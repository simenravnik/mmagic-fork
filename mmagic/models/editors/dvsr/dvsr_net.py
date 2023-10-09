import torch
import torch.nn as nn
import torch.nn.functional as F

import mmagic.utils.common as common
from mmagic.utils.tools import extract_image_patches, reverse_patches

from mmengine.model import BaseModule
from mmagic.registry import MODELS

from mmcv.cnn import ConvModule
from mmengine import MMLogger, print_log
from mmengine.runner import load_checkpoint

from mmagic.models.archs import PixelShufflePack, ResidualBlockNoBN
from mmagic.models.utils import flow_warp, make_layer

from mmagic.models.editors.basicvsr.basicvsr_net import SPyNet
from mmagic.models.editors.basicvsr.basicvsr_net import BasicVSRNet
from mmagic.models.editors.basicvsr_modified.basicvsr_net_modified import BasicVSRNetModified


class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x) 
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, features, scaling_factor=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1)
        self.scaling_factor = scaling_factor
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = out * self.scaling_factor
        out = out + residual

        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_heads, embed_dim, patch_size=3, alpha=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(patch_size, stride=patch_size)
        self.transformer_encoder = nn.TransformerEncoderLayer(embed_dim * patch_size * patch_size, num_heads)
        self.alpha = alpha
        
    def forward(self, x):
        residual = x

        # b, c, h, w = x.size()

        # # Unfold into patches
        # x = self.unfold(x)
        # x = x.permute(2, 0, 1)

        out = self.transformer_encoder(x)

        # Reshape back to original size
        # fold = nn.Fold(output_size=(h, w), kernel_size=self.patch_size, stride=self.patch_size)
        # out = out.permute(1, 2, 0)
        # out = fold(out)

        out = out * self.alpha
        out = out + residual

        return out


class Un(nn.Module):
    def __init__(self, features):
        super(Un, self).__init__()
        self.res_block = ResidualBlock(features, features // 2)

        self.reduce = common.default_conv(3 * features, features, 3)

        self.attention = TransformerEncoderBlock(num_heads=features, embed_dim=features, patch_size=3)
        self.alise = common.default_conv(features, features, 3)

    def forward(self,x):

        x1 = self.res_block(x)
        x2 = self.res_block(x1)
        x3 = self.res_block(x2)
        out = x3
        
        _, _, h, w = x3.shape

        x4 = self.reduce(torch.cat([x1, x2, x3], dim=1))

        x4 = extract_image_patches(x4, ksizes=(3, 3), strides=(1, 1), rates=(1, 1), padding='same')
        x4 = x4.permute(0, 2, 1)

        out = self.attention(x4)
        out = out.permute(0, 2, 1)
        out = reverse_patches(out, out_size=(h, w), ksizes=(3, 3), strides=(1, 1), padding=(1, 1))

        out = self.alise(out)

        return x + out

@MODELS.register_module()
class DVSRNet(BaseModule):
    def __init__(self, upscale_factor=4, basicvsr_pretrained=None, conv=common.default_conv):
        
        super().__init__()

        self.features = 64
        n_blocks = 2
        kernel_size = 3
        self.scale = upscale_factor
        self.n_blocks = n_blocks
        
        # define head module
        modules_head = [conv(3, self.features, kernel_size)]
        
        # define body module
        modules_body = nn.ModuleList([Un(features=self.features) for _ in range(n_blocks)])

        # define tail module
        modules_tail = [
            common.Upsampler(conv, self.scale, self.features, act=False),
            conv(self.features, 3, kernel_size)
        ]

        self.up = nn.Sequential(common.Upsampler(conv, self.scale, self.features, act=False), Conv(self.features, 3, 3, 1, 1))
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        
        self.reduce = conv(n_blocks * self.features, self.features, kernel_size)


        # self.basicvsr = BasicVSRNet()
        self.basicvsr_modified = BasicVSRNetModified()
        # self.spynet = SPyNet(pretrained=spynet_pretrained)

        if isinstance(basicvsr_pretrained, str):
            checkpoint = torch.load(basicvsr_pretrained)
            checkpoint = self._remove_basicvsr_prefix(checkpoint['state_dict'])
            self.basicvsr_modified.load_state_dict(checkpoint, strict=False)
        elif basicvsr_pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(basicvsr_pretrained)}.')
        
    def _remove_basicvsr_prefix(self, state_dict):
        modified_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("generator."):
                modified_key = key[len("generator."):]
                modified_state_dict[modified_key] = value

        return modified_state_dict

    def forward(self, inputs):

        b, t, c, h, w = inputs.shape

        temporal = self.basicvsr_modified(inputs)[:, -1, :, :, :]
        temporal = temporal.view(b, c, self.scale * h, self.scale * w)

        inputs = inputs.permute(0, 2, 1, 3, 4)

        b, c, t, h, w = inputs.shape

        # Select the last frame along the frames dimension
        x = inputs[:, :, -1, :, :]
        x = x.view(b, c, h, w)

        x = self.head(x)
        res2 = x

        body_out = []
        for block in self.body:
            x = block(x)
            body_out.append(x)

        res1 = self.reduce(torch.cat(body_out, dim=1))

        x = self.tail(res1)
        x = self.up(res2) + x

        x = x + temporal

        return x

    # def compute_flow(self, lrs):
    #     n, t, c, h, w = lrs.size()
    #     lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
    #     lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

    #     flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

    #     return flows_forward

    # def forward(self, inputs):

    #     b, t, c, h, w = inputs.shape

    #     # compute optical flow
    #     flows_forward = self.compute_flow(inputs)

    #     # forward-time propagation and upsampling
    #     outputs = []
    #     feat_prop = inputs.new_zeros(b, self.features, h, w)
    #     for i in range(0, t):
    #         lr_curr = inputs[:, i, :, :, :]
    #         if i > 0:  # no warping required for the first timestep
    #             flow = flows_forward[:, i - 1, :, :, :]
    #             feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

    #         feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
    #         feat_prop = self.forward_conv(feat_prop)
    #         outputs.append(feat_prop)

    #     temporal = outputs[-1]
    #     temporal = self.up(temporal)

    #     inputs = inputs.permute(0, 2, 1, 3, 4)

    #     b, c, t, h, w = inputs.shape

    #     # Select the last frame along the frames dimension
    #     x = inputs[:, :, -1, :, :]
    #     x = x.view(b, c, h, w)

    #     x = self.head(x)
    #     res2 = x

    #     body_out = []
    #     for block in self.body:
    #         x = block(x)
    #         body_out.append(x)

    #     res1 = self.reduce(torch.cat(body_out, dim=1))

    #     x = self.tail(res1)
    #     x = self.up(res2) + x

    #     x = x + temporal

    #     return x
    