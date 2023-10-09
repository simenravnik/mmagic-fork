_base_ = '../_base_/models/base_dvsr.py'

experiment_name = 'dvsr_reds4'
save_dir = './work_dirs'
work_dir = f'./work_dirs/{experiment_name}'

# model settings
model = dict(
    type='DVSR',
    generator=dict(
        type='DVSRNet',
        upscale_factor=4,
        basicvsr_pretrained='configs/dvsr/basicvsr_reds4_20120409-0e599677.pth',
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='sum'),
    train_cfg=dict(),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[50000, 100000, 150000, 150000, 150000],
    restart_weights=[1, 0.5, 0.5, 0.5, 0.5],
    eta_min=1e-7)

find_unused_parameters = True
