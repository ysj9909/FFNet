_base_ = [
    '../_base_/models/ffnet.py',
    '../_base_/datasets/cityscapes_1024x1024_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]
# model settings
checkpoint = 'ffnet_seg.pth.tar'
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type = 'ffnet_seg_3',
        drop_path_rate = 0.35,
        init_cfg=dict(type='Pretrained', checkpoint = checkpoint)),
    decode_head=dict(
        type='FFNetHead',
        in_channels=[192, 384, 768],
        spatial_ks = 9,
        channel_ks = 9,
        head_stride = 8,
        head_width = 256, 
        head_depth = 5,
        expansion_ratio = 1,
        final_expansion_ratio = 4,
        channels=1024,
        dropout_ratio=0.1,
        drop_path = 0.,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# data
data = dict(samples_per_gpu=2)
evaluation = dict(interval=4000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=4000)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
