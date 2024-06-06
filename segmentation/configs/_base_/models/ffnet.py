# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ffnet_seg',
        drop_path_rate=0.1),
    decode_head=dict(
        type='FFNetHead',
        in_channels=[64, 160, 256],
        in_index=[0, 1, 2],
        spatial_ks = 7,
        channel_ks = 3,
        head_stride = 8,
        head_width = 256, 
        head_depth = 3,
        expansion_ratio = 2,
        final_expansion_ratio = 4,
        channels=1024,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
