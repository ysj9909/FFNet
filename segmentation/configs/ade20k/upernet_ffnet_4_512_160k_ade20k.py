_base_ = [
    './upernet_r50_nopretrain.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
dims = [128, 256, 512, 1024]
model = dict(
    backbone=dict(
        _delete_=True,
        type='ffnet_4',
        layer_scale_init_value = 1.,
        drop_path_rate=0.3,
        init_cfg=dict(type='Pretrained', checkpoint='ffnet_4.pth.tar')
    ),
    decode_head=dict(num_classes=150, in_channels=dims),
    auxiliary_head=dict(num_classes=150, in_channels=dims[2]),
    test_cfg=dict(mode='whole')
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

optimizer=dict(constructor='FFNetLearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
                   lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                   paramwise_cfg={'decay_rate': 0.9,
                                  'decay_type': 'layer_wise',
                                  'num_layers': 12})

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2,
          val=dict(pipeline=test_pipeline),
          test=dict(pipeline=test_pipeline))
runner = dict(type='IterBasedRunner')
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=20)
evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')