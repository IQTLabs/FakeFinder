# model settings
model = dict(
    type='TSN3D',
    backbone=dict(
        type='InceptionV1_I3D',
        pretrained=None,
        modality='RGB'),
    spatial_temporal_module=dict(
        type='SimpleSpatialTemporalModule',
        spatial_type='avg',
        temporal_size=-1,
        spatial_size=-1),
    segmental_consensus=dict(
        type='SimpleConsensus',
        consensus_type='avg'),
    cls_head=dict(
        type='ClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.5,
        in_channels=2048,
        num_classes=51))
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'RawFramesDataset'
data_root_val = 'data/hmdb51/rawframes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    test=dict(
        type=dataset_type,
        ann_file='data/hmdb51/hmdb51_train_split_1_rawframes.txt',
        img_prefix=data_root_val,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=10,
        new_length=64,
        new_step=1,
        random_shift=True,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=256,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample='three_crop',
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=True))

dist_params = dict(backend='nccl')
