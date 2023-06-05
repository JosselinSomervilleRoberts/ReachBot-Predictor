dataset_type = "ReachbotDataset"
data_root = "../datasets/cracks_seg/"


crop_size = (512, 512)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="Resize", scale=640, keep_ratio=True
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=640, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="img_dir/train", seg_map_path="ann_dir/train"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="ReachbotMetric")
test_evaluator = val_evaluator
