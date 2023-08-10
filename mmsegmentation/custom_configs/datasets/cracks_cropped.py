def get_dataset(crop_size, train_batch_size):
    dataset_type = "ReachbotDataset"
    data_root = "../datasets/cracks_seg/"
    num_training_samples = 80
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        dict(type="Resize", scale=512, keep_ratio=True),
        dict(type="RandomFlip", prob=0.5),
        dict(type="PhotoMetricDistortion"),
        dict(type="PackSegInputs"),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=512, keep_ratio=True),
        # add loading annotation after ``Resize`` because ground truth
        # does not need to do resize data transform
        dict(type="LoadAnnotations"),
        dict(type="PackSegInputs"),
    ]
    train_dataloader = dict(
        batch_size=train_batch_size,
        num_workers=min(4, train_batch_size),
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
        num_workers=1,
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

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_pipeline,
        test_pipeline,
        dataset_type,
        data_root,
        num_training_samples,
    )
