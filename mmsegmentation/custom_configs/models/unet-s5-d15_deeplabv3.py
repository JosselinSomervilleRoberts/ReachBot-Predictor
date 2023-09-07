_base_ = [
    "../../configs/_base_/models/deeplabv3_unet_s5-d16.py",
    # "../_base_/datasets/stare.py",
    "../../configs/_base_/default_runtime.py",
    # "../_base_/schedules/schedule_40k.py",
]
crop_size = (128, 128)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(128, 128), stride=(85, 85)),
)
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, clip_grad=None)
