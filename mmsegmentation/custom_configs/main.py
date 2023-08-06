# Associates a loss to its class and name
losses_mappings = {
    "skil": ("SkilLoss", "loss_skill"),
    "dice": ("DiceLoss", "loss_dice"),
    "cross_entropy": ("CrossEntropyLoss", "loss_ce"),
    "cl_dice": ("ClDiceLoss", "loss_cl_dice"),
}

# Associates a model name to its default weights
default_checkpoint = {
    "vit_vit-b16_mln_upernet": "pretrain/upernet_vit-b16_mln_512x512_80k_ade20k_20210624_130547-0403cee1_fix.pth"
}

# Associates a dataset name to its config file
datasets_mappings = {
    "cracks_full": "custom_configs.datasets.cracks_full",
    "cracks_cropped": "custom_configs.datasets.cracks_cropped",
}

# Associates a learning rate schedule name to its config file
schedules_mappings = {
    "aggressive": "custom_configs.schedules.aggressive",
}