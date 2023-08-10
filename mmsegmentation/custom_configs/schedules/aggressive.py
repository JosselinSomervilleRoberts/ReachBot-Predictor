def get_scheduler(num_steps_total):
    MAX_LR = 1e-5
    END_LINEAR = int(0.25 * num_steps_total)
    START_FACTOR = MAX_LR / END_LINEAR
    param_scheduler = [
        dict(
            type="LinearLR",
            start_factor=START_FACTOR,
            by_epoch=False,
            begin=0,
            end=END_LINEAR,
        ),  # int(0.15 * num_steps_total)),
        dict(
            type="PolyLR",
            eta_min=0.0,
            power=1.0,
            begin=END_LINEAR,
            end=num_steps_total,
            by_epoch=False,
        ),
    ]
    return param_scheduler
