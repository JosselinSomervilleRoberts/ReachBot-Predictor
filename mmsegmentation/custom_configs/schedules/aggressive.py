
def get_scheduler(num_steps_total):
    param_scheduler = [
        dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=int(0.15 * num_steps_total)),
        dict(
            type="PolyLR",
            eta_min=0.0,
            power=1.0,
            begin=int(0.15 * num_steps_total),
            end=num_steps_total,
            by_epoch=False,
        ),
    ]
    return param_scheduler