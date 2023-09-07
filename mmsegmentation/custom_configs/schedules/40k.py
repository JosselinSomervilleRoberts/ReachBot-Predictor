def get_scheduler(num_steps_total):
    # learning policy
    param_scheduler = [
        dict(type="PolyLR", eta_min=1e-4, power=0.9, begin=0, end=40000, by_epoch=False)
    ]
    return param_scheduler
