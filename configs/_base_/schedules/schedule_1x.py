# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate scheduler (leave empty, use main config)
param_scheduler = []

# optimizer (leave empty, use main config)
optim_wrapper = {}

# auto scale (disabled, defined in main config)
auto_scale_lr = dict(enable=False, base_batch_size=16)
