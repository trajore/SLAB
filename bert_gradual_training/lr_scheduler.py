import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

def build_scheduler(config, optimizer, n_iter_per_epoch, epochs):
    num_steps = int(epochs * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config.TRAIN.MIN_LR,
        warmup_lr_init=config.TRAIN.WARMUP_LR,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    
    return lr_scheduler 