from .coco_schedule import multi_steps_scheduler

# warmup scheduler for ovdino on O365v1 pre-training
# epochs, decay_epochs, warmup_steps, samples_per_epoch, batch_size, total_steps
lr_multiplier_600k_bs64_24ep_two_steps_warmup = multi_steps_scheduler(
    24, [16, 22], 1000, 600000, 64, 225000
)
