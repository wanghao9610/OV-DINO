import os
import os.path as osp

from detrex.config import get_config

from .models.ovdino_swin_tiny224_bert_base import model

model_root = os.getenv("MODEL_ROOT", "./inits")
init_checkpoint = osp.join(model_root, "./swin", "swin_tiny_patch4_window7_224.pth")

# get default config
dataloader = get_config("common/data/coco_ovd.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config(
    "common/coco_schedule.py"
).lr_multiplier_120k_bs32_24ep_two_steps_warmup
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = init_checkpoint
train.output_dir = "./wkdrs/ovdino_swin_tiny224_bert_base_24ep_ft_coco"

# max training iterations, 120000 / 32 * 24 = 90000 -> 90000
train.max_iter = 90000
train.eval_period = 5000
train.log_period = 50
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device
model.num_classes = 80

# find_unused_parameters
# train.ddp.find_unused_parameters = True

# modify optimizer config
optimizer.lr = 1e-5
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: (
    0.1 if "language_backbone" in module_name else 1.0
)

# modify dataloader config
dataloader.train.num_workers = 8

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 64/4 = 4
dataloader.train.total_batch_size = 32

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
