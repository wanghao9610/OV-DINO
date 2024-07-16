#!/usr/bin/env bash
# set -x

# example:
# sh scripts/app.sh demo_config.py pretrained_model

# project config
root_dir="$(realpath $(dirname $0)/../../)"
code_dir=$root_dir/ovdino

config_file=$1
init_ckpt=$(realpath $2)

# env config
export DETECTRON2_DATASETS="$root_dir/datas/"
export HF_HOME="$root_dir/inits/huggingface"
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=DETAIL

cd $code_dir
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
    python ./demo/app.py \
    --config-file $config_file \
    --opts \
    train.init_checkpoint=$init_ckpt \
    model.app_mode=True