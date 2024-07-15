#!/usr/bin/env bash
# set -x

# project config
root_dir="$(realpath $(dirname $0)/../../)"
code_dir=$root_dir/ovdino
time=$(date "+%Y%m%d-%H%M%S")

config_file=$1
init_ckpt=$(realpath $2)
config_name=$(basename $config_file .py)
output_dir=$root_dir/wkdrs/$config_name

# env config
export DETECTRON2_DATASETS="$root_dir/datas/"
export HF_HOME="$root_dir/inits/huggingface"
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Distributed Training"
evaluation_dir="$output_dir/eval_coco_$time"
mkdir -p $evaluation_dir
cd $code_dir
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
    python ./tools/train_net.py \
    --config-file $config_file \
    --resume \
    train.init_checkpoint=$init_ckpt \
    train.output_dir=$output_dir \
    dataloader.evaluator.output_dir="$evaluation_dir" | tee $output_dir/train_$time.log
    