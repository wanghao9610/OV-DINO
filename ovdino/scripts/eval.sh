#!/usr/bin/env bash
# set -x

# project config
root_dir="$(realpath $(dirname $0)/../../)"
code_dir=$root_dir/ovdino
time=$(date "+%Y%m%d-%H%M%S")

config_file=$1
init_ckpt=$(realpath $2)
output_dir=$3
dataset=$(basename $config_file | sed 's/.*_\(.*\)\.py/\1/')

# env config
export DETECTRON2_DATASETS="$root_dir/datas/"
export HF_HOME="$root_dir/inits/huggingface"
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Distributed Testing on $dataset"
evaluation_dir="$output_dir/eval_${dataset}_$time"
mkdir -p $evaluation_dir
cd $code_dir
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
    python ./tools/train_net.py \
    --config-file $config_file \
    --eval-only \
    --resume \
    train.init_checkpoint=$init_ckpt \
    train.output_dir=$output_dir \
    dataloader.evaluator.output_dir="$evaluation_dir" | tee $evaluation_dir/eval_$time.log