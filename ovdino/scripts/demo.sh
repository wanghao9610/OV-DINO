#!/usr/bin/env bash
# set -x

# project config
root_dir="$(realpath $(dirname $0)/../../)"
code_dir=$root_dir/ovdino

sam_config_file="sam2_hiera_l.yaml"
sam_init_checkpoint="$root_dir/inits/sam2/sam2_hiera_large.pt"

config_file=$1
init_ckpt=$(realpath $2)
category_names=$3
input=$4
output=$5
num_classes=$(echo $category_names | wc -w)

# env config
export DETECTRON2_DATASETS="$root_dir/datas/"
export HF_HOME="$root_dir/inits/huggingface"
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=DETAIL

cd $code_dir
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
    python ./demo/demo.py \
    --config-file $config_file \
    --sam-config-file $sam_config_file \
    --sam-init-checkpoint $sam_init_checkpoint \
    --input $input \
    --output $output \
    --category_names $category_names \
    --opts \
    train.init_checkpoint=$init_ckpt \
    model.num_classes=$num_classes