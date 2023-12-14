#! /bin/bash

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_zero2_config.json"
bert_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 4 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 24 \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save checkpoints/bert_345m \
       --load checkpoints/bert_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --fp32-layernorm \
       --fp32-embedding \
       &> bert_zero_2.log
"
bert_options="${bert_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="numactl --interleave=all \
       deepspeed --num_nodes ${NUM_WORKERS} \
              --num_gpus ${NUM_GPUS_PER_WORKER} \
              pretrain_bert.py $@ ${bert_options}"

echo ${run_cmd}
eval ${run_cmd}

set +x
