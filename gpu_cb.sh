#!/bin/bash

#### local path
GLUE_DIR=../data-superglue
INIT_CKPT_DIR=../../xlnet_large_cased
PROC_DATA_DIR=proc_data/cb
MODEL_DIR=experiment/squad

#### Use 3 GPUs, each with 8 seqlen-512 samples
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier.py \
  --do_train=True \
  --do_eval=True \
  --task_name=cb \
  --data_dir=${GLUE_DIR}/CB \
  --output_dir=proc_data/cb \
  --model_dir=exp/cb \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${LARGE_DIR}/xlnet_config.json \
  --init_checkpoint=${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=4 \
  --learning_rate=5e-5 \
  --train_steps=1200 \
  --warmup_steps=120 \
  --save_steps=600 \
  --is_regression=True

  $@
