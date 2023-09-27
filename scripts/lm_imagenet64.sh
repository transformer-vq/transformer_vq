#!/bin/bash

Help() {
  echo "Syntax: lm_imagenet64.sh [x|c|r|p|t|l|m|s|a|g|e|i|o|d|w|h]"
  echo "options:"
  echo "x     Multihost mode?"
  echo "c     Command: train, val, test, or sample."
  echo "r     Random seed: master PRNG seed for weight init, shuffle, dropout, etc."
  echo "p     Precision: float32 or bfloat16."
  echo "t     Sequence length."
  echo "l     Block length."
  echo "m     Memory length: pre-aggregated cache size."
  echo "s     Shortcodes: aggregated cache size."
  echo "a     Include aggregated cache in attention computations."
  echo "g     Gradient thru cache?"
  echo "e     Extra steps: extra steps for current training phase."
  echo "i     Input checkpoint dir: local dir or GCS bucket."
  echo "o     Output checkpoint dir: local dir or GCS bucket."
  echo "d     Data dir: local dir or GCS bucket."
  echo "w     Optional w&b run id: required for resuming runs with logging continuity."
  echo "h     Print this Help."
  echo
}

MULTIHOST=0;
while getopts "xc:r:p:t:l:m:s:a:g:e:i:o:d:w:h" option; do
  case $option in
    x)
      MULTIHOST=1;;
    c)
      COMMAND=$OPTARG;;
    r)
      PRNG_SEED=$OPTARG;;
    p)
      ACT_DTYPE=$OPTARG;;
    t)
      SEQUENCE_LEN=$OPTARG;;
    l)
      BLOCK_LEN=$OPTARG;;
    m)
      MEM_LEN=$OPTARG;;
    s)
      NUM_CODE=$OPTARG;;
    a)
      USE_AGG_CACHE=$OPTARG;;
    g)
      CACHE_GRADS=$OPTARG;;
    e)
      EXTRA_STEPS=$OPTARG;;
    i)
      INPUT_CHECKPOINT_DIR=$OPTARG;;
    o)
      OUTPUT_CHECKPOINT_DIR=$OPTARG;;
    d)
      DATA_DIR=$OPTARG;;
    w)
      WB_RUN_ID=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done


python3 scripts/launch.py \
  --multihost="$MULTIHOST" \
  --command="$COMMAND" \
  --dataset=imagenet64 \
  --data_dir="$DATA_DIR" \
  --prng_seed="$PRNG_SEED" \
  --global_batch_size=128 \
  --sequence_len="$SEQUENCE_LEN" \
  --update_len=1536 \
  --block_len="$BLOCK_LEN" \
  --mem_len="$MEM_LEN" \
  --grad_thru_cache="$CACHE_GRADS" \
  --agg_cache="$USE_AGG_CACHE" \
  --param_dtype=float32 \
  --dtype="$ACT_DTYPE" \
  --d_model=2048 \
  --d_k=128 \
  --d_v=4096 \
  --d_ff=0 \
  --n_head=1 \
  --n_code="$NUM_CODE" \
  --n_layer=24 \
  --pe_abs=1 \
  --p_dropemb=0.0 \
  --p_dropsin=0.1 \
  --p_dropres=0.0 \
  --p_droplyr=0.0 \
  --c_beta=0.0001 \
  --c_gamma=0.99 \
  --e_tie=1 \
  --e_preln=1 \
  --e_scale=0.005 \
  --optimizer=adafactor \
  --lr_max=0.01 \
  --lr_schedule=cosine \
  --wd_lam=0.0 \
  --p_nucleus=0.999 \
  --n_warmup_step=10000 \
  --n_max_step=500000 \
  --n_extra_step="$EXTRA_STEPS" \
  --n_save_step=1000 \
  --n_eval_step=100 \
  --in_checkpoint_dir="$INPUT_CHECKPOINT_DIR" \
  --out_checkpoint_dir="$OUTPUT_CHECKPOINT_DIR" \
  --model_name=transformer_vq_imagenet64 \
  --run_id="$WB_RUN_ID"
