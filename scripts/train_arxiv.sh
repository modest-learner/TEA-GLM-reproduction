#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LLM="${LLM:-${1:-}}"
: "${LLM:?Usage: bash scripts/train_arxiv.sh /path/to/vicuna-7b-v1.5}"
python train_glm.py --backbone "$LLM" --dataset arxiv --freeze_llama \
  --pretrain_gnn GraphSAGE_arxiv_1000_tp.pth --num_token 5 \
  --batch_size "${BATCH_SIZE:-2}" --grad_steps "${GRAD_STEPS:-2}" --epoch 1 \
  --att_d_model 2048 --gnn_output 4096 --clip_grad_norm 1.0 --weight_decay 0.0 \
  --max_text_length "${MAX_TEXT_LENGTH:-700}" --lr "${LR:-0.001}" --optim "${OPTIM:-adam}" \
  --dtype bfloat16 --seed "${SEED:-0}" --exp_num "${EXP_NUM:-1}" --prefix "${PREFIX:-tea-citation}"
