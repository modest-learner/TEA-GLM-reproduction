#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# < 6 )); then
  echo 'Usage: test_single.sh SOURCE TARGET NUM_TOKEN MAX_LENGTH PREFIX LLM [PROJECTOR]' >&2
  exit 2
fi
extra=()
if [[ -n "${7:-}" ]]; then extra+=(--projector_checkpoint "$7"); fi
result_root="${RESULTS_DIR:-results}"
log_file="$result_root/logs/${5}_${1}_to_${2}.log"
if [[ -e "$log_file" || -e "$result_root/$2/$5" ]]; then
  echo "Run/log already exists; choose a new PREFIX or RESULTS_DIR: $log_file" >&2
  exit 2
fi
mkdir -p "$result_root/logs"
python -u train_glm.py --freeze_llama --inference --dataset "$1" --test_dataset "$2" \
  --num_token "$3" --max_text_length "$4" --prefix "$5" --backbone "$6" \
  --batch_size "${BATCH_SIZE:-4}" --dtype bfloat16 --do_sample --temperature 0.7 \
  --gen_max_length 80 --seed "${SEED:-0}" --exp_num "${EXP_NUM:-1}" \
  --method "${METHOD:-tea_glm}" --results_dir "$result_root" --save_features --verbose \
  "${extra[@]}" 2>&1 | tee "$log_file"
