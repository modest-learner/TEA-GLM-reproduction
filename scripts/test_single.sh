#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == --interventions ]]; then
  if (( $# < 2 || $# > 3 )); then
    echo 'Usage: test_single.sh --interventions LLM [screen|full]' >&2
    exit 2
  fi
  stage="${3:-screen}"
  if [[ "$stage" == screen ]]; then
    datasets="${DATASETS:-children history}"
    variants="${VARIANTS:-normal scale_1e-3 scale_1e-4 scale_1e-5 self_only shuffle}"
    sample_count="${SAMPLE_COUNT:-2048}"
  elif [[ "$stage" == full ]]; then
    datasets="${DATASETS:-children history pubmed}"
    variants="${VARIANTS:?Full validation requires explicit VARIANTS including normal and selected interventions}"
    sample_count=''
    if [[ -n "${SAMPLE_COUNT:-}" ]]; then echo 'Full validation must not set SAMPLE_COUNT' >&2; exit 2; fi
  else
    echo 'Stage must be screen or full' >&2; exit 2
  fi
  if [[ " $variants " != *' normal '* ]]; then echo 'VARIANTS must include normal' >&2; exit 2; fi
  for variant in $variants; do
    case "$variant" in normal|scale_1e-3|scale_1e-4|scale_1e-5|self_only|shuffle) ;; *) echo "Unknown variant: $variant" >&2; exit 2;; esac
  done
  result_root="${RESULTS_DIR:?Set RESULTS_DIR to a NEW experiment directory}"
  for target in $datasets; do
    case "$target" in
      children|history) source=computer; length=1100; checkpoint=saved_model/first_model/TEA-GLM_ecom.pth ;;
      pubmed) source=arxiv; length=700; checkpoint=saved_model/first_model/TEA-GLM_citation.pth ;;
      *) echo "Unsupported diagnostic target: $target" >&2; exit 2 ;;
    esac
    for seed in ${SEEDS:-0}; do
      for variant in $variants; do
        mode=none; scale=1
        case "$variant" in scale_*) scale="${variant#scale_}";; self_only|shuffle) mode="$variant";; esac
        prefix="${variant}_seed${seed}"
        echo "=== $stage: $source -> $target, $variant, seed=$seed, samples=${sample_count:-all} ==="
        METHOD=tea_glm EXP_NUM=1 SEED="$seed" SAMPLE_COUNT="$sample_count" \
          GRAPH_INTERVENTION="$mode" GRAPH_TOKEN_SCALE="$scale" RESULTS_DIR="$result_root" \
          bash scripts/test_single.sh "$source" "$target" 5 "$length" "$prefix" "$2" "$checkpoint"
      done
    done
  done
  exit 0
fi
if (( $# < 6 )); then
  echo 'Usage: test_single.sh SOURCE TARGET NUM_TOKEN MAX_LENGTH PREFIX LLM [PROJECTOR]' >&2
  exit 2
fi
extra=()
if [[ -n "${7:-}" ]]; then extra+=(--projector_checkpoint "$7"); fi
if [[ -n "${SAMPLE_COUNT:-}" ]]; then extra+=(--sample_count "$SAMPLE_COUNT"); fi
extra+=(--sample_seed "${SAMPLE_SEED:-2026}" --shuffle_seed "${SHUFFLE_SEED:-2026}"
        --graph_intervention "${GRAPH_INTERVENTION:-none}" --graph_token_scale "${GRAPH_TOKEN_SCALE:-1}")
result_root="${RESULTS_DIR:-results}"
log_file="$result_root/logs/${5}_${1}_to_${2}.log"
if [[ -e "$log_file" || -e "$result_root/$2/$5" ]]; then
  echo "Run/log already exists; choose a new PREFIX or RESULTS_DIR: $log_file" >&2
  exit 2
fi
command=(python -u train_glm.py --freeze_llama --inference --dataset "$1" --test_dataset "$2" \
  --num_token "$3" --max_text_length "$4" --prefix "$5" --backbone "$6" \
  --batch_size "${BATCH_SIZE:-4}" --dtype bfloat16 --do_sample --temperature 0.7 \
  --gen_max_length 80 --seed "${SEED:-0}" --exp_num "${EXP_NUM:-1}" \
  --method "${METHOD:-tea_glm}" --results_dir "$result_root" --save_features --verbose --evaluation_policy unique \
  "${extra[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
mkdir -p "$result_root/logs"
"${command[@]}" 2>&1 | tee "$log_file"
