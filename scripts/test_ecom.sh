#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LLM="${LLM:-${1:-}}"
: "${LLM:?Usage: bash scripts/test_ecom.sh /path/to/vicuna-7b-v1.5}"
# Children reaches 1035 tokens with the released instructions and Transformers
# 4.28 tokenizer. Raise the original 1000 limit without truncating any prompt.
for pair in children:1100 history:1300 computer:1000 photo:1000 sports:500; do
  IFS=: read -r target max_length <<< "$pair"
  bash scripts/test_single.sh computer "$target" 5 "${MAX_TEXT_LENGTH:-$max_length}" "${PREFIX:-tea-ecom}" "$LLM" "${PROJECTOR:-}"
done
