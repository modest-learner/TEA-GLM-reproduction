#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LLM="${LLM:-${1:-}}"
: "${LLM:?Usage: bash scripts/test_citation.sh /path/to/vicuna-7b-v1.5}"
for pair in arxiv:700 pubmed:700 cora:850; do
  IFS=: read -r target max_length <<< "$pair"
  bash scripts/test_single.sh arxiv "$target" 5 "${MAX_TEXT_LENGTH:-$max_length}" "${PREFIX:-tea-citation}" "$LLM" "${PROJECTOR:-}"
done
