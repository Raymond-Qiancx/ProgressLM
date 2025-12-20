#!/bin/bash
cd /projects/p32958/chengxuan/ProgressLM/eval/qwen3vl/scripts/edit_nega
for f in qwen3vl_*_nothink.sh; do
    echo Running: $f
    bash "$f" || exit 1
done
echo All scripts completed
