#!/usr/bin/env bash

set -euo pipefail

ROOT="/data2/wenb/ylu/HiGGs-GBD/HiGGs-main"
PYTHON_BIN="/data2/wenb/wbai/miniconda3/envs/higgs/bin/python"
DIAG_SCRIPT="$ROOT/tools/diagnostics/cora_teacher_forced_glue.py"

source "$ROOT/logs/sampling/cora_gpu1_pair_majority.latest.env"

CFG="$OUT_DIR/.hydra/config.yaml"
PROCESSED="$ROOT/data/cora/processed/proc_test_X_r2_majority_max500_bs2.pt"

CKPT_ROOT="$ROOT/outputs/2026-07-29/21-56-33/checkpoints/cora_glue_original_r2_e500_resume"

E199="$CKPT_ROOT/epoch=199.ckpt"
LAST="$CKPT_ROOT/last.ckpt"

TAG="$(date +%Y%m%d_%H%M%S)"
COMPARE_DIR="$ROOT/higgs/outputs/${TAG}_cora_glue_checkpoint_compare_gpu0"
LOG_DIR="$ROOT/logs/teacher_forced"

mkdir -p "$COMPARE_DIR" "$LOG_DIR"

for FILE in \
    "$PYTHON_BIN" \
    "$DIAG_SCRIPT" \
    "$CFG" \
    "$PROCESSED" \
    "$E199" \
    "$LAST"
do
    test -s "$FILE" || {
        echo "错误：缺少文件 $FILE"
        exit 1
    }
done

run_checkpoint() {
    local label="$1"
    local checkpoint="$2"
    local output_dir="$COMPARE_DIR/$label"
    local log="$LOG_DIR/cora_teacher_forced_${label}_${TAG}.log"

    mkdir -p "$output_dir"

    echo
    echo "================================================================================"
    echo "开始测试：$label"
    echo "CHECKPOINT=$checkpoint"
    echo "OUTPUT=$output_dir"
    echo "LOG=$log"
    echo "================================================================================"

    CUDA_VISIBLE_DEVICES=0 \
    WANDB_MODE=disabled \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    "$PYTHON_BIN" -u "$DIAG_SCRIPT" \
        --root "$ROOT" \
        --config "$CFG" \
        --checkpoint "$checkpoint" \
        --processed "$PROCESSED" \
        --output-dir "$output_dir" \
        --count 50 \
        --seed 0 \
        2>&1 | tee "$log"

    test -s "$output_dir/teacher_forced_summary.json" || {
        echo "错误：$label 没有生成 summary"
        exit 1
    }

    echo "$label 测试完成"
}

run_checkpoint "epoch199" "$E199"
run_checkpoint "last" "$LAST"

META="$LOG_DIR/cora_glue_checkpoint_compare.latest.env"

{
    printf 'ROOT=%q\n' "$ROOT"
    printf 'COMPARE_DIR=%q\n' "$COMPARE_DIR"
    printf 'TAG=%q\n' "$TAG"
    printf 'E199=%q\n' "$E199"
    printf 'LAST=%q\n' "$LAST"
} > "$META"

echo
echo "================================================================================"
echo "两个 checkpoint 均已完成"
echo "COMPARE_DIR=$COMPARE_DIR"
echo "META=$META"
echo "================================================================================"
