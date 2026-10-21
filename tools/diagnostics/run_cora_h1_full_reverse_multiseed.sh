#!/usr/bin/env bash

ROOT="/data2/wenb/ylu/HiGGs-GBD/HiGGs-main"
PY="/data2/wenb/wbai/miniconda3/envs/higgs/bin/python"
SOURCE="$ROOT/tools/diagnostics/cora_beta_h1_checkpoint_full_reverse_comparison.py"
GENERATED="$ROOT/tools/diagnostics/generated_multiseed"

cd "$ROOT"

mkdir -p "$GENERATED"

for SEED in 20260812 20260813 20260814 20260815
do
    echo
    echo "================================================================================"
    echo "Starting validation full reverse seed=$SEED"
    echo "================================================================================"

    TEMP="$GENERATED/cora_h1_full_reverse_seed_${SEED}.py"

    sed \
      -e "s/^SEED = 20260811$/SEED = ${SEED}/" \
      -e "s/cora_beta_h1_checkpoint_full_reverse_validation_seed20260811/cora_beta_h1_checkpoint_full_reverse_validation_seed${SEED}/g" \
      "$SOURCE" > "$TEMP"

    if [ ! -s "$TEMP" ]; then
        echo "Seed $SEED temporary script was not created"
        break
    fi

    env PYTHONDONTWRITEBYTECODE=1 \
      "$PY" -B -u "$TEMP"

    RUN_CODE=$?

    echo "Seed $SEED finished with code $RUN_CODE"

    if [ "$RUN_CODE" -ne 0 ]; then
        break
    fi
done

echo
echo "Multi-seed runner finished"
