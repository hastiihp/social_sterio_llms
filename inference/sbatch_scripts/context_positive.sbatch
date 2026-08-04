#!/bin/bash
#SBATCH --job-name=ctx_positive
#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --exclude=n[51-55,101,102]
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=../logs/ctx_positive_%j.out
#SBATCH --error=../logs/ctx_positive_%j.err

# PROVENANCE NOTE (added during the data/+results/ reorg): this script reflects
# the exact commands that produced the checked-in results, including the since-
# removed context_staging/ working directory and pre-reorg filenames
# ({ctx}_full_results_{model}.csv etc). It is kept unmodified for historical
# accuracy and is NOT directly re-runnable against the current data/ + results/
# layout -- see README.md's "Running inference" section for current commands
# and inference/context_full_inference.py for the up-to-date script paths. GPU/
# resource settings (partition, gres, mem, time) below are still accurate.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate paper_env
export HF_HOME=/mnt/beegfs/projects/ttessllm/hf_cache
cd ~/ttessllm/social_sterio_llms/context_staging

CTX=positive

echo "=== STEP 1: smoketest gate for $CTX (llama + qwen, 200 rows each) ==="

if [ -f "smoketest_passed_${CTX}.flag" ]; then
    echo "Smoketest already passed for $CTX in a previous run -- skipping."
else
    python3 context_full_inference.py $CTX llama --smoketest
    python3 context_full_inference.py $CTX qwen --smoketest

    echo "=== Checking smoketest compliance for $CTX ==="
    PASS=$(python3 -c "
import pandas as pd
ctx = '$CTX'
models = ['llama', 'qwen']
all_pass = True
for m in models:
    df = pd.read_csv(f'{ctx}_smoketest_results_{m}.csv')
    rate = df['strict_is_valid'].mean()
    print(f'{ctx}/{m} strict_valid_rate: {rate:.4f}')
    if rate < 0.90:
        all_pass = False
print('PASS' if all_pass else 'FAIL')
")
    echo "$PASS"

    if echo "$PASS" | grep -q "FAIL"; then
        echo "=== SMOKETEST FAILED for $CTX. Stopping before full run. ==="
        exit 1
    fi

    touch "smoketest_passed_${CTX}.flag"
    echo "=== SMOKETEST PASSED for $CTX. ==="
fi

echo "=== STEP 2: full run for $CTX, 5 models (resumable) ==="
for model in llama gemma qwen deepseek ministral; do
    outfile="${CTX}_full_results_${model}.csv"
    if [ -f "$outfile" ]; then
        actual_rows=$(($(wc -l < "$outfile") - 1))
        if [ "$actual_rows" -eq 75600 ]; then
            echo "SKIP: $outfile already complete (75600 rows)"
            continue
        else
            echo "INCOMPLETE FOUND: $outfile has $actual_rows rows, expected 75600 -- rerunning"
        fi
    fi
    python3 context_full_inference.py $CTX $model
done

echo "=== $CTX COMPLETE ==="
