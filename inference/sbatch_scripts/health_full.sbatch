#!/bin/bash
#SBATCH --job-name=health_full
#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --exclude=n[51-55,101,102]
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=../logs/health_full_%j.out
#SBATCH --error=../logs/health_full_%j.err

# PROVENANCE NOTE (added during the data/+results/ reorg): this script reflects
# the exact commands that produced the checked-in results, including the since-
# removed health_staging/ working directory and pre-reorg filenames
# (health_prompts_full.csv, health_full_results_{model}.csv). It is kept
# unmodified for historical accuracy and is NOT directly re-runnable against
# the current data/ + results/ layout -- see README.md's "Running inference"
# section for current commands and inference/full_health_inference.py for the
# up-to-date script paths. GPU/resource settings (partition, gres, mem, time)
# below are still accurate. Also note: this smoketest step wrote its 200-row
# sample to the same filename the full run uses, which the current
# full_health_inference.py no longer does (see its module docstring) -- do
# not reuse this smoketest logic verbatim.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate paper_env
export HF_HOME=/mnt/beegfs/projects/ttessllm/hf_cache
cd ~/ttessllm/social_sterio_llms/health_staging

echo "=== STEP 1: quick format-compliance check (llama + qwen, first 200 rows only) ==="
python3 -c "
import pandas as pd
df = pd.read_csv('health_prompts_full.csv').head(200)
df.to_csv('health_prompts_smoketest.csv', index=False)
print('wrote smoketest subset:', len(df), 'rows')
"

python3 -c "
import sys
sys.argv = ['x', 'llama']
import importlib
m = importlib.import_module('full_health_inference')
m.PROMPTS_CSV = 'health_prompts_smoketest.csv'
m.run_full('llama')
"
python3 -c "
import sys
sys.argv = ['x', 'qwen']
import importlib
m = importlib.import_module('full_health_inference')
m.PROMPTS_CSV = 'health_prompts_smoketest.csv'
m.run_full('qwen')
"

echo "=== Checking smoketest compliance rate ==="
PASS=$(python3 -c "
import pandas as pd
llama = pd.read_csv('health_full_results_llama.csv')
qwen = pd.read_csv('health_full_results_qwen.csv')
r_llama = llama['strict_is_valid'].mean()
r_qwen = qwen['strict_is_valid'].mean()
print(f'llama strict_valid_rate: {r_llama:.4f}')
print(f'qwen strict_valid_rate: {r_qwen:.4f}')
print('PASS' if (r_llama >= 0.90 and r_qwen >= 0.90) else 'FAIL')
")
echo "$PASS"

if echo "$PASS" | grep -q "FAIL"; then
    echo "=== SMOKETEST FAILED. Stopping before full run. Check health_full_results_llama.csv / health_full_results_qwen.csv (smoketest, 200 rows) ==="
    exit 1
fi

echo "=== SMOKETEST PASSED. Proceeding to full 5-model run (75,600 rows each). ==="

echo "=== STEP 2: full run, 5 models ==="
python3 full_health_inference.py llama
python3 full_health_inference.py gemma
python3 full_health_inference.py qwen
python3 full_health_inference.py deepseek
python3 full_health_inference.py ministral

echo "=== ALL MODELS COMPLETE ==="
