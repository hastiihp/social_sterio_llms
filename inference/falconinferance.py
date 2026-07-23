"""
Full run: all six models, complete prompts.csv (75,600 rows each).
Same frozen settings, same batching, same strict/salvage parsing and
provenance logging as the pilot script.

Run on a GPU node:
    python full_inference.py llama
    python full_inference.py qwen
    python full_inference.py gemma
    python full_inference.py deepseek
    python full_inference.py ministral
    python full_inference.py falcon
"""

import sys
import os
import re
import csv
import json
import subprocess
import traceback
from datetime import datetime, timezone

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS_CSV = "../data/prompts.csv"
DEFAULT_BATCH_SIZE = 16

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-3-12b-it",
    "qwen": "Qwen/Qwen3-8B",
    "deepseek": "deepseek-ai/deepseek-llm-7b-chat",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
    "falcon": "tiiuae/Falcon-H1-7B-Instruct",
}

# Falcon-H1's hybrid Mamba architecture falls back to an unoptimized kernel
# path, which both (a) uses much more memory per batched sequence at the same
# batch size as the other five models, and (b) has a cache-shape bug when the
# padded sequence length changes between batches. Fix: smaller batch size AND
# a fixed padding length so every batch has identical shape.
BATCH_SIZE_OVERRIDE = {"falcon": 4}
FIXED_LENGTH_OVERRIDE = {"falcon": 320}  # covers the longest rendered prompt with margin

GEN_KWARGS = dict(do_sample=False, max_new_tokens=30, repetition_penalty=1.0)

VALID_A = {"1", "2", "3", "4", "5"}
VALID_B = VALID_A | {"NA"}

SALVAGE_PATTERNS = [
    r"(?:rating|score|response|answer)\s*(?:of|is|:)?\s*([1-5])\b",
    r"\b(?:choose|select|respond with|estimate|rate)\w*\s*(?:a|the)?\s*([1-5])\b",
    r"\bwith\s+a?\s*(?:rating|score)\s+of\s+([1-5])\b",
    r"(?:^|\s)([1-5])[\s.!]*$",
]
REFUSAL_MARKERS = ["cannot provide", "cannot make", "cannot determine",
                    "not possible to", "unable to provide"]


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or "."
        ).decode().strip()
    except Exception:
        return "unknown"


def classify_format(text, cond_label):
    stripped = text.strip()
    valid_set = VALID_A if cond_label.startswith("A_") else VALID_B
    if stripped in valid_set:
        return True, None, "strict_valid"
    for pat in SALVAGE_PATTERNS:
        m = re.search(pat, stripped, re.IGNORECASE)
        if m:
            return False, m.group(1), "salvageable_numeric"
    if re.search(r"\bNA\b", stripped):
        return False, "NA", "salvageable_numeric"
    if any(m in stripped.lower() for m in REFUSAL_MARKERS):
        return False, None, "explicit_refusal"
    return False, None, "other_malformed"


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    return tokenizer, model


def build_batch_inputs(tokenizer, model_id, prompts, fixed_length=None):
    texts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        kwargs = dict(add_generation_prompt=True, tokenize=False)
        if "qwen3" in model_id.lower():
            kwargs["enable_thinking"] = False
        texts.append(tokenizer.apply_chat_template(messages, **kwargs))
    if fixed_length:
        enc = tokenizer(texts, return_tensors="pt", padding="max_length",
                         max_length=fixed_length, truncation=True, add_special_tokens=False)
    else:
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return enc.to("cuda:0")


def generate_batch(tokenizer, model, model_id, prompts, retry=True, fixed_length=None):
    try:
        enc = build_batch_inputs(tokenizer, model_id, prompts, fixed_length=fixed_length)
        with torch.no_grad():
            out = model.generate(**enc, **GEN_KWARGS, pad_token_id=tokenizer.pad_token_id)
        input_len = enc["input_ids"].shape[1]
        results = []
        for i in range(len(prompts)):
            gen_ids = out[i][input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
            if "\u0120" in text:
                text = text.replace("\u0120", " ").strip()
            results.append((text, "none"))
        return results
    except Exception as e:
        if retry:
            print(f"  batch generation failed ({e}), retrying once...", flush=True)
            torch.cuda.empty_cache()
            return generate_batch(tokenizer, model, model_id, prompts, retry=False, fixed_length=fixed_length)
        return [("", "technical_failure")] * len(prompts)


def run_full(model_key):
    model_id = MODELS[model_key]
    batch_size = BATCH_SIZE_OVERRIDE.get(model_key, DEFAULT_BATCH_SIZE)
    fixed_length = FIXED_LENGTH_OVERRIDE.get(model_key, None)
    print(f"\n{'=' * 70}\nFULL RUN: {model_id} (batch_size={batch_size}, "
          f"fixed_length={fixed_length})\n{'=' * 70}", flush=True)

    import transformers
    provenance = dict(
        model_id=model_id,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        prompt_template_version="friend_v2_explicit_gender",
        generation_config=json.dumps(GEN_KWARGS),
        timestamp=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", "none"),
        git_commit=get_git_commit(),
    )
    print(f"Provenance: {provenance}", flush=True)

    tokenizer, model = load_model(model_id)
    df = pd.read_csv(PROMPTS_CSV)
    print(f"Full dataset: {len(df)} prompts", flush=True)

    out_path = f"full_results_{model_key}.csv"
    fieldnames = list(provenance.keys()) + [
        "persona_id", "country", "gender", "age", "profession", "topic",
        "response_condition", "raw_text", "normalized_text",
        "strict_parsed_rating", "strict_is_valid", "salvaged_rating",
        "salvage_method", "is_abstention", "parse_failure_reason",
    ]

    n_written = 0
    n_technical_fail = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]
            prompts = batch["prompt"].tolist()
            results = generate_batch(tokenizer, model, model_id, prompts, fixed_length=fixed_length)

            for row, (text, gen_failure) in zip(batch.itertuples(), results):
                if gen_failure == "technical_failure":
                    n_technical_fail += 1
                    strict_valid, salvage, ftype = False, None, "technical_failure"
                else:
                    strict_valid, salvage, ftype = classify_format(text, row.response_condition)

                is_abstention = strict_valid and text.strip() == "NA"

                writer.writerow({
                    **provenance,
                    "persona_id": row.persona_id, "country": row.country,
                    "gender": row.gender, "age": row.age, "profession": row.profession,
                    "topic": row.topic, "response_condition": row.response_condition,
                    "raw_text": text, "normalized_text": text.strip(),
                    "strict_parsed_rating": (text.strip() if strict_valid else ""),
                    "strict_is_valid": strict_valid,
                    "salvaged_rating": (salvage or ""),
                    "salvage_method": ("regex" if salvage and not strict_valid else ""),
                    "is_abstention": is_abstention,
                    "parse_failure_reason": ("none" if strict_valid else ftype),
                })
                n_written += 1

            if (start // batch_size) % 50 == 0:
                print(f"  {start + len(batch)}/{len(df)} done", flush=True)
                f.flush()

    print(f"Full run complete for {model_id}: {n_written} rows written, "
          f"{n_technical_fail} technical failures -> {out_path}", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        print(f"Usage: python full_inference.py [{'|'.join(MODELS.keys())}]")
        sys.exit(1)
    run_full(sys.argv[1])
    print("\nFull run for this model complete.")
