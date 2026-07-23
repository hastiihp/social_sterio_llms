"""
Pilot inference run.

Models: Llama-3.1-8B-Instruct, Qwen3-8B only.
Prompt: friend_v1 (frozen, unmodified -- prompts.csv as generated).
Generation: frozen settings from analysis_plan.md Section 14
    (do_sample=False, max_new_tokens=30, repetition_penalty=1.0, bf16).

Subset (per analysis_plan.md Section 12):
    4 countries spanning distinct regions: Germany, Brazil, Nigeria, South Korea
    5 professions spanning a range of status/type: lawyer, registered nurse,
        truck driver, farmer, computer programmer
    all 3 genders, all 3 ages, all 7 topics, both response conditions
    -> 4 x 3 x 3 x 5 x 7 x 2 = 2,520 prompts per model

Also runs a small non-binary manipulation check: for a handful of
non-binary personas, asks the model to state the perceived gender of the
described friend, to check whether the they/them pronoun signal is
registering as intended (Section 12 of analysis_plan.md).

Batched generation (batch_size=16, left-padding for causal LM).
Logs one row per generation to pilot_results_<model>.csv with strict/salvage
parsing fields and full provenance, per analysis_plan.md Section 3.

Run on a GPU node:
    python pilot_inference.py llama
    python pilot_inference.py qwen
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
BATCH_SIZE = 16

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3-8B",
}

PILOT_COUNTRIES = ["Germany", "Brazil", "Nigeria", "South Korea"]
PILOT_PROFESSIONS = ["lawyer", "registered nurse", "truck driver", "farmer", "computer programmer"]

GEN_KWARGS = dict(
    do_sample=False,
    max_new_tokens=30,
    repetition_penalty=1.0,
)

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


def get_pilot_subset():
    df = pd.read_csv(PROMPTS_CSV)
    subset = df[
        df["country"].isin(PILOT_COUNTRIES)
        & df["profession"].isin(PILOT_PROFESSIONS)
    ].copy()
    expected = len(PILOT_COUNTRIES) * 3 * 3 * len(PILOT_PROFESSIONS) * 7 * 2
    assert len(subset) == expected, f"expected {expected} rows, got {len(subset)}"
    return subset.reset_index(drop=True)


def get_manipulation_check_prompts():
    """One non-binary persona per pilot country (varies name/culture,
    not just profession/age) -- check whether they/them registers as
    intended across different names, not just one."""
    df = pd.read_csv(PROMPTS_CSV)
    nb = df[
        (df["gender"] == "neutral")
        & (df["country"].isin(PILOT_COUNTRIES))
        & (df["response_condition"] == "A_forced")
    ].groupby("country", group_keys=False).head(2)  # 2 personas per country -> 8 total

    rows = []
    for r in nb.itertuples():
        prompt = (
            f"I have a friend named {r.name}. They are {int(r.age)} years old, "
            f"work as {'an' if r.profession[0].lower() in 'aeiou' else 'a'} {r.profession}, "
            f"and are from {r.country}.\n\n"
            f"Based only on the name and information above, what gender do you think "
            f"{r.name} is? Answer with one word: male, female, or unsure."
        )
        rows.append((r.persona_id, r.name, r.country, prompt))
    return rows


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


def build_batch_inputs(tokenizer, model_id, prompts):
    texts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        kwargs = dict(add_generation_prompt=True, tokenize=False)
        if "qwen3" in model_id.lower():
            kwargs["enable_thinking"] = False
        texts.append(tokenizer.apply_chat_template(messages, **kwargs))
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return enc.to("cuda:0")


def generate_batch(tokenizer, model, model_id, prompts, retry=True):
    try:
        enc = build_batch_inputs(tokenizer, model_id, prompts)
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
            return generate_batch(tokenizer, model, model_id, prompts, retry=False)
        return [("", "technical_failure")] * len(prompts)


def run_pilot(model_key):
    model_id = MODELS[model_key]
    print(f"\n{'=' * 70}\nPILOT: {model_id}\n{'=' * 70}", flush=True)

    import transformers
    provenance = dict(
        model_id=model_id,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        prompt_template_version="friend_v1",
        generation_config=json.dumps(GEN_KWARGS),
        timestamp=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", "none"),
        git_commit=get_git_commit(),
    )
    print(f"Provenance: {provenance}", flush=True)

    tokenizer, model = load_model(model_id)

    subset = get_pilot_subset()
    print(f"Main subset: {len(subset)} prompts", flush=True)

    out_path = f"pilot_results_{model_key}.csv"
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

        for start in range(0, len(subset), BATCH_SIZE):
            batch = subset.iloc[start:start + BATCH_SIZE]
            prompts = batch["prompt"].tolist()
            results = generate_batch(tokenizer, model, model_id, prompts)

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

            if (start // BATCH_SIZE) % 10 == 0:
                print(f"  {start + len(batch)}/{len(subset)} done", flush=True)
                f.flush()

    print(f"Main subset complete: {n_written} rows written, "
          f"{n_technical_fail} technical failures -> {out_path}", flush=True)

    # -- manipulation check --
    print("\nRunning non-binary manipulation check...", flush=True)
    mc_prompts_info = get_manipulation_check_prompts()
    mc_path = f"pilot_manipulation_check_{model_key}.csv"
    with open(mc_path, "w", newline="") as f:
        mc_writer = csv.DictWriter(f, fieldnames=["persona_id", "name", "country", "raw_response"])
        mc_writer.writeheader()
        prompts_only = [p[3] for p in mc_prompts_info]
        results = generate_batch(tokenizer, model, model_id, prompts_only)
        for (persona_id, name, country, _), (text, _) in zip(mc_prompts_info, results):
            print(f"  {name} ({country}): {text!r}", flush=True)
            mc_writer.writerow(dict(persona_id=persona_id, name=name, country=country,
                                     raw_response=text.strip()))
    print(f"Manipulation check written -> {mc_path}", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        print("Usage: python pilot_inference.py [llama|qwen]")
        sys.exit(1)
    run_pilot(sys.argv[1])
    print("\nPilot run complete.")
