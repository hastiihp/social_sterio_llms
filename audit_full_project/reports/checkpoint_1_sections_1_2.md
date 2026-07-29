# Full-project audit — Checkpoint 1

Scope of this checkpoint: Section 1 (complete project/experiment map) and
Section 2 (experimental design reconstructed from code and current files).
No scientific-result reproduction has been performed yet.

## Immediate traceability findings

1. The prior first-person/direct-prompt v9 experiment is not present. It is
   mentioned in `analysis_plan.md`, and some names are described as carried
   from it, but its prompts, inference code, raw outputs, parser, and analyses
   cannot be traced in this repository.
2. The current single-turn friend-frame experiment is traceable from
   `data/render_prompts.py` through `inference/full_inference.py`, the five
   `results/full_results_*.csv` files, `analysis/01_merge_dataset.py`, and the
   current analysis scripts/tables. However, inference-to-results relocation
   is a manual undocumented step: the inference script writes into its current
   working directory, while the retained files live in `results/`.
3. The health experiment has a prompt generator, inference script, five raw
   result files, and four current analysis scripts. All health materials are
   untracked in Git at this checkpoint, so Git history cannot establish when
   the 180-persona analysis subset was chosen or frozen.
4. `analysis_health/audit/output/` contains outputs from a prior adversarial
   audit, but the corresponding `run_adversarial_audit.py` source is absent.
   Only a compiled `__pycache__/run_adversarial_audit.cpython-313.pyc` remains.
   Those audit outputs are therefore not reproducible from checked-in source.
5. `logs/` is empty. There are no retained main, health, pilot, retry, recovery,
   or Falcon execution logs.
6. Pilot outputs, manipulation-check outputs, and prompt-reinforcement test
   outputs reported in `analysis_plan.md` are absent.
7. `data/build_dataset.py` writes to a hard-coded historical
   `/home/claude/stereotype_llm_paper/data/` path. It documents the design code
   but will not regenerate the current workspace files without manual path
   intervention.
8. `data/prompts.csv` labels the template `friend_v1`, while the inference
   results label it `friend_v2_explicit_gender`. The rendered prompt does
   contain the explicit gender label, so this is a provenance-label
   inconsistency, not a row-count discrepancy.
9. `inference/full_health_inference.py` expects
   `health_prompts_full.csv` in its current working directory and writes raw
   outputs there. The retained prompt is in `data_health/` and retained results
   are in `results_health/`; no orchestration/move script or log records the
   actual staging procedure.

## Inventory by role

- Design/source data: `data/names.csv`, `data/personas.csv`,
  `data/topics.csv`, `data/prompts.csv`,
  `data_health/health_prompts_full.csv`.
- Design/prompt generation: `data/build_dataset.py`,
  `data/render_prompts.py`, `data_health/health_render_prompts_full.py`.
- Inference: `inference/pilot_inference.py`,
  `inference/full_inference.py`, `inference/falconinferance.py`,
  `inference/full_health_inference.py`.
- Raw retained main outputs: five `results/full_results_*.csv` files.
- Raw retained health outputs: five
  `results_health/health_full_results_*.csv` files.
- Processed main dataset: `analysis/master_results.csv`.
- Main validation/analysis/figures: `analysis/01_merge_dataset.py` through
  `analysis/10_figures.py`, plus `_common.py` and `_style.py`.
- Health analysis: `analysis_health/01_compare_health_vs_original.py`,
  `02_compare_by_condition.py`, `03_deepseek_health_diagnosis.py`,
  `04_ranking_robustness.py`.
- Main outputs: `tables/*.csv`, `figures/*.png`.
- Health outputs: `analysis_health/output/*`.
- Untraceable prior health-audit outputs:
  `analysis_health/audit/output/*`.
- Documentation/configuration: `analysis_plan.md`, `README.md`,
  `MANIFEST.md`, `FALCON_EXCLUSION.md`, `requirements.txt`.
- Logs: directory exists but contains no files.
- Missing configuration forms: no `environment.yml`, `pyproject.toml`, or
  inference-specific environment lock.

## Experiment integrity and traceability table

“Rows” refers to the experiment's raw design/input population unless a narrower
analysis population is stated.

| Experiment | Exact code | Inputs | Outputs | Models | Personas / topics / conditions | Expected rows | Actual rows | Status |
|---|---|---|---|---|---|---:|---:|---|
| Prior first-person/direct-prompt v9 | **Absent** | **Absent** | **Absent** | Unknown | Historical design unknown from current repo | Unknown | 0 retained | **REPRODUCIBILITY FAILURE — untraceable** |
| Current single-turn friend-frame full inference (“original” in health scripts) | `data/render_prompts.py`; `inference/full_inference.py` | `data/personas.csv`, `data/topics.csv`, `data/prompts.csv` | five `results/full_results_*.csv` | Llama, Gemma, Qwen, Ministral, DeepSeek | 5,400 / 7 / A+B | 75,600/model; 378,000 total | 75,600/model; 378,000 total | Traceable with undocumented staging and template-label mismatch |
| Condition A forced response | same prompt/inference code | A rows of `data/prompts.csv` | A rows in five main result files/master | five main models | 5,400 / 7 / A | 37,800/model | 37,800/model | Count-complete; detailed prompt/inference audit deferred |
| Condition B optional response | same prompt/inference code | B rows of `data/prompts.csv` | B rows in five main result files/master | five main models | 5,400 / 7 / B | 37,800/model | 37,800/model | Count-complete; detailed prompt/inference audit deferred |
| Master merge/validation | `analysis/01_merge_dataset.py`, `02_validate_dataset.py` | five main result files | `analysis/master_results.csv`; console validation | five | 5,400 / 7 / A+B | 378,000 | 378,000 | Traceable; validation has no retained log |
| H1 primary/pooled ratings | `analysis/05_hypothesis_models.py`, `08_variance_ranking.py`, BH additions from `05e_bh_correction.py` | master dataset | `hypothesis_model_*.csv`, `variance_ranking.csv`, BH summary | four primary; DeepSeek separate exploratory/per-model | full main design; primary pooled scope A | 151,200 primary A rows for four models | Source has 151,200 eligible A rows | Traceable; values deferred to confirm pass |
| H2 abstention | `analysis/06_abstention_analysis.py`, `05e_bh_correction.py` | master B rows | rate tables, `abstention_model_qwen_ministral.csv` | descriptive five; fit Qwen+Ministral | 5,400 / 7 / B | 189,000 B rows descriptive; 75,600 Qwen+Ministral before model filtering | Counts available as expected | Traceable; planned interaction model documented as non-estimable/omitted |
| H3 forced-versus-optional pairing | `analysis/07b_paired_comparison.py` | master A+B | paired summary; matched-cell and topic-stratified tables | Llama, Gemma, Qwen, Ministral; DeepSeek skipped | 5,400 / 7 / paired A-B | 37,800 matched cells/model | 37,800/model in each retained matched table | Traceable |
| Clustered-standard-error analyses | `05_hypothesis_models.py`, `05c_topic_specific_models.py`, `05d_country_set_robustness.py`, `06_abstention_analysis.py`, `07b_paired_comparison.py`, `08_variance_ranking.py`; health `01`, `02`, `04` | master or paired health/original sources | corresponding coefficient/test tables | varies by script | persona clusters over applicable cells | Analysis-specific | Outputs retained | Traceable |
| Mixed-model analyses | `analysis/05_hypothesis_models.py` | strict-valid main ratings | `hypothesis_model_*.csv` | five per-model attempts | up to 5,400 persona clusters/model | validity-dependent | tables retained | Traceable; convergence validation deferred |
| Ordinal robustness | `analysis/05b_ordinal_robustness.py`, `05b2_proportional_odds_by_topic.py` | master strict-valid Condition A for four models | per-model ordinal coefficient/pseudo-R²/PO tables; topic PO table | Llama, Gemma, Qwen, Ministral | 5,400 / 7 / A | 37,800/model before outcome validity | Current source population traceable | Traceable; numerical confirmation deferred |
| Topic-specific analyses | `analysis/05c_topic_specific_models.py` | master | five topic-specific CSVs | four rating models; Qwen+Ministral abstention | 5,400/topic/model | 5,400 per topic/model before validity | Summary covers all seven topics per included model | Traceable |
| Country robustness | `analysis/05d_country_set_robustness.py` | master + country-set mapping from personas | two country robustness CSVs | four primary models | 2,700 personas/country set; 7 topics; A | 75,600 rows/set across four models | Source design supports expected counts | Traceable |
| Profession/demographic effects and descriptives | `analysis/04_descriptives.py`, `05_hypothesis_models.py`, `05c`, `05d`, `08` | master | descriptives, coefficient, spread, robustness, variance tables | varies | full main design or documented subsets | Analysis-specific | Outputs retained | Traceable |
| Cross-model agreement | `analysis/07_cross_model_agreement.py` | master Condition A | main-four agreement matrices/table; DeepSeek-pair table | primary four; DeepSeek separate | 5,400 / 7 / A | 37,800 matched cells per main pair | 37,800 stated/available per main pair | Traceable |
| DeepSeek main compliance/diagnostic | `analysis/03_compliance_table.py`, `09_deepseek_report.py`, parts of `07`/`08` | master/raw DeepSeek result | compliance and separate agreement/variance outputs | DeepSeek | 5,400 / 7 / A+B | 75,600 | 75,600 | Traceable |
| Falcon attempt/exclusion | `inference/falconinferance.py`; `FALCON_EXCLUSION.md`; plan note | `data/prompts.csv` intended | no retained raw result or log | Falcon-H1 | intended 5,400 / 7 / A+B | 75,600 intended | 0 usable retained | **PARTIAL — code/docs exist, failure evidence/logs absent** |
| Original Llama/Qwen pilot | `inference/pilot_inference.py` | filtered main prompts | expected `pilot_results_llama.csv`, `pilot_results_qwen.csv` | Llama, Qwen | 180 / 7 / A+B | 2,520/model | 0 retained | **REPRODUCIBILITY FAILURE — outputs absent** |
| Gender manipulation check | `inference/pilot_inference.py` plus plan claims | generated pilot check prompts | expected `pilot_manipulation_check_*.csv` | Llama, Qwen | code currently generates 8 prompts/model; plan reports a different 60-persona check | code expectation 8/model; plan reports 60-persona design | 0 retained | **REPRODUCIBILITY FAILURE / design-record conflict** |
| Prompt-reinforcement comparison | no executable/current source located | absent matched prompt variants/raw outputs | absent | plan says six | plan says 12 cases/version/model | 144 generations if 12 × 2 variants × 6 | 0 retained | **REPRODUCIBILITY FAILURE — narrative only** |
| Full health-conversation inference | `data_health/health_render_prompts_full.py`; `inference/full_health_inference.py` | main personas/topics; health messages CSV | five health result CSVs | five main models | 5,400 / 7 / A+B | 75,600/model; 378,000 total | 75,600/model; 378,000 total | Count-complete; untracked and staging/log lineage incomplete |
| Health-versus-original paired comparison | `analysis_health/01_compare_health_vs_original.py` | five main + five health raw files, filtered in code | `health_vs_original_summary.csv`; Ministral 2×2/topic tables | five; rating inference limited by validity | 180 / 7 / A+B | 2,520/model/family; 12,600 matched across five | 2,520/model/family; 12,600 total matched design | Traceable from current code; subset freeze timing unverifiable |
| Condition-specific health comparison | `analysis_health/02_compare_by_condition.py` | same filtered raw files | `health_vs_original_by_condition.csv` | five | 180 / 7 / separate A and B | 1,260/model/condition/family | 1,260/model/condition/family | Traceable |
| Health abstention and Ministral topic analysis | `analysis_health/01_compare_health_vs_original.py` | filtered and full original/health raw files | `ministral_abstention_2x2.csv`, `ministral_abstention_by_topic.csv` | Ministral primarily; summary includes five | 180 pilot and full 5,400 scopes | 1,260 B rows/model in pilot; 37,800 B rows/model full | Present as expected in sources | Traceable |
| Health ranking robustness | `analysis_health/04_ranking_robustness.py` | original/health raw files filtered to pilot A | four ranking CSVs | Llama, Gemma, Qwen, Ministral | 180 / 7 / A | 1,260/model/family before validity | 1,260/model/family for these four | Traceable |
| DeepSeek health-output diagnostic | `analysis_health/03_deepseek_health_diagnosis.py` | full original and health DeepSeek outputs | compact-parser audit CSV | DeepSeek | 5,400 / 7 / A+B | 75,600/family | 75,600/family | Traceable |
| Prior health adversarial audit | **source missing** (`run_adversarial_audit.pyc` only) | inferable original/health files | 13 CSV/JSON files under `analysis_health/audit/output/` | five / subsets | mixed full and 180-persona scopes | inferable only | outputs exist | **REPRODUCIBILITY FAILURE — no source** |
| Figures 2–7 | `analysis/10_figures.py` | master plus selected current tables | six PNGs | varies by figure | main-design scopes | N/A | six files | Traceable |

## Section 2 — design reconstructed from code

### Main design

The design code constructs:

- 20 countries from 60 country×gender name rows;
- 30 explicit professions;
- 3 genders: male, female, neutral;
- 3 ages: 25, 45, 65;
- 7 explicit topics;
- 2 conditions: `A_forced`, `B_optional`.

Exact calculation:

```text
personas = 20 countries × 30 professions × 3 genders × 3 ages
         = 5,400

prompts per persona = 7 topics × 2 conditions = 14

rows per model = 5,400 × 14 = 75,600

five-model total = 75,600 × 5 = 378,000
```

Observed current files:

| Model | Actual raw rows | A | B | Unique personas | Result |
|---|---:|---:|---:|---:|---|
| Llama | 75,600 | 37,800 | 37,800 | 5,400 | Count confirmed |
| Gemma | 75,600 | 37,800 | 37,800 | 5,400 | Count confirmed |
| Qwen | 75,600 | 37,800 | 37,800 | 5,400 | Count confirmed |
| Ministral | 75,600 | 37,800 | 37,800 | 5,400 | Count confirmed |
| DeepSeek | 75,600 | 37,800 | 37,800 | 5,400 | Count confirmed |
| Master total | 378,000 | 189,000 | 189,000 | 5,400 shared IDs | Count confirmed |

`data/prompts.csv` itself contains 75,600 rows with the same factor
cardinalities. This verifies design size only; persona/prompt key integrity is
reserved for Section 3 at the next checkpoint.

### Health design and 180-persona comparison subset

The health prompt generator is a **full-design generator**, not a 180-persona
pilot generator:

```text
5,400 personas × 7 topics × 2 conditions = 75,600 rows/model
```

`data_health/health_prompts_full.csv` and each of the five health raw result
files contain exactly 75,600 rows, 5,400 personas, all seven topics, and both
conditions.

The health comparison subset is selected later in analysis code using:

```text
countries = Germany, Brazil, Nigeria, South Korea              (4)
professions = lawyer, registered nurse, truck driver,
              farmer, computer programmer                      (5)
genders = 3
ages = 3

pilot personas = 4 × 5 × 3 × 3 = 180
pilot rows/model/family = 180 × 7 × 2 = 2,520
```

Independent count checks found exactly 2,520 rows per model in both the
original and health families, split 1,260 A and 1,260 B. All five original
model files use the same 180 persona IDs; all five health files use the same
180; and the original and health persona sets match exactly.

The same four-country/five-profession constants appear in all four current
health analysis scripts and in the original `pilot_inference.py`. No current
health script was found using a different subset.

What cannot be verified is **temporal freezing**: there is no dedicated subset
manifest/file, the health tree is untracked, and there is no run log or commit
history establishing that these constants were fixed before health outcomes
were inspected. Therefore the subset is currently consistent but not
provenance-frozen.

## Checkpoint-1 verdict

The current main and health raw datasets have the intended design-level row
counts. The current single-turn and health experiments are largely traceable
to code and inputs, but the project does **not** yet have complete experiment
traceability. The absent v9 experiment, missing pilot/manipulation/reinforcement
artifacts, missing Falcon/run logs, missing health-audit source, untracked
health tree, and undocumented inference-output staging are reproducibility
failures that must remain open for later severity assessment.

