# Dependency and reproducibility audit

## Canonical lineage

Main: `data/{names,topics,personas}.csv` → `data/prompts.csv` →
`results/full_results_{model}.csv` → `analysis/01_merge_dataset.py` →
`analysis/master_results.csv` → analysis scripts → `tables/` →
`analysis/10_figures.py` → `figures/`.

Health: `data/personas.csv` + `data/topics.csv` →
`data_health/health_render_prompts_full.py` → ignored canonical prompt
`data_health/health_prompts_full.csv` → `inference/full_health_inference.py` →
`results_health/health_full_results_{model}.csv` → `analysis_health/01-04` and
`analysis_context/01,03,04,05` → `analysis_health/output/` and
`analysis_context/output/`.

Context: `data/personas.csv` + `data/topics.csv` →
`data_context/context_render_prompts_full.py` → ignored canonical prompts →
`inference/context_full_inference.py` → `results_context/` →
`analysis_context/01-05` → `analysis_context/output/`.

## Main analysis graph

| Script | Inputs | Outputs | Downstream users |
|---|---|---|---|
| `01_merge_dataset.py` | five `results/full_results_*.csv` | `analysis/master_results.csv` | all main scripts; context script 05 |
| `02_validate_dataset.py` | master through `_common.load_master` | console only | human gate only; does not exit nonzero on failure |
| `03_compliance_table.py` | master | `table1_compliance.csv` | documentation |
| `04_descriptives.py` | master | five `descriptives_*.csv` | documentation |
| `05_hypothesis_models.py` | master | six `hypothesis_model_*.csv` | `05e`; figure 5 |
| `05b_ordinal_robustness.py` | master | four families of ordinal CSVs | documentation |
| `05b2_proportional_odds_by_topic.py` | master | `ordinal_proportional_odds_by_topic.csv` | documentation |
| `05c_topic_specific_models.py` | master | five topic-analysis CSVs | documentation |
| `05d_country_set_robustness.py` | master + `data/personas.csv` | two country-set CSVs | documentation |
| `06_abstention_analysis.py` | master | rate CSVs + abstention model | `05e`; figure 3 indirectly recomputes from master |
| `05e_bh_correction.py` | model/abstention CSVs | rewrites them + BH summary | final tables; must run last |
| `07_cross_model_agreement.py` | master | pair and matrix CSVs | figure 6 |
| `07b_paired_comparison.py` | master | matched cells and summaries | documentation |
| `08_variance_ranking.py` | master | `variance_ranking.csv` | figure 7; context script 05 |
| `09_deepseek_report.py` | master | DeepSeek compliance CSV | documentation |
| `10_figures.py` | master; pooled coefficients; agreement matrices; variance ranking | figures 2–7 | publication |

No circular dependency was found. `05e` is an intentional in-place postprocessor,
so rerunning producers after it creates stale/BH-incomplete tables.

## Health and context graph

`analysis_health/01`, `02`, `03`, and `04` read canonical raw main/health results
directly and write their named CSVs under `analysis_health/output/`. Nothing in code
consumes most of these outputs; they are publication/documentation endpoints.

`analysis_context/_common.py` defines all canonical result paths. Script 01 writes
per-context matched summaries, by-condition tables, abstention tables, ranking tables,
and bootstraps at pilot and full scope. Script 02 consumes raw frames and writes
cross-context agreement/clustering tables. Script 03 consumes all five raw conditions
and writes abstention stability tables. Script 04 consumes DeepSeek raw files and
writes the compact-parser diagnosis. Script 05 consumes raw context frames plus
`tables/variance_ranking.csv` and `analysis/master_results.csv`, then writes combined
variance CSVs, dominant-factor summaries, and nine figures.

## Figure reproduction map

| Figure | Raw source | Script/intermediate | Final |
|---|---|---|---|
| Main rating distributions | main raw → master | `01`, `10` | `figures/fig2_rating_distributions.png` |
| Main abstention by topic | main raw → master | `01`, `10` | `figures/fig3_abstention_by_topic.png` |
| Main country-topic heatmap | main raw → master | `01`, `10` | `figures/fig4_country_topic_heatmap.png` |
| Pooled coefficients | main raw → master | `05`, `05e`, `10`; `hypothesis_model_pooled.csv` | `figures/fig5_pooled_coefficients.png` |
| Agreement matrix | main raw → master | `07`, `10`; two matrix CSVs | `figures/fig6_agreement_matrix.png` |
| Variance explained | main raw → master | `08`, `10`; `variance_ranking.csv` | `figures/fig7_variance_explained.png` |
| Context variance figures | corresponding raw result files | context `05`; combined variance CSV | `analysis_context/output/variance_explained_*.png` |

All present published figures have a code path to canonical raw inputs. Reproduction
is not currently executable from the active Python environment because required
packages are absent. Main reconstruction also depends on ignored
`analysis/master_results.csv`, although it can be regenerated from tracked main raw
results. Context sources/scripts are untracked and therefore not reproducible from
the recorded commit alone.

## Stale/orphan/duplicate assessment

- `health_staging/` is an abandoned duplicate tree: prompt, five full result files,
  renderer, and inference script are byte-identical to canonical copies.
- Pilot context outputs are unsuffixed while full outputs use `_full5400`; code and
  documentation distinguish them, but filename-only consumers can mistake pilot for
  default/current output.
- Most CSVs are terminal reporting artifacts and have no scripted downstream user;
  this is expected, not automatically stale. No provenance manifest covers the
  context outputs, so freshness cannot be established from dependency metadata.
- Old pilot/manipulation/reinforcement raw outputs and the original health audit
  runner are documented as missing historical artifacts.
- `README.md` and `analysis_plan.md` still describe a pre-pilot/six-model future state.
