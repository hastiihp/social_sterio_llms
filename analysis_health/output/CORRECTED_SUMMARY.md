# Corrected Summary: Health-Conversation Framing vs. Original

**Status note on this document, read first:** you asked for this to use "the exact
'Concise Results' and 'Cautious Discussion' wording already drafted in AUDIT_REPORT.md
Section J." `AUDIT_REPORT.md` does not exist anywhere in this project (checked via a
full-project search, `.git/` excluded) -- only its verified numerical *outputs* survive,
at `analysis_health/audit/output/*.csv` (the script that produced them,
`run_adversarial_audit.py`, is also gone; only a `.pyc` cache remains). I flagged this
gap when you first gave the H1-H7 instructions and am flagging it again here rather than
fabricating "pre-drafted" wording. Every number below is verified either against those
audit CSVs directly, or freshly computed and cross-checked against them where the audit
provided a ground-truth value. The prose is my own synthesis, written in the spirit of
your "Concise Results / Cautious Discussion" structure, not a reproduction of a document
that doesn't exist. Please review before this goes to your supervisor.

All numbers regenerated 2026-07-24 from `analysis_health/01_compare_health_vs_original.py`,
`02_compare_by_condition.py`, `03_deepseek_health_diagnosis.py`, `04_ranking_robustness.py`,
run end-to-end against `results/` and `results_health/` only. `analysis/` and
`master_results.csv` were not touched; no new inference was run.

---

## Concise Results

**Sample.** 180-persona pilot subset (Germany, Brazil, Nigeria, South Korea x lawyer,
registered nurse, truck driver, farmer, computer programmer), matched 1:1 between the
original and health-conversation framings on the full canonical key (model, persona_id,
country, profession, gender, age, topic, response_condition). Merge validated
`one_to_one`: 12,600 rows per framing, 0 unmatched on either side.

**Condition A (forced-choice, primary evidence).** Every model's mean rating shifts
downward under the health framing, all persona-clustered p < 1e-30:
llama -0.174 (p=7.6e-77), gemma -0.185 (p=8.3e-52), qwen -0.156 (p=3.2e-22),
ministral -0.098 (p=2.4e-31), all n=1,260 pairs / 180 persona clusters, 100% valid
in every model. DeepSeek has 0 valid Condition-A ratings in the health framing (see
DeepSeek section below), so no shift is estimable for it.

**Condition B (optional NA, descriptive only -- see labeling note below).** llama/gemma
have no valid-NA option in the prompt design used here and are 100% valid in both
framings (shift -0.034, -0.134). qwen and ministral do have a working NA option, and the
"answered" subset is small and framing-dependent: qwen 80/1,260 = 6.35% valid pairs,
ministral 209/1,260 = 16.59%. Within those small selected samples, qwen's shift is
+0.325 (p=4.5e-11) -- the OPPOSITE sign from its Condition-A shift -- while ministral's
is -0.062 (p=2.5e-4), same sign as Condition A. This sign flip in qwen is the clearest
evidence in this dataset that Condition-B comparisons are not measuring the same thing
as Condition-A comparisons; see Discussion.

**Ministral abstention (Condition B).** Health framing reduces Ministral's abstention
rate substantially, but the correct comparison depends on holding both sample and
denominator fixed:

| sample | denominator | orig | health | shift |
|---|---|---|---|---|
| full 5,400 personas | all rows | 41.73% | 24.26% | -17.48pp |
| full 5,400 personas | Condition B only | 83.47% | 48.51% | -34.95pp |
| 180-persona pilot | all rows | 41.71% | 24.80% | -16.91pp |
| 180-persona pilot | Condition B only | 83.41% | 49.60% | -33.81pp |

Pilot / Condition-B-only, persona-clustered: shift = -33.81pp, p=4.6e-167. This is
concentrated in specific topics, not spread evenly: climate change -81.1pp, economic
redistribution -64.4pp, immigration -56.1pp, vs. gender equality -2.8pp and religion and
secularism -0.6pp (coefficient of variation of the per-topic shift = -0.97).

Qwen's abstention shift, by contrast, is small and in the opposite direction: +1.59pp
(pilot, paired), p=0.071 -- not conventionally significant, and consistent with the
Condition-B sample-composition effect described above rather than a clear opinion shift.

**Rankings (Condition A only, persona-clustered).** Profession rankings are highly
stable across the framing change for gemma, qwen, and ministral (Spearman r = 1.00,
0.9747 exact permutation p=0.017-0.033), and moderately stable for llama (r=0.872,
p=0.10). Truck driver ranks bottom in every model, in both framings. Country rankings
(n=4) cannot reach conventional significance under any circumstance -- see Discussion --
but descriptively: Brazil ranks top for llama, gemma, ministral in both framings; qwen's
top country flips from Germany (original) to Nigeria (health).

**DeepSeek.** Valid Condition-A ratings drop from 63/75,600 (original, already a low
rate) to 0/75,600 (health) -- a complete drop. A conservative compact-text parser
recovers 14.86% (11,234/75,600) of health-condition DeepSeek responses as containing a
plausible rating; the remainder show refusal or other noncompliance patterns. The cause
of the compact-text formatting (tokenizer, chat template, or decoding) is not established
from available logs. (Supplementary note, not required by the audit but worth flagging:
every one of the 11,234 recovered ratings is the digit "4" -- this pattern itself is not
explained by available data either, and should not be treated as evidence about DeepSeek's
substantive opinions.)

---

## Cautious Discussion

**What changes under the health framing, and what doesn't.** Absolute rating levels
move down consistently across every model with a usable Condition A (llama, gemma, qwen,
ministral), with clustered p-values far below any reasonable threshold. Relative rankings
-- which profession or country a model rates higher or lower than another -- are largely
preserved for profession (3 of 4 models at r>=0.97, llama at r=0.87), meaning this looks
like a fairly uniform downward shift riding on a stable underlying pattern, not a
reshuffling of which groups get rated how. Both findings can be, and appear to be, true
at once; they are not in tension.

**Condition B evidence is descriptive only, not primary, for qwen and ministral.** For
these two models specifically, "Condition B" is a self-selected sample: only respondents
who chose to answer rather than abstain appear in it, and which respondents that is
changes between framings (Ministral's Condition-B abstention rate itself drops from
83% to 50% under the health framing -- a huge compositional shift in who's even in the
comparison). Qwen's Condition-B rating shift flips sign relative to its own Condition-A
shift; that flip is best read as a symptom of this selection effect, not as a genuine
reversal of qwen's opinions. No number drawn only from Condition B should be presented as
headline evidence for either model; Condition A is the primary comparison throughout this
document for exactly this reason. llama and gemma have no valid NA option in this design,
so their Condition-B numbers don't carry this selection risk -- but "no valid NA option"
is not the same claim as "the model always produces a usable answer": malformed or
refused output remains possible regardless of condition, and DeepSeek is the clearest
existing example of that distinction mattering.

**Ministral's abstention drop is real but its size is denominator-dependent.** The
83%->50% (pilot, Condition-B-only) and 42%->24% (full dataset, all-rows) figures describe
the same underlying phenomenon at different denominators -- one restricted to the subset
of prompts where abstaining was even a live option, one averaged over all prompts
including the majority (Condition A) where it structurally wasn't. Neither number is
"the" effect size; both are true simultaneously and answer different questions. The
topic concentration (climate change and economic redistribution driving most of the
shift, religion and secularism barely moving) suggests this isn't a uniform "health
framing makes the model more willing to answer" effect, but something more specific to
which topics get reframed as less contentious in a health-conversation context --
this dataset can describe that pattern but can't explain its cause.

**Country rankings (n=4) cannot be treated as formally significance-tested.** With only
4 countries, the smallest possible two-sided exact permutation p-value is 2/24 = 0.083,
achieved only under perfect rank agreement -- p<.05 is mathematically unreachable at this
sample size no matter how strong the true agreement is. This is a structural ceiling from
having 4 categories, not evidence against country-ranking stability. The bootstrap
rank-position probabilities (`ranking_robustness_bootstrap.csv`) describe which country
tends to rank top/bottom across persona-resampled refits and are the more informative
number here, but should be read as descriptive, not as a hypothesis test.

**DeepSeek's health-framing behavior is a formatting failure, not a resolved causal
finding.** The 63->0 drop and the compact-text pattern that a narrow parser can recover
14.86% of are both reproducible facts about the raw text. What produces that
formatting -- tokenizer behavior, chat template handling, or decoding settings -- is not
established from the logs available in this project, and no claim about the mechanism
should be attributed to this analysis.

---

## Source tables

- `health_vs_original_summary.csv` -- per-model Condition A shift + Condition B
  abstention (H1-fixed sign, H6 merge-validated)
- `health_vs_original_by_condition.csv` -- Condition A vs B split, sample-size and
  validity-rate labeled per condition (H3, H7)
- `ministral_abstention_2x2.csv` -- full sample x denominator table (H2)
- `ministral_abstention_by_topic.csv` -- per-topic abstention shift
- `ranking_robustness_profession.csv`, `ranking_robustness_country.csv` -- per-level
  coefficients, both framings
- `ranking_robustness_exact_pvalues.csv` -- exact permutation Spearman tests (H4a)
- `ranking_robustness_bootstrap.csv` -- persona-bootstrap rank-position probabilities
  (H4b; implemented per spec -- persona-cluster resampling, 1,000 resamples, full model
  refit -- but not verifiable byte-for-byte against the audit's own bootstrap output
  since the audit's source script no longer exists, only a compiled `.pyc`)
- `deepseek_health_compact_parser_audit.csv` -- rows recovered by the compact-text parser
  (H5)
