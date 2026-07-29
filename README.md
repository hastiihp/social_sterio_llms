# stereotype_llm_paper

Publication-stage project for the stereotype-based opinion attribution study.
Old pipeline versions (v4-v9) live in the original project folder and are referenced, not copied, here.

## Provenance so far

| File | Produced by | Notes |
|---|---|---|
| data/names.csv | data/build_dataset.py | 60 names, 20 countries x 3 genders, validation tiers logged |
| data/topics.csv | data/build_dataset.py | 7 topics; 4 new topic statements are DRAFT pending review |
| data/personas.csv | data/build_dataset.py | 5,400-row canonical grid |
| data/prompts.csv | data/render_prompts.py | 75,600 rows: personas x topics x 2 response conditions, friend_v1 template, no cues |
| analysis_plan.md | manual, this session | pre-pilot draft, freeze after pilot acceptance criteria met |

## Next steps
1. Confirm/edit draft topic statements in data/topics.csv, rerun render_prompts.py if changed
2. Set up cluster env (Python >=3.10) and run 6-model smoke test
3. Pilot: Llama-3.1-8B + Qwen3-8B, subset per analysis_plan.md pilot spec
4. Review pilot against acceptance criteria in analysis_plan.md Section 12
5. Full 6-model run
