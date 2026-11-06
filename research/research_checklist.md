# Initial research checklist

## Before running real models
- [ ] Read the sources in `literature_review.md`; delete anything unverified; write a short gap analysis.
- [ ] Human-verify dataset cases (ground truth + rubric + acceptable alternatives); promote with `scripts/promote_verified.py`.
- [ ] Check fictional entities in Exp. 3 are not real (search them).
- [ ] Add ≥1 non-Claude-authored or human-authored subset to reduce generator bias.
- [ ] Freeze `research_questions.md`, rubrics (git tag, e.g. `prereg-v1`), dataset version and prompt version.
- [ ] `python scripts/list_models.py` → fix `configs/models.yaml` ids; record `verified: true`; add prices if you want cost tracking.
- [ ] Dry-run: `python scripts/run_all.py --dry-run` and read the plan.
- [ ] Pilot on 1 model with `--limit 5`; inspect raw outputs by hand; fix *tooling* bugs (not rubric thresholds to fit results). Log any rubric change.

## Running
- [ ] Same date window for all models; no changes to prompts/rubrics mid-run.
- [ ] ≥3 repeats for Exp. 1–5 final runs (set `repeats` in `configs/experiments.yaml`), keep Exp. 6 at ≥3.
- [ ] Keep `results/raw/` backed up (it is git-ignored by default).

## After running
- [ ] `scripts/analyse_results.py`; review `review_queue.csv`; label ≥ 50 rows (or all if fewer) with a human; report agreement/κ.
- [ ] Second annotator on a 20–30% subset; report κ.
- [ ] Validate any LLM judge against human labels before using it.
- [ ] Code failures for the taxonomy; report prevalence with CIs.
- [ ] Sensitivity checks: rerun scoring with human labels overriding; drop `needs_review` rows; compare.
- [ ] Fill `experimental_results.md`; draft paper from `research_paper_template.md` (`scripts/generate_report.py`).
- [ ] Write limitations from observed issues; state clearly what is exploratory.

## Scaling toward 600 cases
- [ ] Add cases per category via new builders or by hand; keep generated vs. verified separated; bump `dataset_version`.
- [ ] Deduplicate (`validate_dataset.py`), balance difficulty, add non-factual and domain-diverse items, vary prompt templates.
- [ ] Re-run power considerations: aim for ≥ 80–100 cases per primary comparison for moderate effects.
