# Experimental results

**Status: no real-model experiments have been run. This file intentionally contains no results.**

Numbers in `results_smoke/` (if present) come from an offline *mock* client and describe only the pipeline.

## How to populate this file
1. Verify dataset cases and freeze `research_questions.md` (commit + tag).
2. `python scripts/list_models.py`, then run the experiments (see README) and `python scripts/analyse_results.py`.
3. Copy the primary tables from `results/statistics/report.md` here; keep the figure files in `results/figures/`.
4. For each hypothesis H*, write: result (estimate + 95% CI), test (if any, adjusted p), and a one-sentence cautious interpretation.
5. Record run metadata below.

## Run metadata (fill in)
| Field | Value |
|---|---|
| Run dates | |
| Dataset version / fingerprint | |
| Prompt version | |
| Models (key → exact model id) | |
| Repeats / temperature per experiment | |
| Human-reviewed share of cases | |
| Rubric / code git commit | |

## Results by hypothesis (fill in after running)
| Hypothesis | Estimate (95% CI) | Test (adj. p) | Supported? | Notes |
|---|---|---|---|---|
| H1a | | | | |
| H1b | | | | |
| … | | | | |
