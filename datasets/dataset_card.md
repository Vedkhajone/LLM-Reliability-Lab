# Dataset Card: LLM Reliability Lab seed set (v0.1.0)

## Summary
A small, original, **model-generated and human-unreviewed** seed dataset for six behavioural-reliability experiments.
It is a *pilot instrument* for developing the pipeline and for exploratory results. It is **not** a validated benchmark.

| Experiment | File | Base cases | Prompts per model per repeat |
|---|---|---:|---:|
| 1 Contradictory instructions | `generated/instruction_conflicts.jsonl` | 18 | 18 |
| 2 Misleading context | `generated/misleading_context.jsonl` | 18 | 18 |
| 3 Uncertainty / hallucination | `generated/uncertainty.jsonl` | 24 (18 unanswerable + 6 answerable controls) | 24 |
| 4 Adversarial prompts | `generated/adversarial.jsonl` | 21 (3 tasks × 6 attacks + 3 no-attack controls) | 21 |
| 5 Context sensitivity | `generated/context_sensitivity.jsonl` | 18 | 108 (× conditions C0–C5) |
| 6 Response stability | `generated/response_stability.jsonl` | 18 | 76 (variants; ×3 repeats by default) |

117 base cases, 265 distinct prompts, 417 requests per model with default repeats. The target of 600 cases is a goal for later
versions; the generator and schema are built to expand, but quality is prioritised over count.

## Provenance and review status
* **Origin:** every case is `origin: "model_generated"`: written by an AI assistant (Claude Code) from the author's specification
  by `scripts/build_dataset.py`. Cases are deterministic outputs of that script.
* **Review:** every case is `human_review_status: "unreviewed"`. No case has been verified by a human.
  `tests/test_dataset.py` fails if a case claims otherwise without going through `scripts/promote_verified.py`.
* **Separation:** `generated/` holds unreviewed cases; `processed/` holds only cases promoted after human verification
  (`origin: "human_verified"`). Verified cases override generated ones with the same id.
* **Ground truth:** factual answers are widely known facts. `ground_truth_source` currently says "standard reference works;
  human verification pending". Before reporting results, verify each answer against a cited source and record it.
* **Fictional entities** (Exp. 3) were invented to look plausible. They were *not* searched for; a coincidental real match is
  possible and must be checked by a human.

## Schema (`src/reliability_lab/schema.py`)
`id, category, question, context, system_prompt, ground_truth, expected_behavior, difficulty (easy|medium|hard),
evaluation_criteria[], acceptable_alternatives[], ground_truth_source, origin, human_review_status, dataset_version, tags[], spec{}`.
`spec` holds experiment-specific scoring parameters (rubric inputs), e.g. `checks_a/checks_b/acceptable` for Exp. 1,
`context_type/provenance/wrong_answer` for Exp. 2, `answerable/subtype` for Exp. 3, `canary/task_checks` for Exp. 4.

## Design properties
* Facts are not reused across experiments (tested), to keep experiments independent.
* Questions never contain their own answers (tested); paraphrases likewise.
* Exp. 1 includes *non-conflicting controls* (satisfiable constraints) and order-swapped pairs.
* Exp. 3 includes *answerable controls* so false abstention can be measured.
* Exp. 4 includes *no-attack controls* so task failure can be separated from attack effects. Payloads are benign canary tokens.

## Known limitations and biases
* **Tiny n.** 18–24 cases per experiment gives wide confidence intervals; only large effects are detectable.
* **Easy, famous facts** (Exp. 2, 5, 6) likely cause ceiling effects at baseline and are probably memorised by all evaluated models
  (benchmark-contamination risk is high, though the *combinations* with misleading contexts are novel).
* **Generator bias.** The cases were written by a Claude model; a Claude-family evaluated model may find them more natural
  (stylistic familiarity / self-preference). Treat cross-vendor comparisons with extra caution and consider having a
  non-Claude model or human re-author a subset.
* **English-only**, Western-centric facts; no domain-specialist content.
* **Heuristic scoring.** Rubrics are rule-based; borderline outputs are flagged `needs_review` for human adjudication.
* **Single phrasing per prompt template**; results are conditional on the templates in `src/reliability_lab/experiments/`.

## Versioning
Semantic version in `configs/experiments.yaml:run.dataset_version`, stamped on every case and every stored response.
Changing a case's text changes request keys, so reruns never silently mix versions. Bump the version when cases or rubrics change.

## Intended use
Pipeline development; exploratory comparisons; generating hypotheses and a failure taxonomy for a larger, human-verified set.
**Not** for: claims about overall model safety, leaderboard ranking, or training.
