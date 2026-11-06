# LLM Reliability Lab

Independent AI-safety research project: a reproducible harness for studying the **behavioral reliability of LLMs** under controlled
conditions: instruction conflicts, misleading context, uncertainty/hallucination, adversarial (prompt-injection) robustness, context
sensitivity, and response stability.

> **Status (honest):** the software is implemented and tested with *mocked* responses. **No real-model experiments have been run, there are
> no research findings yet, and the 117-case seed dataset is model-generated and human-unreviewed** (see `datasets/dataset_card.md`).
> The study itself is the work you do next.

## What is here
| Area | Where |
|---|---|
| Model adapters (Claude, OpenAI, Gemini, local Hugging Face, mock) | `src/reliability_lab/models/` |
| Six experiments with rubrics | `src/reliability_lab/experiments/` |
| Scoring, metrics, statistics, failure categories | `src/reliability_lab/evaluation/` |
| Resumable runner, result store, report generator | `src/reliability_lab/pipeline/` |
| Seed dataset + card + builder | `datasets/`, `scripts/build_dataset.py` |
| Dashboard (Streamlit, dark) | `dashboard/app.py` |
| Research docs and paper template | `research/` |
| Tests (97, all offline) | `tests/` |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # or: pip install -e ".[providers,viz,dev]"
cp .env.example .env                                    # Windows: copy .env.example .env   (then add keys)

python -m pytest                                        # offline tests, no API keys needed
python scripts/validate_dataset.py                      # schema + quality checks

# Offline end-to-end smoke test with the synthetic mock client (no cost, no keys)
python scripts/run_all.py --smoke --models mock --yes
python scripts/analyse_results.py --results-dir results_smoke --no-embeddings
streamlit run dashboard/app.py                          # choose "results_smoke" in the sidebar
```
Mock output is synthetic: it exercises the pipeline and says nothing about any model. Reports and the dashboard flag it.

## Running real experiments
1. **Keys:** put the keys you have in `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`; optional `HF_TOKEN`). Models without keys are skipped.
2. **Model ids (manual step):** ids change often. Run `python scripts/list_models.py` and fix `configs/models.yaml`. The OpenAI and Gemini ids shipped there are
   **unverified placeholders** (documentation snapshots disagreed on 2026-11-06); the Claude id comes from Anthropic's published list. Enable/disable models with `enabled:`.
   For an open-weight model set `open_weight.model_id` to an instruction-tuned Hugging Face chat model that fits your hardware, set `enabled: true`, and `pip install torch`.
3. **Prices (optional):** add `price_per_mtok_input/output` from provider pricing pages to get cost estimates and `--max-cost` enforcement.
4. **Look before you spend:**
   ```bash
   python scripts/run_all.py --dry-run                             # plan: request count, token estimate, cost if priced
   python scripts/run_experiment.py -e misleading_context --models claude --limit 5     # pilot
   python scripts/run_experiment.py -e misleading_context uncertainty --models claude openai gemini
   python scripts/run_all.py --models claude openai gemini --max-requests 1500 --max-cost 15
   ```
5. **Analyse:** `python scripts/analyse_results.py` → `results/processed/scored.*`, `results/statistics/{metrics,comparisons}.csv|report.md`, `results/figures/*.png`,
   `results/processed/review_queue.csv`.
6. **Human review:** label rows in the review queue; save as `results/human_labels.jsonl` (`{"item_id","model_key","repeat","label":"pass|fail"}` per line); re-run analysis
   (agreement and κ are reported; `--human-override` lets labels replace automatic ones).
7. **Paper:** `python scripts/generate_report.py` fills the results block of `research/research_paper_template.md` → `research/paper_draft.md`. Write the interpretation yourself.

### Command reference
| Command | Purpose |
|---|---|
| `scripts/run_experiment.py -e <exp…\|all> [--models …] [--limit N] [--smoke] [--dry-run] [--yes] [--max-requests N] [--max-cost USD] [--results-dir D] [--verified-only]` | run experiments |
| `scripts/run_all.py` (same flags minus `-e`) | run all six |
| `scripts/analyse_results.py [--results-dir D] [--no-embeddings] [--judge] [--human-override]` | score + stats + figures |
| `scripts/validate_dataset.py` | dataset validation (exit code 1 on error) |
| `scripts/build_dataset.py` | regenerate the seed set (deterministic) |
| `scripts/promote_verified.py [--dry-run]` | move human-verified cases to `datasets/processed/` |
| `scripts/list_models.py` | check configured ids against live provider model lists |
| `scripts/generate_report.py` | populate paper draft |
Experiments: `instruction_conflicts misleading_context uncertainty adversarial context_sensitivity response_stability`.

## Safety rails against accidental spend
* `--dry-run` sends nothing; `--smoke` = 3 cases, 1 repeat, separate `results_smoke/` directory.
* Confirmation prompt above `confirm_above_requests` (150) unless `--yes`; non-interactive runs refuse without `--yes`.
* Hard caps: `--max-requests` (default 2000) and `--max-cost` (enforced when prices are set). A stopped run resumes where it left off.
* Per-model rate limiting, exponential-backoff retries on 429/5xx/timeouts, request timeouts, API error logging.
* Full default run ≈ **417 requests per model** (18+18+24+21+108+228 with 3 stability repeats).

## Provider caveat: sampling parameters
Not every SDK/API accepts `temperature` or `seed` (e.g. Anthropic SDK 2.6.0 exposes no `temperature` on `messages.create`). Adapters send only what is accepted and store
what was actually applied in `params_applied` for every response. Check it before interpreting Exp. 6 across models.

## Data integrity
* `results/raw/<experiment>__<model>.jsonl` is **append-only**; each line stores run id, timestamp, provider, exact model id, prompt/dataset versions, generation parameters, prompt,
  response, tokens, latency, cost, attempts and error.
* Resume: completed request keys are skipped, errored ones retried; changing the prompt version, dataset version, model id or params creates new keys.
* Scores are derived (`results/processed/`) and can be regenerated at any time, e.g. after improving a rubric (log such changes).

## Configuration
`configs/models.yaml` (models), `configs/experiments.yaml` (repeats, temperature, token limits, versions, budget thresholds), `configs/evaluation.yaml`
(statistics, semantic model, optional judge, human-label path).

## Extending
* **Add a model/provider:** subclass `ModelClient` (`_call` only), register in `models/__init__.py::build_client`, add a YAML entry.
* **Add cases:** edit/extend `scripts/build_dataset.py` or write JSONL by hand into `datasets/generated/`, run `validate_dataset.py`, bump `dataset_version`.
* **Add an experiment:** implement `Experiment.build_items/score`, add metrics to `evaluation/metrics.py`, register in `experiments/__init__.py`.

## Research documents
`research/research_questions.md` (hypotheses to freeze), `methodology.md` (design, metrics, statistics, critical review), `literature_review.md` (verify before citing),
`failure_taxonomy.md`, `limitations.md`, `experimental_results.md` (empty by design), `research_checklist.md`, `research_paper_template.md`.

## Known limitations (short list)
Tiny model-generated unreviewed dataset; heuristic scoring needing human validation; famous-fact ceiling effects; single prompt templates; wide CIs. Details in
`research/limitations.md`. No claims about model internals or overall safety are supported by this repository.

## Requires your manual action
API keys; verifying model ids; (optionally) prices; choosing an open-weight model; human verification of the dataset and review labels; reading and verifying references.

## License
MIT (see `LICENSE`).
