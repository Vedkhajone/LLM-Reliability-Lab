# Limitations (living document)

## Study status
**No real-model experiments have been run.** The software has been tested only with mocked responses. This file lists limitations of the *design*;
extend it with observed limitations after real runs.

## Validity
* **Construct:** "hallucination", "abstention", "attack success" are measured by heuristics on text. They approximate, not equal, the constructs.
* **Internal:** single prompt templates; prompt-version and model-version effects are confounded with model identity unless runs are interleaved in time.
* **External:** small English-only synthetic dataset on famous facts; benign canary attacks. Findings will not transfer to realistic adversaries,
  specialised domains, other languages, tool-using agents or long-context settings.
* **Statistical conclusion:** ~20 cases per experiment → wide intervals, low power, multiplicity. Treat all results as exploratory.

## Data
* Cases are model-generated and unreviewed (see `datasets/dataset_card.md`); ground truths unverified; fictional entities unchecked for coincidental matches.
* Probable contamination/memorisation of facts; possible generator-family bias.

## Measurement
* Rule-based detectors (English phrasings) have unknown precision/recall until validated against human labels; judge models (if used) add their own biases.
* Verbalised confidence ≠ internal probability; ECE/Brier are unreliable at N≈24.
* Semantic similarity (embedding model) mixes meaning and style; the lexical fallback is not semantic.
* Cost estimates exist only if you enter prices; token counts depend on each provider's tokenizer.

## Operational
* Provider APIs change (parameters, aliases, deprecations); some models/SDKs do not accept `temperature` (observed for the Anthropic SDK 2.6.0) or `seed`. Adapters omit unsupported parameters and record `params_applied` in every raw record; Exp. 6 sampling variance is therefore not controlled identically across providers.
* Non-zero temperature or provider nondeterminism means reruns can differ even with fixed seeds.
* Local open-weight models without a system role get the system prompt merged into the user turn, which may change behaviour.

## What the study cannot claim
* Anything about model internals or "why" a failure occurs.
* That a model is "safe" or "unsafe", or a ranking of vendors, on this evidence.
* That absence of a failure in this dataset implies absence in deployment.

## Ethical notes
* All adversarial payloads are benign (a canary string). No harmful content is generated or requested. Follow each provider's usage policy; do not
  extend Exp. 4 to harmful-content elicitation without appropriate review.
* API keys live only in `.env` (git-ignored). Raw responses may contain model-generated falsehoods (Exp. 2/3); label them as such if shared.
