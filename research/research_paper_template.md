# Evaluating Behavioral Reliability in Large Language Models: A Controlled Study of Instruction Conflicts, Contextual Manipulation, Uncertainty, and Adversarial Robustness

**Author:** [name] · **Affiliation:** Independent · **Date:** [date] · **Code & data:** [repository URL]

> TEMPLATE. Sections in [brackets] are for you to write. The only automated content is the results block marked `<!-- AUTO:RESULTS -->`, filled by
> `python scripts/generate_report.py`; it never writes interpretation. Delete these instructions before submission. Do not state any finding that the
> results block does not support.

## Abstract
[~150–200 words: motivation; what was built (harness, dataset of N cases, six experiments); models evaluated (exact ids, dates); main *measured*
findings with intervals; main limitation (small, model-generated, unreviewed/partly reviewed dataset; heuristic scoring); what is released.]

## 1. Introduction
[Why behavioural reliability matters for deployment and safety evaluation. State the six questions. State contributions precisely and modestly (see
`literature_review.md` §What is novel). State that evaluation is behavioural and makes no claims about internal mechanisms.]

## 2. Background and Related Work
[Organise by: LLM evaluation and statistics (HELM; error bars for evals); instruction following and hierarchy (IFEval; instruction hierarchy); prompt injection and
adversarial robustness; hallucination, truthfulness and calibration; context and prompt sensitivity; benchmark contamination. Cite only sources you have read. Identify the gap
this work addresses and how it differs from each cited work.]

## 3. Research Questions and Hypotheses
[Copy from `research_questions.md` as frozen at pre-registration (give the git tag/commit). Mark all as exploratory where applicable.]

## 4. Experimental Methodology
[Pipeline, models and access dates, generation parameters, repeats, retry/rate-limit handling, resumability, prompts (appendix), seeds. See `methodology.md` §2–3.]

## 5. Dataset Construction
[Provenance (model-generated vs human-authored), review process and share verified, ground-truth sources, difficulty distribution, controls, known biases,
contamination discussion, versioning. See `datasets/dataset_card.md`. Report how many cases were human-verified; do not call unverified synthetic cases a benchmark.]

## 6. Evaluation Metrics
[Definitions (see `methodology.md` §4), rubric design, deterministic/semantic/human/judge components, annotation protocol, agreement (κ) between automatic and human
labels, statistical procedures (Wilson, cluster bootstrap, exact McNemar, Holm, Cohen's h).]

## 7. Experimental Results
<!-- AUTO:RESULTS -->

[Interpret hypothesis by hypothesis using the tables above. Report estimates with CIs; do not describe non-significant differences as equivalence; flag ceiling effects.]

## 8. Comparative Model Analysis
[Where the data support it: paired comparisons with adjusted p-values and effect sizes. Otherwise state that comparisons are underpowered. No vendor rankings.]

## 9. Behavioral Failure Taxonomy
[From `failure_taxonomy.md` after manual coding: categories, definitions, examples (quote responses), prevalence with CIs, annotator agreement.]

## 10. Discussion
[What the results do and do not suggest; relation to prior work; practical implications for evaluation design; alternative explanations (prompt template,
generator bias, contamination, heuristic scoring errors).]

## 11. Limitations
[From `limitations.md` plus any observed in the runs.]

## 12. Ethical Considerations
[Benign payloads only; no harmful-content elicitation; provider terms; sharing of model-generated falsehoods with labels; dual-use considerations of releasing attack templates.]

## 13. Conclusion
[Brief, scoped to measured findings; future work: larger human-verified dataset, more models/templates, defences baselines, multilingual, agentic settings.]

## 14. References
[Verified references only.]

## Appendices
A. Full prompt templates (`src/reliability_lab/experiments/`). B. Rubrics. C. Model configs and exact ids. D. Dataset card. E. Reproduction commands.
