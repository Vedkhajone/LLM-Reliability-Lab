# Literature map

> **Verification status:** this list was assembled from memory by an AI assistant **without live lookup**. Titles, arXiv ids and venues are believed
> correct, but **you must open each source and confirm authors, year, venue and that it says what is attributed to it before citing.** Delete any item you
> cannot confirm. Do not add citations you have not read. "Relevance" lines describe how the work relates to this project, not what it concludes beyond its abstract.

## LLM evaluation (general)
* Liang et al., *Holistic Evaluation of Language Models* (HELM), arXiv:2211.09110. Multi-metric, scenario-based evaluation; motivates reporting more than accuracy.
* Miller, *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*, arXiv:2411.00640. Confidence intervals, clustering and paired comparisons for evals; directly informs §Statistics (cluster bootstrap, paired tests).
* Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, arXiv:2306.05685. Judge agreement with humans and judge biases; informs why judges are optional and validated against human labels.

## Instruction following and instruction hierarchy
* Zhou et al., *Instruction-Following Evaluation for Large Language Models* (IFEval), arXiv:2311.07911. Verifiable, rule-checkable instructions; the deterministic-constraint approach in Exp. 1 follows the same idea but targets *conflicting* constraints.
* Wallace et al., *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions*, arXiv:2404.13208. Motivates the system-vs-user priority cases in Exp. 1 and 4.

## Prompt injection and adversarial robustness
* Perez & Ribeiro, *Ignore Previous Prompt: Attack Techniques For Language Models*, arXiv:2211.09527. Goal hijacking via "ignore previous instructions".
* Greshake et al., *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*, arXiv:2302.12173. Injection through retrieved content; motivates the retrieved-document attack type.
* Wei et al., *Jailbroken: How Does LLM Safety Training Fail?*, arXiv:2307.02483. Failure modes of safety training (competing objectives, mismatched generalisation); conceptual background for role-play/format attacks. (This project does not test harmful content.)
* Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models*, arXiv:2307.15043. Automated adversarial suffixes; out of scope here (we use hand-written, benign attacks) but defines the stronger-attack frontier.

## Hallucination, truthfulness, uncertainty and calibration
* Ji et al., *Survey of Hallucination in Natural Language Generation*, ACM Computing Surveys (2023), arXiv:2202.03629. Taxonomy and definitions.
* Lin, Hilton & Evans, *TruthfulQA: Measuring How Models Mimic Human Falsehoods*, arXiv:2109.07958. Truthfulness under imitative falsehoods.
* Kadavath et al., *Language Models (Mostly) Know What They Know*, arXiv:2207.05221. Self-evaluation and calibration of model confidence.
* Xiong et al., *Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs*, arXiv:2306.13063. Verbalised-confidence elicitation; relevant to the "Confidence: N%" protocol in Exp. 3.
* Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017, arXiv:1706.04599. ECE definition.

## Context sensitivity and prompt sensitivity
* Shi et al., *Large Language Models Can Be Easily Distracted by Irrelevant Context*, ICML 2023, arXiv:2302.00093. Irrelevant context degrades accuracy; informs C2/C4.
* Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, arXiv:2307.03172. Position effects for relevant information; informs the mid-document placement in C4 (and is a reason to vary position in v0.2).
* Sclar et al., *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design*, arXiv:2310.11324. Formatting sensitivity; informs Exp. 6.

## Benchmark contamination
* Sainz et al., *NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark*, arXiv:2310.18018. Argues for per-benchmark contamination measurement; motivates the contamination caveats in the dataset card.

## Statistics (classical references)
* McNemar (1947), *Note on the sampling error of the difference between correlated proportions or percentages*, Psychometrika 12(2).
* Wilson (1927), *Probable inference, the law of succession, and statistical inference*, JASA 22(158).
* Efron (1979), *Bootstrap methods: another look at the jackknife*, Annals of Statistics 7(1).
* Holm (1979), *A simple sequentially rejective multiple test procedure*, Scandinavian Journal of Statistics 6(2).

## Official documentation to cite for model access (check current URLs)
Provider API documentation and model cards for each evaluated model (Anthropic, OpenAI, Google, and the chosen open-weight model). Record the exact
model identifier and access date for every model you report.

## What is novel here (be modest)
The individual ideas above are established. The contribution of this project is, at most: (i) a *combined*, reproducible harness for six behavioural
probes with shared statistics; (ii) the explicit separation of epistemic abstention from refusal, of authority from assertion, and of factual from stylistic
agreement; (iii) an openly documented, small, extensible dataset and failure taxonomy. Whether (iii) adds anything beyond existing benchmarks must be argued
from a gap analysis you perform after reading the sources above.
