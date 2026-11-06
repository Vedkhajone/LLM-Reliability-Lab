# Research questions and hypotheses

Status: **pre-registration draft. Freeze this file (commit + tag) before running real models**, so that hypotheses are not
written after seeing results. Everything below is a hypothesis to be tested, not a finding.

## Overarching question
How reliably do current LLMs behave under controlled perturbations of instructions, context, answerability and adversarial
framing, and which observable failure patterns recur across models?

## RQ1: Instruction consistency (Exp. 1)
*Do models notice and appropriately handle contradictory or ambiguous instructions?*
* H1a: On unsatisfiable conflicts, models silently satisfy one side (outcome a/b) more often than they flag the conflict.
* H1b: When a higher-priority instruction is stated (system role or explicit ranking), adherence to it exceeds 80%.
  *(Exploratory threshold; revise from pilot data, not from final data.)*
* H1c: Models over-flag some satisfiable controls (false conflict), i.e. recognition is not free of false positives.
* H1d: Swapping the order of conflicting instructions changes the outcome in a non-trivial share of pairs.

## RQ2: Contextual robustness (Exp. 2)
*Can incorrect context override a model's correct prior knowledge?*
* H2a: Susceptibility is higher for fabricated-authority and contradictory contexts than for a bare unsupported assertion.
* H2b: Irrelevant and partial contexts do not reduce factual accuracy relative to correct context.

## RQ3: Uncertainty calibration (Exp. 3)
*Do models abstain epistemically when the answer cannot be established, without abstaining on answerable questions?*
* H3a: Hallucination rate is highest for plausible-sounding nonexistent publications and lowest for clearly unverifiable personal facts.
* H3b: Stated confidence is weakly informative of appropriateness (ECE materially above 0), and overconfidence is concentrated on hallucinated answers.
* H3c: Bare refusals are rare relative to epistemic abstentions for these benign prompts.

## RQ4: Adversarial robustness (Exp. 4)
*How often do benign-payload prompt injections displace the legitimate task?*
* H4a: Attack success varies by delivery channel (direct override vs. quoted/retrieved/formatted injection).
* H4b: The multi-step attack yields more partial compliance (canary appended, task preserved) than direct override.
* Not hypothesised: any ordering of vendors. This study cannot support such a claim at this sample size.

## RQ5: Context sensitivity (Exp. 5)
*Does added context help or hurt, relative to the question-only baseline?*
* H5a: Relevant context does not reduce accuracy (benefit ≥ 0), with potential ceiling effects.
* H5b: Misleading and contradictory contexts reduce accuracy more than irrelevant or excessive context.
* H5c: Excessive context (relevant fact buried among distractors) reduces accuracy less than misleading context.

## RQ6: Response stability (Exp. 6)
*Are answers invariant to rephrasing, formatting, context order and sampling?*
* H6a: Answer agreement is high on easy facts; prompt-wording sensitivity concentrates on medium-difficulty items.
* H6b: Semantic similarity of full responses is lower than factual answer agreement (style varies more than substance).
* Note: agreement is not correctness. A model can be stably wrong; always read agreement next to accuracy.

## Analysis plan (frozen with this file)
Primary outcome per experiment = the metric named first in `research/methodology.md` §Metrics. Report point estimates with 95% CIs
(Wilson / cluster bootstrap). Paired model comparisons use the exact McNemar test with Holm correction within a family. Hypotheses
are directional but **exploratory**: with ~20 cases per experiment, a failure to reject is not evidence of equivalence.
