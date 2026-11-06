"""Analysis + reporting: metrics, paired tests, figures, markdown report and paper draft."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..evaluation.evaluators import SemanticScorer, apply_human_labels, human_agreement, load_human_labels  # noqa: E402
from ..evaluation.metrics import compute_metrics  # noqa: E402
from ..evaluation.statistical_analysis import (  # noqa: E402
    compare_models_paired, holm_correction, majority_vote, mcnemar_exact,
)
from ..utils.config import load_evaluation_config, project_root  # noqa: E402
from ..utils.logging import get_logger  # noqa: E402
from .result_collector import resolve_dir  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------- comparisons ----

def paired_comparisons(scored: pd.DataFrame, correction: str = "holm") -> pd.DataFrame:
    """Model-vs-model McNemar tests per experiment, plus context-vs-baseline tests per model."""
    rows: list[dict[str, Any]] = []
    for exp, g in scored.groupby("experiment"):
        models = sorted(g["model_key"].unique())
        for a, b in itertools.combinations(models, 2):
            r = compare_models_paired(g, a, b)
            if r:
                rows.append({"family": "model_vs_model", "experiment": exp, "a": a, "b": b, **r})
    cs = scored[scored["experiment"] == "context_sensitivity"]
    for model, g in cs.groupby("model_key"):
        mv = majority_vote(g, ["case_id", "condition"], "correct")
        wide = mv.pivot(index="case_id", columns="condition", values="correct")
        if "C0" not in wide:
            continue
        for cond in [c for c in wide.columns if c != "C0"]:
            sub = wide[["C0", cond]].dropna()
            if len(sub):
                r = mcnemar_exact(sub[cond].astype(bool), sub["C0"].astype(bool))
                rows.append({"family": "condition_vs_baseline", "experiment": "context_sensitivity",
                             "a": f"{model}:{cond}", "b": f"{model}:C0", **r})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if correction == "holm":
        df["p_adjusted"] = np.nan
        for _, idx in df.groupby("family").groups.items():
            df.loc[idx, "p_adjusted"] = holm_correction(df.loc[idx, "p_value"].tolist())
    else:
        df["p_adjusted"] = df["p_value"]
    return df


# ------------------------------------------------------------------------ figures ----

def _fig_metrics(metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for exp, g in metrics[metrics["subset"] == "all"].groupby("experiment"):
        g = g[g["metric"].isin(g["metric"].unique())]
        g = g[~g["metric"].isin(["ece", "brier", "length_variation_cv"])]
        if g.empty:
            continue
        mets = list(g["metric"].unique())
        models = sorted(g["model_key"].unique())
        fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(mets)), 4))
        w = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            s = g[g["model_key"] == m].set_index("metric").reindex(mets)
            yerr = np.vstack([(s["value"] - s["ci_low"]).fillna(0).clip(lower=0),
                              (s["ci_high"] - s["value"]).fillna(0).clip(lower=0)])
            ax.bar(np.arange(len(mets)) + i * w, s["value"], w, yerr=yerr, capsize=2, label=m)
        ax.set_xticks(np.arange(len(mets)) + w * (len(models) - 1) / 2)
        ax.set_xticklabels(mets, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("value (95% CI)")
        ax.set_title(exp)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / f"{exp}_metrics.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths


def _fig_context(metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    g = metrics[(metrics["experiment"] == "context_sensitivity") & (metrics["metric"] == "accuracy")]
    if g.empty:
        return []
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m, s in g.groupby("model_key"):
        s = s.sort_values("subset")
        ax.errorbar(s["subset"], s["value"], yerr=[(s["value"] - s["ci_low"]).fillna(0).clip(lower=0),
                    (s["ci_high"] - s["value"]).fillna(0).clip(lower=0)], marker="o", capsize=3, label=m)
    ax.set_ylabel("accuracy"); ax.set_xlabel("condition (C0 baseline ... C5 contradictory)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.set_title("Accuracy by context condition")
    fig.tight_layout()
    p = out_dir / "context_sensitivity_by_condition.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return [p]


# ------------------------------------------------------------------------- report ----

def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_no data_\n"
    d = df[cols].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)) + "\n"


def build_report(scored: pd.DataFrame, metrics: pd.DataFrame, comps: pd.DataFrame, agreement: dict) -> str:
    mock = scored["provider"].eq("mock").any() if "provider" in scored else False
    parts = ["# Experiment report (auto-generated)\n"]
    if mock:
        parts.append("> **WARNING: this data includes the offline MOCK client. Mock outputs are synthetic; "
                     "these numbers say nothing about any real model.**\n")
    models = scored.groupby(["model_key", "provider", "model_id"]).size().reset_index(name="responses")
    parts.append("## Models and sample sizes\n" + _md_table(models, list(models.columns)))
    parts.append("\nSmall samples: treat CIs as the primary evidence; do not over-interpret point estimates.\n")
    for exp, g in metrics.groupby("experiment"):
        parts.append(f"\n## {exp}\n" + _md_table(g, ["model_key", "metric", "subset", "value", "ci_low", "ci_high", "n"]))
    if not comps.empty:
        parts.append("\n## Paired comparisons (exact McNemar; Holm-adjusted within family)\n")
        parts.append(_md_table(comps, ["family", "experiment", "a", "b", "n", "n10", "n01", "diff", "p_value", "p_adjusted", "cohens_h"]
                               if "cohens_h" in comps else ["family", "experiment", "a", "b", "n", "n10", "n01", "diff", "p_value", "p_adjusted"]))
        parts.append("\nA small p-value without a pre-specified hypothesis and adequate n is not evidence of a general effect.\n")
    if agreement.get("n_reviewed"):
        parts.append(f"\n## Human agreement\nReviewed rows: {agreement['n_reviewed']}, raw agreement {agreement['agreement']:.3f}, Cohen's kappa {agreement['kappa']:.3f}\n")
    return "\n".join(parts)


def run_analysis(results_dir: str = "results", eval_cfg: dict | None = None,
                 use_embeddings: bool | None = None, override_with_human: bool = False) -> dict[str, Any]:
    """Score is assumed to be in results/processed/scored.jsonl (see scripts/analyse_results.py)."""
    from .result_collector import load_scored

    cfg = eval_cfg or load_evaluation_config()
    scored = load_scored(results_dir)
    if scored.empty:
        raise SystemExit(f"No scored results in {results_dir}/processed. Run an experiment, then scripts/analyse_results.py.")
    labels = load_human_labels(resolve_dir(cfg["human_review"]["labels_file"]))
    scored = apply_human_labels(scored, labels, override=override_with_human)
    sem_cfg = cfg.get("semantic", {})
    enabled = sem_cfg.get("enabled", True) if use_embeddings is None else use_embeddings
    embedder = SemanticScorer(sem_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"), enabled=enabled)
    log.info("semantic backend: %s", embedder.backend)
    metrics = compute_metrics(scored, cfg["statistics"], embedder=embedder.embed)
    metrics["semantic_backend"] = embedder.backend
    comps = paired_comparisons(scored, cfg["statistics"].get("multiple_comparison_correction", "holm"))
    agreement = human_agreement(scored)

    root = resolve_dir(results_dir)
    (root / "statistics").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(root / "statistics" / "metrics.csv", index=False)
    comps.to_csv(root / "statistics" / "comparisons.csv", index=False)
    figs = _fig_metrics(metrics, root / "figures") + _fig_context(metrics, root / "figures")
    report = build_report(scored, metrics, comps, agreement)
    (root / "statistics" / "report.md").write_text(report, encoding="utf-8")
    return {"metrics": metrics, "comparisons": comps, "figures": figs, "report": report, "agreement": agreement}


def populate_paper(report_md: str, template: str | Path = "research/research_paper_template.md",
                   out: str | Path = "research/paper_draft.md") -> Path:
    """Fill the AUTO markers in the paper template with the generated report. Prose stays yours."""
    t = (project_root() / template).read_text(encoding="utf-8")
    t = t.replace("<!-- AUTO:RESULTS -->", report_md)
    p = project_root() / out
    p.write_text(t, encoding="utf-8")
    return p
