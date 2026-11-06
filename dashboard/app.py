"""LLM Reliability Lab dashboard.   Launch:  streamlit run dashboard/app.py"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from reliability_lab.evaluation.failures import add_failure_categories  # noqa: E402
from reliability_lab.evaluation.metrics import compute_metrics  # noqa: E402
from reliability_lab.pipeline.report_generator import paired_comparisons  # noqa: E402
from reliability_lab.pipeline.result_collector import load_scored, read_raw  # noqa: E402
from reliability_lab.schema import EXPERIMENTS  # noqa: E402

st.set_page_config(page_title="LLM Reliability Lab", page_icon="🧪", layout="wide")

ACCENT = "#7aa2f7"
PALETTE = ["#7aa2f7", "#bb9af7", "#9ece6a", "#e0af68", "#f7768e", "#73daca", "#ff9e64"]
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            colorway=PALETTE, font=dict(family="Inter, system-ui, sans-serif", size=12))

st.markdown(
    """
<style>
 .block-container {padding-top: 1.6rem; max-width: 1300px;}
 h1, h2, h3 {letter-spacing: -0.01em;}
 [data-testid="stMetric"] {background: #161b26; border: 1px solid #2a3142; border-radius: 10px; padding: 12px 16px;}
 [data-testid="stMetricLabel"] {color: #8b93a7;}
 .lab-sub {color:#8b93a7; margin-top:-0.6rem; margin-bottom:1rem;}
 .warn {background:#3a2a12; border:1px solid #e0af68; color:#f2d9a6; padding:10px 14px; border-radius:8px;}
 div[data-baseweb="tab-list"] {gap: 6px;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- data ----
def result_dirs() -> list[str]:
    return [d.name for d in sorted(ROOT.glob("results*")) if (d / "raw").exists() or (d / "processed").exists()] or ["results"]


@st.cache_data(show_spinner=False)
def load(results_dir: str, mtime: float):
    raw = read_raw(results_dir)
    scored = load_scored(results_dir)
    return raw, scored


@st.cache_data(show_spinner=False)
def metrics_for(results_dir: str, mtime: float, n_boot: int):
    scored = load_scored(results_dir)
    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()
    m = compute_metrics(scored, {"bootstrap_resamples": n_boot})
    return m, paired_comparisons(scored)


def mtime_of(results_dir: str) -> float:
    p = ROOT / results_dir
    files = list(p.rglob("*.jsonl")) if p.exists() else []
    return max((f.stat().st_mtime for f in files), default=0.0)


st.title("🧪 LLM Reliability Lab")
st.markdown('<div class="lab-sub">Behavioral reliability and adversarial evaluation of LLMs · controlled, reproducible, '
            "uncertainty-aware</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Data")
    rdir = st.selectbox("Results directory", result_dirs())
    n_boot = st.select_slider("Bootstrap resamples (CI)", [200, 500, 1000, 2000], value=500)
    st.caption("Raw responses are append-only; scores are recomputed from them by `scripts/analyse_results.py`.")

mt = mtime_of(rdir)
raw, scored = load(rdir, mt)

if raw.empty:
    st.info(f"No results in `{rdir}/raw` yet. Try a mocked smoke test:\n\n"
            "```\npython scripts/run_all.py --smoke --models mock --yes\n"
            "python scripts/analyse_results.py --results-dir results_smoke --no-embeddings\n```")
    st.stop()

if raw["provider"].eq("mock").any():
    st.markdown('<div class="warn"><b>Mock data present.</b> The offline mock client produces synthetic text. '
                "Numbers derived from it describe the pipeline, not any real model.</div>", unsafe_allow_html=True)
    st.write("")

if scored.empty:
    st.warning("Raw results exist but are not scored yet. Run `python scripts/analyse_results.py`.")

metrics, comps = metrics_for(rdir, mt, n_boot) if not scored.empty else (pd.DataFrame(), pd.DataFrame())
tabs = st.tabs(["Overview", "Model comparison", "Experiment analysis", "Failure analysis", "Cost & runtime", "Export"])

# ------------------------------------------------------------------------ overview ----
with tabs[0]:
    ok = raw[raw["error"].isna()]
    c = st.columns(5)
    c[0].metric("Raw requests", f"{len(raw):,}")
    c[1].metric("Completed runs", f"{len(ok):,}", delta=f"{len(raw) - len(ok)} errors" if len(raw) > len(ok) else None,
                delta_color="inverse")
    c[2].metric("Models evaluated", raw["model_key"].nunique())
    c[3].metric("Experiments", raw["experiment"].nunique())
    c[4].metric("Distinct test cases", raw["case_id"].nunique())
    st.subheader("Evaluation progress")
    rows = []
    for exp in EXPERIMENTS:
        n_ok = len(ok[ok["experiment"] == exp])
        n_sc = len(scored[scored["experiment"] == exp]) if not scored.empty else 0
        rows.append({"experiment": exp, "responses": n_ok, "scored": n_sc, "scored %": (n_sc / n_ok) if n_ok else np.nan})
    prog = pd.DataFrame(rows)
    st.dataframe(prog, hide_index=True, use_container_width=True,
                 column_config={"scored %": st.column_config.ProgressColumn("scored %", min_value=0, max_value=1, format="%.0f%%")})
    st.subheader("Models")
    st.dataframe(raw.groupby(["model_key", "provider", "model_id"]).agg(
        responses=("item_id", "count"), errors=("error", lambda s: int(s.notna().sum())),
        dataset_versions=("dataset_version", lambda s: ", ".join(sorted(set(s)))),
        prompt_versions=("prompt_version", lambda s: ", ".join(sorted(set(s))))).reset_index(),
        hide_index=True, use_container_width=True)
    st.caption("Dataset cases are model-generated and unreviewed until you verify them "
               "(see `datasets/dataset_card.md`). Treat results as exploratory.")

# --------------------------------------------------------------- model comparison ----
with tabs[1]:
    if metrics.empty:
        st.info("No scored metrics yet.")
    else:
        exp = st.selectbox("Experiment", sorted(metrics["experiment"].unique()), key="cmp_exp")
        g = metrics[(metrics.experiment == exp)]
        subsets = ["all"] + sorted(s for s in g["subset"].unique() if s != "all")
        sub = st.selectbox("Subset (category-wise view)", subsets)
        gg = g[g.subset == sub]
        fig = go.Figure()
        for i, (mk, d) in enumerate(gg.groupby("model_key")):
            fig.add_bar(x=d["metric"], y=d["value"], name=mk, marker_color=PALETTE[i % len(PALETTE)],
                        error_y=dict(type="data", symmetric=False, array=(d["ci_high"] - d["value"]).fillna(0).clip(lower=0),
                                     arrayminus=(d["value"] - d["ci_low"]).fillna(0).clip(lower=0)))
        fig.update_layout(**PLOT, barmode="group", yaxis_title="value (95% CI)", height=420, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Error bars: Wilson interval (independent cases) or cluster bootstrap over cases (repeated measures). "
                   "With ~18 cases per experiment intervals are wide; do not over-read differences.")
        st.dataframe(gg[["model_key", "metric", "subset", "value", "ci_low", "ci_high", "n"]].round(3),
                     hide_index=True, use_container_width=True)
        with st.expander("Metric definitions"):
            for _, r in gg.drop_duplicates("metric").iterrows():
                st.markdown(f"**{r['metric']}** — {r['definition']}")
        st.subheader("Pass rate: experiment × model")
        pr = scored.groupby(["experiment", "model_key"])["passed"].apply(lambda s: float(np.mean(s.astype(float)))).unstack()
        st.plotly_chart(px.imshow(pr, text_auto=".2f", color_continuous_scale="Blues", aspect="auto", zmin=0, zmax=1)
                        .update_layout(**PLOT, height=320), use_container_width=True)
        if not comps.empty:
            st.subheader("Paired comparisons (exact McNemar, Holm-adjusted)")
            cols = [c for c in ["family", "experiment", "a", "b", "n", "n10", "n01", "diff", "p_value", "p_adjusted"] if c in comps]
            st.dataframe(comps[cols].round(4), hide_index=True, use_container_width=True)
            st.caption("`diff` = accuracy(a) − accuracy(b). A small p-value is not evidence of a general effect without a "
                       "pre-specified hypothesis and adequate sample size.")

# ----------------------------------------------------------- experiment analysis ----
with tabs[2]:
    if scored.empty:
        st.info("No scored results.")
    else:
        c1, c2, c3 = st.columns(3)
        exp = c1.selectbox("Experiment", sorted(scored.experiment.unique()), key="ea_exp")
        d = scored[scored.experiment == exp]
        mdl = c2.selectbox("Model", sorted(d.model_key.unique()), key="ea_model")
        d = d[d.model_key == mdl]
        only_fail = c3.checkbox("Only failures", value=False)
        if only_fail:
            d = d[~d["passed"].astype(bool)]
        st.write(f"{len(d)} rows")
        if len(d):
            sel = st.selectbox("Item", d["item_id"] + "  (repeat " + d["repeat"].astype(str) + ")", key="ea_item")
            iid, rep = sel.split("  (repeat ")[0], int(sel.split("  (repeat ")[1].rstrip(")"))
            row = d[(d.item_id == iid) & (d.repeat == rep)].iloc[0]
            rr = raw[(raw.request_key == row["request_key"])].iloc[-1]
            a, b = st.columns(2)
            with a:
                st.markdown("**Prompt**")
                if rr.get("system"):
                    st.caption("system"); st.code(rr["system"], language=None)
                st.code(rr["prompt"], language=None)
            with b:
                st.markdown("**Model response**")
                st.code(rr["response"], language=None)
                st.markdown("**Scores**")
                skip = {"response", "prompt", "system", "request_key", "run_id"}
                st.json({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.to_dict().items() if k not in skip},
                        expanded=False)
            if bool(row.get("needs_review")):
                st.caption("⚠ flagged `needs_review`: heuristic label — confirm manually "
                           "(see `results/processed/review_queue.csv`).")

# ------------------------------------------------------------------- failures ----
with tabs[3]:
    if scored.empty:
        st.info("No scored results.")
    else:
        fdf = add_failure_categories(scored)
        fails = fdf[fdf["failure_category"].notna()]
        st.write(f"{len(fails)} failures out of {len(fdf)} scored rows "
                 f"({len(fails) / max(1, len(fdf)):.1%}). Categories are operational labels of observable behaviour, "
                 "not claims about internal mechanisms.")
        exps = st.multiselect("Experiments", sorted(fails.experiment.unique()), default=sorted(fails.experiment.unique()))
        f2 = fails[fails.experiment.isin(exps)]
        if f2.empty:
            st.success("No failures in the selection.")
        else:
            dist = f2.groupby(["failure_category", "model_key"]).size().reset_index(name="count")
            st.plotly_chart(px.bar(dist, x="count", y="failure_category", color="model_key", orientation="h",
                                   color_discrete_sequence=PALETTE).update_layout(**PLOT, height=420, barmode="stack",
                                                                                   yaxis=dict(categoryorder="total ascending")),
                            use_container_width=True)
            cat = st.selectbox("Representative failures for category", sorted(f2.failure_category.unique()))
            rep = f2[f2.failure_category == cat].head(8)
            st.dataframe(rep[["experiment", "model_key", "item_id", "condition", "response"]], hide_index=True,
                         use_container_width=True)

# ------------------------------------------------------------------------ cost ----
with tabs[4]:
    r = raw.copy()
    for col in ("input_tokens", "output_tokens", "latency_s", "cost_usd"):
        r[col] = pd.to_numeric(r[col], errors="coerce")
    agg = r.groupby("model_key").agg(requests=("item_id", "count"), input_tokens=("input_tokens", "sum"),
                                     output_tokens=("output_tokens", "sum"), cost_usd=("cost_usd", "sum"),
                                     priced=("cost_usd", "count"), latency_mean_s=("latency_s", "mean"),
                                     latency_p95_s=("latency_s", lambda s: s.quantile(0.95)),
                                     total_runtime_s=("latency_s", "sum")).reset_index()
    st.dataframe(agg.round(4), hide_index=True, use_container_width=True)
    if (agg["priced"] == 0).all():
        st.caption("Costs are blank because no prices are set in `configs/models.yaml` "
                   "(copy them from each provider's pricing page).")
    by = r.groupby(["experiment", "model_key"]).agg(tokens=("output_tokens", "sum"), runtime_s=("latency_s", "sum")).reset_index()
    st.plotly_chart(px.bar(by, x="experiment", y="runtime_s", color="model_key", barmode="group",
                           color_discrete_sequence=PALETTE).update_layout(**PLOT, height=340,
                                                                          yaxis_title="summed API latency (s)"),
                    use_container_width=True)
    st.plotly_chart(px.histogram(r.dropna(subset=["latency_s"]), x="latency_s", color="model_key", nbins=40,
                                 color_discrete_sequence=PALETTE).update_layout(**PLOT, height=300), use_container_width=True)

# ---------------------------------------------------------------------- export ----
with tabs[5]:
    st.subheader("Downloads")
    cA, cB, cC = st.columns(3)
    if not scored.empty:
        cA.download_button("Scored results (CSV)", scored.to_csv(index=False).encode(), "scored.csv", "text/csv")
        cA.download_button("Scored results (JSONL)",
                           "\n".join(json.dumps(x, default=str) for x in scored.to_dict("records")).encode(),
                           "scored.jsonl", "application/json")
    cB.download_button("Raw responses (JSONL)", "\n".join(json.dumps(x, default=str) for x in raw.to_dict("records")).encode(),
                       "raw.jsonl", "application/json")
    if not metrics.empty:
        cB.download_button("Metrics with CIs (CSV)", metrics.to_csv(index=False).encode(), "metrics.csv", "text/csv")
        summary = metrics[["experiment", "model_key", "metric", "subset", "value", "ci_low", "ci_high", "n"]]
        cC.download_button("Statistical summary (CSV)", summary.to_csv(index=False).encode(), "statistical_summary.csv", "text/csv")
        if not comps.empty:
            cC.download_button("Paired comparisons (CSV)", comps.to_csv(index=False).encode(), "comparisons.csv", "text/csv")
    figs = sorted((ROOT / rdir / "figures").glob("*.png"))
    if figs:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f in figs:
                z.write(f, f.name)
        st.download_button(f"Figures ({len(figs)} PNG, zip)", buf.getvalue(), "figures.zip", "application/zip")
    else:
        st.caption("No figures yet; `scripts/analyse_results.py` writes them to `results*/figures/`.")
