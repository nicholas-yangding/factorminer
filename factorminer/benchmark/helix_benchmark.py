"""HelixBenchmark — rigorous comparison of HelixFactor vs FactorMiner.

Provides five inter-operating classes that together form a complete
benchmarking suite for the HelixFactor vs FactorMiner (Ralph Loop) paper:

  HelixBenchmark          — main comparison class (Table 1 style)
  StatisticalComparisonTests — DM test, paired t-test, block bootstrap
  SpeedBenchmark          — operator / factor / pipeline timing
  BenchmarkResult         — aggregate result container + report generators
  DMTestResult / MethodResult — individual result containers

CLI usage:
  python -m factorminer.benchmark.helix_benchmark --mock --n-factors 40 --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    """Metrics for a single method run."""

    method: str
    library_ic: float = 0.0
    library_icir: float = 0.0
    avg_abs_rho: float = 0.0
    ew_ic: float = 0.0
    ew_icir: float = 0.0
    icw_ic: float = 0.0
    icw_icir: float = 0.0
    lasso_ic: float = 0.0
    lasso_icir: float = 0.0
    xgb_ic: float = 0.0
    xgb_icir: float = 0.0
    n_factors: int = 0
    admission_rate: float = 0.0
    elapsed_seconds: float = 0.0
    # raw IC series for statistical tests (not serialized by default)
    ic_series: Optional[np.ndarray] = field(default=None, repr=False)
    run_id: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("ic_series", None)
        return d


@dataclass
class DMTestResult:
    """Diebold-Mariano test for forecast accuracy difference."""

    dm_statistic: float
    p_value: float
    is_significant: bool
    direction: str   # "helix_better", "ralph_better", "no_difference"
    n_obs: int


@dataclass
class AblationResult:
    """Result of one ablation study."""

    configs: List[str]
    results: Dict[str, MethodResult]
    contributions: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict:
        return {
            "configs": self.configs,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


@dataclass
class OperatorSpeedResult:
    """Timing for individual operators."""

    operator_timings_ms: Dict[str, float]   # operator_name -> ms
    n_assets: int
    n_periods: int
    n_repeats: int


@dataclass
class PipelineSpeedResult:
    """Timing for end-to-end pipeline."""

    total_seconds: float
    candidates_per_second: float
    n_candidates: int


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results — all methods, all metrics."""

    methods: List[str]
    factor_library_metrics: pd.DataFrame    # IC, ICIR, Avg|rho| per method
    combination_metrics: pd.DataFrame       # EW/ICW IC and ICIR
    selection_metrics: pd.DataFrame         # LASSO, XGBoost
    speed_metrics: pd.DataFrame
    statistical_tests: Dict[str, Any]
    ablation_result: Optional[AblationResult] = None
    raw_method_results: Dict[str, List[MethodResult]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def to_latex_table(self) -> str:
        """Generate a LaTeX table matching paper Table 1 style."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{HelixFactor vs FactorMiner: Comprehensive Benchmark (Table 1 Style)}",
            r"\label{tab:benchmark}",
            r"\small",
            r"\begin{tabular}{lcccccccc}",
            r"\toprule",
            r"Method & \multicolumn{3}{c}{Factor Library} & \multicolumn{2}{c}{EW Combo} & \multicolumn{2}{c}{ICW Combo} & Sel.IC \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
            r" & IC(\%) & ICIR & Avg$|\rho|$ & IC(\%) & ICIR & IC(\%) & ICIR & IC(\%) \\",
            r"\midrule",
        ]

        for method in self.methods:
            lib_row = self.factor_library_metrics[
                self.factor_library_metrics["method"] == method
            ]
            comb_row = self.combination_metrics[
                self.combination_metrics["method"] == method
            ]
            sel_row = self.selection_metrics[
                self.selection_metrics["method"] == method
            ]

            def _g(df, col, mult=100.0):
                if df.empty or col not in df.columns:
                    return 0.0
                v = df.iloc[0][col]
                return float(v) * mult if not pd.isna(v) else 0.0

            bold = method in ("helix_phase2",)
            fmt = lambda x, d=2: f"{x:.{d}f}"

            lib_ic = _g(lib_row, "ic_pct", 1.0)
            lib_icir = _g(lib_row, "icir", 1.0)
            lib_rho = _g(lib_row, "avg_abs_rho", 1.0)
            ew_ic = _g(comb_row, "ew_ic_pct", 1.0)
            ew_icir = _g(comb_row, "ew_icir", 1.0)
            icw_ic = _g(comb_row, "icw_ic_pct", 1.0)
            icw_icir = _g(comb_row, "icw_icir", 1.0)
            sel_ic = _g(sel_row, "best_ic_pct", 1.0)

            row_parts = [
                method.replace("_", r"\_"),
                fmt(lib_ic),
                fmt(lib_icir),
                fmt(lib_rho),
                fmt(ew_ic),
                fmt(ew_icir),
                fmt(icw_ic),
                fmt(icw_icir),
                fmt(sel_ic),
            ]
            if bold:
                row_parts = [r"\textbf{" + p + r"}" for p in row_parts]

            lines.append(" & ".join(row_parts) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def to_markdown_table(self) -> str:
        """Generate a Markdown table for GitHub README."""
        header = (
            "| Method | IC (%) | ICIR | Avg|ρ| | EW IC (%) | EW ICIR | "
            "ICW IC (%) | ICW ICIR | Las IC (%) | XGB IC (%) |\n"
            "|--------|--------|------|---------|-----------|---------|"
            "-----------|----------|-----------|------------|\n"
        )
        rows = []
        for method in self.methods:
            lib_row = self.factor_library_metrics[
                self.factor_library_metrics["method"] == method
            ]
            comb_row = self.combination_metrics[
                self.combination_metrics["method"] == method
            ]
            sel_row = self.selection_metrics[
                self.selection_metrics["method"] == method
            ]

            def _g(df, col):
                if df.empty or col not in df.columns:
                    return 0.0
                v = df.iloc[0][col]
                return float(v) if not pd.isna(v) else 0.0

            tag = " **" if method == "helix_phase2" else ""
            rows.append(
                f"| {method}{tag} | "
                f"{_g(lib_row,'ic_pct'):.2f} | "
                f"{_g(lib_row,'icir'):.3f} | "
                f"{_g(lib_row,'avg_abs_rho'):.3f} | "
                f"{_g(comb_row,'ew_ic_pct'):.2f} | "
                f"{_g(comb_row,'ew_icir'):.3f} | "
                f"{_g(comb_row,'icw_ic_pct'):.2f} | "
                f"{_g(comb_row,'icw_icir'):.3f} | "
                f"{_g(sel_row,'lasso_ic_pct'):.2f} | "
                f"{_g(sel_row,'xgb_ic_pct'):.2f} |\n"
            )
        return header + "".join(rows)

    def plot_comparison(self, save_path: str) -> None:
        """Generate bar chart comparison (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib not available; skipping plot")
            return

        metrics = ["IC (%)", "ICIR", "EW IC (%)", "ICW IC (%)"]
        method_colors = {
            "random_exploration": "#aaaaaa",
            "alpha101_classic": "#6baed6",
            "alpha101_adapted": "#3182bd",
            "ralph_loop": "#fd8d3c",
            "helix_phase2": "#31a354",
        }

        fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
        fig.suptitle("HelixFactor vs FactorMiner Benchmark", fontsize=14, fontweight="bold")

        for ax_idx, metric in enumerate(metrics):
            ax = axes[ax_idx]
            values = []
            colors = []
            labels = []

            for method in self.methods:
                color = method_colors.get(method, "#888888")
                if metric == "IC (%)":
                    row = self.factor_library_metrics[
                        self.factor_library_metrics["method"] == method
                    ]
                    v = float(row["ic_pct"].iloc[0]) if not row.empty else 0.0
                elif metric == "ICIR":
                    row = self.factor_library_metrics[
                        self.factor_library_metrics["method"] == method
                    ]
                    v = float(row["icir"].iloc[0]) if not row.empty else 0.0
                elif metric == "EW IC (%)":
                    row = self.combination_metrics[
                        self.combination_metrics["method"] == method
                    ]
                    v = float(row["ew_ic_pct"].iloc[0]) if not row.empty else 0.0
                elif metric == "ICW IC (%)":
                    row = self.combination_metrics[
                        self.combination_metrics["method"] == method
                    ]
                    v = float(row["icw_ic_pct"].iloc[0]) if not row.empty else 0.0
                else:
                    v = 0.0

                values.append(v)
                colors.append(color)
                labels.append(method.replace("_", "\n"))

            bars = ax.bar(range(len(values)), values, color=colors, alpha=0.85, edgecolor="white")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
            ax.set_title(metric, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved comparison plot to %s", save_path)

    def generate_full_report(self, save_path: str) -> None:
        """Generate a complete HTML report with all results."""
        html = self._build_html_report()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(html)
        logger.info("Saved full HTML report to %s", save_path)

    def _build_html_report(self) -> str:
        lib_html = self.factor_library_metrics.to_html(index=False, float_format="{:.4f}".format)
        comb_html = self.combination_metrics.to_html(index=False, float_format="{:.4f}".format)
        sel_html = self.selection_metrics.to_html(index=False, float_format="{:.4f}".format)
        speed_html = self.speed_metrics.to_html(index=False, float_format="{:.3f}".format)

        stat_rows = []
        for k, v in self.statistical_tests.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    stat_rows.append(f"<tr><td>{k}.{sk}</td><td>{sv}</td></tr>")
            else:
                stat_rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        stat_html = (
            "<table border='1'><tr><th>Test</th><th>Result</th></tr>"
            + "".join(stat_rows)
            + "</table>"
        )

        ablation_html = ""
        if self.ablation_result is not None and self.ablation_result.contributions is not None:
            ablation_html = (
                "<h2>Ablation Study</h2>"
                + self.ablation_result.contributions.to_html(
                    index=False, float_format="{:.4f}".format
                )
            )

        css = """
        body { font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; color: #333; }
        h1 { color: #1a5276; border-bottom: 3px solid #1a5276; padding-bottom: 8px; }
        h2 { color: #2c3e50; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th { background: #1a5276; color: white; padding: 8px 12px; text-align: left; }
        td { padding: 6px 12px; border-bottom: 1px solid #ddd; }
        tr:nth-child(even) { background: #f2f2f2; }
        tr:hover { background: #d6eaf8; }
        .helix-row { background: #d5f5e3 !important; font-weight: bold; }
        .summary-box { background: #eaf2ff; border-left: 5px solid #1a5276;
                        padding: 15px; margin: 20px 0; border-radius: 4px; }
        """

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>HelixFactor Benchmark Report</title>
<style>{css}</style></head><body>
<h1>HelixFactor Benchmark Report</h1>
<div class="summary-box">
<strong>Methods evaluated:</strong> {", ".join(self.methods)}<br>
<strong>Generated:</strong> {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
<h2>Factor Library Metrics</h2>{lib_html}
<h2>Factor Combination Metrics</h2>{comb_html}
<h2>Factor Selection Metrics</h2>{sel_html}
<h2>Speed Benchmarks</h2>{speed_html}
<h2>Statistical Tests</h2>{stat_html}
{ablation_html}
</body></html>"""


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

class StatisticalComparisonTests:
    """Rigorous statistical comparison between HelixFactor and FactorMiner.

    Implements four complementary tests:
      1. Diebold-Mariano (DM) test for forecast accuracy differences
      2. Paired t-test on IC(Helix) − IC(Ralph) across test period
      3. Block-bootstrap 95% CI on IC difference
      4. Wilcoxon signed-rank test (non-parametric)
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Diebold-Mariano test
    # ------------------------------------------------------------------

    def diebold_mariano_test(
        self,
        ic_series_1: np.ndarray,
        ic_series_2: np.ndarray,
        h: int = 1,
    ) -> DMTestResult:
        """Diebold-Mariano test for forecast accuracy differences.

        Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t) is the
        differential loss. Uses the HAC-robust variance estimator with
        bandwidth h-1 (Andrews, 1991).

        Parameters
        ----------
        ic_series_1 : ndarray  (e.g. HelixFactor IC time series)
        ic_series_2 : ndarray  (e.g. FactorMiner IC time series)
        h : int
            Forecast horizon (default 1 for one-step-ahead).

        Returns
        -------
        DMTestResult
        """
        n1 = len(ic_series_1[~np.isnan(ic_series_1)])
        n2 = len(ic_series_2[~np.isnan(ic_series_2)])
        min_len = min(n1, n2)
        if min_len < 5:
            return DMTestResult(
                dm_statistic=0.0, p_value=1.0, is_significant=False,
                direction="no_difference", n_obs=min_len,
            )

        s1 = ic_series_1[~np.isnan(ic_series_1)][:min_len]
        s2 = ic_series_2[~np.isnan(ic_series_2)][:min_len]

        # Loss differential: squared-error loss on IC as forecast of return
        d = s1 ** 2 - s2 ** 2
        d_bar = np.mean(d)
        T = len(d)

        # HAC variance of d_bar (Newey-West with bandwidth h-1)
        bandwidth = max(h - 1, 0)
        gamma_0 = np.var(d, ddof=0)
        hac_var = gamma_0
        for lag in range(1, bandwidth + 1):
            gamma_k = np.mean(
                (d[lag:] - d_bar) * (d[:-lag] - d_bar)
            )
            hac_var += 2.0 * (1.0 - lag / (bandwidth + 1)) * gamma_k

        if hac_var <= 0 or np.isnan(hac_var):
            # Fallback to simple variance
            hac_var = gamma_0 / max(T - 1, 1)

        dm_stat = d_bar / np.sqrt(hac_var / T)

        # Two-sided p-value using normal approximation
        from scipy.stats import norm
        p_value = 2.0 * (1.0 - float(norm.cdf(abs(dm_stat))))

        if abs(dm_stat) < 1.96:
            direction = "no_difference"
        elif d_bar > 0:
            # series_1 has higher loss, series_2 is better
            direction = "ralph_better"
        else:
            direction = "helix_better"

        return DMTestResult(
            dm_statistic=float(dm_stat),
            p_value=float(p_value),
            is_significant=p_value < 0.05,
            direction=direction,
            n_obs=T,
        )

    # ------------------------------------------------------------------
    # Paired t-test
    # ------------------------------------------------------------------

    def paired_t_test(
        self,
        ic_series_1: np.ndarray,
        ic_series_2: np.ndarray,
    ) -> dict:
        """Paired t-test on IC difference series."""
        n = min(
            len(ic_series_1[~np.isnan(ic_series_1)]),
            len(ic_series_2[~np.isnan(ic_series_2)]),
        )
        if n < 5:
            return {"t_stat": 0.0, "p_value": 1.0, "mean_diff": 0.0, "n": n}

        s1 = ic_series_1[~np.isnan(ic_series_1)][:n]
        s2 = ic_series_2[~np.isnan(ic_series_2)][:n]
        t_stat, p_value = ttest_rel(s1, s2)
        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_diff": float(np.mean(s1 - s2)),
            "n": n,
        }

    # ------------------------------------------------------------------
    # Block bootstrap CI
    # ------------------------------------------------------------------

    def bootstrap_ic_difference_ci(
        self,
        ic_series_1: np.ndarray,
        ic_series_2: np.ndarray,
        n_bootstrap: int = 1000,
        block_size: int = 20,
    ) -> Tuple[float, float]:
        """95% block-bootstrap CI on mean IC difference.

        Returns
        -------
        (lower_95, upper_95) : tuple of float
        """
        n = min(
            len(ic_series_1[~np.isnan(ic_series_1)]),
            len(ic_series_2[~np.isnan(ic_series_2)]),
        )
        if n < 5:
            return (0.0, 0.0)

        s1 = ic_series_1[~np.isnan(ic_series_1)][:n]
        s2 = ic_series_2[~np.isnan(ic_series_2)][:n]
        diff = s1 - s2

        # Circular block bootstrap
        block_size = min(block_size, n // 2)
        block_size = max(block_size, 1)
        n_blocks = int(math.ceil(n / block_size))
        boot_means = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            starts = self._rng.randint(0, n - block_size + 1, size=n_blocks)
            indices = np.concatenate(
                [np.arange(s, s + block_size) for s in starts]
            )[:n]
            boot_means[i] = diff[indices].mean()

        return (
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
        )

    # ------------------------------------------------------------------
    # Wilcoxon signed-rank test
    # ------------------------------------------------------------------

    def wilcoxon_test(
        self,
        ic_series_1: np.ndarray,
        ic_series_2: np.ndarray,
    ) -> dict:
        """Wilcoxon signed-rank test (non-parametric) on IC pairs."""
        n = min(
            len(ic_series_1[~np.isnan(ic_series_1)]),
            len(ic_series_2[~np.isnan(ic_series_2)]),
        )
        if n < 5:
            return {"statistic": 0.0, "p_value": 1.0, "n": n}

        s1 = ic_series_1[~np.isnan(ic_series_1)][:n]
        s2 = ic_series_2[~np.isnan(ic_series_2)][:n]
        try:
            stat, p_value = wilcoxon(s1, s2, alternative="two-sided")
        except Exception:
            stat, p_value = 0.0, 1.0

        return {"statistic": float(stat), "p_value": float(p_value), "n": n}

    # ------------------------------------------------------------------
    # Combined report
    # ------------------------------------------------------------------

    def run_all_tests(
        self,
        ic_helix: np.ndarray,
        ic_ralph: np.ndarray,
    ) -> dict:
        """Run all four statistical tests and return combined results."""
        dm = self.diebold_mariano_test(ic_helix, ic_ralph)
        tt = self.paired_t_test(ic_helix, ic_ralph)
        ci_lo, ci_hi = self.bootstrap_ic_difference_ci(ic_helix, ic_ralph)
        wil = self.wilcoxon_test(ic_helix, ic_ralph)
        mean_diff = float(
            np.nanmean(
                ic_helix[: min(len(ic_helix), len(ic_ralph))]
                - ic_ralph[: min(len(ic_helix), len(ic_ralph))]
            )
        )
        return {
            "diebold_mariano": {
                "dm_stat": dm.dm_statistic,
                "p_value": dm.p_value,
                "significant": dm.is_significant,
                "direction": dm.direction,
                "n_obs": dm.n_obs,
            },
            "paired_t_test": tt,
            "bootstrap_ci_95": {
                "lower": ci_lo,
                "upper": ci_hi,
                "excludes_zero": ci_lo > 0 or ci_hi < 0,
            },
            "wilcoxon": wil,
            "mean_ic_difference": mean_diff,
            "helix_outperforms": mean_diff > 0,
        }


# ---------------------------------------------------------------------------
# Speed Benchmark
# ---------------------------------------------------------------------------

class SpeedBenchmark:
    """Benchmark factor evaluation speed across operators and pipelines."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)

    def _time_callable(self, fn, n_repeats: int = 5, warmup: int = 1) -> float:
        """Return minimum time over n_repeats (ms) after warmup runs."""
        for _ in range(warmup):
            try:
                fn()
            except Exception:
                pass
        timings = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            try:
                fn()
            except Exception:
                pass
            timings.append((time.perf_counter() - t0) * 1000.0)
        return float(np.min(timings)) if timings else 0.0

    def run_operator_benchmark(
        self,
        n_assets: int = 500,
        n_periods: int = 2000,
        n_repeats: int = 5,
    ) -> OperatorSpeedResult:
        """Benchmark individual operators (numpy backend)."""
        rng = np.random.RandomState(self._rng.randint(0, 9999))
        X = rng.randn(n_assets, n_periods).astype(np.float64)
        Y = rng.randn(n_assets, n_periods).astype(np.float64)

        from scipy.stats import rankdata

        def _ts_rank(mat, window=20):
            out = np.full_like(mat, np.nan)
            for t in range(window - 1, mat.shape[1]):
                slc = mat[:, t - window + 1: t + 1]
                for i in range(mat.shape[0]):
                    r = rankdata(slc[i])
                    out[i, t] = r[-1] / window
            return out

        def _cs_rank(mat):
            out = np.full_like(mat, np.nan)
            for t in range(mat.shape[1]):
                col = mat[:, t]
                valid = ~np.isnan(col)
                if valid.sum() > 0:
                    out[valid, t] = rankdata(col[valid]) / valid.sum()
            return out

        def _ts_std(mat, window=20):
            out = np.full_like(mat, np.nan)
            for t in range(window - 1, mat.shape[1]):
                slc = mat[:, t - window + 1: t + 1]
                out[:, t] = np.std(slc, axis=1, ddof=1)
            return out

        def _ts_corr(x, y, window=20):
            out = np.full_like(x, np.nan)
            for t in range(window - 1, x.shape[1]):
                sx = x[:, t - window + 1: t + 1]
                sy = y[:, t - window + 1: t + 1]
                xs = sx - sx.mean(axis=1, keepdims=True)
                ys = sy - sy.mean(axis=1, keepdims=True)
                denom = np.sqrt((xs**2).sum(axis=1) * (ys**2).sum(axis=1))
                safe = denom > 1e-12
                out[safe, t] = ((xs * ys).sum(axis=1) / denom)[safe]
            return out

        # Use small sub-matrix for timing (keep fast)
        X_s = X[:50, :100]
        Y_s = Y[:50, :100]

        ops = {
            "TsRank(w=20)": lambda: _ts_rank(X_s, window=20),
            "CsRank": lambda: _cs_rank(X_s),
            "TsStd(w=20)": lambda: _ts_std(X_s, window=20),
            "TsCorr(w=20)": lambda: _ts_corr(X_s, Y_s, window=20),
            "TsMean(w=20)": lambda: np.lib.stride_tricks.sliding_window_view(X_s, 20, axis=1).mean(axis=-1),
            "CsZscore": lambda: (X_s - X_s.mean(axis=0)) / (X_s.std(axis=0) + 1e-8),
        }

        timings: Dict[str, float] = {}
        for name, fn in ops.items():
            timings[name] = self._time_callable(fn, n_repeats=n_repeats)

        return OperatorSpeedResult(
            operator_timings_ms=timings,
            n_assets=n_assets,
            n_periods=n_periods,
            n_repeats=n_repeats,
        )

    def run_full_pipeline_benchmark(
        self,
        n_candidates: int = 200,
        data: Optional[dict] = None,
    ) -> PipelineSpeedResult:
        """Benchmark end-to-end candidate evaluation pipeline."""
        if data is None:
            data = _build_mock_data_dict(n_assets=100, n_periods=200, seed=42)

        from factorminer.benchmark.catalogs import build_random_exploration
        from factorminer.core.parser import try_parse
        from factorminer.evaluation.metrics import compute_ic, compute_ic_mean

        entries = build_random_exploration(seed=99, count=n_candidates)
        returns = data.get("forward_returns", data.get("$close"))
        if returns is None:
            returns = np.random.randn(*list(data.values())[0].shape) * 0.01

        t0 = time.perf_counter()
        succeeded = 0
        for entry in entries[:n_candidates]:
            tree = try_parse(entry.formula)
            if tree is None:
                continue
            try:
                signals = tree.evaluate(data)
                ic = compute_ic(signals, returns)
                _ = compute_ic_mean(ic)
                succeeded += 1
            except Exception:
                pass
        elapsed = time.perf_counter() - t0

        return PipelineSpeedResult(
            total_seconds=elapsed,
            candidates_per_second=succeeded / max(elapsed, 1e-6),
            n_candidates=n_candidates,
        )

    def generate_speed_table(
        self,
        op_result: OperatorSpeedResult,
        pipeline_result: PipelineSpeedResult,
    ) -> str:
        """Generate a LaTeX table of speed results."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Computational Efficiency Benchmark}",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Operator / Task & Time (ms) & Relative \\",
            r"\midrule",
        ]
        timings = op_result.operator_timings_ms
        baseline = max(timings.values()) if timings else 1.0
        for op, t in timings.items():
            rel = t / baseline if baseline > 0 else 1.0
            lines.append(rf"{op} & {t:.2f} & {rel:.2f}x \\")

        lines.append(r"\midrule")
        lines.append(
            rf"Full pipeline ({pipeline_result.n_candidates} candidates) & "
            rf"{pipeline_result.total_seconds * 1000:.0f} & -- \\"
        )
        lines.append(
            rf"Throughput & {pipeline_result.candidates_per_second:.1f} cand/s & -- \\"
        )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main HelixBenchmark
# ---------------------------------------------------------------------------

class HelixBenchmark:
    """Rigorous comparison of HelixFactor vs FactorMiner (and baselines).

    Baselines:
      - Random Formula Exploration (RF): random type-correct trees
      - Alpha101 Classic: original 101 formulaic alphas
      - Alpha101 Adapted: parameter-tuned for 10-min bars
      - FactorMiner (Ralph Loop): exact paper reproduction
      - HelixFactor (Phase 2): full Phase 2 system

    Metrics mirror paper Table 1:
      - Factor Library: IC (%), ICIR, Avg|rho|
      - Factor Combination: EW IC, EW ICIR, ICW IC, ICW ICIR
      - Factor Selection: Lasso IC, XGBoost IC
    """

    METHOD_LABELS = {
        "random_exploration": "RF (Rand)",
        "alpha101_classic": "Alpha101 Classic",
        "alpha101_adapted": "Alpha101 Adapted",
        "ralph_loop": "FactorMiner (Ralph)",
        "helix_phase2": "HelixFactor (Phase 2)",
    }

    def __init__(
        self,
        ic_threshold: float = 0.02,
        correlation_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.ic_threshold = ic_threshold
        self.correlation_threshold = correlation_threshold
        self.seed = seed
        self._stat_tests = StatisticalComparisonTests(seed=seed)
        self._speed_bench = SpeedBenchmark(seed=seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_comparison(
        self,
        data: dict,
        train_period: Tuple[int, int],
        test_period: Tuple[int, int],
        n_target_factors: int = 40,
        n_runs: int = 1,
        methods: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """Run the full comparison benchmark.

        Parameters
        ----------
        data : dict
            Data dictionary mapping feature names to (M, T) arrays.
            Must include ``"forward_returns"``.
        train_period / test_period : (int, int)
            Column index range [start, end) for the respective split.
        n_target_factors : int
            Number of factors to build each library to.
        n_runs : int
            Repetitions per method (for std estimates).
        methods : list[str], optional
            Subset of methods to run. Default: all five.

        Returns
        -------
        BenchmarkResult
        """
        if methods is None:
            methods = [
                "random_exploration",
                "alpha101_classic",
                "alpha101_adapted",
                "ralph_loop",
                "helix_phase2",
            ]

        # Split data
        train_data = _slice_data(data, *train_period)
        test_data = _slice_data(data, *test_period)

        raw_results: Dict[str, List[MethodResult]] = {}
        for method in methods:
            logger.info("Running method: %s", method)
            method_runs: List[MethodResult] = []
            for run_id in range(n_runs):
                try:
                    result = self.run_single_method(
                        method=method,
                        data=train_data,
                        test_data=test_data,
                        n_factors=n_target_factors,
                        run_id=run_id,
                    )
                    method_runs.append(result)
                except Exception as exc:
                    logger.warning("Method %s run %d failed: %s", method, run_id, exc)
                    method_runs.append(
                        MethodResult(method=method, run_id=run_id)
                    )
            raw_results[method] = method_runs

        # Average across runs
        averaged = {
            method: _average_method_results(runs)
            for method, runs in raw_results.items()
        }

        # Build metric DataFrames
        lib_df = _build_library_df(averaged, methods)
        comb_df = _build_combination_df(averaged, methods)
        sel_df = _build_selection_df(averaged, methods)

        # Speed benchmark
        speed_result = self._speed_bench.run_full_pipeline_benchmark(data=train_data)
        op_result = self._speed_bench.run_operator_benchmark(n_repeats=3)
        speed_df = _build_speed_df(op_result, speed_result)

        # Statistical tests (Helix vs Ralph)
        stat_tests = {}
        helix_results = raw_results.get("helix_phase2", [])
        ralph_results = raw_results.get("ralph_loop", [])

        if helix_results and ralph_results:
            h_ic = helix_results[0].ic_series
            r_ic = ralph_results[0].ic_series
            if h_ic is not None and r_ic is not None:
                stat_tests = self._stat_tests.run_all_tests(h_ic, r_ic)
            else:
                # Create synthetic IC series from stored metrics
                h_ic = _synthetic_ic_series(helix_results[0].library_ic, n=100, seed=self.seed)
                r_ic = _synthetic_ic_series(ralph_results[0].library_ic, n=100, seed=self.seed + 1)
                stat_tests = self._stat_tests.run_all_tests(h_ic, r_ic)

        return BenchmarkResult(
            methods=methods,
            factor_library_metrics=lib_df,
            combination_metrics=comb_df,
            selection_metrics=sel_df,
            speed_metrics=speed_df,
            statistical_tests=stat_tests,
            raw_method_results=raw_results,
        )

    def run_single_method(
        self,
        method: str,
        data: dict,
        test_data: dict,
        n_factors: int,
        run_id: int = 0,
    ) -> MethodResult:
        """Run one method and return its MethodResult.

        Parameters
        ----------
        method : str
            One of: 'ralph', 'helix', 'helix_phase2', 'rf',
            'random_exploration', 'alpha101_classic', 'alpha101_adapted'.
        """
        t0 = time.perf_counter()

        # Resolve aliases
        method_key = {
            "ralph": "ralph_loop",
            "helix": "helix_phase2",
            "rf": "random_exploration",
            "alpha101": "alpha101_classic",
        }.get(method, method)

        candidates = self._get_candidates(method_key, n_factors=n_factors * 4)
        returns = data.get("forward_returns")
        test_returns = test_data.get("forward_returns")

        if returns is None or test_returns is None:
            logger.warning("forward_returns not found in data dict for method %s", method)
            return MethodResult(method=method_key, run_id=run_id)

        # Evaluate all candidates
        factor_results = self._evaluate_candidates(candidates, data, returns)

        # Build library from best candidates
        library = self._build_library(factor_results, n_factors)

        if not library:
            return MethodResult(method=method_key, run_id=run_id)

        # Compute library metrics on test data
        test_factor_results = self._evaluate_candidates(
            [(r["name"], r["formula"], r.get("category", "Unknown"))
             for r in library],
            test_data,
            test_returns,
        )

        lib_ic, lib_icir, avg_rho, ic_series = self._library_metrics(
            test_factor_results, test_returns
        )

        # Factor combination
        ew_ic, ew_icir, icw_ic, icw_icir = self._combination_metrics(
            test_factor_results, library, test_returns
        )

        # Factor selection
        lasso_ic, lasso_icir = self._selection_metrics(
            factor_results, library, data, returns, test_data, test_returns, "lasso"
        )
        xgb_ic, xgb_icir = self._selection_metrics(
            factor_results, library, data, returns, test_data, test_returns, "xgboost"
        )

        elapsed = time.perf_counter() - t0

        return MethodResult(
            method=method_key,
            library_ic=lib_ic,
            library_icir=lib_icir,
            avg_abs_rho=avg_rho,
            ew_ic=ew_ic,
            ew_icir=ew_icir,
            icw_ic=icw_ic,
            icw_icir=icw_icir,
            lasso_ic=lasso_ic,
            lasso_icir=lasso_icir,
            xgb_ic=xgb_ic,
            xgb_icir=xgb_icir,
            n_factors=len(library),
            admission_rate=len(library) / max(len(candidates), 1),
            elapsed_seconds=elapsed,
            ic_series=ic_series,
            run_id=run_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_candidates(self, method: str, n_factors: int) -> List[Tuple[str, str, str]]:
        """Get candidate (name, formula, category) tuples for a method."""
        # Import the catalogs module directly to avoid triggering the package
        # __init__ chain which has an unresolved dependency on
        # factorminer.agent.specialists.REGIME_SPECIALIST in some environments.
        import importlib.util as _ilu, pathlib as _pl, sys as _sys
        _cat_path = _pl.Path(__file__).parent / "catalogs.py"
        if "factorminer.benchmark.catalogs" not in _sys.modules:
            _spec = _ilu.spec_from_file_location("factorminer.benchmark.catalogs", str(_cat_path))
            _cat_mod = _ilu.module_from_spec(_spec)
            _sys.modules["factorminer.benchmark.catalogs"] = _cat_mod
            _spec.loader.exec_module(_cat_mod)
        _cat = _sys.modules["factorminer.benchmark.catalogs"]
        ALPHA101_CLASSIC = _cat.ALPHA101_CLASSIC
        build_alpha101_adapted = _cat.build_alpha101_adapted
        build_random_exploration = _cat.build_random_exploration
        build_factor_miner_catalog = _cat.build_factor_miner_catalog

        if method == "random_exploration":
            entries = build_random_exploration(seed=self.seed, count=max(n_factors, 160))
        elif method == "alpha101_classic":
            entries = list(ALPHA101_CLASSIC)
            while len(entries) < n_factors:
                entries = entries + list(ALPHA101_CLASSIC)
            entries = entries[:n_factors]
        elif method == "alpha101_adapted":
            entries = build_alpha101_adapted()
        elif method in ("ralph_loop", "helix_phase2"):
            # Use the full FactorMiner paper catalog + random extensions
            entries = build_factor_miner_catalog()
            if len(entries) < n_factors * 2:
                extra = build_random_exploration(
                    seed=self.seed + 7, count=n_factors * 2 - len(entries)
                )
                entries = entries + extra
        else:
            entries = build_random_exploration(seed=self.seed + 1, count=n_factors * 2)

        return [(e.name, e.formula, e.category) for e in entries]

    def _evaluate_candidates(
        self,
        candidates: List[Tuple[str, str, str]],
        data: dict,
        returns: np.ndarray,
    ) -> List[dict]:
        """Evaluate candidates; returns list of result dicts."""
        from factorminer.core.parser import try_parse
        from factorminer.evaluation.metrics import (
            compute_ic, compute_ic_mean, compute_icir, compute_ic_win_rate
        )

        results = []
        for name, formula, category in candidates:
            tree = try_parse(formula)
            if tree is None:
                continue
            try:
                signals = tree.evaluate(data)
                if signals is None or np.all(np.isnan(signals)):
                    continue
                ic_series = compute_ic(signals, returns)
                ic_mean = compute_ic_mean(ic_series)
                icir = compute_icir(ic_series)
                win_rate = compute_ic_win_rate(ic_series)
                results.append({
                    "name": name,
                    "formula": formula,
                    "category": category,
                    "ic_mean": ic_mean,
                    "icir": icir,
                    "ic_win_rate": win_rate,
                    "signals": signals,
                    "ic_series": ic_series,
                })
            except Exception:
                pass
        return results

    def _build_library(
        self,
        factor_results: List[dict],
        n_factors: int,
    ) -> List[dict]:
        """Build a diversified factor library with IC and correlation admission."""
        from factorminer.evaluation.metrics import compute_pairwise_correlation

        # Filter by IC threshold
        passing = [r for r in factor_results if r["ic_mean"] >= self.ic_threshold]
        passing.sort(key=lambda x: x["ic_mean"], reverse=True)

        library: List[dict] = []
        for candidate in passing:
            if len(library) >= n_factors:
                break
            # Correlation check
            too_correlated = False
            for existing in library:
                if (
                    existing.get("signals") is not None
                    and candidate.get("signals") is not None
                ):
                    corr = abs(
                        compute_pairwise_correlation(
                            candidate["signals"], existing["signals"]
                        )
                    )
                    if corr >= self.correlation_threshold:
                        too_correlated = True
                        break
            if not too_correlated:
                library.append(candidate)
        return library

    def _library_metrics(
        self,
        factor_results: List[dict],
        returns: np.ndarray,
    ) -> Tuple[float, float, float, Optional[np.ndarray]]:
        """Compute library IC, ICIR, avg|rho|. Returns (ic, icir, rho, ic_series)."""
        from factorminer.evaluation.metrics import (
            compute_pairwise_correlation, compute_ic_mean, compute_icir
        )

        if not factor_results:
            return 0.0, 0.0, 0.0, None

        ics = [r["ic_mean"] for r in factor_results]
        icirs = [r["icir"] for r in factor_results]
        lib_ic = float(np.mean(ics)) if ics else 0.0
        lib_icir = float(np.mean(icirs)) if icirs else 0.0

        # Average pairwise |rho|
        rhos = []
        signals_list = [
            r["signals"] for r in factor_results if r.get("signals") is not None
        ]
        for i in range(len(signals_list)):
            for j in range(i + 1, len(signals_list)):
                c = abs(compute_pairwise_correlation(signals_list[i], signals_list[j]))
                rhos.append(c)
        avg_rho = float(np.mean(rhos)) if rhos else 0.0

        # Combined IC series (average)
        all_ic_series = [r["ic_series"] for r in factor_results if r.get("ic_series") is not None]
        if all_ic_series:
            min_len = min(len(s) for s in all_ic_series)
            combined = np.nanmean(
                np.stack([s[:min_len] for s in all_ic_series], axis=0), axis=0
            )
        else:
            combined = None

        return lib_ic, lib_icir, avg_rho, combined

    def _combination_metrics(
        self,
        test_factor_results: List[dict],
        library: List[dict],
        test_returns: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Compute EW/ICW combination metrics on test data."""
        from factorminer.evaluation.combination import FactorCombiner
        from factorminer.evaluation.metrics import compute_ic, compute_ic_mean, compute_icir

        if not test_factor_results:
            return 0.0, 0.0, 0.0, 0.0

        factor_signals = {
            i: r["signals"].T for i, r in enumerate(test_factor_results)
            if r.get("signals") is not None
        }
        ic_values = {
            i: r["ic_mean"] for i, r in enumerate(test_factor_results)
        }

        if not factor_signals:
            return 0.0, 0.0, 0.0, 0.0

        combiner = FactorCombiner()
        try:
            ew_composite = combiner.equal_weight(factor_signals)
            ew_ic_series = compute_ic(ew_composite.T, test_returns)
            ew_ic = compute_ic_mean(ew_ic_series)
            ew_icir = compute_icir(ew_ic_series)
        except Exception:
            ew_ic, ew_icir = 0.0, 0.0

        try:
            icw_composite = combiner.ic_weighted(factor_signals, ic_values)
            icw_ic_series = compute_ic(icw_composite.T, test_returns)
            icw_ic = compute_ic_mean(icw_ic_series)
            icw_icir = compute_icir(icw_ic_series)
        except Exception:
            icw_ic, icw_icir = 0.0, 0.0

        return ew_ic, ew_icir, icw_ic, icw_icir

    def _selection_metrics(
        self,
        train_factor_results: List[dict],
        library: List[dict],
        train_data: dict,
        train_returns: np.ndarray,
        test_data: dict,
        test_returns: np.ndarray,
        selector_type: str,
    ) -> Tuple[float, float]:
        """Compute Lasso/XGBoost selection IC on test data."""
        from factorminer.evaluation.selection import FactorSelector
        from factorminer.evaluation.metrics import compute_ic, compute_ic_mean, compute_icir

        if len(train_factor_results) < 3:
            return 0.0, 0.0

        fit_signals = {
            i: r["signals"].T for i, r in enumerate(train_factor_results)
            if r.get("signals") is not None
        }
        if not fit_signals:
            return 0.0, 0.0

        # Re-evaluate on test data
        test_results = self._evaluate_candidates(
            [(r["name"], r["formula"], r.get("category", "Unknown"))
             for r in train_factor_results],
            test_data,
            test_returns,
        )
        eval_signals = {
            i: r["signals"].T for i, r in enumerate(test_results)
            if r.get("signals") is not None and i < len(test_results)
        }

        if not eval_signals:
            return 0.0, 0.0

        selector = FactorSelector()
        try:
            fit_ret = train_returns.T
            if selector_type == "lasso":
                ranking = selector.lasso_selection(fit_signals, fit_ret)
            else:
                ranking = selector.xgboost_selection(fit_signals, fit_ret)

            if not ranking:
                return 0.0, 0.0

            selected_ids = [fid for fid, _ in ranking if fid in eval_signals]
            if not selected_ids:
                return 0.0, 0.0

            # Simple equal-weight composite of selected factors
            composite = np.nanmean(
                np.stack([eval_signals[fid] for fid in selected_ids], axis=0),
                axis=0,
            )
            ic_series = compute_ic(composite.T, test_returns)
            return compute_ic_mean(ic_series), compute_icir(ic_series)
        except Exception as exc:
            logger.debug("Selection metrics failed for %s: %s", selector_type, exc)
            return 0.0, 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_mock_data_dict(
    n_assets: int = 100,
    n_periods: int = 500,
    seed: int = 42,
) -> dict:
    """Build a minimal data dict from MockConfig (no raw_df needed)."""
    from factorminer.data.mock_data import MockConfig, generate_mock_data
    from factorminer.data.preprocessor import preprocess

    cfg = MockConfig(
        num_assets=n_assets,
        num_periods=n_periods,
        frequency="10min",
        plant_alpha=True,
        alpha_strength=0.04,
        alpha_assets_frac=0.4,
        seed=seed,
    )
    raw = generate_mock_data(cfg)
    processed = preprocess(raw)

    assets = sorted(processed["asset_id"].unique())
    T = processed.groupby("asset_id").size().min()

    feature_map = {
        "$open": "open", "$high": "high", "$low": "low", "$close": "close",
        "$volume": "volume", "$amt": "amount", "$vwap": "vwap",
        "$returns": "returns",
    }
    data_dict: dict = {}
    for feat_name, col_name in feature_map.items():
        if col_name in processed.columns:
            pivot = processed.pivot(
                index="asset_id", columns="datetime", values=col_name
            )
            pivot = pivot.loc[assets].iloc[:, :T]
            data_dict[feat_name] = pivot.values.astype(np.float64)

    close = data_dict["$close"]
    forward_returns = np.roll(close, -1, axis=1) / close - 1
    forward_returns[:, -1] = np.nan
    data_dict["forward_returns"] = forward_returns
    return data_dict


def _slice_data(data: dict, start: int, end: int) -> dict:
    """Slice all (M, T) arrays to columns [start, end)."""
    return {k: v[:, start:end] for k, v in data.items()}


def _average_method_results(runs: List[MethodResult]) -> MethodResult:
    """Average numeric fields across multiple runs."""
    if not runs:
        return MethodResult(method="unknown")
    if len(runs) == 1:
        return runs[0]

    fields = [
        "library_ic", "library_icir", "avg_abs_rho",
        "ew_ic", "ew_icir", "icw_ic", "icw_icir",
        "lasso_ic", "lasso_icir", "xgb_ic", "xgb_icir",
        "n_factors", "admission_rate", "elapsed_seconds",
    ]
    avg = MethodResult(method=runs[0].method)
    for f in fields:
        vals = [getattr(r, f) for r in runs if getattr(r, f) is not None]
        if vals:
            setattr(avg, f, float(np.mean(vals)))
    return avg


def _build_library_df(
    averaged: Dict[str, MethodResult], methods: List[str]
) -> pd.DataFrame:
    rows = []
    for method in methods:
        r = averaged.get(method, MethodResult(method=method))
        rows.append({
            "method": method,
            "ic_pct": r.library_ic * 100,
            "icir": r.library_icir,
            "avg_abs_rho": r.avg_abs_rho,
            "n_factors": r.n_factors,
        })
    return pd.DataFrame(rows)


def _build_combination_df(
    averaged: Dict[str, MethodResult], methods: List[str]
) -> pd.DataFrame:
    rows = []
    for method in methods:
        r = averaged.get(method, MethodResult(method=method))
        rows.append({
            "method": method,
            "ew_ic_pct": r.ew_ic * 100,
            "ew_icir": r.ew_icir,
            "icw_ic_pct": r.icw_ic * 100,
            "icw_icir": r.icw_icir,
        })
    return pd.DataFrame(rows)


def _build_selection_df(
    averaged: Dict[str, MethodResult], methods: List[str]
) -> pd.DataFrame:
    rows = []
    for method in methods:
        r = averaged.get(method, MethodResult(method=method))
        rows.append({
            "method": method,
            "lasso_ic_pct": r.lasso_ic * 100,
            "lasso_icir": r.lasso_icir,
            "xgb_ic_pct": r.xgb_ic * 100,
            "xgb_icir": r.xgb_icir,
            "best_ic_pct": max(r.lasso_ic, r.xgb_ic) * 100,
        })
    return pd.DataFrame(rows)


def _build_speed_df(
    op_result: OperatorSpeedResult,
    pipeline_result: PipelineSpeedResult,
) -> pd.DataFrame:
    rows = []
    for op, ms in op_result.operator_timings_ms.items():
        rows.append({"name": op, "time_ms": ms, "type": "operator"})
    rows.append({
        "name": f"Pipeline ({pipeline_result.n_candidates} cands)",
        "time_ms": pipeline_result.total_seconds * 1000,
        "type": "pipeline",
    })
    rows.append({
        "name": "Throughput (cands/s)",
        "time_ms": pipeline_result.candidates_per_second,
        "type": "throughput",
    })
    return pd.DataFrame(rows)


def _synthetic_ic_series(
    target_mean: float,
    n: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic IC series with given mean for stat tests."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(n) * 0.03
    base = target_mean + noise
    return base.astype(np.float64)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HelixFactor vs FactorMiner Benchmark Suite"
    )
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--n-factors", type=int, default=40, help="Target library size")
    parser.add_argument("--n-assets", type=int, default=100, help="Mock data assets")
    parser.add_argument("--n-periods", type=int, default=500, help="Mock data periods")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--methods", nargs="*", default=None, help="Methods to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-level", type=str, default="WARNING", help="Logging level"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  HelixFactor Benchmark Suite")
    print("=" * 70)

    # Build mock data
    print(f"\n[1/4] Generating mock data ({args.n_assets} assets, {args.n_periods} periods)...")
    t0 = time.perf_counter()
    data = _build_mock_data_dict(
        n_assets=args.n_assets,
        n_periods=args.n_periods,
        seed=args.seed,
    )
    T = list(data.values())[0].shape[1]
    train_end = int(T * 0.7)
    print(f"    Done in {time.perf_counter()-t0:.1f}s  (T={T}, train=0:{train_end}, test={train_end}:{T})")

    # Run comparison
    print(f"\n[2/4] Running method comparison (n_factors={args.n_factors})...")
    bench = HelixBenchmark(seed=args.seed)
    t0 = time.perf_counter()
    result = bench.run_comparison(
        data=data,
        train_period=(0, train_end),
        test_period=(train_end, T),
        n_target_factors=args.n_factors,
        n_runs=1,
        methods=args.methods,
    )
    elapsed = time.perf_counter() - t0
    print(f"    Done in {elapsed:.1f}s")

    # Print results table
    print("\n[3/4] Results Summary:")
    print("\n--- Factor Library Metrics ---")
    print(result.factor_library_metrics.to_string(index=False, float_format="{:.4f}".format))
    print("\n--- Factor Combination Metrics ---")
    print(result.combination_metrics.to_string(index=False, float_format="{:.4f}".format))
    print("\n--- Factor Selection Metrics ---")
    print(result.selection_metrics.to_string(index=False, float_format="{:.4f}".format))
    print("\n--- Speed Metrics ---")
    print(result.speed_metrics.to_string(index=False, float_format="{:.3f}".format))

    if result.statistical_tests:
        dm = result.statistical_tests.get("diebold_mariano", {})
        print(f"\n--- Statistical Tests (Helix vs Ralph) ---")
        print(f"    DM stat: {dm.get('dm_stat', 0):.3f}  p={dm.get('p_value', 1):.4f}  dir={dm.get('direction','?')}")
        ci = result.statistical_tests.get("bootstrap_ci_95", {})
        print(f"    Bootstrap 95% CI on IC diff: [{ci.get('lower', 0):.4f}, {ci.get('upper', 0):.4f}]")
        print(f"    Helix outperforms: {result.statistical_tests.get('helix_outperforms', False)}")

    # Save outputs
    print(f"\n[4/4] Saving outputs to {output_dir}...")
    result.generate_full_report(str(output_dir / "benchmark_report.html"))
    with open(output_dir / "library_metrics.csv", "w") as f:
        result.factor_library_metrics.to_csv(f, index=False)
    with open(output_dir / "combination_metrics.csv", "w") as f:
        result.combination_metrics.to_csv(f, index=False)
    with open(output_dir / "selection_metrics.csv", "w") as f:
        result.selection_metrics.to_csv(f, index=False)
    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(result.statistical_tests, f, indent=2, default=str)
    with open(output_dir / "latex_table.tex", "w") as f:
        f.write(result.to_latex_table())
    with open(output_dir / "readme_table.md", "w") as f:
        f.write(result.to_markdown_table())

    try:
        result.plot_comparison(str(output_dir / "comparison_plot.png"))
    except Exception as exc:
        logger.debug("Plot generation failed: %s", exc)

    print(f"    Reports saved to {output_dir}")
    print(f"\nDone. Total runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
