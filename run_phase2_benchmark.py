#!/usr/bin/env python3
"""HelixFactor Phase 2 Comprehensive Benchmark Runner

Generates a complete publication-quality benchmarking report comparing
HelixFactor (Phase 2) against FactorMiner (Ralph Loop) and all baselines.

Usage:
    python run_phase2_benchmark.py --mock                  # quick mock data run
    python run_phase2_benchmark.py --mock --n-factors 40   # custom factor count
    python run_phase2_benchmark.py --mock --full-ablation  # include all ablations
    python run_phase2_benchmark.py --data path/to/data.csv # real data

Outputs (in results/phase2_benchmark/):
    benchmark_report.html      — full interactive HTML report
    benchmark_report.md        — GitHub-ready Markdown table
    latex_table.tex            — publication LaTeX Table 1
    ablation_table.tex         — ablation study LaTeX table
    statistical_tests.json     — all statistical test results
    library_metrics.csv        — per-method library metrics
    combination_metrics.csv    — per-method combination metrics
    selection_metrics.csv      — per-method selection metrics
    comparison_plot.png        — bar chart comparison figure
    ablation_contributions.csv — component contribution summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# Insert the repo root so that direct module imports bypass the package __init__
# (the package __init__ chains through factorminer.agent which has a known
#  import issue with build_critic_scoring_prompt; all benchmark code is
#  self-contained in helix_benchmark.py and ablation.py)
_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_module_direct(module_name: str, file_path: Path):
    """Load a Python module directly from a file path, bypassing package init."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so lazy imports inside the module work
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HelixFactor Phase 2 Comprehensive Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use synthetic mock data (no API keys needed)",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to real market data CSV",
    )
    parser.add_argument(
        "--n-factors", type=int, default=40,
        help="Target library size per method",
    )
    parser.add_argument(
        "--n-assets", type=int, default=100,
        help="Number of assets in mock data",
    )
    parser.add_argument(
        "--n-periods", type=int, default=600,
        help="Number of time periods in mock data",
    )
    parser.add_argument(
        "--output", type=str, default="results/phase2_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--methods", nargs="*", default=None,
        help="Methods to benchmark (default: all 5)",
    )
    parser.add_argument(
        "--full-ablation", action="store_true",
        help="Run full ablation study (slower)",
    )
    parser.add_argument(
        "--skip-ablation", action="store_true",
        help="Skip ablation study entirely",
    )
    parser.add_argument(
        "--log-level", type=str, default="WARNING",
        help="Logging level",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


def _subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _print_improvement_table(bench_result) -> None:
    """Print clear table showing HelixFactor improvement over FactorMiner."""
    lib = bench_result.factor_library_metrics
    comb = bench_result.combination_metrics
    sel = bench_result.selection_metrics

    helix_lib = lib[lib["method"] == "helix_phase2"]
    ralph_lib = lib[lib["method"] == "ralph_loop"]
    helix_comb = comb[comb["method"] == "helix_phase2"]
    ralph_comb = comb[comb["method"] == "ralph_loop"]
    helix_sel = sel[sel["method"] == "helix_phase2"]
    ralph_sel = sel[sel["method"] == "ralph_loop"]

    if helix_lib.empty or ralph_lib.empty:
        print("  (Could not compute improvement — method results missing)")
        return

    def _get(df, col, default=0.0):
        if df.empty or col not in df.columns:
            return default
        v = df.iloc[0][col]
        return float(v) if v == v else default  # NaN check

    h_ic = _get(helix_lib, "ic_pct")
    r_ic = _get(ralph_lib, "ic_pct")
    h_icir = _get(helix_lib, "icir")
    r_icir = _get(ralph_lib, "icir")
    h_ew = _get(helix_comb, "ew_ic_pct")
    r_ew = _get(ralph_comb, "ew_ic_pct")
    h_icw = _get(helix_comb, "icw_ic_pct")
    r_icw = _get(ralph_comb, "icw_ic_pct")
    h_las = _get(helix_sel, "lasso_ic_pct")
    r_las = _get(ralph_sel, "lasso_ic_pct")
    h_xgb = _get(helix_sel, "xgb_ic_pct")
    r_xgb = _get(ralph_sel, "xgb_ic_pct")

    def _delta(h, r):
        if r < 1e-8:
            return "N/A"
        return f"+{(h - r) / r * 100:.1f}%"

    print(f"\n  {'Metric':<28} {'FactorMiner':>12} {'HelixFactor':>12} {'Improvement':>12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12}")
    metrics = [
        ("Library IC (%)", r_ic, h_ic),
        ("Library ICIR", r_icir, h_icir),
        ("EW Combo IC (%)", r_ew, h_ew),
        ("ICW Combo IC (%)", r_icw, h_icw),
        ("LASSO Sel IC (%)", r_las, h_las),
        ("XGBoost Sel IC (%)", r_xgb, h_xgb),
    ]
    for name, r_val, h_val in metrics:
        print(
            f"  {name:<28} {r_val:>12.4f} {h_val:>12.4f} {_delta(h_val, r_val):>12}"
        )


def _fmt_stat(v, fmt=".4f") -> str:
    """Format a stat value, showing N/A for NaN."""
    if v is None:
        return "N/A"
    try:
        f = float(v)
        if f != f:  # NaN
            return "N/A"
        return format(f, fmt)
    except (TypeError, ValueError):
        return str(v)


def _print_stat_tests(stat_tests: dict) -> None:
    dm = stat_tests.get("diebold_mariano", {})
    boot = stat_tests.get("bootstrap_ci_95", {})
    tt = stat_tests.get("paired_t_test", {})
    wil = stat_tests.get("wilcoxon", {})
    mean_diff = stat_tests.get("mean_ic_difference", 0.0)

    print(f"  Mean IC difference (Helix - Ralph): {_fmt_stat(mean_diff, '+.4f')}")
    print(f"  Helix outperforms: {stat_tests.get('helix_outperforms', '?')}")
    print()

    dm_p = dm.get("p_value", float("nan"))
    dm_stat_val = dm.get("dm_stat", float("nan"))
    try:
        sig_dm = "  *" if float(dm_p) < 0.05 else ""
    except (TypeError, ValueError):
        sig_dm = ""
    print(f"  Diebold-Mariano test:")
    print(f"    DM statistic = {_fmt_stat(dm_stat_val)}{sig_dm}")
    print(f"    p-value      = {_fmt_stat(dm_p)}")
    print(f"    Direction    = {dm.get('direction', '?')}")
    print()

    tt_p = tt.get("p_value", float("nan"))
    try:
        sig_tt = "  *" if float(tt_p) < 0.05 else ""
    except (TypeError, ValueError):
        sig_tt = ""
    print(f"  Paired t-test:")
    print(f"    t-stat  = {_fmt_stat(tt.get('t_stat', float('nan')))}{sig_tt}")
    print(f"    p-value = {_fmt_stat(tt_p)}")
    print(f"    n       = {tt.get('n', 0)}")
    print()

    lo = boot.get("lower", 0.0)
    hi = boot.get("upper", 0.0)
    print(f"  Block-bootstrap 95% CI on IC difference:")
    print(f"    [{_fmt_stat(lo)}, {_fmt_stat(hi)}]  "
          f"{'(excludes zero **)' if boot.get('excludes_zero') else ''}")
    print()

    wil_p = wil.get("p_value", float("nan"))
    try:
        sig_wil = "  *" if float(wil_p) < 0.05 else ""
    except (TypeError, ValueError):
        sig_wil = ""
    print(f"  Wilcoxon signed-rank:")
    print(f"    stat    = {_fmt_stat(wil.get('statistic', 0.0), '.1f')}{sig_wil}")
    print(f"    p-value = {_fmt_stat(wil_p)}")


def _generate_markdown_report(bench_result, ablation_result, output_dir: Path) -> str:
    """Build and write a comprehensive Markdown report."""
    md = ["# HelixFactor Phase 2 Benchmark Report\n"]
    md.append(f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    md.append("\n## Table 1: Factor Library Metrics\n")
    md.append(bench_result.factor_library_metrics.to_markdown(index=False, floatfmt=".4f"))

    md.append("\n\n## Table 2: Factor Combination Metrics\n")
    md.append(bench_result.combination_metrics.to_markdown(index=False, floatfmt=".4f"))

    md.append("\n\n## Table 3: Factor Selection Metrics\n")
    md.append(bench_result.selection_metrics.to_markdown(index=False, floatfmt=".4f"))

    md.append("\n\n## Table 4: Speed Benchmarks\n")
    md.append(bench_result.speed_metrics.to_markdown(index=False, floatfmt=".3f"))

    # Statistical tests
    stat = bench_result.statistical_tests
    if stat:
        md.append("\n\n## Statistical Tests (HelixFactor vs FactorMiner)\n")
        dm = stat.get("diebold_mariano", {})
        boot = stat.get("bootstrap_ci_95", {})
        tt = stat.get("paired_t_test", {})
        md.append(f"| Test | Statistic | p-value | Significant |\n|---|---|---|---|\n")
        md.append(f"| Diebold-Mariano | {dm.get('dm_stat', 0):.4f} | {dm.get('p_value', 1):.4f} | {dm.get('significant', False)} |\n")
        md.append(f"| Paired t-test | {tt.get('t_stat', 0):.4f} | {tt.get('p_value', 1):.4f} | {tt.get('p_value', 1) < 0.05} |\n")
        md.append(f"| Bootstrap CI (95%) | [{boot.get('lower', 0):.4f}, {boot.get('upper', 0):.4f}] | — | {boot.get('excludes_zero', False)} |\n")

    if ablation_result is not None and ablation_result.contributions is not None:
        md.append("\n\n## Ablation Study: Component Contributions\n")
        md.append(ablation_result.contributions.to_markdown(index=False, floatfmt=".4f"))

    content = "\n".join(md)
    path = output_dir / "benchmark_report.md"
    with open(path, "w") as f:
        f.write(content)
    return str(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_t0 = time.perf_counter()

    # ================================================================
    # STEP 1: Data
    # ================================================================
    _section("STEP 1: Prepare Data")

    # Load benchmark modules directly to avoid triggering the package __init__
    _hb = _load_module_direct(
        "factorminer.benchmark.helix_benchmark",
        _REPO_ROOT / "factorminer" / "benchmark" / "helix_benchmark.py",
    )
    _build_mock_data_dict = _hb._build_mock_data_dict
    _slice_data = _hb._slice_data
    HelixBenchmark = _hb.HelixBenchmark

    _abl = _load_module_direct(
        "factorminer.benchmark.ablation",
        _REPO_ROOT / "factorminer" / "benchmark" / "ablation.py",
    )
    AblationStudy = _abl.AblationStudy
    ABLATION_CONFIGS = _abl.ABLATION_CONFIGS

    if args.mock or args.data is None:
        print(f"  Using mock data: {args.n_assets} assets x {args.n_periods} periods")
        np.random.seed(args.seed)
        t0 = time.perf_counter()
        data = _build_mock_data_dict(
            n_assets=args.n_assets,
            n_periods=args.n_periods,
            seed=args.seed,
        )
        print(f"  Generated in {time.perf_counter()-t0:.1f}s")
    else:
        from factorminer.data.loader import load_market_data
        from factorminer.data.preprocessor import preprocess
        print(f"  Loading real data from: {args.data}")
        t0 = time.perf_counter()
        raw = load_market_data(args.data)
        processed = preprocess(raw)
        assets = sorted(processed["asset_id"].unique())
        T = processed.groupby("asset_id").size().min()
        feature_map = {
            "$open": "open", "$high": "high", "$low": "low", "$close": "close",
            "$volume": "volume", "$amt": "amount", "$vwap": "vwap",
            "$returns": "returns",
        }
        data = {}
        for feat, col in feature_map.items():
            if col in processed.columns:
                pivot = processed.pivot(index="asset_id", columns="datetime", values=col)
                pivot = pivot.loc[assets].iloc[:, :T]
                data[feat] = pivot.values.astype(np.float64)
        close = data["$close"]
        fwd = np.roll(close, -1, axis=1) / close - 1
        fwd[:, -1] = np.nan
        data["forward_returns"] = fwd
        print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    T = list(data.values())[0].shape[1]
    train_end = int(T * 0.7)
    print(f"  Shape: M={list(data.values())[0].shape[0]}, T={T}")
    print(f"  Train: [0, {train_end})  Test: [{train_end}, {T})")

    # ================================================================
    # STEP 2: Main Comparison Benchmark
    # ================================================================
    _section("STEP 2: Main Method Comparison")

    # HelixBenchmark already loaded via _load_module_direct above

    methods = args.methods or [
        "random_exploration",
        "alpha101_classic",
        "alpha101_adapted",
        "ralph_loop",
        "helix_phase2",
    ]
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Target library size: {args.n_factors}")
    print()

    bench = HelixBenchmark(seed=args.seed)
    t0 = time.perf_counter()
    bench_result = bench.run_comparison(
        data=data,
        train_period=(0, train_end),
        test_period=(train_end, T),
        n_target_factors=args.n_factors,
        n_runs=1,
        methods=methods,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Print method-by-method summary
    _subsection("Factor Library Metrics")
    print(bench_result.factor_library_metrics.to_string(index=False, float_format="{:.4f}".format))

    _subsection("Factor Combination Metrics")
    print(bench_result.combination_metrics.to_string(index=False, float_format="{:.4f}".format))

    _subsection("Factor Selection Metrics")
    print(bench_result.selection_metrics.to_string(index=False, float_format="{:.4f}".format))

    # ================================================================
    # STEP 3: HelixFactor vs FactorMiner Improvement Table
    # ================================================================
    _section("STEP 3: HelixFactor vs FactorMiner — Improvement Summary")
    _print_improvement_table(bench_result)

    # ================================================================
    # STEP 4: Statistical Tests
    # ================================================================
    _section("STEP 4: Statistical Significance Tests")
    if bench_result.statistical_tests:
        _print_stat_tests(bench_result.statistical_tests)
    else:
        print("  (No statistical tests available — need both helix_phase2 and ralph_loop methods)")

    # ================================================================
    # STEP 5: Speed Benchmark
    # ================================================================
    _section("STEP 5: Computational Speed Benchmark")
    print(bench_result.speed_metrics.to_string(index=False, float_format="{:.3f}".format))

    # ================================================================
    # STEP 6: Ablation Study
    # ================================================================
    ablation_result = None
    if not args.skip_ablation:
        _section("STEP 6: Ablation Study")

        # AblationStudy and ABLATION_CONFIGS already loaded via _load_module_direct above

        if args.full_ablation:
            configs_to_run = list(ABLATION_CONFIGS.keys())
        else:
            # Run a focused subset for speed
            configs_to_run = [
                "full", "no_debate", "no_causal", "no_canonicalize",
                "no_regime", "no_significance", "no_memory",
            ]

        print(f"  Configurations: {', '.join(configs_to_run)}")
        t0 = time.perf_counter()

        study = AblationStudy(seed=args.seed)
        ablation_result = study.run_ablation(
            data=data,
            train_period=(0, train_end),
            test_period=(train_end, T),
            n_factors=args.n_factors,
            configs_to_run=configs_to_run,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Completed in {elapsed:.1f}s")
        study.print_summary(ablation_result)

        # Attach ablation result to bench_result
        bench_result.ablation_result = ablation_result
    else:
        _section("STEP 6: Ablation Study")
        print("  (Skipped via --skip-ablation)")

    # ================================================================
    # STEP 7: Save All Outputs
    # ================================================================
    _section("STEP 7: Save Outputs")

    # CSV tables
    bench_result.factor_library_metrics.to_csv(output_dir / "library_metrics.csv", index=False)
    bench_result.combination_metrics.to_csv(output_dir / "combination_metrics.csv", index=False)
    bench_result.selection_metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    bench_result.speed_metrics.to_csv(output_dir / "speed_metrics.csv", index=False)

    # Statistical tests JSON
    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(bench_result.statistical_tests, f, indent=2, default=str)

    # LaTeX table (Table 1 style)
    with open(output_dir / "latex_table.tex", "w") as f:
        f.write(bench_result.to_latex_table())

    # Markdown table
    with open(output_dir / "readme_table.md", "w") as f:
        f.write(bench_result.to_markdown_table())

    # HTML report
    bench_result.generate_full_report(str(output_dir / "benchmark_report.html"))

    # Comprehensive Markdown report
    md_path = _generate_markdown_report(bench_result, ablation_result, output_dir)

    # Ablation outputs
    if ablation_result is not None:
        if ablation_result.contributions is not None:
            ablation_result.contributions.to_csv(
                output_dir / "ablation_contributions.csv", index=False
            )
        # AblationStudy already available from _load_module_direct above
        abl_study = AblationStudy(seed=args.seed)
        with open(output_dir / "ablation_table.tex", "w") as f:
            f.write(abl_study.to_latex_table(ablation_result))

    # Bar chart comparison
    try:
        bench_result.plot_comparison(str(output_dir / "comparison_plot.png"))
        print(f"  comparison_plot.png saved")
    except Exception as exc:
        print(f"  (Plot skipped: {exc})")

    print(f"\n  Output files saved to: {output_dir.resolve()}")
    for fpath in sorted(output_dir.glob("*")):
        size = fpath.stat().st_size
        print(f"    {fpath.name:<40} {size:>8,} bytes")

    # ================================================================
    # SUMMARY
    # ================================================================
    _section("BENCHMARK COMPLETE")

    total_elapsed = time.perf_counter() - total_t0
    print(f"  Total runtime: {total_elapsed:.1f}s")
    print()
    print(f"  Methods benchmarked: {len(methods)}")
    print(f"  Factors per method: {args.n_factors}")

    if ablation_result is not None:
        print(f"  Ablation configs: {len(ablation_result.configs)}")

    if bench_result.statistical_tests.get("helix_outperforms"):
        print()
        print("  *** HelixFactor OUTPERFORMS FactorMiner ***")
        dm = bench_result.statistical_tests.get("diebold_mariano", {})
        if dm.get("significant"):
            print(f"  *** DM test significant: p={dm.get('p_value', 1):.4f} ***")
    print()
    print(f"  Full report: {output_dir.resolve() / 'benchmark_report.html'}")
    print(f"  Markdown:    {md_path}")
    print()


if __name__ == "__main__":
    main()
