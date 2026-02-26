"""Click-based CLI for FactorMiner."""

from __future__ import annotations

from pathlib import Path

import click

from factorminer.utils.config import load_config


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a YAML config file (merges with defaults).",
)
@click.option("--gpu/--cpu", default=True, help="Enable or disable GPU evaluation backend.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug-level logging.")
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False),
    default="output",
    help="Directory for all output artifacts.",
)
@click.version_option(package_name="factorminer")
@click.pass_context
def main(ctx: click.Context, config: str | None, gpu: bool, verbose: bool, output_dir: str) -> None:
    """FactorMiner -- LLM-powered quantitative factor mining."""
    overrides: dict = {}
    if not gpu:
        overrides.setdefault("evaluation", {})["backend"] = "numpy"

    cfg = load_config(config_path=config, overrides=overrides if overrides else None)
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["verbose"] = verbose
    ctx.obj["output_dir"] = Path(output_dir)


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------

@main.command()
@click.option("--iterations", "-n", type=int, default=None, help="Override max_iterations.")
@click.option("--batch-size", "-b", type=int, default=None, help="Override batch_size.")
@click.option("--target", "-t", type=int, default=None, help="Override target_library_size.")
@click.option("--resume", type=click.Path(exists=True), default=None, help="Resume from a saved library.")
@click.pass_context
def mine(ctx: click.Context, iterations: int | None, batch_size: int | None, target: int | None, resume: str | None) -> None:
    """Run a factor mining session."""
    cfg = ctx.obj["config"]

    if iterations is not None:
        cfg.mining.max_iterations = iterations
    if batch_size is not None:
        cfg.mining.batch_size = batch_size
    if target is not None:
        cfg.mining.target_library_size = target

    cfg.validate()

    output_dir = ctx.obj["output_dir"]
    verbose = ctx.obj["verbose"]

    click.echo(f"Starting mining session (target={cfg.mining.target_library_size}, "
               f"batch={cfg.mining.batch_size}, max_iter={cfg.mining.max_iterations})")
    click.echo(f"Output directory: {output_dir}")

    if resume:
        click.echo(f"Resuming from: {resume}")

    # Actual mining loop will be wired in by the core module
    click.echo("Mining engine not yet connected. Infrastructure ready.")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--period", type=click.Choice(["train", "test", "both"]), default="test", help="Evaluation period.")
@click.option("--top-k", type=int, default=None, help="Evaluate only the top-K factors by IC.")
@click.pass_context
def evaluate(ctx: click.Context, library_path: str, period: str, top_k: int | None) -> None:
    """Evaluate a factor library on historical data."""
    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]

    click.echo(f"Evaluating library: {library_path}")
    click.echo(f"Period: {period} | Backend: {cfg.evaluation.backend}")

    if top_k:
        click.echo(f"Evaluating top {top_k} factors only")

    click.echo("Evaluation engine not yet connected. Infrastructure ready.")


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option(
    "--method", "-m",
    type=click.Choice(["ridge", "lasso", "xgboost", "lightgbm", "ensemble"]),
    default="xgboost",
    help="Factor combination method.",
)
@click.option("--top-k", type=int, default=None, help="Select top-K factors before combining.")
@click.pass_context
def combine(ctx: click.Context, library_path: str, method: str, top_k: int | None) -> None:
    """Run factor combination and selection methods."""
    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]

    click.echo(f"Combining factors from: {library_path}")
    click.echo(f"Method: {method}")

    if top_k:
        click.echo(f"Pre-selecting top {top_k} factors")

    click.echo("Combination engine not yet connected. Infrastructure ready.")


# ---------------------------------------------------------------------------
# visualize
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--tear-sheet", is_flag=True, help="Generate a full factor tear sheet.")
@click.option("--ic-decay", is_flag=True, help="Plot IC decay curves.")
@click.option("--correlation-matrix", is_flag=True, help="Plot factor correlation heatmap.")
@click.option("--format", "fmt", type=click.Choice(["png", "pdf", "svg"]), default="png", help="Output format.")
@click.pass_context
def visualize(ctx: click.Context, library_path: str, tear_sheet: bool, ic_decay: bool, correlation_matrix: bool, fmt: str) -> None:
    """Generate plots and tear sheets for a factor library."""
    output_dir = ctx.obj["output_dir"]

    click.echo(f"Generating visualizations for: {library_path}")
    click.echo(f"Output format: {fmt}")

    plots = []
    if tear_sheet:
        plots.append("tear-sheet")
    if ic_decay:
        plots.append("ic-decay")
    if correlation_matrix:
        plots.append("correlation-matrix")

    if not plots:
        plots = ["tear-sheet", "ic-decay", "correlation-matrix"]
        click.echo("No specific plots requested; generating all.")

    for p in plots:
        click.echo(f"  - {p}")

    click.echo("Visualization engine not yet connected. Infrastructure ready.")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@main.command(name="export")
@click.argument("library_path", type=click.Path(exists=True))
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "csv", "parquet", "pickle"]),
    default="json",
    help="Export format.",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.pass_context
def export_cmd(ctx: click.Context, library_path: str, fmt: str, output: str | None) -> None:
    """Export a factor library to various formats."""
    output_dir = ctx.obj["output_dir"]

    if output is None:
        output = str(output_dir / f"library.{fmt}")

    click.echo(f"Exporting library: {library_path}")
    click.echo(f"Format: {fmt} -> {output}")
    click.echo("Export engine not yet connected. Infrastructure ready.")


# ---------------------------------------------------------------------------
# helix
# ---------------------------------------------------------------------------

@main.command()
@click.option("--iterations", "-n", type=int, default=None, help="Override max_iterations.")
@click.option("--batch-size", "-b", type=int, default=None, help="Override batch_size.")
@click.option("--target", "-t", type=int, default=None, help="Override target_library_size.")
@click.option("--resume", type=click.Path(exists=True), default=None, help="Resume from a saved library.")
@click.option("--causal/--no-causal", default=None, help="Enable/disable causal validation.")
@click.option("--regime/--no-regime", default=None, help="Enable/disable regime-conditional evaluation.")
@click.option("--debate/--no-debate", default=None, help="Enable/disable multi-specialist debate generation.")
@click.option("--canonicalize/--no-canonicalize", default=None, help="Enable/disable SymPy canonicalization.")
@click.pass_context
def helix(
    ctx: click.Context,
    iterations: int | None,
    batch_size: int | None,
    target: int | None,
    resume: str | None,
    causal: bool | None,
    regime: bool | None,
    debate: bool | None,
    canonicalize: bool | None,
) -> None:
    """Run the enhanced Helix Loop with Phase 2 features."""
    cfg = ctx.obj["config"]

    if iterations is not None:
        cfg.mining.max_iterations = iterations
    if batch_size is not None:
        cfg.mining.batch_size = batch_size
    if target is not None:
        cfg.mining.target_library_size = target

    if causal is not None:
        cfg.phase2.causal.enabled = causal
    if regime is not None:
        cfg.phase2.regime.enabled = regime
    if debate is not None:
        cfg.phase2.debate.enabled = debate
    if canonicalize is not None:
        cfg.phase2.helix.enable_canonicalization = canonicalize

    cfg.validate()

    output_dir = ctx.obj["output_dir"]
    verbose = ctx.obj["verbose"]

    enabled_features = []
    if cfg.phase2.causal.enabled:
        enabled_features.append("causal")
    if cfg.phase2.regime.enabled:
        enabled_features.append("regime")
    if cfg.phase2.capacity.enabled:
        enabled_features.append("capacity")
    if cfg.phase2.significance.enabled:
        enabled_features.append("significance")
    if cfg.phase2.debate.enabled:
        enabled_features.append("debate")
    if cfg.phase2.auto_inventor.enabled:
        enabled_features.append("auto-inventor")
    if cfg.phase2.helix.enabled:
        enabled_features.append("helix-memory")

    click.echo("HelixFactor Phase 2 mining engine.")
    click.echo(f"  Target: {cfg.mining.target_library_size} | "
               f"Batch: {cfg.mining.batch_size} | "
               f"Max iterations: {cfg.mining.max_iterations}")
    click.echo(f"  Output directory: {output_dir}")

    if enabled_features:
        click.echo(f"  Active Phase 2 features: {', '.join(enabled_features)}")
    else:
        click.echo("  No Phase 2 features enabled. Configure phase2.* in your config to enable features.")

    if resume:
        click.echo(f"  Resuming from: {resume}")

    click.echo("Helix loop not yet connected. Infrastructure ready.")


if __name__ == "__main__":
    main()
