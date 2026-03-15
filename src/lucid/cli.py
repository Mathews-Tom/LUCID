"""Click-based CLI for LUCID."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console

from lucid import __version__
from lucid.config import LUCIDConfig, load_config

_SUPPORTED_EXTENSIONS = frozenset({".tex", ".ltx", ".latex", ".md", ".markdown", ".txt", ".text"})

logger = logging.getLogger("lucid")


def _resolve_inputs(input_path: Path) -> list[Path]:
    """Resolve a path to a list of processable files.

    If input_path is a directory, find all supported files within it.
    Otherwise return the single file.
    """
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.rglob("*") if p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
        if not files:
            raise click.BadParameter(
                f"No supported files found in {input_path}", param_hint="INPUT_PATH"
            )
        return files
    return [input_path]


@click.group()
@click.version_option(version=__version__, prog_name="lucid")
@click.option(
    "--profile",
    type=click.Choice(["fast", "balanced", "quality"]),
    default=None,
    help="Quality profile (overrides config file).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to user config TOML file.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output.")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress all output.")
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Write all log output to this file (in addition to stderr).",
)
@click.pass_context
def main(
    ctx: click.Context,
    profile: str | None,
    config_path: Path | None,
    verbose: bool,
    quiet: bool,
    log_file: Path | None,
) -> None:
    """LUCID -- AI content detection and transformation engine."""
    ctx.ensure_object(dict)
    cfg = load_config(profile=profile, user_config_path=config_path)
    ctx.obj = {
        "config": cfg,
        "verbose": verbose,
        "quiet": quiet,
    }

    # Configure logging
    level = logging.DEBUG if verbose else logging.WARNING
    if quiet:
        level = logging.CRITICAL

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    fmt = logging.Formatter("%(name)s: %(message)s")

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    root_logger.addHandler(stderr_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)

    logging.getLogger("markdown_it").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Report output format.",
)
@click.option("--threshold", type=float, default=None, help="Detection threshold override.")
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.pass_context
def detect(
    ctx: click.Context,
    input_path: Path,
    output_format: str,
    threshold: float | None,
    output_path: Path | None,
) -> None:
    """Detect AI-generated content in a document."""
    from lucid.output import OutputFormatter
    from lucid.pipeline import LUCIDPipeline
    from lucid.progress import ProgressReporter

    obj = ctx.obj
    config: LUCIDConfig = obj["config"]
    if threshold is not None:
        config = load_config(
            profile=config.general.profile,
            cli_overrides={"detection.thresholds.ai_min": str(threshold)},
        )
    console = Console(stderr=True, quiet=obj["quiet"])
    reporter = ProgressReporter(console, verbose=obj["verbose"], quiet=obj["quiet"])

    files = _resolve_inputs(input_path)
    pipeline = LUCIDPipeline(config)
    formatter = OutputFormatter()

    for fpath in files:
        reporter.start(total_chunks=0)
        result = pipeline.run_detect_only(fpath, progress_callback=reporter.callback)

        if output_path is not None:
            formatter.write(result, output_path, output_format, config=config)
        else:
            if output_format == "json":
                click.echo(formatter.format_json(result, config))
            else:
                click.echo(formatter.format_text(result))

        reporter.finish(result)


@main.command(name="transform")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--report", type=click.Path(path_type=Path), default=None, help="Write report file.")
@click.option(
    "--report-format",
    type=click.Choice(["json", "text", "annotated"]),
    default="json",
    help="Report format when --report is set.",
)
@click.option("--model", default=None, help="Override Ollama model tag.")
@click.option("--search/--no-search", default=True, help="Enable search loop.")
@click.option(
    "--skip-eval",
    is_flag=True,
    default=False,
    help="Skip semantic evaluation and apply all transformed text directly.",
)
@click.pass_context
def transform_cmd(
    ctx: click.Context,
    input_path: Path,
    output_path: Path | None,
    report: Path | None,
    report_format: str,
    model: str | None,
    search: bool,
    skip_eval: bool,
) -> None:
    """Transform AI-generated content in a document."""
    from lucid.output import OutputFormatter
    from lucid.pipeline import LUCIDPipeline
    from lucid.progress import ProgressReporter

    obj = ctx.obj
    config: LUCIDConfig = obj["config"]

    # Apply CLI overrides to config
    overrides: dict[str, str] = {}
    if model is not None:
        overrides[f"ollama.models.{config.general.profile}"] = model
    if not search:
        overrides["transform.search_iterations"] = "1"
    if overrides:
        config = load_config(profile=config.general.profile, cli_overrides=overrides)

    console = Console(stderr=True, quiet=obj["quiet"])
    reporter = ProgressReporter(console, verbose=obj["verbose"], quiet=obj["quiet"])

    files = _resolve_inputs(input_path)
    pipeline = LUCIDPipeline(config, skip_eval=skip_eval)
    formatter = OutputFormatter()

    for fpath in files:
        prose_count = 0
        reporter.start(total_chunks=prose_count)
        result = pipeline.run(fpath, output_path=output_path, progress_callback=reporter.callback)
        reporter.finish(result)

        if report is not None:
            original = fpath.read_text(encoding="utf-8") if report_format == "annotated" else None
            formatter.write(
                result, report, report_format, config=config, original_content=original
            )
            click.echo(f"Report: {report}")

        if result.output_path:
            click.echo(f"Output: {result.output_path}")


@main.command(name="pipeline")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--report", type=click.Path(path_type=Path), default=None, help="Write report file.")
@click.option(
    "--output-format",
    type=click.Choice(["json", "text", "annotated"]),
    default="json",
    help="Report format.",
)
@click.option("--resume/--no-resume", default=True, help="Resume from checkpoint if available.")
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Checkpoint directory.",
)
@click.option(
    "--skip-eval",
    is_flag=True,
    default=False,
    help="Skip semantic evaluation and apply all transformed text directly.",
)
@click.pass_context
def pipeline_cmd(
    ctx: click.Context,
    input_path: Path,
    output_path: Path | None,
    report: Path | None,
    output_format: str,
    resume: bool,
    checkpoint_dir: Path | None,
    skip_eval: bool,
) -> None:
    """Run the full detect-transform-reconstruct pipeline."""
    from lucid.output import OutputFormatter
    from lucid.pipeline import LUCIDPipeline
    from lucid.progress import ProgressReporter

    obj = ctx.obj
    config: LUCIDConfig = obj["config"]
    console = Console(stderr=True, quiet=obj["quiet"])
    reporter = ProgressReporter(console, verbose=obj["verbose"], quiet=obj["quiet"])

    ckpt_dir = checkpoint_dir if resume else None
    if ckpt_dir is None and resume:
        ckpt_dir = Path.cwd() / ".lucid_checkpoints"

    files = _resolve_inputs(input_path)
    pipeline_obj = LUCIDPipeline(config, checkpoint_dir=ckpt_dir, skip_eval=skip_eval)
    formatter = OutputFormatter()

    for fpath in files:
        reporter.start(total_chunks=0)
        result = pipeline_obj.run(
            fpath, output_path=output_path, progress_callback=reporter.callback
        )
        reporter.finish(result)

        if report is not None:
            original = fpath.read_text(encoding="utf-8") if output_format == "annotated" else None
            formatter.write(
                result, report, output_format, config=config, original_content=original
            )
            click.echo(f"Report: {report}")

        if result.output_path:
            click.echo(f"Output: {result.output_path}")


@main.command(name="config")
@click.option("--set", "set_kv", nargs=2, multiple=True, help="Set KEY VALUE.")
@click.option("--profile", "show_profile", default=None, help="Show a specific profile.")
@click.pass_context
def config_cmd(
    ctx: click.Context,
    set_kv: tuple[tuple[str, str], ...],
    show_profile: str | None,
) -> None:
    """View or modify LUCID configuration."""
    import json as json_mod

    from rich.syntax import Syntax

    obj = ctx.obj
    config: LUCIDConfig = obj["config"]
    console = Console(quiet=obj["quiet"])

    if set_kv:
        overrides = dict(set_kv)
        new_config = load_config(
            profile=show_profile or config.general.profile,
            cli_overrides=overrides,
        )
        # Display the resulting config
        config = new_config

    # Serialize config to dict for display
    config_dict = {
        "general": {
            "profile": config.general.profile,
            "language": config.general.language,
            "log_level": config.general.log_level,
            "output_dir": config.general.output_dir,
        },
        "ollama": {
            "host": config.ollama.host,
            "timeout_seconds": config.ollama.timeout_seconds,
        },
        "detection": {
            "enabled": config.detection.enabled,
            "roberta_model": config.detection.roberta_model,
            "use_statistical": config.detection.use_statistical,
            "use_binoculars": config.detection.use_binoculars,
        },
        "evaluator": {
            "embedding_model": config.evaluator.embedding_model,
            "embedding_threshold": config.evaluator.embedding_threshold,
            "nli_model": config.evaluator.nli_model,
            "bertscore_threshold": config.evaluator.bertscore_threshold,
        },
    }

    json_str = json_mod.dumps(config_dict, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai")
    console.print(syntax)


@main.command(name="models")
@click.option("--download", is_flag=True, default=False, help="Download missing models.")
@click.pass_context
def models_cmd(
    ctx: click.Context,
    download: bool,
) -> None:
    """Check or download required models."""
    from rich.table import Table

    from lucid.models.download import ModelDownloader

    obj = ctx.obj
    config: LUCIDConfig = obj["config"]
    console = Console(quiet=obj["quiet"])

    downloader = ModelDownloader(config)
    statuses = downloader.check_all()

    table = Table(title="Model Status", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Source", style="blue")
    table.add_column("Available", style="green")

    for status in statuses:
        available_str = "Yes" if status.available else "[red]No[/red]"
        table.add_row(status.name, status.source, available_str)

    console.print(table)

    if download:
        missing = [s for s in statuses if not s.available]
        for status in missing:
            console.print(f"Downloading {status.name}...")
            if status.source == "huggingface":
                downloader.download_huggingface(status.name)
            elif status.source == "ollama":
                downloader.pull_ollama_model(status.name)
            console.print(f"  Done: {status.name}")


@main.group()
def bench() -> None:
    """Benchmark commands."""


@bench.command(name="run")
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_dir", type=click.Path(path_type=Path), default=None)
@click.pass_context
def bench_run(ctx: click.Context, manifest_path: Path, output_dir: Path | None) -> None:
    """Run a benchmark experiment from a manifest file."""
    from lucid.bench.datasets import DatasetLoader
    from lucid.bench.experiment import ExperimentRunner
    from lucid.bench.manifests import ExperimentManifest
    from lucid.bench.runner import BenchRunner

    manifest = ExperimentManifest.from_yaml(manifest_path)
    dataset_path = Path(manifest.dataset)

    if dataset_path.is_dir():
        samples = DatasetLoader.load_corpus(dataset_path)
    else:
        samples = DatasetLoader.load_jsonl(dataset_path)

    runner = ExperimentRunner(manifest)
    result = runner.run(samples)

    if output_dir is None:
        output_dir = Path("benchmarks/results") / manifest.name

    bench_runner = BenchRunner(output_dir)
    bench_runner.save_results(result, output_dir)
    click.echo(f"Results saved to {output_dir}")


@bench.command(name="report")
@click.argument("results_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format", "output_format",
    type=click.Choice(["markdown", "csv"]),
    default="markdown",
)
@click.pass_context
def bench_report(ctx: click.Context, results_dir: Path, output_format: str) -> None:
    """Generate a report from benchmark results."""
    import json as json_mod

    from lucid.bench.experiment import ExperimentResult
    from lucid.bench.reporting import ReportWriter
    from lucid.core.types import DetectionRecord

    detections_path = results_dir / "detections.jsonl"
    if not detections_path.exists():
        raise click.BadParameter(f"No detections.jsonl found in {results_dir}")

    detections: list[DetectionRecord] = []
    with detections_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                detections.append(DetectionRecord.from_dict(json_mod.loads(stripped)))

    # Build a minimal ExperimentResult for reporting
    result = ExperimentResult(
        manifest_name=results_dir.name,
        detections=tuple(detections),
        metrics=(),
        timestamp="",
        duration_seconds=0.0,
    )

    if output_format == "csv":
        output_path = results_dir / "metrics.csv"
        ReportWriter.write_metrics_csv(result.metrics, output_path)
    else:
        output_path = results_dir / "summary.md"
        ReportWriter.write_summary_markdown(result, output_path)

    click.echo(f"Report written to {output_path}")


@main.command(name="calibrate")
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.pass_context
def calibrate_cmd(ctx: click.Context, dataset_path: Path, output_path: Path | None) -> None:
    """Calibrate detector scores using a labeled dataset."""
    from lucid.bench.datasets import DatasetLoader
    from lucid.detector.calibrate import fit_temperature_scaling

    samples = DatasetLoader.load_jsonl(dataset_path)

    # Extract scores and labels from detection records embedded in metadata
    scores: list[float] = []
    labels: list[int] = []
    for sample in samples:
        score = sample.metadata.get("score")
        if score is None:
            continue
        scores.append(float(score))
        label = 1 if sample.source_class in {"ai_raw", "ai_edited_light", "ai_edited_heavy"} else 0
        labels.append(label)

    if not scores:
        raise click.ClickException("No samples with 'score' in metadata found for calibration.")

    calibrator = fit_temperature_scaling(scores, labels)

    if output_path is None:
        output_path = Path("calibration.json")
    calibrator.save(output_path)
    click.echo(f"Calibration saved to {output_path}")


@main.command(name="explain")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def explain_cmd(ctx: click.Context, input_path: Path) -> None:
    """Explain detection results for a document."""
    import json as json_mod

    from lucid.detector.explain import DetectionExplainer
    from lucid.pipeline import LUCIDPipeline
    from lucid.progress import ProgressReporter

    obj = ctx.obj
    config: LUCIDConfig = obj["config"]
    console = Console(stderr=True, quiet=obj["quiet"])
    reporter = ProgressReporter(console, verbose=obj["verbose"], quiet=obj["quiet"])

    files = _resolve_inputs(input_path)
    pipeline = LUCIDPipeline(config)
    explainer = DetectionExplainer()

    for fpath in files:
        reporter.start(total_chunks=0)
        result = pipeline.run_detect_only(fpath, progress_callback=reporter.callback)
        reporter.finish(result)

        for chunk_result in result.chunk_results:
            explanation = explainer.explain(chunk_result.detection)
            click.echo(json_mod.dumps(explanation, indent=2))


@main.command()
@click.option(
    "--profile",
    "setup_profile",
    type=click.Choice(["fast", "balanced", "quality"]),
    default="balanced",
    help="Quality profile to set up models for.",
)
@click.pass_context
def setup(ctx: click.Context, setup_profile: str) -> None:
    """First-run setup: check Ollama, download required models."""
    import shutil

    from rich.table import Table

    from lucid.models.download import ModelDownloader

    obj = ctx.obj
    config: LUCIDConfig = obj["config"]
    if setup_profile != config.general.profile:
        config = load_config(profile=setup_profile)
    console = Console(quiet=obj.get("quiet", False))

    # Step 1: Check Ollama binary
    console.print("\n[bold]Step 1:[/bold] Checking Ollama installation...")
    ollama_path = shutil.which("ollama")
    if ollama_path is None:
        console.print("[red]Ollama not found.[/red]")
        console.print("Install Ollama from: https://ollama.ai/download")
        ctx.exit(1)
        return
    console.print(f"  Ollama found: {ollama_path}")

    # Step 2: Check Ollama server
    console.print("\n[bold]Step 2:[/bold] Checking Ollama server...")
    downloader = ModelDownloader(config)
    if not downloader.check_ollama():
        console.print("[red]Ollama server not reachable.[/red]")
        console.print("Start Ollama with: ollama serve")
        ctx.exit(1)
        return
    console.print("  Ollama server is running.")

    # Step 3: Check/pull Ollama model
    ollama_model: str = getattr(config.ollama.models, setup_profile)
    console.print(f"\n[bold]Step 3:[/bold] Checking Ollama model ({ollama_model})...")
    if downloader.check_ollama_model(ollama_model):
        console.print(f"  Model {ollama_model} is available.")
    else:
        console.print(f"  Pulling model {ollama_model}...")
        downloader.pull_ollama_model(ollama_model)
        console.print(f"  Model {ollama_model} pulled successfully.")

    # Step 4: Download HuggingFace models
    hf_models = [
        config.detection.roberta_model,
        config.evaluator.embedding_model,
        config.evaluator.nli_model,
        config.evaluator.bertscore_model,
    ]
    console.print("\n[bold]Step 4:[/bold] Checking HuggingFace models...")
    for model_id in hf_models:
        if downloader._check_huggingface_cache(model_id):
            console.print(f"  {model_id}: cached")
        else:
            console.print(f"  Downloading {model_id}...")
            downloader.download_huggingface(model_id)
            console.print(f"  {model_id}: downloaded")

    # Step 5: Health check
    console.print("\n[bold]Step 5:[/bold] Running health check...")
    statuses = downloader.check_all()

    table = Table(title="Model Status", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Source", style="blue")
    table.add_column("Available", style="green")

    all_available = True
    for status in statuses:
        available_str = "Yes" if status.available else "[red]No[/red]"
        if not status.available:
            all_available = False
        table.add_row(status.name, status.source, available_str)

    console.print(table)

    if all_available:
        console.print("\n[bold green]Setup complete.[/bold green] All models are available.")
    else:
        console.print("\n[bold yellow]Setup incomplete.[/bold yellow] Some models are missing.")
        ctx.exit(1)
