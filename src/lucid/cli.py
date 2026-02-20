"""Click-based CLI for LUCID."""

from __future__ import annotations

from pathlib import Path

import click

from lucid import __version__
from lucid.config import LUCIDConfig, load_config

pass_config = click.make_pass_decorator(LUCIDConfig, ensure=True)


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
@click.pass_context
def main(ctx: click.Context, profile: str | None, config_path: Path | None) -> None:
    """LUCID — AI content detection and humanization engine."""
    ctx.ensure_object(dict)
    cfg = load_config(profile=profile, user_config_path=config_path)
    ctx.obj = cfg


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.pass_obj
def detect(config: LUCIDConfig, input_path: Path) -> None:
    """Detect AI-generated content in a document."""
    raise NotImplementedError("Detection pipeline not yet implemented (Phase 2)")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.pass_obj
def humanize(config: LUCIDConfig, input_path: Path) -> None:
    """Humanize AI-generated content in a document."""
    raise NotImplementedError("Humanization pipeline not yet implemented (Phase 3)")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.pass_obj
def pipeline(config: LUCIDConfig, input_path: Path) -> None:
    """Run the full detect-humanize-reconstruct pipeline."""
    raise NotImplementedError("Full pipeline not yet implemented (Phase 5)")
