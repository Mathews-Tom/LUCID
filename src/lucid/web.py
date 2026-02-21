"""Gradio web interface for the LUCID pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


def _ensure_gradio() -> Any:
    """Import gradio with a clear error if not installed."""
    try:
        import gradio as gr
        return gr
    except ImportError:
        raise RuntimeError(
            "Gradio is not installed. Install web extras: uv add 'lucid-ai[web]'"
        ) from None


def run_detection(file_path: str, profile: str) -> tuple[str, str]:
    """Run detection-only pipeline on an uploaded file.

    Args:
        file_path: Path to uploaded file from Gradio.
        profile: Quality profile name.

    Returns:
        Tuple of (text report, JSON report string).
    """
    from lucid.config import load_config
    from lucid.output import OutputFormatter
    from lucid.pipeline import LUCIDPipeline

    config = load_config(profile=profile)
    pipeline = LUCIDPipeline(config)
    result = pipeline.run_detect_only(Path(file_path))

    formatter = OutputFormatter()
    return formatter.format_text(result), formatter.format_json(result, config)


def run_pipeline(
    file_path: str,
    profile: str,
    adversarial: bool,
) -> tuple[str, str | None]:
    """Run the full LUCID pipeline on an uploaded file.

    Args:
        file_path: Path to uploaded file from Gradio.
        profile: Quality profile name.
        adversarial: Whether to enable adversarial refinement.

    Returns:
        Tuple of (text report, output file path or None).
    """
    from lucid.config import load_config
    from lucid.output import OutputFormatter
    from lucid.pipeline import LUCIDPipeline

    overrides: dict[str, str] = {}
    if not adversarial:
        overrides["humanizer.adversarial_iterations"] = "1"

    config = load_config(profile=profile, cli_overrides=overrides if overrides else None)
    pipeline = LUCIDPipeline(config)

    input_path = Path(file_path)
    output_dir = Path(tempfile.mkdtemp(prefix="lucid_"))
    output_path = output_dir / (input_path.stem + "_humanized" + input_path.suffix)

    result = pipeline.run(input_path, output_path=output_path)

    formatter = OutputFormatter()
    text_report = formatter.format_text(result)
    return text_report, result.output_path


def create_app() -> Any:
    """Create and configure the Gradio web application.

    Returns:
        A gr.Blocks application instance.
    """
    gr = _ensure_gradio()

    with gr.Blocks(title="LUCID - AI Content Detection & Humanization") as app:
        gr.Markdown("# LUCID\n**AI Content Detection & Humanization Engine**")

        with gr.Tab("Detect"):
            with gr.Row():
                with gr.Column():
                    detect_file = gr.File(
                        label="Upload Document",
                        file_types=[".tex", ".ltx", ".md", ".markdown", ".txt"],
                    )
                    detect_profile = gr.Dropdown(
                        choices=["fast", "balanced", "quality"],
                        value="balanced",
                        label="Profile",
                    )
                    detect_btn = gr.Button("Run Detection", variant="primary")
                with gr.Column():
                    detect_text = gr.Textbox(label="Detection Report", lines=20, interactive=False)
                    detect_json = gr.Textbox(label="JSON Results", lines=10, interactive=False)

            detect_btn.click(
                fn=run_detection,
                inputs=[detect_file, detect_profile],
                outputs=[detect_text, detect_json],
            )

        with gr.Tab("Full Pipeline"):
            with gr.Row():
                with gr.Column():
                    pipeline_file = gr.File(
                        label="Upload Document",
                        file_types=[".tex", ".ltx", ".md", ".markdown", ".txt"],
                    )
                    pipeline_profile = gr.Dropdown(
                        choices=["fast", "balanced", "quality"],
                        value="balanced",
                        label="Profile",
                    )
                    pipeline_adversarial = gr.Checkbox(
                        label="Enable Adversarial Refinement", value=True,
                    )
                    pipeline_btn = gr.Button("Run Pipeline", variant="primary")
                with gr.Column():
                    pipeline_text = gr.Textbox(label="Pipeline Report", lines=20, interactive=False)
                    pipeline_download = gr.File(
                        label="Download Humanized Document", interactive=False
                    )

            pipeline_btn.click(
                fn=run_pipeline,
                inputs=[pipeline_file, pipeline_profile, pipeline_adversarial],
                outputs=[pipeline_text, pipeline_download],
            )

    return app


def main() -> None:
    """Launch the Gradio web interface."""
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
