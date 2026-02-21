# Responsible Use Policy

LUCID is a dual-use tool. It detects AI-generated content (defensive) and humanizes it (offensive). This document outlines the ethical framework governing its development and intended use.

## This Tool Is Dual-Use

AI detection tools and AI humanization tools are two sides of the same coin. LUCID provides both capabilities in a single pipeline, enabling users to:

- **Detect** AI-generated content in documents with per-paragraph scoring
- **Humanize** flagged content through iterative paraphrasing with semantic preservation

Acknowledging both capabilities transparently is a deliberate design choice. Hiding the humanization capability would not prevent its development by others, and transparency enables informed discussion about appropriate use.

## Legitimate Use Cases

- **False positive defense.** AI detectors frequently misclassify non-native English speakers, neurodivergent writers, and technical authors. LUCID helps users understand and address these false flags.
- **Writing quality improvement.** Transforms robotic AI drafts into natural-sounding prose while preserving technical accuracy.
- **Research tool.** Enables study of detection robustness, adversarial text generation, and the fundamental limits of AI text detection.
- **Accessibility.** Allows users who rely on AI writing assistance to produce text that reflects their intended voice without triggering automated penalties.

## Ethical Guidelines

1. **Human-in-the-loop.** All LUCID outputs require human review. The system provides scores and suggestions, not autonomous publishing. Users are responsible for reviewing and approving all output.
2. **No autonomous publishing.** LUCID is not designed to produce publish-ready content without human oversight. Outputs should be reviewed, edited, and approved before submission.
3. **Disclosed use.** Users should comply with the disclosure requirements of their institution, publisher, or jurisdiction when submitting LUCID-processed content.
4. **Academic integrity.** Users in academic settings must follow their institution's policies regarding AI-assisted writing tools.

## What This Tool Does Not Do

- **Not a cheating tool.** LUCID does not write content. It paraphrases existing text to reduce AI detection scores while preserving meaning. The intellectual contribution remains the user's responsibility.
- **Not plagiarism.** LUCID operates on the user's own text (or text the user has the right to modify). It does not copy from external sources.
- **Not a guaranteed bypass.** Detection evasion rates vary by detector, model, and document type. LUCID makes no guarantees about defeating specific commercial detection services.
- **Not a SaaS product.** LUCID runs entirely offline on local hardware. No data is transmitted to external servers.

## EU AI Act Compliance

The EU AI Act (Article 50, effective August 2026) establishes requirements for AI-generated content disclosure and watermarking. LUCID's position:

- **No watermark stripping.** LUCID does not specifically target or remove AI watermarks. The humanization process operates at the linguistic level (paraphrasing, restructuring) rather than targeting encoding-level watermarks.
- **Linguistic-level processing only.** All transformations are applied to the surface text through standard NLP techniques (paraphrasing, vocabulary substitution, structural reordering). No steganographic or encoding-level modifications are performed.
- **Transparency.** All detection accuracy and evasion rates are published with methodology, enabling regulators and researchers to assess the tool's capabilities.

## Model Licensing

LUCID depends on several pre-trained models, each with its own license:

| Model | License | Commercial Use |
|-------|---------|----------------|
| Phi-3 / Phi-4 | MIT | Unrestricted |
| Mistral 7B | Apache 2.0 | Unrestricted |
| Qwen2.5 | Apache 2.0 | Unrestricted |
| Llama 3.x | Meta Community License | With attribution, <700M MAU |
| Gemma 2 | Google Use Terms | Google reserves remote restriction rights |
| RoBERTa | MIT | Unrestricted |
| DeBERTa | MIT | Unrestricted |

**Recommended default models for lowest licensing risk:** Phi-3-mini (3.8B, MIT) and Qwen2.5-7B (Apache 2.0).

## Responsible Development Commitments

- **Transparency.** All detection accuracy and evasion rates are published with full methodology.
- **No watermark targeting.** The system does not specifically target or remove AI watermarks.
- **Open benchmarking.** Detection and humanization results are reproducible via the included benchmark suite (`tests/benchmarks/`).
- **Human oversight required.** The tool is designed for human-in-the-loop workflows, not automated content generation.
- **Open source.** LUCID is released under the MIT license, enabling community review and audit of all capabilities.
