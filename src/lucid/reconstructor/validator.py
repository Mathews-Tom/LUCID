"""Post-reconstruction validation for LaTeX and Markdown documents.

Advisory only — does not block output. Reports compilation errors
and structural issues.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt


@dataclass(frozen=True, slots=True)
class ValidationError:
    """A single validation error.

    Args:
        message: Human-readable error description.
        line: Line number where the error occurred (0 if unknown).
        context: Surrounding text or log excerpt.
    """

    message: str
    line: int = 0
    context: str = ""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of a validation check.

    Args:
        valid: True if valid, False if errors found, None if validation
            could not be performed (e.g., compiler not installed).
        errors: List of validation errors found.
    """

    valid: bool | None
    errors: tuple[ValidationError, ...] = ()


_LATEX_ERROR_RE = re.compile(r"^! (.+?)$", re.MULTILINE)
_LATEX_LINE_RE = re.compile(r"^l\.(\d+)", re.MULTILINE)


def validate_latex(
    content: str,
    compiler: str = "pdflatex",
) -> ValidationResult:
    """Validate LaTeX content by attempting compilation.

    Writes content to a temporary file and runs the compiler in
    nonstopmode. Parses the log for errors.

    Args:
        content: LaTeX document string.
        compiler: LaTeX compiler command (pdflatex, xelatex, lualatex).

    Returns:
        ValidationResult with compilation status.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "document.tex"
        tex_path.write_text(content, encoding="utf-8")

        try:
            result = subprocess.run(
                [
                    compiler,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    tmpdir,
                    str(tex_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmpdir,
            )
        except FileNotFoundError:
            # Compiler not installed
            return ValidationResult(valid=None)
        except subprocess.TimeoutExpired:
            return ValidationResult(
                valid=False,
                errors=(ValidationError(message="Compilation timed out"),),
            )

        if result.returncode == 0:
            return ValidationResult(valid=True)

        # Parse log for errors
        log_path = Path(tmpdir) / "document.log"
        log_content = ""
        if log_path.exists():
            log_content = log_path.read_text(encoding="utf-8", errors="replace")

        errors = _parse_latex_log(log_content)
        if not errors:
            # Fallback: use stderr/stdout
            errors = [
                ValidationError(
                    message=result.stdout[:200] or result.stderr[:200] or "Unknown error",
                )
            ]

        return ValidationResult(valid=False, errors=tuple(errors))


def _parse_latex_log(log: str) -> list[ValidationError]:
    """Extract errors from a LaTeX log file.

    Args:
        log: Contents of the .log file.

    Returns:
        List of ValidationError objects.
    """
    errors: list[ValidationError] = []
    error_matches = list(_LATEX_ERROR_RE.finditer(log))

    for match in error_matches:
        message = match.group(1)
        line_num = 0

        # Look for line number after the error
        remaining = log[match.end() :]
        line_match = _LATEX_LINE_RE.search(remaining[:200])
        if line_match:
            line_num = int(line_match.group(1))

        # Context: a few characters around the error
        ctx_start = max(0, match.start() - 50)
        ctx_end = min(len(log), match.end() + 100)
        context = log[ctx_start:ctx_end].strip()

        errors.append(
            ValidationError(
                message=message,
                line=line_num,
                context=context,
            )
        )

    return errors


def validate_markdown(content: str) -> ValidationResult:
    """Validate Markdown content for structural issues.

    Checks for parse errors and unbalanced dollar-sign math delimiters.

    Args:
        content: Markdown document string.

    Returns:
        ValidationResult with validation status.
    """
    errors: list[ValidationError] = []

    # Attempt to parse
    md = MarkdownIt()
    try:
        md.parse(content)
    except Exception as exc:
        errors.append(
            ValidationError(
                message=f"Parse error: {exc}",
            )
        )

    # Check balanced $ delimiters (ignoring escaped \$ and code blocks)
    dollar_count = _count_unescaped_dollars(content)
    if dollar_count % 2 != 0:
        errors.append(
            ValidationError(
                message=f"Unbalanced $ delimiters: {dollar_count} found (odd count)",
            )
        )

    if errors:
        return ValidationResult(valid=False, errors=tuple(errors))

    return ValidationResult(valid=True)


def _count_unescaped_dollars(content: str) -> int:
    """Count $ signs not inside code blocks/spans and not escaped.

    Args:
        content: Markdown document string.

    Returns:
        Count of unescaped dollar signs.
    """
    # Remove fenced code blocks
    cleaned = re.sub(r"```[\s\S]*?```", "", content)
    # Remove inline code
    cleaned = re.sub(r"`[^`]+`", "", cleaned)
    # Count unescaped $
    count = 0
    for i, ch in enumerate(cleaned):
        if ch == "$" and (i == 0 or cleaned[i - 1] != "\\"):
            count += 1
    return count
