"""Result data models and model lifecycle management."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.models.download import ModelDownloader, ModelStatus
    from lucid.models.manager import ModelManager

__all__ = ["ModelDownloader", "ModelManager", "ModelStatus"]


def __getattr__(name: str) -> type:
    """Lazy-load model management classes on first access."""
    if name == "ModelManager":
        from lucid.models.manager import ModelManager

        return ModelManager
    if name == "ModelDownloader":
        from lucid.models.download import ModelDownloader

        return ModelDownloader
    if name == "ModelStatus":
        from lucid.models.download import ModelStatus

        return ModelStatus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
