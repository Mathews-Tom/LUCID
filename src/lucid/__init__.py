"""LUCID: Offline-first AI content detection and transformation engine."""

from __future__ import annotations

import os

# Tokenizer-internal worker pools do not play well with the shared threaded
# evaluator path on macOS/MPS, and can leak OS-level semaphores at shutdown.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
