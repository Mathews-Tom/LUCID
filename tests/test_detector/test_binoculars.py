"""Unit tests for Binoculars Tier 3 detector scaffolding.

No actual model downloads are performed — all HuggingFace and torch
dependencies are mocked throughout.
"""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from lucid.detector import BinocularsUnavailableError
from lucid.detector.binoculars import (
    BINOCULARS_THRESHOLD,
    SIGMOID_SCALE,
    BinocularsDetector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_normalized(raw: float) -> float:
    """Replicate the normalization formula for expected-value assertions."""
    exponent = (raw - BINOCULARS_THRESHOLD) * SIGMOID_SCALE
    if exponent >= 709.0:
        return 0.0
    if exponent <= -709.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(exponent))))


def _make_torch_stub() -> types.ModuleType:
    """Build a minimal torch stub sufficient for BinocularsDetector._ensure_loaded."""
    torch_mod = types.ModuleType("torch")
    torch_mod.float16 = "float16"  # type: ignore[attr-defined]
    torch_mod.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))  # type: ignore[attr-defined]
    torch_mod.cuda = MagicMock()  # type: ignore[attr-defined]
    torch_mod.cuda.is_available = MagicMock(return_value=False)  # type: ignore[attr-defined]
    return torch_mod


def _make_transformers_stub() -> types.ModuleType:
    """Build a minimal transformers stub with AutoTokenizer and AutoModelForCausalLM."""
    transformers_mod = types.ModuleType("transformers")

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.eval = MagicMock(return_value=None)

    auto_tokenizer = MagicMock()
    auto_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)

    auto_model = MagicMock()
    auto_model.from_pretrained = MagicMock(return_value=mock_model)

    transformers_mod.AutoTokenizer = auto_tokenizer  # type: ignore[attr-defined]
    transformers_mod.AutoModelForCausalLM = auto_model  # type: ignore[attr-defined]

    return transformers_mod


# ---------------------------------------------------------------------------
# _normalize_score tests
# ---------------------------------------------------------------------------


class TestNormalizeScore:
    """_normalize_score maps raw ratios to [0, 1] with correct polarity."""

    def test_low_raw_near_one(self) -> None:
        """Raw ratio well below threshold should produce a score near 1.0 (AI).

        At raw=0.4 (0.5 below threshold=0.9015, scale=10): sigmoid input=-5,
        giving ~0.9933.
        """
        score = BinocularsDetector._normalize_score(0.4)
        assert score > 0.9, f"Expected >0.9 for raw=0.4, got {score}"

    def test_high_raw_near_zero(self) -> None:
        """Raw ratio well above threshold should produce a score near 0.0 (human).

        At raw=1.4 (0.5 above threshold=0.9015, scale=10): sigmoid input=+5,
        giving ~0.0067.
        """
        score = BinocularsDetector._normalize_score(1.4)
        assert score < 0.1, f"Expected <0.1 for raw=1.4, got {score}"

    def test_threshold_maps_to_half(self) -> None:
        """Raw ratio exactly at threshold should produce approximately 0.5."""
        score = BinocularsDetector._normalize_score(BINOCULARS_THRESHOLD)
        assert abs(score - 0.5) < 1e-6, f"Expected ~0.5 at threshold, got {score}"

    def test_output_clamped_to_unit_interval(self) -> None:
        """Output is always within [0.0, 1.0] regardless of extreme raw values.

        Extreme values would overflow math.exp without overflow guarding.
        """
        high = BinocularsDetector._normalize_score(-1000.0)
        assert 0.0 <= high <= 1.0, f"Score out of range for raw=-1000: {high}"
        low = BinocularsDetector._normalize_score(1000.0)
        assert 0.0 <= low <= 1.0, f"Score out of range for raw=1000: {low}"

    def test_formula_matches_reference(self) -> None:
        """Spot-check several values against the reference formula."""
        for raw in [0.5, 0.7, BINOCULARS_THRESHOLD, 1.0, 1.2]:
            expected = _expected_normalized(raw)
            actual = BinocularsDetector._normalize_score(raw)
            assert abs(actual - expected) < 1e-9, (
                f"Mismatch at raw={raw}: expected={expected}, actual={actual}"
            )


# ---------------------------------------------------------------------------
# Lazy loading tests
# ---------------------------------------------------------------------------


class TestLazyLoading:
    """BinocularsDetector defers model loading until score() is called."""

    def test_not_loaded_on_init(self) -> None:
        """Models must not be loaded at construction time."""
        detector = BinocularsDetector()
        assert detector.loaded is False

    def test_loaded_property_reflects_state(self) -> None:
        """loaded property returns False before and True after _ensure_loaded."""
        detector = BinocularsDetector()
        assert not detector.loaded

        torch_stub = _make_torch_stub()
        transformers_stub = _make_transformers_stub()

        with (
            patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}),
        ):
            detector._ensure_loaded()

        assert detector.loaded is True

    def test_ensure_loaded_is_idempotent(self) -> None:
        """Calling _ensure_loaded twice should not attempt model loading twice."""
        detector = BinocularsDetector()

        torch_stub = _make_torch_stub()
        transformers_stub = _make_transformers_stub()

        with patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}):
            detector._ensure_loaded()
            # Second call must not re-invoke from_pretrained
            call_count_before = transformers_stub.AutoTokenizer.from_pretrained.call_count
            detector._ensure_loaded()
            assert transformers_stub.AutoTokenizer.from_pretrained.call_count == call_count_before


# ---------------------------------------------------------------------------
# unload() tests
# ---------------------------------------------------------------------------


class TestUnload:
    """unload() releases model references and resets loaded state."""

    def test_unload_sets_loaded_false(self) -> None:
        """After unload(), loaded property must be False."""
        detector = BinocularsDetector()

        torch_stub = _make_torch_stub()
        transformers_stub = _make_transformers_stub()

        with patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}):
            detector._ensure_loaded()
            assert detector.loaded is True
            detector.unload()

        assert detector.loaded is False

    def test_unload_clears_model_references(self) -> None:
        """After unload(), internal model attributes must be None."""
        detector = BinocularsDetector()

        torch_stub = _make_torch_stub()
        transformers_stub = _make_transformers_stub()

        with patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}):
            detector._ensure_loaded()
            detector.unload()

        assert detector._observer is None
        assert detector._performer is None
        assert detector._tokenizer is None

    def test_unload_without_loading_is_safe(self) -> None:
        """unload() on a never-loaded detector must not raise."""
        detector = BinocularsDetector()
        # torch not needed for unload on an unloaded detector
        with patch.dict(sys.modules, {}):
            detector.unload()  # must not raise
        assert detector.loaded is False

    def test_unload_cuda_cache_cleared_when_available(self) -> None:
        """unload() calls torch.cuda.empty_cache() when CUDA is available."""
        detector = BinocularsDetector()

        torch_stub = _make_torch_stub()
        transformers_stub = _make_transformers_stub()

        with patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}):
            detector._ensure_loaded()
            torch_stub.cuda.is_available = MagicMock(return_value=True)  # type: ignore[attr-defined]
            torch_stub.cuda.empty_cache = MagicMock()  # type: ignore[attr-defined]
            detector.unload()
            torch_stub.cuda.empty_cache.assert_called_once()


# ---------------------------------------------------------------------------
# BinocularsUnavailableError on missing torch
# ---------------------------------------------------------------------------


class TestBinocularsUnavailableError:
    """score() raises BinocularsUnavailableError when torch is absent."""

    def test_raises_when_torch_missing(self) -> None:
        """ImportError for torch must be converted to BinocularsUnavailableError."""
        detector = BinocularsDetector()

        # Remove torch from sys.modules to simulate it being absent
        saved = sys.modules.pop("torch", None)
        saved_transformers = sys.modules.pop("transformers", None)
        try:
            with pytest.raises(BinocularsUnavailableError, match="torch and transformers required"):
                detector._ensure_loaded()
        finally:
            if saved is not None:
                sys.modules["torch"] = saved
            if saved_transformers is not None:
                sys.modules["transformers"] = saved_transformers

    def test_raises_when_model_load_fails(self) -> None:
        """Exceptions during model loading must be wrapped in BinocularsUnavailableError."""
        detector = BinocularsDetector()

        torch_stub = _make_torch_stub()
        transformers_stub = _make_transformers_stub()
        transformers_stub.AutoModelForCausalLM.from_pretrained.side_effect = RuntimeError("OOM")

        with (
            patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}),
            pytest.raises(BinocularsUnavailableError, match="Failed to load Binoculars models"),
        ):
            detector._ensure_loaded()

    def test_unavailable_error_is_detector_error(self) -> None:
        """BinocularsUnavailableError must be a DetectorError subclass."""
        from lucid.detector import DetectorError

        assert issubclass(BinocularsUnavailableError, DetectorError)
