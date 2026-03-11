"""Tests for lucid.core.registry generic plugin registry."""

from __future__ import annotations

import pytest

from lucid.core.registry import Registry


class TestRegistry:
    def test_register_and_get(self) -> None:
        reg: Registry[object] = Registry("test")

        @reg.register("foo")
        class Foo:
            pass

        assert reg.get("foo") is Foo

    def test_duplicate_registration_raises(self) -> None:
        reg: Registry[object] = Registry("test")

        @reg.register("dup")
        class First:
            pass

        with pytest.raises(ValueError, match="already registered"):
            @reg.register("dup")
            class Second:
                pass

    def test_get_unknown_key_raises(self) -> None:
        reg: Registry[object] = Registry("test")

        @reg.register("alpha")
        class Alpha:
            pass

        with pytest.raises(KeyError, match="not found.*Available: alpha"):
            reg.get("missing")

    def test_list_registered_sorted(self) -> None:
        reg: Registry[object] = Registry("test")

        @reg.register("zebra")
        class Z:
            pass

        @reg.register("apple")
        class A:
            pass

        assert reg.list_registered() == ["apple", "zebra"]

    def test_contains(self) -> None:
        reg: Registry[object] = Registry("test")

        @reg.register("item")
        class Item:
            pass

        assert "item" in reg
        assert "missing" not in reg

    def test_len(self) -> None:
        reg: Registry[object] = Registry("test")
        assert len(reg) == 0

        @reg.register("a")
        class A:
            pass

        @reg.register("b")
        class B:
            pass

        assert len(reg) == 2

    def test_decorator_preserves_class_identity(self) -> None:
        reg: Registry[object] = Registry("test")

        @reg.register("cls")
        class MyClass:
            """A docstring."""
            x: int = 42

        assert MyClass.__name__ == "MyClass"
        assert MyClass.__doc__ == "A docstring."
        assert MyClass.x == 42
        assert reg.get("cls") is MyClass
