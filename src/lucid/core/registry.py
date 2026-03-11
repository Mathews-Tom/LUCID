"""Generic plugin registry for named component lookup."""

from __future__ import annotations

from collections.abc import Callable


class Registry[T]:
    """Generic registry for named plugin components.

    Supports decorator-based registration and lookup by name.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Register a class under a key. Use as decorator."""
        def decorator(cls: type[T]) -> type[T]:
            if key in self._entries:
                raise ValueError(
                    f"{self._name} registry: key {key!r} already registered"
                )
            self._entries[key] = cls
            return cls
        return decorator

    def get(self, key: str) -> type[T]:
        """Look up a registered class by key."""
        if key not in self._entries:
            available = ", ".join(sorted(self._entries))
            raise KeyError(
                f"{self._name} registry: {key!r} not found. "
                f"Available: {available}"
            )
        return self._entries[key]

    def list_registered(self) -> list[str]:
        """Return sorted list of registered keys."""
        return sorted(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)
