"""
RailNet — DType abstraction.

Every numeric format implements RailDType.
Compiler / runtime / artifact layer is dtype-agnostic and
dispatches through this interface.

Status labels per SPEC.md:
  PROVEN / EXPERIMENTAL / PLANNED
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DTypeInfo:
    name: str
    bits: int
    is_float: bool
    is_integer: bool
    status: str  # PROVEN | EXPERIMENTAL | PLANNED | READY


class RailDType(abc.ABC):
    """Generic dtype contract. All formats subclass this."""

    info: DTypeInfo

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def bits(self) -> int:
        return self.info.bits

    @property
    def is_float(self) -> bool:
        return self.info.is_float

    @property
    def is_integer(self) -> bool:
        return self.info.is_integer

    # --- encode / decode ---

    @abc.abstractmethod
    def encode(self, value: Any) -> int:
        """Python scalar -> raw bit pattern (uint)."""
        ...

    @abc.abstractmethod
    def decode(self, bits: int) -> Any:
        """Raw bit pattern -> Python scalar."""
        ...

    def encode_array(self, values: np.ndarray) -> np.ndarray:
        return np.vectorize(self.encode, otypes=[np.uint16])(values)

    def decode_array(self, bits_arr: np.ndarray) -> np.ndarray:
        return np.vectorize(self.decode, otypes=[np.float64])(bits_arr)

    # --- quantize ---

    @abc.abstractmethod
    def quantize(self, value: Any) -> Any:
        """Round-trip through dtype representation."""
        ...

    # --- exact equality ---

    def exact_equal(self, a_bits: int, b_bits: int) -> bool:
        return int(a_bits) == int(b_bits)

    def exact_equal_array(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.dtype == b.dtype:
            st = self.storage_dtype
            return a.view(st) == b.view(st)
        return a == b

    # --- numpy helpers ---

    @property
    @abc.abstractmethod
    def numpy_dtype(self):
        ...

    @property
    @abc.abstractmethod
    def storage_dtype(self):
        """uint dtype that stores raw bits."""
        ...

    def __repr__(self) -> str:
        return f"{self.info.name}(bits={self.info.bits}, status={self.info.status})"


_REGISTRY: dict[str, RailDType] = {}


def register(dtype_or_cls) -> RailDType:
    # supports @register on class (instantiate) or instance
    import inspect
    if inspect.isclass(dtype_or_cls) and issubclass(dtype_or_cls, RailDType):
        inst = dtype_or_cls()
        dtype = inst
        cls = dtype_or_cls
    else:
        dtype = dtype_or_cls
        cls = dtype_or_cls
    _REGISTRY[dtype.name] = dtype
    _REGISTRY[dtype.name.lower()] = dtype
    return cls if inspect.isclass(dtype_or_cls) else dtype


def get_dtype(name: str) -> RailDType:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown dtype '{name}'. Registered: {sorted(set(k for k in _REGISTRY if k.islower()))}")
    return _REGISTRY[key]


def registered_dtypes() -> dict[str, RailDType]:
    return dict(_REGISTRY)
