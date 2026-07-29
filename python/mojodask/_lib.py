"""ctypes bridge to the single Mojo compilation unit."""

from __future__ import annotations

import ctypes
import os
import subprocess
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJODASK_LIB", os.path.join(ROOT, "dist", "libmojo-dask.so"))
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "md_binary": ([I, I, I, I, I], None),
    "md_scalar": ([I, F, I, I, I], None),
    "md_scalar_binary": ([I, I, F, I, I, I, I], None),
    "md_unary": ([I, I, I, I], None),
    "md_stats": ([I, I, I], None),
    "md_sum": ([I, I], F),
    "md_dot": ([I, I, I], F),
    "md_matmul": ([I, I, I, I, I, I, I], None),
    "md_matmul_gpu": ([I, I, I, I, I, I], I),
    "md_group_stats": ([I] * 10, None),
}

_library: ctypes.CDLL | None = None
_MAX_EXACT_INTEGER = 1 << 53


def build() -> str:
    if not os.path.exists(LIB):
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode:
            raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = args
            fn.restype = result
    return _library


def f64(value, *, name: str = "array") -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "bifu":
        raise TypeError(f"{name} must contain real numeric values")
    if source.dtype.kind in "iu" and source.size:
        minimum = int(source.min())
        maximum = int(source.max())
        if minimum < -_MAX_EXACT_INTEGER or maximum > _MAX_EXACT_INTEGER:
            raise OverflowError(
                f"{name} contains integers that cannot be represented exactly as float64"
            )
    if source.dtype.kind == "f" and source.dtype.itemsize > 8:
        narrowed = source.astype(np.float64)
        if not np.array_equal(source, narrowed, equal_nan=True):
            raise OverflowError(f"{name} cannot be represented exactly as float64")
    return np.ascontiguousarray(source, dtype=np.float64)


def addr(value: np.ndarray) -> int:
    address = int(value.ctypes.data)
    if value.size and not address:
        raise ValueError("NumPy supplied a null pointer for a non-empty buffer")
    return address


def _same_size(left: np.ndarray, right: np.ndarray) -> None:
    if left.size != right.size:
        raise ValueError("kernel operands must have the same number of elements")


def binary(a: np.ndarray, b: np.ndarray, op: int) -> np.ndarray:
    if op not in range(5):
        raise ValueError("invalid binary kernel operation")
    left, right = f64(a, name="left operand"), f64(b, name="right operand")
    _same_size(left, right)
    result = np.empty_like(left)
    if not left.size:
        return result
    lib().md_binary(addr(left), addr(right), addr(result), left.size, op)
    return result


def scalar(a: np.ndarray, value: float, op: int) -> np.ndarray:
    if op not in range(7):
        raise ValueError("invalid scalar kernel operation")
    source = f64(a)
    scalar_value = f64(np.asarray(value), name="scalar").item()
    result = np.empty_like(source)
    if not source.size:
        return result
    lib().md_scalar(addr(source), scalar_value, addr(result), source.size, op)
    return result


def scalar_binary(
    a: np.ndarray, b: np.ndarray, value: float, scalar_op: int, binary_op: int
) -> np.ndarray:
    if scalar_op not in range(4) or binary_op not in range(4):
        raise ValueError("invalid fused kernel operation")
    left, right = f64(a, name="left operand"), f64(b, name="right operand")
    _same_size(left, right)
    scalar_value = f64(np.asarray(value), name="scalar").item()
    result = np.empty_like(left)
    if not left.size:
        return result
    lib().md_scalar_binary(
        addr(left), addr(right), scalar_value, addr(result), left.size,
        scalar_op, binary_op,
    )
    return result


def unary(a: np.ndarray, op: int) -> np.ndarray:
    if op not in range(5):
        raise ValueError("invalid unary kernel operation")
    source = f64(a)
    result = np.empty_like(source)
    if not source.size:
        return result
    lib().md_unary(addr(source), addr(result), source.size, op)
    return result


def stats(a: np.ndarray) -> np.ndarray:
    source = f64(a)
    if not source.size:
        return np.array([0.0, 0.0, 0.0, 0.0, np.inf, -np.inf])
    result = np.empty(6, dtype=np.float64)
    lib().md_stats(addr(source), source.size, addr(result))
    return result


def sum(a: np.ndarray) -> float:
    source = f64(a)
    if not source.size:
        return 0.0
    return float(lib().md_sum(addr(source), source.size))


def dot(a: np.ndarray, b: np.ndarray) -> float:
    left, right = f64(a, name="left operand"), f64(b, name="right operand")
    _same_size(left, right)
    if not left.size:
        return 0.0
    return float(lib().md_dot(addr(left), addr(right), left.size))


def matmul(a: np.ndarray, b: np.ndarray, device: str = "cpu") -> np.ndarray:
    left, right = f64(a, name="left matrix"), f64(b, name="right matrix")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("matmul kernel requires two 2-D arrays")
    if left.shape[1] != right.shape[0]:
        raise ValueError("matmul kernel inner dimensions do not match")
    result = np.empty((left.shape[0], right.shape[1]), dtype=np.float64)
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    if not result.size or not left.shape[1]:
        result.fill(0.0)
        return result
    if device == "gpu":
        device_bytes = left.nbytes + right.nbytes + result.nbytes
        if device_bytes < 2_000_000_000 and lib().md_matmul_gpu(
            addr(left), addr(right), addr(result),
            left.shape[0], left.shape[1], right.shape[1],
        ):
            return result
        warnings.warn(
            "GPU matmul was unavailable; falling back to the CPU kernel",
            RuntimeWarning,
            stacklevel=2,
        )
    lib().md_matmul(
        addr(left), addr(right), addr(result),
        left.shape[0], left.shape[1], right.shape[1], 0,
    )
    return result


def matmul_accumulate(
    result: np.ndarray, a: np.ndarray, b: np.ndarray
) -> None:
    destination, left, right = f64(result), f64(a), f64(b)
    if destination is not result:
        raise ValueError("matmul accumulation requires a contiguous float64 result")
    if destination.ndim != 2 or left.ndim != 2 or right.ndim != 2:
        raise ValueError("matmul accumulation requires 2-D arrays")
    if left.shape[1] != right.shape[0] or destination.shape != (
        left.shape[0], right.shape[1]
    ):
        raise ValueError("matmul accumulation shapes are not aligned")
    if not destination.size or not left.shape[1]:
        return
    lib().md_matmul(
        addr(left), addr(right), addr(destination),
        left.shape[0], left.shape[1], right.shape[1], 1,
    )


def group_stats(codes: np.ndarray, values: np.ndarray, ngroups: int):
    raw_codes = np.asarray(codes)
    if raw_codes.dtype.kind not in "iu":
        raise TypeError("group codes must be integers")
    if raw_codes.size and (
        int(raw_codes.min()) < np.iinfo(np.int64).min
        or int(raw_codes.max()) > np.iinfo(np.int64).max
    ):
        raise OverflowError("group codes do not fit in int64")
    c = np.ascontiguousarray(raw_codes, dtype=np.int64)
    v = f64(values, name="group values")
    if c.ndim != 1 or v.ndim != 1 or len(c) != len(v):
        raise ValueError("group codes and values must be equal-length 1-D arrays")
    ngroups = int(ngroups)
    if ngroups < 0:
        raise ValueError("number of groups must be non-negative")
    count = np.zeros(ngroups, dtype=np.int64)
    arrays = [
        np.zeros(ngroups, dtype=np.float64),
        np.zeros(ngroups, dtype=np.float64),
        np.zeros(ngroups, dtype=np.float64),
        np.full(ngroups, np.inf, dtype=np.float64),
        np.full(ngroups, -np.inf, dtype=np.float64),
    ]
    if not len(c):
        return count, *arrays
    lib().md_group_stats(
        addr(c), addr(v), len(c), ngroups, addr(count), *(addr(x) for x in arrays)
    )
    return count, *arrays
