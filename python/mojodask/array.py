"""Dask-array-shaped lazy blocked arrays backed by Mojo kernels."""

from __future__ import annotations

import builtins
import math
from itertools import product

import numpy as np

from . import _lib
from .core import Scalar, Task, execute, merge_graphs, new_key

_PARALLEL_REDUCTION_SIZE = 1_000_000


def _require_float64_dtype(dtype):
    if dtype is not None and np.dtype(dtype) != np.dtype(np.float64):
        raise NotImplementedError("mojo-dask arrays have float64 dtype")


def _normalise_chunks(shape, chunks):
    if chunks == "auto":
        target = 1_000_000
        if len(shape) == 1:
            chunks = min(shape[0], target)
        else:
            edge = max(1, int(target ** (1 / len(shape))))
            chunks = tuple(min(size, edge) for size in shape)
    if isinstance(chunks, int):
        chunks = (chunks,) * len(shape)
    if len(shape) == 1 and chunks and isinstance(chunks[0], int):
        chunks = tuple(chunks)
        if len(chunks) > 1 and sum(chunks) == shape[0]:
            return (chunks,)
    if len(chunks) != len(shape):
        raise ValueError("chunks must have one entry per dimension")
    result = []
    for size, spec in zip(shape, chunks):
        if isinstance(spec, int):
            if spec <= 0:
                raise ValueError("chunk sizes must be positive")
            full, tail = divmod(size, spec)
            result.append((spec,) * full + ((tail,) if tail else ()))
        else:
            sizes = tuple(int(value) for value in spec)
            if sum(sizes) != size:
                raise ValueError("explicit chunks must add up to the dimension")
            result.append(sizes)
    return tuple(result)


def _slices(chunks):
    dimensions = []
    for sizes in chunks:
        start = 0
        dimension = []
        for size in sizes:
            dimension.append(slice(start, start + size))
            start += size
        dimensions.append(dimension)
    return dimensions


def _assemble(shape, chunks, indices, *blocks):
    result = np.empty(shape, dtype=np.float64)
    slices = _slices(chunks)
    for index, block in zip(indices, blocks):
        result[tuple(slices[axis][part] for axis, part in enumerate(index))] = block
    return result


def _block_indices(chunks):
    return tuple(product(*(range(len(axis)) for axis in chunks)))


class Array:
    __array_priority__ = 1000

    def __init__(self, graph, blocks, shape, chunks, name=None, block_expressions=None):
        self._graph = graph
        self._blocks = dict(blocks)
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.chunks = tuple(tuple(axis) for axis in chunks)
        self.dtype = np.dtype("float64")
        self.name = name or new_key("array")
        self._block_expressions = block_expressions or {}
        self._key = new_key("assemble")
        indices = _block_indices(self.chunks)
        dependencies = tuple(self._blocks[index] for index in indices)
        self._graph[self._key] = Task(
            lambda *values, shape=self.shape, chunks=self.chunks, indices=indices:
                _assemble(shape, chunks, indices, *values),
            dependencies,
        )

    @property
    def numblocks(self):
        return tuple(len(axis) for axis in self.chunks)

    @property
    def size(self):
        return math.prod(self.shape)

    def compute(self, scheduler=None, **kwargs):
        return execute(
            self._graph, (self._key,), scheduler, kwargs.get("num_workers")
        )[0]

    def persist(self, scheduler=None, **kwargs):
        values = execute(
            self._graph,
            tuple(self._blocks[index] for index in _block_indices(self.chunks)),
            scheduler,
            kwargs.get("num_workers"),
        )
        graph = {}
        blocks = {}
        for index, value in zip(_block_indices(self.chunks), values):
            key = new_key("persist")
            graph[key] = Task(lambda value=value: value)
            blocks[index] = key
        return Array(graph, blocks, self.shape, self.chunks, self.name)

    def rechunk(self, chunks="auto", threshold=None, block_size_limit=None, balance=False):
        del threshold, block_size_limit, balance
        target = _normalise_chunks(self.shape, chunks)
        slices = _slices(target)
        graph = dict(self._graph)
        blocks = {}
        for index in _block_indices(target):
            selection = tuple(
                slices[axis][part] for axis, part in enumerate(index)
            )
            key = new_key("rechunk")
            graph[key] = Task(
                lambda whole, selection=selection:
                    _lib.f64(whole[selection]),
                (self._key,),
            )
            blocks[index] = key
        return Array(graph, blocks, self.shape, target)

    def map_blocks(self, func, *args, dtype=None, chunks=None, **kwargs):
        del dtype
        if args:
            raise NotImplementedError("collection arguments to map_blocks are not covered")
        graph = dict(self._graph)
        blocks = {}
        for index, dependency in self._blocks.items():
            key = new_key("map-blocks")
            graph[key] = Task(
                lambda block, func=func, kwargs=kwargs:
                    _lib.f64(func(block, **kwargs), name="map_blocks result"),
                (dependency,),
            )
            blocks[index] = key
        target = self.chunks if chunks is None else _normalise_chunks(self.shape, chunks)
        return Array(graph, blocks, self.shape, target)

    def _binary(self, other, opcode):
        if np.isscalar(other):
            graph = dict(self._graph)
            blocks = {}
            expressions = {}
            for index, dependency in self._blocks.items():
                key = new_key("scalar-op")
                value = other
                graph[key] = Task(
                    lambda block, value=value, opcode=opcode:
                        _lib.scalar(block, value, opcode),
                    (dependency,),
                )
                blocks[index] = key
                if opcode < 4:
                    expressions[index] = (dependency, value, opcode)
            return Array(
                graph, blocks, self.shape, self.chunks,
                block_expressions=expressions,
            )
        other = asarray(other, chunks=self.chunks)
        if self.shape != other.shape:
            raise ValueError("covered elementwise operations require equal shapes")
        if self.chunks != other.chunks:
            other = other.rechunk(self.chunks)
        graph = merge_graphs(self._graph, other._graph)
        blocks = {}
        for index in self._blocks:
            key = new_key("binary-op")
            expression = self._block_expressions.get(index)
            if expression is not None and opcode < 4:
                source, value, scalar_opcode = expression
                graph[key] = Task(
                    lambda left, right, value=value, scalar_opcode=scalar_opcode,
                           opcode=opcode:
                        _lib.scalar_binary(
                            left, right, value, scalar_opcode, opcode
                        ),
                    (source, other._blocks[index]),
                )
            else:
                graph[key] = Task(
                    lambda left, right, opcode=opcode:
                        _lib.binary(left, right, opcode),
                    (self._blocks[index], other._blocks[index]),
                )
            blocks[index] = key
        return Array(graph, blocks, self.shape, self.chunks)

    def __add__(self, other):
        return self._binary(other, 0)

    def __radd__(self, other):
        return self._binary(other, 0)

    def __sub__(self, other):
        return self._binary(other, 1)

    def __rsub__(self, other):
        return self._binary(other, 5)

    def __mul__(self, other):
        return self._binary(other, 2)

    def __rmul__(self, other):
        return self._binary(other, 2)

    def __truediv__(self, other):
        return self._binary(other, 3)

    def __rtruediv__(self, other):
        return self._binary(other, 6)

    def __pow__(self, other):
        return self._binary(other, 4)

    def __neg__(self):
        return _unary(self, 0)

    def __abs__(self):
        return _unary(self, 1)

    def _reduce(self, operation, axis=None, dtype=None, keepdims=False, ddof=0):
        _require_float64_dtype(dtype)
        graph = dict(self._graph)
        key = new_key(operation)
        if axis is not None:
            functions = {
                "sum": np.sum, "mean": np.mean, "var": np.var, "std": np.std,
                "min": np.min, "max": np.max,
            }
            graph[key] = Task(
                lambda whole, fn=functions[operation], axis=axis,
                       keepdims=keepdims, ddof=ddof:
                    fn(whole, axis=axis, keepdims=keepdims, **(
                        {"ddof": ddof} if operation in ("var", "std") else {}
                    )),
                (self._key,),
            )
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            if any(
                not isinstance(value, (int, np.integer))
                or value < -self.ndim
                or value >= self.ndim
                for value in axes
            ):
                raise np.exceptions.AxisError(axis, ndim=self.ndim)
            axes = tuple(int(value) % self.ndim for value in axes)
            if len(set(axes)) != len(axes):
                raise ValueError("duplicate value in 'axis'")
            result_shape = tuple(
                1 if keepdims and dimension in axes else size
                for dimension, size in enumerate(self.shape)
                if keepdims or dimension not in axes
            )
            if not result_shape:
                return Scalar(graph, key)
            result_chunks = tuple((size,) for size in result_shape)
            block_index = tuple(0 for _ in result_shape)
            return Array(graph, {block_index: key}, result_shape, result_chunks)

        dependencies = tuple(
            self._blocks[index] for index in _block_indices(self.chunks)
        )
        precomputed_parts = len(dependencies) > 1 and (
            self.size >= _PARALLEL_REDUCTION_SIZE
        )
        if precomputed_parts:
            part_function = (
                _lib.sum if operation in ("sum", "mean") else _lib.stats
            )
            part_dependencies = []
            for dependency in dependencies:
                part_key = new_key(f"{operation}-part")
                graph[part_key] = Task(part_function, (dependency,))
                part_dependencies.append(part_key)
            dependencies = tuple(part_dependencies)

        def reduce_blocks(*values):
            if not values:
                if operation == "sum":
                    return 0.0
                raise ValueError("zero-size array reduction")
            if operation in ("sum", "mean"):
                totals = (
                    list(values)
                    if precomputed_parts
                    else [_lib.sum(block) for block in values]
                )
                total = sum(totals)
                if any(np.isnan(value) for value in totals):
                    return float("nan")
                value = total if operation == "sum" else total / self.size
                if keepdims:
                    return np.full((1,) * self.ndim, value)
                return float(value)
            parts = (
                list(values)
                if precomputed_parts
                else [_lib.stats(block) for block in values]
            )
            if any(np.isnan(part[1]) for part in parts):
                return float("nan")
            count = sum(part[0] for part in parts)
            total = sum(part[1] for part in parts)
            if operation == "min":
                value = min(part[4] for part in parts)
            elif operation == "max":
                value = max(part[5] for part in parts)
            else:
                mean = 0.0
                m2 = 0.0
                seen = 0.0
                for part in parts:
                    n, _, part_mean, part_m2 = part[:4]
                    if not n:
                        continue
                    delta = part_mean - mean
                    combined = seen + n
                    m2 += part_m2 + delta * delta * seen * n / combined
                    mean += delta * n / combined
                    seen = combined
                variance = m2 / (seen - ddof)
                value = math.sqrt(variance) if operation == "std" else variance
            if keepdims:
                return np.full((1,) * self.ndim, value)
            return float(value)

        graph[key] = Task(reduce_blocks, dependencies)
        if keepdims:
            shape = (1,) * self.ndim
            chunks = ((1,),) * self.ndim
            return Array(
                graph,
                {tuple(0 for _ in shape): key},
                shape,
                chunks,
            )
        return Scalar(graph, key)

    def sum(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        del split_every, out
        return self._reduce("sum", axis, dtype, keepdims)

    def mean(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        del split_every, out
        return self._reduce("mean", axis, dtype, keepdims)

    def var(self, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None):
        del split_every, out
        return self._reduce("var", axis, dtype, keepdims, ddof)

    def std(self, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None):
        del split_every, out
        return self._reduce("std", axis, dtype, keepdims, ddof)

    def min(self, axis=None, keepdims=False, split_every=None, out=None):
        del split_every, out
        return self._reduce("min", axis, None, keepdims)

    def max(self, axis=None, keepdims=False, split_every=None, out=None):
        del split_every, out
        return self._reduce("max", axis, None, keepdims)

    def dot(self, other):
        return dot(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __getitem__(self, selection):
        sample_shape = np.empty(self.shape)[selection].shape
        graph = dict(self._graph)
        key = new_key("getitem")
        graph[key] = Task(
            lambda whole, selection=selection:
                _lib.f64(whole[selection]),
            (self._key,),
        )
        if not sample_shape:
            return Scalar(graph, key)
        chunks = tuple((size,) for size in sample_shape)
        return Array(graph, {tuple(0 for _ in sample_shape): key}, sample_shape, chunks)

    def reshape(self, *shape, merge_chunks=True, limit=None):
        del merge_chunks, limit
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        resolved = np.empty(self.shape).reshape(*shape).shape
        graph = dict(self._graph)
        key = new_key("reshape")
        graph[key] = Task(
            lambda whole, shape=resolved:
                _lib.f64(whole.reshape(shape)),
            (self._key,),
        )
        chunks = tuple((size,) for size in resolved)
        return Array(graph, {tuple(0 for _ in resolved): key}, resolved, chunks)

    @property
    def T(self):
        graph = dict(self._graph)
        key = new_key("transpose")
        graph[key] = Task(
            lambda whole: _lib.f64(whole.T),
            (self._key,),
        )
        shape = self.shape[::-1]
        chunks = tuple((size,) for size in shape)
        return Array(graph, {tuple(0 for _ in shape): key}, shape, chunks)


def from_array(x, chunks="auto", name=None, lock=False, asarray=None, fancy=True,
               getitem=None, meta=None, inline_array=False):
    del lock, asarray, fancy, getitem, meta, inline_array
    source = np.asarray(x)
    shape = source.shape
    normal = _normalise_chunks(shape, chunks)
    slices = _slices(normal)
    graph = {}
    blocks = {}
    for index in _block_indices(normal):
        selection = tuple(slices[axis][part] for axis, part in enumerate(index))
        key = new_key(name or "from-array")
        graph[key] = Task(
            lambda source=source, selection=selection:
                _lib.f64(source[selection], name="array input")
        )
        blocks[index] = key
    return Array(graph, blocks, shape, normal, name)


def asarray(a, allow_unknown_chunksizes=False, dtype=None, order=None, like=None,
            inline_array=False, chunks="auto"):
    del allow_unknown_chunksizes, order, like, inline_array
    _require_float64_dtype(dtype)
    if isinstance(a, Array):
        return a
    return from_array(a, chunks=chunks)


def arange(*args, chunks="auto", like=None, dtype=None, **kwargs):
    del like
    _require_float64_dtype(dtype)
    return from_array(np.arange(*args, dtype=np.float64, **kwargs), chunks=chunks)


def ones(shape, chunks="auto", dtype=float, order="C", meta=None):
    del meta
    _require_float64_dtype(dtype)
    return from_array(np.ones(shape, dtype=np.float64, order=order), chunks=chunks)


def zeros(shape, chunks="auto", dtype=float, order="C", meta=None):
    del meta
    _require_float64_dtype(dtype)
    return from_array(np.zeros(shape, dtype=np.float64, order=order), chunks=chunks)


def _unary(a, opcode):
    a = asarray(a)
    graph = dict(a._graph)
    blocks = {}
    for index, dependency in a._blocks.items():
        key = new_key("unary")
        graph[key] = Task(
            lambda block, opcode=opcode: _lib.unary(block, opcode),
            (dependency,),
        )
        blocks[index] = key
    return Array(graph, blocks, a.shape, a.chunks)


def exp(a):
    return _unary(a, 2)


def log(a):
    return _unary(a, 3)


def sqrt(a):
    return _unary(a, 4)


def absolute(a):
    return _unary(a, 1)


def dot(a, b):
    left, right = asarray(a), asarray(b)
    if left.ndim == right.ndim == 1:
        if left.shape != right.shape:
            raise ValueError("shapes are not aligned")
        if left.chunks != right.chunks:
            right = right.rechunk(left.chunks)
        graph = merge_graphs(left._graph, right._graph)
        key = new_key("dot")
        dependencies = []
        for index in _block_indices(left.chunks):
            dependencies.extend((left._blocks[index], right._blocks[index]))

        def dot_blocks(*values):
            return builtins.sum(
                _lib.dot(values[i], values[i + 1])
                for i in range(0, len(values), 2)
            )

        graph[key] = Task(dot_blocks, tuple(dependencies))
        return Scalar(graph, key)
    return matmul(left, right)


def matmul(a, b, device="cpu"):
    left, right = asarray(a), asarray(b)
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("covered matmul requires two 2-D arrays")
    if left.shape[1] != right.shape[0]:
        raise ValueError("shapes are not aligned")
    if device == "gpu":
        graph = merge_graphs(left._graph, right._graph)
        key = new_key("matmul-gpu")
        graph[key] = Task(
            lambda left_value, right_value:
                _lib.matmul(left_value, right_value, device="gpu"),
            (left._key, right._key),
        )
        shape = (left.shape[0], right.shape[1])
        chunks = tuple((size,) for size in shape)
        return Array(graph, {(0, 0): key}, shape, chunks)
    if left.chunks[1] != right.chunks[0]:
        right = right.rechunk((left.chunks[1], right.chunks[1]))
    chunks = (left.chunks[0], right.chunks[1])
    graph = merge_graphs(left._graph, right._graph)
    blocks = {}
    for i in range(len(chunks[0])):
        for j in range(len(chunks[1])):
            dependencies = []
            for k in range(len(left.chunks[1])):
                dependencies.extend(
                    (left._blocks[(i, k)], right._blocks[(k, j)])
                )
            key = new_key("matmul")

            def multiply_parts(*values):
                result = _lib.matmul(values[0], values[1])
                for part in range(2, len(values), 2):
                    _lib.matmul_accumulate(
                        result, values[part], values[part + 1]
                    )
                return result

            graph[key] = Task(multiply_parts, tuple(dependencies))
            blocks[(i, j)] = key
    return Array(graph, blocks, (left.shape[0], right.shape[1]), chunks)


def concatenate(seq, axis=0, allow_unknown_chunksizes=False):
    del allow_unknown_chunksizes
    arrays = [asarray(value) for value in seq]
    if not arrays:
        raise ValueError("need at least one array to concatenate")
    axis %= arrays[0].ndim
    for position, value in enumerate(arrays[1:], start=1):
        if value.ndim != arrays[0].ndim:
            raise ValueError("all arrays must have the same number of dimensions")
        if any(
            value.shape[dimension] != arrays[0].shape[dimension]
            for dimension in range(value.ndim)
            if dimension != axis
        ):
            raise ValueError("array shapes must match except on the concatenation axis")
        target = tuple(
            arrays[0].chunks[dimension]
            if dimension != axis else value.chunks[dimension]
            for dimension in range(value.ndim)
        )
        if any(
            value.chunks[dimension] != arrays[0].chunks[dimension]
            for dimension in range(value.ndim)
            if dimension != axis
        ):
            arrays[position] = value.rechunk(target)
    graph = merge_graphs(*(value._graph for value in arrays))
    blocks = {}
    offset = 0
    for value in arrays:
        for index, key in value._blocks.items():
            output_index = list(index)
            output_index[axis] += offset
            blocks[tuple(output_index)] = key
        offset += len(value.chunks[axis])
    shape = list(arrays[0].shape)
    shape[axis] = sum(value.shape[axis] for value in arrays)
    chunks = list(arrays[0].chunks)
    chunks[axis] = tuple(
        size for value in arrays for size in value.chunks[axis]
    )
    return Array(graph, blocks, tuple(shape), tuple(chunks))
