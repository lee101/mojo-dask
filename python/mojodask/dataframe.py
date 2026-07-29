"""Partitioned pandas-compatible dataframes with Mojo aggregation kernels."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import _lib
from .core import Scalar, Task, execute, merge_graphs, new_key


def _concat(*partitions):
    if not partitions:
        return pd.DataFrame()
    return pd.concat(partitions, axis=0)


class _Frame:
    def __init__(self, graph, partitions, meta):
        self._graph = graph
        self._partitions = tuple(partitions)
        self._meta = meta.iloc[:0].copy()
        self.npartitions = len(self._partitions)
        self._key = new_key("frame")
        self._graph[self._key] = Task(_concat, self._partitions)

    @property
    def divisions(self):
        return (None,) * (self.npartitions + 1)

    def compute(self, scheduler=None, **kwargs):
        return execute(
            self._graph, (self._key,), scheduler, kwargs.get("num_workers")
        )[0]

    def persist(self, scheduler=None, **kwargs):
        values = execute(
            self._graph, self._partitions, scheduler, kwargs.get("num_workers")
        )
        graph = {}
        keys = []
        for value in values:
            key = new_key("persist-frame")
            graph[key] = Task(lambda value=value: value)
            keys.append(key)
        return type(self)(graph, keys, self._meta)

    def map_partitions(self, func, *args, meta=None, enforce_metadata=True,
                       transform_divisions=True, clear_divisions=False,
                       align_dataframes=False, parent_meta=None,
                       required_columns=None, **kwargs):
        del enforce_metadata, transform_divisions, clear_divisions
        del align_dataframes, parent_meta, required_columns
        graph = dict(self._graph)
        keys = []
        for dependency in self._partitions:
            key = new_key("map-partitions")
            graph[key] = Task(
                lambda partition, func=func, args=args, kwargs=kwargs:
                    func(partition, *args, **kwargs),
                (dependency,),
            )
            keys.append(key)
        if meta is None:
            meta = func(self._meta.copy(), *args, **kwargs)
        cls = Series if isinstance(meta, pd.Series) else DataFrame
        return cls(graph, keys, meta)


class Series(_Frame):
    @property
    def name(self):
        return self._meta.name

    @property
    def dtype(self):
        return self._meta.dtype

    def _reduce(self, operation, split_every=False, **kwargs):
        del split_every
        graph = dict(self._graph)
        local_keys = []
        for dependency in self._partitions:
            key = new_key("series-stats")

            def local(partition):
                values = np.asarray(partition.dropna(), dtype=np.float64)
                return _lib.stats(values)

            graph[key] = Task(local, (dependency,))
            local_keys.append(key)
        result_key = new_key(f"series-{operation}")

        def finish(*parts):
            count = sum(part[0] for part in parts)
            total = sum(part[1] for part in parts)
            if operation == "count":
                return int(count)
            if operation == "sum":
                return float(total)
            if operation == "mean":
                return float(total / count) if count else float("nan")
            if operation == "min":
                return float(min(
                    (part[4] for part in parts if part[0]),
                    default=float("nan"),
                ))
            if operation == "max":
                return float(max(
                    (part[5] for part in parts if part[0]),
                    default=float("nan"),
                ))
            ddof = kwargs.get("ddof", 1)
            mean = m2 = seen = 0.0
            for part in parts:
                n, _, part_mean, part_m2 = part[:4]
                if not n:
                    continue
                delta = part_mean - mean
                combined = seen + n
                m2 += part_m2 + delta * delta * seen * n / combined
                mean += delta * n / combined
                seen = combined
            value = m2 / (seen - ddof) if seen > ddof else float("nan")
            return math.sqrt(value) if operation == "std" else value

        graph[result_key] = Task(finish, tuple(local_keys))
        return Scalar(graph, result_key)

    def sum(self, axis=0, skipna=True, split_every=False, dtype=None,
            out=None, min_count=0, numeric_only=False):
        del axis, dtype, out, numeric_only
        if not skipna or min_count:
            raise NotImplementedError("Series.sum covers skipna=True and min_count=0")
        return self._reduce("sum", split_every)

    def mean(self, axis=0, skipna=True, numeric_only=False, split_every=False):
        del axis, numeric_only
        if not skipna:
            raise NotImplementedError("Series.mean covers skipna=True")
        return self._reduce("mean", split_every)

    def min(self, axis=0, skipna=True, split_every=False, numeric_only=False):
        del axis, numeric_only
        if not skipna:
            raise NotImplementedError("Series.min covers skipna=True")
        return self._reduce("min", split_every)

    def max(self, axis=0, skipna=True, split_every=False, numeric_only=False):
        del axis, numeric_only
        if not skipna:
            raise NotImplementedError("Series.max covers skipna=True")
        return self._reduce("max", split_every)

    def count(self, split_every=False):
        return self._reduce("count", split_every)

    def var(self, axis=0, skipna=True, ddof=1, numeric_only=False,
            split_every=False):
        del axis, numeric_only
        if not skipna:
            raise NotImplementedError("Series.var covers skipna=True")
        return self._reduce("var", split_every, ddof=ddof)

    def std(self, axis=0, skipna=True, ddof=1, numeric_only=False,
            split_every=False):
        del axis, numeric_only
        if not skipna:
            raise NotImplementedError("Series.std covers skipna=True")
        return self._reduce("std", split_every, ddof=ddof)


def _numeric_stats(partition, columns):
    result = {}
    for column in columns:
        values = np.asarray(partition[column].dropna(), dtype=np.float64)
        result[column] = _lib.stats(values)
    return result


def _merge_column_stats(parts, columns, operation, ddof):
    result = {}
    for column in columns:
        stats = [part[column] for part in parts]
        count = sum(part[0] for part in stats)
        total = sum(part[1] for part in stats)
        if operation == "count":
            result[column] = int(count)
        elif operation == "sum":
            result[column] = total
        elif operation == "mean":
            result[column] = total / count if count else np.nan
        elif operation == "min":
            result[column] = min(
                (part[4] for part in stats if part[0]), default=np.nan
            )
        elif operation == "max":
            result[column] = max(
                (part[5] for part in stats if part[0]), default=np.nan
            )
        else:
            mean = m2 = seen = 0.0
            for part in stats:
                n, _, part_mean, part_m2 = part[:4]
                if not n:
                    continue
                delta = part_mean - mean
                combined = seen + n
                m2 += part_m2 + delta * delta * seen * n / combined
                mean += delta * n / combined
                seen = combined
            variance = m2 / (seen - ddof) if seen > ddof else np.nan
            result[column] = math.sqrt(variance) if operation == "std" else variance
    return pd.Series(result)


class DataFrame(_Frame):
    @property
    def columns(self):
        return self._meta.columns

    @property
    def dtypes(self):
        return self._meta.dtypes

    def __getitem__(self, key):
        graph = dict(self._graph)
        keys = []
        for dependency in self._partitions:
            task_key = new_key("getitem-frame")
            graph[task_key] = Task(
                lambda partition, key=key: partition[key],
                (dependency,),
            )
            keys.append(task_key)
        meta = self._meta[key]
        cls = Series if isinstance(meta, pd.Series) else DataFrame
        return cls(graph, keys, meta)

    def _reduce(self, operation, axis=0, skipna=True, numeric_only=False,
                split_every=False, ddof=1):
        del split_every
        if not skipna:
            raise NotImplementedError(
                "dataframe reductions currently cover skipna=True"
            )
        if axis not in (0, "index"):
            return self.map_partitions(
                lambda frame: getattr(frame, operation)(
                    axis=axis, numeric_only=numeric_only
                )
            )
        columns = list(
            self._meta.select_dtypes(include=[np.number, "bool"]).columns
            if numeric_only or operation in ("mean", "var", "std")
            else self._meta.columns
        )
        if any(not pd.api.types.is_numeric_dtype(self._meta[column]) for column in columns):
            raise TypeError("covered dataframe reductions operate on numeric columns")
        graph = dict(self._graph)
        local_keys = []
        for dependency in self._partitions:
            key = new_key("dataframe-stats")
            graph[key] = Task(
                lambda partition, columns=columns:
                    _numeric_stats(partition, columns),
                (dependency,),
            )
            local_keys.append(key)
        result_key = new_key(f"dataframe-{operation}")
        graph[result_key] = Task(
            lambda *parts, columns=columns, operation=operation, ddof=ddof:
                _merge_column_stats(parts, columns, operation, ddof),
            tuple(local_keys),
        )
        return Scalar(graph, result_key)

    def sum(self, axis=0, skipna=True, numeric_only=False, split_every=False,
            min_count=0, **kwargs):
        del kwargs
        if min_count:
            raise NotImplementedError("DataFrame.sum covers min_count=0")
        return self._reduce("sum", axis, skipna, numeric_only, split_every)

    def mean(self, axis=0, skipna=True, numeric_only=False, split_every=False,
             **kwargs):
        del kwargs
        return self._reduce("mean", axis, skipna, numeric_only, split_every)

    def min(self, axis=0, skipna=True, numeric_only=False, split_every=False,
            **kwargs):
        del kwargs
        return self._reduce("min", axis, skipna, numeric_only, split_every)

    def max(self, axis=0, skipna=True, numeric_only=False, split_every=False,
            **kwargs):
        del kwargs
        return self._reduce("max", axis, skipna, numeric_only, split_every)

    def count(self, numeric_only=False, split_every=False):
        return self._reduce("count", 0, True, numeric_only, split_every)

    def var(self, axis=0, skipna=True, ddof=1, numeric_only=False,
            split_every=False, **kwargs):
        del kwargs
        return self._reduce("var", axis, skipna, numeric_only, split_every, ddof)

    def std(self, axis=0, skipna=True, ddof=1, numeric_only=False,
            split_every=False, **kwargs):
        del kwargs
        return self._reduce("std", axis, skipna, numeric_only, split_every, ddof)

    def groupby(self, by, group_keys=True, sort=None, observed=None,
                dropna=None, **kwargs):
        del group_keys, observed
        return DataFrameGroupBy(
            self, by, sort=True if sort is None else sort,
            dropna=True if dropna is None else dropna,
            as_index=kwargs.pop("as_index", True),
        )


def _local_groups(partition, by, columns, dropna):
    keys = partition[by]
    codes, uniques = pd.factorize(keys, sort=False, use_na_sentinel=dropna)
    result = {}
    for column in columns:
        result[column] = _lib.group_stats(
            codes, np.asarray(partition[column], dtype=np.float64), len(uniques)
        )
    return uniques, result


def _finish_groups(
    parts, by, columns, operation, sort, as_index, dtypes, scalar_selection
):
    groups = {}
    order = []
    for labels, column_stats in parts:
        for local_index, label in enumerate(labels):
            if label not in groups:
                groups[label] = {
                    column: [0, 0.0, 0.0, 0.0, float("inf"), -float("inf")]
                    for column in columns
                }
                order.append(label)
            for column in columns:
                count, total, mean, m2, minimum, maximum = (
                    stats[local_index] for stats in column_stats[column]
                )
                target = groups[label][column]
                old_count = target[0]
                combined = old_count + count
                if count:
                    delta = mean - target[2]
                    target[3] += m2 + delta * delta * old_count * count / combined
                    target[2] += delta * count / combined
                    target[4] = min(target[4], minimum)
                    target[5] = max(target[5], maximum)
                target[0] = combined
                target[1] += total
    if sort:
        present = [label for label in order if not pd.isna(label)]
        missing = [label for label in order if pd.isna(label)]
        labels = sorted(present) + missing
    else:
        labels = order
    rows = []
    for label in labels:
        row = {}
        for column in columns:
            count, total, mean, m2, minimum, maximum = groups[label][column]
            if operation == "count":
                row[column] = int(count)
            elif operation == "sum":
                row[column] = total
            elif operation == "mean":
                row[column] = mean if count else np.nan
            elif operation == "min":
                row[column] = minimum if count else np.nan
            elif operation == "max":
                row[column] = maximum if count else np.nan
            elif operation == "var":
                row[column] = m2 / (count - 1) if count > 1 else np.nan
            else:
                row[column] = math.sqrt(m2 / (count - 1)) if count > 1 else np.nan
        rows.append(row)
    label_dtype = parts[0][0].dtype if parts else None
    result = pd.DataFrame(
        rows, index=pd.Index(labels, dtype=label_dtype, name=by)
    )
    for column in columns:
        if operation == "count":
            result[column] = result[column].astype(np.int64)
        elif operation in ("sum", "min", "max") and pd.api.types.is_integer_dtype(
            dtypes[column]
        ):
            result[column] = result[column].astype(dtypes[column])
    if scalar_selection and as_index:
        return result[columns[0]]
    if not as_index:
        result = result.reset_index()
    return result


class DataFrameGroupBy:
    def __init__(self, frame, by, sort=True, dropna=True, as_index=True,
                 selection=None, scalar_selection=False):
        if not isinstance(by, str):
            raise NotImplementedError("covered groupby accepts one column name")
        self.frame = frame
        self.by = by
        self.sort = sort
        self.dropna = dropna
        self.as_index = as_index
        self.selection = selection
        self.scalar_selection = scalar_selection

    def __getitem__(self, key):
        selection = [key] if isinstance(key, str) else list(key)
        return DataFrameGroupBy(
            self.frame, self.by, self.sort, self.dropna, self.as_index, selection,
            isinstance(key, str),
        )

    def _aggregate(self, operation, split_every=None, split_out=None,
                   shuffle_method=None, numeric_only=False, **kwargs):
        del split_every, split_out, shuffle_method, kwargs
        columns = self.selection or [
            column for column in self.frame.columns
            if column != self.by
            and pd.api.types.is_numeric_dtype(self.frame._meta[column])
        ]
        if numeric_only:
            columns = [
                column for column in columns
                if pd.api.types.is_numeric_dtype(self.frame._meta[column])
            ]
        if any(
            not pd.api.types.is_numeric_dtype(self.frame._meta[column])
            for column in columns
        ):
            raise TypeError("covered groupby aggregations require numeric columns")
        graph = dict(self.frame._graph)
        local_keys = []
        for dependency in self.frame._partitions:
            key = new_key("group-partition")
            graph[key] = Task(
                lambda partition, by=self.by, columns=columns, dropna=self.dropna:
                    _local_groups(partition, by, columns, dropna),
                (dependency,),
            )
            local_keys.append(key)
        result_key = new_key(f"group-{operation}")
        dtypes = self.frame._meta.dtypes
        graph[result_key] = Task(
            lambda *parts, by=self.by, columns=columns, operation=operation,
                   sort=self.sort, as_index=self.as_index, dtypes=dtypes,
                   scalar_selection=self.scalar_selection:
                _finish_groups(
                    parts, by, columns, operation, sort, as_index, dtypes,
                    scalar_selection,
                ),
            tuple(local_keys),
        )
        return Scalar(graph, result_key)

    def sum(self, numeric_only=False, min_count=None, **kwargs):
        del min_count
        return self._aggregate("sum", numeric_only=numeric_only, **kwargs)

    def mean(self, numeric_only=False, **kwargs):
        return self._aggregate("mean", numeric_only=numeric_only, **kwargs)

    def count(self, **kwargs):
        return self._aggregate("count", **kwargs)

    def min(self, numeric_only=False, min_count=None, **kwargs):
        del min_count
        return self._aggregate("min", numeric_only=numeric_only, **kwargs)

    def max(self, numeric_only=False, min_count=None, **kwargs):
        del min_count
        return self._aggregate("max", numeric_only=numeric_only, **kwargs)

    def var(self, ddof=1, numeric_only=False, **kwargs):
        if ddof != 1:
            raise NotImplementedError("covered groupby variance uses ddof=1")
        return self._aggregate("var", numeric_only=numeric_only, **kwargs)

    def std(self, ddof=1, numeric_only=False, **kwargs):
        if ddof != 1:
            raise NotImplementedError("covered groupby standard deviation uses ddof=1")
        return self._aggregate("std", numeric_only=numeric_only, **kwargs)


def from_pandas(data, npartitions=None, sort=True, chunksize=None):
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("data must be a pandas DataFrame or Series")
    if isinstance(data, pd.DataFrame):
        data = data.copy()
        for column in data.columns:
            if pd.api.types.is_string_dtype(data[column].dtype):
                data[column] = data[column].astype("string")
    elif pd.api.types.is_string_dtype(data.dtype):
        data = data.astype("string")
    if sort and not data.index.is_monotonic_increasing:
        data = data.sort_index()
    if chunksize is not None:
        if chunksize <= 0:
            raise ValueError("chunksize must be positive")
        npartitions = max(1, math.ceil(len(data) / chunksize))
    npartitions = 1 if npartitions is None else int(npartitions)
    if npartitions <= 0:
        raise ValueError("npartitions must be positive")
    boundaries = np.linspace(0, len(data), npartitions + 1, dtype=int)
    graph = {}
    keys = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        key = new_key("from-pandas")
        graph[key] = Task(
            lambda data=data, start=int(start), stop=int(stop):
                data.iloc[start:stop].copy()
        )
        keys.append(key)
    cls = Series if isinstance(data, pd.Series) else DataFrame
    return cls(graph, keys, data.iloc[:0])
