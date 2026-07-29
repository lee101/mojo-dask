"""Small task graph and schedulers shared by arrays, dataframes, and delayed."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable

_ids = count()


def new_key(prefix: str) -> str:
    return f"{prefix}-{next(_ids)}"


@dataclass(frozen=True)
class Task:
    function: Callable[..., Any]
    dependencies: tuple[str, ...] = ()


def merge_graphs(*graphs: dict[str, Task]) -> dict[str, Task]:
    merged: dict[str, Task] = {}
    for graph in graphs:
        merged.update(graph)
    return merged


def _reachable(graph: dict[str, Task], outputs: tuple[str, ...]) -> set[str]:
    found: set[str] = set()
    stack = list(outputs)
    while stack:
        key = stack.pop()
        if key not in found:
            found.add(key)
            stack.extend(graph[key].dependencies)
    return found


def execute(
    graph: dict[str, Task],
    outputs: tuple[str, ...],
    scheduler: str | None = None,
    num_workers: int | None = None,
) -> list[Any]:
    scheduler = scheduler or "threads"
    required = _reachable(graph, outputs)
    values: dict[str, Any] = {}
    pending = set(required)

    if scheduler in ("sync", "synchronous", "single-threaded"):
        def resolve(key: str):
            if key not in values:
                task = graph[key]
                values[key] = task.function(*(resolve(dep) for dep in task.dependencies))
            return values[key]

        return [resolve(key) for key in outputs]
    if scheduler not in ("threads", "threading"):
        raise ValueError("scheduler must be 'threads' or 'synchronous'")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        while pending:
            ready = [
                key for key in pending
                if all(dep in values for dep in graph[key].dependencies)
            ]
            if not ready:
                raise RuntimeError("task graph contains a cycle")
            futures = {
                key: pool.submit(
                    graph[key].function,
                    *(values[dep] for dep in graph[key].dependencies),
                )
                for key in ready
            }
            for key, future in futures.items():
                values[key] = future.result()
                pending.remove(key)
    return [values[key] for key in outputs]


class Scalar:
    def __init__(self, graph: dict[str, Task], key: str):
        self._graph = graph
        self._key = key

    def compute(self, scheduler=None, **kwargs):
        return execute(
            self._graph, (self._key,), scheduler, kwargs.get("num_workers")
        )[0]

    def persist(self, **kwargs):
        value = self.compute(**kwargs)
        key = new_key("scalar")
        return Scalar({key: Task(lambda value=value: value)}, key)


class Delayed(Scalar):
    pass


def delayed(obj=None, name=None, pure=None, nout=None, traverse=True):
    del pure, nout, traverse
    if obj is None:
        return lambda fn: delayed(fn, name=name)
    if callable(obj):
        def wrapper(*args, **kwargs):
            collections = [
                value for value in (*args, *kwargs.values())
                if hasattr(value, "_graph") and hasattr(value, "_key")
            ]
            graph = merge_graphs(*(value._graph for value in collections))
            deps = tuple(value._key for value in collections)

            def invoke(*resolved):
                replacements = dict(zip(map(id, collections), resolved))
                call_args = [
                    replacements.get(id(value), value) for value in args
                ]
                call_kwargs = {
                    key: replacements.get(id(value), value)
                    for key, value in kwargs.items()
                }
                return obj(*call_args, **call_kwargs)

            key = new_key(name or getattr(obj, "__name__", "delayed"))
            graph[key] = Task(invoke, deps)
            return Delayed(graph, key)
        return wrapper
    key = new_key(name or "delayed")
    return Delayed({key: Task(lambda value=obj: value)}, key)


def compute(*args, traverse=True, optimize_graph=True, scheduler=None, **kwargs):
    del traverse, optimize_graph
    collections = [
        value for value in args
        if hasattr(value, "_graph") and hasattr(value, "_key")
    ]
    if not collections:
        return args
    graph = merge_graphs(*(value._graph for value in collections))
    values = execute(
        graph, tuple(value._key for value in collections), scheduler,
        kwargs.get("num_workers"),
    )
    replacements = iter(values)
    return tuple(
        next(replacements) if hasattr(value, "_key") else value
        for value in args
    )
