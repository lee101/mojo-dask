"""Locked same-process benchmarks against upstream Dask."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import dask  # noqa: E402
import dask.array as upstream_array  # noqa: E402
import dask.dataframe as upstream_dataframe  # noqa: E402
import mojodask.array as mojo_array  # noqa: E402
import mojodask.dataframe as mojo_dataframe  # noqa: E402


def best_time(function, repetitions=3):
    function()
    best = float("inf")
    value = None
    for _ in range(repetitions):
        start = time.perf_counter()
        value = function()
        best = min(best, time.perf_counter() - start)
    return best, value


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf8") as source:
            for line in source:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_memory_free_mib():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return min(int(line.strip()) for line in result.stdout.splitlines())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def measure(name, mojo_function, dask_function, check):
    mojo_seconds, mojo_value = best_time(mojo_function)
    dask_seconds, dask_value = best_time(dask_function)
    check(mojo_value, dask_value)
    return name, mojo_seconds, dask_seconds


def main():
    rng = np.random.default_rng(2026)
    rows = []
    gpu_free_mib = gpu_memory_free_mib()

    n = 5_000_000
    x = rng.normal(size=n)
    mojo_x = mojo_array.from_array(x, chunks=1_000_000)
    dask_x = upstream_array.from_array(x, chunks=1_000_000)
    rows.append(measure(
        "elementwise `(x + 1.5) * x`, 5M",
        lambda: ((mojo_x + 1.5) * mojo_x).compute(),
        lambda: ((dask_x + 1.5) * dask_x).compute(),
        lambda a, b: np.testing.assert_allclose(a, b),
    ))
    rows.append(measure(
        "mean + variance, 5M",
        lambda: (mojo_x.mean().compute(), mojo_x.var().compute()),
        lambda: (dask_x.mean().compute(), dask_x.var().compute()),
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11),
    ))

    y = rng.normal(size=n)
    mojo_y = mojo_array.from_array(y, chunks=1_000_000)
    dask_y = upstream_array.from_array(y, chunks=1_000_000)
    rows.append(measure(
        "dot, 5M",
        lambda: mojo_array.dot(mojo_x, mojo_y).compute(),
        lambda: upstream_array.dot(dask_x, dask_y).compute(),
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11),
    ))

    side = 768
    left = rng.normal(size=(side, side))
    right = rng.normal(size=(side, side))
    mojo_left = mojo_array.from_array(left, chunks=(256, 256))
    mojo_right = mojo_array.from_array(right, chunks=(256, 256))
    dask_left = upstream_array.from_array(left, chunks=(256, 256))
    dask_right = upstream_array.from_array(right, chunks=(256, 256))
    rows.append(measure(
        "blocked matmul, 768x768",
        lambda: (mojo_left @ mojo_right).compute(),
        lambda: (dask_left @ dask_right).compute(),
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10),
    ))
    if gpu_free_mib is not None and gpu_free_mib >= 4000:
        rows.append(measure(
            "blocked matmul GPU, 768x768",
            lambda: mojo_array.matmul(
                mojo_left, mojo_right, device="gpu"
            ).compute(),
            lambda: (dask_left @ dask_right).compute(),
            lambda a, b: np.testing.assert_allclose(
                a, b, rtol=1e-10, atol=1e-10
            ),
        ))

    frame_n = 2_000_000
    frame = pd.DataFrame({
        "group": rng.integers(0, 100, size=frame_n),
        "x": rng.normal(size=frame_n),
        "y": rng.normal(size=frame_n),
        "z": rng.normal(size=frame_n),
    })
    mojo_frame = mojo_dataframe.from_pandas(frame, npartitions=8, sort=False)
    dask_frame = upstream_dataframe.from_pandas(frame, npartitions=8, sort=False)
    rows.append(measure(
        "dataframe mean, 2M x 3",
        lambda: mojo_frame[["x", "y", "z"]].mean().compute(),
        lambda: dask_frame[["x", "y", "z"]].mean().compute(),
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11),
    ))
    rows.append(measure(
        "groupby mean, 2M / 100 groups",
        lambda: mojo_frame.groupby("group").mean().compute(),
        lambda: dask_frame.groupby("group").mean().compute(),
        lambda a, b: np.testing.assert_allclose(
            a.sort_index(), b.sort_index(), rtol=1e-10
        ),
    ))

    print(f"Machine: {cpu_name()}, {os.cpu_count()} logical CPUs")
    print(
        f"Software: Python {platform.python_version()}, NumPy {np.__version__}, "
        f"Dask {dask.__version__}"
    )
    if gpu_free_mib is None:
        print("GPU benchmark skipped: no NVIDIA device was detected.")
    elif gpu_free_mib < 4000:
        print(
            f"GPU benchmark skipped: only {gpu_free_mib} MiB device memory free."
        )
    print()
    print("| kernel | mojo-dask | upstream Dask | Mojo speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_seconds, dask_seconds in rows:
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{dask_seconds * 1000:.2f} ms | {dask_seconds / mojo_seconds:.2f}x |"
        )


if __name__ == "__main__":
    main()
