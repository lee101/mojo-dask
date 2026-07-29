# mojo-dask

`mojo-dask` is a standalone Mojo implementation of Dask's compute-heavy blocked
array and dataframe core. It builds lazy task graphs in Python, schedules independent
blocks synchronously or on a thread pool, and runs numerical block kernels in a single
compiled Mojo shared library.

It is not a binding to Dask and does not require Dask at runtime. The public modules are
shaped so code can use the familiar aliases:

```python
import mojodask.array as da
import mojodask.dataframe as dd
```

Dask itself is a development dependency and the test suite checks the same graphs and
data against upstream Dask 2026.7.1.

## Covered subset

Arrays:

- `from_array`, `asarray`, `arange`, `ones`, `zeros`, and `concatenate`
- lazy blocked `+`, `-`, `*`, `/`, power, negation, and absolute value
- `exp`, `log`, and `sqrt`
- `sum`, `mean`, `var`, `std`, `min`, and `max`, including `axis` and `keepdims`
- blocked vector `dot` and 2-D `matmul`
- slicing, transpose, reshape, rechunk, `map_blocks`, `persist`, and `compute`

Dataframes:

- `from_pandas`, column selection, partitioning, `map_partitions`, and `persist`
- numeric Series/DataFrame `sum`, `mean`, `count`, `min`, `max`, `var`, and `std`
- single-column groupby with `sum`, `mean`, `count`, `min`, `max`, `var`, and `std`
- groupby selection and the `sort`, `dropna`, and `as_index` options

Scheduling:

- Dask-shaped `delayed(...)` and `compute(...)`
- `scheduler="threads"` and `scheduler="synchronous"`
- dependency caching within a computation and parallel execution of ready blocks

The array kernel boundary is C-contiguous `float64`. Real numeric inputs are converted
only when their values are representable; complex values, unsupported output dtypes,
and integers beyond float64's exact range are rejected instead of silently narrowed.
Elementwise broadcasting other than scalars, arbitrary NumPy dtypes,
advanced indexing, n-dimensional tensor contraction, rolling/window operations,
multi-key dataframe groupby, joins, categorical aggregation, Dask's distributed
scheduler, spilling, graph optimization, and storage readers are not covered. Axis
reductions are API-compatible but currently assemble the result before reducing;
whole-array reductions remain blockwise. This scope is deliberate rather than a claim
to reimplement all of Dask.

## Install and build

The repository pins the Mojo nightly used by the kernels.

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` compiles `src/kernels.mojo` with `mojo build --emit shared-lib` and
writes `dist/libmojo-dask.so`. The Python package can also find an externally packaged
library through `MOJODASK_LIB=/path/to/libmojo-dask.so`.

## Usage

This example is exercised by the same APIs used in the tests:

```python
import numpy as np
import pandas as pd
import mojodask.array as da
import mojodask.dataframe as dd

x = da.from_array(np.arange(24.0).reshape(6, 4), chunks=(2, 2))
y = ((x + 1) * x).mean().compute()
product = (x.T @ x).compute()

frame = pd.DataFrame({
    "region": ["east", "west", "east", "north"],
    "sales": [10.0, 7.0, 12.0, 5.0],
})
totals = (
    dd.from_pandas(frame, npartitions=2, sort=False)
    .groupby("region")
    .sum()
    .compute()
)

print(y)
print(product.shape)
print(totals)
```

Collections are lazy. Pass `scheduler="synchronous"` to `.compute()` for deterministic
single-threaded graph execution or leave the default thread scheduler to run independent
blocks concurrently. Matrix multiplication uses the CPU by default. The optional
`da.matmul(left, right, device="gpu")` path uses a compatible accelerator when available
and emits a runtime warning before falling back to CPU if GPU setup or execution fails.

## Benchmarks

Measured with `pixi run bench`, which takes a machine-wide `flock` before running.
Times are the best of three warm same-process runs and include graph scheduling, ctypes,
and result assembly. Both implementations see identical NumPy/pandas inputs, chunk
shapes, and correctness checks.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 logical CPUs.
Software: Python 3.13.14, NumPy 2.5.1, Dask 2026.7.1.

| kernel | mojo-dask | upstream Dask | Mojo speedup |
| --- | ---: | ---: | ---: |
| elementwise `(x + 1.5) * x`, 5M | 44.58 ms | 46.25 ms | 1.04x |
| mean + variance, 5M | 9.04 ms | 25.16 ms | 2.78x |
| dot, 5M | 10.58 ms | 25.56 ms | 2.42x |
| blocked matmul, 768x768 | 28.54 ms | 19.65 ms | 0.69x |
| blocked matmul GPU, 768x768 | 15.92 ms | 26.29 ms | 1.65x |
| dataframe mean, 2M x 3 | 35.57 ms | 51.86 ms | 1.46x |
| groupby mean, 2M / 100 groups | 40.57 ms | 129.74 ms | 3.20x |

The benchmark only attempts GPU work when `nvidia-smi` reports at least 4000 MiB free.
Otherwise it prints that the GPU row was skipped. Runtime device allocations are capped
below 2 GB and buffers are released when the Mojo device context exits.

Reproduce the table rather than copying it to another machine:

```bash
pixi run bench
```

## How it works

Every lazy collection owns a graph of small Python task records. Array tasks correspond
to an n-dimensional grid of chunks; dataframe tasks correspond to row partitions.
The scheduler finds the graph reachable from requested outputs, caches each completed
dependency once, and executes each ready level either directly or through a
`ThreadPoolExecutor`.

All compiled functions live in one Mojo compilation unit. Python owns allocation and
passes C-contiguous NumPy buffer addresses through `ctypes` as 64-bit integers. The
exported Mojo functions use C ABI effects, reconstruct
`UnsafePointer[..., AnyOrigin[mut=True]]` values from those addresses, and write into
caller-owned output buffers. Task closures and local call references keep every NumPy
owner alive until the synchronous foreign call returns. Empty buffers never cross the
boundary, and Python checks lengths, dimensions, dtypes, and non-null addresses before
Mojo can dereference them. There is no cross-language allocator.

Array blocks use row-major `float64` storage. Elementwise, sum, dot, and Welford
reduction loops use the host's native SIMD width with scalar remainder loops.
Scalar-then-binary elementwise pairs execute in one kernel and one output allocation.
Whole-array reductions at or above one million elements schedule independent block
statistics in parallel; smaller reductions stay serial. Matrix block products
accumulate directly into caller-owned output buffers instead of allocating intermediate
products.

Reductions compute mergeable Welford `(count, sum, mean, M2, min, max)` states.
Dataframe partitions factorize group keys in Python, aggregate numeric columns by
integer code in Mojo, then merge the same Welford states across partition boundaries.
That last merge is what makes variance and standard deviation correct when a group
spans many partitions. High-arithmetic-intensity matrix multiplication also has an
explicit `std.gpu` path; streaming elementwise and reduction kernels stay on CPU because
their arithmetic intensity is too low to repay device transfers.

## Verification

The pytest suite contains 95 tests. It asserts values, chunk layouts, NaN behavior,
degrees of freedom, index/order behavior, and result dtypes against real upstream Dask
and pandas rather than merely checking that calls complete.

The project is licensed under the [MIT License](LICENSE).
