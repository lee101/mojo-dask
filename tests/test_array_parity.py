import numpy as np
import pytest

import dask.array as dask_array
import mojodask.array as da
from mojodask import _lib
from mojodask import compute, delayed

rng = np.random.default_rng(42)


@pytest.fixture
def matrix():
    return rng.normal(size=(37, 19))


def test_from_array_matches_dask_chunks_and_values(matrix):
    ours = da.from_array(matrix, chunks=(8, 6))
    theirs = dask_array.from_array(matrix, chunks=(8, 6))
    assert ours.chunks == theirs.chunks
    np.testing.assert_array_equal(ours.compute(), theirs.compute())


def test_constructors_and_source_lifetime():
    source = np.arange(12.0).reshape(3, 4)[:, ::2]
    value = da.asarray(source, chunks=(2, 1))
    del source
    np.testing.assert_array_equal(value.compute(), np.arange(12.0).reshape(3, 4)[:, ::2])
    np.testing.assert_array_equal(da.ones((2, 3), chunks=2).compute(), np.ones((2, 3)))
    np.testing.assert_array_equal(da.zeros((2, 3), chunks=2).compute(), np.zeros((2, 3)))


def test_float64_contract_rejects_silent_narrowing():
    with pytest.raises(OverflowError, match="float64"):
        da.from_array(np.array([2**53 + 1], dtype=np.int64), chunks=1).compute()
    with pytest.raises(OverflowError, match="float64"):
        (da.ones(1, chunks=1) + (2**53 + 1)).compute()
    with pytest.raises(NotImplementedError, match="float64"):
        da.zeros(3, dtype=np.float32)


def test_ffi_rejects_mismatched_lengths_and_shapes():
    with pytest.raises(ValueError, match="same number"):
        _lib.binary(np.ones(8), np.ones(7), 0)
    with pytest.raises(ValueError, match="same number"):
        _lib.dot(np.ones(8), np.ones(7))
    with pytest.raises(ValueError, match="inner dimensions"):
        _lib.matmul(np.ones((2, 3)), np.ones((2, 4)))
    with pytest.raises(ValueError, match="equal-length"):
        _lib.group_stats(np.array([0, 1]), np.array([1.0]), 2)


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: x + 2.5,
        lambda x: 2.5 + x,
        lambda x: x - 2.5,
        lambda x: 2.5 - x,
        lambda x: x * 2.5,
        lambda x: x / 2.5,
        lambda x: 2.5 / x,
        lambda x: x ** 2,
        lambda x: -x,
        lambda x: abs(x),
    ],
)
def test_elementwise_scalar_parity(matrix, operation):
    positive = matrix + 3.0
    ours = operation(da.from_array(positive, chunks=(9, 5))).compute()
    theirs = operation(dask_array.from_array(positive, chunks=(9, 5))).compute()
    np.testing.assert_allclose(ours, theirs, rtol=1e-13, atol=1e-13)


def test_elementwise_array_parity(matrix):
    other = rng.normal(size=matrix.shape)
    ours_a = da.from_array(matrix, chunks=(7, 4))
    ours_b = da.from_array(other, chunks=(7, 4))
    dask_a = dask_array.from_array(matrix, chunks=(7, 4))
    dask_b = dask_array.from_array(other, chunks=(7, 4))
    np.testing.assert_allclose(
        ((ours_a + ours_b) * (ours_a - ours_b)).compute(),
        ((dask_a + dask_b) * (dask_a - dask_b)).compute(),
    )


@pytest.mark.parametrize("size", [1, 3, 4, 5, 15, 16, 17])
def test_fused_elementwise_simd_tails(size):
    values = rng.normal(size=size)
    ours = da.from_array(values, chunks=size)
    theirs = dask_array.from_array(values, chunks=size)
    np.testing.assert_allclose(
        ((ours + 1.5) * ours).compute(),
        ((theirs + 1.5) * theirs).compute(),
        rtol=1e-13,
        atol=1e-13,
    )


def test_large_fused_elementwise_parallel_writes_and_tails():
    values = rng.normal(size=1_000_003)
    ours = da.from_array(values, chunks=200_001)
    result = (ours + 1.5) * ours
    assert any(key.startswith("fused-write-") for key in result._graph)
    np.testing.assert_allclose(
        result.compute(),
        (values + 1.5) * values,
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize("name", ["sum", "mean", "var", "std", "min", "max"])
def test_whole_array_reduction_parity(matrix, name):
    ours = da.from_array(matrix, chunks=(7, 5))
    theirs = dask_array.from_array(matrix, chunks=(7, 5))
    assert getattr(ours, name)().compute() == pytest.approx(
        getattr(theirs, name)().compute(), rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("size", [1, 3, 4, 5, 15, 16, 17])
def test_reduction_simd_tails(size):
    values = rng.normal(size=size)
    chunks = max(1, size - 1)
    ours = da.from_array(values, chunks=chunks)
    theirs = dask_array.from_array(values, chunks=chunks)
    for name in ("sum", "mean", "var", "std", "min", "max"):
        assert getattr(ours, name)().compute() == pytest.approx(
            getattr(theirs, name)().compute(), rel=1e-12, abs=1e-12
        )


def test_parallel_reduction_threshold():
    small_values = np.arange(999.0)
    small = da.from_array(small_values, chunks=333)
    small_mean = small.mean()
    assert not any("-part-" in key for key in small_mean._graph)
    assert small_mean.compute() == pytest.approx(np.mean(small_values))

    large_values = rng.normal(size=1_000_003)
    large = da.from_array(large_values, chunks=500_001)
    large_variance = large.var()
    assert any("var-part-" in key for key in large_variance._graph)
    assert large_variance.compute() == pytest.approx(
        np.var(large_values), rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("name", ["sum", "mean", "var", "std", "min", "max"])
@pytest.mark.parametrize("axis", [0, 1])
def test_axis_reduction_parity(matrix, name, axis):
    ours = da.from_array(matrix, chunks=(7, 5))
    theirs = dask_array.from_array(matrix, chunks=(7, 5))
    ours_reduced = getattr(ours, name)(axis=axis)
    theirs_reduced = getattr(theirs, name)(axis=axis)
    np.testing.assert_allclose(
        ours_reduced.compute(),
        theirs_reduced.compute(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert ours_reduced.shape == theirs_reduced.shape


def test_keepdims_and_invalid_axes(matrix):
    ours = da.from_array(matrix, chunks=(7, 5))
    theirs = dask_array.from_array(matrix, chunks=(7, 5))
    np.testing.assert_allclose(
        ours.mean(keepdims=True).compute(),
        theirs.mean(keepdims=True).compute(),
    )
    assert ours.mean(keepdims=True).shape == (1, 1)
    with pytest.raises(np.exceptions.AxisError):
        ours.sum(axis=2)
    with pytest.raises(ValueError, match="duplicate"):
        ours.sum(axis=(0, 0))


def test_nondefault_reduction_ddof(matrix):
    ours = da.from_array(matrix, chunks=(7, 5))
    theirs = dask_array.from_array(matrix, chunks=(7, 5))
    assert ours.var(ddof=1).compute() == pytest.approx(
        theirs.var(ddof=1).compute(), rel=1e-12, abs=1e-12
    )


def test_nan_propagation_matches_dask():
    values = np.array([1.0, 2.0, np.nan, 4.0])
    ours = da.from_array(values, chunks=2)
    theirs = dask_array.from_array(values, chunks=2)
    for name in ("sum", "mean", "var", "std", "min", "max"):
        assert np.isnan(getattr(ours, name)().compute())
        assert np.isnan(getattr(theirs, name)().compute())


def test_math_functions_match_dask(matrix):
    positive = np.abs(matrix) + 0.2
    ours = da.sqrt(da.exp(da.log(da.from_array(positive, chunks=(8, 5)))))
    theirs = dask_array.sqrt(
        dask_array.exp(dask_array.log(dask_array.from_array(positive, chunks=(8, 5))))
    )
    np.testing.assert_allclose(ours.compute(), theirs.compute(), rtol=1e-9, atol=1e-12)


def test_vector_dot_matches_dask():
    a, b = rng.normal(size=10_003), rng.normal(size=10_003)
    ours = da.dot(da.from_array(a, chunks=777), da.from_array(b, chunks=777))
    theirs = dask_array.dot(
        dask_array.from_array(a, chunks=777),
        dask_array.from_array(b, chunks=777),
    )
    assert ours.compute() == pytest.approx(theirs.compute(), rel=1e-12)


def test_blocked_matmul_matches_dask():
    a = rng.normal(size=(43, 31))
    b = rng.normal(size=(31, 27))
    ours = da.from_array(a, chunks=(11, 8)) @ da.from_array(b, chunks=(8, 7))
    theirs = dask_array.from_array(a, chunks=(11, 8)) @ dask_array.from_array(
        b, chunks=(8, 7)
    )
    assert ours.chunks == theirs.chunks
    np.testing.assert_allclose(ours.compute(), theirs.compute(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("size", [1, 3, 4, 5, 15, 16, 17])
def test_inplace_add_simd_tails(size):
    left = rng.normal(size=size)
    right = rng.normal(size=size)
    expected = left + right
    _lib.add_inplace(left, right)
    np.testing.assert_allclose(left, expected, rtol=1e-13, atol=1e-13)


def test_matmul_parallel_threshold_and_simd_tails():
    small_a = rng.normal(size=(99, 200))
    shared_b = rng.normal(size=(200, 200))
    small = da.from_array(small_a, chunks=(50, 100)) @ da.from_array(
        shared_b, chunks=(100, 100)
    )
    assert any(key.startswith("matmul-serial-") for key in small._graph)
    assert not any(key.startswith("matmul-product-") for key in small._graph)
    np.testing.assert_allclose(
        small.compute(), small_a @ shared_b, rtol=1e-12, atol=1e-12
    )

    large_a = rng.normal(size=(100, 200))
    large = da.from_array(large_a, chunks=(50, 100)) @ da.from_array(
        shared_b, chunks=(100, 100)
    )
    assert any(key.startswith("matmul-product-") for key in large._graph)
    np.testing.assert_allclose(
        large.compute(), large_a @ shared_b, rtol=1e-12, atol=1e-12
    )


def test_gpu_matmul_or_cpu_fallback_matches_dask():
    a = rng.normal(size=(17, 13))
    b = rng.normal(size=(13, 11))
    ours = da.matmul(
        da.from_array(a, chunks=(6, 5)),
        da.from_array(b, chunks=(5, 4)),
        device="gpu",
    )
    theirs = dask_array.from_array(a, chunks=(6, 5)) @ dask_array.from_array(
        b, chunks=(5, 4)
    )
    np.testing.assert_allclose(
        ours.compute(), theirs.compute(), rtol=1e-12, atol=1e-12
    )


def test_gpu_low_memory_falls_back_silently(monkeypatch):
    monkeypatch.setattr(_lib, "_gpu_memory_cache", None)
    monkeypatch.setattr(_lib, "_gpu_memory_free_mib", lambda: 3999)
    library = _lib.lib()

    def unexpected_gpu_call(*args):
        raise AssertionError("GPU kernel must not run below the memory threshold")

    monkeypatch.setattr(library, "md_matmul_gpu", unexpected_gpu_call)
    left = rng.normal(size=(9, 7))
    right = rng.normal(size=(7, 5))
    np.testing.assert_allclose(
        _lib.matmul(left, right, device="gpu"),
        left @ right,
        rtol=1e-12,
        atol=1e-12,
    )


def test_invalid_matmul_device_rejected():
    values = da.from_array(np.eye(2), chunks=2)
    with pytest.raises(ValueError, match="device"):
        da.matmul(values, values, device="tpu")


def test_rechunk_reshape_transpose_and_slice(matrix):
    ours = da.from_array(matrix, chunks=(8, 5)).rechunk((13, 7))
    theirs = dask_array.from_array(matrix, chunks=(8, 5)).rechunk((13, 7))
    assert ours.chunks == theirs.chunks
    np.testing.assert_allclose(ours.T[2:8, 3:20].compute(), theirs.T[2:8, 3:20].compute())
    np.testing.assert_allclose(
        ours.reshape(703).compute(), theirs.reshape(703).compute()
    )


def test_map_blocks_matches_dask(matrix):
    fn = lambda block: block * block + 1
    ours = da.from_array(matrix, chunks=(8, 5)).map_blocks(fn)
    theirs = dask_array.from_array(matrix, chunks=(8, 5)).map_blocks(fn)
    np.testing.assert_allclose(ours.compute(), theirs.compute())


def test_lazy_concatenate_matches_dask():
    values = [np.arange(12.0).reshape(3, 4), np.arange(8.0).reshape(2, 4)]
    ours = da.concatenate([da.from_array(value, chunks=(2, 2)) for value in values])
    theirs = dask_array.concatenate(
        [dask_array.from_array(value, chunks=(2, 2)) for value in values]
    )
    assert ours.chunks == theirs.chunks
    np.testing.assert_array_equal(ours.compute(), theirs.compute())


def test_persist_keeps_results_and_chunks(matrix):
    source = da.from_array(matrix, chunks=(8, 5))
    persisted = (source * 3 + 1).persist(scheduler="synchronous")
    assert persisted.chunks == source.chunks
    np.testing.assert_allclose(persisted.compute(), matrix * 3 + 1)


def test_delayed_and_compute_match_dask_style():
    @delayed
    def total(a, b):
        return a + b

    one = delayed(10)
    result = total(one, 5)
    array = da.arange(12, chunks=5) ** 2
    got_result, got_array = compute(result, array, scheduler="threads")
    assert got_result == 15
    np.testing.assert_array_equal(got_array, np.arange(12) ** 2)


def test_shared_dependencies_are_computed_once():
    calls = []

    @delayed
    def source():
        calls.append(1)
        return 3

    @delayed
    def add(a, b):
        return a + b

    shared = source()
    assert compute(add(shared, shared), scheduler="threads") == (6,)
    assert calls == [1]


def test_synchronous_and_threaded_schedulers_agree(matrix):
    value = (da.from_array(matrix, chunks=(4, 3)) + 1) ** 2
    np.testing.assert_array_equal(
        value.compute(scheduler="synchronous"),
        value.compute(scheduler="threads", num_workers=4),
    )
