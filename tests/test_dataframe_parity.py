import numpy as np
import pandas as pd
import pytest

import dask.dataframe as dask_dataframe
import mojodask.dataframe as dd

rng = np.random.default_rng(7)


@pytest.fixture
def frame():
    n = 401
    result = pd.DataFrame(
        {
            "group": rng.choice(["north", "south", "east", "west"], n),
            "x": rng.normal(size=n),
            "y": rng.normal(loc=4, scale=2, size=n),
            "z": rng.integers(0, 20, size=n),
        }
    )
    result.loc[::17, "x"] = np.nan
    result.loc[::23, "y"] = np.nan
    return result


def test_from_pandas_partition_and_compute_parity(frame):
    ours = dd.from_pandas(frame, npartitions=7, sort=False)
    theirs = dask_dataframe.from_pandas(frame, npartitions=7, sort=False)
    assert ours.npartitions == theirs.npartitions
    pd.testing.assert_frame_equal(ours.compute(), theirs.compute())


def test_column_and_column_subset_parity(frame):
    ours = dd.from_pandas(frame, npartitions=7, sort=False)
    theirs = dask_dataframe.from_pandas(frame, npartitions=7, sort=False)
    pd.testing.assert_series_equal(ours["x"].compute(), theirs["x"].compute())
    pd.testing.assert_frame_equal(
        ours[["x", "z"]].compute(), theirs[["x", "z"]].compute()
    )


@pytest.mark.parametrize("name", ["sum", "mean", "min", "max", "count", "var", "std"])
def test_series_reduction_parity(frame, name):
    ours = dd.from_pandas(frame, npartitions=9, sort=False)["x"]
    theirs = dask_dataframe.from_pandas(frame, npartitions=9, sort=False)["x"]
    assert getattr(ours, name)().compute() == pytest.approx(
        getattr(theirs, name)().compute(), rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("name", ["sum", "mean", "min", "max", "count", "var", "std"])
def test_dataframe_reduction_parity(frame, name):
    columns = ["x", "y", "z"]
    ours = dd.from_pandas(frame[columns], npartitions=8, sort=False)
    theirs = dask_dataframe.from_pandas(frame[columns], npartitions=8, sort=False)
    pd.testing.assert_series_equal(
        getattr(ours, name)(numeric_only=True).compute(),
        getattr(theirs, name)(numeric_only=True).compute(),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("name", ["sum", "mean", "min", "max", "count", "var", "std"])
def test_groupby_parity_across_partitions(frame, name):
    ours = dd.from_pandas(frame, npartitions=11, sort=False)
    theirs = dask_dataframe.from_pandas(frame, npartitions=11, sort=False)
    got = getattr(ours.groupby("group"), name)().compute()
    expected = getattr(theirs.groupby("group"), name)().compute()
    pd.testing.assert_frame_equal(
        got[["x", "y", "z"]],
        expected[["x", "y", "z"]],
        check_exact=False,
        rtol=1e-11,
        atol=1e-12,
    )


def test_groupby_unsorted_order_matches_pandas_first_seen(frame):
    ours = dd.from_pandas(frame, npartitions=5, sort=False)
    got = ours.groupby("group", sort=False).sum().compute()
    normalized = frame.assign(group=frame.group.astype("string"))
    expected = normalized.groupby("group", sort=False).sum(numeric_only=True)
    pd.testing.assert_frame_equal(got, expected, check_exact=False)


def test_groupby_as_index_false(frame):
    ours = dd.from_pandas(frame, npartitions=6, sort=False)
    got = ours.groupby("group", as_index=False).mean().compute()
    normalized = frame.assign(group=frame.group.astype("string"))
    expected = normalized.groupby("group", as_index=False).mean(numeric_only=True)
    pd.testing.assert_frame_equal(got, expected, check_exact=False)


def test_map_partitions_parity(frame):
    ours = dd.from_pandas(frame, npartitions=7, sort=False)
    theirs = dask_dataframe.from_pandas(frame, npartitions=7, sort=False)
    fn = lambda part: part.assign(x=part.x * 3)
    pd.testing.assert_frame_equal(
        ours.map_partitions(fn).compute(), theirs.map_partitions(fn).compute()
    )


def test_map_partitions_does_not_swallow_metadata_errors(frame):
    ours = dd.from_pandas(frame, npartitions=2, sort=False)

    def broken(partition):
        raise RuntimeError("metadata failure")

    with pytest.raises(RuntimeError, match="metadata failure"):
        ours.map_partitions(broken)


def test_unsupported_reduction_options_fail_explicitly(frame):
    ours = dd.from_pandas(frame, npartitions=2, sort=False)
    with pytest.raises(NotImplementedError, match="skipna"):
        ours["x"].mean(skipna=False)
    with pytest.raises(NotImplementedError, match="min_count"):
        ours.sum(min_count=2)


def test_invalid_partition_sizes_are_rejected(frame):
    with pytest.raises(ValueError, match="npartitions"):
        dd.from_pandas(frame, npartitions=0)
    with pytest.raises(ValueError, match="chunksize"):
        dd.from_pandas(frame, chunksize=0)


def test_selected_groupby_series_parity(frame):
    ours = dd.from_pandas(frame, npartitions=7, sort=False)
    theirs = dask_dataframe.from_pandas(frame, npartitions=7, sort=False)
    pd.testing.assert_series_equal(
        ours.groupby("group")["x"].mean().compute(),
        theirs.groupby("group")["x"].mean().compute(),
        check_exact=False,
        rtol=1e-12,
    )


def test_groupby_dropna_false_parity(frame):
    source = frame.copy()
    source.loc[::13, "group"] = None
    ours = dd.from_pandas(source, npartitions=7, sort=False)
    theirs = dask_dataframe.from_pandas(source, npartitions=7, sort=False)
    pd.testing.assert_frame_equal(
        ours.groupby("group", dropna=False).mean().compute(),
        theirs.groupby("group", dropna=False).mean().compute(),
        check_exact=False,
        rtol=1e-12,
    )


def test_dataframe_persist(frame):
    source = dd.from_pandas(frame, npartitions=4, sort=False)
    persisted = source.map_partitions(lambda part: part[["x", "y"]]).persist()
    pd.testing.assert_frame_equal(persisted.compute(), frame[["x", "y"]])
