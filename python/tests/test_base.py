from anndata_rs import AnnData, AnnDataSet, read

import pytest
import math
import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import uuid
from scipy.sparse import csr_matrix, csc_matrix
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *

BACKENDS = ["hdf5"]


def h5ad(dir=Path("./")):
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))


@pytest.mark.parametrize("backend", BACKENDS)
@given(
    x=arrays(
        integer_dtypes(endianness="=")
        | floating_dtypes(endianness="=", sizes=(32, 64))
        | unsigned_integer_dtypes(endianness="="),
        array_shapes(min_dims=2, max_dims=2, min_side=0, max_side=5),
    )
)
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_basic(x, tmp_path, backend):
    adata = AnnData(filename=h5ad(tmp_path), backend=backend)

    assert adata.obs is None
    assert adata.var is None

    adata.X = x
    adata.obs_names = [str(i) for i in range(x.shape[0])]
    np.testing.assert_array_equal(x, adata.X[:])
    np.testing.assert_array_equal(x, adata.X[...])

    adata.X = csr_matrix(x)
    np.testing.assert_array_equal(x, adata.X[:].todense())

    adata.X = csc_matrix(x)
    np.testing.assert_array_equal(x, adata.X[:].todense())

    adata.layers["X1"] = x
    adata.layers["X2"] = csr_matrix(x)
    adata.layers["X3"] = csc_matrix(x)

    adata.uns["array"] = np.array([1, 2, 3, 4, 5])
    adata.uns["array"] = np.array(["one", "two", "three", "four", "five"])
    adata.uns["array"] = np.array(
        ["one", "two", "three", "four", "five"], dtype="object"
    )

    # Dataframe
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["one", "two", "three", "four", "five"],
        }
    )
    adata.uns["df"] = df
    assert df.equals(adata.uns["df"])
    df = pl.DataFrame({"a": [], "b": []}, schema=[("a", pl.Float32), ("b", pl.Float32)])
    adata.uns["df"] = df
    assert df.equals(adata.uns["df"])

    adata.uns["df"] = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["one", "two", "three"],
        }
    )
    output = h5ad(tmp_path)
    adata.write(output, backend=backend)
    read(output, backend=backend).close()
    if backend == "hdf5":
        ad.read_h5ad(output).write(output, compression="gzip")
        read(output, backend="hdf5").close()


@pytest.mark.parametrize("backend", BACKENDS)
@given(
    x=arrays(
        integer_dtypes(endianness="=")
        | floating_dtypes(endianness="=", sizes=(32, 64))
        | unsigned_integer_dtypes(endianness="="),
        array_shapes(min_dims=2, max_dims=2, min_side=0, max_side=5),
    )
)
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_assign_arrays(x, tmp_path, backend):
    adata = AnnData(filename=h5ad(tmp_path), backend=backend)
    adata.uns["x"] = x
    x_ = adata.uns["x"]
    np.testing.assert_array_equal(x_, x)


@pytest.mark.parametrize("backend", BACKENDS)
@given(x=st.floats())
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_assign_floats(x, tmp_path, backend):
    adata = AnnData(filename=h5ad(tmp_path), backend=backend)
    adata.uns["x"] = x
    x_ = adata.uns["x"]
    assert x_ == x or (math.isnan(x) and math.isnan(x_))


@pytest.mark.parametrize("backend", BACKENDS)
def test_creation(tmp_path, backend):
    adata = AnnData(filename=h5ad(tmp_path), backend=backend)
    assert adata.n_obs == 0
    assert adata.n_vars == 0

    adata.obsm = dict(X_pca=np.array([[1, 2], [3, 4]]))
    assert adata.n_obs == 2
    assert adata.obsm["X_pca"].shape == (2, 2)

    adata.varm = dict(X_pca=np.array([[1, 2, 3], [3, 4, 5]]))
    assert adata.n_vars == 2

    adata.uns["df"] = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    adata.uns["df"] = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])

    adata.obs = pl.DataFrame({"a": ["1", "2"], "b": [3, 4]})
    adata.obs["c"] = np.array([1, 2])
    obs_names = list(adata.obs["a"])
    adata.obs_names = obs_names
    adata.obs = pl.DataFrame()
    assert adata.obs_names == obs_names

    adata.var = pl.DataFrame({"a": [1, 2], "b": ["3", "4"]})
    var_names = list(adata.var["b"])
    adata.var_names = var_names
    adata.var = pl.DataFrame()
    assert adata.var_names == var_names
    assert adata.to_memory().var_names.to_list() == var_names


@pytest.mark.parametrize("backend", BACKENDS)
def test_to_memory(tmp_path, backend):
    adata = AnnData(
        X=np.array([[1, 2, 3], [4, 5, 6]]), filename=h5ad(tmp_path), backend=backend
    )
    adata.obs_names = ["a", "b"]
    adata.var_names = ["a", "b", "c"]

    adata = adata.to_memory()
    assert adata.n_obs == 2
    assert adata.n_vars == 3
    assert adata.obs_names.tolist() == ["a", "b"]
    assert adata.var_names.tolist() == ["a", "b", "c"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_sparse_roundtrip(tmp_path, backend):
    x = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.int64)
    adata = AnnData(filename=h5ad(tmp_path), backend=backend)

    adata.X = csr_matrix(x)
    csr = adata.X[:]
    assert csr.getformat() == "csr"
    assert csr.dtype == x.dtype
    np.testing.assert_array_equal(csr.toarray(), x)

    adata.X = csc_matrix(x)
    csc = adata.X[:]
    assert csc.getformat() == "csc"
    assert csc.dtype == x.dtype
    np.testing.assert_array_equal(csc.toarray(), x)


@pytest.mark.parametrize("backend", BACKENDS)
def test_resize(tmp_path, backend):
    adata = AnnData(filename=h5ad(tmp_path), backend=backend)
    adata.obsm = dict(X_pca=np.array([[1, 2], [3, 4]]))
    adata.var_names = ["a", "b", "c", "d"]
    adata.X = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])


def test_nullable(tmp_path):
    file = h5ad(tmp_path)
    adata = ad.AnnData(X=np.array([[float("nan"), float("inf")]]))
    adata.write(file)

    adata = read(file)
    assert math.isnan(adata.X[:][0, 0])
    assert math.isinf(adata.X[:][0, 1])

    adata.uns["df"] = pd.DataFrame(
        {"test": pd.Series(["a", "b", np.nan, "a"], dtype="category")}
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_type(tmp_path, backend):
    adata = AnnData(
        filename=h5ad(tmp_path), X=np.array([[1, 2], [3, 4]]), backend=backend
    )

    x = "test"
    adata.uns["str"] = x
    assert adata.uns["str"] == x

    x = {
        "a": 1,
        "b": 2.0,
        "c": {"1": 2, "2": 5},
        "d": "test",
    }
    adata.uns["dict"] = x
    assert adata.uns["dict"] == x


@pytest.mark.parametrize("backend", BACKENDS)
@given(
    x1=arrays(np.int64, (7, 13)),
    x2=arrays(np.int64, (9, 13)),
    x3=arrays(np.int64, (3, 13)),
)
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_create_anndataset(x1, x2, x3, tmp_path, backend):
    # empty dataset
    adata1 = AnnData(filename=h5ad(tmp_path), backend=backend)
    adata2 = AnnData(filename=h5ad(tmp_path), backend=backend)
    adata3 = AnnData(filename=h5ad(tmp_path), backend=backend)
    dataset = AnnDataSet(
        adatas=[("1", adata1), ("2", adata2), ("3", adata3)],
        filename=h5ad(tmp_path),
        add_key="batch",
        backend=backend,
    )
    assert dataset.n_obs == 0
    assert dataset.n_vars == 0

    dataset.to_adata()

    # dense array
    adata1 = AnnData(X=x1, filename=h5ad(tmp_path), backend=backend)
    adata2 = AnnData(X=x2, filename=h5ad(tmp_path), backend=backend)
    adata3 = AnnData(X=x3, filename=h5ad(tmp_path), backend=backend)
    merged = np.concatenate([x1, x2, x3], axis=0)
    dataset = AnnDataSet(
        adatas=[("1", adata1), ("2", adata2), ("3", adata3)],
        filename=h5ad(tmp_path),
        add_key="batch",
        backend=backend,
    )
    np.testing.assert_array_equal(merged, dataset.X[:])
    dataset.obs["obs"] = [1] * (x1.shape[0] + x2.shape[0] + x3.shape[0])

    # sparse array
    adata1 = AnnData(X=csr_matrix(x1), filename=h5ad(tmp_path), backend=backend)
    adata2 = AnnData(X=csr_matrix(x2), filename=h5ad(tmp_path), backend=backend)
    adata3 = AnnData(X=csr_matrix(x3), filename=h5ad(tmp_path), backend=backend)
    dataset = AnnDataSet(
        adatas=[("1", adata1), ("2", adata2), ("3", adata3)],
        filename=h5ad(tmp_path),
        add_key="batch",
        backend=backend,
    )
    np.testing.assert_array_equal(merged, dataset.X[:].todense())

    # indexing
    x = dataset.X[:]
    np.testing.assert_array_equal(
        x[:, [1, 2, 3]].todense(), dataset.X[:, [1, 2, 3]].todense()
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_noncanonical_csr(tmp_path, backend):
    def assert_csr_equal(a, b):
        np.testing.assert_array_equal(a.shape, b.shape)
        np.testing.assert_array_equal(a.data, b.data)
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_array_equal(a.indptr, b.indptr)

    csr = csr_matrix(
        ([1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 2, 3, 1, 3], [0, 1, 4, 5, 6, 7]),
        (5, 4),
    )
    assert not csr.has_canonical_format

    file = h5ad(tmp_path)
    adata = AnnData(filename=file, backend=backend)
    adata.X = csr
    adata.obs = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert_csr_equal(csr, adata.X[:])
    adata.close()

    if backend == "hdf5":
        adata = ad.read_h5ad(file)
        assert_csr_equal(csr, adata.X)
        file = h5ad(tmp_path)
        adata.write(file)
        adata = read(file, backed=None, backend="hdf5")
        assert_csr_equal(csr, adata.X)
