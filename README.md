# AnnData-RS

A high-performance, out-of-core implementation of the [AnnData](https://anndata.readthedocs.io/) format for Rust, with Python bindings via the `anndata_rs` package.

`anndata-rs` complements the original Python `anndata` package by focusing on lazy, backed access to datasets that are too large to fit in memory. Opening a file reads metadata only; arrays, sparse matrices, and tables are loaded when requested.

## AnnData model

AnnData stores an annotated matrix for single-cell and other high-dimensional data:

- `X`: the primary observations × variables matrix, dense or sparse CSR/CSC.
- `obs` / `var`: observation and variable annotations.
- `obsm` / `varm`: multi-dimensional annotations.
- `obsp` / `varp`: pairwise annotations.
- `layers`: alternative matrices with the same shape as `X`.
- `uns`: unstructured metadata.

`anndata-rs` also provides `AnnDataSet` for lazily combining multiple AnnData objects without materializing the full concatenated matrix in memory.

## Key features

- **Backed, lazy access**: fields are loaded on demand; opening a large file uses minimal memory.
- **HDF5 and Zarr backends**: Rust supports `.h5ad`/HDF5 and Zarr stores, including local and object-store-backed Zarr paths.
- **Cloud native Zarr**: read and write Zarr through `object_store` integrations such as S3 and HTTP.
- **Fast sparse matrices**: canonical sparse matrices use `sprs::CsMatI` with `u64` index pointers and preserved index dtypes.
- **Efficient full reads**: full sparse reads move backend buffers directly into `sprs` after validation, avoiding extra full-matrix copies.
- **Selection and streaming**: subset, split, concatenate, and chunk data without loading entire AnnData objects.

## Quick start: Rust

```rust
use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend};
use anndata::data::SelectInfoElem;
use anndata_hdf5::H5;

// Open a file lazily.
let adata = AnnData::<H5>::open(H5::open("data.h5ad")?)?;

// Load X into memory only when requested.
let x = adata.x().get::<ArrayData>()?;

// Load a row subset of X.
let subset = adata.x().slice::<ArrayData, _>(&[
    SelectInfoElem::from(0..100),
    SelectInfoElem::full(),
])?;
```

For detailed Rust examples on cloud storage, subsetting, and memory-efficient loading, see the [Usage Guide](USAGE.md).

## Quick start: Python

The user-facing Python package is `anndata_rs` and lives under `python/` in this repository. The `pyanndata/` crate is the internal PyO3 implementation used by that package.

```bash
pip install anndata_rs
```

```python
import anndata_rs as ad

# Backed open; data stays on disk until accessed.
adata = ad.read("data.h5ad", backed="r")

# Load X into memory.
x = adata.X[:]

# Convert selected fields to an in-memory Python anndata.AnnData.
mem = adata.to_memory(partial=["X", "obs", "var"])
```

Current Python bindings are centered on backed HDF5/`.h5ad` workflows. Use the Rust API directly for the full set of Rust backend features, including object-store Zarr workflows.

## Installation

### Rust

Add the crates you need to `Cargo.toml`:

```toml
[dependencies]
anndata = "0.7"
anndata-hdf5 = "0.6" # HDF5/.h5ad support
anndata-zarr = "0.2" # Zarr support
```

### Python from source

```bash
git clone https://github.com/kaizhang/anndata-rs.git
cd anndata-rs/python
pip install .
```

## Documentation

- [Usage Guide](USAGE.md)
- [Python docs](python/docs/index.rst)
- [API Reference](https://docs.rs/anndata)
- [AnnData specification](https://anndata.readthedocs.io/)

## License

Licensed under the Apache License, Version 2.0.
