# AnnData-RS Usage Guide

`anndata-rs` provides a high-performance, out-of-core implementation of the AnnData format in Rust. It supports both HDF5 (`.h5ad`) and Zarr V3 formats with native cloud storage integration.

## Table of Contents
1. [Backend Selection](#1-backend-selection)
2. [Opening and Creating Files](#2-opening-and-creating-files)
3. [Accessing Data](#3-accessing-data)
4. [Subsetting](#4-subsetting)
5. [Lean Extraction (Memory Management)](#5-lean-extraction-memory-management)
6. [Parallel Processing](#6-parallel-processing)

---

## 1. Backend Selection

The library uses a generic `Backend` trait. You must choose a backend depending on your file format:

- **HDF5**: Use `anndata_hdf5::H5` (for `.h5ad` files).
- **Zarr**: Use `anndata_zarr::Zarr` (for Zarr V3 stores).

---

## 2. Opening and Creating Files

### Opening an HDF5 file
```rust
use anndata::AnnData;
use anndata_hdf5::H5;

let adata = AnnData::<H5>::open(H5::open("data.h5ad")?)?;
println!("Rows: {}, Cols: {}", adata.n_obs(), adata.n_vars());
```

### Opening a Zarr Store (Local or Remote)
Zarr V3 supports local paths and cloud URIs (`s3://`, `http://`).
```rust
use anndata_zarr::Zarr;

// Local
let adata = AnnData::<Zarr>::open(Zarr::open("data.zarr")?)?;

// Remote S3
let adata = AnnData::<Zarr>::open(Zarr::open("s3://bucket/data.zarr")?)?;
```

---

## 3. Accessing Data

Data is loaded **lazily**. Accessing a field returns a proxy object, and data is only read when you call `.get()`.

```rust
// Access X matrix
let x: ArrayData = adata.x().get()?.unwrap();

// Access observation annotations (Dataframe)
let obs = adata.read_obs()?;

// Access named matrices in obsm
let pca = adata.obsm().get_item::<ArrayData>("X_pca")?.unwrap();
```

---

## 4. Subsetting

`anndata-rs` supports in-place subsetting and writing subsets to new files.

### In-place Subsetting
```rust
use anndata::data::SelectInfoElem;

// Subset to first 100 rows
let selection = [SelectInfoElem::from(0..100), SelectInfoElem::full()];
adata.subset(&selection)?; 
```

### Writing a Subset to a New File
```rust
adata.write_select::<H5, _, _>(&selection, "subset.h5ad")?;
```

---

## 5. Lean Extraction (Memory Management)

This is the most efficient way to load data into memory for use in other libraries (like `anndata-memory`) without duplicating buffers.

### Ownership Transfer (`take`)
The `take()` method reads data from disk OR moves it out of the internal cache, leaving the `AnnData` object empty for that field.

```rust
// Move X into your ownership. adata.x() will now be empty in memory.
let x: ArrayData = adata.take_x()?.unwrap();
```

### Consuming Collections (`drain`)
Use `drain()` to extract all elements from a collection (like `obsm` or `layers`) as an iterator of owned objects.

```rust
let obsm_data: HashMap<String, ArrayData> = adata.obsm().drain().collect();
// adata.obsm() is now empty.
```

### Cache Control
You can manually manage the in-memory cache to speed up repetitive access.
```rust
adata.x().enable_cache();
let x1 = adata.x().get()?; // Loads from disk into cache
let x2 = adata.x().get()?; // Returns a clone from cache
```

---

## 6. Parallel Processing

`AnnDataSet` allows you to lazily concatenate multiple files and process them in parallel using Rayon.

```rust
use anndata::AnnDataSet;

let dataset = AnnDataSet::<H5>::new(
    [("batch1", adata1), ("ann2", adata2)],
    "dataset.h5ad",
    "sample_col",
    false,
)?;

// Parallel read across multiple files
let x = dataset.x().get::<ArrayData>()?;
```

## Performance Tips
- **Monotonic Selections**: Subsetting with strictly increasing indices (e.g., `0..100` or `[1, 5, 10]`) uses a fast-path that avoids expensive sorting and extra allocations.
- **Sparse Indices**: The library uses `u64` for sparse index pointers, allowing support for datasets with more than 2 billion non-zero elements.
