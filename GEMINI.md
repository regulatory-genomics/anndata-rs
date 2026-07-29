# anndata-rs

## Project Overview

`anndata-rs` is a high-performance, out-of-core implementation of the AnnData format written in Rust, with Python bindings available via `pyanndata` / `anndata_rs`. It is designed for datasets that are too large to fit in memory, providing efficient partial I/O and cloud-native storage.

Key features include:
*   **Multi-Backend Support:** Native support for HDF5 (`.h5ad`) and Zarr V3.
*   **Cloud Native:** Read/write Zarr data directly from S3, HTTP, and Azure Blob Storage.
*   **Zero-Memory Loading:** Lazy loading of fields.
*   **Lean Extraction:** Special `take()` and `drain()` semantics to move data directly from disk into ownership without memory duplication.
*   **High-Performance Sparse Matrices:** Support for datasets with billions of non-zero elements using `u64` index pointers.
*   **Parallel I/O:** Multi-threaded operations via `AnnDataSet` and Rayon.

## Architecture & Directory Structure

This is a Cargo workspace containing the following main crates:

*   **`anndata/`**: The core Rust library providing the `AnnData` structure and logic.
*   **`anndata-hdf5/`**: The HDF5 backend implementation (`anndata_hdf5::H5`).
*   **`anndata-zarr/`**: The Zarr V3 backend implementation (`anndata_zarr::Zarr`).
*   **`pyanndata/`**: Rust crate containing the PyO3 bindings for Python.
*   **`python/`**: The Python package (`anndata_rs`) structure, including `pyproject.toml` (uses `maturin` as the build backend) and Python tests.
*   **`anndata-test-utils/`**: Utilities for testing across the workspace.

## Building and Running

### Rust
Standard Cargo commands apply from the workspace root:
*   `cargo build` / `cargo build --release`
*   `cargo test`

### Python
The Python bindings are built using `maturin`.
*   To build the Python extension in development mode, navigate to the `python/` directory or use maturin from the root:
    ```bash
    maturin develop
    ```
*   Python tests are located in `python/tests/` and use `pytest` and `hypothesis`.
    ```bash
    cd python && pytest
    ```

## Development Conventions

*   **Backend Trait:** The core library uses a generic `Backend` trait. Implementations like HDF5 and Zarr are provided in separate crates.
*   **Memory Management:** The library prioritizes low memory usage. Operations often return proxy objects, and explicit actions like `.get()`, `.take()`, or `.drain()` are used to load data into memory or transfer ownership.
*   **Python Integration:** Rust structs are exposed to Python using PyO3. The `pyanndata` crate acts as the bridge.
