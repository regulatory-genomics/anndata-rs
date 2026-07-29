//! Manual benchmark for h5ad write/read throughput at both compression settings.
//!
//! Run with:
//! ```text
//! cargo test --release -p anndata-test-utils --test io_bench -- --ignored --nocapture
//! ```
use std::hint::black_box;
use std::time::Instant;

use anndata::backend::{Compression, WriteConfig, set_default_write_config};
use anndata::{AnnData, AnnDataOp, ArrayElemOp, Backend};
use anndata_hdf5::H5;
use sprs::CsMatI;
use tempfile::tempdir;

fn make_csr(rows: usize, cols: usize, per_row: usize) -> CsMatI<f32, i32, u64> {
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::with_capacity(rows * per_row);
    let mut data = Vec::with_capacity(rows * per_row);
    indptr.push(0);

    let step = 7919usize;
    for row in 0..rows {
        let mut cols_for_row = (0..per_row)
            .map(|k| ((row + k * step) % cols) as i32)
            .collect::<Vec<_>>();
        cols_for_row.sort_unstable();
        cols_for_row.dedup();
        data.extend((0..cols_for_row.len()).map(|k| k as f32));
        indices.extend(cols_for_row);
        indptr.push(indices.len() as u64);
    }
    CsMatI::new((rows, cols), indptr, indices, data)
}

fn bench_param(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(default)
}

fn run_case(label: &str, compression: Option<Compression>, csr: &CsMatI<f32, i32, u64>) {
    let dir = tempdir().unwrap();
    let file = dir.path().join("bench.h5ad");
    set_default_write_config(WriteConfig {
        compression,
        block_size: None,
    });

    let start = Instant::now();
    let adata = AnnData::<H5>::new(&file).unwrap();
    adata.set_x(csr).unwrap();
    adata.close().unwrap();
    let write_ms = start.elapsed().as_secs_f64() * 1000.0;

    let size_mb = std::fs::metadata(&file).unwrap().len() as f64 / (1024.0 * 1024.0);

    let start = Instant::now();
    let adata = AnnData::<H5>::open(H5::open(&file).unwrap()).unwrap();
    let x = adata.x().get::<CsMatI<f32, i32, u64>>().unwrap().unwrap();
    let read_ms = start.elapsed().as_secs_f64() * 1000.0;
    black_box(x.nnz());

    println!(
        "{label:<26} write {write_ms:>8.1} ms   read {read_ms:>8.1} ms   file {size_mb:>7.1} MB"
    );
}

#[test]
#[ignore = "manual h5ad io benchmark"]
fn bench_h5ad_io() {
    let rows = bench_param("ANNDATA_BENCH_ROWS", 20_000);
    let cols = bench_param("ANNDATA_BENCH_COLS", 10_000);
    let per_row = bench_param("ANNDATA_BENCH_PER_ROW", 500);

    let csr = make_csr(rows, cols, per_row);
    println!(
        "\n=== {rows}x{cols}, nnz={}, density={:.2}% ===",
        csr.nnz(),
        100.0 * csr.nnz() as f64 / (rows as f64 * cols as f64)
    );

    run_case("zstd-5 (default)", Some(Compression::Zst(5)), &csr);
    run_case("gzip-5", Some(Compression::Gzip(5)), &csr);
    run_case("uncompressed", None, &csr);

    set_default_write_config(WriteConfig::default());
}
