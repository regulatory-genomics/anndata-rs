//! Manual benchmark for sparse `select` (subsetting) throughput.
//!
//! Run with:
//! ```text
//! cargo test --release -p anndata-test-utils --test select_bench -- --ignored --nocapture
//! ```
use std::hint::black_box;
use std::time::Instant;

use anndata::backend::BackendData;
use anndata::data::{SelectInfoElem, Selectable};
use num::ToPrimitive;
use sprs::{CsMatI, SpIndex};

/// Generic over the index type so the benchmark can match scipy's `int32`
/// indices as well as the wider `i64` this crate also supports.
fn make_csr<T>(rows: usize, cols: usize, per_row: usize) -> CsMatI<f32, T, u64>
where
    T: BackendData + SpIndex + ToPrimitive + num::Integer + num::FromPrimitive,
{
    let nnz = rows * per_row;
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    indptr.push(0);

    // Deterministic, sorted, duplicate-free row pattern without RNG overhead.
    let step = 7919usize;
    for row in 0..rows {
        let mut cols_for_row = (0..per_row)
            .map(|k| <T as SpIndex>::from_usize((row + k * step) % cols))
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

fn time_it<T>(label: &str, repeats: usize, mut f: impl FnMut() -> T) {
    // Warmup outside the measured loop.
    black_box(f());
    let start = Instant::now();
    for _ in 0..repeats {
        black_box(f());
    }
    let elapsed = start.elapsed();
    println!(
        "{label:<44} {:>9.3} ms/iter",
        elapsed.as_secs_f64() * 1000.0 / repeats as f64
    );
}

fn run_case<T>(label: &str, rows: usize, cols: usize, per_row: usize, repeats: usize)
where
    T: BackendData + SpIndex + ToPrimitive + num::Integer + num::FromPrimitive,
{
    let csr = make_csr::<T>(rows, cols, per_row);
    println!(
        "\n=== {label}: {rows}x{cols}, nnz={}, density={:.2}% ===",
        csr.nnz(),
        100.0 * csr.nnz() as f64 / (rows as f64 * cols as f64)
    );

    // Half the rows, every column (fast path: contiguous minor axis).
    let row_half: Vec<usize> = (0..rows / 2).collect();
    let full = SelectInfoElem::full();
    let sel_rows = [SelectInfoElem::Index(row_half.clone()), full.clone()];
    time_it("subset rows (index), all cols", repeats, || {
        csr.select(&sel_rows)
    });

    // Half the rows, half the columns (the slow HashMap path).
    let col_half: Vec<usize> = (0..cols / 2).collect();
    let sel_both = [
        SelectInfoElem::Index(row_half.clone()),
        SelectInfoElem::Index(col_half.clone()),
    ];
    time_it("subset rows + cols (both index)", repeats, || {
        csr.select(&sel_both)
    });

    // Strided column selection (monotonic, sparse in the minor axis).
    let col_stride: Vec<usize> = (0..cols).step_by(3).collect();
    let sel_stride = [
        SelectInfoElem::Index(row_half.clone()),
        SelectInfoElem::Index(col_stride),
    ];
    time_it("subset rows + strided cols", repeats, || {
        csr.select(&sel_stride)
    });

    // Shuffled (non-monotonic) column selection: exercises the sorting path.
    let mut col_shuffled: Vec<usize> = (0..cols / 2).collect();
    col_shuffled.reverse();
    let sel_shuffled = [
        SelectInfoElem::Index(row_half),
        SelectInfoElem::Index(col_shuffled),
    ];
    time_it(
        "subset rows + reversed cols (non-monotonic)",
        repeats,
        || csr.select(&sel_shuffled),
    );
}

#[test]
#[ignore = "manual sparse select benchmark"]
fn bench_sparse_select() {
    let repeats = bench_param("ANNDATA_BENCH_REPEATS", 10);
    // i32 indices match what scipy uses, so the two move the same bytes.
    run_case::<i32>("small (i32 idx)", 5_000, 2_000, 100, repeats);
    run_case::<i32>("medium (i32 idx)", 20_000, 10_000, 500, repeats);
    run_case::<i64>("medium (i64 idx)", 20_000, 10_000, 500, repeats);
}
