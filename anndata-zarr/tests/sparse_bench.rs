use std::hint::black_box;
use std::time::Instant;

use anndata::{AnnData, AnnDataOp, ArrayElemOp, Backend};
use anndata_zarr::Zarr;
use sprs::CsMatI;
use tempfile::tempdir;

fn make_csr(rows: usize, cols: usize, per_row: usize) -> CsMatI<f32, i64, u64> {
    let nnz = rows * per_row;
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    indptr.push(0);

    let step = 7919usize;
    for row in 0..rows {
        let mut cols_for_row = (0..per_row)
            .map(|k| ((row + k * step) % cols) as i64)
            .collect::<Vec<_>>();
        cols_for_row.sort_unstable();
        indices.extend(cols_for_row);
        data.extend((0..per_row).map(|k| k as f32));
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

#[test]
#[ignore = "manual sparse full-read benchmark"]
fn bench_sparse_full_read_zarr() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("sparse-full-read.zarr");
    let rows = bench_param("ANNDATA_BENCH_ROWS", 100_000);
    let cols = bench_param("ANNDATA_BENCH_COLS", 20_000);
    let per_row = bench_param("ANNDATA_BENCH_PER_ROW", 64);
    let repeats = bench_param("ANNDATA_BENCH_REPEATS", 5);

    let csr = make_csr(rows, cols, per_row);
    let adata = AnnData::<Zarr>::new(&file).unwrap();
    adata.set_x(&csr).unwrap();
    adata.close().unwrap();

    let adata = AnnData::<Zarr>::open(Zarr::open(&file).unwrap()).unwrap();

    let warmup = adata.x().get::<CsMatI<f32, i64, u64>>().unwrap().unwrap();
    black_box(warmup.nnz());

    let start = Instant::now();
    let mut total_nnz = 0usize;
    for _ in 0..repeats {
        let mat = adata.x().get::<CsMatI<f32, i64, u64>>().unwrap().unwrap();
        total_nnz += black_box(mat.nnz());
    }
    let elapsed = start.elapsed();
    let avg = elapsed / repeats as u32;
    let bytes_per_read = std::mem::size_of_val(csr.data())
        + std::mem::size_of_val(csr.indices())
        + std::mem::size_of_val(csr.indptr().as_slice().unwrap());
    let mib_per_read = bytes_per_read as f64 / (1024.0 * 1024.0);
    let throughput_mib_s = (mib_per_read * repeats as f64) / elapsed.as_secs_f64();

    eprintln!(
        "zarr: shape={rows}x{cols}, nnz={}, payload={mib_per_read:.2} MiB/read, repeats={repeats}, avg={avg:?}, throughput={throughput_mib_s:.2} MiB/s, total_nnz={total_nnz}",
        csr.nnz(),
    );
}
