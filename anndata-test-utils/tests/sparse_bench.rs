use std::hint::black_box;
use std::time::Instant;

use anndata::backend::{AttributeOp, DatasetOp, GroupOp};
use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend, HasShape};
use anndata_hdf5::H5;
use anndata_zarr::Zarr;
use ndarray::Ix1;
use sprs::{CompressedStorage, CsMatI};
use tempfile::tempdir;

fn make_csr(rows: usize, cols: usize, per_row: usize) -> CsMatI<f32, i64, u64> {
    let nnz = rows * per_row;
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    indptr.push(0);

    // Deterministic, sorted, duplicate-free row pattern without RNG overhead.
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

fn bench_full_read<B: Backend>(label: &str) {
    let dir = tempdir().unwrap();
    let file = dir.path().join(label);
    let rows = bench_param("ANNDATA_BENCH_ROWS", 100_000);
    let cols = bench_param("ANNDATA_BENCH_COLS", 20_000);
    let per_row = bench_param("ANNDATA_BENCH_PER_ROW", 64);
    let repeats = bench_param("ANNDATA_BENCH_REPEATS", 5);

    let csr = make_csr(rows, cols, per_row);
    let adata = AnnData::<B>::new(&file).unwrap();
    adata.set_x(&csr).unwrap();
    adata.close().unwrap();

    let adata = AnnData::<B>::open(B::open(&file).unwrap()).unwrap();

    // Warmup: populate backend/file-system caches outside measured loop.
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
        "{label}: shape={rows}x{cols}, nnz={}, payload={mib_per_read:.2} MiB/read, repeats={repeats}, avg={avg:?}, throughput={throughput_mib_s:.2} MiB/s, total_nnz={total_nnz}",
        csr.nnz(),
    );
}

#[test]
#[ignore = "manual sparse full-read benchmark"]
fn bench_sparse_full_read_zarr() {
    bench_full_read::<Zarr>("sparse-full-read.zarr");
}

#[test]
#[ignore = "manual real-world sparse full-read benchmark"]
fn bench_real_pbmc_5k_h5() {
    let path =
        std::env::var("ANNDATA_BENCH_H5AD").unwrap_or_else(|_| "data/pbmc_5k.h5ad".to_string());
    let repeats = bench_param("ANNDATA_BENCH_REPEATS", 20);
    let adata = AnnData::<H5>::open(H5::open(&path).unwrap()).unwrap();

    eprintln!(
        "pbmc_5k: path={path}, dtype={:?}, shape={:?}",
        adata.x().dtype(),
        adata.x().shape()
    );

    let warmup = adata.x().get::<ArrayData>().unwrap().unwrap();
    black_box(warmup.shape());

    let start = Instant::now();
    let mut total_cells = 0usize;
    for _ in 0..repeats {
        let x = adata.x().get::<ArrayData>().unwrap().unwrap();
        let shape = black_box(x.shape());
        total_cells += shape.as_ref().iter().product::<usize>();
    }
    let elapsed = start.elapsed();
    let avg = elapsed / repeats as u32;
    eprintln!("pbmc_5k ArrayData: repeats={repeats}, avg={avg:?}, total_cells={total_cells}");

    let warmup = adata.x().get::<CsMatI<i64, i32, u64>>().unwrap().unwrap();
    black_box(warmup.nnz());

    let start = Instant::now();
    let mut total_nnz = 0usize;
    for _ in 0..repeats {
        let x = adata.x().get::<CsMatI<i64, i32, u64>>().unwrap().unwrap();
        total_nnz += black_box(x.nnz());
    }
    let elapsed = start.elapsed();
    let avg = elapsed / repeats as u32;
    eprintln!("pbmc_5k CsMatI<i64,i32,u64>: repeats={repeats}, avg={avg:?}, total_nnz={total_nnz}");
}

#[test]
#[ignore = "manual real-world sparse Zarr full-read benchmark"]
fn bench_real_pbmc_5k_zarr() {
    let path =
        std::env::var("ANNDATA_BENCH_ZARR").unwrap_or_else(|_| "data/pbmc_5k.zarr".to_string());
    let repeats = bench_param("ANNDATA_BENCH_REPEATS", 20);
    let adata = AnnData::<Zarr>::open(Zarr::open(&path).unwrap()).unwrap();

    eprintln!(
        "pbmc_5k_zarr: path={path}, dtype={:?}, shape={:?}",
        adata.x().dtype(),
        adata.x().shape()
    );

    let warmup = adata.x().get::<ArrayData>().unwrap().unwrap();
    black_box(warmup.shape());

    let start = Instant::now();
    let mut total_cells = 0usize;
    for _ in 0..repeats {
        let x = adata.x().get::<ArrayData>().unwrap().unwrap();
        let shape = black_box(x.shape());
        total_cells += shape.as_ref().iter().product::<usize>();
    }
    let elapsed = start.elapsed();
    let avg = elapsed / repeats as u32;
    eprintln!("pbmc_5k_zarr ArrayData: repeats={repeats}, avg={avg:?}, total_cells={total_cells}");

    let warmup = adata.x().get::<CsMatI<i64, i32, u64>>().unwrap().unwrap();
    black_box(warmup.nnz());

    let start = Instant::now();
    let mut total_nnz = 0usize;
    for _ in 0..repeats {
        let x = adata.x().get::<CsMatI<i64, i32, u64>>().unwrap().unwrap();
        total_nnz += black_box(x.nnz());
    }
    let elapsed = start.elapsed();
    let avg = elapsed / repeats as u32;
    eprintln!(
        "pbmc_5k_zarr CsMatI<i64,i32,u64>: repeats={repeats}, avg={avg:?}, total_nnz={total_nnz}"
    );
}

#[test]
#[ignore = "manual real-world sparse Zarr raw-read benchmark"]
fn bench_real_pbmc_5k_zarr_raw_parts() {
    let path =
        std::env::var("ANNDATA_BENCH_ZARR").unwrap_or_else(|_| "data/pbmc_5k.zarr".to_string());
    let repeats = bench_param("ANNDATA_BENCH_REPEATS", 20);
    let store = Zarr::open(&path).unwrap();
    let group = store.open_group("X").unwrap();
    let shape: Vec<u64> = group.get_attr("shape").unwrap();
    let data_ds = group.open_dataset("data").unwrap();
    let indices_ds = group.open_dataset("indices").unwrap();
    let indptr_ds = group.open_dataset("indptr").unwrap();

    eprintln!(
        "pbmc_5k_zarr raw parts: path={path}, shape={shape:?}, data_shape={:?}, indices_shape={:?}, indptr_shape={:?}",
        data_ds.shape(),
        indices_ds.shape(),
        indptr_ds.shape()
    );

    let warmup_data = data_ds.read_array::<i64, Ix1>().unwrap();
    let warmup_indices = indices_ds.read_array::<i32, Ix1>().unwrap();
    let warmup_indptr = indptr_ds.read_array::<i32, Ix1>().unwrap();
    black_box((warmup_data.len(), warmup_indices.len(), warmup_indptr.len()));

    let start = Instant::now();
    let mut total_len = 0usize;
    for _ in 0..repeats {
        let data = data_ds.read_array::<i64, Ix1>().unwrap();
        let indices = indices_ds.read_array::<i32, Ix1>().unwrap();
        let indptr = indptr_ds.read_array::<i32, Ix1>().unwrap();
        total_len += black_box(data.len() + indices.len() + indptr.len());
    }
    let elapsed = start.elapsed();
    eprintln!(
        "pbmc_5k_zarr backend read_array sequential: repeats={repeats}, avg={:?}, total_len={total_len}",
        elapsed / repeats as u32
    );

    let start = Instant::now();
    let mut total_len = 0usize;
    for _ in 0..repeats {
        let (data, (indptr, indices)) = rayon::join(
            || data_ds.read_array::<i64, Ix1>().unwrap(),
            || {
                rayon::join(
                    || indptr_ds.read_array::<i32, Ix1>().unwrap(),
                    || indices_ds.read_array::<i32, Ix1>().unwrap(),
                )
            },
        );
        total_len += black_box(data.len() + indices.len() + indptr.len());
    }
    let elapsed = start.elapsed();
    eprintln!(
        "pbmc_5k_zarr backend read_array rayon: repeats={repeats}, avg={:?}, total_len={total_len}",
        elapsed / repeats as u32
    );
}

#[test]
#[ignore = "manual real-world sparse HDF5 raw-read benchmark"]
fn bench_real_pbmc_5k_h5_raw_parts() {
    let path =
        std::env::var("ANNDATA_BENCH_H5AD").unwrap_or_else(|_| "data/pbmc_5k.h5ad".to_string());
    let repeats = bench_param("ANNDATA_BENCH_REPEATS", 20);
    let file = H5::open(&path).unwrap();
    let group = file.open_group("X").unwrap();
    let shape: Vec<u64> = group.get_attr("shape").unwrap();
    let data_ds = group.open_dataset("data").unwrap();
    let indices_ds = group.open_dataset("indices").unwrap();
    let indptr_ds = group.open_dataset("indptr").unwrap();

    eprintln!(
        "pbmc_5k raw parts: path={path}, shape={shape:?}, data_shape={:?}, indices_shape={:?}, indptr_shape={:?}",
        data_ds.shape(),
        indices_ds.shape(),
        indptr_ds.shape()
    );

    let warmup_data = data_ds.read_array::<i64, Ix1>().unwrap();
    let warmup_indices = indices_ds.read_array::<i32, Ix1>().unwrap();
    let warmup_indptr = indptr_ds.read_array::<u64, Ix1>().unwrap();
    black_box((warmup_data.len(), warmup_indices.len(), warmup_indptr.len()));

    let start = Instant::now();
    let mut total_len = 0usize;
    for _ in 0..repeats {
        let data = data_ds.read_array::<i64, Ix1>().unwrap();
        let indices = indices_ds.read_array::<i32, Ix1>().unwrap();
        let indptr = indptr_ds.read_array::<u64, Ix1>().unwrap();
        total_len += black_box(data.len() + indices.len() + indptr.len());
    }
    let elapsed = start.elapsed();
    eprintln!(
        "pbmc_5k backend read_array sequential: repeats={repeats}, avg={:?}, total_len={total_len}",
        elapsed / repeats as u32
    );

    let start = Instant::now();
    let mut total_len = 0usize;
    for _ in 0..repeats {
        let (data, (indptr, indices)) = rayon::join(
            || data_ds.read_array::<i64, Ix1>().unwrap(),
            || {
                rayon::join(
                    || indptr_ds.read_array::<u64, Ix1>().unwrap(),
                    || indices_ds.read_array::<i32, Ix1>().unwrap(),
                )
            },
        );
        total_len += black_box(data.len() + indices.len() + indptr.len());
    }
    let elapsed = start.elapsed();
    eprintln!(
        "pbmc_5k backend read_array rayon: repeats={repeats}, avg={:?}, total_len={total_len}",
        elapsed / repeats as u32
    );

    let start = Instant::now();
    let mut total_nnz = 0usize;
    for _ in 0..repeats {
        let data = data_ds
            .read_array::<i64, Ix1>()
            .unwrap()
            .into_raw_vec_and_offset()
            .0;
        let indices = indices_ds
            .read_array::<i32, Ix1>()
            .unwrap()
            .into_raw_vec_and_offset()
            .0;
        let indptr = indptr_ds
            .read_array::<u64, Ix1>()
            .unwrap()
            .into_raw_vec_and_offset()
            .0;
        let mat = CsMatI::try_new_csc(
            (shape[0] as usize, shape[1] as usize),
            indptr,
            indices,
            data,
        )
        .unwrap();
        total_nnz += black_box(mat.nnz());
    }
    let elapsed = start.elapsed();
    eprintln!(
        "pbmc_5k sequential read_array + try_new_csc: repeats={repeats}, avg={:?}, total_nnz={total_nnz}",
        elapsed / repeats as u32
    );

    let start = Instant::now();
    let mut total_nnz = 0usize;
    for _ in 0..repeats {
        let data = data_ds
            .read_array::<i64, Ix1>()
            .unwrap()
            .into_raw_vec_and_offset()
            .0;
        let indices = indices_ds
            .read_array::<i32, Ix1>()
            .unwrap()
            .into_raw_vec_and_offset()
            .0;
        let indptr = indptr_ds
            .read_array::<u64, Ix1>()
            .unwrap()
            .into_raw_vec_and_offset()
            .0;
        let mat = unsafe {
            CsMatI::new_unchecked(
                CompressedStorage::CSC,
                (shape[0] as usize, shape[1] as usize),
                indptr,
                indices,
                data,
            )
        };
        total_nnz += black_box(mat.nnz());
    }
    let elapsed = start.elapsed();
    eprintln!(
        "pbmc_5k sequential read_array + new_unchecked: repeats={repeats}, avg={:?}, total_nnz={total_nnz}",
        elapsed / repeats as u32
    );
}
