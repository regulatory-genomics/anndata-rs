use anndata::concat::{JoinType, concat};
use anndata::data::SelectInfoElem;
use anndata::{AnnData, AnnDataOp, ArrayElemOp, Backend, Selectable};
use anndata_hdf5::H5;
use anndata_test_utils as utils;
use anndata_test_utils::with_tmp_dir;
use anndata_zarr::Zarr;
use sprs::CsMatI;

#[test]
fn test_basic() {
    utils::test_basic::<H5>();
    utils::test_basic::<Zarr>();
}

#[test]
fn test_complex_dataframe() {
    let input = "tests/data/sample.h5ad";
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata = AnnData::<H5>::open(H5::open(input).unwrap()).unwrap();
        adata.write::<H5, _>(file, None, None).unwrap();
    });

    with_tmp_dir(|dir| {
        let file = dir.join("test.zarr");
        let adata = AnnData::<H5>::open(H5::open(input).unwrap()).unwrap();
        adata.write::<Zarr, _>(file, None, None).unwrap();
    });
}

#[test]
fn test_mixed_layers() {
    utils::test_mixed_layers::<H5>();
    utils::test_mixed_layers::<Zarr>();
}

#[test]
fn test_pairwise() {
    utils::test_pairwise::<H5>();
    utils::test_pairwise::<Zarr>();
}

#[test]
fn test_sparse_edge_cases() {
    utils::test_sparse_edge_cases::<H5>();
    utils::test_sparse_edge_cases::<Zarr>();
}

#[test]
fn test_corrupt_sparse_full_read() {
    utils::test_corrupt_sparse_full_read::<H5>();
    utils::test_corrupt_sparse_full_read::<Zarr>();
}

#[test]
fn test_anndataset_mixed_layouts() {
    utils::test_anndataset_mixed_layouts::<H5>();
    utils::test_anndataset_mixed_layouts::<Zarr>();
}

#[test]
fn test_sparse_extraction_select() {
    utils::test_sparse_extraction_select::<H5>();
    utils::test_sparse_extraction_select::<Zarr>();
}

#[test]
fn test_parallel_reading_stress() {
    utils::test_parallel_reading_stress::<H5>();
    utils::test_parallel_reading_stress::<Zarr>();
}

#[test]
fn test_save() {
    utils::test_save::<H5>();
    utils::test_save::<Zarr>();
}

#[test]
fn test_speacial_cases() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_speacial_cases(adata_gen);

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_speacial_cases(adata_gen);
    })
}

#[test]
fn test_noncanonical() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_noncanonical(adata_gen);

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_noncanonical(adata_gen);
    })
}

#[test]
fn test_io() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_io(adata_gen);

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_io(adata_gen);
    })
}

#[test]
fn test_index() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_index(adata_gen);

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_index(adata_gen);
    })
}

#[test]
fn test_iterator() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_iterator(adata_gen);

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_iterator(adata_gen);
    })
}

#[test]
fn test_concat_sparse_outer() {
    with_tmp_dir(|dir| {
        let adata1 = AnnData::<H5>::new(dir.join("input1.h5ad")).unwrap();
        let adata2 = AnnData::<H5>::new(dir.join("input2.h5ad")).unwrap();
        let output = AnnData::<H5>::new(dir.join("output.h5ad")).unwrap();

        let x1 = CsMatI::<i64, i64, u64>::new((2, 2), vec![0, 2, 3], vec![0, 1, 1], vec![1, 2, 3]);
        let x2 = CsMatI::<i64, i64, u64>::new((1, 2), vec![0, 2], vec![0, 1], vec![4, 5]);
        adata1.set_x(x1).unwrap();
        adata1
            .set_obs_names(vec!["o1".to_string(), "o2".to_string()].into())
            .unwrap();
        adata1
            .set_var_names(vec!["a".to_string(), "b".to_string()].into())
            .unwrap();
        adata2.set_x(x2).unwrap();
        adata2.set_obs_names(vec!["o3".to_string()].into()).unwrap();
        adata2
            .set_var_names(vec!["b".to_string(), "c".to_string()].into())
            .unwrap();

        concat::<_, _, String>(&[adata1, adata2], JoinType::Outer, None, None, &output).unwrap();

        let expected = CsMatI::<i64, i64, u64>::new(
            (3, 3),
            vec![0, 2, 3, 5],
            vec![0, 1, 1, 1, 2],
            vec![1, 2, 3, 4, 5],
        );
        assert_eq!(
            output.x().get::<CsMatI<i64, i64, u64>>().unwrap().unwrap(),
            expected
        );
    });
}

#[test]
fn test_split_sparse() {
    with_tmp_dir(|dir| {
        let adata = AnnData::<H5>::new(dir.join("input.h5ad")).unwrap();
        let x = CsMatI::<i64, i64, u64>::new(
            (5, 3),
            vec![0, 2, 3, 4, 6, 7],
            vec![0, 2, 1, 2, 0, 1, 2],
            vec![1, 2, 3, 4, 5, 6, 7],
        );
        adata.set_x(x.clone()).unwrap();
        adata
            .set_obs_names((0..5).map(|x| x.to_string()).collect())
            .unwrap();
        adata
            .set_var_names((0..3).map(|x| x.to_string()).collect())
            .unwrap();

        let keys = ["A", "A", "B", "A", "B"].map(|key| Some(key.to_string()));
        let split = adata
            .split_obs_by::<H5, _>(&keys, dir.join("split"))
            .unwrap();
        let expected_a = x.select(&[SelectInfoElem::from(vec![0, 1, 3]), SelectInfoElem::full()]);

        assert_eq!(
            split["A"]
                .x()
                .get::<CsMatI<i64, i64, u64>>()
                .unwrap()
                .unwrap(),
            expected_a
        );
    });
}

#[test]
fn test_take_x() {
    utils::test_take_x::<H5>();
    utils::test_take_x::<Zarr>();
}

#[test]
fn test_obsm_drain() {
    utils::test_obsm_drain::<H5>();
    utils::test_obsm_drain::<Zarr>();
}

#[test]
fn test_backend_interop() {
    utils::test_backend_interop::<H5, Zarr>();
    utils::test_backend_interop::<Zarr, H5>();
}

#[test]
fn test_uns_nesting() {
    utils::test_uns_nesting::<H5>();
    utils::test_uns_nesting::<Zarr>();
}

/// `set_default_write_config` must reach the sparse write path. It previously
/// built its own `WriteConfig::default()`, so sparse data was always written
/// with Zstd no matter what the caller configured.
#[test]
fn test_write_config_reaches_sparse_path() {
    use anndata::backend::{
        Compression, WriteConfig, get_default_write_config, set_default_write_config,
    };

    // Highly compressible payload, so the two settings give clearly different
    // file sizes when the config is actually honoured.
    let (rows, cols, per_row) = (500usize, 400usize, 40usize);
    let mut indptr = vec![0u64];
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    for r in 0..rows {
        let mut row: Vec<i32> = (0..per_row).map(|k| ((k * 7 + r) % cols) as i32).collect();
        row.sort_unstable();
        row.dedup();
        data.extend(row.iter().map(|c| ((r + *c as usize) % 3) as f64));
        indices.extend(row);
        indptr.push(indices.len() as u64);
    }
    let mtx: CsMatI<f64, i32, u64> = CsMatI::new((rows, cols), indptr, indices, data);

    let file_size = |compression: Option<Compression>| -> u64 {
        with_tmp_dir(|dir| {
            let path = dir.join("config.h5");
            set_default_write_config(WriteConfig {
                compression,
                block_size: None,
            });
            let adata = AnnData::<H5>::new(&path).unwrap();
            adata.set_x(&mtx).unwrap();
            adata.close().unwrap();
            std::fs::metadata(&path).unwrap().len()
        })
    };

    let compressed = file_size(Some(Compression::Zst(5)));
    let uncompressed = file_size(None);

    // Restore the default for any other test sharing this thread.
    set_default_write_config(WriteConfig::default());
    assert_eq!(
        get_default_write_config().compression,
        Some(Compression::Zst(5))
    );

    assert!(
        compressed < uncompressed,
        "compression setting was ignored: zstd={compressed} bytes, none={uncompressed} bytes"
    );

    // File size alone is too weak: it still moves if only some of the three
    // arrays honour the config. The payload is chosen to compress hard, so an
    // uncompressed file must be at least as large as the raw arrays. If any of
    // data/indices/indptr were still forced through Zstd, this would not hold.
    let nnz = mtx.nnz() as u64;
    let raw_bytes = nnz * (size_of::<f64>() + size_of::<i32>()) as u64
        + (rows as u64 + 1) * size_of::<u64>() as u64;
    assert!(
        uncompressed >= raw_bytes,
        "some sparse array is still compressed: file={uncompressed} bytes < raw={raw_bytes} bytes"
    );

    // The data must survive a round-trip either way.
    with_tmp_dir(|dir| {
        let path = dir.join("roundtrip.h5");
        set_default_write_config(WriteConfig {
            compression: None,
            block_size: None,
        });
        let adata = AnnData::<H5>::new(&path).unwrap();
        adata.set_x(&mtx).unwrap();
        adata.close().unwrap();

        let reopened = AnnData::<H5>::open(H5::open(&path).unwrap()).unwrap();
        let back = reopened
            .x()
            .get::<CsMatI<f64, i32, u64>>()
            .unwrap()
            .unwrap();
        set_default_write_config(WriteConfig::default());
        assert_eq!(back, mtx);
    });
}
