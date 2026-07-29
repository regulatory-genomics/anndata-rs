mod common;
pub use common::*;

use anndata::backend::{AttributeOp, GroupOp, StoreOp};
use anndata::concat::{JoinType, concat};
use anndata::data::SelectInfoElem;
use anndata::{data::CsrNonCanonical, *};
use data::ArrayConvert;
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use proptest::prelude::*;
use sprs::{CsMatI, TriMatI};
use std::collections::HashMap;
use std::path::Path;

pub fn test_basic<B: Backend>() {
    with_tmp_dir(|dir| {
        let ann1 = AnnData::<B>::new(dir.join("test1")).unwrap();
        let csc = rand_csc::<i32>(10, 5, 3, 1, 100);
        ann1.obsm().add("csc", &csc).unwrap();
        assert!(
            ann1.obsm()
                .get_item::<CsMatI<i32, i64, u64>>("csc")
                .unwrap()
                .unwrap()
                .is_csc()
        );

        let ann2 = AnnData::<B>::new(dir.join("test2")).unwrap();
        AnnDataSet::<B>::new(
            [("ann1", ann1), ("ann2", ann2)],
            dir.join("dataset"),
            "sample",
            false,
        )
        .unwrap();
    })
}

pub fn test_save<B: Backend>() {
    with_tmp_dir(|dir| {
        let input = dir.join("input");
        let output = dir.join("output");
        let anndatas = ((0_usize..100), (0_usize..100)).prop_flat_map(|(n_obs, n_vars)| {
            (
                anndata_strat::<B, _>(&input, n_obs, n_vars),
                select_strat(n_obs),
                select_strat(n_vars),
            )
        });
        proptest!(ProptestConfig::with_cases(100), |((adata, slice_obs, slice_var) in anndatas)| {
            adata.write::<B, _>(&output, None, None).unwrap();
            let adata_in = AnnData::<B>::open(B::open(&output).unwrap()).unwrap();
            prop_assert!(anndata_eq(&adata, &adata_in).unwrap());
            adata_in.close().unwrap();

            let index = adata.obs_names().select(&slice_obs);
            assert_eq!(index.len(), index.into_vec().len());

            let select = [slice_obs, slice_var];
            adata.write_select::<B, _, _>(&select, &output).unwrap();
            adata.subset(&select).unwrap();
            let adata_in = AnnData::<B>::open(B::open(&output).unwrap()).unwrap();
            prop_assert!(anndata_eq(&adata, &adata_in).unwrap());
            adata_in.close().unwrap();
        });
    });
}

pub fn test_speacial_cases<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let adata = adata_gen();

    let arr = Array2::<i32>::zeros((0, 0));
    adata.set_x(&arr).unwrap();

    // Adding matrices with wrong shapes should fail
    let arr2 = Array2::<i32>::zeros((10, 20));
    assert!(adata.obsm().add("test", &arr2).is_err());

    // Data type casting
    let _: Array2<f64> = adata
        .x()
        .get::<ArrayData>()
        .unwrap()
        .unwrap()
        .try_convert()
        .expect("data type casting failed");
}

pub fn test_noncanonical<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let adata = adata_gen();
    // Construct a non-canonical matrix with duplicate entries
    let coo: TriMatI<i32, u64> = TriMatI::from_triplets(
        (5, 4),
        vec![0, 0, 1, 1, 1, 2, 3, 4], // Duplicate (0,0) and (1,0)
        vec![0, 0, 0, 0, 2, 3, 1, 3],
        vec![1, 10, 2, 20, 4, 5, 6, 7],
    );
    adata.set_x(CsrNonCanonical::from(&coo)).unwrap();

    // Attempting to get as CsMatI should fail because it's non-canonical on disk (duplicates)
    assert!(adata.x().get::<CsMatI<i32, i64, u64>>().is_err());

    // Getting as ArrayData should succeed and return CsrNonCanonical variant
    let data = adata.x().get::<ArrayData>().unwrap().unwrap();
    assert!(matches!(data, ArrayData::CsrNonCanonical(_)));

    // Convert to CsrNonCanonical specifically
    let non_canonical: CsrNonCanonical<i32> = data.try_into().unwrap();
    assert_eq!(non_canonical.nrows(), 5);

    // Currently canonicalize() only works if there are no duplicates.
    // So it should return Err(self) here.
    assert!(non_canonical.canonicalize().is_err());
}

pub fn test_mixed_layers<B: Backend>() {
    with_tmp_dir(|dir| {
        let adata = AnnData::<B>::new(dir.join("mixed_layers")).unwrap();
        let csr = rand_csr::<f64>(50, 20, 10, 0.0, 1.0);
        let csc = rand_csc::<i32>(50, 20, 10, 0, 100);
        let dense = Array2::<f32>::from_elem((50, 20), 1.0);

        adata.set_x(&csr).unwrap();
        adata.layers().add("csc_layer", &csc).unwrap();
        adata.layers().add("dense_layer", &dense).unwrap();

        // Verify layouts are preserved
        assert!(
            adata
                .x()
                .get::<CsMatI<f64, i64, u64>>()
                .unwrap()
                .unwrap()
                .is_csr()
        );
        assert!(
            adata
                .layers()
                .get_item::<CsMatI<i32, i64, u64>>("csc_layer")
                .unwrap()
                .unwrap()
                .is_csc()
        );
        assert_eq!(
            adata
                .layers()
                .get_item::<Array2<f32>>("dense_layer")
                .unwrap()
                .unwrap(),
            dense
        );

        // Test subsetting across all mixed layers
        let select = [SelectInfoElem::from(0..10), SelectInfoElem::full()];
        adata.subset(&select).unwrap();

        assert_eq!(adata.n_obs(), 10);
        assert!(
            adata
                .x()
                .get::<CsMatI<f64, i64, u64>>()
                .unwrap()
                .unwrap()
                .is_csr()
        );
        assert_eq!(
            adata
                .x()
                .get::<CsMatI<f64, i64, u64>>()
                .unwrap()
                .unwrap()
                .rows(),
            10
        );
        assert!(
            adata
                .layers()
                .get_item::<CsMatI<i32, i64, u64>>("csc_layer")
                .unwrap()
                .unwrap()
                .is_csc()
        );
        assert_eq!(
            adata
                .layers()
                .get_item::<CsMatI<i32, i64, u64>>("csc_layer")
                .unwrap()
                .unwrap()
                .rows(),
            10
        );
        assert_eq!(
            adata
                .layers()
                .get_item::<Array2<f32>>("dense_layer")
                .unwrap()
                .unwrap()
                .shape(),
            &[10, 20]
        );
    });
}

pub fn test_pairwise<B: Backend>() {
    with_tmp_dir(|dir| {
        let adata = AnnData::<B>::new(dir.join("pairwise")).unwrap();
        adata.set_n_obs(100).unwrap();
        adata.set_n_vars(50).unwrap();

        // Create square sparse matrix for obsp (100x100)
        let obsp_data = rand_csr::<f64>(100, 100, 50, 0.0, 1.0);
        adata.obsp().add("distances", &obsp_data).unwrap();

        // Attempting to add non-square matrix to obsp should fail
        let bad_data = rand_csr::<f64>(100, 50, 10, 0.0, 1.0);
        assert!(adata.obsp().add("bad", &bad_data).is_err());

        // Subset adata (rows 10..20)
        let select = [SelectInfoElem::from(10..20), SelectInfoElem::full()];
        adata.subset(&select).unwrap();

        assert_eq!(adata.n_obs(), 10);
        // Pairwise matrix should now be 10x10 (subsetted on both axes)
        let sliced_obsp = adata
            .obsp()
            .get_item::<CsMatI<f64, i64, u64>>("distances")
            .unwrap()
            .unwrap();
        assert_eq!(sliced_obsp.rows(), 10);
        assert_eq!(sliced_obsp.cols(), 10);
    });
}

pub fn test_sparse_edge_cases<B: Backend>() {
    with_tmp_dir(|dir| {
        let adata = AnnData::<B>::new(dir.join("edge_cases")).unwrap();

        // Case 1: Empty matrix (0x0)
        let empty = rand_csr::<f64>(0, 0, 0, 0.0, 1.0);
        adata.set_x(&empty).unwrap();
        assert_eq!(adata.n_obs(), 0);
        assert_eq!(adata.n_vars(), 0);

        // Case 2: Sparse matrix with an entirely empty row in the middle
        let adata2 = AnnData::<B>::new(dir.join("empty_row")).unwrap();
        let indptr = [0, 1, 1, 2]; // row 1 is empty
        let indices = [0, 1];
        let data = vec![1.0, 2.0];
        let sparse = CsMatI::<f64, i64, u64>::new(
            (3, 3),
            indptr.iter().map(|&x| x as u64).collect(),
            indices.iter().map(|&x| x as i64).collect(),
            data,
        );
        adata2.set_x(&sparse).unwrap();

        let read_back = adata2.x().get::<CsMatI<f64, i64, u64>>().unwrap().unwrap();
        assert_eq!(
            read_back.indptr().as_slice().unwrap()[2],
            read_back.indptr().as_slice().unwrap()[1]
        );

        // Case 3: NNZ = 0 but shape is non-zero
        let adata3 = AnnData::<B>::new(dir.join("all_zeros")).unwrap();
        let all_zeros = CsMatI::<f64, i64, u64>::new((10, 10), vec![0; 11], vec![], vec![]);
        adata3.obsm().add("zeros", &all_zeros).unwrap();
        let read_zeros = adata3
            .obsm()
            .get_item::<CsMatI<f64, i64, u64>>("zeros")
            .unwrap()
            .unwrap();
        assert_eq!(read_zeros.nnz(), 0);
        assert_eq!(read_zeros.rows(), 10);
    });
}

pub fn test_anndataset_mixed_layouts<B: Backend>() {
    with_tmp_dir(|dir| {
        let adata1 = AnnData::<B>::new(dir.join("adata1")).unwrap();
        let csr1 = rand_csr::<f64>(10, 5, 5, 0.0, 1.0);
        adata1.set_x(&csr1).unwrap();

        let adata2 = AnnData::<B>::new(dir.join("adata2")).unwrap();
        let csr2 = rand_csr::<f64>(20, 5, 5, 0.0, 1.0);
        adata2.set_x(&csr2).unwrap();

        let dataset = AnnDataSet::<B>::new(
            [("ann1", adata1), ("ann2", adata2)],
            dir.join("dataset_csr"),
            "sample",
            false,
        )
        .unwrap();

        assert_eq!(dataset.n_obs(), 30);
        let x = dataset.x().get::<CsMatI<f64, i64, u64>>().unwrap().unwrap();
        assert!(x.is_csr());
    });
}

#[derive(Clone)]
struct SparseFixture {
    encoding_type: &'static str,
    shape: Vec<u64>,
    indptr: Vec<u64>,
    indices: Vec<i32>,
    data: Vec<i64>,
}

fn write_sparse_x<B: Backend>(path: &Path, fixture: &SparseFixture) {
    let file = B::new(path).unwrap();
    {
        let mut group = file.new_group("X").unwrap();
        group
            .new_attr("encoding-type", fixture.encoding_type)
            .unwrap();
        group.new_attr("encoding-version", "0.1.0").unwrap();
        group.new_attr("shape", fixture.shape.clone()).unwrap();
        group
            .new_array_dataset(
                "indptr",
                fixture.indptr.as_slice().into(),
                Default::default(),
            )
            .unwrap();
        group
            .new_array_dataset(
                "indices",
                fixture.indices.as_slice().into(),
                Default::default(),
            )
            .unwrap();
        group
            .new_array_dataset("data", fixture.data.as_slice().into(), Default::default())
            .unwrap();
    }
    file.close().unwrap();
}

fn append_corrupt_sparse_cases(
    cases: &mut Vec<(String, SparseFixture)>,
    prefix: &str,
    valid: SparseFixture,
) {
    let mut bad = valid.clone();
    bad.indices = vec![2, 0, 1];
    cases.push((format!("{prefix}_unsorted_indices"), bad));

    let mut bad = valid.clone();
    bad.indices = vec![0, 3, 1];
    cases.push((format!("{prefix}_out_of_bounds_indices"), bad));

    let mut bad = valid.clone();
    bad.indptr = vec![0, 2, 1];
    cases.push((format!("{prefix}_non_monotonic_indptr"), bad));

    let mut bad = valid.clone();
    bad.indptr = vec![0, 2];
    bad.indices = vec![0, 2];
    bad.data = vec![1, 2];
    cases.push((format!("{prefix}_bad_indptr_length"), bad));

    let mut bad = valid.clone();
    bad.data = vec![1, 2];
    cases.push((format!("{prefix}_mismatched_data_indices_lengths"), bad));

    let mut bad = valid.clone();
    bad.indptr = vec![0, 2, 4];
    cases.push((format!("{prefix}_indptr_nnz_mismatch"), bad));
}

pub fn test_corrupt_sparse_full_read<B: Backend>() {
    let mut cases = Vec::new();

    append_corrupt_sparse_cases(
        &mut cases,
        "csr",
        SparseFixture {
            encoding_type: "csr_matrix",
            shape: vec![2, 3],
            indptr: vec![0, 2, 3],
            indices: vec![0, 2, 1],
            data: vec![1, 2, 3],
        },
    );
    append_corrupt_sparse_cases(
        &mut cases,
        "csc",
        SparseFixture {
            encoding_type: "csc_matrix",
            shape: vec![3, 2],
            indptr: vec![0, 2, 3],
            indices: vec![0, 2, 1],
            data: vec![1, 2, 3],
        },
    );

    with_tmp_dir(|dir| {
        for (name, fixture) in &cases {
            let path = dir.join(name);
            write_sparse_x::<B>(&path, fixture);

            let result = AnnData::<B>::open(B::open(&path).unwrap())
                .and_then(|adata| adata.x().get::<CsMatI<i64, i32, u64>>().map(|_| ()));
            assert!(
                result.is_err(),
                "corrupt sparse fixture should fail: {name}"
            );
        }
    });
}

pub fn test_sparse_extraction_select<B: Backend>() {
    with_tmp_dir(|dir| {
        let adata = AnnData::<B>::new(dir.join("sparse_select")).unwrap();
        let csr = CsMatI::<i32, i64, u64>::new(
            (5, 5),
            vec![0, 2, 3, 5, 6, 7],
            vec![0, 3, 1, 2, 4, 3, 0],
            vec![1, 2, 3, 4, 5, 6, 7],
        );
        adata.set_x(&csr).unwrap();

        // Non-zero major slice plus minor selection exercises the optimized
        // extraction path and ensures the minor filter is applied to the
        // extracted submatrix, not to the original major coordinates again.
        let select = [
            SelectInfoElem::from(2..5),
            SelectInfoElem::from(vec![0usize, 3, 4]),
        ];
        let sliced = adata
            .x()
            .slice::<CsMatI<i32, i64, u64>, _>(&select)
            .unwrap()
            .unwrap();
        let expected =
            CsMatI::<i32, i64, u64>::new((3, 3), vec![0, 1, 2, 3], vec![2, 1, 0], vec![5, 6, 7]);
        assert_eq!(sliced, expected);

        // Arbitrary/repeated major extraction must preserve output order and
        // duplicates without falling back to whole-matrix extraction.
        let select = [
            SelectInfoElem::from(vec![4usize, 2, 2]),
            SelectInfoElem::full(),
        ];
        let sliced = adata
            .x()
            .slice::<CsMatI<i32, i64, u64>, _>(&select)
            .unwrap()
            .unwrap();
        let expected = CsMatI::<i32, i64, u64>::new(
            (3, 5),
            vec![0, 1, 3, 5],
            vec![0, 2, 4, 2, 4],
            vec![7, 4, 5, 4, 5],
        );
        assert_eq!(sliced, expected);
    });
}

pub fn test_parallel_reading_stress<B: Backend>() {
    with_tmp_dir(|dir| {
        let n_adatas = 20;
        let n_obs_per_adata = 10;
        let n_vars = 5;
        let mut adatas = Vec::new();

        for i in 0..n_adatas {
            let adata = AnnData::<B>::new(dir.join(format!("adata_{i}"))).unwrap();
            let csr = rand_csr::<f64>(n_obs_per_adata, n_vars, 5, 0.0, 1.0);
            adata.set_x(&csr).unwrap();
            adatas.push((format!("ann_{i}"), adata));
        }

        let dataset =
            AnnDataSet::<B>::new(adatas, dir.join("dataset_stress"), "sample", false).unwrap();

        // Standard sequential read
        let x_seq = dataset.x().get::<CsMatI<f64, i64, u64>>().unwrap().unwrap();

        // StackedArrayElem::select uses Rayon internally for parallel reading
        // We select the entire range to compare with full read
        let select = [SelectInfoElem::full(), SelectInfoElem::full()];
        let x_par = dataset
            .x()
            .slice::<CsMatI<f64, i64, u64>, _>(&select)
            .unwrap()
            .unwrap();

        assert_eq!(x_seq.rows(), n_adatas * n_obs_per_adata);
        assert_eq!(x_seq.nnz(), x_par.nnz());
        assert_eq!(x_seq.indptr(), x_par.indptr());
        assert_eq!(x_seq.indices(), x_par.indices());
        assert_eq!(x_seq.data(), x_par.data());
    });
}

pub fn test_io<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays =
        proptest::collection::vec(0_usize..50, 2..4).prop_flat_map(|shape| array_strat(&shape));
    proptest!(ProptestConfig::with_cases(256), |(x in arrays)| {
        let adata = adata_gen();
        adata.set_x(&x).unwrap();
        prop_assert_eq!(adata.x().get::<ArrayData>().unwrap().unwrap(), x);
    });
}

pub fn test_index<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays = proptest::collection::vec(0_usize..50, 2..4)
        .prop_flat_map(|shape| array_slice_strat(&shape));
    proptest!(ProptestConfig::with_cases(256), |((x, select) in arrays)| {
        let adata = adata_gen();
        adata.set_x(&x).unwrap();
        prop_assert_eq!(
            adata.x().slice::<ArrayData, _>(&select).unwrap().unwrap(),
            array_select(&x, select.as_slice())
        );

        adata.obsm().add("test", &x).unwrap();
        prop_assert_eq!(
            adata.obsm().get_item_slice::<ArrayData, _>("test", &select).unwrap().unwrap(),
            array_select(&x, select.as_slice())
        );
    });
}

pub fn test_iterator<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays =
        proptest::collection::vec(20_usize..50, 2..3).prop_flat_map(|shape| array_strat(&shape));
    proptest!(ProptestConfig::with_cases(10), |(x in arrays)| {
        if let ArrayData::CscMatrix(_) = x {
        } else {
            let adata = adata_gen();
            adata.obsm().add_iter("test", array_chunks(&x, 7)).unwrap();
            prop_assert_eq!(adata.obsm().get_item::<ArrayData>("test").unwrap().unwrap(), x.clone());

            adata.obsm().add_iter("test2", adata.obsm().get_item_iter::<ArrayData>("test", 7).unwrap().map(|x| x.0)).unwrap();
            prop_assert_eq!(adata.obsm().get_item::<ArrayData>("test2").unwrap().unwrap(), x);
        }
    });
}

pub fn test_concat<B: Backend>() {
    with_tmp_dir(|dir| {
        let input1 = dir.join("input1");
        let input2 = dir.join("input2");
        let output = dir.join("output");
        let anndatas = (
            (0_usize..100),
            (0_usize..100),
            (0_usize..100),
            (0_usize..100),
        )
            .prop_flat_map(|(n_obs1, n_vars1, n_obs2, n_vars2)| {
                (
                    anndata_strat::<B, _>(&input1, n_obs1, n_vars1),
                    anndata_strat::<B, _>(&input2, n_obs2, n_vars2),
                )
            });

        proptest!(ProptestConfig::with_cases(100), |((adata1, adata2) in anndatas)| {
            let adatas = [adata1, adata2];

            let out = AnnData::<B>::new(&output).unwrap();
            concat::<_, _, String>(&adatas, JoinType::Inner, None, None, &out).unwrap();

            let out = AnnData::<B>::new(&output).unwrap();
            concat::<_, _, String>(&adatas, JoinType::Outer, None, None, &out).unwrap();
        })
    });
}

pub fn test_take_x<B: Backend>() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata = AnnData::<B>::new(&file).unwrap();
        let x: ArrayData = Array::random((10, 50), Uniform::new(0, 100).unwrap()).into();
        adata.set_x(&x).unwrap();

        // Ensure data is cached first
        adata.x().enable_cache();
        let _ = adata.x().get::<ArrayData>().unwrap();
        assert!(adata.x().is_cached());

        let taken_x: ArrayData = adata.take_x().unwrap().unwrap();
        assert_eq!(taken_x, x);

        // Internal cache should now be empty
        assert!(!adata.x().is_cached());

        // Data should still be accessible on disk (reading it again works)
        let read_again = adata.x().get::<ArrayData>().unwrap().unwrap();
        assert_eq!(read_again, x);
    });
}

pub fn test_obsm_drain<B: Backend>() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata = AnnData::<B>::new(&file).unwrap();
        let x: ArrayData = Array::random((10, 50), Uniform::new(0, 100).unwrap()).into();
        let y: ArrayData = Array::random((10, 20), Uniform::new(0, 100).unwrap()).into();

        adata.obsm().add("x", &x).unwrap();
        adata.obsm().add("y", &y).unwrap();

        let drained: HashMap<String, ArrayData> = adata.obsm().drain().collect();

        assert_eq!(drained.len(), 2);
        assert_eq!(drained.get("x").unwrap(), &x);
        assert_eq!(drained.get("y").unwrap(), &y);

        // obsm should now be empty in the original object
        assert!(adata.obsm().keys().is_empty());
    });
}

pub fn test_backend_interop<B1: Backend, B2: Backend>() {
    with_tmp_dir(|dir| {
        let file1 = dir.join("test.h5ad");
        let file2 = dir.join("test.zarr");

        // 1. Create a complex dataset in Backend 1
        let adata1 = AnnData::<B1>::new(&file1).unwrap();
        let x: ArrayData = rand_csr::<f64>(50, 100, 20, 0.0, 1.0).into();
        adata1.set_x(&x).unwrap();

        let mut config_map = std::collections::HashMap::new();
        config_map.insert(
            "version".to_string(),
            Data::Scalar(anndata::data::DynScalar::I32(1)),
        );
        config_map.insert(
            "author".to_string(),
            Data::Scalar(anndata::data::DynScalar::String("ian".to_string())),
        );
        adata1
            .set_uns([("config".to_string(), Data::Mapping(config_map.into()))])
            .unwrap();

        let obsm_data: ArrayData = Array::random((50, 5), Uniform::new(0, 100).unwrap()).into();
        adata1.obsm().add("pca", &obsm_data).unwrap();

        // 2. Write it to Backend 2
        adata1.write::<B2, _>(&file2, None, None).unwrap();
        adata1.close().unwrap();

        // 3. Open Backend 2 and verify
        let adata2 = AnnData::<B2>::open(B2::open(&file2).unwrap()).unwrap();

        let x2 = adata2.x().get::<ArrayData>().unwrap().unwrap();
        assert_eq!(x, x2);

        let obsm2 = adata2.obsm().get_item::<ArrayData>("pca").unwrap().unwrap();
        assert_eq!(obsm_data, obsm2);

        assert_eq!(adata2.n_obs(), 50);
        assert_eq!(adata2.n_vars(), 100);
    });
}

pub fn test_uns_nesting<B: Backend>() {
    with_tmp_dir(|dir| {
        let file = dir.join("test_uns");
        let adata = AnnData::<B>::new(&file).unwrap();

        // Create deeply nested data
        let mut inner_map = std::collections::HashMap::new();
        inner_map.insert(
            "val".to_string(),
            Data::Scalar(anndata::data::DynScalar::I32(42)),
        );

        let mut middle_map = std::collections::HashMap::new();
        middle_map.insert("inner".to_string(), Data::Mapping(inner_map.into()));

        // Save to uns
        adata
            .uns()
            .add("config", Data::Mapping(middle_map.into()))
            .unwrap();

        // Close and reopen to test serialization
        adata.close().unwrap();
        let adata_read = AnnData::<B>::open(B::open(&file).unwrap()).unwrap();

        let read_back = adata_read
            .uns()
            .get_item::<Data>("config")
            .unwrap()
            .unwrap();

        if let Data::Mapping(middle) = read_back {
            if let Data::Mapping(inner) = middle.get("inner").unwrap() {
                if let Data::Scalar(anndata::data::DynScalar::I32(val)) = inner.get("val").unwrap()
                {
                    assert_eq!(*val, 42);
                    return;
                }
            }
        }
        panic!("Nested uns extraction failed to match structure");
    });
}
