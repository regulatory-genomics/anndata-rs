mod common;
pub use common::*;

use anndata::backend::{DataContainer, GroupOp};
use anndata::concat::{JoinType, concat};
use anndata::data::{DataFrameIndex, SelectInfoElem};
use anndata::{data::CsrNonCanonical, *};
use data::ArrayConvert;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::Array2;
use polars::df;
use proptest::prelude::*;

pub fn test_basic<B: Backend>() {
    with_tmp_dir(|dir| {
        let ann1 = AnnData::<B>::new(dir.join("test1")).unwrap();
        let csc = rand_csc::<i32>(10, 5, 3, 1, 100);
        ann1.obsm().add("csc", &csc).unwrap();
        assert!(ann1.obsm().get_item::<CsrMatrix<i32>>("csc").is_err());

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
    let coo: CooMatrix<i32> = CooMatrix::try_from_triplets(
        5,
        4,
        vec![0, 1, 1, 1, 2, 3, 4],
        vec![0, 0, 0, 2, 3, 1, 3],
        vec![1, 2, 3, 4, 5, 6, 7],
    )
    .unwrap();
    adata.set_x(CsrNonCanonical::from(&coo)).unwrap();
    assert!(adata.x().get::<CsrMatrix<i32>>().is_err());
    adata.x().get::<CsrNonCanonical<i32>>().unwrap().unwrap();
    adata.x().get::<ArrayData>().unwrap().unwrap();
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

/// Dataframes stored in obsm/varm must be indexed by the obs/var names, as
/// polars dataframes carry no index of their own.
pub fn test_dataframe_index_set_before<B: Backend>() {
    fn index_on_disk<B: Backend>(path: &std::path::Path, key: &str) -> DataFrameIndex {
        let store = B::open(path).unwrap();
        let container = DataContainer::<B>::open(&store.open_group("obsm").unwrap(), key).unwrap();
        let index = DataFrameElem::<B>::try_from(container)
            .unwrap()
            .inner()
            .index
            .clone();
        index
    }

    with_tmp_dir(|dir| {
        let names: DataFrameIndex = ["a", "b", "c"].iter().map(|x| x.to_string()).collect();
        let df = df!("x" => [1i32, 2, 3]).unwrap();

        // Names known before the dataframe is written
        let path = dir.join("before");
        let adata = AnnData::<B>::new(&path).unwrap();
        adata.set_obs_names(names.clone()).unwrap();
        adata.obsm().add("df", df.clone()).unwrap();
        adata.close().unwrap();
        assert_eq!(index_on_disk::<B>(&path, "df"), names);

        // Names set after the dataframe is written
        let path = dir.join("after");
        let adata = AnnData::<B>::new(&path).unwrap();
        adata.obsm().add("df", df.clone()).unwrap();
        adata.set_obs_names(names.clone()).unwrap();
        adata.close().unwrap();
        assert_eq!(index_on_disk::<B>(&path, "df"), names);

        // Subsetting keeps the index in sync
        let path = dir.join("subset");
        let adata = AnnData::<B>::new(&path).unwrap();
        adata.set_obs_names(names.clone()).unwrap();
        adata.obsm().add("df", df).unwrap();
        adata
            .subset([SelectInfoElem::from(vec![0, 2]), SelectInfoElem::full()])
            .unwrap();
        adata.close().unwrap();
        assert_eq!(
            index_on_disk::<B>(&path, "df").into_vec(),
            ["a", "c"].map(String::from)
        );
    })
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
