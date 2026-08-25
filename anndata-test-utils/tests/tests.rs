use anndata::{AnnData, Backend};
use anndata_hdf5::H5;
use anndata_test_utils as utils;
use anndata_test_utils::with_tmp_dir;
use anndata_zarr::Zarr;

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

        let file = dir.join("test.zarr");
        adata.write::<Zarr, _>(file, None, None).unwrap();
    })
}

#[test]
fn test_dataframe_index() {
    utils::test_dataframe_index::<H5>();
    utils::test_dataframe_index::<Zarr>();
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

        //let file = dir.join("test.zarr");
        //let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        //utils::test_io(|| adata_gen());
    })
}

#[test]
fn test_index() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_index(adata_gen);

        //let file = dir.join("test.zarr");
        //let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        //utils::test_index(|| adata_gen());
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
