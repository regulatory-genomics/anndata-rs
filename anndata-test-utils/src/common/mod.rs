use anndata::*;

use anndata::data::index::Interval;
use anndata::data::{DataFrameIndex, SelectInfoElem};
use anyhow::Result;
use itertools::Itertools;
use nalgebra::{ClosedAddAssign, Scalar};
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Zero;
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;
use rand::seq::IteratorRandom;
use sprs::{CsMatI, TriMatI};
use std::path::{Path, PathBuf};
use tempfile::tempdir;

pub fn with_tmp_dir<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    func(path)
}

////////////////////////////////////////////////////////////////////////////////
// Strategies
////////////////////////////////////////////////////////////////////////////////

/// Strategy for generating a random AnnData
pub fn anndata_strat<B: Backend, P: AsRef<Path> + Clone>(
    file: P,
    n_obs: usize,
    n_vars: usize,
) -> impl Strategy<Value = AnnData<B>> {
    let x = array_strat(&[n_obs, n_vars]);
    let obs_names = index_strat(n_obs);
    let obsm = proptest::collection::vec(0_usize..100, 0..3).prop_flat_map(move |shapes| {
        shapes
            .into_iter()
            .map(|d| array_strat(&[n_obs, d]))
            .collect::<Vec<_>>()
    });
    let obsp = (0_usize..3).prop_flat_map(move |d| {
        std::iter::repeat_with(|| array_strat(&[n_obs, n_obs]))
            .take(d)
            .collect::<Vec<_>>()
    });
    let var_names = index_strat(n_vars);
    let varm = proptest::collection::vec(0_usize..100, 0..3).prop_flat_map(move |shapes| {
        shapes
            .into_iter()
            .map(|d| array_strat(&[n_vars, d]))
            .collect::<Vec<_>>()
    });
    let varp = (0_usize..3).prop_flat_map(move |d| {
        std::iter::repeat_with(|| array_strat(&[n_vars, n_vars]))
            .take(d)
            .collect::<Vec<_>>()
    });
    let layers = (0_usize..3).prop_flat_map(move |d| {
        std::iter::repeat_with(|| array_strat(&[n_obs, n_vars]))
            .take(d)
            .collect::<Vec<_>>()
    });
    (x, obs_names, obsm, obsp, var_names, varm, varp, layers).prop_map(
        move |(x, obs_names, obsm, obsp, var_names, varm, varp, layers)| {
            let adata: AnnData<B> = AnnData::new(file.clone()).unwrap();
            adata.set_x(x).unwrap();
            adata.set_obs_names(obs_names).unwrap();
            adata.set_var_names(var_names).unwrap();
            obsm.into_iter().enumerate().for_each(|(i, arr)| {
                adata.obsm().add(&format!("varm_{i}"), arr).unwrap();
            });
            obsp.into_iter().enumerate().for_each(|(i, arr)| {
                adata.obsp().add(&format!("obsp_{i}"), arr).unwrap();
            });
            varm.into_iter().enumerate().for_each(|(i, arr)| {
                adata.varm().add(&format!("varm_{i}"), arr).unwrap();
            });
            varp.into_iter().enumerate().for_each(|(i, arr)| {
                adata.varp().add(&format!("varp_{i}"), arr).unwrap();
            });
            layers.into_iter().enumerate().for_each(|(i, arr)| {
                adata.layers().add(&format!("layer_{i}"), arr).unwrap();
            });
            adata
        },
    )
}

pub fn index_strat(n: usize) -> BoxedStrategy<DataFrameIndex> {
    if n == 0 {
        Just(DataFrameIndex::empty()).boxed()
    } else {
        let list = (0..n).map(|i| format!("i_{i}")).collect();
        let range = n.into();
        let interval = (0..n).prop_flat_map(move |i| {
            (Just(i), (0..n - i)).prop_flat_map(move |(a, b)| {
                let c = n - a - b;
                [a, b, c]
                    .into_iter()
                    .filter(|x| *x != 0)
                    .map(interval_strat)
                    .collect::<Vec<_>>()
                    .prop_map(move |x| {
                        x.into_iter()
                            .enumerate()
                            .map(|(i, x)| (i.to_string(), x))
                            .collect::<DataFrameIndex>()
                    })
            })
        });
        prop_oneof![Just(list), Just(range), interval].boxed()
    }
}

fn interval_strat(n: usize) -> impl Strategy<Value = Interval> {
    (1_usize..100, 1_usize..100).prop_map(move |(size, step)| Interval {
        start: 0,
        end: n * step,
        size,
        step,
    })
}

pub fn array_slice_strat(
    shape: &[usize],
) -> impl Strategy<Value = (ArrayData, Vec<SelectInfoElem>)> {
    array_strat(shape).prop_flat_map(|x| {
        let select = x
            .shape()
            .as_ref()
            .iter()
            .map(|&s| select_strat(s))
            .collect::<Vec<_>>();
        (Just(x), select)
    })
}

pub fn array_strat(shape: &[usize]) -> impl Strategy<Value = ArrayData> {
    prop_oneof![
        csr_strat(shape[0], shape[1]),
        csc_strat(shape[0], shape[1]),
        dense_array_strat(shape),
    ]
}

/// Strategy for generating a random CsrMatrix
pub fn csr_strat(num_rows: usize, num_cols: usize) -> impl Strategy<Value = ArrayData> {
    let max = num_rows * num_cols;
    let nnz = max / 10;
    prop_oneof![
        Just(rand_csr::<u8>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csr::<u16>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csr::<u32>(num_rows, num_cols, nnz, 1, 2550).into()),
        Just(rand_csr::<u64>(num_rows, num_cols, nnz, 1, 25500).into()),
        Just(rand_csr::<i8>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csr::<i16>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csr::<i32>(num_rows, num_cols, nnz, 1, 2550).into()),
        Just(rand_csr::<i64>(num_rows, num_cols, nnz, 1, 25500).into()),
        Just(rand_csr::<f32>(num_rows, num_cols, nnz, 1.0, 255.0).into()),
        Just(rand_csr::<f64>(num_rows, num_cols, nnz, 1.0, 255.0).into()),
    ]
}

/// Strategy for generating a random CscMatrix
pub fn csc_strat(num_rows: usize, num_cols: usize) -> impl Strategy<Value = ArrayData> {
    let max = num_rows * num_cols;
    let nnz = max / 10;
    prop_oneof![
        Just(rand_csc::<u8>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csc::<u16>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csc::<u32>(num_rows, num_cols, nnz, 1, 2550).into()),
        Just(rand_csc::<u64>(num_rows, num_cols, nnz, 1, 25500).into()),
        Just(rand_csc::<i8>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csc::<i16>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csc::<i32>(num_rows, num_cols, nnz, 1, 2550).into()),
        Just(rand_csc::<i64>(num_rows, num_cols, nnz, 1, 25500).into()),
        Just(rand_csc::<f32>(num_rows, num_cols, nnz, 1.0, 255.0).into()),
        Just(rand_csc::<f64>(num_rows, num_cols, nnz, 1.0, 255.0).into()),
    ]
}

fn dense_array_strat(shape: &[usize]) -> impl Strategy<Value = ArrayData> {
    let shape = shape.to_vec();
    let s: Vec<_> = shape.clone().into_iter().rev().collect();
    prop_oneof![
        Just(Array::random(shape.clone(), Uniform::new(0u8, 255u8).unwrap()).into()),
        Just(Array::random(shape.clone(), Uniform::new(0u16, 255u16).unwrap()).into()),
        Just(Array::random(shape.clone(), Uniform::new(0u32, 255u32).unwrap()).into()),
        Just(Array::random(shape.clone(), Uniform::new(0u64, 255u64).unwrap()).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128i8, 127i8).unwrap()).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128i16, 127i16).unwrap()).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128i32, 127i32).unwrap()).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128i64, 127i64).unwrap()).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128f32, 127f32).unwrap()).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128f64, 127f64).unwrap()).into()),
        Just(
            Array::random(shape.clone(), Uniform::new(0u8, 1u8).unwrap())
                .mapv(|x| x == 1)
                .into()
        ),
        Just(
            Array::random(shape.clone(), Uniform::new(-1000f32, 1000f32).unwrap())
                .mapv(|x| x.to_string())
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(0u8, 255u8).unwrap())
                .reversed_axes()
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(0u16, 255u16).unwrap())
                .reversed_axes()
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(0u32, 255u32).unwrap())
                .reversed_axes()
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(0u64, 255u64).unwrap())
                .reversed_axes()
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(-128i8, 127i8).unwrap())
                .reversed_axes()
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(-128i16, 127i16).unwrap())
                .reversed_axes()
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(-128i32, 127i32).unwrap())
                .reversed_axes()
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(-128i64, 127i64).unwrap())
                .reversed_axes()
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(-128f32, 127f32).unwrap())
                .reversed_axes()
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(-128f64, 127f64).unwrap())
                .reversed_axes()
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(0u8, 1u8).unwrap())
                .mapv(|x| x == 1)
                .reversed_axes()
                .into()
        ),
        Just(
            Array::random(s.clone(), Uniform::new(-1000f32, 1000f32).unwrap())
                .mapv(|x| x.to_string())
                .reversed_axes()
                .into()
        ),
    ]
}

/// Generate a random compressed sparse row matrix
pub fn rand_csr<T>(nrow: usize, ncol: usize, nnz: usize, low: T, high: T) -> CsMatI<T, i64, u64>
where
    T: Scalar + Zero + ClosedAddAssign + SampleUniform + PartialOrd,
{
    let mut rng = rand::rng();
    let values: Vec<T> = Array::random((nnz,), Uniform::new(low, high).unwrap()).to_vec();
    let (row_indices, col_indices): (Vec<usize>, Vec<usize>) = (0..nrow)
        .cartesian_product(0..ncol)
        .choose_multiple(&mut rng, nnz)
        .into_iter()
        .unzip();
    let row_indices = row_indices.into_iter().map(|x| x as i64).collect();
    let col_indices = col_indices.into_iter().map(|x| x as i64).collect();
    let coo = TriMatI::from_triplets((nrow, ncol), row_indices, col_indices, values);
    let csr = coo.to_csr();
    CsMatI::new(
        csr.shape(),
        csr.indptr()
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x: &usize| x as u64)
            .collect(),
        csr.indices().to_vec(),
        csr.data().to_vec(),
    )
}

/// Generate a random compressed sparse column matrix
pub fn rand_csc<T>(nrow: usize, ncol: usize, nnz: usize, low: T, high: T) -> CsMatI<T, i64, u64>
where
    T: Scalar + Zero + ClosedAddAssign + SampleUniform + PartialOrd,
{
    let mut rng = rand::rng();
    let values: Vec<T> = Array::random((nnz,), Uniform::new(low, high).unwrap()).to_vec();
    let (row_indices, col_indices): (Vec<usize>, Vec<usize>) = (0..nrow)
        .cartesian_product(0..ncol)
        .choose_multiple(&mut rng, nnz)
        .into_iter()
        .unzip();
    let row_indices = row_indices.into_iter().map(|x| x as i64).collect();
    let col_indices = col_indices.into_iter().map(|x| x as i64).collect();
    let coo = TriMatI::from_triplets((nrow, ncol), row_indices, col_indices, values);
    let csc = coo.to_csc();
    CsMatI::new_csc(
        csc.shape(),
        csc.indptr()
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x: &usize| x as u64)
            .collect(),
        csc.indices().to_vec(),
        csc.data().to_vec(),
    )
}

pub fn select_strat(n: usize) -> BoxedStrategy<SelectInfoElem> {
    if n == 0 {
        Just(Vec::new().into()).boxed()
    } else {
        let indices = proptest::collection::vec(0..n, 0..2 * n).prop_map(|i| i.into());
        let slice = (0..n).prop_flat_map(move |start| {
            (Just(start), (start + 1)..=n).prop_map(|(start, stop)| (start..stop).into())
        });
        prop_oneof![indices, slice,].boxed()
    }
}

////////////////////////////////////////////////////////////////////////////////
// AnnData operations
////////////////////////////////////////////////////////////////////////////////

pub fn anndata_eq<B1: Backend, B2: Backend>(
    adata1: &AnnData<B1>,
    adata2: &AnnData<B2>,
) -> Result<bool> {
    let is_equal = adata1.n_obs() == adata2.n_obs()
        && adata1.n_vars() == adata2.n_vars()
        && adata1.obs_names() == adata2.obs_names()
        && adata1.var_names() == adata2.var_names()
        && {
            let a = adata1.read_obs()?;
            let b = adata2.read_obs()?;
            a == b || (a.height() == 0 && b.height() == 0)
        }
        && {
            let a = adata1.read_var()?;
            let b = adata2.read_var()?;
            a == b || (a.height() == 0 && b.height() == 0)
        }
        && adata1.x().get::<ArrayData>()? == adata2.x().get()?
        && adata1.obsm().keys().iter().all(|k| {
            adata1.obsm().get_item::<ArrayData>(k).unwrap() == adata2.obsm().get_item(k).unwrap()
        })
        && adata1.obsp().keys().iter().all(|k| {
            adata1.obsp().get_item::<ArrayData>(k).unwrap() == adata2.obsp().get_item(k).unwrap()
        })
        && adata1.varm().keys().iter().all(|k| {
            adata1.varm().get_item::<ArrayData>(k).unwrap() == adata2.varm().get_item(k).unwrap()
        })
        && adata1.varp().keys().iter().all(|k| {
            adata1.varp().get_item::<ArrayData>(k).unwrap() == adata2.varp().get_item(k).unwrap()
        })
        && adata1.uns().keys().iter().all(|k| {
            adata1.uns().get_item::<Data>(k).unwrap() == adata2.uns().get_item(k).unwrap()
        })
        && adata1.layers().keys().iter().all(|k| {
            adata1.layers().get_item::<ArrayData>(k).unwrap()
                == adata2.layers().get_item(k).unwrap()
        });
    Ok(is_equal)
}

////////////////////////////////////////////////////////////////////////////////
// Array operations
////////////////////////////////////////////////////////////////////////////////

pub fn array_select(arr: &ArrayData, select: &[SelectInfoElem]) -> ArrayData {
    arr.select(select)
}

pub fn array_chunks(
    arr: &ArrayData,
    chunk_size: usize,
) -> Box<dyn Iterator<Item = ArrayData> + '_> {
    let nrow = arr.shape()[0];
    Box::new((0..nrow).step_by(chunk_size).map(move |i| {
        let j = (i + chunk_size).min(nrow);
        arr.select(&[SelectInfoElem::from(i..j), SelectInfoElem::full()])
    }))
}
