use crate::data::utils::array_major_minor_index_default;
use crate::data::{ArrayData, Data, DataFrameIndex, DynArray, DynIndSparseMatrix, DynSparseMatrix};
use crate::{AnnDataOp, ArrayElemOp, AxisArraysOp, ElemCollectionOp, HasShape};
use anyhow::{Result, ensure};
use indexmap::IndexSet;
use itertools::Itertools;
use log::warn;
use polars::chunked_array::builder::CategoricalChunkedBuilder;
use polars::frame::DataFrame;
use polars::prelude::{AnyValue, Categorical32Type, Column, DataType, IntoLazy, NamedFrom};
use polars::series::{IntoSeries, Series};
use sprs::{CsMatI, SpIndex};

#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner,
    Outer,
}

/// Concatenate multiple AnnData objects into one.
///
/// This function concatenates multiple AnnData objects along the observation axis (`obs`),
/// aligning the variable axis (`var`) according to the specified join type (inner or outer).
/// It also concatenates associated data structures such as `obsm`, `obsp`, `layers`, and shared `uns` elements.
///
/// # Arguments
/// - `adatas`: A slice of AnnData objects to concatenate.
/// - `join`: The type of join to perform on the variables (`var`).
/// - `label`: An optional label for the keys column in `obs`.
/// - `keys`: An optional slice of keys to label each AnnData object in `obs`.
/// - `out`: The output AnnData object to store the concatenated result.
pub fn concat<A, O, S>(
    adatas: &[A],
    join: JoinType,
    label: Option<&str>,
    keys: Option<&[S]>,
    out: &O,
) -> Result<()>
where
    A: AnnDataOp,
    O: AnnDataOp,
    S: ToString,
{
    // Concatenate var_names
    let common_vars = adatas
        .iter()
        .map(|x| x.var_names().into_iter().collect::<IndexSet<_>>());
    let common_vars: IndexSet<String> = match join {
        JoinType::Inner => common_vars.reduce(|a, b| a.intersection(&b).cloned().collect()),
        JoinType::Outer => common_vars.reduce(|a, b| a.union(&b).cloned().collect()),
    }
    .unwrap();
    out.set_var_names(common_vars.iter().cloned().collect())?;

    // Concatenate vars
    {
        let df_var = adatas
            .iter()
            .map(|adata| {
                let var = adata.read_var().unwrap();
                let var_names = adata.var_names();
                // Creating the series
                let columns = var
                    .columns()
                    .iter()
                    .map(|s| align_series(s, &var_names, &common_vars))
                    .collect::<Result<Vec<_>>>()?;
                Ok(DataFrame::new_infer_height(columns)?)
            })
            .reduce(|a, b| {
                let mut a = a?;
                merge_df(&mut a, &b?)?;
                anyhow::Ok(a)
            })
            .unwrap()?;
        out.set_var(df_var)?;
    }

    // Concatenate obs
    {
        let obs_names = adatas.iter().flat_map(|adata| adata.obs_names()).collect();
        out.set_obs_names(obs_names)?;

        let mut dfs = adatas
            .iter()
            .map(|adata| adata.read_obs().unwrap())
            .collect::<Vec<_>>();
        if let Some(keys) = keys {
            dfs.iter_mut().zip_eq(keys.iter()).for_each(|(df, key)| {
                let s = Series::new(
                    label.unwrap_or("label").into(),
                    vec![key.to_string(); df.height()],
                );
                df.insert_column(0, s.into()).unwrap();
            });
        }
        let dfs = dfs.into_iter().map(|df| df.lazy()).collect::<Vec<_>>();
        let mut args = polars::prelude::UnionArgs::default();
        match join {
            JoinType::Inner => args.diagonal = false,
            JoinType::Outer => args.diagonal = true,
        };
        let df_obs = polars::prelude::concat(&dfs, args)?.collect()?;
        out.set_obs(df_obs)?;
    }

    // Concatenate X
    if adatas.iter().any(|adata| adata.x().is_none()) {
        warn!("Some AnnData objects have no X matrix. The concatenated X matrix will be None.");
    } else {
        out.set_x_from_iter(concat_x(adatas, &common_vars))?;
    }

    // Concatenate obsm
    {
        let obsm: Vec<_> = adatas.iter().map(|x| x.obsm()).collect();
        let common_keys = obsm
            .iter()
            .map(|x| x.keys().into_iter().collect::<IndexSet<_>>())
            .reduce(|a, b| a.intersection(&b).cloned().collect())
            .unwrap();
        for key in common_keys {
            let arr = concat_axis_arrays(&obsm, &key);
            out.obsm().add_iter(&key, arr)?;
        }
    }

    // Concatenate obsp
    {
        let obsp: Vec<_> = adatas.iter().map(|x| x.obsp()).collect();
        let common_keys = obsp
            .iter()
            .map(|x| x.keys().into_iter().collect::<IndexSet<_>>())
            .reduce(|a, b| a.intersection(&b).cloned().collect())
            .unwrap();
        for key in common_keys {
            let arr = concat_axis_arrays(&obsp, &key);
            out.obsp().add_iter(&key, arr)?;
        }
    }

    // Concat layers
    {
        let layers: Vec<_> = adatas.iter().map(|x| x.layers()).collect();
        let common_keys = layers
            .iter()
            .map(|x| x.keys().into_iter().collect::<IndexSet<_>>())
            .reduce(|a, b| a.intersection(&b).cloned().collect())
            .unwrap();
        for key in common_keys {
            let arr = concat_axis_arrays(&layers, &key);
            out.layers().add_iter(&key, arr)?;
        }
    }

    // Add shared uns elements.
    {
        let uns: Vec<_> = adatas.iter().map(|x| x.uns()).collect();
        let common_keys = uns
            .iter()
            .map(|x| x.keys().into_iter().collect::<IndexSet<_>>())
            .reduce(|a, b| a.intersection(&b).cloned().collect())
            .unwrap();
        for key in common_keys {
            if uns
                .iter()
                .map(|x| x.get_item::<Data>(&key).unwrap().unwrap())
                .all_equal()
            {
                out.uns()
                    .add(&key, uns.first().unwrap().get_item::<Data>(&key)?.unwrap())?;
            }
        }
    }

    Ok(())
}

fn merge_df(this: &mut DataFrame, other: &DataFrame) -> Result<()> {
    if other.height() == 0 {
        return Ok(());
    }
    ensure!(
        this.height() == other.height(),
        "DataFrames must have the same number of rows"
    );
    other.columns().iter().try_for_each(|other_s| {
        let name = other_s.name();
        if let Some(i) = this.get_column_index(name) {
            let this_s = this.column(name)?;
            let new_column = this_s
                .as_series()
                .unwrap()
                .iter()
                .zip(other_s.as_series().unwrap().iter())
                .map(|(this_v, other_v)| {
                    if other_v.is_null() {
                        this_v.clone()
                    } else {
                        other_v.clone()
                    }
                })
                .collect::<Vec<_>>();
            let dtype = match (this_s.dtype(), other_s.dtype()) {
                (DataType::Categorical(_, _), _) => this_s.dtype(),
                (_, DataType::Categorical(_, _)) => other_s.dtype(),
                _ => this_s.dtype(),
            };
            let new_column = match dtype {
                DataType::Categorical(_, _) => {
                    let mut builder: CategoricalChunkedBuilder<Categorical32Type> =
                        CategoricalChunkedBuilder::new(name.clone(), dtype.clone());
                    new_column.iter().for_each(|x| {
                        if let Some(x) = x.get_str() {
                            builder.append_str(x).unwrap();
                        } else {
                            builder.append_null();
                        }
                    });
                    builder.finish().into_series()
                }
                _ => Series::from_any_values_and_dtype(name.clone(), &new_column, dtype, false)?,
            };
            this.replace_column(i, new_column.into())?;
        } else {
            this.insert_column(this.width(), other_s.clone())?;
        }
        anyhow::Ok(())
    })?;
    Ok(())
}

/// Reorganize a column to match the new row names, filling in missing values with `None`.
fn align_series(
    series: &Column,
    row_names: &DataFrameIndex,
    new_row_names: &IndexSet<String>,
) -> Result<Column> {
    let name = series.name();
    let dtype = series.dtype();
    let new_series = match dtype {
        DataType::Categorical(_, _) => {
            let mut builder: CategoricalChunkedBuilder<Categorical32Type> =
                CategoricalChunkedBuilder::new(name.clone(), dtype.clone());
            new_row_names.iter().for_each(|key| {
                let item = row_names.get_index(key).map(|i| series.get(i).unwrap());
                if let Some(s) = item.as_ref().and_then(|x| x.get_str()) {
                    builder.append_str(s).unwrap();
                } else {
                    builder.append_null();
                }
            });
            builder.finish().into_series()
        }
        _ => {
            let values: Result<Vec<_>> = new_row_names
                .iter()
                .map(|key| {
                    if let Some(i) = row_names.get_index(key) {
                        Ok(series.get(i)?)
                    } else {
                        Ok(AnyValue::Null)
                    }
                })
                .collect();
            Series::from_any_values_and_dtype(name.clone(), &values?, dtype, false)?
        }
    };
    Ok(new_series.into())
}

fn index_array(
    arr: ArrayData,
    row_indices: &[Option<usize>],
    col_indices: &[Option<usize>],
) -> ArrayData {
    macro_rules! fun_array {
        ($variant:ident, $value:expr) => {
            array_major_minor_index_default(
                row_indices,
                col_indices,
                &$value.into_dimensionality().unwrap(),
            )
            .into()
        };
    }

    macro_rules! fun_csr {
        ($variant:ident, $value:expr) => {{ DynSparseMatrix::$variant(index_csr(row_indices, col_indices, &$value)) }};
    }

    match arr {
        ArrayData::Array(x) => crate::macros::dyn_map!(x, DynArray, fun_array),
        ArrayData::CsrMatrix(x) => match x {
            DynIndSparseMatrix::I16(x) => {
                DynIndSparseMatrix::I16(crate::macros::dyn_map!(x, DynSparseMatrix, fun_csr)).into()
            }
            DynIndSparseMatrix::I32(x) => {
                DynIndSparseMatrix::I32(crate::macros::dyn_map!(x, DynSparseMatrix, fun_csr)).into()
            }
            DynIndSparseMatrix::I64(x) => {
                DynIndSparseMatrix::I64(crate::macros::dyn_map!(x, DynSparseMatrix, fun_csr)).into()
            }
            DynIndSparseMatrix::U16(x) => {
                DynIndSparseMatrix::U16(crate::macros::dyn_map!(x, DynSparseMatrix, fun_csr)).into()
            }
            DynIndSparseMatrix::U32(x) => {
                DynIndSparseMatrix::U32(crate::macros::dyn_map!(x, DynSparseMatrix, fun_csr)).into()
            }
            DynIndSparseMatrix::U64(x) => {
                DynIndSparseMatrix::U64(crate::macros::dyn_map!(x, DynSparseMatrix, fun_csr)).into()
            }
        },
        _ => todo!(),
    }
}

fn index_csr<N: Clone, T: SpIndex>(
    row_indices: &[Option<usize>],
    col_indices: &[Option<usize>],
    matrix: &CsMatI<N, T, u64>,
) -> CsMatI<N, T, u64> {
    debug_assert!(matrix.is_csr());

    // Map each source column to its position(s) in the aligned output. Missing
    // output columns have no source entry and therefore remain all-zero.
    let mut col_mapping = vec![Vec::new(); matrix.cols()];
    for (output_col, source_col) in col_indices.iter().enumerate() {
        if let Some(source_col) = source_col {
            col_mapping[*source_col].push(output_col);
        }
    }

    let matrix_indptr = matrix.indptr();
    let matrix_indptr = matrix_indptr.as_slice().unwrap();
    let mut output_indptr = Vec::with_capacity(row_indices.len() + 1);
    let mut output_indices = Vec::new();
    let mut output_data = Vec::new();
    let mut row = Vec::new();
    output_indptr.push(0);

    for source_row in row_indices {
        if let Some(source_row) = source_row {
            let start = matrix_indptr[*source_row] as usize;
            let end = matrix_indptr[*source_row + 1] as usize;
            row.clear();

            for i in start..end {
                let source_col = matrix.indices()[i].to_usize().unwrap();
                for &output_col in &col_mapping[source_col] {
                    row.push((output_col, matrix.data()[i].clone()));
                }
            }

            // Variable orders may differ between inputs, so restore canonical
            // CSR ordering after remapping the column indices.
            row.sort_by_key(|(col, _)| *col);
            for (col, value) in row.drain(..) {
                output_indices.push(T::from_usize(col));
                output_data.push(value);
            }
        }
        output_indptr.push(output_indices.len() as u64);
    }

    CsMatI::new(
        (row_indices.len(), col_indices.len()),
        output_indptr,
        output_indices,
        output_data,
    )
}

fn concat_x<A: AnnDataOp>(
    adatas: &[A],
    common_vars: &IndexSet<String>,
) -> impl Iterator<Item = ArrayData> {
    adatas.iter().map(move |adata| {
        let var_names = adata.var_names();
        let arr = adata.x().get().unwrap().unwrap();
        index_array(
            arr,
            &(0..adata.n_obs()).map(Some).collect::<Vec<_>>(),
            &common_vars
                .iter()
                .map(|x| var_names.get_index(x))
                .collect::<Vec<_>>(),
        )
    })
}

fn concat_axis_arrays<A: AxisArraysOp>(
    axis_arrays: &[A],
    key: &str,
) -> impl Iterator<Item = ArrayData> {
    let size = axis_arrays[0].get(key).unwrap().shape().unwrap()[1];
    axis_arrays.iter().map(move |arr| {
        let arr: ArrayData = arr.get_item(key).unwrap().unwrap();
        assert_eq!(arr.shape()[1], size, "dimension mismatch for key: {key}");
        arr
    })
}
