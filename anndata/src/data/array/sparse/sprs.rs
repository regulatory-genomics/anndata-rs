use std::collections::HashMap;

use anyhow::bail;
use ndarray::Ix1;
use num::ToPrimitive;
use rayon::prelude::*;
use sprs::{CompressedStorage, CsMatI, SpIndex};

use crate::{
    HasShape, Readable, ReadableArray, Selectable, Writable, WritableArray,
    backend::{AttributeOp, Backend, BackendData, DataType, DatasetOp, GroupOp},
    data::{DynArray, Element, MetaData, SelectInfoElem, SelectInfoElemBounds, Shape, Stackable},
};

fn read_indices<B: Backend, D: DatasetOp<B>, T: BackendData + SpIndex + num::FromPrimitive>(
    dataset: &D,
) -> anyhow::Result<Vec<T>> {
    if dataset.dtype()? == T::DTYPE {
        Ok(dataset.read_array::<T, Ix1>()?.into_raw_vec_and_offset().0)
    } else {
        match dataset.read_dyn_array()? {
            DynArray::I8(arr) => Ok(arr.into_iter().map(|x| T::from_i8(x).unwrap()).collect()),
            DynArray::I16(arr) => Ok(arr.into_iter().map(|x| T::from_i16(x).unwrap()).collect()),
            DynArray::I32(arr) => Ok(arr.into_iter().map(|x| T::from_i32(x).unwrap()).collect()),
            DynArray::I64(arr) => Ok(arr.into_iter().map(|x| T::from_i64(x).unwrap()).collect()),
            DynArray::U8(arr) => Ok(arr.into_iter().map(|x| T::from_u8(x).unwrap()).collect()),
            DynArray::U16(arr) => Ok(arr.into_iter().map(|x| T::from_u16(x).unwrap()).collect()),
            DynArray::U32(arr) => Ok(arr.into_iter().map(|x| T::from_u32(x).unwrap()).collect()),
            DynArray::U64(arr) => Ok(arr.into_iter().map(|x| T::from_u64(x).unwrap()).collect()),
            _ => bail!("Unsupported index type"),
        }
    }
}

#[allow(dead_code)]
fn read_indices_slice<
    B: Backend,
    D: DatasetOp<B>,
    T: BackendData + SpIndex + num::FromPrimitive,
    S: AsRef<SelectInfoElem>,
>(
    dataset: &D,
    selection: &[S],
) -> anyhow::Result<Vec<T>> {
    if dataset.dtype()? == T::DTYPE {
        Ok(dataset
            .read_array_slice::<T, _, Ix1>(selection)?
            .into_raw_vec_and_offset()
            .0)
    } else {
        match dataset.read_dyn_array_slice(selection)? {
            DynArray::I8(arr) => Ok(arr.into_iter().map(|x| T::from_i8(x).unwrap()).collect()),
            DynArray::I16(arr) => Ok(arr.into_iter().map(|x| T::from_i16(x).unwrap()).collect()),
            DynArray::I32(arr) => Ok(arr.into_iter().map(|x| T::from_i32(x).unwrap()).collect()),
            DynArray::I64(arr) => Ok(arr.into_iter().map(|x| T::from_i64(x).unwrap()).collect()),
            DynArray::U8(arr) => Ok(arr.into_iter().map(|x| T::from_u8(x).unwrap()).collect()),
            DynArray::U16(arr) => Ok(arr.into_iter().map(|x| T::from_u16(x).unwrap()).collect()),
            DynArray::U32(arr) => Ok(arr.into_iter().map(|x| T::from_u32(x).unwrap()).collect()),
            DynArray::U64(arr) => Ok(arr.into_iter().map(|x| T::from_u64(x).unwrap()).collect()),
            _ => bail!("Unsupported index type"),
        }
    }
}

fn try_new_checked_parallel<N, T: SpIndex>(
    storage: CompressedStorage,
    shape: (usize, usize),
    indptr: Vec<u64>,
    indices: Vec<T>,
    data: Vec<N>,
) -> anyhow::Result<CsMatI<N, T, u64>> {
    let (inner, outer) = match storage {
        CompressedStorage::CSR => (shape.1, shape.0),
        CompressedStorage::CSC => (shape.0, shape.1),
    };

    if data.len() != indices.len() {
        bail!("data and indices have different sizes");
    }
    let expected_indptr_len = outer
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("sparse matrix outer dimension is too large"))?;
    if indptr.len() != expected_indptr_len {
        bail!("indptr length does not match sparse matrix shape");
    }
    if T::try_from_usize(inner).is_none() {
        bail!("index type is not large enough for this matrix");
    }

    let max_indptr = u64::try_from(usize::MAX / 2).unwrap_or(u64::MAX);
    if indptr.is_empty() || indptr.iter().any(|&x| x > max_indptr) {
        bail!("indptr contains values out of range for this platform");
    }
    if !indptr.windows(2).all(|x| x[0] <= x[1]) {
        bail!("indptr values are not sorted");
    }

    let offset = indptr[0];
    let nnz = indptr[indptr.len() - 1] - offset;
    if nnz as usize != indices.len() {
        bail!("indices length and indptr nnz do not match");
    }

    indptr.par_windows(2).try_for_each(|range| {
        let start = (range[0] - offset) as usize;
        let end = (range[1] - offset) as usize;
        let mut prev = None;
        for &idx in &indices[start..end] {
            let idx_usize = idx
                .try_index()
                .ok_or_else(|| anyhow::anyhow!("index value out of range"))?;
            if idx_usize >= inner {
                bail!("index value is larger than sparse matrix inner dimension");
            }
            if let Some(prev) = prev
                && idx <= prev
            {
                bail!("sparse matrix indices are not sorted");
            }
            prev = Some(idx);
        }
        Ok(())
    })?;

    // SAFETY: all invariants checked above mirror sprs' compressed structure
    // validation: matching data/index lengths, valid indptr length and monotonicity,
    // representable indices, in-bounds strictly sorted inner indices per outer slice.
    Ok(unsafe { CsMatI::new_unchecked(storage, shape, indptr, indices, data) })
}

impl<N, T: SpIndex> HasShape for CsMatI<N, T, u64> {
    fn shape(&self) -> Shape {
        let rows = self.rows();
        let cols = self.cols();
        vec![rows, cols].into()
    }
}

impl<N: BackendData, T: BackendData + SpIndex> Element for CsMatI<N, T, u64> {
    fn data_type(&self) -> crate::backend::DataType {
        if self.is_csr() {
            DataType::CsrMatrix(N::DTYPE, T::DTYPE)
        } else {
            DataType::CscMatrix(N::DTYPE, T::DTYPE)
        }
    }

    fn metadata(&self) -> crate::data::MetaData {
        let mut metadata = HashMap::new();

        metadata.insert("shape".to_string(), HasShape::shape(self).into());
        let mtx_type = if self.is_csc() {
            "csc_matrix"
        } else {
            "csr_matrix"
        };
        MetaData::new(mtx_type, "0.1.0", Some(metadata))
    }
}

impl<N: BackendData, T: BackendData + SpIndex + ToPrimitive + num::Integer + num::FromPrimitive>
    Selectable for CsMatI<N, T, u64>
{
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<crate::data::SelectInfoElem>,
    {
        let is_csr = self.is_csr();
        let rows = self.rows();
        let cols = self.cols();

        let row_bounds = SelectInfoElemBounds::new(info[0].as_ref(), rows);
        let full = SelectInfoElem::full();
        let col_bounds = if info.len() > 1 {
            SelectInfoElemBounds::new(info[1].as_ref(), cols)
        } else {
            SelectInfoElemBounds::new(&full, cols)
        };

        if is_csr {
            // CSR Selection
            let indptr_obj = self.indptr();
            let indptr_slice = indptr_obj.as_slice().unwrap();
            let mut new_indptr = Vec::with_capacity(row_bounds.len() + 1);
            let mut temp_indices = Vec::new();
            let mut temp_data = Vec::new();
            new_indptr.push(0);

            if col_bounds.is_full(cols) {
                for i in 0..row_bounds.len() {
                    let row_idx = row_bounds.index(i);
                    let start = indptr_slice[row_idx] as usize;
                    let end = indptr_slice[row_idx + 1] as usize;
                    temp_indices.extend_from_slice(&self.indices()[start..end]);
                    temp_data.extend_from_slice(&self.data()[start..end]);
                    new_indptr.push(temp_indices.len() as u64);
                }
                return CsMatI::try_new(
                    (row_bounds.len(), cols),
                    new_indptr,
                    temp_indices,
                    temp_data,
                )
                .unwrap();
            }

            // Check if minor axis selection is monotonically increasing
            let is_monotonic = col_bounds
                .iter()
                .zip(col_bounds.iter().skip(1))
                .all(|(a, b)| a < b);

            // Use SmallVec to avoid heap allocations for unique indices
            let mut col_lookup: HashMap<usize, smallvec::SmallVec<[usize; 1]>> = HashMap::new();
            for (i, x) in col_bounds.iter().enumerate() {
                col_lookup.entry(x).or_default().push(i);
            }

            let mut row_workspace: Vec<(usize, N)> = Vec::new();

            for i in 0..row_bounds.len() {
                let row_idx = row_bounds.index(i);
                let start = indptr_slice[row_idx] as usize;
                let end = indptr_slice[row_idx + 1] as usize;

                if is_monotonic {
                    // Fast path: skip sorting
                    for j in start..end {
                        let col_idx = self.indices()[j].to_usize().unwrap();
                        if let Some(new_col_indices) = col_lookup.get(&col_idx) {
                            for &new_col_idx in new_col_indices {
                                temp_indices.push(SpIndex::from_usize(new_col_idx));
                                temp_data.push(self.data()[j].clone());
                            }
                        }
                    }
                } else {
                    // Slow path: out-of-order or duplicate selection requires sorting
                    row_workspace.clear();
                    for j in start..end {
                        let col_idx = self.indices()[j].to_usize().unwrap();
                        if let Some(new_col_indices) = col_lookup.get(&col_idx) {
                            for &new_col_idx in new_col_indices {
                                row_workspace.push((new_col_idx, self.data()[j].clone()));
                            }
                        }
                    }
                    row_workspace.sort_by_key(|x| x.0);
                    for (new_col_idx, val) in &row_workspace {
                        temp_indices.push(SpIndex::from_usize(*new_col_idx));
                        temp_data.push(val.clone());
                    }
                }
                new_indptr.push(temp_indices.len() as u64);
            }
            CsMatI::try_new(
                (row_bounds.len(), col_bounds.len()),
                new_indptr,
                temp_indices,
                temp_data,
            )
            .unwrap()
        } else {
            // CSC Selection
            let indptr_obj = self.indptr();
            let indptr_slice = indptr_obj.as_slice().unwrap();
            let mut new_indptr = Vec::with_capacity(col_bounds.len() + 1);
            let mut temp_indices = Vec::new();
            let mut temp_data = Vec::new();
            new_indptr.push(0);

            if row_bounds.is_full(rows) {
                for i in 0..col_bounds.len() {
                    let col_idx = col_bounds.index(i);
                    let start = indptr_slice[col_idx] as usize;
                    let end = indptr_slice[col_idx + 1] as usize;
                    temp_indices.extend_from_slice(&self.indices()[start..end]);
                    temp_data.extend_from_slice(&self.data()[start..end]);
                    new_indptr.push(temp_indices.len() as u64);
                }
                return CsMatI::try_new_csc(
                    (rows, col_bounds.len()),
                    new_indptr,
                    temp_indices,
                    temp_data,
                )
                .unwrap();
            }

            // Check if minor axis selection is monotonically increasing
            let is_monotonic = row_bounds
                .iter()
                .zip(row_bounds.iter().skip(1))
                .all(|(a, b)| a < b);

            // Use SmallVec to avoid heap allocations for unique indices
            let mut row_lookup: HashMap<usize, smallvec::SmallVec<[usize; 1]>> = HashMap::new();
            for (i, x) in row_bounds.iter().enumerate() {
                row_lookup.entry(x).or_default().push(i);
            }

            let mut col_workspace: Vec<(usize, N)> = Vec::new();

            for i in 0..col_bounds.len() {
                let col_idx = col_bounds.index(i);
                let start = indptr_slice[col_idx] as usize;
                let end = indptr_slice[col_idx + 1] as usize;

                if is_monotonic {
                    // Fast path: skip sorting
                    for j in start..end {
                        let row_idx = self.indices()[j].to_usize().unwrap();
                        if let Some(new_row_indices) = row_lookup.get(&row_idx) {
                            for &new_row_idx in new_row_indices {
                                temp_indices.push(SpIndex::from_usize(new_row_idx));
                                temp_data.push(self.data()[j].clone());
                            }
                        }
                    }
                } else {
                    // Slow path: out-of-order or duplicate selection requires sorting
                    col_workspace.clear();
                    for j in start..end {
                        let row_idx = self.indices()[j].to_usize().unwrap();
                        if let Some(new_row_indices) = row_lookup.get(&row_idx) {
                            for &new_row_idx in new_row_indices {
                                col_workspace.push((new_row_idx, self.data()[j].clone()));
                            }
                        }
                    }
                    col_workspace.sort_by_key(|x| x.0);
                    for (new_row_idx, val) in &col_workspace {
                        temp_indices.push(SpIndex::from_usize(*new_row_idx));
                        temp_data.push(val.clone());
                    }
                }
                new_indptr.push(temp_indices.len() as u64);
            }
            CsMatI::try_new_csc(
                (row_bounds.len(), col_bounds.len()),
                new_indptr,
                temp_indices,
                temp_data,
            )
            .unwrap()
        }
    }
}

impl<N: BackendData + std::fmt::Debug, T: BackendData + SpIndex + num::Integer + num::FromPrimitive>
    Stackable for CsMatI<N, T, u64>
{
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> anyhow::Result<Self> {
        let mut iter = iter.peekable();
        let first = iter.peek().ok_or(anyhow::anyhow!("Empty iterator"))?;
        let is_csr = first.is_csr();
        let cols = first.cols();

        if !is_csr {
            bail!("vstack is only implemented for CSR matrices");
        }

        let mut new_rows = 0;
        let mut new_indptr = Vec::new();
        let mut new_indices = Vec::new();
        let mut new_data = Vec::new();
        let mut current_nnz: u64 = 0;
        new_indptr.push(0);

        for m in iter {
            if !m.is_csr() {
                bail!("Cannot vstack matrices with different layouts (CSR vs CSC)");
            }
            if m.cols() != cols {
                bail!("Cannot vstack matrices with different number of columns");
            }
            new_rows += m.rows();

            let indptr = m.indptr();
            let indptr_slice = indptr.as_slice().unwrap();
            new_indptr.reserve(indptr_slice.len().saturating_sub(1));
            new_indices.reserve(m.nnz());
            new_data.reserve(m.nnz());
            for &p in &indptr_slice[1..] {
                new_indptr.push(current_nnz + p);
            }
            current_nnz += m.nnz() as u64;
            new_indices.extend_from_slice(m.indices());
            new_data.extend_from_slice(m.data());
        }

        Ok(CsMatI::new(
            (new_rows, cols),
            new_indptr,
            new_indices,
            new_data,
        ))
    }
}

impl<N: BackendData, T: BackendData + SpIndex> Writable for CsMatI<N, T, u64> {
    fn write<B: crate::Backend, G: crate::backend::GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> anyhow::Result<crate::backend::DataContainer<B>> {
        let mut group = location.new_group(name)?;

        self.metadata().save(&mut group)?;
        group.new_array_dataset("data", self.data().into(), Default::default())?;

        let indptr = self.indptr();
        group.new_array_dataset(
            "indptr",
            indptr.as_slice().unwrap().into(),
            Default::default(),
        )?;
        let min_idx = self.indices();
        group.new_array_dataset("indices", min_idx.into(), Default::default())?;
        Ok(crate::backend::DataContainer::Group(group))
    }
}

impl<N: BackendData, T: BackendData + SpIndex + num::FromPrimitive> Readable for CsMatI<N, T, u64> {
    fn read<B: crate::Backend>(container: &crate::backend::DataContainer<B>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let group = container.as_group()?;
        let encoding: String = group.get_attr("encoding-type")?;
        let is_csr = match encoding.as_str() {
            "csr_matrix" => true,
            "csc_matrix" => false,
            _ => bail!("Cannot read CSMatI from container of type {encoding}"),
        };

        let shape: Vec<u64> = group.get_attr("shape")?;
        if shape.len() != 2 {
            bail!("sparse matrix shape must have length 2");
        }
        let data_ds = group.open_dataset("data")?;
        let indptr_ds = group.open_dataset("indptr")?;
        let indices_ds = group.open_dataset("indices")?;

        let (data, (indptr, indices)) = rayon::join(
            || {
                data_ds
                    .read_array::<N, Ix1>()
                    .map(|x| x.into_raw_vec_and_offset().0)
            },
            || {
                rayon::join(
                    || read_indices::<B, _, u64>(&indptr_ds),
                    || read_indices::<B, _, T>(&indices_ds),
                )
            },
        );

        let storage = if is_csr {
            CompressedStorage::CSR
        } else {
            CompressedStorage::CSC
        };
        try_new_checked_parallel(
            storage,
            (shape[0] as usize, shape[1] as usize),
            indptr?,
            indices?,
            data?,
        )
    }
}

impl<N: BackendData, T: BackendData + SpIndex + ToPrimitive + num::Integer + num::FromPrimitive>
    ReadableArray for CsMatI<N, T, u64>
{
    fn get_shape<B: crate::Backend>(
        container: &crate::backend::DataContainer<B>,
    ) -> anyhow::Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    fn read_select<B, S>(
        container: &crate::backend::DataContainer<B>,
        info: &[S],
    ) -> anyhow::Result<Self>
    where
        B: crate::Backend,
        S: AsRef<crate::data::SelectInfoElem>,
        Self: Sized,
    {
        use rayon::prelude::*;

        fn contiguous_runs(bounds: &SelectInfoElemBounds<'_>) -> (usize, Vec<(usize, usize)>) {
            if let SelectInfoElemBounds::Slice(slice) = bounds
                && slice.step == 1
            {
                let len = slice.end - slice.start;
                let runs = if len == 0 {
                    Vec::new()
                } else {
                    vec![(slice.start, slice.end)]
                };
                return (len, runs);
            }

            let mut indices = bounds.iter();
            let len = indices.len();
            let Some(mut start) = indices.next() else {
                return (0, Vec::new());
            };
            let mut prev = start;
            let mut runs = Vec::new();
            for idx in indices {
                if idx == prev + 1 {
                    prev = idx;
                } else {
                    runs.push((start, prev + 1));
                    start = idx;
                    prev = idx;
                }
            }
            runs.push((start, prev + 1));
            (len, runs)
        }

        let data_type = container.encoding_type()?;

        match data_type {
            DataType::CsrMatrix(_, _) | DataType::CscMatrix(_, _) => {
                let selection = [
                    info.first()
                        .map(|x| x.as_ref().clone())
                        .unwrap_or_else(SelectInfoElem::full),
                    info.get(1)
                        .map(|x| x.as_ref().clone())
                        .unwrap_or_else(SelectInfoElem::full),
                ];

                if selection.iter().all(SelectInfoElem::is_full) {
                    return Self::read(container);
                }

                let shape = Self::get_shape(container)?;
                let major_axis = if let DataType::CsrMatrix(_, _) = data_type {
                    0
                } else {
                    1
                };
                let minor_axis = 1 - major_axis;
                let major_bounds =
                    SelectInfoElemBounds::new(&selection[major_axis], shape[major_axis]);
                let (major_len, runs) = contiguous_runs(&major_bounds);
                let minor_len = shape[minor_axis];

                let res = if runs.is_empty() {
                    if major_axis == 0 {
                        CsMatI::try_new((0, minor_len), vec![0], Vec::new(), Vec::new())
                            .map_err(|(_, _, _, e)| anyhow::anyhow!("Cannot read csr matrix {e}"))?
                    } else {
                        CsMatI::try_new_csc((minor_len, 0), vec![0], Vec::new(), Vec::new())
                            .map_err(|(_, _, _, e)| anyhow::anyhow!("Cannot read csc matrix {e}"))?
                    }
                } else {
                    let group = container.as_group()?;
                    let indptr_ds = group.open_dataset("indptr")?;
                    let data_ds = group.open_dataset("data")?;
                    let indices_ds = group.open_dataset("indices")?;

                    type SparseRun<N, T> = (Vec<u64>, Vec<T>, Vec<N>);
                    let run_results: Vec<anyhow::Result<SparseRun<N, T>>> = runs
                        .par_iter()
                        .map(|&(start, end)| {
                            let indptr = indptr_ds
                                .read_array_slice::<u64, _, Ix1>(&[SelectInfoElem::from(
                                    start..end + 1,
                                )])?
                                .into_raw_vec_and_offset()
                                .0;
                            let nnz_start = indptr[0] as usize;
                            let nnz_end = *indptr.last().unwrap() as usize;
                            let nnz_slice = [SelectInfoElem::from(nnz_start..nnz_end)];
                            let (data, indices) = rayon::join(
                                || {
                                    data_ds
                                        .read_array_slice::<N, _, Ix1>(&nnz_slice)
                                        .map(|x| x.into_raw_vec_and_offset().0)
                                },
                                || {
                                    indices_ds
                                        .read_array_slice::<T, _, Ix1>(&nnz_slice)
                                        .map(|x| x.into_raw_vec_and_offset().0)
                                },
                            );
                            Ok((indptr, indices?, data?))
                        })
                        .collect();
                    let run_parts: Vec<(Vec<u64>, Vec<T>, Vec<N>)> =
                        run_results.into_iter().collect::<anyhow::Result<_>>()?;

                    let total_nnz: usize =
                        run_parts.iter().map(|(_, indices, _)| indices.len()).sum();
                    let mut new_indptr = Vec::with_capacity(major_len + 1);
                    let mut new_indices = Vec::with_capacity(total_nnz);
                    let mut new_data = Vec::with_capacity(total_nnz);
                    let mut current_nnz = 0_u64;
                    new_indptr.push(0);

                    for (indptr, indices, data) in run_parts {
                        let base = indptr[0];
                        for offset in &indptr[1..] {
                            new_indptr.push(current_nnz + offset - base);
                        }
                        current_nnz += indptr.last().copied().unwrap_or(base) - base;
                        new_indices.extend(indices);
                        new_data.extend(data);
                    }

                    if major_axis == 0 {
                        CsMatI::try_new((major_len, minor_len), new_indptr, new_indices, new_data)
                            .map_err(|(_, _, _, e)| anyhow::anyhow!("Cannot read csr matrix {e}"))?
                    } else {
                        CsMatI::try_new_csc(
                            (minor_len, major_len),
                            new_indptr,
                            new_indices,
                            new_data,
                        )
                        .map_err(|(_, _, _, e)| anyhow::anyhow!("Cannot read csc matrix {e}"))?
                    }
                };

                if selection[minor_axis].is_full() {
                    Ok(res)
                } else {
                    let mut minor_only = [SelectInfoElem::full(), SelectInfoElem::full()];
                    minor_only[minor_axis] = selection[minor_axis].clone();
                    Ok(res.select(&minor_only))
                }
            }
            _ => bail!("Cannot read sparse matrix from group."),
        }
    }
}

impl<N: BackendData, T: BackendData + SpIndex> WritableArray for &CsMatI<N, T, u64> {}
impl<N: BackendData, T: BackendData + SpIndex> WritableArray for CsMatI<N, T, u64> {}

impl<N: BackendData, T: BackendData + SpIndex> crate::data::data_traits::Indexable
    for CsMatI<N, T, u64>
{
    fn get(&self, index: &[usize]) -> Option<crate::data::DynScalar> {
        self.get(index[0], index[1]).map(|v| v.as_dyn())
    }
}

impl<N: BackendData + ToPrimitive, T: BackendData + SpIndex>
    crate::data::data_traits::ArrayArithmetic for CsMatI<N, T, u64>
{
    fn sum(&self) -> f64 {
        self.data().iter().map(|x| x.to_f64().unwrap()).sum()
    }

    fn sum_axis(&self, axis: usize) -> anyhow::Result<ndarray::ArrayD<f64>> {
        let rows = self.rows();
        let cols = self.cols();
        let indptr_obj = self.indptr();
        let indptr_slice = indptr_obj.as_slice().unwrap();
        if self.is_csr() {
            if axis == 0 {
                let mut col_sums = vec![0.0; cols];
                for i in 0..rows {
                    let start = indptr_slice[i] as usize;
                    let end = indptr_slice[i + 1] as usize;
                    for (&col, val) in self.indices()[start..end]
                        .iter()
                        .zip(&self.data()[start..end])
                    {
                        col_sums[col.to_usize().unwrap()] += val.to_f64().unwrap();
                    }
                }
                Ok(ndarray::Array1::from_vec(col_sums).into_dyn())
            } else if axis == 1 {
                let row_sums: Vec<f64> = (0..rows)
                    .map(|i| {
                        let start = indptr_slice[i] as usize;
                        let end = indptr_slice[i + 1] as usize;
                        self.data()[start..end]
                            .iter()
                            .map(|x| x.to_f64().unwrap())
                            .sum()
                    })
                    .collect();
                Ok(ndarray::Array1::from_vec(row_sums).into_dyn())
            } else {
                bail!("Axis {axis} out of bounds for 2D matrix");
            }
        } else {
            if axis == 0 {
                let col_sums: Vec<f64> = (0..cols)
                    .map(|i| {
                        let start = indptr_slice[i] as usize;
                        let end = indptr_slice[i + 1] as usize;
                        self.data()[start..end]
                            .iter()
                            .map(|x| x.to_f64().unwrap())
                            .sum()
                    })
                    .collect();
                Ok(ndarray::Array1::from_vec(col_sums).into_dyn())
            } else if axis == 1 {
                let mut row_sums = vec![0.0; rows];
                for i in 0..cols {
                    let start = indptr_slice[i] as usize;
                    let end = indptr_slice[i + 1] as usize;
                    for (&row, val) in self.indices()[start..end]
                        .iter()
                        .zip(&self.data()[start..end])
                    {
                        row_sums[row.to_usize().unwrap()] += val.to_f64().unwrap();
                    }
                }
                Ok(ndarray::Array1::from_vec(row_sums).into_dyn())
            } else {
                bail!("Axis {axis} out of bounds for 2D matrix");
            }
        }
    }

    fn min(&self) -> f64 {
        self.data()
            .iter()
            .map(|x| x.to_f64().unwrap())
            .fold(f64::INFINITY, f64::min)
    }

    fn max(&self) -> f64 {
        self.data()
            .iter()
            .map(|x| x.to_f64().unwrap())
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

impl<N, T: SpIndex> crate::data::data_traits::SparseMatrixLayout for CsMatI<N, T, u64> {
    fn get_sparse_layout(&self) -> crate::data::SparseMatrixLayoutE {
        if self.is_csr() {
            crate::data::SparseMatrixLayoutE::CSR
        } else {
            crate::data::SparseMatrixLayoutE::CSC
        }
    }
}

#[cfg(test)]
mod sparse_tests {
    use super::*;
    use sprs::CsMatI;

    #[test]
    fn test_vstack() {
        let m1: CsMatI<f64, u32, u64> =
            CsMatI::new((2, 3), vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]);
        let m2: CsMatI<f64, u32, u64> = CsMatI::new((1, 3), vec![0, 1], vec![2], vec![3.0]);

        let stacked = CsMatI::vstack(vec![m1.clone(), m2.clone()].into_iter()).unwrap();
        assert_eq!(stacked.rows(), 3);
        assert_eq!(stacked.cols(), 3);
        assert_eq!(stacked.nnz(), 3);
        assert_eq!(stacked.indptr().as_slice().unwrap(), &[0, 1, 2, 3]);
        assert_eq!(stacked.indices(), &[0, 1, 2]);
        assert_eq!(stacked.data(), &[1.0, 2.0, 3.0]);

        // Verify values
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(stacked.get(i, j), m1.get(i, j));
            }
        }
        for j in 0..3 {
            assert_eq!(stacked.get(2, j), m2.get(0, j));
        }
    }
}
