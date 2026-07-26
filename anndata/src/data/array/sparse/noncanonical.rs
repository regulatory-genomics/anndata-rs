use std::collections::HashMap;

use crate::backend::*;
use crate::data::{
    SelectInfoBounds,
    array::utils::{coo_to_unsorted_cs, cs_major_minor_index},
    data_traits::*,
    slice::{SelectInfoElem, Shape},
};

use anyhow::{Result, bail};
use ndarray::{ArrayD, Ix1};
use num::ToPrimitive;
use sprs::{CsMatI, TriMatI};

use super::DynSparseMatrix;

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum DynIndCsrNonCanonical {
    I16(DynCsrNonCanonical<i16>),
    I32(DynCsrNonCanonical<i32>),
    I64(DynCsrNonCanonical<i64>),
    U16(DynCsrNonCanonical<u16>),
    U32(DynCsrNonCanonical<u32>),
    U64(DynCsrNonCanonical<u64>),
}

impl Readable for DynIndCsrNonCanonical {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => {
                let indices_dtype = group.open_dataset("indices")?.dtype()?;
                macro_rules! fun {
                    ($type:ident, $variant:ident) => {
                        DynCsrNonCanonical::<$type>::read(container)
                            .map(DynIndCsrNonCanonical::$variant)
                    };
                }
                crate::macros::dyn_index_match!(indices_dtype, ScalarType, fun)
            }
            _ => bail!("Can't read sparse matrix from non-group container"),
        }
    }
}

impl HasShape for DynIndCsrNonCanonical {
    fn shape(&self) -> Shape {
        crate::macros::dyn_index_map_fun!(self, DynIndCsrNonCanonical, shape)
    }
}

impl Selectable for DynIndCsrNonCanonical {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        crate::macros::dyn_index_sparse_map!(self, DynIndCsrNonCanonical, select, info)
    }
}

impl Element for DynIndCsrNonCanonical {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_index_map_fun!(self, DynIndCsrNonCanonical, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_index_map_fun!(self, DynIndCsrNonCanonical, metadata)
    }
}

impl Writable for DynIndCsrNonCanonical {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_index_map_fun!(self, DynIndCsrNonCanonical, write, location, name)
    }
}

impl ReadableArray for DynIndCsrNonCanonical {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        let indices_dtype = container.as_group()?.open_dataset("indices")?.dtype()?;
        macro_rules! fun {
            ($type:ident, $variant:ident) => {
                DynCsrNonCanonical::<$type>::read_select(container, info)
                    .map(DynIndCsrNonCanonical::$variant)
            };
        }
        crate::macros::dyn_index_match!(indices_dtype, ScalarType, fun)
    }
}

impl WritableArray for DynIndCsrNonCanonical {}

#[derive(Debug, Clone, PartialEq)]
pub enum DynCsrNonCanonical<I = u64> {
    I8(CsrNonCanonical<i8, I>),
    I16(CsrNonCanonical<i16, I>),
    I32(CsrNonCanonical<i32, I>),
    I64(CsrNonCanonical<i64, I>),
    U8(CsrNonCanonical<u8, I>),
    U16(CsrNonCanonical<u16, I>),
    U32(CsrNonCanonical<u32, I>),
    U64(CsrNonCanonical<u64, I>),
    F32(CsrNonCanonical<f32, I>),
    F64(CsrNonCanonical<f64, I>),
    Bool(CsrNonCanonical<bool, I>),
    String(CsrNonCanonical<String, I>),
}

impl<I: sprs::SpIndex + BackendData + num::FromPrimitive + num::Integer> DynCsrNonCanonical<I> {
    pub fn canonicalize(self) -> Result<DynSparseMatrix<I>, Self> {
        match self {
            DynCsrNonCanonical::I8(data) => data
                .canonicalize()
                .map(DynSparseMatrix::I8)
                .map_err(Into::into),
            DynCsrNonCanonical::I16(data) => data
                .canonicalize()
                .map(DynSparseMatrix::I16)
                .map_err(Into::into),
            DynCsrNonCanonical::I32(data) => data
                .canonicalize()
                .map(DynSparseMatrix::I32)
                .map_err(Into::into),
            DynCsrNonCanonical::I64(data) => data
                .canonicalize()
                .map(DynSparseMatrix::I64)
                .map_err(Into::into),
            DynCsrNonCanonical::U8(data) => data
                .canonicalize()
                .map(DynSparseMatrix::U8)
                .map_err(Into::into),
            DynCsrNonCanonical::U16(data) => data
                .canonicalize()
                .map(DynSparseMatrix::U16)
                .map_err(Into::into),
            DynCsrNonCanonical::U32(data) => data
                .canonicalize()
                .map(DynSparseMatrix::U32)
                .map_err(Into::into),
            DynCsrNonCanonical::U64(data) => data
                .canonicalize()
                .map(DynSparseMatrix::U64)
                .map_err(Into::into),
            DynCsrNonCanonical::F32(data) => data
                .canonicalize()
                .map(DynSparseMatrix::F32)
                .map_err(Into::into),
            DynCsrNonCanonical::F64(data) => data
                .canonicalize()
                .map(DynSparseMatrix::F64)
                .map_err(Into::into),
            DynCsrNonCanonical::Bool(data) => data
                .canonicalize()
                .map(DynSparseMatrix::Bool)
                .map_err(Into::into),
            DynCsrNonCanonical::String(data) => data
                .canonicalize()
                .map(DynSparseMatrix::String)
                .map_err(Into::into),
        }
    }
}

macro_rules! impl_noncanonicalcsr_traits {
    ($($from_type:ty, $to_type:ident),*) => {
        $(
            impl<I: sprs::SpIndex + BackendData + num::FromPrimitive> From<CsrNonCanonical<$from_type, I>> for DynCsrNonCanonical<I> {
                fn from(data: CsrNonCanonical<$from_type, I>) -> Self {
                    DynCsrNonCanonical::$to_type(data)
                }
            }
            impl<I: sprs::SpIndex + BackendData + num::FromPrimitive> TryFrom<DynCsrNonCanonical<I>> for CsrNonCanonical<$from_type, I> {
                type Error = anyhow::Error;
                fn try_from(data: DynCsrNonCanonical<I>) -> Result<Self> {
                    if let DynCsrNonCanonical::$to_type(data) = data {
                        Ok(data)
                    } else {
                        bail!("cannot convert {:?} to CsrNonCanonical<{}>", data.data_type(), stringify!($from_type));
                    }
                }
            }
        )*
    };
}

impl_noncanonicalcsr_traits!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

impl<I: sprs::SpIndex + BackendData + num::FromPrimitive> Element for DynCsrNonCanonical<I> {
    fn data_type(&self) -> DataType {
        match self {
            DynCsrNonCanonical::I8(data) => data.data_type(),
            DynCsrNonCanonical::I16(data) => data.data_type(),
            DynCsrNonCanonical::I32(data) => data.data_type(),
            DynCsrNonCanonical::I64(data) => data.data_type(),
            DynCsrNonCanonical::U8(data) => data.data_type(),
            DynCsrNonCanonical::U16(data) => data.data_type(),
            DynCsrNonCanonical::U32(data) => data.data_type(),
            DynCsrNonCanonical::U64(data) => data.data_type(),
            DynCsrNonCanonical::F32(data) => data.data_type(),
            DynCsrNonCanonical::F64(data) => data.data_type(),
            DynCsrNonCanonical::Bool(data) => data.data_type(),
            DynCsrNonCanonical::String(data) => data.data_type(),
        }
    }

    fn metadata(&self) -> MetaData {
        match self {
            DynCsrNonCanonical::I8(data) => data.metadata(),
            DynCsrNonCanonical::I16(data) => data.metadata(),
            DynCsrNonCanonical::I32(data) => data.metadata(),
            DynCsrNonCanonical::I64(data) => data.metadata(),
            DynCsrNonCanonical::U8(data) => data.metadata(),
            DynCsrNonCanonical::U16(data) => data.metadata(),
            DynCsrNonCanonical::U32(data) => data.metadata(),
            DynCsrNonCanonical::U64(data) => data.metadata(),
            DynCsrNonCanonical::F32(data) => data.metadata(),
            DynCsrNonCanonical::F64(data) => data.metadata(),
            DynCsrNonCanonical::Bool(data) => data.metadata(),
            DynCsrNonCanonical::String(data) => data.metadata(),
        }
    }
}

impl<I: sprs::SpIndex + BackendData + num::FromPrimitive> Writable for DynCsrNonCanonical<I> {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        match self {
            DynCsrNonCanonical::I8(data) => data.write(location, name),
            DynCsrNonCanonical::I16(data) => data.write(location, name),
            DynCsrNonCanonical::I32(data) => data.write(location, name),
            DynCsrNonCanonical::I64(data) => data.write(location, name),
            DynCsrNonCanonical::U8(data) => data.write(location, name),
            DynCsrNonCanonical::U16(data) => data.write(location, name),
            DynCsrNonCanonical::U32(data) => data.write(location, name),
            DynCsrNonCanonical::U64(data) => data.write(location, name),
            DynCsrNonCanonical::F32(data) => data.write(location, name),
            DynCsrNonCanonical::F64(data) => data.write(location, name),
            DynCsrNonCanonical::Bool(data) => data.write(location, name),
            DynCsrNonCanonical::String(data) => data.write(location, name),
        }
    }
}

impl<I: sprs::SpIndex + BackendData + num::FromPrimitive> Readable for DynCsrNonCanonical<I> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($variant:ident) => {
                        CsrNonCanonical::<$variant, I>::read(container).map(Into::into)
                    };
                }
                crate::macros::dyn_match!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            }
            _ => bail!("cannot read csr matrix from non-group container"),
        }
    }
}

impl<I: sprs::SpIndex + BackendData + num::FromPrimitive> HasShape for DynCsrNonCanonical<I> {
    fn shape(&self) -> Shape {
        match self {
            DynCsrNonCanonical::I8(data) => data.shape(),
            DynCsrNonCanonical::I16(data) => data.shape(),
            DynCsrNonCanonical::I32(data) => data.shape(),
            DynCsrNonCanonical::I64(data) => data.shape(),
            DynCsrNonCanonical::U8(data) => data.shape(),
            DynCsrNonCanonical::U16(data) => data.shape(),
            DynCsrNonCanonical::U32(data) => data.shape(),
            DynCsrNonCanonical::U64(data) => data.shape(),
            DynCsrNonCanonical::F32(data) => data.shape(),
            DynCsrNonCanonical::F64(data) => data.shape(),
            DynCsrNonCanonical::Bool(data) => data.shape(),
            DynCsrNonCanonical::String(data) => data.shape(),
        }
    }
}

impl<I: sprs::SpIndex + BackendData + num::FromPrimitive + num::Integer> Selectable
    for DynCsrNonCanonical<I>
{
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        match self {
            DynCsrNonCanonical::I8(data) => data.select(info).into(),
            DynCsrNonCanonical::I16(data) => data.select(info).into(),
            DynCsrNonCanonical::I32(data) => data.select(info).into(),
            DynCsrNonCanonical::I64(data) => data.select(info).into(),
            DynCsrNonCanonical::U8(data) => data.select(info).into(),
            DynCsrNonCanonical::U16(data) => data.select(info).into(),
            DynCsrNonCanonical::U32(data) => data.select(info).into(),
            DynCsrNonCanonical::U64(data) => data.select(info).into(),
            DynCsrNonCanonical::F32(data) => data.select(info).into(),
            DynCsrNonCanonical::F64(data) => data.select(info).into(),
            DynCsrNonCanonical::Bool(data) => data.select(info).into(),
            DynCsrNonCanonical::String(data) => data.select(info).into(),
        }
    }
}

impl<I: sprs::SpIndex + BackendData + num::FromPrimitive> Stackable for DynCsrNonCanonical<I> {
    fn vstack<I2: Iterator<Item = Self>>(iter: I2) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynCsrNonCanonical::I8(_) => Ok(DynCsrNonCanonical::I8(
                CsrNonCanonical::<i8, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::I16(_) => Ok(DynCsrNonCanonical::I16(
                CsrNonCanonical::<i16, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::I32(_) => Ok(DynCsrNonCanonical::I32(
                CsrNonCanonical::<i32, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::I64(_) => Ok(DynCsrNonCanonical::I64(
                CsrNonCanonical::<i64, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::U8(_) => Ok(DynCsrNonCanonical::U8(
                CsrNonCanonical::<u8, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::U16(_) => Ok(DynCsrNonCanonical::U16(
                CsrNonCanonical::<u16, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::U32(_) => Ok(DynCsrNonCanonical::U32(
                CsrNonCanonical::<u32, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::U64(_) => Ok(DynCsrNonCanonical::U64(
                CsrNonCanonical::<u64, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::F32(_) => Ok(DynCsrNonCanonical::F32(
                CsrNonCanonical::<f32, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::F64(_) => Ok(DynCsrNonCanonical::F64(
                CsrNonCanonical::<f64, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::Bool(_) => Ok(DynCsrNonCanonical::Bool(
                CsrNonCanonical::<bool, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::String(_) => {
                Ok(DynCsrNonCanonical::String(
                    CsrNonCanonical::<String, I>::vstack(iter.map(|x| x.try_into().unwrap()))?,
                ))
            }
        }
    }
}

impl<I: sprs::SpIndex + BackendData + num::FromPrimitive + num::Integer> ReadableArray
    for DynCsrNonCanonical<I>
{
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        if let DataType::CsrMatrix(_ty, tp) = container.encoding_type()? {
            macro_rules! fun {
                ($variant:ident) => {
                    CsrNonCanonical::<$variant, I>::read_select(container, info).map(Into::into)
                };
            }
            crate::macros::dyn_match!(tp, ScalarType, fun)
        } else {
            bail!("the container does not contain a csr matrix");
        }
    }
}

impl<I: sprs::SpIndex + BackendData + num::FromPrimitive> WritableArray for DynCsrNonCanonical<I> {}

impl<I: sprs::SpIndex + BackendData + num::FromPrimitive + num::Integer> ArrayArithmetic
    for DynCsrNonCanonical<I>
{
    fn sum(&self) -> f64 {
        match self {
            DynCsrNonCanonical::I8(data) => data.sum(),
            DynCsrNonCanonical::I16(data) => data.sum(),
            DynCsrNonCanonical::I32(data) => data.sum(),
            DynCsrNonCanonical::I64(data) => data.sum(),
            DynCsrNonCanonical::U8(data) => data.sum(),
            DynCsrNonCanonical::U16(data) => data.sum(),
            DynCsrNonCanonical::U32(data) => data.sum(),
            DynCsrNonCanonical::U64(data) => data.sum(),
            DynCsrNonCanonical::F32(data) => data.sum(),
            DynCsrNonCanonical::F64(data) => data.sum(),
            DynCsrNonCanonical::Bool(_) => panic!("Cannot compute sum for Bool sparse matrix"),
            DynCsrNonCanonical::String(_) => panic!("Cannot compute sum for String sparse matrix"),
        }
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayD<f64>> {
        match self {
            DynCsrNonCanonical::I8(data) => data.sum_axis(axis),
            DynCsrNonCanonical::I16(data) => data.sum_axis(axis),
            DynCsrNonCanonical::I32(data) => data.sum_axis(axis),
            DynCsrNonCanonical::I64(data) => data.sum_axis(axis),
            DynCsrNonCanonical::U8(data) => data.sum_axis(axis),
            DynCsrNonCanonical::U16(data) => data.sum_axis(axis),
            DynCsrNonCanonical::U32(data) => data.sum_axis(axis),
            DynCsrNonCanonical::U64(data) => data.sum_axis(axis),
            DynCsrNonCanonical::F32(data) => data.sum_axis(axis),
            DynCsrNonCanonical::F64(data) => data.sum_axis(axis),
            DynCsrNonCanonical::Bool(_) => bail!("Cannot compute sum for Bool sparse matrix"),
            DynCsrNonCanonical::String(_) => bail!("Cannot compute sum for String sparse matrix"),
        }
    }

    fn min(&self) -> f64 {
        match self {
            DynCsrNonCanonical::I8(data) => data.min(),
            DynCsrNonCanonical::I16(data) => data.min(),
            DynCsrNonCanonical::I32(data) => data.min(),
            DynCsrNonCanonical::I64(data) => data.min(),
            DynCsrNonCanonical::U8(data) => data.min(),
            DynCsrNonCanonical::U16(data) => data.min(),
            DynCsrNonCanonical::U32(data) => data.min(),
            DynCsrNonCanonical::U64(data) => data.min(),
            DynCsrNonCanonical::F32(data) => data.min(),
            DynCsrNonCanonical::F64(data) => data.min(),
            DynCsrNonCanonical::Bool(_) => panic!("Cannot compute min for Bool sparse matrix"),
            DynCsrNonCanonical::String(_) => panic!("Cannot compute min for String sparse matrix"),
        }
    }

    fn max(&self) -> f64 {
        match self {
            DynCsrNonCanonical::I8(data) => data.max(),
            DynCsrNonCanonical::I16(data) => data.max(),
            DynCsrNonCanonical::I32(data) => data.max(),
            DynCsrNonCanonical::I64(data) => data.max(),
            DynCsrNonCanonical::U8(data) => data.max(),
            DynCsrNonCanonical::U16(data) => data.max(),
            DynCsrNonCanonical::U32(data) => data.max(),
            DynCsrNonCanonical::U64(data) => data.max(),
            DynCsrNonCanonical::F32(data) => data.max(),
            DynCsrNonCanonical::F64(data) => data.max(),
            DynCsrNonCanonical::Bool(_) => panic!("Cannot compute max for Bool sparse matrix"),
            DynCsrNonCanonical::String(_) => panic!("Cannot compute max for String sparse matrix"),
        }
    }
}

/// Compressed sparse row matrix with potentially duplicate column indices.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrNonCanonical<N, I = u64> {
    offsets: Vec<u64>,
    indices: Vec<I>,
    values: Vec<N>,
    num_rows: usize,
    num_cols: usize,
}

impl<N, I: sprs::SpIndex> CsrNonCanonical<N, I> {
    pub fn nrows(&self) -> usize {
        self.num_rows
    }

    pub fn ncols(&self) -> usize {
        self.num_cols
    }

    pub fn row_offsets(&self) -> &[u64] {
        &self.offsets
    }

    pub fn col_indices(&self) -> &[I] {
        &self.indices
    }

    pub fn values(&self) -> &[N] {
        &self.values
    }

    pub fn csr_data(&self) -> (&[u64], &[I], &[N]) {
        (&self.offsets, &self.indices, &self.values)
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn disassemble(self) -> (Vec<u64>, Vec<I>, Vec<N>) {
        (self.offsets, self.indices, self.values)
    }

    pub fn from_csr_data(
        num_rows: usize,
        num_cols: usize,
        row_offsets: Vec<u64>,
        col_indices: Vec<I>,
        data: Vec<N>,
    ) -> Self {
        Self {
            offsets: row_offsets,
            indices: col_indices,
            values: data,
            num_rows,
            num_cols,
        }
    }

    pub fn canonicalize(self) -> Result<CsMatI<N, I, u64>, Self>
    where
        I: num::Integer,
    {
        let nrows = self.nrows();
        let ncols = self.ncols();
        if crate::data::utils::check_format(nrows, ncols, self.row_offsets(), self.col_indices())
            .is_ok()
        {
            Ok(CsMatI::new(
                (nrows, ncols),
                self.offsets,
                self.indices,
                self.values,
            ))
        } else {
            Err(self)
        }
    }
}

impl<N: Clone, Ix: sprs::SpIndex> From<CsMatI<N, Ix, u64>> for CsrNonCanonical<N, Ix> {
    fn from(csr: CsMatI<N, Ix, u64>) -> Self {
        assert!(csr.is_csr());
        let num_rows = csr.rows();
        let num_cols = csr.cols();
        let (offsets, indices, data) = csr.into_raw_storage();
        Self::from_csr_data(num_rows, num_cols, offsets, indices, data)
    }
}

impl<N: Clone, Ix: sprs::SpIndex> From<&TriMatI<N, Ix>> for CsrNonCanonical<N, Ix> {
    fn from(coo: &TriMatI<N, Ix>) -> Self {
        let nnz = coo.nnz();
        let mut offsets = vec![0_u64; coo.rows() + 1];
        let mut indices = vec![Ix::from_usize(0); nnz];
        let mut data = coo.data().to_vec();
        let rows: Vec<usize> = coo.row_inds().iter().map(|x| x.index()).collect();
        coo_to_unsorted_cs(
            &mut offsets,
            &mut indices,
            &mut data,
            coo.rows(),
            &rows,
            coo.col_inds(),
            coo.data(),
        );
        Self::from_csr_data(coo.rows(), coo.cols(), offsets, indices, data)
    }
}

impl<N: BackendData, I: sprs::SpIndex + BackendData> Element for CsrNonCanonical<N, I> {
    fn data_type(&self) -> DataType {
        DataType::CsrMatrix(N::DTYPE, I::DTYPE)
    }

    fn metadata(&self) -> MetaData {
        let mut metadata = HashMap::new();
        metadata.insert(
            "shape".to_string(),
            vec![self.num_rows, self.num_cols].into(),
        );
        MetaData::new("csr_matrix", "0.1.0", Some(metadata))
    }
}

impl<N: BackendData, I: sprs::SpIndex + BackendData> Writable for CsrNonCanonical<N, I> {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let mut group = location.new_group(name)?;
        self.metadata().save(&mut group)?;
        group.new_array_dataset("data", self.values.as_slice().into(), Default::default())?;
        group.new_array_dataset("indptr", self.offsets.as_slice().into(), Default::default())?;
        group.new_array_dataset(
            "indices",
            self.indices.as_slice().into(),
            Default::default(),
        )?;
        Ok(DataContainer::Group(group))
    }
}

impl<N: BackendData, I: sprs::SpIndex + BackendData + num::FromPrimitive> Readable
    for CsrNonCanonical<N, I>
{
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let group = container.as_group()?;
        let shape: Vec<u64> = group.get_attr("shape")?;
        let (data, (indptr, indices)) = rayon::join(
            || {
                group
                    .open_dataset("data")?
                    .read_array::<N, Ix1>()
                    .map(|x| x.into_raw_vec_and_offset().0)
            },
            || {
                rayon::join(
                    || {
                        group
                            .open_dataset("indptr")?
                            .read_array::<u64, Ix1>()
                            .map(|x| x.into_raw_vec_and_offset().0)
                    },
                    || {
                        group
                            .open_dataset("indices")?
                            .read_array::<I, Ix1>()
                            .map(|x| x.into_raw_vec_and_offset().0)
                    },
                )
            },
        );

        Ok(Self::from_csr_data(
            shape[0] as usize,
            shape[1] as usize,
            indptr?,
            indices?,
            data?,
        ))
    }
}

impl<N: BackendData, I: sprs::SpIndex + BackendData + num::FromPrimitive + num::Integer>
    ReadableArray for CsrNonCanonical<N, I>
{
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        if info.as_ref().len() != 2 {
            panic!("index must have length 2");
        }

        if info.iter().all(|s| s.as_ref().is_full()) {
            return Self::read(container);
        }

        let data = if let SelectInfoElem::Slice(s) = info[0].as_ref() {
            let group = container.as_group()?;
            let indptr_slice = if let Some(end) = s.end {
                SelectInfoElem::from(s.start..end + 1)
            } else {
                SelectInfoElem::from(s.start..)
            };
            let mut indptr: Vec<u64> = group
                .open_dataset("indptr")?
                .read_array_slice::<u64, _, Ix1>(&[indptr_slice])?
                .into_raw_vec_and_offset()
                .0;
            let lo = indptr[0];
            let slice = SelectInfoElem::from(lo as usize..indptr[indptr.len() - 1] as usize);
            let (data, indices) = rayon::join(
                || {
                    group
                        .open_dataset("data")?
                        .read_array_slice::<N, _, Ix1>(&[&slice])
                        .map(|x| x.into_raw_vec_and_offset().0)
                },
                || {
                    group
                        .open_dataset("indices")?
                        .read_array_slice::<I, _, Ix1>(&[&slice])
                        .map(|x| x.into_raw_vec_and_offset().0)
                },
            );
            indptr.iter_mut().for_each(|x| *x -= lo);
            Self::from_csr_data(
                indptr.len() - 1,
                Self::get_shape(container)?[1],
                indptr,
                indices?,
                data?,
            )
            .select_axis(1, info[1].as_ref())
        } else {
            Self::read(container)?.select(info)
        };
        Ok(data)
    }
}

impl<N: BackendData, I: sprs::SpIndex + BackendData> WritableArray for &CsrNonCanonical<N, I> {}
impl<N: BackendData, I: sprs::SpIndex + BackendData> WritableArray for CsrNonCanonical<N, I> {}

impl<N: Clone, I: sprs::SpIndex> HasShape for CsrNonCanonical<N, I> {
    fn shape(&self) -> Shape {
        vec![self.num_rows, self.num_cols].into()
    }
}

impl<N: ToPrimitive + Clone, I: sprs::SpIndex> ArrayArithmetic for CsrNonCanonical<N, I> {
    fn sum(&self) -> f64 {
        self.values.iter().map(|x| x.to_f64().unwrap()).sum()
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayD<f64>> {
        if axis == 0 {
            let mut col_sums = vec![0.0; self.num_cols];
            for row in 0..self.num_rows {
                let start = self.offsets[row] as usize;
                let end = self.offsets[row + 1] as usize;
                for (&col, val) in self.indices[start..end]
                    .iter()
                    .zip(&self.values[start..end])
                {
                    col_sums[col.to_usize().unwrap()] += val.to_f64().unwrap();
                }
            }
            Ok(ndarray::Array1::from_vec(col_sums).into_dyn())
        } else if axis == 1 {
            let row_sums: Vec<f64> = (0..self.num_rows)
                .map(|row| {
                    let start = self.offsets[row] as usize;
                    let end = self.offsets[row + 1] as usize;
                    self.values[start..end]
                        .iter()
                        .map(|x| x.to_f64().unwrap())
                        .sum()
                })
                .collect();
            Ok(ndarray::Array1::from_vec(row_sums).into_dyn())
        } else {
            bail!("Axis {axis} out of bounds for 2D matrix")
        }
    }

    fn min(&self) -> f64 {
        self.values
            .iter()
            .map(|x| x.to_f64().unwrap())
            .fold(f64::INFINITY, f64::min)
    }

    fn max(&self) -> f64 {
        self.values
            .iter()
            .map(|x| x.to_f64().unwrap())
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

impl<N: Clone, I: sprs::SpIndex + num::Integer + num::FromPrimitive> Selectable
    for CsrNonCanonical<N, I>
{
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        if info.as_ref().len() != 2 {
            panic!("DataFrame only support 2D selection");
        }
        let select = SelectInfoBounds::new(&info, &self.shape());
        let major = select.as_ref()[0].to_vec();
        let minor = select.as_ref()[1].to_vec();
        let (indptr, indices, data) = cs_major_minor_index(
            major.iter().copied(),
            minor.iter().copied(),
            self.num_cols,
            &self.offsets,
            &self.indices,
            &self.values,
        );
        Self::from_csr_data(
            select.as_ref()[0].len(),
            select.as_ref()[1].len(),
            indptr,
            indices,
            data,
        )
    }
}

impl<N: Clone, I: sprs::SpIndex> Stackable for CsrNonCanonical<N, I> {
    fn vstack<I2: Iterator<Item = Self>>(iter: I2) -> Result<Self> {
        let mut iter = iter.peekable();
        let first = iter.peek().ok_or(anyhow::anyhow!("Empty iterator"))?;
        let num_cols = first.num_cols;

        let mut num_rows = 0;
        let mut total_nnz = 0;
        let mut matrices = Vec::new();

        for m in iter {
            if m.num_cols != num_cols {
                bail!("Cannot vstack matrices with different number of columns");
            }
            num_rows += m.num_rows;
            total_nnz += m.nnz();
            matrices.push(m);
        }

        let mut new_offsets = Vec::with_capacity(num_rows + 1);
        let mut new_indices = Vec::with_capacity(total_nnz);
        let mut new_values = Vec::with_capacity(total_nnz);

        new_offsets.push(0);
        let mut current_nnz: u64 = 0;

        for m in matrices {
            for &p in &m.offsets[1..] {
                new_offsets.push(current_nnz + p);
            }
            current_nnz += m.nnz() as u64;
            new_indices.extend_from_slice(&m.indices);
            new_values.extend_from_slice(&m.values);
        }

        Ok(Self::from_csr_data(
            num_rows,
            num_cols,
            new_offsets,
            new_indices,
            new_values,
        ))
    }
}

impl<N, I: sprs::SpIndex + ToPrimitive> From<&CsrNonCanonical<N, I>> for TriMatI<N, I>
where
    N: Clone,
{
    fn from(csr: &CsrNonCanonical<N, I>) -> Self {
        let mut coo = TriMatI::new((csr.num_rows, csr.num_cols));
        for row in 0..csr.num_rows {
            let start = csr.offsets[row] as usize;
            let end = csr.offsets[row + 1] as usize;
            for i in start..end {
                coo.add_triplet(
                    row,
                    csr.indices[i].to_usize().unwrap(),
                    csr.values[i].clone(),
                );
            }
        }
        coo
    }
}
