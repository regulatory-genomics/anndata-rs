use core::panic;

use crate::backend::*;
use crate::data::ArrayConvert;
use crate::data::{
    data_traits::*,
    slice::{SelectInfoElem, Shape},
};

use crate::data::DynScalar;
use anyhow::{Result, bail};
use ndarray::ArrayD;
use num::{FromPrimitive, ToPrimitive};
use sprs::{CsMatI, SpIndex};

#[derive(Debug, Clone, PartialEq)]
pub enum DynIndSparseMatrix {
    I16(DynSparseMatrix<i16>),
    I32(DynSparseMatrix<i32>),
    I64(DynSparseMatrix<i64>),
    U16(DynSparseMatrix<u16>),
    U32(DynSparseMatrix<u32>),
    U64(DynSparseMatrix<u64>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DynSparseMatrix<T>
where
    T: SpIndex + num::Integer + num::FromPrimitive,
{
    I8(CsMatI<i8, T, u64>),
    I16(CsMatI<i16, T, u64>),
    I32(CsMatI<i32, T, u64>),
    I64(CsMatI<i64, T, u64>),
    U8(CsMatI<u8, T, u64>),
    U16(CsMatI<u16, T, u64>),
    U32(CsMatI<u32, T, u64>),
    U64(CsMatI<u64, T, u64>),
    F32(CsMatI<f32, T, u64>),
    F64(CsMatI<f64, T, u64>),
    Bool(CsMatI<bool, T, u64>),
    String(CsMatI<String, T, u64>),
}

////////////////////////////////////////////////////////////////////////////////
// ArrayConvert implementations
////////////////////////////////////////////////////////////////////////////////

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> ArrayConvert<CsMatI<u32, T, u64>>
    for DynSparseMatrix<T>
{
    fn try_convert(self) -> Result<CsMatI<u32, T, u64>> {
        match self {
            DynSparseMatrix::U32(data) => Ok(data),
            DynSparseMatrix::I8(data) => convert_sparse_with(data, |x| Ok(x.try_into()?)),
            DynSparseMatrix::I16(data) => convert_sparse_with(data, |x| Ok(x.try_into()?)),
            DynSparseMatrix::I32(data) => convert_sparse_with(data, |x| Ok(x.try_into()?)),
            DynSparseMatrix::I64(data) => convert_sparse_with(data, |x| Ok(x.try_into()?)),
            DynSparseMatrix::U8(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::U16(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::U64(data) => convert_sparse_with(data, |x| Ok(x.try_into()?)),
            DynSparseMatrix::Bool(data) => convert_sparse_with(data, |x| Ok(x.into())),
            v => bail!(
                "Cannot convert {} to CsMatI<u32, {}, u64>",
                v.data_type(),
                std::any::type_name::<T>()
            ),
        }
    }
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> ArrayConvert<CsMatI<f32, T, u64>>
    for DynSparseMatrix<T>
{
    fn try_convert(self) -> Result<CsMatI<f32, T, u64>> {
        match self {
            DynSparseMatrix::F32(data) => Ok(data),
            DynSparseMatrix::I8(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::I16(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::I32(data) => {
                convert_sparse_with(data, |x| Ok(f32::from_i32(x).unwrap()))
            }
            DynSparseMatrix::I64(data) => {
                convert_sparse_with(data, |x| Ok(f32::from_i64(x).unwrap()))
            }
            DynSparseMatrix::U8(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::U16(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::U32(data) => {
                convert_sparse_with(data, |x| Ok(f32::from_u32(x).unwrap()))
            }
            DynSparseMatrix::U64(data) => {
                convert_sparse_with(data, |x| Ok(f32::from_u64(x).unwrap()))
            }
            DynSparseMatrix::F64(data) => {
                convert_sparse_with(data, |x| Ok(f32::from_f64(x).unwrap()))
            }
            DynSparseMatrix::Bool(data) => convert_sparse_with(data, |x| Ok(x.into())),
            v => bail!(
                "Cannot convert {} to CsMatI<f32, {}, u64>",
                v.data_type(),
                std::any::type_name::<T>()
            ),
        }
    }
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> ArrayConvert<CsMatI<f64, T, u64>>
    for DynSparseMatrix<T>
{
    fn try_convert(self) -> Result<CsMatI<f64, T, u64>> {
        match self {
            DynSparseMatrix::F64(data) => Ok(data),
            DynSparseMatrix::I8(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::I16(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::I32(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::I64(data) => {
                convert_sparse_with(data, |x| Ok(f64::from_i64(x).unwrap()))
            }
            DynSparseMatrix::U8(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::U16(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::U32(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::U64(data) => {
                convert_sparse_with(data, |x| Ok(f64::from_u64(x).unwrap()))
            }
            DynSparseMatrix::F32(data) => convert_sparse_with(data, |x| Ok(x.into())),
            DynSparseMatrix::Bool(data) => convert_sparse_with(data, |x| Ok(x.into())),
            v => bail!(
                "Cannot convert {} to CsMatI<f64, {}, u64>",
                v.data_type(),
                std::any::type_name::<T>()
            ),
        }
    }
}

fn convert_sparse_with<T, U, Ix, F>(mat: CsMatI<T, Ix, u64>, f: F) -> Result<CsMatI<U, Ix, u64>>
where
    T: BackendData,
    U: BackendData,
    Ix: SpIndex + BackendData,
    F: Fn(T) -> Result<U>,
{
    let shape = mat.shape();
    let is_csr = mat.is_csr();
    let indptr = mat.indptr().as_slice().unwrap().to_vec();
    let indices = mat.indices().to_vec();
    let data = mat.data().to_vec();
    let new_data = data.into_iter().map(f).collect::<Result<Vec<U>>>()?;
    if is_csr {
        Ok(CsMatI::new(shape, indptr, indices, new_data))
    } else {
        Ok(CsMatI::new_csc(shape, indptr, indices, new_data))
    }
}

// Dynamic sparse matrix implementations.

impl<T: SpIndex + BackendData + num::Integer + num::FromPrimitive> Element for DynSparseMatrix<T> {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynSparseMatrix, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_map_fun!(self, DynSparseMatrix, metadata)
    }
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> Writable for DynSparseMatrix<T> {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynSparseMatrix, write, location, name)
    }
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> Readable for DynSparseMatrix<T> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self>
    where
        Self: Sized,
    {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($type:ident, $variant:ident) => {
                        CsMatI::<$type, T, u64>::read(container).map(DynSparseMatrix::$variant)
                    };
                }
                crate::macros::dyn_match_new!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            }
            _ => bail!("Can't read sparse matrix from non-group container!"),
        }
    }
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> HasShape for DynSparseMatrix<T> {
    fn shape(&self) -> Shape {
        match self {
            DynSparseMatrix::I8(data) => HasShape::shape(data),
            DynSparseMatrix::I16(data) => HasShape::shape(data),
            DynSparseMatrix::I32(data) => HasShape::shape(data),
            DynSparseMatrix::I64(data) => HasShape::shape(data),
            DynSparseMatrix::U8(data) => HasShape::shape(data),
            DynSparseMatrix::U16(data) => HasShape::shape(data),
            DynSparseMatrix::U32(data) => HasShape::shape(data),
            DynSparseMatrix::U64(data) => HasShape::shape(data),
            DynSparseMatrix::F32(data) => HasShape::shape(data),
            DynSparseMatrix::F64(data) => HasShape::shape(data),
            DynSparseMatrix::Bool(data) => HasShape::shape(data),
            DynSparseMatrix::String(data) => HasShape::shape(data),
        }
    }
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> Selectable
    for DynSparseMatrix<T>
{
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        crate::macros::dyn_sparse_map!(self, DynSparseMatrix, select, info)
    }
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> SparseMatrixLayout
    for DynSparseMatrix<T>
{
    fn get_sparse_layout(&self) -> SparseMatrixLayoutE {
        crate::macros::dyn_map_fun!(self, DynSparseMatrix, get_sparse_layout)
    }
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> Indexable
    for DynSparseMatrix<T>
{
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        if index.len() != 2 {
            panic!("index must have length 2");
        }
        match self {
            DynSparseMatrix::I8(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
            DynSparseMatrix::I16(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
            DynSparseMatrix::I32(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
            DynSparseMatrix::I64(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
            DynSparseMatrix::U8(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
            DynSparseMatrix::U16(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
            DynSparseMatrix::U32(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
            DynSparseMatrix::U64(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
            DynSparseMatrix::F32(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
            DynSparseMatrix::F64(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
            DynSparseMatrix::Bool(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
            DynSparseMatrix::String(data) => data.get(index[0], index[1]).map(|v| v.as_dyn()),
        }
    }
}

impl<T: BackendData + SpIndex + ToPrimitive + num::Integer + num::FromPrimitive> ArrayArithmetic
    for DynSparseMatrix<T>
{
    fn sum(&self) -> f64 {
        match self {
            DynSparseMatrix::I8(arr) => ArrayArithmetic::sum(arr),
            DynSparseMatrix::I16(arr) => ArrayArithmetic::sum(arr),
            DynSparseMatrix::I32(arr) => ArrayArithmetic::sum(arr),
            DynSparseMatrix::I64(arr) => ArrayArithmetic::sum(arr),
            DynSparseMatrix::U8(arr) => ArrayArithmetic::sum(arr),
            DynSparseMatrix::U16(arr) => ArrayArithmetic::sum(arr),
            DynSparseMatrix::U32(arr) => ArrayArithmetic::sum(arr),
            DynSparseMatrix::U64(arr) => ArrayArithmetic::sum(arr),
            DynSparseMatrix::F32(arr) => ArrayArithmetic::sum(arr),
            DynSparseMatrix::F64(arr) => ArrayArithmetic::sum(arr),
            DynSparseMatrix::Bool(_) => panic!("Cannot compute sum for Bool sparse matrix"),
            DynSparseMatrix::String(_) => panic!("Cannot compute sum for String sparse matrix"),
        }
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayD<f64>> {
        match self {
            DynSparseMatrix::I8(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynSparseMatrix::I16(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynSparseMatrix::I32(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynSparseMatrix::I64(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynSparseMatrix::U8(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynSparseMatrix::U16(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynSparseMatrix::U32(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynSparseMatrix::U64(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynSparseMatrix::F32(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynSparseMatrix::F64(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynSparseMatrix::Bool(_) => panic!("Cannot compute sum for Bool sparse matrix"),
            DynSparseMatrix::String(_) => bail!("Cannot compute sum for String sparse matrix"),
        }
    }

    fn min(&self) -> f64 {
        match self {
            DynSparseMatrix::I8(arr) => ArrayArithmetic::min(arr),
            DynSparseMatrix::I16(arr) => ArrayArithmetic::min(arr),
            DynSparseMatrix::I32(arr) => ArrayArithmetic::min(arr),
            DynSparseMatrix::I64(arr) => ArrayArithmetic::min(arr),
            DynSparseMatrix::U8(arr) => ArrayArithmetic::min(arr),
            DynSparseMatrix::U16(arr) => ArrayArithmetic::min(arr),
            DynSparseMatrix::U32(arr) => ArrayArithmetic::min(arr),
            DynSparseMatrix::U64(arr) => ArrayArithmetic::min(arr),
            DynSparseMatrix::F32(arr) => ArrayArithmetic::min(arr),
            DynSparseMatrix::F64(arr) => ArrayArithmetic::min(arr),
            DynSparseMatrix::Bool(_) => panic!("Cannot compute min for Bool sparse matrix"),
            DynSparseMatrix::String(_) => panic!("Cannot compute min for String sparse matrix"),
        }
    }

    fn max(&self) -> f64 {
        match self {
            DynSparseMatrix::I8(arr) => ArrayArithmetic::max(arr),
            DynSparseMatrix::I16(arr) => ArrayArithmetic::max(arr),
            DynSparseMatrix::I32(arr) => ArrayArithmetic::max(arr),
            DynSparseMatrix::I64(arr) => ArrayArithmetic::max(arr),
            DynSparseMatrix::U8(arr) => ArrayArithmetic::max(arr),
            DynSparseMatrix::U16(arr) => ArrayArithmetic::max(arr),
            DynSparseMatrix::U32(arr) => ArrayArithmetic::max(arr),
            DynSparseMatrix::U64(arr) => ArrayArithmetic::max(arr),
            DynSparseMatrix::F32(arr) => ArrayArithmetic::max(arr),
            DynSparseMatrix::F64(arr) => ArrayArithmetic::max(arr),
            DynSparseMatrix::Bool(_) => panic!("Cannot compute max for Bool sparse matrix"),
            DynSparseMatrix::String(_) => panic!("Cannot compute max for String sparse matrix"),
        }
    }
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> Stackable
    for DynSparseMatrix<T>
{
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter
            .peek()
            .ok_or(anyhow::anyhow!("Cannot stack empty iterator"))?
        {
            DynSparseMatrix::I8(_) => Ok(DynSparseMatrix::I8(CsMatI::<i8, T, u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynSparseMatrix::I16(_) => Ok(DynSparseMatrix::I16(CsMatI::<i16, T, u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynSparseMatrix::I32(_) => Ok(DynSparseMatrix::I32(CsMatI::<i32, T, u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynSparseMatrix::I64(_) => Ok(DynSparseMatrix::I64(CsMatI::<i64, T, u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynSparseMatrix::U8(_) => Ok(DynSparseMatrix::U8(CsMatI::<u8, T, u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynSparseMatrix::U16(_) => Ok(DynSparseMatrix::U16(CsMatI::<u16, T, u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynSparseMatrix::U32(_) => Ok(DynSparseMatrix::U32(CsMatI::<u32, T, u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynSparseMatrix::U64(_) => Ok(DynSparseMatrix::U64(CsMatI::<u64, T, u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynSparseMatrix::F32(_) => Ok(DynSparseMatrix::F32(CsMatI::<f32, T, u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynSparseMatrix::F64(_) => Ok(DynSparseMatrix::F64(CsMatI::<f64, T, u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynSparseMatrix::Bool(_) => Ok(DynSparseMatrix::Bool(CsMatI::<bool, T, u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynSparseMatrix::String(_) => Ok(DynSparseMatrix::String(
                CsMatI::<String, T, u64>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
        }
    }
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> WritableArray
    for DynSparseMatrix<T>
{
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> ReadableArray
    for DynSparseMatrix<T>
{
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
        Self: Sized,
    {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($type: ident, $variant:ident) => {
                        CsMatI::<$type, T, u64>::read_select(container, info)
                            .map(DynSparseMatrix::$variant)
                    };
                }
                crate::macros::dyn_match_new!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            }
            _ => bail!("Unable to read sparse matrix from non-group container"),
        }
    }
}

impl Element for DynIndSparseMatrix {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, metadata)
    }
}

impl Writable for DynIndSparseMatrix {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, write, location, name)
    }
}

impl DynIndSparseMatrix {
    pub(crate) fn read_with_types<B: Backend>(
        container: &DataContainer<B>,
        value_dtype: ScalarType,
        index_dtype: ScalarType,
    ) -> Result<Self> {
        macro_rules! index_fun {
            ($index_type:ident, $index_variant:ident) => {{
                macro_rules! value_fun {
                    ($value_type:ident, $value_variant:ident) => {
                        CsMatI::<$value_type, $index_type, u64>::read(container)
                            .map(DynSparseMatrix::$value_variant)
                            .map(DynIndSparseMatrix::$index_variant)
                    };
                }
                crate::macros::dyn_match_new!(value_dtype, ScalarType, value_fun)
            }};
        }

        crate::macros::dyn_index_match!(index_dtype, ScalarType, index_fun)
    }
}

impl Readable for DynIndSparseMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self>
    where
        Self: Sized,
    {
        match container {
            DataContainer::Group(group) => {
                let indices_dtype = group.open_dataset("indices")?.dtype()?;
                macro_rules! fun {
                    ($type:ident, $variant:ident) => {
                        DynSparseMatrix::<$type>::read(container).map(DynIndSparseMatrix::$variant)
                    };
                }

                crate::macros::dyn_index_match!(indices_dtype, ScalarType, fun)
            }
            _ => bail!("Can't read sparse matrix from non-group container"),
        }
    }
}

impl HasShape for DynIndSparseMatrix {
    fn shape(&self) -> Shape {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, shape)
    }
}

impl Selectable for DynIndSparseMatrix {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        crate::macros::dyn_index_sparse_map!(self, DynIndSparseMatrix, select, info)
    }
}

impl Indexable for DynIndSparseMatrix {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, get, index)
    }
}

impl ArrayArithmetic for DynIndSparseMatrix {
    fn sum(&self) -> f64 {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, sum)
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayD<f64>> {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, sum_axis, axis)
    }

    fn min(&self) -> f64 {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, min)
    }

    fn max(&self) -> f64 {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, max)
    }
}

impl SparseMatrixLayout for DynIndSparseMatrix {
    fn get_sparse_layout(&self) -> SparseMatrixLayoutE {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, get_sparse_layout)
    }
}

impl WritableArray for DynIndSparseMatrix {}

impl ReadableArray for DynIndSparseMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
        Self: Sized,
    {
        match container {
            DataContainer::Group(group) => {
                let indices_type = group.open_dataset("indices")?.dtype()?;
                macro_rules! fun {
                    ($type:ident, $variant:ident) => {
                        DynSparseMatrix::<$type>::read_select(container, info)
                            .map(DynIndSparseMatrix::$variant)
                    };
                }
                crate::macros::dyn_index_match!(indices_type, ScalarType, fun)
            }
            _ => bail!("Cant read sparse matrix from non group container"),
        }
    }
}

impl Stackable for DynIndSparseMatrix {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter
            .peek()
            .ok_or(anyhow::anyhow!("Cannot stack empty iterator"))?
        {
            DynIndSparseMatrix::I16(_) => Ok(DynIndSparseMatrix::I16(
                DynSparseMatrix::<i16>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynIndSparseMatrix::I32(_) => Ok(DynIndSparseMatrix::I32(
                DynSparseMatrix::<i32>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynIndSparseMatrix::I64(_) => Ok(DynIndSparseMatrix::I64(
                DynSparseMatrix::<i64>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynIndSparseMatrix::U16(_) => Ok(DynIndSparseMatrix::U16(
                DynSparseMatrix::<u16>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynIndSparseMatrix::U32(_) => Ok(DynIndSparseMatrix::U32(
                DynSparseMatrix::<u32>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynIndSparseMatrix::U64(_) => Ok(DynIndSparseMatrix::U64(
                DynSparseMatrix::<u64>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
        }
    }
}

#[inline]
pub fn vec_usize_to_u64(v: Vec<usize>) -> Vec<u64> {
    #[cfg(target_pointer_width = "64")]
    unsafe {
        use std::mem::ManuallyDrop;

        let mut v = ManuallyDrop::new(v);
        Vec::from_raw_parts(v.as_mut_ptr() as *mut u64, v.len(), v.capacity())
    }

    #[cfg(not(target_pointer_width = "64"))]
    {
        v.into_iter().map(|x| x as u64).collect()
    }
}

macro_rules! impl_from_dynsparse_to_dynind {
    ($($type:ty, $variant:ident),*) => {
        $(
            impl From<DynSparseMatrix<$type>> for DynIndSparseMatrix {
                fn from(value: DynSparseMatrix<$type>) -> Self {
                    DynIndSparseMatrix::$variant(value)
                }
            }
        )*
    };
}

impl_from_dynsparse_to_dynind!(i16, I16, i32, I32, i64, I64, u16, U16, u32, U32, u64, U64);

macro_rules! impl_from_csmati_to_dynsparse {
    ($($value_type:ty, $value_variant:ident),*) => {
        $(
            impl<T: SpIndex + BackendData + num::Integer + num::FromPrimitive> From<CsMatI<$value_type, T, u64>> for DynSparseMatrix<T> {
                fn from(value: CsMatI<$value_type, T, u64>) -> Self {
                    DynSparseMatrix::$value_variant(value)
                }
            }
        )*
    };
}

impl_from_csmati_to_dynsparse!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

macro_rules! impl_tryfrom_dynsparse_to_csmati {
    ($($value_type:ty, $value_variant:ident),*) => {
        $(
            impl<T: SpIndex + BackendData + num::Integer + num::FromPrimitive> TryFrom<DynSparseMatrix<T>> for CsMatI<$value_type, T, u64> {
                type Error = anyhow::Error;

                fn try_from(value: DynSparseMatrix<T>) -> Result<Self> {
                    match value {
                        DynSparseMatrix::$value_variant(data) => Ok(data),
                        _ => bail!(
                            "Cannot convert {} to CsMatI<{}, {}, u64>",
                            value.data_type(),
                            stringify!($value_type),
                            std::any::type_name::<T>()
                        ),
                    }
                }
            }
        )*
    };
}

impl_tryfrom_dynsparse_to_csmati!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

macro_rules! impl_tryfrom_dynind_to_dynsparse {
    ($($index_type:ty, $index_variant:ident),*) => {
        $(
            impl TryFrom<DynIndSparseMatrix> for DynSparseMatrix<$index_type> {
                type Error = anyhow::Error;

                fn try_from(value: DynIndSparseMatrix) -> Result<Self> {
                    match value {
                        DynIndSparseMatrix::$index_variant(data) => Ok(data),
                        _ => bail!(
                            "Cannot convert DynIndSparseMatrix with index type {:?} to DynSparseMatrix<{}>",
                            value.data_type(),
                            stringify!($index_type)
                        ),
                    }
                }
            }
        )*
    };
}

impl_tryfrom_dynind_to_dynsparse!(i16, I16, i32, I32, i64, I64, u16, U16, u32, U32, u64, U64);
