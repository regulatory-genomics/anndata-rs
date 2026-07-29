mod chunks;
pub mod dataframe;
mod dense;
pub mod slice;
mod sparse;
pub mod utils;

pub use chunks::{ArrayChunk, MatrixBuilder};
pub use dataframe::DataFrameIndex;
pub use dense::{ArrayConvert, CategoricalArray, DynArray, DynCowArray, DynScalar};
pub use slice::{SelectInfo, SelectInfoBounds, SelectInfoElem, SelectInfoElemBounds, Shape};
pub use sparse::{CsrNonCanonical, DynCsrNonCanonical, DynIndSparseMatrix, DynSparseMatrix};
use sprs::{CsMatI, SpIndex};

use crate::backend::*;
use crate::data::{DataType, data_traits::*};

use anyhow::{Result, bail};
use ndarray::{Array, ArrayD, RemoveAxis};
use polars::prelude::DataFrame;

#[derive(Debug, Clone, PartialEq)]
pub enum ArrayData {
    Array(DynArray),
    CsrMatrix(DynIndSparseMatrix),
    CsrNonCanonical(DynCsrNonCanonical),
    CscMatrix(DynIndSparseMatrix),
    DataFrame(DataFrame),
}

impl<T: Clone + Into<ArrayData>> From<&T> for ArrayData {
    fn from(data: &T) -> Self {
        data.clone().into()
    }
}

impl From<DataFrame> for ArrayData {
    fn from(data: DataFrame) -> Self {
        ArrayData::DataFrame(data)
    }
}

impl From<DynArray> for ArrayData {
    fn from(data: DynArray) -> Self {
        ArrayData::Array(data)
    }
}

impl From<DynCsrNonCanonical> for ArrayData {
    fn from(data: DynCsrNonCanonical) -> Self {
        ArrayData::CsrNonCanonical(data)
    }
}

impl From<DynIndSparseMatrix> for ArrayData {
    fn from(value: DynIndSparseMatrix) -> Self {
        match value.get_sparse_layout() {
            SparseMatrixLayoutE::CSR => ArrayData::CsrMatrix(value),
            SparseMatrixLayoutE::CSC => ArrayData::CscMatrix(value),
            SparseMatrixLayoutE::COO => todo!(),
            SparseMatrixLayoutE::NONE => {
                panic!("A matrix with a none layout cannot be added into an array object");
            }
        }
    }
}

impl TryFrom<ArrayData> for DynArray {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::Array(data) => Ok(data),
            _ => bail!("Cannot convert {:?} to DynArray", value.data_type()),
        }
    }
}

impl TryFrom<ArrayData> for DynCsrNonCanonical {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::CsrNonCanonical(data) => Ok(data),
            _ => bail!(
                "Cannot convert {:?} to DynCsrNonCanonical",
                value.data_type()
            ),
        }
    }
}

impl<T> TryFrom<ArrayData> for CsrNonCanonical<T>
where
    CsrNonCanonical<T>: TryFrom<DynCsrNonCanonical, Error = anyhow::Error>,
{
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        DynCsrNonCanonical::try_from(value)?.try_into()
    }
}

impl TryFrom<ArrayData> for DynIndSparseMatrix {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::CsrMatrix(data) => Ok(data),
            ArrayData::CscMatrix(data) => Ok(data),
            _ => bail!(
                "Cannot convert {:?} to DynIndSparseMatrix",
                value.data_type()
            ),
        }
    }
}

impl TryFrom<ArrayData> for DataFrame {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::DataFrame(data) => Ok(data),
            _ => bail!("Cannot convert {:?} to DataFrame", value.data_type()),
        }
    }
}

impl<T, D> TryFrom<ArrayData> for Array<T, D>
where
    Array<T, D>: TryFrom<DynArray, Error = anyhow::Error>,
{
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        DynArray::try_from(value)?.try_into()
    }
}

fn downcast_sparse_matrix<N, T, V, I>(matrix: CsMatI<V, I, u64>) -> Result<CsMatI<N, T, u64>>
where
    N: BackendData,
    T: BackendData + SpIndex,
    V: BackendData,
    I: BackendData + SpIndex,
{
    let matrix: Box<dyn std::any::Any> = Box::new(matrix);
    matrix
        .downcast::<CsMatI<N, T, u64>>()
        .map(|matrix| *matrix)
        .map_err(|_| anyhow::anyhow!("sparse matrix type mismatch"))
}

impl<N, T> TryFrom<ArrayData> for CsMatI<N, T, u64>
where
    N: BackendData,
    T: BackendData + SpIndex,
{
    type Error = anyhow::Error;

    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        use std::any::TypeId;

        let dyn_ind = DynIndSparseMatrix::try_from(value)?;

        macro_rules! try_extract {
            ($data:expr, $index_ty:ty) => {{
                if TypeId::of::<N>() == TypeId::of::<i8>() {
                    let result = CsMatI::<i8, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else if TypeId::of::<N>() == TypeId::of::<i16>() {
                    let result = CsMatI::<i16, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else if TypeId::of::<N>() == TypeId::of::<i32>() {
                    let result = CsMatI::<i32, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else if TypeId::of::<N>() == TypeId::of::<i64>() {
                    let result = CsMatI::<i64, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else if TypeId::of::<N>() == TypeId::of::<u8>() {
                    let result = CsMatI::<u8, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else if TypeId::of::<N>() == TypeId::of::<u16>() {
                    let result = CsMatI::<u16, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else if TypeId::of::<N>() == TypeId::of::<u32>() {
                    let result = CsMatI::<u32, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else if TypeId::of::<N>() == TypeId::of::<u64>() {
                    let result = CsMatI::<u64, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else if TypeId::of::<N>() == TypeId::of::<f32>() {
                    let result = CsMatI::<f32, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else if TypeId::of::<N>() == TypeId::of::<f64>() {
                    let result = CsMatI::<f64, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else if TypeId::of::<N>() == TypeId::of::<bool>() {
                    let result = CsMatI::<bool, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else if TypeId::of::<N>() == TypeId::of::<String>() {
                    let result = CsMatI::<String, $index_ty, u64>::try_from($data)?;
                    downcast_sparse_matrix(result)
                } else {
                    bail!("Unsupported value type: {}", std::any::type_name::<N>())
                }
            }};
        }

        match dyn_ind {
            DynIndSparseMatrix::I16(data) if TypeId::of::<T>() == TypeId::of::<i16>() => {
                try_extract!(data, i16)
            }
            DynIndSparseMatrix::I32(data) if TypeId::of::<T>() == TypeId::of::<i32>() => {
                try_extract!(data, i32)
            }
            DynIndSparseMatrix::I64(data) if TypeId::of::<T>() == TypeId::of::<i64>() => {
                try_extract!(data, i64)
            }
            DynIndSparseMatrix::U16(data) if TypeId::of::<T>() == TypeId::of::<u16>() => {
                try_extract!(data, u16)
            }
            DynIndSparseMatrix::U32(data) if TypeId::of::<T>() == TypeId::of::<u32>() => {
                try_extract!(data, u32)
            }
            DynIndSparseMatrix::U64(data) if TypeId::of::<T>() == TypeId::of::<u64>() => {
                try_extract!(data, u64)
            }
            _ => bail!(
                "Index type mismatch: expected {}, found {:?}",
                std::any::type_name::<T>(),
                dyn_ind.data_type()
            ),
        }
    }
}

impl<T, D> ArrayConvert<Array<T, D>> for ArrayData
where
    DynArray: ArrayConvert<Array<T, D>>,
{
    fn try_convert(self) -> Result<Array<T, D>> {
        DynArray::try_from(self)?.try_convert()
    }
}

macro_rules! impl_arraydata_traits {
    ($($ty:ty),*) => {
        $(
            impl<D: RemoveAxis> From<Array<$ty, D>> for ArrayData {
                fn from(data: Array<$ty, D>) -> Self {
                    ArrayData::Array(data.into())
                }
            }
            impl From<CsrNonCanonical<$ty>> for ArrayData {
                fn from(data: CsrNonCanonical<$ty>) -> Self {
                    ArrayData::CsrNonCanonical(data.into())
                }
            }
        )*
    };
}

macro_rules! impl_csmati_to_arraydata {
    ($value_ty:ty; $($index_ty:ty),*) => {
        $(
            impl From<CsMatI<$value_ty, $index_ty, u64>> for ArrayData {
                fn from(data: CsMatI<$value_ty, $index_ty, u64>) -> Self {
                    let dyn_sparse: DynSparseMatrix<$index_ty> = data.into();
                    let dyn_ind: DynIndSparseMatrix = dyn_sparse.into();
                    match dyn_ind.get_sparse_layout() {
                        SparseMatrixLayoutE::CSR => ArrayData::CsrMatrix(dyn_ind),
                        SparseMatrixLayoutE::CSC => ArrayData::CscMatrix(dyn_ind),
                        SparseMatrixLayoutE::COO => panic!("COO layout not supported"),
                        SparseMatrixLayoutE::NONE => panic!("NONE layout not supported"),
                    }
                }
            }
        )*
    };
}

impl_csmati_to_arraydata!(i8; i16, i32, i64, u16, u32, u64);
impl_csmati_to_arraydata!(i16; i16, i32, i64, u16, u32, u64);
impl_csmati_to_arraydata!(i32; i16, i32, i64, u16, u32, u64);
impl_csmati_to_arraydata!(i64; i16, i32, i64, u16, u32, u64);
impl_csmati_to_arraydata!(u8; i16, i32, i64, u16, u32, u64);
impl_csmati_to_arraydata!(u16; i16, i32, i64, u16, u32, u64);
impl_csmati_to_arraydata!(u32; i16, i32, i64, u16, u32, u64);
impl_csmati_to_arraydata!(u64; i16, i32, i64, u16, u32, u64);
impl_csmati_to_arraydata!(f32; i16, i32, i64, u16, u32, u64);
impl_csmati_to_arraydata!(f64; i16, i32, i64, u16, u32, u64);
impl_csmati_to_arraydata!(bool; i16, i32, i64, u16, u32, u64);
impl_csmati_to_arraydata!(String; i16, i32, i64, u16, u32, u64);

impl_arraydata_traits!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool, String);

impl Readable for ArrayData {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => {
                DynArray::read(container).map(ArrayData::Array)
            }
            DataType::CsrMatrix(value_dtype, index_dtype) => {
                match DynIndSparseMatrix::read_with_types(container, value_dtype, index_dtype) {
                    Ok(data) => Ok(ArrayData::CsrMatrix(data)),
                    Err(_) => DynCsrNonCanonical::read(container).map(ArrayData::CsrNonCanonical),
                }
            }
            DataType::CscMatrix(value_dtype, index_dtype) => {
                DynIndSparseMatrix::read_with_types(container, value_dtype, index_dtype)
                    .map(ArrayData::CscMatrix)
            }
            DataType::DataFrame => DataFrame::read(container).map(ArrayData::DataFrame),
            ty => bail!("Cannot read type '{ty:?}' as matrix data"),
        }
    }
}

impl Element for ArrayData {
    fn data_type(&self) -> DataType {
        match self {
            ArrayData::Array(data) => data.data_type(),
            ArrayData::CsrMatrix(data) => data.data_type(),
            ArrayData::CsrNonCanonical(data) => data.data_type(),
            ArrayData::CscMatrix(data) => data.data_type(),
            ArrayData::DataFrame(data) => data.data_type(),
        }
    }

    fn metadata(&self) -> MetaData {
        match self {
            ArrayData::Array(data) => data.metadata(),
            ArrayData::CsrMatrix(data) => data.metadata(),
            ArrayData::CsrNonCanonical(data) => data.metadata(),
            ArrayData::CscMatrix(data) => data.metadata(),
            ArrayData::DataFrame(data) => data.metadata(),
        }
    }
}

impl Writable for ArrayData {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        match self {
            ArrayData::Array(data) => data.write(location, name),
            ArrayData::CsrMatrix(data) => data.write(location, name),
            ArrayData::CsrNonCanonical(data) => data.write(location, name),
            ArrayData::CscMatrix(data) => data.write(location, name),
            ArrayData::DataFrame(data) => data.write(location, name),
        }
    }
}

impl HasShape for ArrayData {
    fn shape(&self) -> Shape {
        match self {
            ArrayData::Array(data) => data.shape(),
            ArrayData::CsrMatrix(data) => data.shape(),
            ArrayData::CsrNonCanonical(data) => data.shape(),
            ArrayData::CscMatrix(data) => data.shape(),
            ArrayData::DataFrame(data) => HasShape::shape(data),
        }
    }
}

impl Indexable for ArrayData {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        match self {
            ArrayData::Array(data) => data.get(index),
            _ => todo!(),
        }
    }
}

impl Selectable for ArrayData {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        match self {
            ArrayData::Array(data) => data.select(info).into(),
            ArrayData::CsrMatrix(data) => data.select(info).into(),
            ArrayData::CsrNonCanonical(data) => data.select(info).into(),
            ArrayData::CscMatrix(data) => data.select(info).into(),
            ArrayData::DataFrame(data) => Selectable::select(data, info).into(),
        }
    }
}

impl Stackable for ArrayData {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        let item = iter.peek();
        if item.is_none() {
            bail!("Cannot stack empty iterator");
        }
        match item.unwrap() {
            ArrayData::Array(_) => {
                DynArray::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            ArrayData::CsrMatrix(_) => {
                DynIndSparseMatrix::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            ArrayData::CsrNonCanonical(_data) => {
                DynCsrNonCanonical::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            ArrayData::CscMatrix(_) => {
                DynIndSparseMatrix::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            ArrayData::DataFrame(_) => {
                <DataFrame as Stackable>::vstack(iter.map(|x| x.try_into().unwrap()))
                    .map(|x| x.into())
            }
        }
    }
}

impl ArrayArithmetic for ArrayData {
    fn sum(&self) -> f64 {
        match self {
            ArrayData::Array(data) => ArrayArithmetic::sum(data),
            ArrayData::CsrMatrix(data) => ArrayArithmetic::sum(data),
            ArrayData::CsrNonCanonical(data) => ArrayArithmetic::sum(data),
            ArrayData::CscMatrix(data) => ArrayArithmetic::sum(data),
            ArrayData::DataFrame(_) => panic!("Cannot compute sum for DataFrame"),
        }
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayD<f64>> {
        match self {
            ArrayData::Array(data) => ArrayArithmetic::sum_axis(data, axis),
            ArrayData::CsrMatrix(data) => ArrayArithmetic::sum_axis(data, axis),
            ArrayData::CsrNonCanonical(data) => ArrayArithmetic::sum_axis(data, axis),
            ArrayData::CscMatrix(data) => ArrayArithmetic::sum_axis(data, axis),
            ArrayData::DataFrame(_) => bail!("Cannot compute sum for DataFrame"),
        }
    }

    fn min(&self) -> f64 {
        match self {
            ArrayData::Array(data) => ArrayArithmetic::min(data),
            ArrayData::CsrMatrix(data) => ArrayArithmetic::min(data),
            ArrayData::CsrNonCanonical(data) => ArrayArithmetic::min(data),
            ArrayData::CscMatrix(data) => ArrayArithmetic::min(data),
            ArrayData::DataFrame(_) => panic!("Cannot compute min for DataFrame"),
        }
    }

    fn max(&self) -> f64 {
        match self {
            ArrayData::Array(data) => ArrayArithmetic::max(data),
            ArrayData::CsrMatrix(data) => ArrayArithmetic::max(data),
            ArrayData::CsrNonCanonical(data) => ArrayArithmetic::max(data),
            ArrayData::CscMatrix(data) => ArrayArithmetic::max(data),
            ArrayData::DataFrame(_) => panic!("Cannot compute max for DataFrame"),
        }
    }
}

impl ReadableArray for ArrayData {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => DynArray::get_shape(container),
            DataType::CsrMatrix(_, _) => DynIndSparseMatrix::get_shape(container),
            DataType::CscMatrix(_, _) => DynIndSparseMatrix::get_shape(container),
            DataType::DataFrame => DataFrame::get_shape(container),
            ty => bail!("Cannot read shape information from type '{ty}'"),
        }
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => {
                DynArray::read_select(container, info).map(ArrayData::Array)
            }
            DataType::CsrMatrix(_, _) => {
                DynIndSparseMatrix::read_select(container, info).map(ArrayData::CsrMatrix)
            }
            DataType::CscMatrix(_, _) => {
                DynIndSparseMatrix::read_select(container, info).map(ArrayData::CscMatrix)
            }
            DataType::DataFrame => {
                DataFrame::read_select(container, info).map(ArrayData::DataFrame)
            }
            ty => bail!("Cannot read type '{ty:?}' as matrix data"),
        }
    }
}
impl WritableArray for ArrayData {}

impl WritableArray for &ArrayData {}

// Helper

// fn read_csr<B: Backend>(container: &DataContainer<B>) -> Result<ArrayData> {
//     fn _read_csr<B: Backend, T: BackendData>(container: &DataContainer<B>) -> Result<ArrayData>
//     where
//         CsrMatrix<T>: Into<ArrayData>,
//         CsrNonCanonical<T>: Into<ArrayData>,
//     {
//         let group = container.as_group()?;
//         let shape: Vec<u64> = group.get_attr("shape")?;
//         let data = group
//             .open_dataset("data")?
//             .read_array::<_, Ix1>()?
//             .into_raw_vec_and_offset()
//             .0;
//         let indptr: Vec<usize> = group
//             .open_dataset("indptr")?
//             .read_array_cast::<_, Ix1>()?
//             .into_raw_vec_and_offset()
//             .0;
//         let indices: Vec<usize> = group
//             .open_dataset("indices")?
//             .read_array_cast::<_, Ix1>()?
//             .into_raw_vec_and_offset()
//             .0;
//         from_csr_data::<T>(shape[0] as usize, shape[1] as usize, indptr, indices, data)
//     }

//     match container {
//         DataContainer::Group(group) => match group.open_dataset("data")?.dtype()? {
//             ScalarType::I8 => _read_csr::<B, i8>(container),
//             ScalarType::I16 => _read_csr::<B, i16>(container),
//             ScalarType::I32 => _read_csr::<B, i32>(container),
//             ScalarType::I64 => _read_csr::<B, i64>(container),
//             ScalarType::U8 => _read_csr::<B, u8>(container),
//             ScalarType::U16 => _read_csr::<B, u16>(container),
//             ScalarType::U32 => _read_csr::<B, u32>(container),
//             ScalarType::U64 => _read_csr::<B, u64>(container),
//             ScalarType::F32 => _read_csr::<B, f32>(container),
//             ScalarType::F64 => _read_csr::<B, f64>(container),
//             ScalarType::Bool => _read_csr::<B, bool>(container),
//             ScalarType::String => _read_csr::<B, String>(container),
//         },
//         _ => bail!("cannot read csr matrix from non-group container"),
//     }
// }

// fn read_csr_select<B: Backend, S>(container: &DataContainer<B>, info: &[S]) -> Result<ArrayData>
// where
//     B: Backend,
//     S: AsRef<SelectInfoElem>,
// {
//     fn _read_csr<B: Backend, T: BackendData, S>(
//         container: &DataContainer<B>,
//         info: &[S],
//     ) -> Result<ArrayData>
//     where
//         CsrMatrix<T>: Into<ArrayData>,
//         CsrNonCanonical<T>: Into<ArrayData>,
//         S: AsRef<SelectInfoElem>,
//     {
//         if info.as_ref().len() != 2 {
//             panic!("index must have length 2");
//         }

//         if info.iter().all(|s| s.as_ref().is_full()) {
//             return read_csr(container);
//         }

//         let data = if let SelectInfoElem::Slice(s) = info[0].as_ref() {
//             let group = container.as_group()?;
//             let shape: Vec<u64> = group.get_attr("shape")?;
//             let indptr_slice = if let Some(end) = s.end {
//                 SelectInfoElem::from(s.start..end + 1)
//             } else {
//                 SelectInfoElem::from(s.start..)
//             };
//             let mut indptr: Vec<usize> = group
//                 .open_dataset("indptr")?
//                 .read_array_slice_cast(&[indptr_slice])?
//                 .to_vec();
//             let lo = indptr[0];
//             let slice = SelectInfoElem::from(lo..indptr[indptr.len() - 1]);
//             let data: Vec<T> = group
//                 .open_dataset("data")?
//                 .read_array_slice(&[&slice])?
//                 .to_vec();
//             let indices: Vec<usize> = group
//                 .open_dataset("indices")?
//                 .read_array_slice_cast(&[&slice])?
//                 .to_vec();
//             indptr.iter_mut().for_each(|x| *x -= lo);

//             from_csr_data::<T>(indptr.len() - 1, shape[1] as usize, indptr, indices, data)
//                 .unwrap()
//                 .select_axis(1, info[1].as_ref())
//         } else {
//             read_csr(container)?.select(info)
//         };
//         Ok(data)
//     }

//     match container {
//         DataContainer::Group(group) => match group.open_dataset("data")?.dtype()? {
//             ScalarType::I8 => _read_csr::<B, i8, _>(container, info),
//             ScalarType::I16 => _read_csr::<B, i16, _>(container, info),
//             ScalarType::I32 => _read_csr::<B, i32, _>(container, info),
//             ScalarType::I64 => _read_csr::<B, i64, _>(container, info),
//             ScalarType::U8 => _read_csr::<B, u8, _>(container, info),
//             ScalarType::U16 => _read_csr::<B, u16, _>(container, info),
//             ScalarType::U32 => _read_csr::<B, u32, _>(container, info),
//             ScalarType::U64 => _read_csr::<B, u64, _>(container, info),
//             ScalarType::F32 => _read_csr::<B, f32, _>(container, info),
//             ScalarType::F64 => _read_csr::<B, f64, _>(container, info),
//             ScalarType::Bool => _read_csr::<B, bool, _>(container, info),
//             ScalarType::String => _read_csr::<B, String, _>(container, info),
//         },
//         _ => bail!("cannot read csr matrix from non-group container"),
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::CsMatI;

    #[test]
    fn test_dyn_ind_sparse_matrix_csr_csc_conversions() {
        let csr: CsMatI<f64, u32, u64> =
            CsMatI::new_csc((3, 3), vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let data: ArrayData = csr.into();
        assert!(matches!(data, ArrayData::CscMatrix(_)));

        let csr2: CsMatI<f64, u32, u64> =
            CsMatI::new((3, 3), vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let data2: ArrayData = csr2.into();
        assert!(matches!(data2, ArrayData::CsrMatrix(_)));
    }

    #[test]
    fn test_dyn_ind_sparse_matrix_try_from_arraydata() {
        let csr: CsMatI<f64, u32, u64> =
            CsMatI::new((3, 3), vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let data: ArrayData = csr.clone().into();
        let extracted: CsMatI<f64, u32, u64> = CsMatI::try_from(data).unwrap();
        assert_eq!(csr.indptr(), extracted.indptr());
        assert_eq!(csr.indices(), extracted.indices());
        assert_eq!(csr.data(), extracted.data());
    }

    #[test]
    fn test_arraydata_stackable_vstack_sprs() {
        let csr1: CsMatI<f64, u32, u64> =
            CsMatI::new((2, 3), vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]);
        let csr2: CsMatI<f64, u32, u64> = CsMatI::new((1, 3), vec![0, 1], vec![2], vec![3.0]);
        let d1: ArrayData = csr1.into();
        let d2: ArrayData = csr2.into();

        let stacked = ArrayData::vstack(vec![d1, d2].into_iter()).unwrap();

        if let ArrayData::CsrMatrix(DynIndSparseMatrix::U32(DynSparseMatrix::F64(m))) = stacked {
            assert_eq!(m.rows(), 3);
            assert_eq!(m.cols(), 3);
            assert_eq!(m.indptr().as_slice().unwrap(), &[0, 1, 2, 3]);
            assert_eq!(m.indices(), &[0, 1, 2]);
            assert_eq!(m.data(), &[1.0, 2.0, 3.0]);
        } else {
            panic!("Expected CsrMatrix of correct type");
        }
    }
}
