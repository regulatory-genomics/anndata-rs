use crate::backend::{AttributeOp, Backend, BackendData, DataContainer, GroupOp, ScalarType};
use crate::data::SparseMatrixLayoutE;
use crate::data::{ArrayData, array::DynArray, array::utils::ExtendableDataset};
use crate::{ArrayElem, Selectable};

use super::{CsrNonCanonical, DynCsrNonCanonical, DynIndSparseMatrix, DynSparseMatrix};
use crate::backend::get_default_write_config;
use anyhow::{Context, Result, bail};
use ndarray::{Array, Array1, ArrayD, ArrayView1, RemoveAxis};
use sprs::{CsMatI, SpIndex};

pub enum MatrixBuilder<B: Backend> {
    CsrMatrix(CsrMatrixBuilder<B>),
    Array(ArrayBuilder<B>),
}

impl<B: Backend> MatrixBuilder<B> {
    pub fn new_dense<G: GroupOp<B>>(location: &G, name: &str, dtype: ScalarType) -> Result<Self> {
        Ok(Self::Array(ArrayBuilder::new(location, name, dtype, 2)?))
    }

    pub fn new_sparse<G: GroupOp<B>>(location: &G, name: &str, dtype: ScalarType) -> Result<Self> {
        Ok(Self::CsrMatrix(CsrMatrixBuilder::new(
            location, name, dtype,
        )?))
    }

    pub fn add(&mut self, csr: ArrayData) -> Result<()> {
        match self {
            Self::CsrMatrix(builder) => builder.add(csr),
            Self::Array(builder) => builder.add(csr),
        }
    }

    pub fn finish(self) -> Result<ArrayElem<B>> {
        let container = match self {
            Self::CsrMatrix(builder) => builder.finish(),
            Self::Array(builder) => builder.finish(),
        }?;
        ArrayElem::try_from(container)
    }
}

pub struct CsrMatrixBuilder<B: Backend> {
    indices: ExtendableDataset<B, i64>,
    data: ArrayBuilder<B>,
    indptr: Vec<u64>,
    num_rows: usize,
    num_cols: Option<usize>,
    nnz: u64,
    group: B::Group,
}

impl<B: Backend> CsrMatrixBuilder<B> {
    fn new<G: GroupOp<B>>(location: &G, name: &str, dtype: ScalarType) -> Result<Self> {
        let mut group = location.new_group(name)?;
        group.new_attr("encoding-type", "csr_matrix")?;
        group.new_attr("encoding-version", "0.1.0")?;
        group.new_attr("h5sparse_format", "csr")?;

        let data = ArrayBuilder::new(&group, "data", dtype, 1)?;
        let indices: ExtendableDataset<B, i64> =
            ExtendableDataset::with_capacity(&group, "indices", 1000.into())?;

        Ok(Self {
            indices,
            data,
            indptr: Vec::new(),
            num_rows: 0,
            num_cols: None,
            nnz: 0,
            group,
        })
    }

    fn add(&mut self, csr: ArrayData) -> Result<()> {
        fn helper<B, T, V>(builder: &mut CsrMatrixBuilder<B>, mat: CsMatI<V, T, u64>) -> Result<()>
        where
            B: Backend,
            T: SpIndex + BackendData,
            V: BackendData + Clone,
            ArrayData: From<Array1<V>>,
        {
            let c = mat.cols();
            if builder.num_cols.is_none() {
                builder.num_cols = Some(c);
            }
            if builder.num_cols.unwrap() == c {
                builder.num_rows += mat.rows();

                let ipt = mat.indptr();
                let indptr = ipt.as_slice().unwrap();
                let indices = mat.indices();
                let data = mat.data();

                indptr[..indptr.len() - 1].iter().for_each(|x| {
                    builder.indptr.push(*x + builder.nnz);
                });

                builder.nnz += *indptr.last().unwrap_or(&0);

                builder.data.add(Array::from_vec(data.to_vec()).into())?;

                builder.indices.extend(
                    0,
                    Array::from_vec(indices.iter().map(|&x| x.to_i64().unwrap()).collect()).view(),
                )?;

                Ok(())
            } else {
                bail!("All matrices must have the same number of columns");
            }
        }

        macro_rules! match_ind_sparse {
            ($mat:expr, $self:expr) => {
                match $mat {
                    DynIndSparseMatrix::I16(inner) => match_sparse!(inner, $self),
                    DynIndSparseMatrix::I32(inner) => match_sparse!(inner, $self),
                    DynIndSparseMatrix::I64(inner) => match_sparse!(inner, $self),
                    DynIndSparseMatrix::U16(inner) => match_sparse!(inner, $self),
                    DynIndSparseMatrix::U32(inner) => match_sparse!(inner, $self),
                    DynIndSparseMatrix::U64(inner) => match_sparse!(inner, $self),
                }
            };
        }

        macro_rules! match_sparse {
            ($mat:expr, $self:expr) => {
                match $mat {
                    DynSparseMatrix::I8(m) => helper($self, m),
                    DynSparseMatrix::I16(m) => helper($self, m),
                    DynSparseMatrix::I32(m) => helper($self, m),
                    DynSparseMatrix::I64(m) => helper($self, m),
                    DynSparseMatrix::U8(m) => helper($self, m),
                    DynSparseMatrix::U16(m) => helper($self, m),
                    DynSparseMatrix::U32(m) => helper($self, m),
                    DynSparseMatrix::U64(m) => helper($self, m),
                    DynSparseMatrix::F32(m) => helper($self, m),
                    DynSparseMatrix::F64(m) => helper($self, m),
                    DynSparseMatrix::Bool(m) => helper($self, m),
                    DynSparseMatrix::String(m) => helper($self, m),
                }
            };
        }

        match csr {
            ArrayData::CsrMatrix(mat) => match_ind_sparse!(mat, self),
            _ => bail!("Expected CsrMatrix"),
        }?;
        Ok(())
    }

    fn finish(mut self) -> Result<DataContainer<B>> {
        self.indices.finish()?;
        self.data.finish()?;
        self.indptr.push(self.nnz);
        self.group
            .new_array_dataset("indptr", self.indptr.into(), get_default_write_config())?;
        self.group.new_attr(
            "shape",
            [self.num_rows as u64, self.num_cols.unwrap_or(0) as u64].as_slice(),
        )?;
        Ok(DataContainer::Group(self.group))
    }
}

pub enum ArrayBuilder<B: Backend> {
    U8(ExtendableDataset<B, u8>),
    U16(ExtendableDataset<B, u16>),
    U32(ExtendableDataset<B, u32>),
    U64(ExtendableDataset<B, u64>),
    I8(ExtendableDataset<B, i8>),
    I16(ExtendableDataset<B, i16>),
    I32(ExtendableDataset<B, i32>),
    I64(ExtendableDataset<B, i64>),
    F32(ExtendableDataset<B, f32>),
    F64(ExtendableDataset<B, f64>),
    Bool(ExtendableDataset<B, bool>),
    String(ExtendableDataset<B, String>),
}

impl<B: Backend> ArrayBuilder<B> {
    fn new<G>(location: &G, name: &str, dtype: ScalarType, ndim: usize) -> Result<Self>
    where
        G: GroupOp<B>,
    {
        let chunk_size = vec![1000; ndim].into();
        let data = match dtype {
            ScalarType::U8 => Self::U8(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::U16 => Self::U16(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::U32 => Self::U32(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::U64 => Self::U64(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::I8 => Self::I8(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::I16 => Self::I16(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::I32 => Self::I32(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::I64 => Self::I64(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::F32 => Self::F32(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::F64 => Self::F64(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::Bool => Self::Bool(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::String => Self::String(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
        };
        Ok(data)
    }

    fn add(&mut self, array: ArrayData) -> Result<()> {
        match self {
            Self::U8(data) => data.extend(0, ArrayD::<u8>::try_from(array)?.view()),
            Self::U16(data) => data.extend(0, ArrayD::<u16>::try_from(array)?.view()),
            Self::U32(data) => data.extend(0, ArrayD::<u32>::try_from(array)?.view()),
            Self::U64(data) => data.extend(0, ArrayD::<u64>::try_from(array)?.view()),
            Self::I8(data) => data.extend(0, ArrayD::<i8>::try_from(array)?.view()),
            Self::I16(data) => data.extend(0, ArrayD::<i16>::try_from(array)?.view()),
            Self::I32(data) => data.extend(0, ArrayD::<i32>::try_from(array)?.view()),
            Self::I64(data) => data.extend(0, ArrayD::<i64>::try_from(array)?.view()),
            Self::F32(data) => data.extend(0, ArrayD::<f32>::try_from(array)?.view()),
            Self::F64(data) => data.extend(0, ArrayD::<f64>::try_from(array)?.view()),
            Self::Bool(data) => data.extend(0, ArrayD::<bool>::try_from(array)?.view()),
            Self::String(data) => data.extend(0, ArrayD::<String>::try_from(array)?.view()),
        }
    }

    fn finish(self) -> Result<DataContainer<B>> {
        let (dataset, encoding_type) = match self {
            Self::U8(data) => (data.finish()?, "array"),
            Self::U16(data) => (data.finish()?, "array"),
            Self::U32(data) => (data.finish()?, "array"),
            Self::U64(data) => (data.finish()?, "array"),
            Self::I8(data) => (data.finish()?, "array"),
            Self::I16(data) => (data.finish()?, "array"),
            Self::I32(data) => (data.finish()?, "array"),
            Self::I64(data) => (data.finish()?, "array"),
            Self::F32(data) => (data.finish()?, "array"),
            Self::F64(data) => (data.finish()?, "array"),
            Self::Bool(data) => (data.finish()?, "array"),
            Self::String(data) => (data.finish()?, "string-array"),
        };
        let mut container = DataContainer::<B>::Dataset(dataset);
        container.new_attr("encoding-type", encoding_type)?;
        container.new_attr("encoding-version", "0.2.0")?;
        Ok(container)
    }
}

pub trait ArrayChunk: Selectable {
    fn write_by_chunk<B, G, I>(
        iter: I,
        location: &G,
        name: &str,
        format: Option<SparseMatrixLayoutE>,
    ) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>;
}

impl ArrayChunk for ArrayData {
    fn write_by_chunk<B, G, I>(
        iter: I,
        location: &G,
        name: &str,
        _format: Option<SparseMatrixLayoutE>,
    ) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().context("input iterator is empty")? {
            ArrayData::Array(_) => {
                DynArray::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name, None)
            }
            ArrayData::CsrMatrix(_) | ArrayData::CsrNonCanonical(_) => {
                DynIndSparseMatrix::write_by_chunk(
                    iter.map(|x| x.try_into().unwrap()),
                    location,
                    name,
                    Some(SparseMatrixLayoutE::CSR),
                )
            }
            ArrayData::CscMatrix(_) => DynIndSparseMatrix::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                Some(SparseMatrixLayoutE::CSC),
            ),
            ArrayData::DataFrame(_) => todo!(),
        }
    }
}

impl ArrayChunk for DynArray {
    fn write_by_chunk<B, G, I>(
        iter: I,
        location: &G,
        name: &str,
        _format: Option<SparseMatrixLayoutE>,
    ) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().context("input iterator is empty")? {
            DynArray::U8(_) => ArrayD::<u8>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynArray::U16(_) => ArrayD::<u16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynArray::U32(_) => ArrayD::<u32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynArray::U64(_) => ArrayD::<u64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynArray::I8(_) => ArrayD::<i8>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynArray::I16(_) => ArrayD::<i16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynArray::I32(_) => ArrayD::<i32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynArray::I64(_) => ArrayD::<i64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynArray::F32(_) => ArrayD::<f32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynArray::F64(_) => ArrayD::<f64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynArray::Bool(_) => ArrayD::<bool>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynArray::String(_) => ArrayD::<String>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
        }
    }
}

impl<D: RemoveAxis, T: BackendData> ArrayChunk for Array<T, D> {
    fn write_by_chunk<B, G, I>(
        iter: I,
        location: &G,
        name: &str,
        _format: Option<SparseMatrixLayoutE>,
    ) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();
        let chunk_size = iter
            .peek()
            .unwrap()
            .shape()
            .iter()
            .map(|&x| x.min(1000))
            .collect();
        let mut data: ExtendableDataset<B, T> =
            ExtendableDataset::with_capacity(location, name, chunk_size)?;

        iter.try_for_each(|x| data.extend(0, x.view()))?;
        let dataset = data.finish()?;
        let encoding_type = if T::DTYPE == ScalarType::String {
            "string-array"
        } else {
            "array"
        };
        let mut container = DataContainer::<B>::Dataset(dataset);
        container.new_attr("encoding-type", encoding_type)?;
        container.new_attr("encoding-version", "0.2.0")?;
        Ok(container)
    }
}

impl ArrayChunk for DynCsrNonCanonical {
    fn write_by_chunk<B, G, I>(
        iter: I,
        location: &G,
        name: &str,
        _format: Option<SparseMatrixLayoutE>,
    ) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().context("input iterator is empty")? {
            DynCsrNonCanonical::U8(_) => CsrNonCanonical::<u8>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynCsrNonCanonical::U16(_) => CsrNonCanonical::<u16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynCsrNonCanonical::U32(_) => CsrNonCanonical::<u32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynCsrNonCanonical::U64(_) => CsrNonCanonical::<u64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynCsrNonCanonical::I8(_) => CsrNonCanonical::<i8>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynCsrNonCanonical::I16(_) => CsrNonCanonical::<i16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynCsrNonCanonical::I32(_) => CsrNonCanonical::<i32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynCsrNonCanonical::I64(_) => CsrNonCanonical::<i64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynCsrNonCanonical::F32(_) => CsrNonCanonical::<f32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynCsrNonCanonical::F64(_) => CsrNonCanonical::<f64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynCsrNonCanonical::Bool(_) => CsrNonCanonical::<bool>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
            DynCsrNonCanonical::String(_) => CsrNonCanonical::<String>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                None,
            ),
        }
    }
}

impl<T: BackendData> ArrayChunk for CsrNonCanonical<T> {
    fn write_by_chunk<B, G, I>(
        mut iter: I,
        location: &G,
        name: &str,
        _format: Option<SparseMatrixLayoutE>,
    ) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut group = location.new_group(name)?;
        group.new_attr("encoding-type", "csr_matrix")?;
        group.new_attr("encoding-version", "0.1.0")?;
        group.new_attr("h5sparse_format", "csr")?;

        let mut data: ExtendableDataset<B, T> =
            ExtendableDataset::with_capacity(&group, "data", 1000.into())?;
        let mut indices: ExtendableDataset<B, i64> =
            ExtendableDataset::with_capacity(&group, "indices", 1000.into())?;
        let mut indptr: Vec<i64> = Vec::new();
        let mut num_rows = 0;
        let mut num_cols: Option<usize> = None;
        let mut nnz = 0;

        iter.try_for_each(|csr| {
            let c = csr.ncols();
            if num_cols.is_none() {
                num_cols = Some(c);
            }
            if num_cols.unwrap() == c {
                num_rows += csr.nrows();
                let (indptr_, indices_, data_) = csr.csr_data();
                indptr_[..indptr_.len() - 1]
                    .iter()
                    .for_each(|x| indptr.push(i64::try_from(*x).unwrap() + nnz));
                nnz += *indptr_.last().unwrap_or(&0) as i64;
                data.extend(0, ArrayView1::from_shape(data_.len(), data_)?)?;
                indices.extend(
                    0,
                    ArrayView1::from_shape(indices_.len(), indices_)?
                        .mapv(|x| i64::try_from(x).unwrap())
                        .view(),
                )
            } else {
                bail!("All matrices must have the same number of columns");
            }
        })?;

        indices.finish()?;
        data.finish()?;
        indptr.push(nnz);
        group.new_array_dataset("indptr", indptr.into(), get_default_write_config())?;
        group.new_attr(
            "shape",
            [num_rows as u64, num_cols.unwrap_or(0) as u64].as_slice(),
        )?;
        Ok(DataContainer::Group(group))
    }
}

impl ArrayChunk for DynIndSparseMatrix {
    fn write_by_chunk<B, G, I>(
        iter: I,
        location: &G,
        name: &str,
        format: Option<SparseMatrixLayoutE>,
    ) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();

        match iter.peek().context("Input iterator is empty")? {
            DynIndSparseMatrix::I16(_) => DynSparseMatrix::<i16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                format,
            ),
            DynIndSparseMatrix::I32(_) => DynSparseMatrix::<i32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                format,
            ),
            DynIndSparseMatrix::I64(_) => DynSparseMatrix::<i64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                format,
            ),
            DynIndSparseMatrix::U16(_) => DynSparseMatrix::<u16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                format,
            ),
            DynIndSparseMatrix::U32(_) => DynSparseMatrix::<u32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                format,
            ),
            DynIndSparseMatrix::U64(_) => DynSparseMatrix::<u64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
                format,
            ),
        }
    }
}

impl<T: BackendData + SpIndex + num::Integer + num::FromPrimitive> ArrayChunk
    for DynSparseMatrix<T>
{
    fn write_by_chunk<B, G, I>(
        iter: I,
        location: &G,
        name: &str,
        format: Option<SparseMatrixLayoutE>,
    ) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();

        macro_rules! write_sparse {
            ($type:ty) => {
                CsMatI::<$type, T, u64>::write_by_chunk(
                    iter.map(|x| x.try_into().unwrap()),
                    location,
                    name,
                    format,
                )
            };
        }

        match iter.peek().context("iterator is empty")? {
            DynSparseMatrix::I8(_) => write_sparse!(i8),
            DynSparseMatrix::I16(_) => write_sparse!(i16),
            DynSparseMatrix::I32(_) => write_sparse!(i32),
            DynSparseMatrix::I64(_) => write_sparse!(i64),
            DynSparseMatrix::U8(_) => write_sparse!(u8),
            DynSparseMatrix::U16(_) => write_sparse!(u16),
            DynSparseMatrix::U32(_) => write_sparse!(u32),
            DynSparseMatrix::U64(_) => write_sparse!(u64),
            DynSparseMatrix::F32(_) => write_sparse!(f32),
            DynSparseMatrix::F64(_) => write_sparse!(f64),
            DynSparseMatrix::Bool(_) => write_sparse!(bool),
            DynSparseMatrix::String(_) => write_sparse!(String),
        }
    }
}

impl<N: BackendData + std::fmt::Debug, T: BackendData + SpIndex + num::Integer + num::FromPrimitive>
    ArrayChunk for CsMatI<N, T, u64>
{
    fn write_by_chunk<B, G, I>(
        mut iter: I,
        location: &G,
        name: &str,
        format: Option<SparseMatrixLayoutE>,
    ) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let format = match format {
            Some(f) => f,
            None => bail!("Writing a sparse matrix requires passing a format type!"),
        };

        let (encoding, format) = match format {
            SparseMatrixLayoutE::CSR => ("csr_matrix", "csr"),
            SparseMatrixLayoutE::CSC => ("csc_matrix", "csc"),
            _ => {
                bail!("Writing a sparse matrix requires the type to be CSR or CSC, not {format:?}")
            }
        };

        let mut group = location.new_group(name)?;
        group.new_attr("encoding-type", encoding)?;
        group.new_attr("encoding-version", "0.1.0")?;
        group.new_attr("h5sparse_format", format)?;

        let mut indptr: ExtendableDataset<B, u64> =
            ExtendableDataset::with_capacity(&group, "indptr", 1000.into())?;
        let mut data: ExtendableDataset<B, N> =
            ExtendableDataset::with_capacity(&group, "data", 1000.into())?;
        let mut indices: ExtendableDataset<B, T> =
            ExtendableDataset::with_capacity(&group, "indices", 1000.into())?;

        let mut num_rows = 0;
        let mut num_cols: Option<usize> = None;
        let mut nnz = 0;

        let mut indptr_workspace: Vec<u64> = Vec::new();

        iter.try_for_each(|mat| {
            if format != if mat.is_csr() { "csr" } else { "csc" } {
                bail!("All matrices must have the same layout for now!");
            }

            if format == "csc" {
                let r = mat.rows();
                if num_rows == 0 {
                    num_rows = r;
                } else if num_rows != r {
                    bail!("All matrices must have the same number of rows");
                }

                let c = mat.cols();
                let prev_cols = num_cols.unwrap_or(0);
                num_cols = Some(prev_cols + c);

                let in_ptr_raw = mat.indptr();
                let (indptr_, indices_, data_) =
                    (in_ptr_raw.raw_storage(), mat.indices(), mat.data());

                indptr_workspace.clear();
                indptr_workspace.extend(indptr_[..indptr_.len() - 1].iter().map(|&x| x + nnz));
                indptr.extend(
                    0,
                    ArrayView1::from_shape(indptr_workspace.len(), &indptr_workspace)?,
                )?;

                nnz += *indptr_.last().unwrap_or(&0);

                data.extend(0, ArrayView1::from_shape(data_.len(), data_)?)?;

                indices.extend(0, ArrayView1::from_shape(indices_.len(), indices_)?)?;
            } else if format == "csr" {
                let c = mat.cols();
                if num_cols.is_none() {
                    num_cols = Some(c);
                }
                if num_cols.unwrap() == c {
                    num_rows += mat.rows();
                    let in_ptr_raw = mat.indptr();
                    let (indptr_, indices_, data_) =
                        (in_ptr_raw.raw_storage(), mat.indices(), mat.data());

                    indptr_workspace.clear();
                    indptr_workspace.extend(indptr_[..indptr_.len() - 1].iter().map(|&x| x + nnz));
                    indptr.extend(
                        0,
                        ArrayView1::from_shape(indptr_workspace.len(), &indptr_workspace)?,
                    )?;

                    nnz += *indptr_.last().unwrap_or(&0);

                    data.extend(0, ArrayView1::from_shape(data_.len(), data_)?)?;

                    indices.extend(0, ArrayView1::from_shape(indices_.len(), indices_)?)?;
                } else {
                    bail!("All matrices must have the same number of columns");
                }
            }
            Ok(())
        })?;

        // Write the final nnz value for indptr
        indptr_workspace.clear();
        indptr_workspace.push(nnz);
        indptr.extend(
            0,
            ArrayView1::from_shape(indptr_workspace.len(), &indptr_workspace)?,
        )?;
        indptr.finish()?;
        indices.finish()?;
        data.finish()?;

        group.new_attr(
            "shape",
            [num_rows as u64, num_cols.unwrap_or(0) as u64].as_slice(),
        )?;
        Ok(DataContainer::Group(group))
    }
}
