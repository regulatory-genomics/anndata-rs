use anndata::{
    backend::*,
    data::{DynArray, DynCowArray, DynScalar, SelectInfoBounds, SelectInfoElem, Shape},
};

use anyhow::{Ok, Result, bail};
use hdf5::{
    File, Group, H5Type, Location, Selection,
    dataset::Dataset,
    types::IntSize::*,
    types::{FloatSize, TypeDescriptor, VarLenUnicode},
};
use ndarray::{Array, ArrayD, ArrayView, CowArray, Dimension, IxDyn, SliceInfo};
use std::ops::Deref;
use std::path::{Path, PathBuf};

///////////////////////////////////////////////////////////////////////////////
// Type definitions
///////////////////////////////////////////////////////////////////////////////

pub struct H5;

pub struct H5File(File);

impl Deref for H5File {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct H5Group(Group);

impl Deref for H5Group {
    type Target = Group;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct H5Dataset(Dataset);

impl Deref for H5Dataset {
    type Target = Dataset;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

///////////////////////////////////////////////////////////////////////////////
// Backend implementation
///////////////////////////////////////////////////////////////////////////////

impl Backend for H5 {
    const NAME: &'static str = "hdf5";

    type Store = H5File;
    type Group = H5Group;
    type Dataset = H5Dataset;

    fn new<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        Ok(H5File(File::create(path)?))
    }

    /// Opens a file as read-only, file must exist.
    fn open<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        Ok(File::open(path).map(H5File)?)
    }

    /// Opens a file as read/write, file must exist.
    fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        Ok(File::open_rw(path).map(H5File)?)
    }
}

impl StoreOp<H5> for H5File {
    fn filename(&self) -> PathBuf {
        hdf5::Location::filename(self).into()
    }

    fn close(self) -> Result<()> {
        Ok(self.0.close()?)
    }
}

// Generic GroupOp implementations

fn list(group: &Group) -> Result<Vec<String>> {
    Ok(group.member_names()?)
}

fn create_group(group: &Group, name: &str) -> Result<H5Group> {
    Ok(H5Group(group.create_group(name)?))
}

fn open_group(group: &Group, name: &str) -> Result<H5Group> {
    Ok(H5Group(group.group(name)?))
}

fn new_dataset<T: BackendData>(
    group: &Group,
    name: &str,
    shape: &Shape,
    config: WriteConfig,
) -> Result<H5Dataset> {
    let dtype = T::DTYPE;
    let mut builder = match dtype {
        ScalarType::U8 => group.new_dataset::<u8>(),
        ScalarType::U16 => group.new_dataset::<u16>(),
        ScalarType::U32 => group.new_dataset::<u32>(),
        ScalarType::U64 => group.new_dataset::<u64>(),
        ScalarType::I8 => group.new_dataset::<i8>(),
        ScalarType::I16 => group.new_dataset::<i16>(),
        ScalarType::I32 => group.new_dataset::<i32>(),
        ScalarType::I64 => group.new_dataset::<i64>(),
        ScalarType::F32 => group.new_dataset::<f32>(),
        ScalarType::F64 => group.new_dataset::<f64>(),
        ScalarType::Bool => group.new_dataset::<bool>(),
        ScalarType::String => group.new_dataset::<VarLenUnicode>(),
    };

    builder = if let Some(compression) = config.compression {
        match compression {
            Compression::Gzip(lvl) => builder.deflate(lvl),
            Compression::Zst(lvl) => match dtype {
                ScalarType::String => builder.deflate(3),
                _ => builder.blosc_zstd(lvl, hdf5::filters::BloscShuffle::Byte),
            },
        }
    } else {
        builder
    };

    builder = if let Some(s) = config.block_size {
        if s.as_ref().iter().all(|&x| x > 0) {
            builder.chunk(s.as_ref())
        } else {
            builder
        }
    } else {
        builder
    };

    let s: hdf5::Extents = hdf5::SimpleExtents::resizable(shape.as_ref()).into();
    let dataset = builder.shape(s).create(name)?;
    Ok(H5Dataset(dataset))
}

fn open_dataset(group: &Group, name: &str) -> Result<H5Dataset> {
    Ok(H5Dataset(group.dataset(name)?))
}

fn delete(group: &Group, name: &str) -> Result<()> {
    Ok(group.unlink(name)?)
}

fn exists(group: &Group, name: &str) -> Result<bool> {
    Ok(group.link_exists(name))
}

fn create_scalar_data<D: BackendData>(group: &Group, name: &str, data: &D) -> Result<H5Dataset> {
    match data.as_dyn() {
        DynScalar::U8(x) => {
            let dataset = group.new_dataset::<u8>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::U16(x) => {
            let dataset = group.new_dataset::<u16>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::U32(x) => {
            let dataset = group.new_dataset::<u32>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::U64(x) => {
            let dataset = group.new_dataset::<u64>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::I8(x) => {
            let dataset = group.new_dataset::<i8>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::I16(x) => {
            let dataset = group.new_dataset::<i16>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::I32(x) => {
            let dataset = group.new_dataset::<i32>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::I64(x) => {
            let dataset = group.new_dataset::<i64>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::F32(x) => {
            let dataset = group.new_dataset::<f32>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::F64(x) => {
            let dataset = group.new_dataset::<f64>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::Bool(x) => {
            let dataset = group.new_dataset::<bool>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::String(x) => {
            let dataset = group.new_dataset::<VarLenUnicode>().create(name)?;
            dataset.write_scalar(&x.parse::<VarLenUnicode>().unwrap())?;
            Ok(dataset)
        }
    }
    .map(H5Dataset)
}

impl DatasetOp<H5> for H5Dataset {
    fn dtype(&self) -> Result<ScalarType> {
        let ty = match hdf5::Container::dtype(self)?.to_descriptor()? {
            TypeDescriptor::Unsigned(U1) => ScalarType::U8,
            TypeDescriptor::Unsigned(U2) => ScalarType::U16,
            TypeDescriptor::Unsigned(U4) => ScalarType::U32,
            TypeDescriptor::Unsigned(U8) => ScalarType::U64,
            TypeDescriptor::Integer(U1) => ScalarType::I8,
            TypeDescriptor::Integer(U2) => ScalarType::I16,
            TypeDescriptor::Integer(U4) => ScalarType::I32,
            TypeDescriptor::Integer(U8) => ScalarType::I64,
            TypeDescriptor::Float(FloatSize::U4) => ScalarType::F32,
            TypeDescriptor::Float(FloatSize::U8) => ScalarType::F64,
            TypeDescriptor::Boolean => ScalarType::Bool,
            TypeDescriptor::VarLenAscii => ScalarType::String,
            TypeDescriptor::VarLenUnicode => ScalarType::String,
            ty => bail!("Unsupported type: {ty:?}"),
        };
        Ok(ty)
    }

    fn shape(&self) -> Shape {
        hdf5::Container::shape(self).into()
    }

    fn reshape(&mut self, shape: &Shape) -> Result<()> {
        Ok(Dataset::resize(self, shape.as_ref())?)
    }

    fn read_scalar<T: BackendData>(&self) -> Result<T> {
        let val = match T::DTYPE {
            ScalarType::Bool => self.deref().read_scalar::<bool>()?.as_dyn(),
            ScalarType::U8 => self.deref().read_scalar::<u8>()?.as_dyn(),
            ScalarType::U16 => self.deref().read_scalar::<u16>()?.as_dyn(),
            ScalarType::U32 => self.deref().read_scalar::<u32>()?.as_dyn(),
            ScalarType::U64 => self.deref().read_scalar::<u64>()?.as_dyn(),
            ScalarType::I8 => self.deref().read_scalar::<i8>()?.as_dyn(),
            ScalarType::I16 => self.deref().read_scalar::<i16>()?.as_dyn(),
            ScalarType::I32 => self.deref().read_scalar::<i32>()?.as_dyn(),
            ScalarType::I64 => self.deref().read_scalar::<i64>()?.as_dyn(),
            ScalarType::F32 => self.deref().read_scalar::<f32>()?.as_dyn(),
            ScalarType::F64 => self.deref().read_scalar::<f64>()?.as_dyn(),
            ScalarType::String => {
                let s = self.deref().read_scalar::<VarLenUnicode>()?;
                s.to_string().as_dyn()
            }
        };
        BackendData::from_dyn(val)
    }

    fn read_array<T: BackendData, D>(&self) -> Result<Array<T, D>>
    where
        D: Dimension,
    {
        macro_rules! read_typed {
            ($ty:ty) => {{
                let arr: DynArray = self.deref().read::<$ty, D>()?.into_dyn().into();
                T::from_dyn_arr(arr)?
                    .into_dimensionality::<D>()
                    .map_err(Into::into)
            }};
        }

        match T::DTYPE {
            ScalarType::I8 => read_typed!(i8),
            ScalarType::I16 => read_typed!(i16),
            ScalarType::I32 => read_typed!(i32),
            ScalarType::I64 => read_typed!(i64),
            ScalarType::U8 => read_typed!(u8),
            ScalarType::U16 => read_typed!(u16),
            ScalarType::U32 => read_typed!(u32),
            ScalarType::U64 => read_typed!(u64),
            ScalarType::F32 => read_typed!(f32),
            ScalarType::F64 => read_typed!(f64),
            ScalarType::Bool => read_typed!(bool),
            ScalarType::String => {
                let arr = self.deref().read::<VarLenUnicode, D>()?;
                let shape = arr.raw_dim();
                let vec = arr.into_raw_vec_and_offset().0;
                let strings: Vec<String> = vec.into_iter().map(|s| s.to_string()).collect();
                let arr: DynArray = Array::from_shape_vec(shape, strings)
                    .unwrap()
                    .into_dyn()
                    .into();
                T::from_dyn_arr(arr)?
                    .into_dimensionality::<D>()
                    .map_err(Into::into)
            }
        }
    }

    fn read_dyn_array(&self) -> Result<DynArray> {
        match self.dtype()? {
            ScalarType::I8 => Ok(self.deref().read::<i8, IxDyn>()?.into()),
            ScalarType::I16 => Ok(self.deref().read::<i16, IxDyn>()?.into()),
            ScalarType::I32 => Ok(self.deref().read::<i32, IxDyn>()?.into()),
            ScalarType::I64 => Ok(self.deref().read::<i64, IxDyn>()?.into()),
            ScalarType::U8 => Ok(self.deref().read::<u8, IxDyn>()?.into()),
            ScalarType::U16 => Ok(self.deref().read::<u16, IxDyn>()?.into()),
            ScalarType::U32 => Ok(self.deref().read::<u32, IxDyn>()?.into()),
            ScalarType::U64 => Ok(self.deref().read::<u64, IxDyn>()?.into()),
            ScalarType::F32 => Ok(self.deref().read::<f32, IxDyn>()?.into()),
            ScalarType::F64 => Ok(self.deref().read::<f64, IxDyn>()?.into()),
            ScalarType::Bool => Ok(self.deref().read::<bool, IxDyn>()?.into()),
            ScalarType::String => {
                let arr = self.deref().read::<VarLenUnicode, IxDyn>()?;
                let shape = arr.raw_dim();
                let vec = arr.into_raw_vec_and_offset().0;
                let strings: Vec<String> = vec.into_iter().map(|s| s.to_string()).collect();
                Ok(Array::from_shape_vec(shape, strings).unwrap().into())
            }
        }
    }

    fn read_array_slice<T, S, D>(&self, selection: &[S]) -> Result<Array<T, D>>
    where
        T: BackendData,
        S: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        let (select, shape) = into_selection(selection, self.shape());

        macro_rules! read_slice_typed {
            ($ty:ty) => {{
                let arr = if matches!(select, Selection::Points(_)) {
                    let slice_1d = hdf5::Container::read_slice_1d::<$ty, _>(self, select)?;
                    slice_1d
                        .into_shape_with_order(shape.as_ref())?
                        .into_dimensionality::<D>()?
                } else {
                    hdf5::Container::read_slice::<$ty, _, D>(self, select)?
                };
                let arr: DynArray = arr.into_dyn().into();
                T::from_dyn_arr(arr)?
                    .into_dimensionality::<D>()
                    .map_err(Into::into)
            }};
        }

        match T::DTYPE {
            ScalarType::I8 => read_slice_typed!(i8),
            ScalarType::I16 => read_slice_typed!(i16),
            ScalarType::I32 => read_slice_typed!(i32),
            ScalarType::I64 => read_slice_typed!(i64),
            ScalarType::U8 => read_slice_typed!(u8),
            ScalarType::U16 => read_slice_typed!(u16),
            ScalarType::U32 => read_slice_typed!(u32),
            ScalarType::U64 => read_slice_typed!(u64),
            ScalarType::F32 => read_slice_typed!(f32),
            ScalarType::F64 => read_slice_typed!(f64),
            ScalarType::Bool => read_slice_typed!(bool),
            ScalarType::String => {
                let arr = if matches!(select, Selection::Points(_)) {
                    let slice_1d = self.deref().read_slice_1d::<VarLenUnicode, _>(select)?;
                    slice_1d
                        .into_shape_with_order(shape.as_ref())?
                        .into_dimensionality::<D>()?
                } else {
                    self.deref().read_slice::<VarLenUnicode, _, D>(select)?
                };
                let shape_d = arr.raw_dim();
                let vec = arr.into_raw_vec_and_offset().0;
                let strings: Vec<String> = vec.into_iter().map(|s| s.to_string()).collect();
                let arr: DynArray = Array::from_shape_vec(shape_d, strings)
                    .unwrap()
                    .into_dyn()
                    .into();
                T::from_dyn_arr(arr)?
                    .into_dimensionality::<D>()
                    .map_err(Into::into)
            }
        }
    }

    fn write_array_slice<S, T, D>(&self, data: CowArray<'_, T, D>, selection: &[S]) -> Result<()>
    where
        T: BackendData,
        S: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        fn write_array_impl<T, S>(
            container: &H5Dataset,
            arr: CowArray<'_, T, IxDyn>,
            selection: &[S],
        ) -> Result<()>
        where
            T: H5Type + Clone,
            S: AsRef<SelectInfoElem>,
        {
            let (select, _) = into_selection(selection, container.shape());
            container
                .deref()
                .write_slice(&arr.as_standard_layout(), select)?;
            Ok(())
        }

        match BackendData::into_dyn_arr(data.into_dyn()) {
            DynCowArray::U8(x) => write_array_impl(self, x, selection),
            DynCowArray::U16(x) => write_array_impl(self, x, selection),
            DynCowArray::U32(x) => write_array_impl(self, x, selection),
            DynCowArray::U64(x) => write_array_impl(self, x, selection),
            DynCowArray::I8(x) => write_array_impl(self, x, selection),
            DynCowArray::I16(x) => write_array_impl(self, x, selection),
            DynCowArray::I32(x) => write_array_impl(self, x, selection),
            DynCowArray::I64(x) => write_array_impl(self, x, selection),
            DynCowArray::F32(x) => write_array_impl(self, x, selection),
            DynCowArray::F64(x) => write_array_impl(self, x, selection),
            DynCowArray::Bool(x) => write_array_impl(self, x, selection),
            DynCowArray::String(x) => {
                let data: Array<VarLenUnicode, _> = x.map(|x| x.parse().unwrap());
                write_array_impl(self, data.into(), selection)
            }
        }
    }
}

// Generic `LocationOp` functions

fn file(loc: &Location) -> Result<H5File> {
    Ok(H5File(hdf5::Location::file(loc)?))
}

fn path(loc: &Location) -> PathBuf {
    hdf5::Location::name(loc).into()
}

////////////////////////////////////////////////////////////////////////////////
// Derived implementations
////////////////////////////////////////////////////////////////////////////////

impl GroupOp<H5> for H5File {
    fn list(&self) -> Result<Vec<String>> {
        list(self)
    }

    fn new_group(&self, name: &str) -> Result<<H5 as Backend>::Group> {
        create_group(self, name)
    }

    fn open_group(&self, name: &str) -> Result<<H5 as Backend>::Group> {
        open_group(self, name)
    }

    fn new_empty_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<H5 as Backend>::Dataset> {
        new_dataset::<T>(self, name, shape, config)
    }

    fn open_dataset(&self, name: &str) -> Result<<H5 as Backend>::Dataset> {
        open_dataset(self, name)
    }

    fn delete(&self, name: &str) -> Result<()> {
        delete(self, name)
    }

    fn exists(&self, name: &str) -> Result<bool> {
        exists(self, name)
    }

    fn new_scalar_dataset<D: BackendData>(
        &self,
        name: &str,
        data: &D,
    ) -> Result<<H5 as Backend>::Dataset> {
        create_scalar_data(self, name, data)
    }
}

impl GroupOp<H5> for H5Group {
    fn list(&self) -> Result<Vec<String>> {
        list(self)
    }

    fn new_group(&self, name: &str) -> Result<<H5 as Backend>::Group> {
        create_group(self, name)
    }

    fn open_group(&self, name: &str) -> Result<<H5 as Backend>::Group> {
        open_group(self, name)
    }

    fn new_empty_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<H5 as Backend>::Dataset> {
        new_dataset::<T>(self, name, shape, config)
    }

    fn open_dataset(&self, name: &str) -> Result<<H5 as Backend>::Dataset> {
        open_dataset(self, name)
    }

    fn delete(&self, name: &str) -> Result<()> {
        delete(self, name)
    }

    fn exists(&self, name: &str) -> Result<bool> {
        exists(self, name)
    }

    fn new_scalar_dataset<D: BackendData>(
        &self,
        name: &str,
        data: &D,
    ) -> Result<<H5 as Backend>::Dataset> {
        create_scalar_data(self, name, data)
    }
}

impl AttributeOp<H5> for H5Group {
    fn store(&self) -> Result<<H5 as Backend>::Store> {
        file(self)
    }

    fn path(&self) -> PathBuf {
        path(self)
    }

    fn new_json_attr(&mut self, name: &str, value: &Value) -> Result<()> {
        match value {
            Value::Null => Ok(()),
            Value::Bool(b) => write_scalar_attr(self, name, *b),
            Value::Number(n) => n
                .as_u64()
                .map(|i| write_scalar_attr(self, name, i))
                .or_else(|| n.as_i64().map(|i| write_scalar_attr(self, name, i)))
                .or_else(|| n.as_f64().map(|i| write_scalar_attr(self, name, i)))
                .expect("number cannot be converted to u64, i64 or f64"),
            Value::String(s) => write_scalar_attr(self, name, s.clone()),
            Value::Array(_) => json_to_ndarray(value, |x| x.as_i64())?
                .map(|x| write_array_attr(self, name, &x))
                .or_else(|| {
                    json_to_ndarray(value, |x| x.as_f64())
                        .unwrap()
                        .map(|x| write_array_attr(self, name, &x))
                })
                .or_else(|| {
                    json_to_ndarray(value, |x| x.as_str().map(|s| s.to_string()))
                        .unwrap()
                        .map(|x| write_array_attr(self, name, &x))
                })
                .expect("array cannot be converted to i64, f64 or string"),
            Value::Object(_) => bail!("attributes of object type are not supported"),
        }
    }

    fn get_json_attr(&self, name: &str) -> Result<Value> {
        if self.attr(name)?.is_scalar() {
            read_scalar_attr(self, name)
        } else {
            read_array_attr(self, name)
        }
    }
}

impl AttributeOp<H5> for H5Dataset {
    fn store(&self) -> Result<<H5 as Backend>::Store> {
        file(self)
    }

    fn path(&self) -> PathBuf {
        path(self)
    }

    fn new_json_attr(&mut self, name: &str, value: &Value) -> Result<()> {
        match value {
            Value::Null => Ok(()),
            Value::Bool(b) => write_scalar_attr(self, name, *b),
            Value::Number(n) => n
                .as_u64()
                .map(|i| write_scalar_attr(self, name, i))
                .or_else(|| n.as_i64().map(|i| write_scalar_attr(self, name, i)))
                .or_else(|| n.as_f64().map(|i| write_scalar_attr(self, name, i)))
                .expect("number cannot be converted to u64, i64 or f64"),
            Value::String(s) => write_scalar_attr(self, name, s.clone()),
            Value::Array(_) => json_to_ndarray(value, |x| x.as_i64())?
                .map(|x| write_array_attr(self, name, &x))
                .or_else(|| {
                    json_to_ndarray(value, |x| x.as_f64())
                        .unwrap()
                        .map(|x| write_array_attr(self, name, &x))
                })
                .or_else(|| {
                    json_to_ndarray(value, |x| x.as_str().map(|s| s.to_string()))
                        .unwrap()
                        .map(|x| write_array_attr(self, name, &x))
                })
                .expect("array cannot be converted to i64, f64 or string"),
            Value::Object(_) => bail!("attributes of object type are not supported"),
        }
    }

    fn get_json_attr(&self, name: &str) -> Result<Value> {
        if self.attr(name)?.is_scalar() {
            read_scalar_attr(self, name)
        } else {
            read_array_attr(self, name)
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// Auxiliary functions
///////////////////////////////////////////////////////////////////////////////
fn read_scalar_attr(loc: &Location, name: &str) -> Result<Value> {
    let attr = loc.attr(name)?;
    let result = match attr.dtype()?.to_descriptor()? {
        TypeDescriptor::VarLenUnicode => attr.read_scalar::<VarLenUnicode>()?.to_string().into(),
        TypeDescriptor::VarLenAscii => attr.read_scalar::<VarLenUnicode>()?.to_string().into(),
        TypeDescriptor::Boolean => attr.read_scalar::<bool>()?.into(),
        TypeDescriptor::Unsigned(_) => attr.read_scalar::<u64>()?.into(),
        TypeDescriptor::Integer(_) => attr.read_scalar::<i64>()?.into(),
        TypeDescriptor::Float(_) => attr.read_scalar::<f64>()?.into(),
        v => bail!("Unsupported type {v}"),
    };
    Ok(result)
}

fn read_array_attr(loc: &Location, name: &str) -> Result<Value> {
    let attr = loc.attr(name)?;
    let result = match attr.dtype()?.to_descriptor()? {
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
            let arr = attr.read::<VarLenUnicode, IxDyn>()?;
            let shape = arr.raw_dim();
            let vec = arr.into_raw_vec_and_offset().0;
            let strings: Vec<String> = vec.into_iter().map(|s| s.to_string()).collect();
            let arr_string = Array::from_shape_vec(shape, strings).unwrap();
            ndarray_to_json(&arr_string)
        }
        TypeDescriptor::Boolean => ndarray_to_json(&attr.read::<bool, IxDyn>()?),
        TypeDescriptor::Unsigned(_) => ndarray_to_json(&attr.read::<u64, IxDyn>()?),
        TypeDescriptor::Integer(_) => ndarray_to_json(&attr.read::<i64, IxDyn>()?),
        TypeDescriptor::Float(_) => ndarray_to_json(&attr.read::<f64, IxDyn>()?),
        v => bail!("Unsupported type {v}"),
    };
    Ok(result)
}

fn write_array_attr<'a, A, D, Dim>(loc: &Location, name: &str, value: A) -> Result<()>
where
    A: Into<ArrayView<'a, D, Dim>>,
    D: BackendData,
    Dim: Dimension,
{
    del_attr(loc, name);
    let value = value.into().into_dyn().into();
    match BackendData::into_dyn_arr(value) {
        DynCowArray::U8(x) => loc.new_attr_builder().with_data(x.view()).create(name)?,
        DynCowArray::U16(x) => loc.new_attr_builder().with_data(x.view()).create(name)?,
        DynCowArray::U32(x) => loc.new_attr_builder().with_data(x.view()).create(name)?,
        DynCowArray::U64(x) => loc.new_attr_builder().with_data(x.view()).create(name)?,
        DynCowArray::I8(x) => loc.new_attr_builder().with_data(x.view()).create(name)?,
        DynCowArray::I16(x) => loc.new_attr_builder().with_data(x.view()).create(name)?,
        DynCowArray::I32(x) => loc.new_attr_builder().with_data(x.view()).create(name)?,
        DynCowArray::I64(x) => loc.new_attr_builder().with_data(x.view()).create(name)?,
        DynCowArray::F32(x) => loc.new_attr_builder().with_data(x.view()).create(name)?,
        DynCowArray::F64(x) => loc.new_attr_builder().with_data(x.view()).create(name)?,
        DynCowArray::Bool(x) => loc.new_attr_builder().with_data(x.view()).create(name)?,
        DynCowArray::String(x) => {
            let data: Array<VarLenUnicode, Dim> =
                x.map(|x| x.parse().unwrap()).into_dimensionality()?;
            loc.new_attr_builder().with_data(data.view()).create(name)?
        }
    };
    Ok(())
}

fn write_scalar_attr<D: BackendData>(loc: &Location, name: &str, value: D) -> Result<()> {
    del_attr(loc, name);
    match value.as_dyn() {
        DynScalar::U8(x) => loc.new_attr::<u8>().create(name)?.write_scalar(&x)?,
        DynScalar::U16(x) => loc.new_attr::<u16>().create(name)?.write_scalar(&x)?,
        DynScalar::U32(x) => loc.new_attr::<u32>().create(name)?.write_scalar(&x)?,
        DynScalar::U64(x) => loc.new_attr::<u64>().create(name)?.write_scalar(&x)?,
        DynScalar::I8(x) => loc.new_attr::<i8>().create(name)?.write_scalar(&x)?,
        DynScalar::I16(x) => loc.new_attr::<i16>().create(name)?.write_scalar(&x)?,
        DynScalar::I32(x) => loc.new_attr::<i32>().create(name)?.write_scalar(&x)?,
        DynScalar::I64(x) => loc.new_attr::<i64>().create(name)?.write_scalar(&x)?,
        DynScalar::F32(x) => loc.new_attr::<f32>().create(name)?.write_scalar(&x)?,
        DynScalar::F64(x) => loc.new_attr::<f64>().create(name)?.write_scalar(&x)?,
        DynScalar::Bool(x) => loc.new_attr::<bool>().create(name)?.write_scalar(&x)?,
        DynScalar::String(x) => {
            let value_: VarLenUnicode = x.parse().unwrap();
            loc.new_attr::<VarLenUnicode>()
                .create(name)?
                .write_scalar(&value_)?
        }
    };
    Ok(())
}

fn into_selection<S, E>(selection: S, shape: Shape) -> (Selection, Shape)
where
    S: AsRef<[E]>,
    E: AsRef<SelectInfoElem>,
{
    if selection.as_ref().iter().all(|x| x.as_ref().is_full()) {
        (Selection::All, shape)
    } else {
        let bounded_selection = SelectInfoBounds::new(&selection, &shape);
        let out_shape = bounded_selection.out_shape();
        if let Some(idx) = bounded_selection.try_into_indices() {
            (Selection::from(idx), out_shape)
        } else {
            let slice: SliceInfo<_, _, _> = bounded_selection.try_into().unwrap();
            (Selection::try_from(slice).unwrap(), out_shape)
        }
    }
}

fn del_attr(loc: &Location, name: &str) {
    unsafe {
        let c_name = std::ffi::CString::new(name).unwrap().into_raw();
        if hdf5_sys::h5a::H5Aexists(loc.id(), c_name) != 0 {
            hdf5_sys::h5a::H5Adelete(loc.id(), c_name);
        }
    }
}

fn json_to_ndarray<F, T>(json: &Value, f: F) -> Result<Option<ArrayD<T>>>
where
    F: Fn(&Value) -> Option<T>,
{
    // Recursively determine the shape of the JSON array
    fn get_shape(value: &Value) -> Vec<usize> {
        let mut shape = Vec::new();
        let mut current = value;
        while let Value::Array(arr) = current {
            shape.push(arr.len());
            if arr.is_empty() {
                break;
            }
            current = &arr[0];
        }
        shape
    }

    // Recursively flatten the JSON array to a Vec<i32>
    fn flatten_array<F, T>(value: &Value, f: &F) -> Option<Vec<T>>
    where
        F: Fn(&Value) -> Option<T>,
    {
        match value {
            Value::Array(arr) => {
                let mut flattened = Vec::new();
                for item in arr {
                    flattened.extend(flatten_array(item, f)?);
                }
                Some(flattened)
            }
            v => Some(vec![f(v)?]),
        }
    }

    // Get the shape and flatten the JSON
    let shape = get_shape(json);
    if let Some(flattened_data) = flatten_array(json, &f) {
        Ok(Some(ArrayD::from_shape_vec(IxDyn(&shape), flattened_data)?))
    } else {
        Ok(None)
    }
}

fn ndarray_to_json<T: Into<Value> + Clone>(array: &ArrayD<T>) -> Value {
    // Helper function to recursively convert ndarray to nested Vecs
    fn recursive_convert<T: Into<Value> + Clone>(array: &ArrayD<T>) -> Value {
        if array.ndim() == 1 {
            // Base case: 1D array, convert to Vec and serialize
            let vec = array.iter().cloned().collect::<Vec<T>>();
            vec.into()
        } else {
            // Recursive case: split along the first axis and apply recursively
            let nested_vec = array
                .outer_iter()
                .map(|sub_array| recursive_convert(&sub_array.to_owned().into_dyn()))
                .collect::<Vec<Value>>();
            Value::Array(nested_vec)
        }
    }

    recursive_convert(array)
}

/// test module
#[cfg(test)]
mod tests {
    use super::*;
    use anndata::s;
    use ndarray::{Array1, Axis, Ix1, concatenate};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use std::path::PathBuf;
    use tempfile::tempdir;

    pub fn with_tmp_dir<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();
        func(path)
    }

    fn with_tmp_path<T, F: Fn(PathBuf) -> T>(func: F) -> T {
        with_tmp_dir(|dir| func(dir.join("temp.h5")))
    }

    #[test]
    fn test_basic() -> Result<()> {
        with_tmp_path(|path| {
            let file = H5::new(path.clone())?;
            let group = file.new_group("group")?;
            let subgroup = group.new_group("subgroup")?;

            assert_eq!(subgroup.path(), PathBuf::from("/group/subgroup"));
            Ok(())
        })
    }

    #[test]
    fn test_write_empty() -> Result<()> {
        with_tmp_path(|path| {
            let file = H5::new(&path)?;
            let group = file.new_group("group")?;
            let config = WriteConfig {
                ..Default::default()
            };

            let empty = Array1::<u8>::from_vec(Vec::new());
            let dataset = group.new_array_dataset("test", empty.view().into(), config)?;
            assert_eq!(empty, dataset.read_array::<u8, Ix1>()?);
            Ok(())
        })
    }

    #[test]
    fn test_write_slice() -> Result<()> {
        with_tmp_path(|path| -> Result<()> {
            let file = H5::new(&path)?;
            let config = WriteConfig {
                ..Default::default()
            };

            let mut dataset =
                file.new_empty_dataset::<i32>("test", &[20, 50].as_slice().into(), config)?;
            let arr = Array::random((20, 50), Uniform::new(0, 100).unwrap());

            // Repeatitive writes
            dataset.write_array_slice(arr.view().into(), s![.., ..].as_ref())?;
            dataset.write_array_slice(arr.view().into(), s![.., ..].as_ref())?;

            // Out-of-bounds writes should fail
            assert!(
                dataset
                    .write_array_slice(arr.view().into(), s![20..40, ..].as_ref())
                    .is_err()
            );

            // Reshape and write
            dataset.reshape(&[40, 50].as_slice().into())?;
            dataset.write_array_slice(arr.view().into(), s![20..40, ..].as_ref())?;

            // Read back is OK
            let merged = concatenate(Axis(0), &[arr.view(), arr.view()])?;
            assert_eq!(merged, dataset.read_array::<i32, _>()?);

            // Shrinking is OK
            dataset.reshape(&[20, 50].as_slice().into())?;
            assert_eq!(arr, dataset.read_array::<i32, _>()?);

            Ok(())
        })
    }
}
