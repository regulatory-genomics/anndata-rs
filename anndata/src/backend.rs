mod datatype;
use crate::data::{ArrayConvert, DynArray, SelectInfo, SelectInfoElem, Shape};
pub use datatype::{BackendData, DataType, ScalarType};

use anyhow::{Result, bail};
use core::fmt::{Debug, Formatter};
use ndarray::{Array, CowArray, Dimension, Ix0, IxDyn, arr0};
use serde::Deserialize;
pub use serde_json::Value;
use std::cell::RefCell;
use std::path::{Path, PathBuf};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Compression {
    Gzip(u8),
    Zst(u8),
}

#[derive(Debug, Clone)]
pub struct WriteConfig {
    pub compression: Option<Compression>,
    pub block_size: Option<Shape>,
}

impl Default for WriteConfig {
    fn default() -> Self {
        Self {
            compression: Some(Compression::Zst(5)),
            block_size: None,
        }
    }
}

thread_local! {
    static DEFAULT_WRITE_CONFIG: RefCell<WriteConfig> = RefCell::new(WriteConfig::default());
}

/// Set the default write configuration for all subsequent write operations.
pub fn set_default_write_config(config: WriteConfig) {
    DEFAULT_WRITE_CONFIG.with(|c| *c.borrow_mut() = config);
}

/// Get the current default write configuration.
pub fn get_default_write_config() -> WriteConfig {
    DEFAULT_WRITE_CONFIG.with(|c| c.borrow().clone())
}

pub trait Backend: 'static {
    /// The name of the backend.
    const NAME: &'static str;

    /// Data store
    type Store: StoreOp<Self> + GroupOp<Self> + Send + Sync;

    /// Groups work like directories and can contain groups or datasets.
    type Group: GroupOp<Self> + AttributeOp<Self> + Send + Sync;

    /// Datasets store multi-dimensional arrays.
    type Dataset: DatasetOp<Self> + AttributeOp<Self> + Send + Sync;

    /// Create a new file at the given path.
    fn new<P: AsRef<Path>>(path: P) -> Result<Self::Store>;

    /// Opens a file as read-only, file must exist.
    fn open<P: AsRef<Path>>(path: P) -> Result<Self::Store>;

    /// Opens a file as read/write, file must exist.
    fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self::Store>;
}

pub trait StoreOp<B: Backend + ?Sized> {
    /// Returns the file path.
    fn filename(&self) -> PathBuf;

    /// Close the file.
    fn close(self) -> Result<()>;
}

pub trait GroupOp<B: Backend + ?Sized> {
    /// List all groups and datasets in this group.
    fn list(&self) -> Result<Vec<String>>;

    /// Create a new group.
    fn new_group(&self, name: &str) -> Result<B::Group>;

    /// Open an existing group.
    fn open_group(&self, name: &str) -> Result<B::Group>;

    /// Create an empty dataset holding an array value.
    fn new_empty_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<B::Dataset>;
    fn open_dataset(&self, name: &str) -> Result<B::Dataset>;

    /// Delete a group or dataset.
    fn delete(&self, name: &str) -> Result<()>;

    /// Check if a group or dataset exists.
    fn exists(&self, name: &str) -> Result<bool>;

    fn new_array_dataset<'a, D, Dim>(
        &self,
        name: &str,
        arr: CowArray<'a, D, Dim>,
        config: WriteConfig,
    ) -> Result<B::Dataset>
    where
        D: BackendData,
        Dim: Dimension,
    {
        let shape = arr.shape();
        let block_size = config.block_size.unwrap_or_else(|| {
            if shape.len() == 1 {
                shape[0].min(16384).into()
            } else {
                shape.iter().map(|&x| x.min(128)).collect()
            }
        });
        let compression = if arr.len() > 128 {
            config.compression
        } else {
            None
        };
        let new_config = WriteConfig {
            compression,
            block_size: Some(block_size),
        };
        let dataset = self.new_empty_dataset::<D>(name, &shape.into(), new_config)?;
        dataset.write_array(arr)?;
        Ok(dataset)
    }

    fn new_scalar_dataset<D: BackendData>(&self, name: &str, data: &D) -> Result<B::Dataset> {
        self.new_array_dataset(name, arr0(data.clone()).into(), get_default_write_config())
    }
}

pub trait AttributeOp<B: Backend + ?Sized> {
    /// Returns the Root.
    fn store(&self) -> Result<B::Store>;

    /// Returns the path of the location relative to the file root.
    fn path(&self) -> PathBuf;

    /// Write a attribute at a given location.
    fn new_json_attr(&mut self, name: &str, value: &Value) -> Result<()>;

    fn get_json_attr(&self, name: &str) -> Result<Value>;

    fn get_attr<'de, T>(&self, name: &str) -> Result<T>
    where
        T: Deserialize<'de>,
    {
        let value = self.get_json_attr(name)?;
        T::deserialize(value).map_err(|e| e.into())
    }

    fn new_attr<T: Into<Value>>(&mut self, name: &str, value: T) -> Result<()> {
        self.new_json_attr(name, &value.into())
    }
}

pub trait DatasetOp<B: Backend + ?Sized> {
    /// Required methods
    fn dtype(&self) -> Result<ScalarType>;
    fn shape(&self) -> Shape;
    fn reshape(&mut self, shape: &Shape) -> Result<()>;

    fn write_array_slice<S, T, D>(&self, arr: CowArray<'_, T, D>, selection: &[S]) -> Result<()>
    where
        T: BackendData,
        S: AsRef<SelectInfoElem>,
        D: Dimension;

    fn read_array_slice<T: BackendData, S, D>(&self, selection: &[S]) -> Result<Array<T, D>>
    where
        S: AsRef<SelectInfoElem>,
        D: Dimension;

    /// Optional methods
    fn read_dyn_array_slice<S>(&self, selection: &[S]) -> Result<DynArray>
    where
        S: AsRef<SelectInfoElem>,
    {
        let arr = match self.dtype()? {
            ScalarType::I8 => self.read_array_slice::<i8, _, IxDyn>(selection)?.into(),
            ScalarType::I16 => self.read_array_slice::<i16, _, IxDyn>(selection)?.into(),
            ScalarType::I32 => self.read_array_slice::<i32, _, IxDyn>(selection)?.into(),
            ScalarType::I64 => self.read_array_slice::<i64, _, IxDyn>(selection)?.into(),
            ScalarType::U8 => self.read_array_slice::<u8, _, IxDyn>(selection)?.into(),
            ScalarType::U16 => self.read_array_slice::<u16, _, IxDyn>(selection)?.into(),
            ScalarType::U32 => self.read_array_slice::<u32, _, IxDyn>(selection)?.into(),
            ScalarType::U64 => self.read_array_slice::<u64, _, IxDyn>(selection)?.into(),
            ScalarType::F32 => self.read_array_slice::<f32, _, IxDyn>(selection)?.into(),
            ScalarType::F64 => self.read_array_slice::<f64, _, IxDyn>(selection)?.into(),
            ScalarType::Bool => self.read_array_slice::<bool, _, IxDyn>(selection)?.into(),
            ScalarType::String => self.read_array_slice::<String, _, IxDyn>(selection)?.into(),
        };
        Ok(arr)
    }

    fn read_array_slice_cast<T, D, S>(&self, selection: &[S]) -> Result<Array<T, D>>
    where
        DynArray: ArrayConvert<Array<T, D>>,
        D: Dimension,
        S: AsRef<SelectInfoElem>,
    {
        self.read_dyn_array_slice(selection)?.try_convert()
    }

    fn read_array<T: BackendData, D>(&self) -> Result<Array<T, D>>
    where
        D: Dimension,
    {
        self.read_array_slice(SelectInfo::full_slice(self.shape().ndim()).as_ref())
    }

    fn read_dyn_array(&self) -> Result<DynArray> {
        self.read_dyn_array_slice(SelectInfo::full_slice(self.shape().ndim()).as_ref())
    }

    fn read_array_cast<T, D>(&self) -> Result<Array<T, D>>
    where
        DynArray: ArrayConvert<Array<T, D>>,
        D: Dimension,
    {
        self.read_dyn_array()?.try_convert()
    }

    fn read_scalar<T: BackendData>(&self) -> Result<T> {
        self.read_array::<T, Ix0>().map(|x| x.into_scalar())
    }

    fn write_array<D, Dim>(&self, arr: CowArray<'_, D, Dim>) -> Result<()>
    where
        D: BackendData,
        Dim: Dimension,
    {
        let ndim = arr.ndim();
        self.write_array_slice(arr, SelectInfo::full_slice(ndim).as_ref())
    }
}

#[derive(Default)]
pub enum DataContainer<B: Backend> {
    Group(B::Group),
    Dataset(B::Dataset),
    #[default]
    Null,
}

impl<B: Backend> Debug for DataContainer<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            DataContainer::Group(g) => write!(f, "Group({:?})", g.path()),
            DataContainer::Dataset(d) => write!(f, "Dataset({:?})", d.path()),
            DataContainer::Null => write!(f, "Null"),
        }
    }
}

impl<B: Backend> AttributeOp<B> for DataContainer<B> {
    fn store(&self) -> Result<B::Store> {
        match self {
            DataContainer::Group(g) => g.store(),
            DataContainer::Dataset(d) => d.store(),
            DataContainer::Null => bail!("Null container"),
        }
    }
    fn path(&self) -> PathBuf {
        match self {
            DataContainer::Group(g) => g.path(),
            DataContainer::Dataset(d) => d.path(),
            DataContainer::Null => PathBuf::new(),
        }
    }

    fn new_json_attr(&mut self, name: &str, value: &Value) -> Result<()> {
        match self {
            DataContainer::Group(g) => g.new_json_attr(name, value),
            DataContainer::Dataset(d) => d.new_json_attr(name, value),
            DataContainer::Null => bail!("Null container"),
        }
    }
    fn get_json_attr(&self, name: &str) -> Result<Value> {
        match self {
            DataContainer::Group(g) => g.get_json_attr(name),
            DataContainer::Dataset(d) => d.get_json_attr(name),
            DataContainer::Null => bail!("Null container"),
        }
    }
}

impl<B: Backend> DataContainer<B> {
    pub fn open<G: GroupOp<B>>(group: &G, name: &str) -> Result<Self> {
        if group.exists(name)? {
            match group.open_dataset(name) {
                Ok(gr) => Ok(DataContainer::Dataset(gr)),
                Err(e1) => group
                    .open_group(name)
                    .map(DataContainer::Group)
                    .map_err(|e2| {
                        e2.context(e1).context(format!(
                            "Error opening group or dataset named '{name}' in group"
                        ))
                    }),
            }
        } else {
            bail!("No group or dataset named '{name}' in group");
        }
    }

    pub fn delete(container: DataContainer<B>) -> Result<()> {
        container
            .store()?
            .delete(&container.path().to_string_lossy())
    }

    pub fn encoding_type(&self) -> Result<DataType> {
        let enc = match self {
            DataContainer::Group(group) => group
                .get_attr("encoding-type")
                .unwrap_or("dict".to_string()),
            DataContainer::Dataset(dataset) => dataset
                .get_attr("encoding-type")
                .unwrap_or("numeric-scalar".to_string()),
            DataContainer::Null => bail!("Null container"),
        };
        let ty = match enc.as_str() {
            "string" => DataType::Scalar(ScalarType::String),
            "numeric-scalar" => DataType::Scalar(self.as_dataset()?.dtype()?),
            "categorical" => DataType::Categorical,
            "string-array" => DataType::Array(ScalarType::String),
            "array" => DataType::Array(self.as_dataset()?.dtype()?),
            "csc_matrix" => {
                let ty = self.as_group()?.open_dataset("data")?.dtype()?;
                let tp = self.as_group()?.open_dataset("indices")?.dtype()?;
                DataType::CscMatrix(ty, tp)
            }
            "csr_matrix" => {
                let ty = self.as_group()?.open_dataset("data")?.dtype()?;
                let tp = self.as_group()?.open_dataset("indices")?.dtype()?;
                DataType::CsrMatrix(ty, tp)
            }
            "dataframe" => DataType::DataFrame,
            "mapping" | "dict" => DataType::Mapping,
            "nullable-integer" | "nullable-boolean" => DataType::NullableArray,
            ty => bail!("the anndata file contains an unsupported encoding type: '{ty}'"),
        };
        Ok(ty)
    }

    pub fn as_group(&self) -> Result<&B::Group> {
        match self {
            Self::Group(x) => Ok(x),
            _ => bail!("Expecting Group"),
        }
    }

    pub fn as_dataset(&self) -> Result<&B::Dataset> {
        match self {
            Self::Dataset(x) => Ok(x),
            _ => bail!("Expecting Dataset"),
        }
    }
}
