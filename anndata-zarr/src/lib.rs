use anndata::{
    backend::*,
    data::{DynArray, DynCowArray, SelectInfoBounds, SelectInfoElem, SelectInfoElemBounds, Shape},
};

use anyhow::{Result, bail};
use ndarray::{Array, ArrayD, ArrayView, CowArray, Dimension, IxDyn, SliceInfoElem};
use object_store::ObjectStore;
use once_cell::sync::Lazy;
use std::num::NonZeroU64;
use std::{
    borrow::Cow,
    ops::{Deref, Index},
    path::{Path, PathBuf},
};
use std::{sync::Arc, vec};
use url::Url;
use zarrs::array::{
    ArrayShardedReadableExt, ArraySubset, ChunkShape, CodecOptions, Element, ElementOwned,
    FillValue, data_type,
};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::storage::{ReadableWritableListableStorageTraits, StorePrefix};
use zarrs_object_store::AsyncObjectStore;
use zarrs_storage::storage_adapter::async_to_sync::{
    AsyncToSyncBlockOn, AsyncToSyncStorageAdapter,
};

static RUNTIME: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
});

struct TokioBlockOn;
impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        RUNTIME.block_on(future)
    }
}

/// The Zarr backend.
pub struct Zarr;

#[derive(Clone)]
pub struct ZarrStore {
    inner: Arc<dyn ReadableWritableListableStorageTraits>,
    path: PathBuf,
}

impl Deref for ZarrStore {
    type Target = Arc<dyn ReadableWritableListableStorageTraits>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct ZarrGroup {
    group: Group<dyn ReadableWritableListableStorageTraits>,
    store: ZarrStore,
}

pub struct ZarrDataset {
    dataset: zarrs::array::Array<dyn ReadableWritableListableStorageTraits>,
    cache: zarrs::array::ArrayShardedReadableExtCache,
    store: ZarrStore,
}

fn get_storage<P: AsRef<Path>>(
    path: P,
) -> Result<(Arc<dyn ReadableWritableListableStorageTraits>, PathBuf)> {
    let path_str = path
        .as_ref()
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
    if path_str.starts_with("s3://")
        || path_str.starts_with("http://")
        || path_str.starts_with("https://")
    {
        let url = Url::parse(path_str)?;
        let (store, prefix) = object_store::parse_url(&url)?;
        let rooted_store: Arc<dyn ObjectStore> =
            Arc::new(object_store::prefix::PrefixStore::new(store, prefix));
        let async_zarr_store = Arc::new(AsyncObjectStore::new(rooted_store));
        let sync_zarr_store = Arc::new(AsyncToSyncStorageAdapter::new(
            async_zarr_store,
            TokioBlockOn,
        ));
        Ok((sync_zarr_store, path.as_ref().to_path_buf()))
    } else {
        let inner = Arc::new(FilesystemStore::new(path.as_ref())?);
        Ok((inner, path.as_ref().to_path_buf()))
    }
}

impl Backend for Zarr {
    const NAME: &'static str = "zarr";

    type Store = ZarrStore;

    type Group = ZarrGroup;

    /// datasets contain arrays.
    type Dataset = ZarrDataset;

    fn new<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
        if !path_str.starts_with("s3://")
            && !path_str.starts_with("http://")
            && !path_str.starts_with("https://")
            && path.as_ref().try_exists()?
        {
            let metadata = std::fs::metadata(&path)?;
            if metadata.is_file() {
                std::fs::remove_file(&path)?;
            } else {
                std::fs::remove_dir_all(&path)?;
            }
        }

        let (inner, path) = get_storage(path)?;
        zarrs::group::GroupBuilder::new()
            .build(inner.clone(), "/")?
            .store_metadata()?;
        Ok(ZarrStore { path, inner })
    }

    /// Opens a file as read-only, file must exist.
    fn open<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        let (inner, path) = get_storage(path)?;
        Ok(ZarrStore { path, inner })
    }

    /// Opens a file as read/write, file must exist.
    fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        let (inner, path) = get_storage(path)?;
        Ok(ZarrStore { path, inner })
    }
}

impl StoreOp<Zarr> for ZarrStore {
    /// Returns the file path.
    fn filename(&self) -> PathBuf {
        self.path.clone()
    }

    /// Close the file.
    fn close(self) -> Result<()> {
        drop(self);
        Ok(())
    }
}

impl GroupOp<Zarr> for ZarrStore {
    /// List all groups and datasets in this group.
    fn list(&self) -> Result<Vec<String>> {
        let result = self.list_dir(&StorePrefix::root())?;
        Ok(result
            .prefixes()
            .iter()
            .map(|x| {
                x.as_str()
                    .trim_start_matches("/")
                    .trim_end_matches("/")
                    .to_string()
            })
            .collect())
    }

    /// Create a new group.
    fn new_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = canoincalize_path(name);
        let group = zarrs::group::GroupBuilder::new().build(self.inner.clone(), &path)?;
        group.store_metadata()?;
        Ok(ZarrGroup {
            group,
            store: self.clone(),
        })
    }

    /// Open an existing group.
    fn open_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let group = zarrs::group::Group::open(self.inner.clone(), &canoincalize_path(name))?;
        Ok(ZarrGroup {
            group,
            store: self.clone(),
        })
    }

    /// Create an empty dataset holding an array value.
    fn new_empty_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Zarr as Backend>::Dataset> {
        let path = canoincalize_path(name);
        let array = new_empty_dataset_helper::<T, _>(self.inner.clone(), &path, shape, config)?;
        array.store_metadata()?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.clone(),
        })
    }

    fn open_dataset(&self, name: &str) -> Result<<Zarr as Backend>::Dataset> {
        let array = zarrs::array::Array::open(self.inner.clone(), &canoincalize_path(name))?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.clone(),
        })
    }

    /// Delete a group or dataset.
    fn delete(&self, name: &str) -> Result<()> {
        self.inner.erase_prefix(&str_to_prefix(name))?;
        Ok(())
    }

    /// Check if a group or dataset exists.
    fn exists(&self, name: &str) -> Result<bool> {
        let path = format!("/{name}");
        Ok(zarrs::node::node_exists(
            &self.inner,
            &path.as_str().try_into()?,
        )?)
    }
}

impl GroupOp<Zarr> for ZarrGroup {
    fn list(&self) -> Result<Vec<String>> {
        let current_path = str_to_prefix(self.group.path().as_str());
        let result = self
            .store
            .list_dir(&current_path.as_str().try_into()?)?
            .prefixes()
            .iter()
            .map(|x| {
                x.as_str()
                    .strip_prefix(current_path.as_str())
                    .unwrap()
                    .strip_suffix("/")
                    .unwrap()
                    .to_owned()
            })
            .collect();
        Ok(result)
    }

    /// Create a new group.
    fn new_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = self.group.path().as_path().join(name);
        let group = zarrs::group::GroupBuilder::new()
            .build(self.store.inner.clone(), path.to_str().unwrap())?;
        group.store_metadata()?;
        Ok(ZarrGroup {
            group,
            store: self.store.clone(),
        })
    }

    /// Open an existing group.
    fn open_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = self.group.path().as_path().join(name);
        let group = zarrs::group::Group::open(self.store.inner.clone(), path.to_str().unwrap())?;
        Ok(ZarrGroup {
            group,
            store: self.store.clone(),
        })
    }

    /// Create an empty dataset holding an array value.
    fn new_empty_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Zarr as Backend>::Dataset> {
        let path = self.group.path().as_path().join(name);
        let array = new_empty_dataset_helper::<T, _>(
            self.store.inner.clone(),
            path.to_str().unwrap(),
            shape,
            config,
        )?;
        array.store_metadata()?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.store.clone(),
        })
    }

    fn open_dataset(&self, name: &str) -> Result<<Zarr as Backend>::Dataset> {
        let path = self.group.path().as_path().join(name);
        let array = zarrs::array::Array::open(self.store.inner.clone(), path.to_str().unwrap())?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.store.clone(),
        })
    }

    /// Delete a group or dataset.
    fn delete(&self, name: &str) -> Result<()> {
        let path = format!("{}/{}", self.group.path().as_str(), name);
        self.store.erase_prefix(&str_to_prefix(&path))?;
        Ok(())
    }

    /// Check if a group or dataset exists.
    fn exists(&self, name: &str) -> Result<bool> {
        let path = self
            .group
            .path()
            .as_path()
            .join(name)
            .as_os_str()
            .to_str()
            .unwrap()
            .try_into()?;
        Ok(zarrs::node::node_exists(&self.store.inner, &path)?)
    }
}

impl AttributeOp<Zarr> for ZarrGroup {
    /// Returns the Root.
    fn store(&self) -> Result<<Zarr as Backend>::Store> {
        Ok(self.store.clone())
    }

    /// Returns the path of the location relative to the file root.
    fn path(&self) -> PathBuf {
        self.group.path().as_path().to_path_buf()
    }

    /// Write an attribute at a given location.
    fn new_json_attr(&mut self, name: &str, value: &Value) -> Result<()> {
        self.group
            .attributes_mut()
            .insert(name.to_string(), value.clone());
        self.group.store_metadata()?;
        Ok(())
    }

    /// Get an attribute at a given location.
    fn get_json_attr(&self, name: &str) -> Result<Value> {
        self.group
            .attributes()
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Attribute {name} not found"))
    }
}

impl DatasetOp<Zarr> for ZarrDataset {
    fn dtype(&self) -> Result<ScalarType> {
        let name = format!("{:?}", self.dataset.data_type());
        let name = name.to_lowercase();
        if name.contains("uint8") {
            Ok(ScalarType::U8)
        } else if name.contains("uint16") {
            Ok(ScalarType::U16)
        } else if name.contains("uint32") {
            Ok(ScalarType::U32)
        } else if name.contains("uint64") {
            Ok(ScalarType::U64)
        } else if name.contains("int8") {
            Ok(ScalarType::I8)
        } else if name.contains("int16") {
            Ok(ScalarType::I16)
        } else if name.contains("int32") {
            Ok(ScalarType::I32)
        } else if name.contains("int64") {
            Ok(ScalarType::I64)
        } else if name.contains("float32") {
            Ok(ScalarType::F32)
        } else if name.contains("float64") {
            Ok(ScalarType::F64)
        } else if name.contains("bool") {
            Ok(ScalarType::Bool)
        } else if name.contains("string") {
            Ok(ScalarType::String)
        } else {
            bail!("Unsupported data type: {name}")
        }
    }

    fn shape(&self) -> Shape {
        self.dataset.shape().iter().map(|x| *x as usize).collect()
    }

    fn reshape(&mut self, shape: &Shape) -> Result<()> {
        self.dataset
            .set_shape(shape.as_ref().iter().map(|x| *x as u64).collect())?;
        self.dataset.store_metadata()?;
        Ok(())
    }

    fn read_array_slice<T, S, D>(&self, selection: &[S]) -> Result<Array<T, D>>
    where
        T: BackendData,
        S: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        fn read_arr<T, S, D>(dataset: &ZarrDataset, selection: &[S]) -> Result<Array<T, D>>
        where
            T: ElementOwned + 'static,
            S: AsRef<SelectInfoElem>,
            D: Dimension,
        {
            let selection_bounds = SelectInfoBounds::new(&selection, &dataset.shape());
            if let Some(subset) = to_array_subset(selection_bounds.clone()) {
                let arr = dataset
                    .dataset
                    .retrieve_array_subset_sharded_opt::<ndarray::ArrayD<T>>(
                        &dataset.cache,
                        &subset,
                        &CodecOptions::default(),
                    )?
                    .into_dimensionality::<D>()?;
                Ok(arr)
            } else {
                // Read the entire array and then select the slice.
                let arr = dataset
                    .dataset
                    .retrieve_array_subset_sharded_opt::<ndarray::ArrayD<T>>(
                        &dataset.cache,
                        &dataset.dataset.subset_all(),
                        &CodecOptions::default(),
                    )?
                    .into_dimensionality::<D>()?;
                Ok(select(arr.view(), selection))
            }
        }

        let array: DynArray = match T::DTYPE {
            ScalarType::U8 => read_arr::<u8, _, D>(self, selection)?.into(),
            ScalarType::U16 => read_arr::<u16, _, D>(self, selection)?.into(),
            ScalarType::U32 => read_arr::<u32, _, D>(self, selection)?.into(),
            ScalarType::U64 => read_arr::<u64, _, D>(self, selection)?.into(),
            ScalarType::I8 => read_arr::<i8, _, D>(self, selection)?.into(),
            ScalarType::I16 => read_arr::<i16, _, D>(self, selection)?.into(),
            ScalarType::I32 => read_arr::<i32, _, D>(self, selection)?.into(),
            ScalarType::I64 => read_arr::<i64, _, D>(self, selection)?.into(),
            ScalarType::F32 => read_arr::<f32, _, D>(self, selection)?.into(),
            ScalarType::F64 => read_arr::<f64, _, D>(self, selection)?.into(),
            ScalarType::Bool => read_arr::<bool, _, D>(self, selection)?.into(),
            ScalarType::String => read_arr::<String, _, D>(self, selection)?.into(),
        };
        Ok(BackendData::from_dyn_arr(array)?.into_dimensionality::<D>()?)
    }

    fn write_array_slice<S, T, D>(&self, arr: CowArray<'_, T, D>, selection: &[S]) -> Result<()>
    where
        T: BackendData,
        S: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        fn write_array_impl<T, S>(
            container: &ZarrDataset,
            arr: CowArray<'_, T, IxDyn>,
            selection: &[S],
        ) -> Result<()>
        where
            T: Element + 'static,
            S: AsRef<SelectInfoElem>,
        {
            let selection_bounds = SelectInfoBounds::new(&selection, &container.shape());
            let starts: Vec<_> = selection_bounds
                .iter()
                .flat_map(|x| {
                    if let SelectInfoElemBounds::Slice(slice) = x {
                        if slice.step == 1 {
                            Some(slice.start as u64)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            if starts.len() == selection_bounds.ndim() {
                let arr = arr.as_standard_layout().into_owned();
                let subset = ArraySubset::new_with_start_shape(
                    starts,
                    arr.shape().iter().map(|&x| x as u64).collect(),
                )?;
                container.dataset.store_array_subset(&subset, arr)?;
            } else {
                panic!("Not implemented");
            }
            Ok(())
        }

        match BackendData::into_dyn_arr(arr.into_dyn()) {
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
            DynCowArray::String(x) => write_array_impl(self, x, selection),
        }
    }
}

impl AttributeOp<Zarr> for ZarrDataset {
    /// Returns the Root.
    fn store(&self) -> Result<<Zarr as Backend>::Store> {
        Ok(self.store.clone())
    }

    /// Returns the path of the location relative to the file root.
    fn path(&self) -> PathBuf {
        self.dataset.path().as_path().to_path_buf()
    }

    /// Write an attribute at a given location.
    fn new_json_attr(&mut self, name: &str, value: &Value) -> Result<()> {
        self.dataset
            .attributes_mut()
            .insert(name.to_string(), value.clone());
        self.dataset.store_metadata()?;
        Ok(())
    }

    /// Get an attribute at a given location.
    fn get_json_attr(&self, name: &str) -> Result<Value> {
        self.dataset
            .attributes()
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Attribute {name} not found"))
    }
}

fn select<'a, S, T, D>(arr: ArrayView<'a, T, D>, info: &[S]) -> Array<T, D>
where
    S: AsRef<SelectInfoElem>,
    T: Clone,
    D: Dimension,
{
    let arr = arr.into_dyn();
    let slices = info
        .as_ref()
        .iter()
        .map(|x| match x.as_ref() {
            SelectInfoElem::Slice(slice) => Some(SliceInfoElem::from(*slice)),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    if let Some(slices) = slices {
        arr.slice(slices.as_slice()).into_owned()
    } else {
        let shape = arr.shape();
        let select: Vec<_> = info
            .as_ref()
            .iter()
            .zip(shape)
            .map(|(x, n)| SelectInfoElemBounds::new(x.as_ref(), *n))
            .collect();
        let new_shape = select.iter().map(|x| x.len()).collect::<Vec<_>>();
        ArrayD::from_shape_fn(new_shape, |idx| {
            let new_idx: Vec<_> = (0..idx.ndim()).map(|i| select[i].index(idx[i])).collect();
            arr.index(new_idx.as_slice()).clone()
        })
    }
    .into_dimensionality::<D>()
    .unwrap()
}

fn str_to_prefix(s: &str) -> StorePrefix {
    if s.is_empty() {
        StorePrefix::root()
    } else {
        let s = s.trim_matches('/').to_string();
        StorePrefix::new((s + "/").as_str()).unwrap()
    }
}

fn canoincalize_path<'a>(path: &'a str) -> Cow<'a, str> {
    if path.starts_with("/") {
        path.into()
    } else {
        format!("/{path}").into()
    }
}

fn to_array_subset(info: SelectInfoBounds) -> Option<ArraySubset> {
    let ranges = info
        .iter()
        .map(|x| {
            if let SelectInfoElemBounds::Slice(slice) = x {
                if slice.step == 1 {
                    Some(slice.start as u64..slice.end as u64)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ArraySubset::new_with_ranges(&ranges))
}

fn new_empty_dataset_helper<T: BackendData, S: ?Sized + ReadableWritableListableStorageTraits>(
    store: Arc<S>,
    path: &str,
    shape: &Shape,
    config: WriteConfig,
) -> Result<zarrs::array::Array<S>> {
    let (datatype, fill): (zarrs::array::DataType, FillValue) = match T::DTYPE {
        ScalarType::U8 => (data_type::uint8(), 0u8.into()),
        ScalarType::U16 => (data_type::uint16(), 0u16.into()),
        ScalarType::U32 => (data_type::uint32(), 0u32.into()),
        ScalarType::U64 => (data_type::uint64(), 0u64.into()),
        ScalarType::I8 => (data_type::int8(), 0i8.into()),
        ScalarType::I16 => (data_type::int16(), 0i16.into()),
        ScalarType::I32 => (data_type::int32(), 0i32.into()),
        ScalarType::I64 => (data_type::int64(), 0i64.into()),
        ScalarType::F32 => (data_type::float32(), zarrs::array::ZARR_NAN_F32.into()),
        ScalarType::F64 => (data_type::float64(), zarrs::array::ZARR_NAN_F64.into()),
        ScalarType::Bool => (data_type::bool(), false.into()),
        ScalarType::String => (data_type::string(), "".into()),
    };

    let shape_ref = shape.as_ref();
    let chunk_size: Vec<u64> = match config.block_size {
        Some(ref s) => s.as_ref().iter().map(|x| (*x).max(1) as u64).collect(),
        _ => {
            if shape_ref.len() == 1 {
                vec![shape_ref[0].clamp(1, 16384) as u64]
            } else {
                shape_ref.iter().map(|&x| x.clamp(1, 128) as u64).collect()
            }
        }
    };

    let chunk_shape: ChunkShape = chunk_size
        .iter()
        .map(|&x| NonZeroU64::new(x).unwrap())
        .collect::<Vec<_>>();

    let builder = zarrs::array::ArrayBuilder::new(
        shape_ref.iter().map(|x| *x as u64).collect::<Vec<_>>(),
        chunk_shape.clone(),
        datatype.clone(),
        fill,
    );

    let array = { builder.build(store, path)? };

    Ok(array)
}

/// test module
#[cfg(test)]
mod tests {
    use super::*;
    use anndata::s;
    use ndarray::{Array2, Axis, Ix2, array, concatenate};
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
        with_tmp_dir(|dir| func(dir.join("temp")))
    }

    #[test]
    fn test_basic() -> Result<()> {
        with_tmp_path(|path| {
            let store = Zarr::new(&path)?;
            store.open_group("/")?;

            store.new_scalar_dataset("data", &4)?;
            store.open_dataset("data")?;

            let group = store.new_group("group")?;
            assert!(store.exists("group")?);

            let subgroup = group.new_group("group")?;
            assert!(group.exists("group")?);

            let subsubgroup = subgroup.new_group("group")?;
            assert!(subgroup.exists("group")?);

            let data = subsubgroup.new_scalar_dataset("group", &4)?;
            assert!(subsubgroup.exists("group")?);
            subsubgroup.open_dataset("group")?;

            {
                let store = Zarr::open(&path)?;
                DataContainer::open(&store, "group")?;
            }

            assert_eq!(group.path(), PathBuf::from("/group"));
            assert_eq!(subgroup.path(), PathBuf::from("/group/group"));
            assert_eq!(subsubgroup.path(), PathBuf::from("/group/group/group"));
            assert_eq!(data.path(), PathBuf::from("/group/group/group/group"));
            Ok(())
        })
    }

    #[test]
    fn test_write_empty() -> Result<()> {
        with_tmp_path(|path| {
            let store = Zarr::new(&path)?;
            let group = store.new_group("group")?;
            let config = WriteConfig {
                ..Default::default()
            };

            let empty: Array2<i64> = array![[]];
            let dataset = group.new_array_dataset("test", empty.view().into(), config)?;
            assert_eq!(empty, dataset.read_array::<i64, Ix2>()?);
            Ok(())
        })
    }

    #[test]
    fn test_write_slice() -> Result<()> {
        let store = Zarr::new("test_zarr")?;
        let config = WriteConfig {
            block_size: Some(vec![2, 2].as_slice().into()),
            ..Default::default()
        };

        let group = store.new_group("group")?;
        let mut dataset =
            group.new_empty_dataset::<i32>("test", &[20, 50].as_slice().into(), config)?;

        let arr = Array::random((10, 10), Uniform::new(0, 100).unwrap());
        dataset.write_array_slice(arr.view().into(), s![5..15, 10..20].as_ref())?;
        assert_eq!(
            arr,
            dataset.read_array_slice::<i32, _, _>(s![5..15, 10..20].as_ref())?
        );

        // Repeatitive writes
        let arr = Array::random((20, 50), Uniform::new(0, 100).unwrap());
        dataset.write_array_slice(arr.view().into(), s![.., ..].as_ref())?;
        dataset.write_array_slice(arr.view().into(), s![.., ..].as_ref())?;

        // Out-of-bounds writes should fail
        //assert!(dataset.write_array_slice(&arr, s![20..40, ..].as_ref()).is_err());

        // Reshape and write
        dataset.reshape(&[40, 50].as_slice().into())?;
        dataset.write_array_slice(arr.view().into(), s![20..40, ..].as_ref())?;

        // Read back is OK
        let merged = concatenate(Axis(0), &[arr.view(), arr.view()])?;
        assert_eq!(merged, dataset.read_array::<i32, _>()?);

        // Shrinking is OK
        dataset.reshape(&[20, 50].as_slice().into())?;
        assert_eq!(arr, dataset.read_array::<i32, _>()?);

        dataset.reshape(&[50, 50].as_slice().into())?;
        assert_eq!(
            [50, 50],
            store
                .open_group("group")?
                .open_dataset("test")?
                .shape()
                .as_ref(),
        );

        assert_eq!(vec!["group"], store.list()?);
        assert_eq!(vec!["test"], group.list()?);

        assert!(store.exists("group")?);
        assert!(group.exists("test")?);

        store.delete("group")?;
        assert!(!store.exists("group")?);
        assert!(!group.exists("test")?);

        Ok(())
    }
}
