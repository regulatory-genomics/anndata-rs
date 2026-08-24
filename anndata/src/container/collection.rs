use crate::{
    ElemCollectionOp,
    backend::{AttributeOp, Backend, DataContainer, GroupOp, iter_containers},
    container::base::*,
    data::*,
};

use anyhow::{Context, Result, bail, ensure};
use itertools::Itertools;
use log::warn;
use parking_lot::{Mutex, MutexGuard};
use smallvec::{SmallVec, smallvec};
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub struct InnerElemCollection<B: Backend> {
    container: B::Group,
    data: HashMap<String, Elem<B>>,
}

impl<B: Backend> std::fmt::Debug for InnerElemCollection<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl<B: Backend> Deref for InnerElemCollection<B> {
    type Target = HashMap<String, Elem<B>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<B: Backend> DerefMut for InnerElemCollection<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<B: Backend> std::fmt::Display for InnerElemCollection<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let keys = self
            .keys()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Dict with keys: {keys}")
    }
}

impl<B: Backend> InnerElemCollection<B> {
    pub fn add_data(&mut self, key: &str, data: Data) -> Result<()> {
        match self.get_mut(key) {
            None => {
                let container = data.write(&self.container, key)?;
                self.insert(key.to_string(), container.try_into()?);
            }
            Some(elem) => elem.inner().save(data)?,
        }
        Ok(())
    }

    pub fn remove_data(&mut self, key: &str) -> Result<()> {
        self.remove(key).map(|x| x.clear()).transpose()?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct ElemCollection<B: Backend>(Slot<InnerElemCollection<B>>);

impl<B: Backend> Clone for ElemCollection<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B: Backend> Deref for ElemCollection<B> {
    type Target = Slot<InnerElemCollection<B>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend> Display for ElemCollection<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<B: Backend> DerefMut for ElemCollection<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<B: Backend> ElemCollectionOp for &ElemCollection<B> {
    fn keys(&self) -> Vec<String> {
        if self.is_empty() {
            Vec::new()
        } else {
            self.inner().keys().cloned().collect()
        }
    }

    fn get_item<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: TryFrom<Data>,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>,
    {
        let mut lock = self.lock();
        if let Some(elem) = lock.as_mut().and_then(|x| x.get_mut(key)) {
            Ok(Some(elem.inner().data()?.try_into().map_err(Into::into)?))
        } else {
            Ok(None)
        }
    }

    fn add<D: Into<Data>>(&self, key: &str, data: D) -> Result<()> {
        self.inner().add_data(key, data.into())
    }

    fn remove(&self, key: &str) -> Result<()> {
        self.inner().remove_data(key)
    }
}

impl<B: Backend> ElemCollection<B> {
    pub fn empty() -> Self {
        Self(Slot::none())
    }

    pub fn is_empty(&self) -> bool {
        self.is_none() || self.inner().data.is_empty()
    }

    pub fn new(container: B::Group) -> Result<Self> {
        let data: Result<HashMap<_, _>> = iter_containers(&container)
            .map(|(k, v)| Ok((k, Elem::try_from(v)?)))
            .collect();
        let collection = InnerElemCollection {
            container,
            data: data?,
        };
        Ok(Self(Slot::new(collection)))
    }

    pub fn clear(&self) -> Result<()> {
        self.0
            .lock()
            .as_ref()
            .map(|x| {
                let g = &x.container;
                g.store()?.delete(&g.path().to_string_lossy())
            })
            .transpose()?;
        self.0.drop();
        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Axis {
    Row,       // Can perform row-wise operations.
    RowColumn, // Can perform row-wise and/or column-wise operations.
    Pairwise,  // Operations are carried out on both rows and columns at the same time.
}

/// Nullable dimension. None means that the dimension is not set.
/// Dimension can be set only once. Once set, it cannot be changed.
/// Dim is wrapped in Arc<Mutex<>> to allow sharing between multiple AxisArrays
/// and other anndata structures.
/// Dim is used to ensure that the shapes of arrays added to AnnData are consistent.
#[derive(Debug, Clone)]
pub struct Dim(Arc<Mutex<Option<usize>>>);

impl Dim {
    pub fn empty() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }

    pub fn new(n: usize) -> Self {
        Self(Arc::new(Mutex::new(Some(n))))
    }

    pub fn lock(&self) -> DimLock<'_> {
        DimLock(self.0.lock())
    }

    pub fn try_lock(&self) -> Option<DimLock<'_>> {
        self.0.try_lock().map(DimLock)
    }

    pub fn get(&self) -> usize {
        self.lock().get()
    }

    pub fn try_set(&self, n: usize) -> Result<()> {
        self.lock().try_set(n)
    }
}

impl Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lock = self.lock().0;
        if lock.is_none() {
            write!(f, "None")?;
        } else {
            write!(f, "{}", lock.unwrap())?;
        }
        Ok(())
    }
}

/// A lock guard for Dim.
pub struct DimLock<'a>(MutexGuard<'a, Option<usize>>);

impl DimLock<'_> {
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    pub fn get(&self) -> usize {
        self.0.unwrap_or(0)
    }

    pub fn try_set(&mut self, n: usize) -> Result<()> {
        if self.0.is_some() && self.0.unwrap() != n {
            bail!(
                "dimension cannot be changed from {} to {}",
                self.0.unwrap(),
                n
            );
        } else {
            *self.0 = Some(n);
        }
        Ok(())
    }

    pub(crate) fn set(&mut self, n: usize) {
        *self.0 = Some(n);
    }
}

pub struct InnerAxisArrays<B: Backend> {
    axis: Axis,
    container: B::Group,
    dim1: Option<Dim>, // If dim1 is None, mutating AxisArrays is not allowed.
    dim2: Option<Dim>, // If dim2 is None, mutating RowColumn AxisArrays is not allowed.
    names: Option<&'static str>, // Root dataframe ("obs"/"var") holding the names of the first axis.
    data: HashMap<String, ArrayElem<B>>,
}

impl<B: Backend> std::fmt::Debug for InnerAxisArrays<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl<B: Backend> Deref for InnerAxisArrays<B> {
    type Target = HashMap<String, ArrayElem<B>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<B: Backend> DerefMut for InnerAxisArrays<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<B: Backend> std::fmt::Display for InnerAxisArrays<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = match self.axis {
            Axis::Row => "row",
            Axis::RowColumn => "row/column",
            Axis::Pairwise => "pairwise",
        };
        let keys = self
            .keys()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "AxisArrays ({ty}) with keys: {keys}")
    }
}

impl<B: Backend> InnerAxisArrays<B> {
    /// Return the length of the first dimension.
    pub fn size(&self) -> usize {
        if let Some(dim) = &self.dim1 {
            dim.get()
        } else {
            self.data
                .values()
                .next()
                .map_or(0, |x| x.inner().shape()[0])
        }
    }

    fn dim1(&mut self) -> &Dim {
        self.dim1.as_ref().expect("Dim1 is not set!")
    }

    fn dim2(&mut self) -> &Dim {
        self.dim2.as_ref().expect("Dim2 is not set!")
    }

    /// Names along the first axis, i.e. the index dataframes stored here must have.
    fn axis_names(&self) -> Result<Option<DataFrameIndex>> {
        let Some(names) = self.names else {
            return Ok(None);
        };
        let store = self.container.store()?;
        if !store.exists(names)? {
            return Ok(None);
        }
        let index = DataFrameIndex::read(&DataContainer::<B>::open(&store, names)?)?;
        Ok((!index.is_empty()).then_some(index))
    }

    /// Re-index the dataframes stored here with the current axis names.
    pub(crate) fn update_index(&mut self) -> Result<()> {
        if let Some(index) = self.axis_names()? {
            self.data
                .values()
                .try_for_each(|x| x.inner().set_index(&index))?;
        }
        Ok(())
    }

    pub fn add_data<D: Into<ArrayData>>(&mut self, key: &str, data: D) -> Result<()> {
        // Check if the data is compatible with the current size
        let data = data.into();
        let shape = data.shape();
        let is_df = matches!(data, ArrayData::DataFrame(_));
        match self.axis {
            Axis::Row => {
                self.dim1().try_set(shape[0])?;
            }
            Axis::RowColumn => {
                self.dim1().try_set(shape[0])?;
                self.dim2().try_set(shape[1])?;
            }
            Axis::Pairwise => {
                ensure!(
                    shape[0] == shape[1],
                    "expecting a square array, but receive a {shape:?} array"
                );
                self.dim1().try_set(shape[0])?;
            }
        }

        match self.get_mut(key) {
            None => {
                let container = data.write(&self.container, key)?;
                let elem = container.try_into()?;
                self.insert(key.to_string(), elem);
            }
            Some(elem) => elem.inner().save(data)?,
        }

        // A polars dataframe carries no index, so stamp the axis names onto the
        // one that was just written.
        if is_df && let Some(index) = self.axis_names()? {
            self.data[key].inner().set_index(&index)?;
        }
        Ok(())
    }

    pub fn add_data_from_iter<I, D>(&mut self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk,
    {
        if let Some(elem) = self.get(key) {
            elem.clear()?;
        }
        let elem = ArrayChunk::write_by_chunk(data, &self.container, key)
            .with_context(|| format!("failed to write data to AxisArrays with key: '{key}'"))?;
        let elem = ArrayElem::try_from(elem)?;

        let shape = { elem.inner().shape().clone() };
        match self.axis {
            Axis::Row => {
                if let Err(e) = self.dim1().try_set(shape[0]) {
                    elem.clear()?;
                    bail!(e)
                } else {
                    self.insert(key.to_string(), elem);
                    Ok(())
                }
            }
            Axis::RowColumn => {
                if let Err(e) = self
                    .dim1()
                    .try_set(shape[0])
                    .and(self.dim2.as_ref().unwrap().try_set(shape[1]))
                {
                    elem.clear()?;
                    bail!(e)
                } else {
                    self.insert(key.to_string(), elem);
                    Ok(())
                }
            }
            Axis::Pairwise => {
                if shape[0] != shape[1] {
                    elem.clear()?;
                    bail!("expecting a square array, but receive a {shape:?} array")
                } else if let Err(e) = self.dim1().try_set(shape[0]) {
                    elem.clear()?;
                    bail!(e)
                } else {
                    self.insert(key.to_string(), elem);
                    Ok(())
                }
            }
        }
    }

    pub fn remove_data(&mut self, key: &str) -> Result<()> {
        self.remove(key).map(|x| x.clear()).transpose()?;
        Ok(())
    }

    pub(crate) fn subset(&mut self, selection: &[&SelectInfoElem]) -> Result<()> {
        match self.axis {
            Axis::Row => {
                if selection.len() != 1 {
                    bail!("selection dimension must be 1 for row AxisArrays");
                }
                self.values()
                    .try_for_each(|x| x.inner().subset_axis(0, selection[0]))?;

                // We use try_lock here to avoid deadlock in case dim1 is shared with other structures.
                if let Some(mut lock) = self.dim1.as_ref().unwrap().try_lock() {
                    lock.set(SelectInfoElemBounds::new(selection[0], lock.get()).len());
                }
            }
            Axis::RowColumn => {
                if selection.len() != 2 {
                    bail!("selection dimension must be 2 for row/column AxisArrays");
                }
                self.values()
                    .try_for_each(|x| x.inner().subset(selection))?;
                if let Some(mut lock) = self.dim1.as_ref().unwrap().try_lock() {
                    lock.set(SelectInfoElemBounds::new(selection[0], lock.get()).len());
                }
                if let Some(mut lock) = self.dim2.as_ref().unwrap().try_lock() {
                    lock.set(SelectInfoElemBounds::new(selection[1], lock.get()).len());
                }
            }
            Axis::Pairwise => {
                if selection.len() != 1 {
                    bail!("selection dimension must be 1 for pairwise AxisArrays");
                }
                self.values().try_for_each(|x| {
                    let full = SelectInfoElem::full();
                    let mut slice: SmallVec<[_; 3]> = smallvec![&full; x.inner().shape().ndim()];
                    slice[0] = selection[0];
                    slice[1] = selection[0];
                    x.inner().subset(slice.as_slice())
                })?;
                if let Some(mut lock) = self.dim1.as_ref().unwrap().try_lock() {
                    lock.set(SelectInfoElemBounds::new(selection[0], lock.get()).len());
                }
            }
        }
        self.update_index()
    }
}

#[derive(Debug)]
pub struct AxisArrays<B: Backend>(Slot<InnerAxisArrays<B>>);

impl<B: Backend> std::fmt::Display for AxisArrays<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<B: Backend> Clone for AxisArrays<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B: Backend> Deref for AxisArrays<B> {
    type Target = Slot<InnerAxisArrays<B>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend> DerefMut for AxisArrays<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<B: Backend> AxisArrays<B> {
    pub fn empty() -> Self {
        Self(Slot::none())
    }

    pub fn is_empty(&self) -> bool {
        self.is_none() || self.inner().data.is_empty()
    }

    pub fn new(
        group: B::Group,
        axis: Axis,
        dim1: Option<&Dim>,
        dim2: Option<&Dim>,
        names: Option<&'static str>,
    ) -> Result<Self> {
        let data: HashMap<_, _> = iter_containers::<B>(&group)
            .map(|(k, v)| (k, ArrayElem::try_from(v).unwrap()))
            .collect();

        // Get shapes of arrays
        let shapes = data
            .values()
            .map(|v| v.inner().shape().clone())
            .collect::<Vec<_>>();

        // Check if shapes of arrays conform to axis
        ensure!(
            shapes.iter().map(|x| x[0]).all_equal(),
            "the size of the 1st dimension of arrays must be equal"
        );
        if let Axis::Pairwise = axis {
            ensure!(
                shapes.iter().all(|x| x[0] == x[1]),
                "the size of the 1st and 2nd dimension of arrays must be equal"
            );
        }
        if let Axis::RowColumn = axis {
            ensure!(
                shapes.iter().map(|x| x[1]).all_equal(),
                "the size of the 2nd dimension of arrays must be equal"
            );
        }

        if let Some(s) = shapes.first() {
            if let Some(d) = dim1 {
                d.try_set(s[0])?;
            }
            if let Axis::RowColumn = axis
                && let Some(d) = dim2
            {
                d.try_set(s[1])?;
            }
        }

        let arrays = InnerAxisArrays {
            container: group,
            dim1: dim1.cloned(),
            dim2: dim2.cloned(),
            names,
            axis,
            data,
        };
        Ok(Self(Slot::new(arrays)))
    }

    pub fn clear(&self) -> Result<()> {
        self.0
            .lock()
            .as_ref()
            .map(|x| {
                let g = &x.container;
                g.store()?.delete(&g.path().to_string_lossy())
            })
            .transpose()?;
        self.0.drop();
        Ok(())
    }
}

/// Stacked axis arrays, providing Read-only access to the data.
pub struct StackedAxisArrays<B: Backend> {
    axis: Axis,
    pub(crate) data: Arc<HashMap<String, StackedArrayElem<B>>>,
}

impl<B: Backend> Clone for StackedAxisArrays<B> {
    fn clone(&self) -> Self {
        Self {
            axis: self.axis,
            data: self.data.clone(),
        }
    }
}

impl<B: Backend> Deref for StackedAxisArrays<B> {
    type Target = HashMap<String, StackedArrayElem<B>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<B: Backend> std::fmt::Display for StackedAxisArrays<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = match self.axis {
            Axis::Row => "row",
            Axis::RowColumn => "row/column",
            Axis::Pairwise => "pairwise",
        };
        let keys = self
            .data
            .keys()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Stacked AxisArrays ({ty}) with keys: {keys}")
    }
}

impl<B: Backend> StackedAxisArrays<B> {
    pub fn empty(axis: Axis) -> Self {
        Self {
            axis,
            data: Arc::new(HashMap::new()),
        }
    }

    pub(crate) fn new(axis: Axis, arrays: Vec<AxisArrays<B>>) -> Result<Self> {
        if arrays.iter().any(|x| x.is_none()) {
            return Ok(Self::empty(axis));
        }

        ensure!(
            arrays.iter().all(|x| x.inner().axis == axis),
            "Axis mismatch"
        );

        let shared_keys: HashSet<String> = arrays
            .iter()
            .map(|x| x.inner().keys().cloned().collect::<HashSet<_>>())
            .reduce(|a, b| a.intersection(&b).cloned().collect())
            .unwrap_or(HashSet::new());

        let mut ignore_keys = Vec::new();
        let data = shared_keys
            .into_iter()
            .flat_map(|k| {
                let elems = arrays
                    .iter()
                    .map(|x| x.inner().get(&k).unwrap().clone())
                    .collect();
                if let Ok(arr) = StackedArrayElem::new(elems) {
                    Some((k, arr))
                } else {
                    ignore_keys.push(k);
                    None
                }
            })
            .collect::<HashMap<_, _>>();
        if !ignore_keys.is_empty() {
            warn!(
                "Unable to create stacked arrays for these keys: {}",
                ignore_keys.join(",")
            );
        }
        Ok(Self {
            axis,
            data: Arc::new(data),
        })
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
}
