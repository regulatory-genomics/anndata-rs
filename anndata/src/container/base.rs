use crate::{
    backend::{AttributeOp, Backend, DataContainer, DataType, GroupOp},
    data::index::VecVecIndex,
    data::*,
};

use anyhow::{Result, bail, ensure};
use indexmap::set::IndexSet;
use itertools::Itertools;
use num::integer::div_rem;
use parking_lot::{Mutex, MutexGuard};
use polars::{
    frame::DataFrame,
    prelude::{Column, IntoColumn, IntoLazy, UnionArgs, concat},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

/// Slot stores an optional object wrapped by Arc and Mutex.
/// Encapsulating an object inside a slot allows us to drop the object from all references.
#[derive(Debug)]
pub struct Slot<T>(pub(crate) Arc<Mutex<Option<T>>>);

impl<T> Clone for Slot<T> {
    fn clone(&self) -> Self {
        Slot(self.0.clone())
    }
}

impl<T> std::fmt::Display for Slot<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_none() {
            write!(f, "Empty or closed slot")
        } else {
            write!(f, "{}", self.inner().deref())
        }
    }
}

impl<T> Slot<T> {
    /// Create a slot from data.
    pub fn new(x: T) -> Self {
        Slot(Arc::new(Mutex::new(Some(x))))
    }

    /// Create an empty slot.
    pub fn none() -> Self {
        Slot(Arc::new(Mutex::new(None)))
    }

    pub fn is_none(&self) -> bool {
        self.0.lock().is_none()
    }

    /// Get a locked reference to the inner Option data.
    pub fn lock(&self) -> MutexGuard<'_, Option<T>> {
        self.0.lock()
    }

    /// Get a locked reference to the inner data.
    pub fn inner(&self) -> Inner<'_, T> {
        Inner(self.0.lock())
    }

    /// Insert data to the slot, and return the old data.
    pub fn insert(&self, data: T) -> Option<T> {
        self.0.lock().deref_mut().replace(data)
    }

    /// Extract the data from the slot. The slot becomes empty after this operation.
    pub fn extract(&self) -> Option<T> {
        self.0.lock().deref_mut().take()
    }

    /// Remove the data from the slot.
    pub fn drop(&self) {
        let _ = self.extract();
    }

    /// Swap the data between two slots.
    pub fn swap(&self, other: &Self) {
        let mut self_lock = self.0.lock();
        let mut other_lock = other.0.lock();
        std::mem::swap(self_lock.deref_mut(), other_lock.deref_mut());
    }
}

/// Locked inner data of a slot.
pub struct Inner<'a, T>(pub MutexGuard<'a, Option<T>>);

impl<T> Deref for Inner<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self.0.deref() {
            Some(x) => x,
            None => panic!("accessing an empty slot"),
        }
    }
}

impl<T> DerefMut for Inner<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.0.deref_mut() {
            None => panic!("accessing an empty slot"),
            Some(x) => x,
        }
    }
}

#[derive(Debug)]
pub struct InnerDataFrameElem<B: Backend> {
    element: Option<DataFrame>,
    container: DataContainer<B>,
    column_names: IndexSet<String>,
    pub index: DataFrameIndex,
}

impl<B: Backend> InnerDataFrameElem<B> {
    pub fn new<G: GroupOp<B>>(
        location: &G,
        name: &str,
        index: Option<DataFrameIndex>,
        df: &DataFrame,
    ) -> Result<Self> {
        let index = index.unwrap_or(df.height().into());
        ensure!(
            df.height() == 0 || index.len() == df.height(),
            "cannot create dataframe element as lengths of index and dataframe differ"
        );
        let mut container = df.write(location, name)?;
        index.overwrite(&mut container)?;

        let column_names = df
            .get_column_names()
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        Ok(Self {
            element: None,
            container,
            column_names,
            index,
        })
    }
}

impl<B: Backend> std::fmt::Display for InnerDataFrameElem<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dataframe element")
    }
}

impl<B: Backend> InnerDataFrameElem<B> {
    pub fn width(&self) -> usize {
        self.column_names.len()
    }

    pub fn height(&self) -> usize {
        self.index.len()
    }

    pub fn column(&mut self, name: &str) -> Result<&Column> {
        self.data().and_then(|x| Ok(x.column(name)?))
    }

    pub fn get_column_names(&self) -> &IndexSet<String> {
        &self.column_names
    }

    /// Set a column with a Series.
    //TODO: this is not efficient. We should be able to replace a column without reading the whole dataframe.
    pub fn set_column<C: IntoColumn>(&mut self, name: &str, new_col: C) -> Result<()> {
        let mut column = new_col.into_column();
        column.rename(name.into());

        let mut df = self.data()?.clone();
        df.with_column(column)?;
        self.save(df)
    }

    pub fn set_index(&mut self, index: DataFrameIndex) -> Result<()> {
        ensure!(
            self.index.len() == index.len(),
            "cannot change the index as the lengths differ"
        );
        self.index = index;
        self.index.overwrite(&mut self.container)?;
        Ok(())
    }

    pub fn data(&mut self) -> Result<&DataFrame> {
        match self.element {
            Some(ref df) => Ok(df),
            None => {
                let df = DataFrame::read(&self.container)?;
                self.element = Some(df);
                Ok(self.element.as_ref().unwrap())
            }
        }
    }

    pub fn select<S>(&mut self, selection: &[S]) -> Result<DataFrame>
    where
        S: AsRef<SelectInfoElem>,
    {
        Ok(Selectable::select(self.data()?, selection))
    }

    pub fn select_axis<S>(&mut self, axis: usize, selection: S) -> Result<DataFrame>
    where
        S: AsRef<SelectInfoElem>,
    {
        Ok(Selectable::select_axis(self.data()?, axis, selection))
    }

    pub fn save(&mut self, data: DataFrame) -> Result<()> {
        let num_recs = data.height();
        ensure!(
            num_recs == 0 || self.index.len() == num_recs,
            "cannot update dataframe as lengths differ"
        );
        let new = data.overwrite(std::mem::take(&mut self.container))?;
        let _ = std::mem::replace(&mut self.container, new);
        self.column_names = data
            .get_column_names()
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        if self.element.is_some() {
            self.element = Some(data);
        }
        Ok(())
    }

    /// inplace subsetting the DataFrameElem.
    pub fn subset<S>(&mut self, selection: &[S]) -> Result<()>
    where
        S: AsRef<SelectInfoElem>,
    {
        self.index = self.index.select(selection[0].as_ref());
        self.index.overwrite(&mut self.container)?;
        let df = self.select(selection)?;
        self.save(df)
    }

    pub fn subset_axis<S>(&mut self, axis: usize, selection: S) -> Result<()>
    where
        S: AsRef<SelectInfoElem>,
    {
        let full = SelectInfoElem::full();
        let slice = selection.as_ref().set_axis(axis, 2, &full);
        self.subset(slice.as_slice())
    }
}

pub type DataFrameElem<B> = Slot<InnerDataFrameElem<B>>;

impl<B: Backend> TryFrom<DataContainer<B>> for DataFrameElem<B> {
    type Error = anyhow::Error;

    fn try_from(container: DataContainer<B>) -> Result<Self> {
        match container.encoding_type()? {
            DataType::DataFrame => {
                //let grp = container.as_group()?;
                let index = DataFrameIndex::read(&container)?;
                let column_names = container
                    .get_attr::<Vec<String>>("column-order")?
                    .into_iter()
                    .collect();
                let df = InnerDataFrameElem {
                    element: None,
                    container,
                    column_names,
                    index,
                };
                Ok(Slot::new(df))
            }
            ty => bail!("Expecting a dataframe but found: '{ty}'"),
        }
    }
}

impl<B: Backend> DataFrameElem<B> {
    /// Delete and Remove the data from the element.
    pub fn clear(&self) -> Result<()> {
        if let Some(elem) = self.extract() {
            DataContainer::delete(elem.container)?;
        }
        Ok(())
    }
}

/// Container holding general data types.
#[derive(Debug)]
pub struct InnerElem<B: Backend> {
    dtype: DataType,
    cache_enabled: bool,
    container: DataContainer<B>,
    element: Option<Data>,
}

impl<B: Backend> std::fmt::Display for InnerElem<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} element, cache_enabled: {}, cached: {}",
            self.dtype,
            if self.cache_enabled { "yes" } else { "no" },
            if self.element.is_some() { "yes" } else { "no" },
        )
    }
}

impl<B: Backend> InnerElem<B> {
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn enable_cache(&mut self) {
        self.cache_enabled = true;
    }

    pub fn disable_cache(&mut self) {
        if self.element.is_some() {
            self.element = None;
        }
        self.cache_enabled = false;
    }

    pub fn data(&mut self) -> Result<Data> {
        match self.element.as_ref() {
            Some(data) => Ok(data.clone()),
            None => {
                let data = Data::read(&self.container)?;
                if self.cache_enabled {
                    self.element = Some(data.clone());
                }
                Ok(data)
            }
        }
    }

    pub(crate) fn save(&mut self, data: Data) -> Result<()> {
        let new = data.overwrite(std::mem::take(&mut self.container))?;
        let _ = std::mem::replace(&mut self.container, new);
        self.dtype = data.data_type();
        if self.element.is_some() {
            self.element = Some(data);
        }
        Ok(())
    }
}

pub type Elem<B> = Slot<InnerElem<B>>;

impl<B: Backend> TryFrom<DataContainer<B>> for Elem<B> {
    type Error = anyhow::Error;

    fn try_from(container: DataContainer<B>) -> Result<Self> {
        let dtype = container.encoding_type()?;
        let elem = InnerElem {
            dtype,
            cache_enabled: false,
            element: None,
            container,
        };
        Ok(Slot::new(elem))
    }
}

impl<B: Backend> Elem<B> {
    /// Delete and Remove the data from the element.
    pub fn clear(&self) -> Result<()> {
        if let Some(elem) = self.extract() {
            DataContainer::delete(elem.container)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct InnerArrayElem<B: Backend> {
    dtype: DataType,
    shape: Shape,
    cache_enabled: bool,
    container: DataContainer<B>,
    element: Option<ArrayData>,
}

impl<B: Backend> std::fmt::Display for InnerArrayElem<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} element, cache_enabled: {}, cached: {}",
            self.dtype,
            if self.cache_enabled { "yes" } else { "no" },
            if self.element.is_some() { "yes" } else { "no" },
        )
    }
}

impl<B: Backend> InnerArrayElem<B> {
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn is_cached(&self) -> bool {
        self.element.is_some()
    }

    pub fn enable_cache(&mut self) {
        self.cache_enabled = true;
    }

    pub fn disable_cache(&mut self) {
        if self.element.is_some() {
            self.element = None;
        }
        self.cache_enabled = false;
    }

    pub fn data(&mut self) -> Result<ArrayData> {
        match self.element.as_ref() {
            Some(data) => Ok(data.clone()),
            None => {
                let data = ArrayData::read(&self.container)?;
                if self.cache_enabled {
                    self.element = Some(data.clone());
                }
                Ok(data)
            }
        }
    }

    pub fn take(&mut self) -> Result<ArrayData> {
        match self.element.take() {
            Some(data) => Ok(data),
            None => ArrayData::read(&self.container),
        }
    }

    pub(crate) fn save(&mut self, data: ArrayData) -> Result<()> {
        let new = data.overwrite(std::mem::take(&mut self.container))?;
        let _ = std::mem::replace(&mut self.container, new);
        self.dtype = data.data_type();
        self.shape = data.shape();
        if self.element.is_some() {
            self.element = Some(data);
        }
        Ok(())
    }

    pub fn select<S>(&mut self, selection: &[S]) -> Result<ArrayData>
    where
        S: AsRef<SelectInfoElem>,
    {
        let ndim = self.shape().ndim();
        let full = SelectInfoElem::full();
        if selection
            .iter()
            .map(|x| x.as_ref())
            .chain(std::iter::repeat(&full))
            .take(ndim)
            .all(|x| x.is_full())
        {
            self.data()
        } else {
            match self.element.as_ref() {
                Some(data) => Ok(data.select(selection)),
                None => ArrayData::read_select(&self.container, selection),
            }
        }
    }

    pub fn select_axis<S>(&mut self, axis: usize, selection: S) -> Result<ArrayData>
    where
        S: AsRef<SelectInfoElem>,
    {
        let full = SelectInfoElem::full();
        let slice = selection
            .as_ref()
            .set_axis(axis, self.shape().ndim(), &full);
        self.select(slice.as_slice())
    }

    pub(crate) fn subset<S>(&mut self, selection: &[S]) -> Result<()>
    where
        S: AsRef<SelectInfoElem>,
    {
        let data = match self.element.as_ref() {
            Some(data) => data.select(selection),
            None => ArrayData::read_select(&self.container, selection)?,
        };

        self.shape = data.shape();
        let new = data.overwrite(std::mem::take(&mut self.container))?;
        let _ = std::mem::replace(&mut self.container, new);
        if self.element.is_some() {
            self.element = Some(data);
        }
        Ok(())
    }

    pub(crate) fn subset_axis<S>(&mut self, axis: usize, selection: S) -> Result<()>
    where
        S: AsRef<SelectInfoElem>,
    {
        let full = SelectInfoElem::full();
        let slice = selection
            .as_ref()
            .set_axis(axis, self.shape().ndim(), &full);
        self.subset(slice.as_slice())
    }
}

pub type ArrayElem<B> = Slot<InnerArrayElem<B>>;

/// Container holding matrix data types.
impl<B: Backend> TryFrom<DataContainer<B>> for ArrayElem<B> {
    type Error = anyhow::Error;

    fn try_from(container: DataContainer<B>) -> Result<Self> {
        let dtype = container.encoding_type()?;
        let elem = InnerArrayElem {
            dtype,
            shape: ArrayData::get_shape(&container)?,
            cache_enabled: false,
            element: None,
            container,
        };
        Ok(Slot::new(elem))
    }
}

impl<B: Backend> ArrayElem<B> {
    /// Delete and Remove the data from the element.
    pub fn clear(&self) -> Result<()> {
        if let Some(elem) = self.extract() {
            DataContainer::delete(elem.container)?;
        }
        Ok(())
    }

    pub fn chunked<D>(&self, chunk_size: usize) -> ChunkedArrayElem<B, D>
    where
        D: TryFrom<ArrayData>,
    {
        ChunkedArrayElem::new(self.clone(), chunk_size)
    }
}

/// Horizontal concatenated dataframe elements.
pub struct StackedDataFrame<B: Backend> {
    column_names: IndexSet<String>,
    elems: Arc<Vec<DataFrameElem<B>>>,
    index: VecVecIndex,
}

impl<B: Backend> Clone for StackedDataFrame<B> {
    fn clone(&self) -> Self {
        Self {
            column_names: self.column_names.clone(),
            elems: self.elems.clone(),
            index: self.index.clone(),
        }
    }
}

impl<B: Backend> std::fmt::Display for StackedDataFrame<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "stacked dataframe with columns: '{}'",
            self.column_names.iter().join("', '")
        )
    }
}

impl<B: Backend> StackedDataFrame<B> {
    pub fn is_empty(&self) -> bool {
        self.elems.iter().all(|x| x.is_none())
    }

    pub fn width(&self) -> usize {
        self.column_names.len()
    }

    pub fn height(&self) -> usize {
        self.elems.iter().map(|x| x.inner().height()).sum()
    }

    pub fn new(elems: Vec<DataFrameElem<B>>) -> Result<Self> {
        let index = elems
            .iter()
            .map(|x| x.lock().as_ref().map(|x| x.height()).unwrap_or(0))
            .collect();
        if elems.iter().all(|x| x.is_none()) {
            Ok(Self {
                column_names: IndexSet::new(),
                elems: Arc::new(elems),
                index,
            })
        } else if elems.iter().all(|x| !x.is_none()) {
            let column_names = elems
                .iter()
                .map(|x| x.inner().get_column_names().clone())
                .reduce(|shared_keys, next_keys| {
                    shared_keys
                        .intersection(&next_keys)
                        .map(|x| x.to_owned())
                        .collect()
                })
                .unwrap_or(IndexSet::new());
            Ok(Self {
                column_names,
                elems: Arc::new(elems),
                index,
            })
        } else {
            bail!("slots must be either all empty or all full");
        }
    }

    pub fn get_column_names(&self) -> &IndexSet<String> {
        &self.column_names
    }

    pub fn data(&self) -> Result<DataFrame> {
        let names = &self.column_names;
        let df = self
            .elems
            .iter()
            .map(|x| {
                let mut x = x.inner();
                let df = x.data().unwrap();
                names
                    .iter()
                    .map(|i| df[i.as_str()].clone())
                    .collect::<Vec<_>>()
            })
            .reduce(|mut acc, next| {
                acc.iter_mut().zip(next).for_each(|(a, b)| {
                    a.append(&b).unwrap();
                });
                acc
            })
            .map(|columns| DataFrame::new_infer_height(columns).unwrap())
            .unwrap_or(DataFrame::empty());
        Ok(df)
    }

    pub fn select<S>(&self, selection: &[S]) -> Result<DataFrame>
    where
        S: AsRef<SelectInfoElem>,
    {
        let (indices, mapping) = self.index.split_select(selection[0].as_ref());
        let dfs = self
            .elems
            .iter()
            .enumerate()
            .flat_map(|(i, el)| {
                indices.get(&i).and_then(|idx| {
                    let select: SmallVec<[_; 3]> = std::iter::once(idx)
                        .chain(selection[1..].iter().map(|x| x.as_ref()))
                        .collect();
                    el.lock()
                        .as_mut()
                        .map(|x| x.select(select.as_slice()).unwrap().lazy())
                })
            })
            .collect::<Vec<_>>();
        let df = concat(
            &dfs,
            UnionArgs {
                parallel: true,
                rechunk: true,
                ..Default::default()
            },
        )?
        .collect()?;
        if let Some(m) = mapping {
            let select: SmallVec<[SelectInfoElem; 3]> = std::iter::once(m.into())
                .chain(std::iter::repeat_n(
                    (..).into(),
                    selection.as_ref().len() - 1,
                ))
                .collect();
            Ok(Selectable::select(&df, select.as_slice()))
        } else {
            Ok(df)
        }
    }

    // TODO: this is not efficient, we should use the index to select the columns
    pub fn column(&self, name: &str) -> Result<Column> {
        if self.column_names.contains(name) {
            Ok(self.data()?.column(name)?.clone())
        } else {
            bail!("key is not present");
        }
    }
}

pub struct InnerStackedArrayElem<B: Backend> {
    pub(crate) shape: Option<Shape>,
    pub(crate) elems: SmallVec<[ArrayElem<B>; 96]>,
    index: VecVecIndex,
}

impl<B: Backend> std::fmt::Display for InnerStackedArrayElem<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.elems.is_empty() {
            write!(f, "empty stacked elements")
        } else {
            write!(
                f,
                "{} stacked elements ({}) with {}",
                self.shape.as_ref().unwrap(),
                self.elems.len(),
                self.elems[0].inner().dtype(),
            )
        }
    }
}

impl<B: Backend> InnerStackedArrayElem<B> {
    pub fn is_none(&self) -> bool {
        self.elems.is_empty() || self.elems.iter().all(|x| x.is_none())
    }

    pub fn shape(&self) -> &Option<Shape> {
        &self.shape
    }

    pub fn data<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let data = if self.is_none() {
            None
        } else {
            let array = self
                .elems
                .iter()
                .flat_map(|x| x.lock().as_mut().map(|i| i.data()))
                .process_results(|x| Stackable::vstack(x).unwrap())?;
            Some(array.try_into().map_err(Into::into)?)
        };
        Ok(data)
    }

    pub fn par_data<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let data = if self.is_none() {
            None
        } else {
            let array = self
                .elems
                .par_iter()
                .flat_map(|x| x.lock().as_mut().map(|i| i.data()))
                .collect::<Vec<_>>()
                .into_iter()
                .process_results(|x| Stackable::vstack(x).unwrap())?;
            Some(array.try_into().map_err(Into::into)?)
        };
        Ok(data)
    }

    pub fn select_axis<D, S>(&self, axis: usize, selection: S) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        S: AsRef<SelectInfoElem>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.shape
            .as_ref()
            .map(|s| {
                let n = s.ndim();
                let full = SelectInfoElem::full();
                let slice = selection.as_ref().set_axis(axis, n, &full);
                self.select(slice.as_slice()).map(|x| x.unwrap())
            })
            .transpose()
    }

    pub fn select<D, S>(&self, selection: &[S]) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        S: AsRef<SelectInfoElem>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let data = if self.is_none() {
            None
        } else {
            let (indices, mapping) = self.index.split_select(selection[0].as_ref());
            let array: ArrayData = self
                .elems
                .iter()
                .enumerate()
                .map(|(i, el)| {
                    if let Some(idx) = indices.get(&i) {
                        let select: SmallVec<[_; 3]> = std::iter::once(idx)
                            .chain(selection[1..].iter().map(|x| x.as_ref()))
                            .collect();
                        el.inner().select(select.as_slice())
                    } else {
                        let idx = SelectInfoElem::empty();
                        let select: SmallVec<[_; 3]> = std::iter::once(&idx)
                            .chain(selection[1..].iter().map(|x| x.as_ref()))
                            .collect();
                        el.inner().select(select.as_slice())
                    }
                })
                .process_results(|x| Stackable::vstack(x).unwrap())?;
            if let Some(m) = mapping {
                Some(
                    array
                        .select_axis(0, SelectInfoElem::from(reverse_mapping(m)))
                        .try_into()
                        .map_err(Into::into)?,
                )
            } else {
                Some(array.try_into().map_err(Into::into)?)
            }
        };
        Ok(data)
    }

    pub fn par_select<D, S>(&self, selection: &[S]) -> Result<Option<D>>
    where
        D: Into<ArrayData> + TryFrom<ArrayData> + ReadableArray + Clone,
        S: AsRef<SelectInfoElem> + Sync,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let data = if self.is_none() {
            None
        } else {
            let (indices, mapping) = self.index.split_select(selection[0].as_ref());
            let array: ArrayData = self
                .elems
                .par_iter()
                .enumerate()
                .flat_map(|(i, el)| {
                    indices.get(&i).map(|idx| {
                        let select: SmallVec<[_; 3]> = std::iter::once(idx)
                            .chain(selection[1..].iter().map(|x| x.as_ref()))
                            .collect();
                        el.inner().select(select.as_slice())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .process_results(|x| Stackable::vstack(x).unwrap())?;
            if let Some(m) = mapping {
                Some(
                    array
                        .select_axis(0, SelectInfoElem::from(reverse_mapping(m)))
                        .try_into()
                        .map_err(Into::into)?,
                )
            } else {
                Some(array.try_into().map_err(Into::into)?)
            }
        };
        Ok(data)
    }

    /// Activate the cache for all elements.
    pub fn enable_cache(&self) {
        for el in self.elems.iter() {
            if let Some(x) = el.lock().as_mut() {
                x.enable_cache();
            }
        }
    }

    /// Deactivate the cache for all elements.
    pub fn disable_cache(&self) {
        for el in self.elems.iter() {
            if let Some(x) = el.lock().as_mut() {
                x.disable_cache();
            }
        }
    }
}

pub struct StackedArrayElem<B: Backend>(pub(crate) Arc<InnerStackedArrayElem<B>>);

impl<B: Backend> Clone for StackedArrayElem<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B: Backend> std::fmt::Display for StackedArrayElem<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<B: Backend> Deref for StackedArrayElem<B> {
    type Target = InnerStackedArrayElem<B>;
    fn deref(&self) -> &InnerStackedArrayElem<B> {
        &self.0
    }
}

impl<B: Backend> StackedArrayElem<B> {
    pub fn empty() -> Self {
        Self(Arc::new(InnerStackedArrayElem {
            shape: None,
            elems: SmallVec::new(),
            index: std::iter::empty().collect(),
        }))
    }

    pub(crate) fn new(elems: SmallVec<[ArrayElem<B>; 96]>) -> Result<Self> {
        ensure!(
            elems
                .iter()
                .map(|x| x.lock().as_ref().map(|x| x.dtype()))
                .all_equal(),
            "all elements must have the same dtype"
        );

        let shapes: Vec<_> = elems
            .iter()
            .map(|x| x.lock().as_ref().map(|x| x.shape().clone()))
            .collect();
        ensure!(
            shapes
                .iter()
                .map(|x| x.as_ref().map(|s| &s.as_ref()[1..]))
                .all_equal(),
            "all elements must have the same shape except for the first axis"
        );
        let index: VecVecIndex = shapes.iter().flatten().map(|x| x.as_ref()[0]).collect();
        let shape = shapes.first().and_then(|x| {
            x.as_ref().map(|s| {
                let mut ss = s.clone();
                ss[0] = index.len();
                ss
            })
        });
        Ok(Self(Arc::new(InnerStackedArrayElem {
            shape,
            elems,
            index,
        })))
    }

    pub fn chunked<D>(&self, chunk_size: usize) -> StackedChunkedArrayElem<B, D>
    where
        D: TryFrom<ArrayData>,
    {
        StackedChunkedArrayElem::new(self.elems.iter().cloned(), chunk_size)
    }
}

/// Chunked Arrays
pub struct ChunkedArrayElem<B: Backend, D> {
    /// The underlying array element.
    elem: ArrayElem<B>,
    /// The chunk size.
    chunk_size: usize,
    num_items: usize,
    current_position: usize,
    phantom: std::marker::PhantomData<D>,
}

impl<B: Backend, D> ChunkedArrayElem<B, D> {
    pub fn new(elem: ArrayElem<B>, chunk_size: usize) -> Self {
        let num_items = elem.inner().shape()[0];
        Self {
            elem,
            chunk_size,
            num_items,
            current_position: 0,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<B, D> Iterator for ChunkedArrayElem<B, D>
where
    B: Backend,
    D: TryFrom<ArrayData>,
    <D as TryFrom<ArrayData>>::Error: std::fmt::Debug,
{
    type Item = (D, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_position >= self.num_items {
            if self.current_position == 0 {
                // return an empty array
                self.current_position = 1;
                Some((self.elem.inner().data().unwrap().try_into().unwrap(), 0, 0))
            } else {
                None
            }
        } else {
            let i = self.current_position;
            let j = std::cmp::min(self.num_items, self.current_position + self.chunk_size);
            self.current_position = j;
            let data = self
                .elem
                .inner()
                .select_axis(0, SelectInfoElem::from(i..j))
                .unwrap()
                .try_into()
                .unwrap();
            Some((data, i, j))
        }
    }
}

impl<B, D> ExactSizeIterator for ChunkedArrayElem<B, D>
where
    B: Backend,
    D: TryFrom<ArrayData>,
    <D as TryFrom<ArrayData>>::Error: std::fmt::Debug,
{
    fn len(&self) -> usize {
        let (n, remain) = div_rem(self.num_items, self.chunk_size);
        if remain == 0 { n } else { n + 1 }
    }
}

pub struct StackedChunkedArrayElem<B: Backend, D> {
    arrays: SmallVec<[ChunkedArrayElem<B, D>; 96]>,
    current_position: usize,
    current_array: usize,
}

impl<B: Backend, D> StackedChunkedArrayElem<B, D> {
    pub(crate) fn new<I: Iterator<Item = ArrayElem<B>>>(elems: I, chunk_size: usize) -> Self {
        Self {
            arrays: elems
                .map(|x| ChunkedArrayElem::new(x, chunk_size))
                .collect(),
            current_position: 0,
            current_array: 0,
        }
    }
}

impl<B, D> Iterator for StackedChunkedArrayElem<B, D>
where
    B: Backend,
    D: TryFrom<ArrayData>,
    <D as TryFrom<ArrayData>>::Error: std::fmt::Debug,
{
    type Item = (D, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(mat) = self.arrays.get_mut(self.current_array) {
            if let Some((data, start, stop)) = mat.next() {
                let new_start = self.current_position;
                let new_stop = new_start + stop - start;
                self.current_position = new_stop;
                Some((data, new_start, new_stop))
            } else {
                self.current_array += 1;
                self.next()
            }
        } else {
            if self.current_position == 0 {
                // return an empty array
                self.current_position = 1;
                Some((
                    self.arrays[0]
                        .elem
                        .inner()
                        .data()
                        .unwrap()
                        .try_into()
                        .unwrap(),
                    0,
                    0,
                ))
            } else {
                None
            }
        }
    }
}

impl<B, D> ExactSizeIterator for StackedChunkedArrayElem<B, D>
where
    B: Backend,
    D: TryFrom<ArrayData>,
    <D as TryFrom<ArrayData>>::Error: std::fmt::Debug,
{
    fn len(&self) -> usize {
        self.arrays.iter().map(|x| x.len()).sum()
    }
}

fn reverse_mapping(mapping: Vec<usize>) -> Vec<usize> {
    let mut res = vec![0; mapping.len()];
    for (i, x) in mapping.into_iter().enumerate() {
        res[x] = i;
    }
    res
}
