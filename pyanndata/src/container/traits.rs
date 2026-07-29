use std::ops::Deref;

use crate::data::{PyArrayData, PyData, is_none_slice, to_select_info};

use anndata::backend::DataType;
use anndata::container::{ChunkedArrayElem, StackedChunkedArrayElem};
use anndata::data::SelectInfoElem;
use anndata::{
    ArrayData, ArrayElem, AxisArrays, Backend, DataFrameElem, Elem, ElemCollection,
    StackedArrayElem, StackedAxisArrays, StackedDataFrame,
};
use anyhow::{Context, Result, bail};
use polars::series::Series;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};
use rand::Rng;
use rand::SeedableRng;

use super::{PyArrayElem, PyChunkedArray, PyElem};

/// Trait for `Elem` to abtract over different backends.
pub trait ElemTrait: Send + Sync {
    fn enable_cache(&self);
    fn disable_cache(&self);
    fn is_scalar(&self) -> bool;
    fn get<'py>(&self, subscript: &Bound<'py, PyAny>) -> Result<PyData>;
    fn show(&self) -> String;
}

impl<B: Backend> ElemTrait for Elem<B> {
    fn enable_cache(&self) {
        if let Some(x) = self.lock().as_mut() {
            x.enable_cache();
        }
    }

    fn disable_cache(&self) {
        if let Some(x) = self.lock().as_mut() {
            x.disable_cache();
        }
    }

    fn is_scalar(&self) -> bool {
        matches!(self.inner().dtype(), DataType::Scalar(_))
    }

    fn get<'py>(&self, slice: &Bound<'py, PyAny>) -> Result<PyData> {
        if is_none_slice(slice)? {
            Ok(self.inner().data()?.into())
        } else {
            bail!("Please use None slice to retrieve data.")
        }
    }

    fn show(&self) -> String {
        format!("{self}")
    }
}

pub trait ArrayElemTrait: Send + Sync {
    fn enable_cache(&self);
    fn disable_cache(&self);
    fn show(&self) -> String;
    fn get(&self, subscript: &Bound<'_, PyAny>) -> Result<PyArrayData>;
    fn take(&self) -> Result<PyArrayData>;
    fn shape(&self) -> Vec<usize>;
    fn chunk(&self, size: usize, replace: bool, seed: u64) -> Result<ArrayData>;
    fn chunked(&self, chunk_size: usize) -> PyChunkedArray;
}

impl<B: Backend + 'static> ArrayElemTrait for ArrayElem<B> {
    fn enable_cache(&self) {
        if let Some(x) = self.lock().as_mut() {
            x.enable_cache();
        }
    }

    fn disable_cache(&self) {
        if let Some(x) = self.lock().as_mut() {
            x.disable_cache();
        }
    }

    fn get(&self, subscript: &Bound<'_, PyAny>) -> Result<PyArrayData> {
        let slice = to_select_info(subscript, self.inner().shape())?;
        self.inner().select::<_>(slice.as_ref()).map(|x| x.into())
    }

    fn take(&self) -> Result<PyArrayData> {
        self.inner().take().map(|x| x.into())
    }

    fn show(&self) -> String {
        format!("{self}")
    }

    fn shape(&self) -> Vec<usize> {
        self.inner().shape().as_ref().to_vec()
    }

    fn chunk(&self, size: usize, replace: bool, seed: u64) -> Result<ArrayData> {
        let length = self.shape()[0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let idx: Vec<usize> = if replace {
            std::iter::repeat_with(|| rng.random_range(0..length))
                .take(size)
                .collect()
        } else {
            rand::seq::index::sample(&mut rng, length, size).into_vec()
        };
        self.inner().select_axis::<_>(0, &SelectInfoElem::from(idx))
    }

    fn chunked(&self, chunk_size: usize) -> PyChunkedArray {
        self.chunked(chunk_size).into()
    }
}

impl<B: Backend + 'static> ArrayElemTrait for StackedArrayElem<B> {
    fn enable_cache(&self) {
        self.deref().enable_cache();
    }

    fn disable_cache(&self) {
        self.deref().disable_cache();
    }

    fn get(&self, subscript: &Bound<'_, PyAny>) -> Result<PyArrayData> {
        let slice = to_select_info(subscript, self.deref().shape().as_ref().unwrap())?;
        self.select::<ArrayData, _>(slice.as_ref())
            .map(|x| x.unwrap().into())
    }

    fn take(&self) -> Result<PyArrayData> {
        self.deref()
            .data::<ArrayData>()?
            .context("StackedArrayElem is empty")
            .map(|x| x.into())
    }

    fn show(&self) -> String {
        format!("{self}")
    }

    fn shape(&self) -> Vec<usize> {
        self.deref().shape().as_ref().unwrap().as_ref().to_vec()
    }

    fn chunk(&self, size: usize, replace: bool, seed: u64) -> Result<ArrayData> {
        let length = self.shape()[0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let idx: Vec<usize> = if replace {
            std::iter::repeat_with(|| rng.random_range(0..length))
                .take(size)
                .collect()
        } else {
            rand::seq::index::sample(&mut rng, length, size).into_vec()
        };
        self.select_axis::<ArrayData, _>(0, &SelectInfoElem::from(idx))
            .map(|x| x.unwrap())
    }

    fn chunked(&self, chunk_size: usize) -> PyChunkedArray {
        self.chunked(chunk_size).into()
    }
}

pub trait DataFrameElemTrait: Send + Sync {
    fn get<'py>(&self, subscript: &Bound<'py, PyAny>) -> Result<Bound<'py, PyAny>>;
    fn set(&self, key: &str, data: Series) -> Result<()>;
    fn contains(&self, key: &str) -> bool;
    fn show(&self) -> String;
}

impl<B: Backend> DataFrameElemTrait for DataFrameElem<B> {
    fn get<'py>(&self, subscript: &Bound<'py, PyAny>) -> Result<Bound<'py, PyAny>> {
        let py = subscript.py();
        if let Ok(key) = subscript.extract::<&str>() {
            Ok(
                PySeries(self.inner().column(key)?.clone().take_materialized_series())
                    .into_pyobject(py)?,
            )
        } else {
            let width = self.inner().width();
            let height = self.inner().height();
            let shape = [width, height].as_slice().into();
            let slice = to_select_info(subscript, &shape)?;
            let df = self.inner().select(slice.as_ref())?;
            Ok(PyDataFrame(df).into_pyobject(py)?)
        }
    }

    fn set(&self, key: &str, mut data: Series) -> Result<()> {
        data.rename(key.into());
        self.inner().set_column(key, data)
    }

    fn contains(&self, key: &str) -> bool {
        self.lock()
            .as_ref()
            .map(|x| x.get_column_names().contains(key))
            .unwrap_or(false)
    }

    fn show(&self) -> String {
        format!("{self}")
    }
}

impl<B: Backend> DataFrameElemTrait for StackedDataFrame<B> {
    fn get<'py>(&self, subscript: &Bound<'py, PyAny>) -> Result<Bound<'py, PyAny>> {
        let py = subscript.py();
        if let Ok(key) = subscript.extract::<&str>() {
            Ok(PySeries(self.column(key)?.clone().take_materialized_series()).into_pyobject(py)?)
        } else {
            let width = self.width();
            let height = self.height();
            let shape = [width, height].as_slice().into();
            let slice = to_select_info(subscript, &shape)?;
            let df = self.select(slice.as_ref())?;
            Ok(PyDataFrame(df).into_pyobject(py)?)
        }
    }

    fn set(&self, _: &str, _: Series) -> Result<()> {
        bail!("Cannot set column in stacked dataframe")
    }

    fn contains(&self, key: &str) -> bool {
        self.get_column_names().contains(key)
    }

    fn show(&self) -> String {
        format!("{self}")
    }
}

pub trait AxisArrayTrait: Send + Sync {
    fn keys(&self) -> Vec<String>;
    fn contains(&self, key: &str) -> bool;
    fn get(&self, key: &str) -> Result<PyArrayData>;
    fn el(&self, key: &str) -> Result<PyArrayElem>;
    fn set(&self, key: &str, data: PyArrayData) -> Result<()>;
    fn show(&self) -> String;
}

impl<B: Backend + 'static> AxisArrayTrait for AxisArrays<B> {
    fn keys(&self) -> Vec<String> {
        self.inner().keys().map(|x| x.to_string()).collect()
    }

    fn contains(&self, key: &str) -> bool {
        self.inner().contains_key(key)
    }

    fn get(&self, key: &str) -> Result<PyArrayData> {
        Ok(self
            .inner()
            .get(key)
            .context(format!("No such key: {key}"))?
            .inner()
            .data()?
            .into())
    }

    fn el(&self, key: &str) -> Result<PyArrayElem> {
        Ok(self
            .inner()
            .get(key)
            .context(format!("No such key: {key}"))?
            .clone()
            .into())
    }

    fn set(&self, key: &str, data: PyArrayData) -> Result<()> {
        self.inner().add_data::<ArrayData>(key, data.into())
    }

    fn show(&self) -> String {
        format!("{self}")
    }
}

impl<B: Backend + 'static> AxisArrayTrait for StackedAxisArrays<B> {
    fn keys(&self) -> Vec<String> {
        self.deref().keys().map(|x| x.to_string()).collect()
    }

    fn contains(&self, key: &str) -> bool {
        self.deref().contains_key(key)
    }

    fn get(&self, key: &str) -> Result<PyArrayData> {
        Ok(self
            .deref()
            .get(key)
            .context(format!("No such key: {key}"))?
            .data::<ArrayData>()?
            .unwrap()
            .into())
    }

    fn el(&self, key: &str) -> Result<PyArrayElem> {
        Ok(self
            .deref()
            .get(key)
            .context(format!("No such key: {key}"))?
            .clone()
            .into())
    }

    fn set(&self, _: &str, _: PyArrayData) -> Result<()> {
        bail!("mutations are not allowed on stacked axis arrays")
    }

    fn show(&self) -> String {
        format!("{self}")
    }
}

pub trait ElemCollectionTrait: Send + Sync {
    fn keys(&self) -> Vec<String>;
    fn contains(&self, key: &str) -> bool;
    fn get(&self, key: &str) -> Result<PyData>;
    fn el(&self, key: &str) -> Result<PyElem>;
    fn set(&self, key: &str, data: PyData) -> Result<()>;
    fn show(&self) -> String;
}

impl<B: Backend + 'static> ElemCollectionTrait for ElemCollection<B> {
    fn keys(&self) -> Vec<String> {
        self.inner().keys().map(|x| x.to_string()).collect()
    }

    fn contains(&self, key: &str) -> bool {
        self.inner().contains_key(key)
    }

    fn get(&self, key: &str) -> Result<PyData> {
        Ok(self
            .inner()
            .get(key)
            .context(format!("No such key: {key}"))?
            .inner()
            .data()?
            .into())
    }

    fn el(&self, key: &str) -> Result<PyElem> {
        Ok(self
            .inner()
            .get(key)
            .context(format!("No such key: {key}"))?
            .clone()
            .into())
    }

    fn set(&self, key: &str, data: PyData) -> Result<()> {
        self.inner().add_data(key, data.into())
    }

    fn show(&self) -> String {
        format!("{self}")
    }
}

pub trait ChunkedArrayTrait:
    ExactSizeIterator<Item = (ArrayData, usize, usize)> + Send + Sync
{
}

impl<B: Backend> ChunkedArrayTrait for ChunkedArrayElem<B, ArrayData> {}
impl<B: Backend> ChunkedArrayTrait for StackedChunkedArrayElem<B, ArrayData> {}
