use crate::data::{
    PyArrayData, PyData, isinstance_of_arr, isinstance_of_csc, isinstance_of_csr,
    isinstance_of_polars, isinstance_of_pyanndata,
};

use anndata::backend::{DataType, ScalarType};
use anndata::data::{ArrayChunk, DataFrameIndex, SelectInfoElem, Shape, Stackable};
use anndata::{self, ArrayElemOp, ElemCollectionOp, Selectable};
use anndata::{AnnDataOp, ArrayData, AxisArraysOp, Backend, Data, HasShape};
use anyhow::{Result, bail};
use polars::prelude::DataFrame;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3_polars::PyDataFrame;
use std::collections::HashSet;
use std::ops::Deref;

pub struct PyAnnData<'py>(Bound<'py, PyAny>);

impl<'py> Deref for PyAnnData<'py> {
    type Target = Bound<'py, PyAny>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'py> FromPyObject<'_, 'py> for PyAnnData<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if isinstance_of_pyanndata(&obj)? {
            Ok(PyAnnData(obj.to_owned()))
        } else {
            Err(PyTypeError::new_err("Not a Python AnnData object"))
        }
    }
}

impl<'py> IntoPyObject<'py> for PyAnnData<'py> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, _: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

impl<'py> PyAnnData<'py> {
    pub fn new(py: Python<'py>) -> PyResult<Self> {
        PyModule::import(py, "anndata")?
            .call_method0("AnnData")?
            .extract()
    }

    pub fn from_anndata<B: Backend>(
        py: Python<'py>,
        inner: &anndata::AnnData<B>,
        partial: Option<HashSet<String>>,
    ) -> Result<Self> {
        let partial = partial.unwrap_or_default();
        let adata = PyAnnData::new(py)?;
        adata.set_n_obs(inner.n_obs())?;
        adata.set_n_vars(inner.n_vars())?;

        if (partial.is_empty() || partial.contains("X"))
            && let Some(x) = inner.x().get::<ArrayData>()?
        {
            adata.set_x(x)?;
        }

        if partial.is_empty() || partial.contains("obs") {
            adata.set_obs_names(inner.obs_names())?;
            adata.set_obs(inner.read_obs()?)?;
        }

        if partial.is_empty() || partial.contains("var") {
            adata.set_var_names(inner.var_names())?;
            adata.set_var(inner.read_var()?)?;
        }

        if partial.is_empty() || partial.contains("uns") {
            inner.uns().keys().into_iter().try_for_each(|k| {
                adata
                    .uns()
                    .add(&k, inner.uns().get_item::<Data>(&k)?.unwrap())
            })?;
        }

        if partial.is_empty() || partial.contains("obsm") {
            inner.obsm().keys().into_iter().try_for_each(|k| {
                adata
                    .obsm()
                    .add(&k, inner.obsm().get_item::<ArrayData>(&k)?.unwrap())
            })?;
        }

        if partial.is_empty() || partial.contains("obsp") {
            inner.obsp().keys().into_iter().try_for_each(|k| {
                adata
                    .obsp()
                    .add(&k, inner.obsp().get_item::<ArrayData>(&k)?.unwrap())
            })?;
        }

        if partial.is_empty() || partial.contains("varm") {
            inner.varm().keys().into_iter().try_for_each(|k| {
                adata
                    .varm()
                    .add(&k, inner.varm().get_item::<ArrayData>(&k)?.unwrap())
            })?;
        }

        if partial.is_empty() || partial.contains("varp") {
            inner.varp().keys().into_iter().try_for_each(|k| {
                adata
                    .varp()
                    .add(&k, inner.varp().get_item::<ArrayData>(&k)?.unwrap())
            })?;
        }

        if partial.is_empty() || partial.contains("layers") {
            inner.layers().keys().into_iter().try_for_each(|k| {
                adata
                    .layers()
                    .add(&k, inner.layers().get_item::<ArrayData>(&k)?.unwrap())
            })?;
        }

        Ok(adata)
    }

    pub fn get_x(&self) -> ArrayElem<'py> {
        self.x()
    }

    pub fn take_x_py(&self) -> Result<Option<PyArrayData>> {
        AnnDataOp::take_x::<ArrayData>(self).map(|x| x.map(Into::into))
    }

    pub fn set_x_py(&self, data: Option<PyArrayData>) -> Result<()> {
        if let Some(d) = data {
            self.set_x(d)?;
        } else {
            self.del_x()?;
        }
        Ok(())
    }
}

impl<'py> AnnDataOp for PyAnnData<'py> {
    type X = ArrayElem<'py>;
    type ElemCollectionRef<'a>
        = ElemCollection<'a>
    where
        Self: 'a;
    type AxisArraysRef<'a>
        = AxisArrays<'a>
    where
        Self: 'a;

    fn x(&self) -> Self::X {
        ArrayElem(self.0.getattr("X").unwrap())
    }

    fn take_x<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.x().take()
    }

    fn set_x_from_iter<I, D>(&self, iter: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: Into<ArrayData>,
    {
        let array = Stackable::vstack(iter.map(|x| x.into()))?;
        let shape = array.shape();
        self.set_n_obs(shape[0])?;
        self.set_n_vars(shape[1])?;
        self.setattr("X", PyArrayData::from(array))?;
        Ok(())
    }

    fn set_x<D: Into<ArrayData>>(&self, data: D) -> Result<()> {
        let data = data.into();
        let shape = data.shape();
        self.set_n_obs(shape[0])?;
        self.set_n_vars(shape[1])?;
        self.setattr("X", PyArrayData::from(data))?;
        Ok(())
    }

    fn del_x(&self) -> Result<()> {
        self.setattr("X", None::<Py<PyAny>>)?;
        Ok(())
    }

    fn n_obs(&self) -> usize {
        self.0.getattr("n_obs").unwrap().extract().unwrap()
    }
    fn n_vars(&self) -> usize {
        self.0.getattr("n_vars").unwrap().extract().unwrap()
    }

    fn set_n_obs(&self, n_obs: usize) -> Result<()> {
        let n = self.n_obs();
        if n == n_obs {
            Ok(())
        } else if n == 0 {
            if self.0.hasattr("_n_obs")? {
                self.0.setattr("_n_obs", n_obs)?;
            } else {
                let index = (0..n_obs).map(|x| x.to_string()).collect::<Vec<_>>();
                self.0.getattr("obs")?.setattr("index", index)?;
            }
            Ok(())
        } else {
            bail!("cannot set n_obs unless n_obs == 0")
        }
    }

    fn set_n_vars(&self, n_vars: usize) -> Result<()> {
        let n = self.n_vars();
        if n == n_vars {
            Ok(())
        } else if n == 0 {
            if self.0.hasattr("_n_vars")? {
                self.0.setattr("_n_vars", n_vars)?;
            } else {
                let index = (0..n_vars).map(|x| x.to_string()).collect::<Vec<_>>();
                self.0.getattr("var")?.setattr("index", index)?;
            }
            Ok(())
        } else {
            bail!("cannot set n_vars unless n_vars == 0")
        }
    }

    fn obs_names(&self) -> DataFrameIndex {
        self.0
            .getattr("obs_names")
            .unwrap()
            .extract::<Vec<String>>()
            .unwrap()
            .into()
    }
    fn var_names(&self) -> DataFrameIndex {
        self.0
            .getattr("var_names")
            .unwrap()
            .extract::<Vec<String>>()
            .unwrap()
            .into()
    }

    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        if self
            .getattr("obs")?
            .getattr("empty")?
            .cast()
            .unwrap()
            .is_true()
        {
            let py = self.py();
            let df = py.import("pandas")?.call_method(
                "DataFrame",
                (),
                Some(&[("index", index.into_vec())].into_py_dict(py)?),
            )?;
            self.setattr("obs", df)?;
        } else {
            self.setattr("obs_names", index.into_vec())?;
        }
        Ok(())
    }
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        if self
            .getattr("var")?
            .getattr("empty")?
            .cast()
            .unwrap()
            .is_true()
        {
            let py = self.py();
            let df = py.import("pandas")?.call_method(
                "DataFrame",
                (),
                Some(&[("index", index.into_vec())].into_py_dict(py)?),
            )?;
            self.setattr("var", df)?;
        } else {
            self.setattr("var_names", index.into_vec())?;
        }
        Ok(())
    }

    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, _names: I) -> Result<Vec<usize>> {
        todo!()
    }
    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, _names: I) -> Result<Vec<usize>> {
        todo!()
    }

    fn read_obs(&self) -> Result<DataFrame> {
        let df: PyDataFrame = self
            .py()
            .import("polars")?
            .call_method1("from_pandas", (self.0.getattr("obs")?,))?
            .extract()?;
        Ok(df.into())
    }
    fn read_var(&self) -> Result<DataFrame> {
        let df: PyDataFrame = self
            .py()
            .import("polars")?
            .call_method1("from_pandas", (self.0.getattr("var")?,))?
            .extract()?;
        Ok(df.into())
    }

    fn set_obs(&self, obs: DataFrame) -> Result<()> {
        let py = self.py();
        let is_empty = obs.height() == 0 || obs.width() == 0;
        let index = self.getattr("obs")?.getattr("index")?;
        let mut df = PyDataFrame(obs)
            .into_pyobject(py)?
            .call_method0("to_pandas")?;

        if is_empty {
            df = df.call_method1("reindex", (index,))?;
        } else {
            df = df.call_method1("set_index", (index,))?;
        }

        self.setattr("obs", df)?;
        Ok(())
    }

    fn set_var(&self, var: DataFrame) -> Result<()> {
        let py = self.py();
        let is_empty = var.height() == 0 || var.width() == 0;
        let index = self.getattr("var")?.getattr("index")?;
        let mut df = PyDataFrame(var)
            .into_pyobject(py)?
            .call_method0("to_pandas")?;

        if is_empty {
            df = df.call_method1("reindex", (index,))?;
        } else {
            df = df.call_method1("set_index", (index,))?;
        }

        self.setattr("var", df)?;
        Ok(())
    }

    fn del_obs(&self) -> Result<()> {
        self.0.setattr("obs", None::<Py<PyAny>>)?;
        Ok(())
    }

    fn del_var(&self) -> Result<()> {
        self.0.setattr("var", None::<Py<PyAny>>)?;
        Ok(())
    }

    fn uns(&self) -> Self::ElemCollectionRef<'_> {
        ElemCollection(self.getattr("uns").unwrap())
    }
    fn obsm(&self) -> Self::AxisArraysRef<'_> {
        AxisArrays {
            arrays: self.getattr("obsm").unwrap(),
            adata: self,
            axis: 0,
        }
    }
    fn obsp(&self) -> Self::AxisArraysRef<'_> {
        AxisArrays {
            arrays: self.getattr("obsp").unwrap(),
            adata: self,
            axis: 0,
        }
    }
    fn varm(&self) -> Self::AxisArraysRef<'_> {
        AxisArrays {
            arrays: self.getattr("varm").unwrap(),
            adata: self,
            axis: 1,
        }
    }
    fn varp(&self) -> Self::AxisArraysRef<'_> {
        AxisArrays {
            arrays: self.getattr("varp").unwrap(),
            adata: self,
            axis: 1,
        }
    }

    fn layers(&self) -> Self::AxisArraysRef<'_> {
        AxisArrays {
            arrays: self.getattr("layers").unwrap(),
            adata: self,
            axis: 2,
        }
    }

    fn del_uns(&self) -> Result<()> {
        self.0.setattr("uns", pyo3::types::PyDict::new(self.py()))?;
        Ok(())
    }

    fn del_obsm(&self) -> Result<()> {
        self.0.setattr("obsm", None::<Py<PyAny>>)?;
        Ok(())
    }

    fn del_obsp(&self) -> Result<()> {
        self.0.setattr("obsp", None::<Py<PyAny>>)?;
        Ok(())
    }

    fn del_varm(&self) -> Result<()> {
        self.0.setattr("varm", None::<Py<PyAny>>)?;
        Ok(())
    }

    fn del_varp(&self) -> Result<()> {
        self.0.setattr("varp", None::<Py<PyAny>>)?;
        Ok(())
    }

    fn del_layers(&self) -> Result<()> {
        self.0.setattr("layers", None::<Py<PyAny>>)?;
        Ok(())
    }
}

pub struct PyArrayIterator<D> {
    array: PyArrayData,
    chunk_size: usize,
    total_rows: usize,
    current_row: usize,
    phamtom: std::marker::PhantomData<D>,
}

impl<D> PyArrayIterator<D> {
    pub(crate) fn new(array: PyArrayData, chunk_size: usize) -> PyResult<Self> {
        let total_rows = array.shape()[0];
        Ok(Self {
            array,
            chunk_size,
            total_rows,
            current_row: 0,
            phamtom: std::marker::PhantomData,
        })
    }
}

impl<D> Iterator for PyArrayIterator<D>
where
    D: TryFrom<ArrayData>,
    <D as TryFrom<ArrayData>>::Error: std::fmt::Debug,
{
    type Item = (D, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.total_rows {
            None
        } else {
            let i = self.current_row;
            let j = std::cmp::min(self.total_rows, self.current_row + self.chunk_size);
            self.current_row = j;
            let slice = SelectInfoElem::from(i..j);
            let data = self.array.select_axis(0, slice);
            Some((data.try_into().unwrap(), i, j))
        }
    }
}

impl<D> ExactSizeIterator for PyArrayIterator<D>
where
    D: TryFrom<ArrayData>,
    <D as TryFrom<ArrayData>>::Error: std::fmt::Debug,
{
    fn len(&self) -> usize {
        let n = self.total_rows / self.chunk_size;
        if self.total_rows.is_multiple_of(self.chunk_size) {
            n
        } else {
            n + 1
        }
    }
}

pub struct ElemCollection<'a>(Bound<'a, PyAny>);

impl ElemCollectionOp for ElemCollection<'_> {
    fn keys(&self) -> Vec<String> {
        self.0
            .call_method0("keys")
            .unwrap()
            .try_iter()
            .unwrap()
            .map(|x| x.unwrap().extract().unwrap())
            .collect()
    }

    fn get_item<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: TryFrom<Data>,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>,
    {
        self.0
            .call_method1("__getitem__", (key,))
            .ok()
            .map(|x| {
                let data: Data = x.extract::<PyData>()?.into();
                data.try_into().map_err(Into::into)
            })
            .transpose()
    }

    fn add<D: Into<Data>>(&self, key: &str, data: D) -> Result<()> {
        let py = self.0.py();
        let d = PyData::from(data.into()).into_pyobject(py)?.into_any();
        let new_d = if isinstance_of_polars(&d)? {
            d.call_method0("to_pandas")?
        } else {
            d
        };
        self.0.call_method1("__setitem__", (key, new_d))?;
        Ok(())
    }

    fn remove(&self, key: &str) -> Result<()> {
        self.0.call_method1("__delitem__", (key,))?;
        Ok(())
    }
}

pub struct AxisArrays<'a> {
    arrays: Bound<'a, PyAny>,
    adata: &'a PyAnnData<'a>,
    axis: u8,
}

impl<'py> AxisArraysOp for AxisArrays<'py> {
    type ArrayElem = ArrayElem<'py>;

    fn keys(&self) -> Vec<String> {
        self.arrays
            .call_method0("keys")
            .unwrap()
            .try_iter()
            .unwrap()
            .map(|x| x.unwrap().extract().unwrap())
            .collect()
    }

    fn get(&self, key: &str) -> Option<Self::ArrayElem> {
        self.arrays
            .call_method1("__getitem__", (key,))
            .ok()
            .map(ArrayElem)
    }

    fn add<D: Into<ArrayData>>(&self, key: &str, data: D) -> Result<()> {
        let data = data.into();
        let py = self.arrays.py();
        let shape = data.shape();
        if self.axis == 0 {
            self.adata.set_n_obs(shape[0])?;
        } else if self.axis == 1 {
            self.adata.set_n_vars(shape[0])?;
        } else {
            self.adata.set_n_obs(shape[0])?;
            self.adata.set_n_vars(shape[1])?;
        }
        let d = PyArrayData::from(data).into_pyobject(py)?.into_any();
        let new_d = if isinstance_of_polars(&d)? {
            d.call_method0("to_pandas")?
        } else {
            d
        };
        self.arrays.call_method1("__setitem__", (key, new_d))?;
        Ok(())
    }

    fn add_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk + Into<ArrayData>,
    {
        let array = Stackable::vstack(data.map(|x| x.into()))?;
        let shape = array.shape();
        if self.axis == 0 {
            self.adata.set_n_obs(shape[0])?;
        } else if self.axis == 1 {
            self.adata.set_n_vars(shape[0])?;
        } else {
            self.adata.set_n_obs(shape[0])?;
            self.adata.set_n_vars(shape[1])?;
        }
        self.arrays
            .call_method1("__setitem__", (key, PyArrayData::from(array)))?;
        Ok(())
    }

    fn remove(&self, key: &str) -> Result<()> {
        self.arrays.call_method1("__delitem__", (key,))?;
        Ok(())
    }
}

pub struct ArrayElem<'a>(Bound<'a, PyAny>);

impl ArrayElemOp for ArrayElem<'_> {
    type ArrayIter<D>
        = PyArrayIterator<D>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: std::fmt::Debug;

    fn is_none(&self) -> bool {
        self.0.is_none()
    }

    fn is_cached(&self) -> bool {
        true
    }

    fn enable_cache(&self) {}
    fn disable_cache(&self) {}

    fn dtype(&self) -> Option<DataType> {
        if self.is_none() {
            return None;
        }

        let py_dtype = self.0.getattr("dtype").ok()?;
        let ty_name = py_dtype.getattr("name").ok()?.extract::<String>().ok()?;
        let scalar_type = match ty_name.as_str() {
            "int8" => ScalarType::I8,
            "int16" => ScalarType::I16,
            "int32" => ScalarType::I32,
            "int64" => ScalarType::I64,
            "uint8" => ScalarType::U8,
            "uint16" => ScalarType::U16,
            "uint32" => ScalarType::U32,
            "uint64" => ScalarType::U64,
            "float32" => ScalarType::F32,
            "float64" => ScalarType::F64,
            "bool" => ScalarType::Bool,
            _ => return None,
        };

        if isinstance_of_arr(&self.0).unwrap_or(false) {
            Some(DataType::Array(scalar_type))
        } else if isinstance_of_csr(&self.0).unwrap_or(false) {
            Some(DataType::CsrMatrix(scalar_type, ScalarType::U64))
        } else if isinstance_of_csc(&self.0).unwrap_or(false) {
            Some(DataType::CscMatrix(scalar_type, ScalarType::U64))
        } else {
            None
        }
    }

    fn shape(&self) -> Option<Shape> {
        let shape: Vec<usize> = self.0.getattr("shape").ok()?.extract().ok()?;
        Some(shape.into())
    }

    fn get<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let data: Option<ArrayData> = self.0.extract::<Option<PyArrayData>>()?.map(Into::into);
        data.map(|x| x.try_into().map_err(Into::into)).transpose()
    }

    fn take<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let res = self.get()?;
        if res.is_some() {
            // Since this is a reference to an element in an AnnData object,
            // we should probably clear it from the AnnData object if we want "take" semantics.
            // However, ArrayElem doesn't have a direct reference back to the container attribute name.
            // For now, we'll just return the data, which is already owned by the Rust side after get().
        }
        Ok(res)
    }

    fn slice<D, S>(&self, slice: S) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        if let Some(data) = self.get::<ArrayData>()? {
            data.select(slice.as_ref())
                .try_into()
                .map_err(Into::into)
                .map(Some)
        } else {
            Ok(None)
        }
    }

    fn iter<D>(&self, chunk_size: usize) -> Self::ArrayIter<D>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: std::fmt::Debug,
    {
        let array = self.0.extract::<PyArrayData>().unwrap();
        PyArrayIterator::new(array, chunk_size).unwrap()
    }
}
