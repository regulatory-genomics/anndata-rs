use crate::container::{
    PyArrayElem, PyAxisArrays, PyChunkedArray, PyDataFrameElem, PyElemCollection,
};
use crate::data::{PyArrayData, PyData, isinstance_of_pandas, to_select_elem};
use crate::{AnnData, PyAnnData};

use anndata::container::Slot;
use anndata::data::{ArrayData, DataFrameIndex, SelectInfoElem, SelectInfoElemBounds};
use anndata::{self, ArrayElemOp, Data, Selectable};
use anndata::{AnnDataOp, Backend};
use anndata::{AxisArraysOp, ElemCollectionOp};
use anndata_hdf5::H5;
use anyhow::{Result, bail};
use downcast_rs::{Downcast, impl_downcast};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;
use std::path::PathBuf;

use super::backed::StackedAnnData;
use super::get_backend;

/** Similar to `AnnData`, `AnnDataSet` contains annotations of
    observations `obs` (`obsm`, `obsp`), variables `var` (`varm`, `varp`),
    and unstructured annotations `uns`. Additionally it provides lazy access to
    concatenated component AnnData objects, including `X`, `obs`, `obsm`, `obsp`.

    Parameters
    ----------
    adatas: list[(str, Path)] | list[(str, AnnData)]
        List of key and file name (or backed AnnData object) pairs.
    filename: Path
        File name of the output file containing the AnnDataSet object.
    add_key: str
        The column name in obs to store the keys
    use_abolute_path: bool
        Whether to store absolute paths of the component anndata files.
    backend: Literal['hdf5']
        The backend to use for the AnnDataSet object.

    Note
    ----
    - AnnDataSet does not copy underlying AnnData objects. It stores the references
    to individual anndata files. If you move the anndata files to a new location,
    remember to update the anndata file locations when opening an AnnDataSet object.
    - AnnDataSet requires all component anndata files to have the same set of var names.
    To concatenate AnnData objects with different var names, please use `concat` function.

    See Also
    --------
    read_dataset
    concat
*/
#[pyclass]
#[repr(transparent)]
pub struct AnnDataSet(Box<dyn AnnDataSetTrait>);

impl Clone for AnnDataSet {
    fn clone(&self) -> Self {
        AnnDataSet(self.0.clone_ref())
    }
}

impl<B: Backend> From<anndata::AnnDataSet<B>> for AnnDataSet {
    fn from(adata: anndata::AnnDataSet<B>) -> Self {
        AnnDataSet(Box::new(Slot::new(adata)))
    }
}

impl AnnDataSet {
    pub fn take_inner<B: Backend>(&self) -> Option<anndata::AnnDataSet<B>> {
        self.0
            .downcast_ref::<Slot<anndata::AnnDataSet<B>>>()
            .expect("downcast to AnnDataSet failed")
            .extract()
    }

    pub fn inner_ref<B: Backend>(&self) -> anndata::container::Inner<'_, anndata::AnnDataSet<B>> {
        self.0
            .downcast_ref::<Slot<anndata::AnnDataSet<B>>>()
            .expect("downcast to AnnDataSet failed")
            .inner()
    }

    fn select_obs(&self, ix: &Bound<'_, PyAny>) -> PyResult<SelectInfoElem> {
        let from_iter = ix
            .try_iter()
            .and_then(|iter| {
                iter.map(|x| x.unwrap().extract::<String>())
                    .collect::<PyResult<Vec<_>>>()
            })
            .map(|names| {
                let index = self.0.obs_names();
                names
                    .into_iter()
                    .map(|name| {
                        index
                            .get_index(&name)
                            .unwrap_or_else(|| panic!("Unknown obs name: {name}"))
                    })
                    .collect::<Vec<_>>()
            });

        if let Ok(indices) = from_iter {
            Ok(indices.into())
        } else {
            let n = self.n_obs();
            to_select_elem(ix, n)
        }
    }

    fn select_var(&self, ix: &Bound<'_, PyAny>) -> PyResult<SelectInfoElem> {
        let from_iter = ix
            .try_iter()
            .and_then(|iter| {
                iter.map(|x| x.unwrap().extract::<String>())
                    .collect::<PyResult<Vec<_>>>()
            })
            .map(|names| {
                let index = self.0.var_names();
                names
                    .into_iter()
                    .map(|name| {
                        index
                            .get_index(&name)
                            .unwrap_or_else(|| panic!("Unknown obs name: {name}"))
                    })
                    .collect::<Vec<_>>()
            });

        if let Ok(indices) = from_iter {
            Ok(indices.into())
        } else {
            let n = self.n_vars();
            to_select_elem(ix, n)
        }
    }
}

#[derive(FromPyObject)]
pub enum AnnDataFile<'py> {
    Path(PathBuf),
    Data(Bound<'py, AnnData>),
}

#[pymethods]
impl AnnDataSet {
    #[new]
    #[pyo3(signature = (adatas, *, filename, add_key="sample", use_absolute_path=false, backend=None))]
    pub fn new(
        adatas: Vec<(String, AnnDataFile)>,
        filename: PathBuf,
        add_key: &str,
        use_absolute_path: bool,
        backend: Option<&str>,
    ) -> Result<Self> {
        let backend = get_backend(&filename, backend);
        match backend {
            H5::NAME => {
                let anndatas = adatas.into_iter().map(|(key, data_file)| {
                    let adata = match data_file {
                        AnnDataFile::Data(data) => data.borrow().take_inner::<H5>().unwrap(),
                        AnnDataFile::Path(path) => {
                            anndata::AnnData::open(H5::open(path).unwrap()).unwrap()
                        }
                    };
                    (key, adata)
                });
                Ok(
                    anndata::AnnDataSet::new(anndatas, filename, add_key, use_absolute_path)?
                        .into(),
                )
            }
            _ => todo!(),
        }
    }

    /// Shape of data matrix (`n_obs`, `n_vars`).
    ///
    /// Returns
    /// -------
    /// tuple[int, int]
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }

    /// Number of observations.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn n_obs(&self) -> usize {
        self.shape().0
    }

    /// Number of variables/features.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn n_vars(&self) -> usize {
        self.shape().1
    }

    /// Names of observations.
    ///
    /// Returns
    /// -------
    /// list[str]
    #[getter]
    pub fn obs_names(&self) -> Vec<String> {
        self.0.obs_names().into_vec()
    }
    #[setter(obs_names)]
    pub fn set_obs_names(&self, names: Bound<'_, PyAny>) -> Result<()> {
        self.0.set_obs_names(names)
    }

    #[pyo3(text_signature = "($self, names)")]
    fn obs_ix(&self, names: &Bound<'_, PyAny>) -> Result<Vec<usize>> {
        self.0.obs_ix(names)
    }

    /// Names of variables.
    ///
    /// Returns
    /// -------
    /// list[str]
    #[getter]
    pub fn var_names(&self) -> Vec<String> {
        self.0.var_names().into_vec()
    }
    #[setter(var_names)]
    pub fn set_var_names(&self, names: Bound<'_, PyAny>) -> Result<()> {
        self.0.set_var_names(names)
    }

    #[pyo3(text_signature = "($self, names)")]
    fn var_ix(&self, names: Bound<'_, PyAny>) -> Result<Vec<usize>> {
        self.0.var_ix(names)
    }

    /// Data matrix of shape n_obs × n_vars.
    ///
    /// Returns
    /// -------
    /// PyArrayElem
    #[getter(X)]
    pub fn get_x(&self) -> Option<PyArrayElem> {
        self.0.get_x()
    }

    /// Observation annotations.
    ///
    /// Returns
    /// -------
    /// PyDataFrameElem
    #[getter(obs)]
    fn get_obs(&self) -> Option<PyDataFrameElem> {
        self.0.get_obs()
    }
    #[setter(obs)]
    fn set_obs(&self, obs: Option<Bound<'_, PyAny>>) -> Result<()> {
        self.0.set_obs(obs)
    }

    /// Variable annotations.
    ///
    /// Returns
    /// -------
    /// PyDataFrameElem
    #[getter(var)]
    fn get_var(&self) -> Option<PyDataFrameElem> {
        self.0.get_var()
    }
    #[setter(var)]
    fn set_var(&self, var: Option<Bound<'_, PyAny>>) -> Result<()> {
        self.0.set_var(var)
    }

    /// Unstructured annotation (ordered dictionary).
    ///
    /// Returns
    /// -------
    /// PyElemCollection
    #[getter(uns)]
    pub fn get_uns(&self) -> Option<PyElemCollection> {
        self.0.get_uns()
    }
    #[setter(uns)]
    pub fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()> {
        self.0.set_uns(uns)
    }

    #[getter(obsm)]
    pub fn get_obsm(&self) -> Option<PyAxisArrays> {
        self.0.get_obsm()
    }
    #[setter(obsm)]
    pub fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.set_obsm(obsm)
    }

    #[getter(obsp)]
    pub fn get_obsp(&self) -> Option<PyAxisArrays> {
        self.0.get_obsp()
    }
    #[setter(obsp)]
    pub fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.set_obsp(obsp)
    }

    #[getter(varm)]
    pub fn get_varm(&self) -> Option<PyAxisArrays> {
        self.0.get_varm()
    }
    #[setter(varm)]
    pub fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.set_varm(varm)
    }

    #[getter(varp)]
    pub fn get_varp(&self) -> Option<PyAxisArrays> {
        self.0.get_varp()
    }
    #[setter(varp)]
    pub fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.set_varp(varp)
    }

    /// View into the component AnnData objects.
    ///
    /// Returns
    /// -------
    /// StackedAnnData
    #[getter(adatas)]
    pub fn adatas(&self) -> StackedAnnData {
        self.0.get_adatas()
    }

    /// Convert AnnDataSet to AnnData object.
    #[pyo3(
        signature = (obs_indices=None, var_indices=None, copy_x=true, file=None, backend=None),
        text_signature = "($self, obs_indices=None, var_indices=None, copy_x=True, file=None, backed=None)",
    )]
    pub fn to_adata<'py>(
        &self,
        py: Python<'py>,
        obs_indices: Option<&Bound<'_, PyAny>>,
        var_indices: Option<&Bound<'_, PyAny>>,
        copy_x: bool,
        file: Option<PathBuf>,
        backend: Option<&str>,
    ) -> Result<Bound<'py, PyAny>> {
        let i = obs_indices
            .map(|x| self.select_obs(x).unwrap())
            .unwrap_or(SelectInfoElem::full());
        let j = var_indices
            .map(|x| self.select_var(x).unwrap())
            .unwrap_or(SelectInfoElem::full());
        self.0.to_adata(py, &[i, j], copy_x, file, backend)
    }

    /// Split the AnnDataSet object into multiple AnnData objects based on grouping keys.
    /// The length of the grouping keys must match the number of observations.
    ///
    /// Parameters
    /// ----------
    /// key: str | list[str]
    ///     Grouping key(s). Can be a single column name or a list of column names in `obs`.
    ///     The grouping keys may contain `None` values, in which case the corresponding
    ///     observations are ignored.
    /// out_dir: Path
    ///     Output directory to save the split AnnData files.
    ///
    /// Returns
    /// -------
    /// dict[str, AnnData]
    ///     A dictionary mapping each unique key to its corresponding AnnData object.
    #[pyo3(
        signature = (key, out_dir="./".into()),
        text_signature = "($self, key, out_dir='./')",
    )]
    pub fn split_obs_by(
        &self,
        key: Bound<'_, PyAny>,
        out_dir: PathBuf,
    ) -> Result<HashMap<String, AnnData>> {
        self.0.split_obs_by(key, out_dir)
    }

    /// Parameters
    /// ----------
    /// chunk_size : int
    ///     Row size of a single chunk. Default: 500.
    #[pyo3(
        signature = (chunk_size=500, /),
        text_signature = "($self, chunk_size=500, /)",
        name = "chunked_X",
    )]
    pub fn chunked_x(&self, chunk_size: usize) -> PyChunkedArray {
        self.0.chunked_x(chunk_size)
    }

    /// Whether the AnnDataSet object is backed. This is always true.
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    pub fn isbacked(&self) -> bool {
        true
    }

    /// Close the AnnDataSet object.
    #[pyo3(text_signature = "($self)")]
    pub fn close(&self) -> Result<()> {
        self.0.close()
    }

    /// If the AnnDataSet object has been closed.
    ///
    /// Returns
    /// -------
    /// bool
    #[pyo3(text_signature = "($self)")]
    pub fn is_closed(&self) -> bool {
        self.0.is_closed()
    }

    #[getter]
    pub fn backend(&self) -> String {
        self.0.backend().to_string()
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/*
/// Lazily concatenated AnnData objects.
#[pyclass]
#[repr(transparent)]
pub struct StackedAnnData(pub Slot<anndata::StackedAnnData>);
*/

trait AnnDataSetTrait: Send + Sync + Downcast {
    fn shape(&self) -> (usize, usize);
    fn obs_names(&self) -> DataFrameIndex;
    fn set_obs_names(&self, names: Bound<'_, PyAny>) -> Result<()>;
    fn obs_ix(&self, index: &Bound<'_, PyAny>) -> Result<Vec<usize>>;
    fn var_names(&self) -> DataFrameIndex;
    fn set_var_names(&self, names: Bound<'_, PyAny>) -> Result<()>;
    fn var_ix(&self, index: Bound<'_, PyAny>) -> Result<Vec<usize>>;

    fn get_x(&self) -> Option<PyArrayElem>;
    fn get_obs(&self) -> Option<PyDataFrameElem>;
    fn get_var(&self) -> Option<PyDataFrameElem>;
    fn get_uns(&self) -> Option<PyElemCollection>;
    fn get_obsm(&self) -> Option<PyAxisArrays>;
    fn get_obsp(&self) -> Option<PyAxisArrays>;
    fn get_varm(&self) -> Option<PyAxisArrays>;
    fn get_varp(&self) -> Option<PyAxisArrays>;

    fn set_obs(&self, obs: Option<Bound<'_, PyAny>>) -> Result<()>;
    fn set_var(&self, var: Option<Bound<'_, PyAny>>) -> Result<()>;
    fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()>;
    fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()>;

    fn get_adatas(&self) -> StackedAnnData;

    fn to_adata<'py>(
        &self,
        py: Python<'py>,
        slice: &[SelectInfoElem],
        copy_x: bool,
        file: Option<PathBuf>,
        backend: Option<&str>,
    ) -> Result<Bound<'py, PyAny>>;

    fn split_obs_by(
        &self,
        key: Bound<'_, PyAny>,
        out_dir: PathBuf,
    ) -> Result<HashMap<String, AnnData>>;

    fn chunked_x(&self, chunk_size: usize) -> PyChunkedArray;

    fn backend(&self) -> &str;
    fn is_closed(&self) -> bool;
    fn show(&self) -> String;

    fn close(&self) -> Result<()>;
    fn clone_ref(&self) -> Box<dyn AnnDataSetTrait>;
}
impl_downcast!(AnnDataSetTrait);

impl<B: Backend> AnnDataSetTrait for Slot<anndata::AnnDataSet<B>> {
    fn shape(&self) -> (usize, usize) {
        let inner = self.inner();
        (inner.n_obs(), inner.n_vars())
    }

    fn obs_names(&self) -> DataFrameIndex {
        self.inner().obs_names()
    }

    fn set_obs_names(&self, names: Bound<'_, PyAny>) -> Result<()> {
        let obs_names: Result<DataFrameIndex> = names
            .try_iter()?
            .map(|x| Ok(x?.extract::<String>()?))
            .collect();
        self.inner().set_obs_names(obs_names?)
    }

    fn obs_ix(&self, index: &Bound<'_, PyAny>) -> Result<Vec<usize>> {
        let bounds: Vec<_> = index.try_iter()?.map(|x| x.unwrap()).collect();
        self.inner()
            .obs_ix(bounds.iter().map(|x| x.extract::<&str>().unwrap()))
    }

    fn var_names(&self) -> DataFrameIndex {
        self.inner().var_names()
    }

    fn set_var_names(&self, names: Bound<'_, PyAny>) -> Result<()> {
        let var_names: Result<DataFrameIndex> = names
            .try_iter()?
            .map(|x| Ok(x?.extract::<String>()?))
            .collect();
        self.inner().set_var_names(var_names?)
    }

    fn var_ix(&self, index: Bound<'_, PyAny>) -> Result<Vec<usize>> {
        let bounds: Vec<_> = index.try_iter()?.map(|x| x.unwrap()).collect();
        self.inner()
            .var_ix(bounds.iter().map(|x| x.extract::<&str>().unwrap()))
    }

    fn get_x(&self) -> Option<PyArrayElem> {
        Some(self.inner().x().into())
    }
    fn get_obs(&self) -> Option<PyDataFrameElem> {
        let inner = self.inner();
        let obs = inner.get_anno().get_obs();
        if obs.is_none() {
            None
        } else {
            Some(obs.clone().into())
        }
    }
    fn get_var(&self) -> Option<PyDataFrameElem> {
        let inner = self.inner();
        let var = inner.get_anno().get_var();
        if var.is_none() {
            None
        } else {
            Some(var.clone().into())
        }
    }
    fn get_uns(&self) -> Option<PyElemCollection> {
        let inner = self.inner();
        let uns = inner.get_anno().uns();
        if uns.is_none() {
            None
        } else {
            Some(uns.clone().into())
        }
    }
    fn get_obsm(&self) -> Option<PyAxisArrays> {
        let inner = self.inner();
        let obsm = inner.get_anno().obsm();
        if obsm.is_none() {
            None
        } else {
            Some(obsm.clone().into())
        }
    }
    fn get_obsp(&self) -> Option<PyAxisArrays> {
        let inner = self.inner();
        let obsp = inner.get_anno().obsp();
        if obsp.is_none() {
            None
        } else {
            Some(obsp.clone().into())
        }
    }
    fn get_varm(&self) -> Option<PyAxisArrays> {
        let inner = self.inner();
        let varm = inner.get_anno().varm();
        if varm.is_none() {
            None
        } else {
            Some(varm.clone().into())
        }
    }
    fn get_varp(&self) -> Option<PyAxisArrays> {
        let inner = self.inner();
        let varp = inner.get_anno().varp();
        if varp.is_none() {
            None
        } else {
            Some(varp.clone().into())
        }
    }

    fn set_obs(&self, obs: Option<Bound<'_, PyAny>>) -> Result<()> {
        let inner = self.inner();
        if let Some(x) = obs {
            let py = x.py();
            let ob = if isinstance_of_pandas(&x)? {
                py.import("polars")?.call_method1("from_pandas", (x,))?
            } else if x.is_instance_of::<pyo3::types::PyDict>() {
                py.import("polars")?.call_method1("from_dict", (x,))?
            } else {
                x
            };
            inner.set_obs(ob.extract::<PyDataFrame>()?.0)?;
        } else {
            inner.del_obs()?;
        }
        Ok(())
    }
    fn set_var(&self, var: Option<Bound<'_, PyAny>>) -> Result<()> {
        let inner = self.inner();
        if let Some(x) = var {
            let py = x.py();
            let ob = if isinstance_of_pandas(&x)? {
                py.import("polars")?.call_method1("from_pandas", (x,))?
            } else if x.is_instance_of::<pyo3::types::PyDict>() {
                py.import("polars")?.call_method1("from_dict", (x,))?
            } else {
                x
            };
            inner.set_var(ob.extract::<PyDataFrame>()?.0)?;
        } else {
            inner.del_var()?;
        }
        Ok(())
    }
    fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()> {
        let inner = self.inner();
        if let Some(u) = uns {
            inner.set_uns(u)?;
        } else {
            inner.del_uns()?;
        }
        Ok(())
    }
    fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.inner();
        if let Some(o) = obsm {
            inner.set_obsm(o)?;
        } else {
            inner.del_obsm()?;
        }
        Ok(())
    }
    fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.inner();
        if let Some(o) = obsp {
            inner.set_obsp(o)?;
        } else {
            inner.del_obsp()?;
        }
        Ok(())
    }
    fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.inner();
        if let Some(v) = varm {
            inner.set_varm(v)?;
        } else {
            inner.del_varm()?;
        }
        Ok(())
    }
    fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.inner();
        if let Some(v) = varp {
            inner.set_varp(v)?;
        } else {
            inner.del_varp()?;
        }
        Ok(())
    }

    fn get_adatas(&self) -> StackedAnnData {
        self.inner().adatas().clone().into()
    }

    fn to_adata<'py>(
        &self,
        py: Python<'py>,
        slice: &[SelectInfoElem],
        copy_x: bool,
        file: Option<PathBuf>,
        backend: Option<&str>,
    ) -> Result<Bound<'py, PyAny>> {
        let inner = self.inner();
        if let Some(file) = file {
            let backend = get_backend(&file, backend);
            match backend {
                H5::NAME => {
                    let adata = inner.to_adata_select::<H5, _, _>(slice, file, copy_x)?;
                    Ok(AnnData::from(adata).into_pyobject(py)?.into_any())
                }
                x => bail!("Unsupported backend: {x}"),
            }
        } else {
            let adata = PyAnnData::new(py)?;
            let obs_slice = SelectInfoElemBounds::new(&slice[0], inner.n_obs());
            let var_slice = SelectInfoElemBounds::new(&slice[1], inner.n_vars());
            let n_obs = obs_slice.len();
            let n_vars = var_slice.len();
            adata.set_n_obs(n_obs)?;
            adata.set_n_vars(n_vars)?;

            if copy_x {
                // Set X
                if let Some(x) = inner.x().slice::<ArrayData, _>(slice)? {
                    adata.set_x(x)?;
                }
            }
            {
                // Set obs and var
                adata.set_obs_names(inner.obs_names().select(&slice[0]))?;
                adata.set_obs(inner.read_obs()?.select_axis(0, &slice[0]))?;
                adata.set_var_names(inner.var_names().select(&slice[1]))?;
                adata.set_var(inner.read_var()?.select_axis(0, &slice[1]))?;
            }
            {
                // Set uns
                inner.uns().keys().into_iter().try_for_each(|k| {
                    adata
                        .uns()
                        .add(&k, inner.uns().get_item::<Data>(&k)?.unwrap())
                })?;
            }
            {
                // Set obsm
                inner.obsm().keys().into_iter().try_for_each(|k| {
                    adata.obsm().add(
                        &k,
                        inner
                            .obsm()
                            .get(&k)
                            .unwrap()
                            .slice_axis::<ArrayData, _>(0, &slice[0])?
                            .unwrap(),
                    )
                })?;
            }
            {
                // Set obsp
                inner.obsp().keys().into_iter().try_for_each(|k| {
                    let elem = inner.obsp().get(&k).unwrap();
                    let n = elem.shape().unwrap().ndim();
                    let mut select = vec![SelectInfoElem::full(); n];
                    select[0] = slice[0].clone();
                    select[1] = slice[0].clone();
                    let data = elem.slice::<ArrayData, _>(select)?.unwrap();
                    adata.obsp().add(&k, data)
                })?;
            }
            {
                // Set varm
                inner.varm().keys().into_iter().try_for_each(|k| {
                    adata.varm().add(
                        &k,
                        inner
                            .varm()
                            .get(&k)
                            .unwrap()
                            .slice_axis::<ArrayData, _>(0, &slice[1])?
                            .unwrap(),
                    )
                })?;
            }
            {
                // Set varp
                inner.varp().keys().into_iter().try_for_each(|k| {
                    let elem = inner.varp().get(&k).unwrap();
                    let n = elem.shape().unwrap().ndim();
                    let mut select = vec![SelectInfoElem::full(); n];
                    select[0] = slice[1].clone();
                    select[1] = slice[1].clone();
                    let data = elem.slice::<ArrayData, _>(select)?.unwrap();
                    adata.varp().add(&k, data)
                })?;
            }
            Ok(adata.into_pyobject(py)?)
        }
    }

    fn split_obs_by(
        &self,
        key: Bound<'_, PyAny>,
        out_dir: PathBuf,
    ) -> Result<HashMap<String, AnnData>> {
        let inner = self.inner();

        let keys = if let Ok(key) = key.extract::<String>() {
            let obs = inner.read_obs()?;
            obs.column(&key)?
                .str()?
                .into_iter()
                .map(|x| x.map(|s| s.to_string()))
                .collect()
        } else {
            key.extract::<Vec<Option<String>>>()?
        };

        let split_data = inner.split_obs_by::<H5, _>(&keys, &out_dir)?;
        split_data
            .into_iter()
            .map(|(k, v)| Ok((k, AnnData::from(v))))
            .collect()
    }

    fn chunked_x(&self, chunk_size: usize) -> PyChunkedArray {
        self.inner().x().chunked(chunk_size).into()
    }

    fn backend(&self) -> &str {
        B::NAME
    }

    fn is_closed(&self) -> bool {
        self.is_none()
    }

    fn show(&self) -> String {
        if self.is_none() {
            "Closed AnnDataSet object".to_string()
        } else {
            format!("{self}")
        }
    }

    fn close(&self) -> Result<()> {
        if let Some(inner) = self.extract() {
            inner.close()?;
        }
        Ok(())
    }

    fn clone_ref(&self) -> Box<dyn AnnDataSetTrait> {
        Box::new(self.clone())
    }
}
