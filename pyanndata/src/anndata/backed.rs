use crate::anndata::PyAnnData;
use crate::container::{
    PyArrayElem, PyAxisArrays, PyChunkedArray, PyDataFrameElem, PyElemCollection,
};
use crate::data::{PyArrayData, PyData, isinstance_of_pandas, to_select_elem};

use anndata::container::Slot;
use anndata::data::{DataFrameIndex, SelectInfoElem, SelectInfoElemBounds};
use anndata::{self, ArrayElemOp, AxisArraysOp, Data, ElemCollectionOp, Selectable};
use anndata::{AnnDataOp, ArrayData, Backend};
use anndata_hdf5::H5;
use anyhow::{Result, bail};
use downcast_rs::{Downcast, impl_downcast};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::path::PathBuf;

use super::get_backend;

/** An annotated data matrix.

    `AnnData` stores a data matrix `X` together with annotations of
    observations `obs` (`obsm`, `obsp`), variables `var` (`varm`, `varp`),
    and unstructured annotations `uns`.
    `AnnData` is stored as a HDF5 file. Opening/creating an AnnData object
    does not read data from the HDF5 file. Data is copied to memory only when
    individual element is requested (or when cache is turned on).

    Parameters
    ----------
    X
        A #observations × #variables data matrix. A view of the data is used if the
        data type matches, otherwise, a copy is made.
    obs
        Key-indexed one-dimensional observations annotation of length #observations.
    var
        Key-indexed one-dimensional variables annotation of length #variables.
    uns
        Key-indexed unstructured annotation.
    obsm
        Key-indexed multi-dimensional observations annotation of length #observations.
        If passing a :class:`~numpy.ndarray`, it needs to have a structured datatype.
    varm
        Key-indexed multi-dimensional variables annotation of length #variables.
        If passing a :class:`~numpy.ndarray`, it needs to have a structured datatype.
    filename
        Name of backing file.
    backend
        The backend to use. The Python bindings currently support "hdf5".

    Note
    ----
    This is a backed AnnData. You can use :func:`~AnnData.to_memory` to convert
    it to an in-memory AnnData object. See
    `here <https://anndata.readthedocs.io/en/latest/index.html>`_
    for the documentation of in-memory AnnData objects.

    See Also
    --------
    read
    read_mtx
    read_csv
*/
#[pyclass]
#[repr(transparent)]
pub struct AnnData(Box<dyn AnnDataTrait>);

impl Clone for AnnData {
    fn clone(&self) -> Self {
        AnnData(self.0.clone_ref())
    }
}

impl AnnData {
    pub fn take_inner<B: Backend>(&self) -> Option<anndata::AnnData<B>> {
        self.0
            .downcast_ref::<InnerAnnData<B>>()
            .expect("downcast to anndata failed")
            .adata
            .extract()
    }

    pub fn inner_ref<B: Backend>(&self) -> anndata::container::Inner<'_, anndata::AnnData<B>> {
        self.0
            .downcast_ref::<InnerAnnData<B>>()
            .expect("downcast to anndata failed")
            .adata
            .inner()
    }

    pub fn new_from(filename: PathBuf, mode: &str, backend: &str) -> Result<Self> {
        match backend {
            H5::NAME => {
                let file = match mode {
                    "r" => H5::open(filename)?,
                    "r+" => H5::open_rw(filename)?,
                    _ => bail!("Unknown mode: {mode}"),
                };
                anndata::AnnData::<H5>::open(file).map(|adata| adata.into())
            }
            x => bail!("Unknown backend: {x}"),
        }
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
                            .unwrap_or_else(|| panic!("Unknown var name: {name}"))
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

impl<B: Backend> From<anndata::AnnData<B>> for AnnData {
    fn from(adata: anndata::AnnData<B>) -> Self {
        let inner = InnerAnnData {
            filename: adata.filename(),
            adata: Slot::new(adata),
        };
        AnnData(Box::new(inner))
    }
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl AnnData {
    #[new]
    #[pyo3(
        signature = (
            *, filename, X=None, obs=None, var=None, obsm=None,
            varm=None, uns=None, backend=None,
        ),
        text_signature = "($self, *, filename, X=None, obs=None, var=None, obsm=None,
            varm=None, uns=None, backend=None)"
    )]
    pub fn new(
        filename: PathBuf,
        #[allow(non_snake_case)] X: Option<PyArrayData>,
        obs: Option<Bound<'_, PyAny>>,
        var: Option<Bound<'_, PyAny>>,
        obsm: Option<HashMap<String, PyArrayData>>,
        varm: Option<HashMap<String, PyArrayData>>,
        uns: Option<HashMap<String, PyData>>,
        backend: Option<&str>,
    ) -> Result<Self> {
        let backend = get_backend(&filename, backend);
        let adata: AnnData = match backend {
            H5::NAME => anndata::AnnData::<H5>::new(filename)?.into(),
            backend => bail!("Unknown backend: {backend}"),
        };

        if X.is_some() {
            adata.set_x(X)?;
        }
        if obs.is_some() {
            adata.set_obs(obs)?;
        }
        if var.is_some() {
            adata.set_var(var)?;
        }
        if obsm.is_some() {
            adata.set_obsm(obsm)?;
        }
        if varm.is_some() {
            adata.set_varm(varm)?;
        }
        if uns.is_some() {
            adata.set_uns(uns)?;
        }
        Ok(adata)
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
    fn obs_ix(&self, names: Bound<'_, PyAny>) -> Result<Vec<usize>> {
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

    #[pyo3(text_signature = "($self)")]
    pub fn take_x(&self) -> Result<Option<PyArrayData>> {
        self.0.take_x()
    }

    #[setter(X)]
    pub fn set_x(&self, data: Option<PyArrayData>) -> Result<()> {
        self.0.set_x(data)
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

    #[getter(layers)]
    pub fn get_layers(&self) -> Option<PyAxisArrays> {
        self.0.get_layers()
    }
    #[setter(layers)]
    pub fn set_layers(&self, layers: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.set_layers(layers)
    }

    /// Subsetting the AnnData object.
    ///
    /// Parameters
    /// ----------
    /// obs_indices
    ///     obs indices
    /// var_indices
    ///     var indices
    /// out: Path | None
    ///     File name of the output `.h5ad` file. When `inplace=False`,
    ///     the result is written to this file. If `None`, an in-memory AnnData
    ///     is returned. This parameter is ignored when `inplace=True`.
    /// inplace: bool
    ///     Whether to modify the AnnData object in place or return a new AnnData object.
    /// backend: str | None
    ///     The backend to use. The Python bindings currently support "hdf5".
    ///
    /// Returns
    /// -------
    /// Optional[AnnData]
    #[pyo3(
        signature = (obs_indices=None, var_indices=None, *, out=None, inplace=true, backend=None),
        text_signature = "($self, obs_indices=None, var_indices=None, *, out=None, inplace=True, backend=None)",
    )]
    pub fn subset<'py>(
        &self,
        py: Python<'py>,
        obs_indices: Option<&Bound<'_, PyAny>>,
        var_indices: Option<&Bound<'_, PyAny>>,
        out: Option<PathBuf>,
        inplace: bool,
        backend: Option<&str>,
    ) -> Result<Option<Bound<'py, PyAny>>> {
        let i = obs_indices
            .map(|x| self.select_obs(x).unwrap())
            .unwrap_or(SelectInfoElem::full());
        let j = var_indices
            .map(|x| self.select_var(x).unwrap())
            .unwrap_or(SelectInfoElem::full());
        self.0.subset(py, &[i, j], out, inplace, backend)
    }

    /// Split the AnnData object into multiple AnnData objects based on grouping keys.
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
    ) -> Result<HashMap<String, Self>> {
        self.0.split_obs_by(key, out_dir)
    }

    /// Return an iterator over the rows of the data matrix X.
    ///
    /// Parameters
    /// ----------
    /// chunk_size : int
    ///     Row size of a single chunk. Default: 500.
    ///
    /// Returns
    /// -------
    /// PyChunkedMatrix
    #[pyo3(
        signature = (chunk_size=500),
        text_signature = "($self, chunk_size=500)",
    )]
    #[pyo3(name = "chunked_X")]
    pub fn chunked_x(&self, chunk_size: usize) -> PyChunkedArray {
        self.0.chunked_x(chunk_size)
    }

    /// Filename of the backing .h5ad file.
    ///
    /// Returns
    /// -------
    /// Path
    #[getter]
    pub fn filename(&self) -> PathBuf {
        self.0.filename()
    }

    #[getter]
    pub fn backend(&self) -> String {
        self.0.backend().to_string()
    }

    /// Whether the AnnData object is backed. This is always true.
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    pub fn isbacked(&self) -> bool {
        true
    }

    /// If the AnnData object has been closed.
    ///
    /// Returns
    /// -------
    /// bool
    #[pyo3(text_signature = "($self)")]
    pub fn is_closed(&self) -> bool {
        self.0.is_closed()
    }

    /// Close the AnnData object.
    #[pyo3(text_signature = "($self)")]
    pub fn close(&self) -> Result<()> {
        self.0.close()
    }

    /// Reopen a closed AnnData object.
    #[pyo3(
        signature = (mode="r"),
        text_signature = "($self, mode='r')",
    )]
    pub fn open(&self, mode: &str) -> Result<()> {
        self.0.open(mode)
    }

    /// Write .h5ad-formatted hdf5 file.
    ///
    /// Parameters
    /// ----------
    /// filename: Path
    ///     File name of the output `.h5ad` file.
    /// backend: Literal['hdf5']
    ///     The backend to use. The Python bindings currently support "hdf5".
    /// partial : list[str] | None
    ///     A list of fields to copy. If None, copies all fields. Possible fields are:
    ///     "X", "obs", "var", "obsm", "obsp", "varm", "varp", "uns", "layers".
    /// chunk_size : int | None
    ///     If None, writes the entire data matrix at once. Otherwise,
    ///     rites the data matrix in chunks of the specified size.
    ///     This can be useful for saving large datasets that do not fit into memory.
    #[pyo3(
        signature = (filename, backend=H5::NAME, partial=None, chunk_size=None),
        text_signature = "($self, filename, backend='hdf5', partial=None, chunk_size=None)",
    )]
    pub fn write(
        &self,
        filename: PathBuf,
        backend: &str,
        partial: Option<Bound<PyAny>>,
        chunk_size: Option<usize>,
    ) -> Result<()> {
        let partial = partial.map(|p| {
            let mut set = HashSet::new();
            if let Ok(s) = p.extract::<String>() {
                set.insert(s);
            } else {
                p.try_iter().unwrap().for_each(|x| {
                    set.insert(x.unwrap().extract::<String>().unwrap());
                })
            }
            set
        });
        self.0.write(filename, backend, partial, chunk_size)
    }

    /// Copy the AnnData object.
    ///
    /// Parameters
    /// ----------
    /// filename
    ///     File name of the output `.h5ad` file.
    /// backend: Literal['hdf5']
    ///     The backend to use. The Python bindings currently support "hdf5".
    /// partial : list[str] | None
    ///     A list of fields to copy. If None, copies all fields. Possible fields are:
    ///     "X", "obs", "var", "obsm", "obsp", "varm", "varp", "uns", "layers".
    /// chunk_size : int | None
    ///     If None, writes the entire data matrix at once. Otherwise,
    ///     rites the data matrix in chunks of the specified size.
    ///     This can be useful for saving large datasets that do not fit into memory.
    ///
    /// Returns
    /// -------
    /// AnnData
    #[pyo3(
        signature = (filename, backend=H5::NAME, partial=None, chunk_size=None),
        text_signature = "($self, filename, backend='hdf5', partial=None, chunk_size=None)",
    )]
    fn copy(
        &self,
        filename: PathBuf,
        backend: &str,
        partial: Option<Bound<PyAny>>,
        chunk_size: Option<usize>,
    ) -> Result<Self> {
        let partial = partial.map(|p| {
            let mut set = HashSet::new();
            if let Ok(s) = p.extract::<String>() {
                set.insert(s);
            } else {
                p.try_iter().unwrap().for_each(|x| {
                    set.insert(x.unwrap().extract::<String>().unwrap());
                })
            }
            set
        });
        self.0.copy(filename, backend, partial, chunk_size)
    }

    /// Return a new AnnData object with all backed arrays loaded into memory.
    ///
    /// Parameters
    /// ----------
    /// partial : list[str] | None
    ///     A list of fields to copy. If None, copies all fields. Possible fields are:
    ///     "X", "obs", "var", "obsm", "obsp", "varm", "varp", "uns", "layers".
    ///
    /// Returns
    /// -------
    /// AnnData
    #[pyo3(
        signature = (partial=None),
        text_signature = "($self, partial=None)",
    )]
    pub fn to_memory<'py>(
        &self,
        py: Python<'py>,
        partial: Option<Bound<PyAny>>,
    ) -> Result<PyAnnData<'py>> {
        let partial = partial.map(|p| {
            let mut set = HashSet::new();
            if let Ok(s) = p.extract::<String>() {
                set.insert(s);
            } else {
                p.try_iter().unwrap().for_each(|x| {
                    set.insert(x.unwrap().extract::<String>().unwrap());
                })
            }
            set
        });
        self.0.to_memory(py, partial)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// A trait for AnnData with abstract backend. Each concrete backend should implement this trait.
trait AnnDataTrait: Send + Sync + Downcast {
    fn shape(&self) -> (usize, usize);
    fn obs_names(&self) -> DataFrameIndex;
    fn set_obs_names(&self, names: Bound<'_, PyAny>) -> Result<()>;
    fn obs_ix(&self, index: Bound<'_, PyAny>) -> Result<Vec<usize>>;
    fn var_names(&self) -> DataFrameIndex;
    fn set_var_names(&self, names: Bound<'_, PyAny>) -> Result<()>;
    fn var_ix(&self, index: Bound<'_, PyAny>) -> Result<Vec<usize>>;

    fn get_x(&self) -> Option<PyArrayElem>;
    fn take_x(&self) -> Result<Option<PyArrayData>>;
    fn get_obs(&self) -> Option<PyDataFrameElem>;
    fn get_var(&self) -> Option<PyDataFrameElem>;
    fn get_uns(&self) -> Option<PyElemCollection>;
    fn get_obsm(&self) -> Option<PyAxisArrays>;
    fn get_obsp(&self) -> Option<PyAxisArrays>;
    fn get_varm(&self) -> Option<PyAxisArrays>;
    fn get_varp(&self) -> Option<PyAxisArrays>;
    fn get_layers(&self) -> Option<PyAxisArrays>;

    fn set_x(&self, data: Option<PyArrayData>) -> Result<()>;
    fn set_obs(&self, obs: Option<Bound<'_, PyAny>>) -> Result<()>;
    fn set_var(&self, var: Option<Bound<'_, PyAny>>) -> Result<()>;
    fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()>;
    fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_layers(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()>;

    fn subset<'py>(
        &self,
        py: Python<'py>,
        slice: &[SelectInfoElem],
        file: Option<PathBuf>,
        inplace: bool,
        backend: Option<&str>,
    ) -> Result<Option<Bound<'py, PyAny>>>;

    fn split_obs_by(
        &self,
        key: Bound<'_, PyAny>,
        out_dir: PathBuf,
    ) -> Result<HashMap<String, AnnData>>;

    fn chunked_x(&self, chunk_size: usize) -> PyChunkedArray;

    fn write(
        &self,
        filename: PathBuf,
        backend: &str,
        partial: Option<HashSet<String>>,
        chunk_size: Option<usize>,
    ) -> Result<()>;
    fn copy(
        &self,
        filename: PathBuf,
        backend: &str,
        partial: Option<HashSet<String>>,
        chunk_size: Option<usize>,
    ) -> Result<AnnData>;
    fn to_memory<'py>(
        &self,
        py: Python<'py>,
        partial: Option<HashSet<String>>,
    ) -> Result<PyAnnData<'py>>;

    fn filename(&self) -> PathBuf;
    fn backend(&self) -> &str;
    fn is_closed(&self) -> bool;
    fn show(&self) -> String;

    /// Reopen a closed AnnData object.
    fn open(&self, mode: &str) -> Result<()>;
    fn close(&self) -> Result<()>;
    fn clone_ref(&self) -> Box<dyn AnnDataTrait>;
}
impl_downcast!(AnnDataTrait);

/// An AnnData object with abstract backend.
struct InnerAnnData<B: Backend> {
    filename: PathBuf,
    adata: Slot<anndata::AnnData<B>>,
}

impl<B: Backend> Clone for InnerAnnData<B> {
    fn clone(&self) -> Self {
        Self {
            filename: self.filename.clone(),
            adata: self.adata.clone(),
        }
    }
}

impl<B: Backend> AnnDataTrait for InnerAnnData<B> {
    fn shape(&self) -> (usize, usize) {
        let inner = self.adata.inner();
        (inner.n_obs(), inner.n_vars())
    }

    fn obs_names(&self) -> DataFrameIndex {
        self.adata.inner().obs_names()
    }

    fn obs_ix(&self, index: Bound<'_, PyAny>) -> Result<Vec<usize>> {
        let bounds: Vec<_> = index.try_iter()?.map(|x| x.unwrap()).collect();
        self.adata
            .inner()
            .obs_ix(bounds.iter().map(|x| x.extract::<&str>().unwrap()))
    }

    fn set_obs_names(&self, names: Bound<'_, PyAny>) -> Result<()> {
        let obs_names: Result<DataFrameIndex> = names
            .try_iter()?
            .map(|x| Ok(x?.extract::<String>()?))
            .collect();
        self.adata.inner().set_obs_names(obs_names?)
    }

    fn var_names(&self) -> DataFrameIndex {
        self.adata.inner().var_names()
    }

    fn var_ix(&self, index: Bound<'_, PyAny>) -> Result<Vec<usize>> {
        let bounds: Vec<_> = index.try_iter()?.map(|x| x.unwrap()).collect();
        self.adata
            .inner()
            .var_ix(bounds.iter().map(|x| x.extract::<&str>().unwrap()))
    }

    fn set_var_names(&self, names: Bound<'_, PyAny>) -> Result<()> {
        let var_names: Result<DataFrameIndex> = names
            .try_iter()?
            .map(|x| Ok(x?.extract::<String>()?))
            .collect();
        self.adata.inner().set_var_names(var_names?)
    }

    fn get_x(&self) -> Option<PyArrayElem> {
        let inner = self.adata.inner();
        let x = inner.get_x();
        if x.is_none() {
            None
        } else {
            Some(x.clone().into())
        }
    }

    fn take_x(&self) -> Result<Option<PyArrayData>> {
        self.adata
            .inner()
            .take_x::<ArrayData>()
            .map(|x| x.map(Into::into))
    }
    fn get_obs(&self) -> Option<PyDataFrameElem> {
        let inner = self.adata.inner();
        let obs = inner.get_obs();
        if obs.is_none() {
            None
        } else {
            Some(obs.clone().into())
        }
    }
    fn get_var(&self) -> Option<PyDataFrameElem> {
        let inner = self.adata.inner();
        let var = inner.get_var();
        if var.is_none() {
            None
        } else {
            Some(var.clone().into())
        }
    }
    fn get_uns(&self) -> Option<PyElemCollection> {
        let inner = self.adata.inner();
        let uns = inner.uns();
        if uns.is_none() {
            None
        } else {
            Some(uns.clone().into())
        }
    }
    fn get_obsm(&self) -> Option<PyAxisArrays> {
        let inner = self.adata.inner();
        let obsm = inner.obsm();
        if obsm.is_none() {
            None
        } else {
            Some(obsm.clone().into())
        }
    }
    fn get_obsp(&self) -> Option<PyAxisArrays> {
        let inner = self.adata.inner();
        let obsp = inner.obsp();
        if obsp.is_none() {
            None
        } else {
            Some(obsp.clone().into())
        }
    }
    fn get_varm(&self) -> Option<PyAxisArrays> {
        let inner = self.adata.inner();
        let varm = inner.varm();
        if varm.is_none() {
            None
        } else {
            Some(varm.clone().into())
        }
    }
    fn get_varp(&self) -> Option<PyAxisArrays> {
        let inner = self.adata.inner();
        let varp = inner.varp();
        if varp.is_none() {
            None
        } else {
            Some(varp.clone().into())
        }
    }

    fn get_layers(&self) -> Option<PyAxisArrays> {
        let inner = self.adata.inner();
        let layers = inner.layers();
        if layers.is_none() {
            None
        } else {
            Some(layers.clone().into())
        }
    }

    fn set_x(&self, data: Option<PyArrayData>) -> Result<()> {
        let inner = self.adata.inner();
        if let Some(d) = data {
            inner.set_x::<ArrayData>(d.into())?;
        } else {
            inner.del_x()?;
        }
        Ok(())
    }
    fn set_obs(&self, obs: Option<Bound<'_, PyAny>>) -> Result<()> {
        let inner = self.adata.inner();
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
        let inner = self.adata.inner();
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
        let inner = self.adata.inner();
        if let Some(u) = uns {
            inner.set_uns(u)?;
        } else {
            inner.del_uns()?;
        }
        Ok(())
    }
    fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.adata.inner();
        if let Some(o) = obsm {
            inner.set_obsm(o)?;
        } else {
            inner.del_obsm()?;
        }
        Ok(())
    }
    fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.adata.inner();
        if let Some(o) = obsp {
            inner.set_obsp(o)?;
        } else {
            inner.del_obsp()?;
        }
        Ok(())
    }
    fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.adata.inner();
        if let Some(v) = varm {
            inner.set_varm(v)?;
        } else {
            inner.del_varm()?;
        }
        Ok(())
    }
    fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.adata.inner();
        if let Some(v) = varp {
            inner.set_varp(v)?;
        } else {
            inner.del_varp()?;
        }
        Ok(())
    }
    fn set_layers(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.adata.inner();
        if let Some(v) = varp {
            inner.set_layers(v)?;
        } else {
            inner.del_layers()?;
        }
        Ok(())
    }

    fn subset<'py>(
        &self,
        py: Python<'py>,
        slice: &[SelectInfoElem],
        file: Option<PathBuf>,
        inplace: bool,
        backend: Option<&str>,
    ) -> Result<Option<Bound<'py, PyAny>>> {
        let inner = self.adata.inner();
        if inplace {
            inner.subset(slice)?;
            Ok(None)
        } else if let Some(file) = file {
            let backend = get_backend(&file, backend);
            match backend {
                H5::NAME => {
                    inner.write_select::<H5, _, _>(slice, &file)?;
                    Ok(Some(
                        AnnData::new_from(file, "r+", backend)?
                            .into_pyobject(py)?
                            .into_any(),
                    ))
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

            {
                if let Some(x) = inner.x().slice::<ArrayData, _>(slice)? {
                    adata.set_x(x)?;
                }
            }
            {
                // Set obs and var
                let obs_idx = inner.obs_names();
                if !obs_idx.is_empty() {
                    adata.set_obs_names(obs_idx.select(&slice[0]))?;
                    adata.set_obs(inner.read_obs()?.select_axis(0, &slice[0]))?;
                }
                let var_idx = inner.var_names();
                if !var_idx.is_empty() {
                    adata.set_var_names(var_idx.select(&slice[1]))?;
                    adata.set_var(inner.read_var()?.select_axis(0, &slice[1]))?;
                }
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
            {
                // Set layers
                inner.layers().keys().into_iter().try_for_each(|k| {
                    let elem = inner.layers().get(&k).unwrap();
                    let n = elem.shape().unwrap().ndim();
                    let mut select = vec![SelectInfoElem::full(); n];
                    select[0] = slice[0].clone();
                    select[1] = slice[1].clone();
                    let data = elem.slice::<ArrayData, _>(select)?.unwrap();
                    adata.layers().add(&k, data)
                })?;
            }
            Ok(Some(adata.into_pyobject(py)?.into_any()))
        }
    }

    fn split_obs_by(
        &self,
        key: Bound<'_, PyAny>,
        out_dir: PathBuf,
    ) -> Result<HashMap<String, AnnData>> {
        let inner = self.adata.inner();

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
        self.adata.inner().get_x().chunked(chunk_size).into()
    }

    fn write(
        &self,
        filename: PathBuf,
        backend: &str,
        partial: Option<HashSet<String>>,
        chunk_size: Option<usize>,
    ) -> Result<()> {
        match backend {
            H5::NAME => self
                .adata
                .inner()
                .write::<H5, _>(filename, partial, chunk_size),
            x => bail!("Unsupported backend: {x}"),
        }
    }

    fn copy(
        &self,
        filename: PathBuf,
        backend: &str,
        partial: Option<HashSet<String>>,
        chunk_size: Option<usize>,
    ) -> Result<AnnData> {
        AnnDataTrait::write(self, filename.clone(), backend, partial, chunk_size)?;
        AnnData::new_from(filename, "r+", backend)
    }

    fn to_memory<'py>(
        &self,
        py: Python<'py>,
        partial: Option<HashSet<String>>,
    ) -> Result<PyAnnData<'py>> {
        PyAnnData::from_anndata(py, self.adata.inner().deref(), partial)
    }

    fn filename(&self) -> PathBuf {
        self.filename.clone()
    }

    fn backend(&self) -> &str {
        B::NAME
    }

    fn is_closed(&self) -> bool {
        self.adata.is_none()
    }

    fn show(&self) -> String {
        if self.is_closed() {
            "Closed AnnData object".to_string()
        } else {
            format!("{}", self.adata.inner().deref())
        }
    }

    fn open(&self, mode: &str) -> Result<()> {
        if self.is_closed() {
            let file = match mode {
                "r" => B::open(self.filename())?,
                "r+" => B::open_rw(self.filename())?,
                _ => bail!("Unknown mode: {mode}"),
            };
            self.adata.insert(anndata::AnnData::<B>::open(file)?);
        }
        Ok(())
    }

    fn close(&self) -> Result<()> {
        if let Some(inner) = self.adata.extract() {
            inner.close()?;
        }
        Ok(())
    }

    fn clone_ref(&self) -> Box<dyn AnnDataTrait> {
        Box::new(self.clone())
    }
}

#[pyclass]
#[repr(transparent)]
pub struct StackedAnnData(Box<dyn StackedAnnDataTrait>);

impl<B: Backend> From<Slot<anndata::StackedAnnData<B>>> for StackedAnnData {
    fn from(x: Slot<anndata::StackedAnnData<B>>) -> Self {
        Self(Box::new(x))
    }
}

#[pymethods]
impl StackedAnnData {
    /// :class:`.PyDataFrame`.
    #[getter(obs)]
    fn get_obs(&self) -> Option<PyDataFrameElem> {
        self.0.get_obs()
    }

    /// :class:`.PyAxisArrays`.
    #[getter(obsm)]
    fn get_obsm(&self) -> Option<PyAxisArrays> {
        self.0.get_obsm()
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

trait StackedAnnDataTrait: Send + Sync + Downcast {
    fn get_obs(&self) -> Option<PyDataFrameElem>;
    fn get_obsm(&self) -> Option<PyAxisArrays>;
    fn show(&self) -> String;
}
impl_downcast!(StackedAnnDataTrait);

impl<B: Backend> StackedAnnDataTrait for Slot<anndata::StackedAnnData<B>> {
    fn get_obs(&self) -> Option<PyDataFrameElem> {
        let inner = self.inner();
        let obs = inner.get_obs();
        if obs.is_empty() {
            None
        } else {
            Some(obs.clone().into())
        }
    }
    fn get_obsm(&self) -> Option<PyAxisArrays> {
        let inner = self.inner();
        let obsm = inner.get_obsm();
        if obsm.is_empty() {
            None
        } else {
            Some(obsm.clone().into())
        }
    }
    fn show(&self) -> String {
        if self.is_none() {
            "Closed AnnData object".to_string()
        } else {
            format!("{}", self.inner().deref())
        }
    }
}
