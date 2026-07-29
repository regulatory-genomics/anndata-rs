mod backed;
mod dataset;
pub mod memory;

pub use backed::AnnData;
pub use dataset::AnnDataSet;
pub use memory::PyAnnData;

use anndata;
use anndata::Backend;
use anndata::concat::JoinType;
use anndata_hdf5::H5;
use anyhow::Result;
use pyo3::prelude::*;
use std::{
    collections::HashMap,
    ops::Deref,
    path::{Path, PathBuf},
};

pub(crate) fn get_backend<P: AsRef<Path>>(filename: P, backend: Option<&str>) -> &str {
    if let Some(backend) = backend {
        backend
    } else {
        if let Some(ext) = filename.as_ref().extension() {
            match ext.to_str().unwrap() {
                "h5ad" | "h5" | "h5ads" => H5::NAME,
                _ => H5::NAME,
            }
        } else {
            H5::NAME
        }
    }
}

/// Read `.h5ad`-formatted hdf5 file.
///
/// Parameters
/// ----------
///
/// filename: Path
///     File name of data file.
/// backed: Literal['r', 'r+'] | None
///     Default is `r+`.
///     If `'r'`, the file is opened in read-only mode.
///     If `'r+'`, the file is opened in read/write mode.
///     If `None`, the AnnData object is read into memory.
/// backend: Literal['hdf5']
#[pyfunction]
#[pyo3(
    signature = (filename, backed="r+", backend=None),
    text_signature = "(filename, backed='r+', backend=None)",
)]
pub fn read<'py>(
    py: Python<'py>,
    filename: PathBuf,
    backed: Option<&str>,
    backend: Option<&str>,
) -> Result<Bound<'py, PyAny>> {
    let adata = match backed {
        Some(m) => {
            let backend = get_backend(&filename, backend);
            AnnData::new_from(filename, m, backend)
                .unwrap()
                .into_pyobject(py)?
                .into_any()
        }
        None => PyModule::import(py, "anndata")?
            .getattr("read_h5ad")?
            .call1((filename,))?
            .into_pyobject(py)?
            .into_any(),
    };
    Ok(adata)
}

/// Concatenates AnnData objects.
///
/// When `file` is provided, this function saves the merged AnnData object on disk
/// in a streaming fashion. This is memory efficient and allows merging large datasets
/// that do not fit into memory.
///
/// Parameters
/// ----------
///
/// adatas: list[AnnData]
///     List of AnnData objects to concatenate.
/// join: Literal['inner', 'outer']
///     How to handle observations and variables that are not shared between all AnnData objects.
/// label: str | None
///     Column in axis annotation (i.e. .obs or .var) to place batch information in. If it’s None, no column is added.
/// keys
///     Names for each object being added. These values are used for column values for label.
/// file: Path | None
///     If provided, the concatenated AnnData will be saved to this file.
/// backend: Literal['hdf5']
///     Backend to use for writing the output file.
///
/// Returns
/// -------
///
/// AnnData
///     The concatenated AnnData object.
///
/// See Also
/// --------
/// AnnDataSet
#[pyfunction]
#[pyo3(
    signature = (adatas, *, join="inner", label=None, keys=None, file=None, backend=None),
    text_signature = "(adatas, *, join='inner', label=None, keys=None, file=None, backend=None)",
)]
pub fn concat<'py>(
    py: Python<'py>,
    adatas: Vec<Py<PyAny>>,
    join: &str,
    label: Option<&str>,
    keys: Option<Vec<String>>,
    file: Option<PathBuf>,
    backend: Option<&str>,
) -> Result<Bound<'py, PyAny>> {
    let join = match join {
        "inner" => JoinType::Inner,
        "outer" => JoinType::Outer,
        _ => panic!("Unknown join type"),
    };

    enum T<'a> {
        H5(anndata::AnnData<H5>),
        Py(PyAnnData<'a>),
    }

    let keys = keys.as_deref();
    let out = if let Some(file) = file {
        let backend = get_backend(&file, backend);
        match backend {
            H5::NAME => {
                let adata = anndata::AnnData::<H5>::new(file)?;
                T::H5(adata)
            }
            backend => todo!("Backend {} is not supported", backend),
        }
    } else {
        T::Py(PyAnnData::new(py)?)
    };

    if !adatas.is_empty() {
        if let Ok(adata) = adatas[0].extract::<AnnData>(py) {
            match adata.backend().as_str() {
                H5::NAME => {
                    let adatas = adatas
                        .into_iter()
                        .map(|x| x.extract::<AnnData>(py).unwrap())
                        .collect::<Vec<_>>();
                    let adatas: Vec<_> = adatas.iter().map(|x| x.inner_ref::<H5>()).collect();
                    let adatas: Vec<_> = adatas.iter().map(|x| x.deref()).collect();
                    match &out {
                        T::H5(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
                        T::Py(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
                    }
                }
                _ => todo!(),
            }
        } else {
            let adatas = adatas
                .into_iter()
                .map(|x| x.extract::<PyAnnData>(py).unwrap())
                .collect::<Vec<_>>();
            match &out {
                T::H5(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
                T::Py(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
            }
        }
    }

    match out {
        T::H5(adata) => Ok(AnnData::from(adata).into_pyobject(py)?.into_any()),
        T::Py(adata) => Ok(adata.into_pyobject(py)?.into_any()),
    }
}

/// Read Matrix Market file.
///
/// Parameters
/// ----------
///
/// mtx_file
///     File name of the input matrix market file.
/// obs_names
///     File that stores the observation names.
/// var_names
///     File that stores the variable names.
/// file
///     File name of the output ".h5ad" file.
/// backend: Literal['hdf5']
///     Backend to use for writing the output file.
/// sorted
///     If true, the input matrix is assumed to be sorted by rows.
///     Sorted input matrix can be read faster.
#[pyfunction]
#[pyo3(
    signature = (mtx_file, *, obs_names=None, var_names=None, file=None, backend=None, sorted=false),
    text_signature = "(mtx_file, *, obs_names=None, var_names=None, file=None, backend=None, sorted=False)",
)]
pub fn read_mtx<'py>(
    py: Python<'py>,
    mtx_file: PathBuf,
    obs_names: Option<PathBuf>,
    var_names: Option<PathBuf>,
    file: Option<PathBuf>,
    backend: Option<&str>,
    sorted: bool,
) -> Result<Bound<'py, PyAny>> {
    let mut reader = anndata::reader::MMReader::from_path(mtx_file)?;
    if let Some(obs_names) = obs_names {
        reader = reader.obs_names(obs_names)?;
    }
    if let Some(var_names) = var_names {
        reader = reader.var_names(var_names)?;
    }
    if sorted {
        reader = reader.is_sorted();
    }
    if let Some(file) = file {
        let backend = get_backend(&file, backend);
        match backend {
            H5::NAME => {
                let adata = anndata::AnnData::<H5>::new(file)?;
                reader.finish(&adata)?;
                Ok(AnnData::from(adata).into_pyobject(py)?.into_any())
            }
            backend => todo!("Backend {} is not supported", backend),
        }
    } else {
        let adata = PyAnnData::new(py)?;
        reader.finish(&adata)?;
        Ok(adata.into_pyobject(py)?)
    }
}

/// Read AnnDataSet object.
///
/// Read AnnDataSet from .h5ads file. If the file paths stored in AnnDataSet
/// object are relative paths, it will look for component .h5ad files in .h5ads file's parent directory.
///
/// Parameters
/// ----------
/// filename: Path
///     File name.
/// adata_files_update: Mapping[str, Path] | Path | None
///     AnnDataSet internally stores links to component anndata files.
///     You can find this information in `.uns['AnnDataSet']`.
///     These links may be invalid if the anndata files are moved to a different location.
///     This parameter provides a way to update the locations of component anndata files.
///     The value of this parameter can be either a mapping from component anndata file names to their new locations,
///     or a directory containing component anndata files.
/// mode: str
///     "r": Read-only mode; "r+": can modify annotation file but not component anndata files.
/// backend: Literal['hdf5']
///     Backend to use for reading the annotation file.
///
/// Returns
/// -------
/// AnnDataSet
#[pyfunction]
#[pyo3(
    signature = (filename, *, adata_files_update=None, mode="r+", backend=None),
    text_signature = "(filename, *, adata_files_update=None, mode='r+', backend=None)",
)]
pub fn read_dataset(
    filename: PathBuf,
    adata_files_update: Option<LocationUpdate>,
    mode: &str,
    backend: Option<&str>,
) -> Result<AnnDataSet> {
    let adata_files_update = match adata_files_update {
        Some(LocationUpdate::Map(map)) => Some(Ok(map)),
        Some(LocationUpdate::Dir(dir)) => Some(Err(dir)),
        None => None,
    };
    let backend = get_backend(&filename, backend);
    match backend {
        H5::NAME => {
            let file = match mode {
                "r" => H5::open(filename)?,
                "r+" => H5::open_rw(filename)?,
                _ => panic!("Unkown mode"),
            };
            Ok(anndata::AnnDataSet::<H5>::open(file, adata_files_update)?.into())
        }
        _ => todo!(),
    }
}

#[derive(FromPyObject)]
pub enum LocationUpdate {
    Map(HashMap<String, PathBuf>),
    Dir(PathBuf),
}
