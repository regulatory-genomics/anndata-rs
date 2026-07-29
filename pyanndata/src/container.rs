mod traits;

use crate::data::{PyArrayData, PyData};

use anyhow::Result;
use pyo3::prelude::*;
use pyo3_polars::PySeries;
use traits::{ArrayElemTrait, AxisArrayTrait, DataFrameElemTrait, ElemTrait};

use self::traits::{ChunkedArrayTrait, ElemCollectionTrait};

#[pyclass]
#[repr(transparent)]
pub struct PyElem(Box<dyn ElemTrait>);

impl<T: ElemTrait + 'static> From<T> for PyElem {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyElem {
    fn enable_cache(&self) {
        self.0.enable_cache();
    }

    fn disable_cache(&self) {
        self.0.disable_cache();
    }

    fn is_scalar(&self) -> bool {
        self.0.is_scalar()
    }

    fn __getitem__<'py>(&self, subscript: &Bound<'py, PyAny>) -> Result<PyData> {
        self.0.get(subscript)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}

/// An element that stores array objects such as dense arrays and sparse matrices.
/// Array elements support row and column indexing.
#[pyclass]
#[repr(transparent)]
pub struct PyArrayElem(Box<dyn ArrayElemTrait>);

impl<T: ArrayElemTrait + 'static> From<T> for PyArrayElem {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyArrayElem {
    /// Enable caching so that data will be stored in memory when the element
    /// is accessed the first time. Subsequent requests for the data will use
    /// the in-memory cache.
    #[pyo3(text_signature = "($self)")]
    fn enable_cache(&self) {
        self.0.enable_cache();
    }

    /// Disable caching. In-memory cache will be cleared immediately.
    #[pyo3(text_signature = "($self)")]
    fn disable_cache(&self) {
        self.0.disable_cache();
    }

    /// Shape of array.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.shape()
    }

    fn __getitem__(&self, subscript: &Bound<'_, PyAny>) -> Result<PyArrayData> {
        self.0.get(subscript)
    }

    /// Take the data from the element.
    #[pyo3(text_signature = "($self)")]
    fn take(&self) -> Result<PyArrayData> {
        self.0.take()
    }

    /// Return a chunk of the matrix with random indices.
    ///
    /// Parameters
    /// ----------
    /// size
    ///     Number of rows of the returned random chunk.
    /// replace
    ///     True means random sampling of indices with replacement, False without replacement.
    /// seed
    ///     Random seed.
    ///
    /// Returns
    /// -------
    /// A array
    #[pyo3(
        signature = (size, replace=true, seed=2022),
        text_signature = "($self, size, replace=True, seed=2022)",
    )]
    fn chunk(&self, size: usize, replace: bool, seed: u64) -> Result<PyArrayData> {
        self.0.chunk(size, replace, seed).map(PyArrayData::from)
    }

    /// Return an iterator over the rows of the matrix.
    ///
    /// Parameters
    /// ----------
    /// chunk_size
    ///     Number of rows of a single chunk.
    ///
    /// Returns
    /// -------
    /// An iterator, of which the elements are matrices.
    #[pyo3(text_signature = "($self, chunk_size)")]
    pub fn chunked(&self, chunk_size: usize) -> PyChunkedArray {
        self.0.chunked(chunk_size)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}

/// An element that stores dataframe objects.
#[pyclass]
#[repr(transparent)]
pub struct PyDataFrameElem(Box<dyn DataFrameElemTrait>);

impl<T: DataFrameElemTrait + 'static> From<T> for PyDataFrameElem {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyDataFrameElem {
    fn __getitem__<'py>(&self, subscript: &Bound<'py, PyAny>) -> Result<Bound<'py, PyAny>> {
        self.0.get(subscript)
    }

    fn __setitem__(&self, key: &str, data: &Bound<'_, PyAny>) -> Result<()> {
        let data: PySeries = data
            .py()
            .import("polars")?
            .call_method1("Series", (data,))?
            .extract()?;
        self.0.set(key, data.into())
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.contains(key)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}

/// A mapping, in which each key is associated with an axisarray
/// (a two or higher-dimensional ndarray).
/// It allows indexing and slicing along the associated axis.
///
/// Examples
/// --------
/// >>> data.obsm
/// AxisArrays (row) with keys: X_umap, insertion, X_spectral
/// >>> data.obsm['X_umap']
/// array([[13.279691  , -3.1859393 ],
///       [12.367847  , -1.9303571 ],
///       [11.376464  ,  0.36262953],
///       ...,
///       [12.1357565 , -2.777369  ],
///       [12.9115095 , -1.9225913 ],
///       [13.247231  , -4.200884  ]], dtype=float32)
/// >>> data.obsm.el('X_umap')
/// Array(Float(U4)) element, cache_enabled: no, cached: no
#[pyclass]
#[repr(transparent)]
pub struct PyAxisArrays(Box<dyn AxisArrayTrait>);

impl<T: AxisArrayTrait + 'static> From<T> for PyAxisArrays {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyAxisArrays {
    /// Return the keys.
    ///
    /// Returns
    /// -------
    /// List[str]
    #[pyo3(text_signature = "($self)")]
    pub fn keys(&self) -> Vec<String> {
        self.0.keys()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.contains(key)
    }

    fn __getitem__(&self, key: &str) -> Result<PyArrayData> {
        self.0.get(key)
    }

    /// Provide a lazy access to the elements.
    ///
    /// This function provides a lazy access to underlying elements. For example,
    /// calling `adata.obsm['elem']` will immediately read the data into memory,
    /// while using `adata.obsm.el('elem')` will return a :class:`.PyMatrixElem` object,
    /// which contains a reference to data stored in the disk.
    ///
    /// /// Examples
    /// --------
    /// >>> data.obsm
    /// AxisArrays (row) with keys: X_umap, insertion, X_spectral
    /// >>> data.obsm['X_umap']
    /// array([[13.279691  , -3.1859393 ],
    ///       [12.367847  , -1.9303571 ],
    ///       [11.376464  ,  0.36262953],
    ///       ...,
    ///       [12.1357565 , -2.777369  ],
    ///       [12.9115095 , -1.9225913 ],
    ///       [13.247231  , -4.200884  ]], dtype=float32)
    /// >>> data.obsm.el('X_umap')
    /// Array(Float(U4)) element, cache_enabled: no, cached: no
    ///
    /// Parameters
    /// ----------
    /// key
    ///     the name of the key.
    ///
    /// Returns
    /// -------
    /// Optional[PyArrayElem]
    #[pyo3(text_signature = "($self, key)")]
    fn el(&self, key: &str) -> Result<PyArrayElem> {
        self.0.el(key)
    }

    fn __setitem__(&self, key: &str, data: PyArrayData) -> Result<()> {
        self.0.set(key, data)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}

/// Unstructured annotations (ordered dictionary).
#[pyclass]
#[repr(transparent)]
pub struct PyElemCollection(Box<dyn ElemCollectionTrait>);

impl<T: ElemCollectionTrait + 'static> From<T> for PyElemCollection {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyElemCollection {
    pub fn keys(&self) -> Vec<String> {
        self.0.keys()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.contains(key)
    }

    fn __getitem__(&self, key: &str) -> Result<PyData> {
        self.0.get(key)
    }

    fn __setitem__(&self, key: &str, data: PyData) -> Result<()> {
        self.0.set(key, data)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyChunkedArray(Box<dyn ChunkedArrayTrait>);

#[pymethods]
impl PyChunkedArray {
    fn n_chunks(&self) -> usize {
        self.0.len()
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<(PyArrayData, usize, usize)> {
        slf.0
            .next()
            .map(|(data, start, end)| (data.into(), start, end))
    }
}

impl<T: ChunkedArrayTrait + 'static> From<T> for PyChunkedArray {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}
