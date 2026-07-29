use std::collections::HashMap;

use crate::backend::{AttributeOp, Backend, DataContainer, DataType, GroupOp};
use crate::data::{
    array::DynScalar,
    array::slice::{SelectInfoElem, Shape},
};

use anyhow::Result;
use ndarray::ArrayD;
use serde_json::Value;

pub(crate) const MAPPING_ENCODING: MetaData = MetaData {
    encoding_type: "dict",
    version: "0.1.0",
    metadata: None,
};

pub struct MetaData {
    encoding_type: &'static str,
    version: &'static str,
    metadata: Option<HashMap<String, Value>>,
}

impl MetaData {
    pub(crate) fn new(
        encoding_type: &'static str,
        version: &'static str,
        metadata: Option<HashMap<String, Value>>,
    ) -> Self {
        Self {
            encoding_type,
            version,
            metadata,
        }
    }

    pub(crate) fn save<B: Backend, A: AttributeOp<B>>(self, loc: &mut A) -> Result<()> {
        loc.new_attr("encoding-type", self.encoding_type)?;
        loc.new_attr("encoding-version", self.version)?;
        if let Some(metadata) = self.metadata {
            for (key, value) in metadata.into_iter() {
                loc.new_attr(&key, value)?;
            }
        }
        Ok(())
    }
}

pub trait Element {
    fn data_type(&self) -> DataType;

    fn metadata(&self) -> MetaData;
}

impl<T> Element for &T
where
    T: Element,
{
    fn data_type(&self) -> DataType {
        (*self).data_type()
    }

    fn metadata(&self) -> MetaData {
        (*self).metadata()
    }
}

/// Read data from a backend
pub trait Readable: Element {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self>
    where
        Self: Sized;
}

/// Write data to a backend
pub trait Writable: Element {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>>;

    /// Overwrite the data in the container. The default implementation deletes the
    /// container and creates a new one. The data is then written to the new container.
    /// Specialized implementations may choose to overwrite the data in place.
    fn overwrite<B: Backend>(&self, container: DataContainer<B>) -> Result<DataContainer<B>> {
        let file = container.store()?;
        let path = container.path();
        let group = file.open_group(path.parent().unwrap().to_str().unwrap())?;
        let name = path.file_name().unwrap().to_str().unwrap();
        group.delete(name)?;
        self.write(&group, name)
    }
}

impl<T> Writable for &T
where
    T: Writable,
{
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        (*self).write(location, name)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Traits for arrays
////////////////////////////////////////////////////////////////////////////////

/// Anything that has a shape.
pub trait HasShape {
    fn shape(&self) -> Shape;
}

impl<T> HasShape for &T
where
    T: HasShape,
{
    fn shape(&self) -> Shape {
        (*self).shape()
    }
}

/// Anything that can be indexed.
pub trait Indexable: HasShape {
    fn get(&self, index: &[usize]) -> Option<DynScalar>;
}

pub trait Selectable: HasShape {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>;

    fn select_axis<S>(&self, axis: usize, slice: S) -> Self
    where
        S: AsRef<SelectInfoElem>,
        Self: Sized,
    {
        let full = SelectInfoElem::full();
        let selection = slice.as_ref().set_axis(axis, self.shape().ndim(), &full);
        self.select(selection.as_slice())
    }
}

pub trait Stackable: HasShape {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self>
    where
        Self: Sized;
}

pub trait ReadableArray: Readable {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape>;

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
        Self: Sized;

    fn read_axis<B, S>(container: &DataContainer<B>, axis: usize, slice: S) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
        Self: Sized,
    {
        let ndim = Self::get_shape(container)?.ndim();
        let full = SelectInfoElem::full();
        let selection = slice.as_ref().set_axis(axis, ndim, &full);
        Self::read_select(container, selection.as_slice())
    }
}

pub trait WritableArray: Writable {}

/// Trait for common array arithmetic operations
/// Supports operations like sum, mean, min, and max
pub trait ArrayArithmetic: HasShape {
    /// Compute the sum of all elements in the array
    fn sum(&self) -> f64;

    /// Compute the mean of all elements in the array
    fn mean(&self) -> f64 {
        let total_elements: usize = self.shape().as_ref().iter().product();
        if total_elements == 0 {
            0.0
        } else {
            self.sum() / total_elements as f64
        }
    }

    /// Compute the sum along a specific axis
    /// Returns a vector of sums for each slice along that axis
    fn sum_axis(&self, axis: usize) -> Result<ArrayD<f64>>;

    /// Compute the mean along a specific axis
    /// Returns a vector of means for each slice along that axis
    fn mean_axis(&self, axis: usize) -> Result<ArrayD<f64>> {
        let mut sum_result = self.sum_axis(axis)?;
        let axis_size = self.shape()[axis];
        if axis_size == 0 {
            sum_result.fill(0.0);
        } else {
            sum_result.mapv_inplace(|x| x / axis_size as f64);
        }
        Ok(sum_result)
    }

    /// Compute the minimum value in the array
    fn min(&self) -> f64;

    /// Compute the maximum value in the array
    fn max(&self) -> f64;
}

#[derive(Debug)]
pub enum SparseMatrixLayoutE {
    CSR,
    CSC,
    COO,
    NONE,
}

pub trait SparseMatrixLayout {
    fn get_sparse_layout(&self) -> SparseMatrixLayoutE {
        SparseMatrixLayoutE::NONE
    }
}
