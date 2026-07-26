use std::collections::HashMap;

use crate::backend::{AttributeOp, Backend, DataContainer, DatasetOp, GroupOp, ScalarType};
use crate::data::array::{
    CategoricalArray, DynArray,
    slice::{SelectInfoElem, Shape},
};
use crate::data::data_traits::*;
use crate::data::index::{Index, Interval};

use anyhow::{Context, Result, bail};
use log::warn;
use ndarray::{Array1, Ix1};
use polars::chunked_array::ChunkedArray;
use polars::datatypes::DataType;
use polars::prelude::{Column, DataFrame, Series};

use super::{BackendData, SelectInfoBounds, SelectInfoElemBounds};

impl Element for DataFrame {
    fn data_type(&self) -> crate::backend::DataType {
        crate::backend::DataType::DataFrame
    }

    fn metadata(&self) -> MetaData {
        let mut metadata = HashMap::new();

        let columns: Vec<String> = self
            .get_column_names()
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        metadata.insert("column-order".to_string(), columns.into());

        MetaData::new("dataframe", "0.2.0", Some(metadata))
    }
}

impl Writable for DataFrame {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let mut group = if location.exists(name)? {
            location.open_group(name)?
        } else {
            location.new_group(name)?
        };
        self.metadata().save(&mut group)?;

        self.columns().iter().try_for_each(|x| {
            write_column(x, &group, x.name())?;
            anyhow::Ok(())
        })?;

        let mut container = DataContainer::Group(group);

        // Create an index as the python anndata package enforce it. This is not used by this library
        DataFrameIndex::from(self.height()).overwrite(&mut container)?;

        Ok(container)
    }

    /// Overwrite the data inplace.
    fn overwrite<B: Backend>(&self, mut container: DataContainer<B>) -> Result<DataContainer<B>> {
        if let Ok(index_name) = container.get_attr::<String>("_index") {
            for obj in container.as_group()?.list()? {
                if obj != index_name {
                    container.as_group()?.delete(&obj)?;
                }
            }
            let n = self.height();
            if n != 0 && n != container.as_group()?.open_dataset(&index_name)?.shape()[0] {
                DataFrameIndex::from(self.height()).overwrite(&mut container)?;
            }
        } else {
            for obj in container.as_group()?.list()? {
                container.as_group()?.delete(&obj)?;
            }
            DataFrameIndex::from(self.height()).overwrite(&mut container)?;
        }

        self.columns().iter().try_for_each(|x| {
            write_column(x, container.as_group()?, x.name())?;
            anyhow::Ok(())
        })?;
        self.metadata().save(&mut container)?;

        Ok(container)
    }
}

impl Readable for DataFrame {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        use rayon::prelude::*;
        let shape = DataFrame::get_shape(container)?;
        let columns: Vec<String> = container.get_attr("column-order")?;
        let group = container.as_group()?;
        Ok(DataFrame::new(
            shape[0],
            columns
                .into_par_iter()
                .map(|name| {
                    let name_str = name.as_str();
                    let series_container = DataContainer::<B>::open(group, name_str)?;
                    let mut series = read_series::<B>(&series_container)
                        .with_context(|| format!("Failed to read series: {name_str}"))?;
                    series.rename(name_str.into());
                    Ok(series.into())
                })
                .collect::<Result<Vec<_>>>()?,
        )?)
    }
}

impl HasShape for DataFrame {
    fn shape(&self) -> Shape {
        self.shape().into()
    }
}

impl Selectable for DataFrame {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        if info.as_ref().len() != 2 {
            panic!("DataFrame only support 2D selection");
        }
        let columns = self.get_column_names();
        let select = SelectInfoBounds::new(&info, &HasShape::shape(self));
        let ridx = select.as_ref()[0].iter().map(|x| x as u32).collect();
        self.select(
            select.as_ref()[1]
                .to_vec()
                .into_iter()
                .map(|i| columns[i].as_str()),
        )
        .unwrap_or_else(|_| panic!("Failed to select columns: {:?}", select.as_ref()[1]))
        .take(&ChunkedArray::from_vec("idx".into(), ridx))
        .unwrap_or_else(|_| {
            panic!(
                "Failed to select rows: {:?}, shape: {:?}",
                select.as_ref()[0],
                self.shape()
            )
        })
    }
}

impl Stackable for DataFrame {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        Ok(iter
            .reduce(|mut a, b| {
                a.vstack_mut(&b).unwrap();
                a
            })
            .unwrap_or(Self::empty()))
    }
}

impl ReadableArray for DataFrame {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        let group = container.as_group()?;
        let index: String = group.get_attr("_index")?;
        let nrows = group.open_dataset(&index)?.shape()[0];
        let columns: Vec<String> = container.get_attr("column-order")?;
        Ok((nrows, columns.len()).into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        let columns: Vec<String> = container.get_attr("column-order")?;
        let columns: Result<Vec<Column>> = SelectInfoElemBounds::new(&info[1], columns.len())
            .iter()
            .map(|i| {
                let name = &columns[i];
                let series = container
                    .as_group()?
                    .open_dataset(name)
                    .map(DataContainer::Dataset)
                    .and_then(|x| read_series::<B>(&x))
                    .with_context(|| format!("Failed to read series: {name}"))?;

                let indices: Vec<u32> = SelectInfoElemBounds::new(&info[0], series.len())
                    .iter()
                    .map(|x| x.try_into().unwrap())
                    .collect();
                let mut series = series.take_slice(indices.as_slice())?;
                series.rename(name.into());
                Ok(series.into())
            })
            .collect();
        Ok(DataFrame::new_infer_height(columns?)?)
    }
}

impl WritableArray for DataFrame {}

#[derive(Debug, Clone)]
pub struct DataFrameIndex {
    pub index_name: String,
    index: Index,
}

impl std::cmp::PartialEq for DataFrameIndex {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl DataFrameIndex {
    pub fn empty() -> Self {
        Self {
            index_name: "index".to_string(),
            index: Index::empty(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn get_index(&self, k: &str) -> Option<usize> {
        self.index.get_index(k)
    }

    pub fn into_vec(self) -> Vec<String> {
        self.index.into_vec()
    }

    pub fn select(&self, select: &SelectInfoElem) -> Self {
        let index = self.index.select(select);
        Self {
            index_name: self.index_name.clone(),
            index,
        }
    }

    pub(crate) fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let index_name: String = container.get_attr("_index")?;
        let dataset = container.as_group()?.open_dataset(&index_name)?;
        match dataset
            .get_attr::<String>("index_type")
            .as_ref()
            .map_or("list", |x| x.as_str())
        {
            "list" => {
                let data = dataset.read_array()?;
                let mut index: DataFrameIndex = data.to_vec().into();
                index.index_name = index_name;
                Ok(index)
            }
            "intervals" => {
                let keys: Vec<String> = dataset.get_attr("names")?;
                let values: Vec<Vec<u64>> = dataset.get_attr("intervals")?;
                Ok(keys
                    .into_iter()
                    .zip(values.into_iter().map(|row| Interval {
                        start: row[0] as usize,
                        end: row[1] as usize,
                        size: row[2] as usize,
                        step: row[3] as usize,
                    }))
                    .collect())
            }
            "range" => {
                let start: u64 = dataset.get_attr("start")?;
                let end: u64 = dataset.get_attr("end")?;
                Ok((start as usize..end as usize).into())
            }
            x => bail!("Unknown index type: {x}"),
        }
    }

    /// Overwrite the index inplace.
    pub(crate) fn overwrite<B: Backend>(&self, container: &mut DataContainer<B>) -> Result<()> {
        if let Ok(index_name) = container.get_attr::<String>("_index") {
            container.as_group()?.delete(&index_name)?;
        }
        container.new_attr("_index", self.index_name.clone())?;
        let group = container.as_group()?;
        let arr: Array1<String> = self.clone().into_iter().collect();
        let mut data = arr.write(group, &self.index_name)?;
        match &self.index {
            Index::List(_) => {
                data.new_attr("index_type", "list")?;
            }
            Index::Intervals(intervals) => {
                let keys: Vec<String> = intervals.keys().cloned().collect();
                let values: Vec<Vec<u64>> = intervals
                    .values()
                    .map(|x| vec![x.start as u64, x.end as u64, x.size as u64, x.step as u64])
                    .collect();
                if data.new_attr("names", keys).is_err()
                    || data.new_attr("intervals", values).is_err()
                {
                    // fallback to "list"
                    data.new_attr("index_type", "list")?;
                    warn!("Failed to save interval index as attributes, fallback to list index");
                } else {
                    data.new_attr("index_type", "intervals")?;
                }
            }
            Index::Range(range) => {
                data.new_attr("index_type", "range")?;
                data.new_attr("start", range.start as u64)?;
                data.new_attr("end", range.end as u64)?;
            }
        }
        Ok(())
    }
}

impl IntoIterator for DataFrameIndex {
    type Item = String;
    type IntoIter = Box<dyn Iterator<Item = String>>;

    fn into_iter(self) -> Self::IntoIter {
        self.index.into_iter()
    }
}

impl<D> From<D> for DataFrameIndex
where
    Index: From<D>,
{
    fn from(data: D) -> Self {
        Self {
            index_name: "index".to_owned(),
            index: data.into(),
        }
    }
}

impl<D> FromIterator<D> for DataFrameIndex
where
    Index: FromIterator<D>,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = D>,
    {
        Self {
            index_name: "index".to_owned(),
            index: iter.into_iter().collect(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

fn write_column<B: Backend, G: GroupOp<B>>(
    series: &Column,
    location: &G,
    name: &str,
) -> Result<DataContainer<B>> {
    match series.dtype() {
        DataType::UInt8 => write_series_helper(series.u8()?, location, name),
        DataType::UInt16 => write_series_helper(series.u16()?, location, name),
        DataType::UInt32 => write_series_helper(series.u32()?, location, name),
        DataType::UInt64 => write_series_helper(series.u64()?, location, name),
        DataType::Int8 => write_series_helper(series.i8()?, location, name),
        DataType::Int16 => write_series_helper(series.i16()?, location, name),
        DataType::Int32 => write_series_helper(series.i32()?, location, name),
        DataType::Int64 => write_series_helper(series.i64()?, location, name),
        DataType::Float32 => series
            .f32()?
            .into_iter()
            .map(|x| x.unwrap_or(f32::NAN))
            .collect::<Array1<f32>>()
            .write(location, name),
        DataType::Float64 => series
            .f64()?
            .into_iter()
            .map(|x| x.unwrap_or(f64::NAN))
            .collect::<Array1<f64>>()
            .write(location, name),
        DataType::Boolean => write_series_helper(series.bool()?, location, name),
        DataType::String => {
            let series = series.str()?;
            if let Some(str_vec) = series
                .iter()
                .map(|x| x.map(|s| s.to_string()))
                .collect::<Option<Vec<_>>>()
            {
                Array1::from(str_vec).write(location, name)
            } else {
                series
                    .into_iter()
                    .collect::<CategoricalArray>()
                    .write(location, name)
            }
        }
        DataType::Categorical(_, _) => series
            .cat32()?
            .iter_str()
            .collect::<CategoricalArray>()
            .write(location, name),
        other => bail!("Unsupported series data type: {other:?}"),
    }
}

fn read_series<B: Backend>(container: &DataContainer<B>) -> Result<Series> {
    let ty = container.encoding_type()?;
    match ty {
        crate::backend::DataType::Categorical => {
            let categories = container.as_group()?.open_dataset("categories")?;
            let s = match categories.dtype()? {
                ScalarType::String => CategoricalArray::read(container)?.try_into()?,
                _ => read_cat_as_series(container)?,
            };
            Ok(s)
        }
        crate::backend::DataType::Array(_) => Ok(DynArray::read(container)?.into()),
        crate::backend::DataType::NullableArray => read_nullable(container),
        _ => bail!("Unsupported data type: {ty:?}"),
    }
}

/// Used to read non-string categorical data into regular arrays. After all, such
/// data should not be stored as categorical data.
fn read_cat_as_series<B: Backend>(container: &DataContainer<B>) -> Result<Series> {
    let group = container.as_group()?;
    let codes: Array1<i32> = group.open_dataset("codes")?.read_array_cast()?;
    let codes = codes.mapv(|x| if x < 0 { None } else { Some(x as usize) });
    let categories = group.open_dataset("categories")?.read_dyn_array().unwrap();

    macro_rules! fun {
        ($variant:ident, $value:expr) => {
            codes.iter().map(|x| x.map(|i| $value[i].clone())).collect()
        };
    }

    let arr = crate::macros::dyn_map!(categories, DynArray, fun);
    Ok(arr)
}

fn write_series_helper<B, G, I, T>(data: I, location: &G, name: &str) -> Result<DataContainer<B>>
where
    B: Backend,
    G: GroupOp<B>,
    I: IntoIterator<Item = Option<T>>,
    T: BackendData + Default,
{
    let mut has_null = false;
    let (values, mask): (Vec<_>, Vec<_>) = data
        .into_iter()
        .map(|x| {
            if let Some(x) = x {
                (x, false)
            } else {
                has_null = true;
                (T::default(), true)
            }
        })
        .unzip();

    if has_null {
        let mut group = location.new_group(name)?;
        let encoding = match T::DTYPE {
            ScalarType::I8
            | ScalarType::I16
            | ScalarType::I32
            | ScalarType::I64
            | ScalarType::U8
            | ScalarType::U16
            | ScalarType::U32
            | ScalarType::U64 => "nullable-integer",
            ScalarType::Bool => "nullable-boolean",
            ScalarType::String => "nullable-string",
            ScalarType::F32 | ScalarType::F64 => bail!("float types are not supported"),
        };
        MetaData::new(encoding, "0.1.0", None).save(&mut group)?;
        Array1::from(values).write(&group, "values")?;
        Array1::from(mask).write(&group, "mask")?;
        Ok(DataContainer::Group(group))
    } else {
        Array1::from(values).write(location, name)
    }
}

fn read_nullable<B>(container: &DataContainer<B>) -> Result<Series>
where
    B: Backend,
{
    let mask: Array1<bool> = container.as_group()?.open_dataset("mask")?.read_array()?;
    let dataset = container.as_group()?.open_dataset("values")?;

    macro_rules! fun {
        ($variant:ident) => {
            mask.iter()
                .zip(dataset.read_array::<$variant, Ix1>()?.into_iter())
                .map(|(m, v)| if *m { None } else { Some(v) })
                .collect()
        };
    }

    Ok(crate::macros::dyn_match!(dataset.dtype()?, ScalarType, fun))
}
