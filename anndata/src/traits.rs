use std::{collections::HashMap, path::Path};

use crate::{
    AnnData, AnnDataSet, ArrayElem, AxisArrays, Backend, ElemCollection, StackedArrayElem,
    StackedAxisArrays,
    anndata::elem_io::{new_mapping, open_layers, open_obsm, open_obsp, open_varm, open_varp},
    backend::DataType,
    container::{ChunkedArrayElem, InnerDataFrameElem, StackedChunkedArrayElem},
    data::*,
};

use anyhow::{Context, Result, bail, ensure};
use polars::prelude::DataFrame;
use smallvec::SmallVec;

/// Trait defining operations on an AnnData container.
pub trait AnnDataOp {
    type X: ArrayElemOp;
    type AxisArraysRef<'a>: AxisArraysOp
    where
        Self: 'a;
    type ElemCollectionRef<'a>: ElemCollectionOp
    where
        Self: 'a;

    /// Returns the 'X' element.
    fn x(&self) -> Self::X;

    /// Sets the 'X' element from an iterator.
    /// Note: The original data will be lost if an error occurs during the writing.
    fn set_x_from_iter<I, D>(&self, iter: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk + Into<ArrayData>;

    /// Sets the 'X' element.
    fn set_x<D: Into<ArrayData>>(&self, data: D) -> Result<()>;

    /// Deletes the 'X' element.
    fn del_x(&self) -> Result<()>;

    /// Returns the number of observations (rows).
    fn n_obs(&self) -> usize;
    /// Returns the number of variables (columns).
    fn n_vars(&self) -> usize;

    /// Sets the number of observations.
    fn set_n_obs(&self, n: usize) -> Result<()>;
    fn set_n_vars(&self, n: usize) -> Result<()>;

    /// Returns the names of observations.
    fn obs_names(&self) -> DataFrameIndex;
    /// Returns the names of variables.
    fn var_names(&self) -> DataFrameIndex;

    /// Changes the names of observations.
    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()>;
    /// Changes the names of variables.
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()>;

    /// Returns the indices of specified observations.
    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>>;
    /// Returns the indices of specified variables.
    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>>;

    /// Reads the observation annotations.
    fn read_obs(&self) -> Result<DataFrame>;
    /// Reads the variable annotations.
    fn read_var(&self) -> Result<DataFrame>;

    /// Takes the 'X' element, moving it out of the anndata object.
    fn take_x<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.x().take()
    }

    /// Changes the observation annotations.
    fn set_obs(&self, obs: DataFrame) -> Result<()>;

    /// Changes the variable annotations.
    fn set_var(&self, var: DataFrame) -> Result<()>;

    /// Deletes the observation annotations.
    fn del_obs(&self) -> Result<()>;

    /// Deletes the variable annotations.
    fn del_var(&self) -> Result<()>;

    /// Returns a reference to the unstructured data.
    fn uns(&self) -> Self::ElemCollectionRef<'_>;
    /// Returns a reference to the observation matrix.
    fn obsm(&self) -> Self::AxisArraysRef<'_>;
    /// Returns a reference to the observation pairwise data.
    fn obsp(&self) -> Self::AxisArraysRef<'_>;
    /// Returns a reference to the variable matrix.
    fn varm(&self) -> Self::AxisArraysRef<'_>;
    /// Returns a reference to the variable pairwise data.
    fn varp(&self) -> Self::AxisArraysRef<'_>;
    /// Returns a reference to the layers.
    fn layers(&self) -> Self::AxisArraysRef<'_>;

    /// Sets the unstructured data.
    fn set_uns<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<Data>,
    {
        self.del_uns()?;
        let uns = self.uns();
        data.into_iter().try_for_each(|(k, v)| uns.add(&k, v))
    }

    /// Sets the observation matrix.
    fn set_obsm<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        self.del_obsm()?;
        let obsm = self.obsm();
        data.into_iter().try_for_each(|(k, v)| obsm.add(&k, v))
    }

    /// Sets the observation pairwise data.
    fn set_obsp<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        self.del_obsp()?;
        let obsp = self.obsp();
        data.into_iter().try_for_each(|(k, v)| obsp.add(&k, v))
    }
    /// Sets the variable matrix.
    fn set_varm<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        self.del_varm()?;
        let varm = self.varm();
        data.into_iter().try_for_each(|(k, v)| varm.add(&k, v))
    }
    /// Sets the variable pairwise data.
    fn set_varp<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        self.del_varp()?;
        let varp = self.varp();
        data.into_iter().try_for_each(|(k, v)| varp.add(&k, v))
    }
    /// Sets the layers.
    fn set_layers<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        self.del_layers()?;
        let layers = self.layers();
        data.into_iter().try_for_each(|(k, v)| layers.add(&k, v))
    }

    /// Deletes the unstructured data.
    fn del_uns(&self) -> Result<()>;
    /// Deletes the observation matrix.
    fn del_obsm(&self) -> Result<()>;
    /// Deletes the observation pairwise data.
    fn del_obsp(&self) -> Result<()>;
    /// Deletes the variable matrix.
    fn del_varm(&self) -> Result<()>;
    /// Deletes the variable pairwise data.
    fn del_varp(&self) -> Result<()>;
    /// Deletes the layers.
    fn del_layers(&self) -> Result<()>;

    /// Split the AnnDataSet object into multiple AnnData objects based on grouping keys.
    /// The length of the grouping keys must match the number of observations.
    ///
    /// # Arguments:
    /// * `keys` - A slice of grouping keys. If key is None, the corresponding observation will not be included in any split AnnData.
    /// * `out_dir` - The output directory to save the split AnnData files.
    ///
    /// # Returns:
    /// A HashMap mapping each unique key to its corresponding AnnData object.
    fn split_obs_by<B: Backend, P: AsRef<Path>>(
        &self,
        keys: &[Option<String>],
        out_dir: P,
    ) -> Result<HashMap<String, AnnData<B>>> {
        let n_vars = self.n_vars();
        ensure!(
            keys.len() == self.n_obs(),
            "Length of grouping keys must match number of observations"
        );

        let mut indices = HashMap::new();
        keys.iter().enumerate().for_each(|(i, key)| {
            if let Some(key) = key {
                indices.entry(key).or_insert(Vec::new()).push(i);
            }
        });

        // Create anndata files
        std::fs::create_dir_all(&out_dir)?;
        let adatas = indices
            .into_iter()
            .map(|(key, idx)| {
                let path = out_dir.as_ref().join(format!("{key}.h5ad"));
                let adata = AnnData::<B>::new(&path)?;
                adata.set_n_obs(idx.len())?;
                adata.set_n_vars(n_vars)?;
                Ok((key, (adata, idx)))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        // Copy obs and var
        {
            let obs_names = self.obs_names();
            let var_names = self.var_names();
            let var = self.read_var()?;
            let obs = self.read_obs()?;
            adatas.iter().try_for_each(|(_, (adata, idx))| {
                let idx = SelectInfoElem::from(idx);
                if !obs_names.is_empty() {
                    adata.set_obs_names(obs_names.select(&idx))?;
                }
                if !var_names.is_empty() {
                    adata.set_var_names(var_names.clone())?;
                }
                adata.set_var(var.clone())?;
                adata.set_obs(Selectable::select_axis(&obs, 0, idx))?;
                anyhow::Ok(())
            })?;
        }

        // Copy X
        if let Some(dtype) = self.x().dtype() {
            let mut x_builders: HashMap<_, _> = match dtype {
                DataType::Array(t) => adatas
                    .iter()
                    .map(|(key, (adata, _))| {
                        (key, MatrixBuilder::new_dense(&adata.file, "X", t).unwrap())
                    })
                    .collect(),
                DataType::CsrMatrix(t, _p) => adatas
                    .iter()
                    .map(|(key, (adata, _))| {
                        (key, MatrixBuilder::new_sparse(&adata.file, "X", t).unwrap())
                    })
                    .collect(),
                _ => anyhow::bail!("Unsupported data type for X"),
            };

            self.x()
                .iter::<ArrayData>(10000)
                .for_each(|(chunk, start, end)| {
                    let mut idx = HashMap::new();
                    keys[start..end].iter().enumerate().for_each(|(i, key)| {
                        idx.entry(key).or_insert(Vec::new()).push(i);
                    });

                    idx.into_iter().for_each(|(key, rows)| {
                        if let Some(key) = key {
                            let arr = chunk.select_axis(0, SelectInfoElem::from(&rows));
                            x_builders.get_mut(&key).unwrap().add(arr).unwrap();
                        }
                    });
                });

            x_builders.into_iter().try_for_each(|(key, builder)| {
                adatas.get(key).unwrap().0.x.swap(&builder.finish()?);
                anyhow::Ok(())
            })?;
        }

        adatas
            .into_iter()
            .map(|(key, (adata, _))| Ok((key.to_string(), adata)))
            .collect()
    }
}

impl<T: AnnDataOp> AnnDataOp for &T {
    type X = T::X;
    type AxisArraysRef<'a>
        = T::AxisArraysRef<'a>
    where
        Self: 'a;
    type ElemCollectionRef<'a>
        = T::ElemCollectionRef<'a>
    where
        Self: 'a;

    fn x(&self) -> Self::X {
        (*self).x()
    }

    fn set_x_from_iter<I, D>(&self, iter: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk + Into<ArrayData>,
    {
        (*self).set_x_from_iter(iter)
    }

    fn set_x<D: Into<ArrayData>>(&self, data: D) -> Result<()> {
        (*self).set_x(data)
    }

    fn del_x(&self) -> Result<()> {
        (*self).del_x()
    }

    fn n_obs(&self) -> usize {
        (*self).n_obs()
    }

    fn n_vars(&self) -> usize {
        (*self).n_vars()
    }

    fn set_n_obs(&self, n: usize) -> Result<()> {
        (*self).set_n_obs(n)
    }

    fn set_n_vars(&self, n: usize) -> Result<()> {
        (*self).set_n_vars(n)
    }

    fn obs_names(&self) -> DataFrameIndex {
        (*self).obs_names()
    }

    fn var_names(&self) -> DataFrameIndex {
        (*self).var_names()
    }

    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        (*self).set_obs_names(index)
    }

    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        (*self).set_var_names(index)
    }

    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {
        (*self).obs_ix(names)
    }

    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {
        (*self).var_ix(names)
    }

    fn read_obs(&self) -> Result<DataFrame> {
        (*self).read_obs()
    }

    fn read_var(&self) -> Result<DataFrame> {
        (*self).read_var()
    }

    fn set_obs(&self, obs: DataFrame) -> Result<()> {
        (*self).set_obs(obs)
    }

    fn set_var(&self, var: DataFrame) -> Result<()> {
        (*self).set_var(var)
    }

    fn del_obs(&self) -> Result<()> {
        (*self).del_obs()
    }

    fn del_var(&self) -> Result<()> {
        (*self).del_var()
    }

    fn uns(&self) -> Self::ElemCollectionRef<'_> {
        (*self).uns()
    }

    fn obsm(&self) -> Self::AxisArraysRef<'_> {
        (*self).obsm()
    }

    fn obsp(&self) -> Self::AxisArraysRef<'_> {
        (*self).obsp()
    }

    fn varm(&self) -> Self::AxisArraysRef<'_> {
        (*self).varm()
    }

    fn varp(&self) -> Self::AxisArraysRef<'_> {
        (*self).varp()
    }

    fn layers(&self) -> Self::AxisArraysRef<'_> {
        (*self).layers()
    }

    fn set_uns<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<Data>,
    {
        (*self).set_uns(data)
    }

    fn set_obsm<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        (*self).set_obsm(data)
    }

    fn set_obsp<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        (*self).set_obsp(data)
    }

    fn set_varm<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        (*self).set_varm(data)
    }

    fn set_varp<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        (*self).set_varp(data)
    }

    fn del_uns(&self) -> Result<()> {
        (*self).del_uns()
    }

    fn del_obsm(&self) -> Result<()> {
        (*self).del_obsm()
    }

    fn del_obsp(&self) -> Result<()> {
        (*self).del_obsp()
    }

    fn del_varm(&self) -> Result<()> {
        (*self).del_varm()
    }

    fn del_varp(&self) -> Result<()> {
        (*self).del_varp()
    }

    fn del_layers(&self) -> Result<()> {
        (*self).del_layers()
    }

    fn set_layers<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        (*self).set_layers(data)
    }
}

impl<B: Backend> AnnDataOp for AnnData<B> {
    type X = ArrayElem<B>;
    type AxisArraysRef<'a> = &'a AxisArrays<B>;
    type ElemCollectionRef<'a> = &'a ElemCollection<B>;

    fn x(&self) -> Self::X {
        self.x.clone()
    }

    fn set_x_from_iter<I: Iterator<Item = D>, D: ArrayChunk>(&self, iter: I) -> Result<()> {
        let mut obs_lock = self.n_obs.lock();
        let mut vars_lock = self.n_vars.lock();
        self.del_x()?;
        let new_elem =
            ArrayElem::try_from(ArrayChunk::write_by_chunk(iter, &self.file, "X", None)?)?;
        let shape = new_elem.inner().shape().clone();

        match obs_lock.try_set(shape[0]).and(vars_lock.try_set(shape[1])) {
            Ok(_) => {
                self.x.swap(&new_elem);
                Ok(())
            }
            Err(e) => {
                new_elem.clear()?;
                Err(e)
            }
        }
    }

    fn set_x<D: Into<ArrayData>>(&self, data: D) -> Result<()> {
        let data = data.into();
        let shape = data.shape();
        ensure!(
            shape.ndim() >= 2,
            "X must be a N dimensional array, where N >= 2"
        );
        self.n_obs.try_set(shape[0])?;
        self.n_vars.try_set(shape[1])?;

        if !self.x.is_none() {
            self.x.inner().save(data)?;
        } else {
            let new_elem = ArrayElem::try_from(data.write(&self.file, "X")?)?;
            self.x.swap(&new_elem);
        }
        Ok(())
    }

    fn del_x(&self) -> Result<()> {
        self.x.clear()
    }

    fn n_obs(&self) -> usize {
        self.n_obs.get()
    }
    fn n_vars(&self) -> usize {
        self.n_vars.get()
    }

    fn set_n_obs(&self, n: usize) -> Result<()> {
        let mut n_obs = self.n_obs.lock();
        if let Err(e) = n_obs.try_set(n) {
            if self.x().is_none()
                && self.obs.is_none()
                && self.obsm().is_empty()
                && self.obsp().is_empty()
                && self.layers().is_empty()
            {
                n_obs.set(n);
            } else {
                return Err(e);
            }
        }
        Ok(())
    }
    fn set_n_vars(&self, n: usize) -> Result<()> {
        let mut n_vars = self.n_vars.lock();
        if let Err(e) = n_vars.try_set(n) {
            if self.x().is_none()
                && self.var.is_none()
                && self.varm().is_empty()
                && self.varp().is_empty()
                && self.layers().is_empty()
            {
                n_vars.set(n);
            } else {
                return Err(e);
            }
        }
        Ok(())
    }

    fn obs_names(&self) -> DataFrameIndex {
        self.obs
            .lock()
            .as_ref()
            .map_or(DataFrameIndex::empty(), |obs| obs.index.clone())
    }

    fn var_names(&self) -> DataFrameIndex {
        self.var
            .lock()
            .as_ref()
            .map_or(DataFrameIndex::empty(), |var| var.index.clone())
    }

    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        self.n_obs.try_set(index.len())?;
        if self.obs.is_none() {
            let df = InnerDataFrameElem::new(&self.file, "obs", Some(index), &DataFrame::empty())?;
            self.obs.insert(df);
        } else {
            self.obs.inner().set_index(index)?;
        }
        Ok(())
    }

    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        self.n_vars.try_set(index.len())?;
        if self.var.is_none() {
            let df = InnerDataFrameElem::new(&self.file, "var", Some(index), &DataFrame::empty())?;
            self.var.insert(df);
        } else {
            self.var.inner().set_index(index)?;
        }
        Ok(())
    }

    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {
        let inner = self.obs.inner();
        names
            .into_iter()
            .map(|i| {
                inner
                    .index
                    .get_index(i)
                    .context(format!("'{i}' does not exist in obs_names"))
            })
            .collect()
    }

    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {
        let inner = self.var.inner();
        names
            .into_iter()
            .map(|i| {
                inner
                    .index
                    .get_index(i)
                    .context(format!("'{i}' does not exist in obs_names"))
            })
            .collect()
    }

    fn read_obs(&self) -> Result<DataFrame> {
        self.get_obs()
            .lock()
            .as_mut()
            .map_or(Ok(DataFrame::empty()), |x| x.data().cloned())
    }
    fn read_var(&self) -> Result<DataFrame> {
        self.get_var()
            .lock()
            .as_mut()
            .map_or(Ok(DataFrame::empty()), |x| x.data().cloned())
    }

    fn set_obs(&self, obs: DataFrame) -> Result<()> {
        let nrows = obs.height();
        if nrows != 0 || obs.width() != 0 {
            self.n_obs.try_set(nrows)?;
            if self.obs.is_none() {
                self.obs
                    .insert(InnerDataFrameElem::new(&self.file, "obs", None, &obs)?);
            } else {
                self.obs.inner().save(obs)?;
            }
        }
        Ok(())
    }

    fn set_var(&self, var: DataFrame) -> Result<()> {
        let nrows = var.height();
        if nrows != 0 || var.width() != 0 {
            self.n_vars.try_set(nrows)?;
            if self.var.is_none() {
                self.var
                    .insert(InnerDataFrameElem::new(&self.file, "var", None, &var)?);
            } else {
                self.var.inner().save(var)?;
            }
        }
        Ok(())
    }

    fn del_obs(&self) -> Result<()> {
        self.get_obs().clear()
    }

    fn del_var(&self) -> Result<()> {
        self.get_var().clear()
    }

    fn uns(&self) -> Self::ElemCollectionRef<'_> {
        if self.uns.is_none() {
            let elems = new_mapping(&self.file, "uns").and_then(ElemCollection::new);
            if let Ok(uns) = elems {
                self.uns.swap(&uns);
            }
        }
        &self.uns
    }
    fn obsm(&self) -> Self::AxisArraysRef<'_> {
        if self.obsm.is_none() {
            let arrays =
                new_mapping(&self.file, "obsm").and_then(|g| open_obsm(g, Some(&self.n_obs)));
            if let Ok(obsm) = arrays {
                self.obsm.swap(&obsm);
            }
        }
        &self.obsm
    }
    fn obsp(&self) -> Self::AxisArraysRef<'_> {
        if self.obsp.is_none() {
            let arrays =
                new_mapping(&self.file, "obsp").and_then(|g| open_obsp(g, Some(&self.n_obs)));
            if let Ok(obsp) = arrays {
                self.obsp.swap(&obsp);
            }
        }
        &self.obsp
    }
    fn varm(&self) -> Self::AxisArraysRef<'_> {
        if self.varm.is_none() {
            let arrays =
                new_mapping(&self.file, "varm").and_then(|g| open_varm(g, Some(&self.n_vars)));
            if let Ok(varm) = arrays {
                self.varm.swap(&varm);
            }
        }
        &self.varm
    }
    fn varp(&self) -> Self::AxisArraysRef<'_> {
        if self.varp.is_none() {
            let arrays =
                new_mapping(&self.file, "varp").and_then(|g| open_varp(g, Some(&self.n_vars)));
            if let Ok(varp) = arrays {
                self.varp.swap(&varp);
            }
        }
        &self.varp
    }
    fn layers(&self) -> Self::AxisArraysRef<'_> {
        if self.layers.is_none() {
            let arrays = new_mapping(&self.file, "layers")
                .and_then(|g| open_layers(g, Some(&self.n_obs), Some(&self.n_vars)));
            if let Ok(layers) = arrays {
                self.layers.swap(&layers);
            }
        }
        &self.layers
    }

    fn del_uns(&self) -> Result<()> {
        self.uns.clear()
    }
    fn del_obsm(&self) -> Result<()> {
        self.obsm.clear()
    }
    fn del_obsp(&self) -> Result<()> {
        self.obsp.clear()
    }
    fn del_varm(&self) -> Result<()> {
        self.varm.clear()
    }
    fn del_varp(&self) -> Result<()> {
        self.varp.clear()
    }
    fn del_layers(&self) -> Result<()> {
        self.layers.clear()
    }
}

impl<B: Backend> AnnDataOp for AnnDataSet<B> {
    type X = StackedArrayElem<B>;
    type AxisArraysRef<'a> = &'a AxisArrays<B>;
    type ElemCollectionRef<'a> = &'a ElemCollection<B>;

    fn x(&self) -> Self::X {
        self.anndatas.inner().x.clone()
    }

    fn set_x_from_iter<I: Iterator<Item = D>, D: ArrayChunk>(&self, _iter: I) -> Result<()> {
        bail!("cannot set X in AnnDataSet")
    }

    fn set_x<D: Into<ArrayData>>(&self, _: D) -> Result<()> {
        bail!("cannot set X in AnnDataSet")
    }

    fn del_x(&self) -> Result<()> {
        bail!("cannot delete X in AnnDataSet")
    }

    fn n_obs(&self) -> usize {
        self.anndatas.inner().n_obs
    }
    fn n_vars(&self) -> usize {
        self.anndatas.inner().n_vars
    }
    fn set_n_obs(&self, n: usize) -> Result<()> {
        self.annotation.set_n_obs(n)
    }
    fn set_n_vars(&self, n: usize) -> Result<()> {
        self.annotation.set_n_vars(n)
    }

    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {
        self.annotation.obs_ix(names)
    }
    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {
        self.annotation.var_ix(names)
    }
    fn obs_names(&self) -> DataFrameIndex {
        self.annotation.obs_names()
    }
    fn var_names(&self) -> DataFrameIndex {
        self.annotation.var_names()
    }
    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        self.annotation.set_obs_names(index)
    }
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        self.annotation.set_var_names(index)
    }

    fn read_obs(&self) -> Result<DataFrame> {
        self.annotation.read_obs()
    }
    fn read_var(&self) -> Result<DataFrame> {
        self.annotation.read_var()
    }
    fn set_obs(&self, obs: DataFrame) -> Result<()> {
        self.annotation.set_obs(obs)
    }
    fn set_var(&self, var: DataFrame) -> Result<()> {
        self.annotation.set_var(var)
    }
    fn del_obs(&self) -> Result<()> {
        self.annotation.del_obs()
    }
    fn del_var(&self) -> Result<()> {
        self.annotation.del_var()
    }

    fn uns(&self) -> Self::ElemCollectionRef<'_> {
        self.annotation.uns()
    }
    fn obsm(&self) -> Self::AxisArraysRef<'_> {
        self.annotation.obsm()
    }
    fn obsp(&self) -> Self::AxisArraysRef<'_> {
        self.annotation.obsp()
    }
    fn varm(&self) -> Self::AxisArraysRef<'_> {
        self.annotation.varm()
    }
    fn varp(&self) -> Self::AxisArraysRef<'_> {
        self.annotation.varp()
    }
    fn layers(&self) -> Self::AxisArraysRef<'_> {
        self.annotation.layers()
    }

    fn set_uns<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<Data>,
    {
        self.annotation.set_uns(data)
    }

    fn set_obsm<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        self.annotation.set_obsm(data)
    }

    fn set_obsp<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        self.annotation.set_obsp(data)
    }

    fn set_varm<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        self.annotation.set_varm(data)
    }

    fn set_varp<I, D>(&self, data: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, D)>,
        D: Into<ArrayData>,
    {
        self.annotation.set_varp(data)
    }

    fn del_uns(&self) -> Result<()> {
        self.annotation.del_uns()
    }
    fn del_obsm(&self) -> Result<()> {
        self.annotation.del_obsm()
    }
    fn del_obsp(&self) -> Result<()> {
        self.annotation.del_obsp()
    }
    fn del_varm(&self) -> Result<()> {
        self.annotation.del_varm()
    }
    fn del_varp(&self) -> Result<()> {
        self.annotation.del_varp()
    }
    fn del_layers(&self) -> Result<()> {
        self.annotation.del_layers()
    }
}

/// Trait for operations on element collections.
pub trait ElemCollectionOp {
    /// Returns the keys of the collection.
    fn keys(&self) -> Vec<String>;

    /// Gets an item from the collection by key.
    fn get_item<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: TryFrom<Data>,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>;

    fn iter_item<D>(&self) -> impl Iterator<Item = (String, D)>
    where
        D: TryFrom<Data>,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>,
    {
        self.keys().into_iter().flat_map(|key| {
            let data = self.get_item::<D>(&key).unwrap()?;
            Some((key, data))
        })
    }

    fn drain<D>(&self) -> impl Iterator<Item = (String, D)>
    where
        D: TryFrom<Data>,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>,
    {
        self.keys().into_iter().flat_map(|key| {
            let data = self.get_item::<D>(&key).unwrap()?;
            self.remove(&key).unwrap();
            Some((key, data))
        })
    }

    /// Adds an item to the collection.
    fn add<D: Into<Data>>(&self, key: &str, data: D) -> Result<()>;

    /// Removes an item from the collection by key.
    fn remove(&self, key: &str) -> Result<()>;
}

/// Trait for accessing arrays with multiple axes.
pub trait AxisArraysOp {
    type ArrayElem: ArrayElemOp;

    /// Returns the keys of the axis arrays.
    fn keys(&self) -> Vec<String>;

    /// Gets an ArrayElem object by key, without reading the data.
    fn get(&self, key: &str) -> Option<Self::ArrayElem>;

    /// Gets the array data by key.
    fn get_item<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get(key)
            .and_then(|x| x.get().transpose())
            .transpose()
            .map_err(|e| e.context(format!("key: {key}")))
    }

    fn iter_item<D>(&self) -> impl Iterator<Item = (String, D)>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.keys().into_iter().flat_map(|key| {
            let data = self.get_item::<D>(&key).unwrap()?;
            Some((key, data))
        })
    }

    fn iter_item_slice<'a, D, S>(&'a self, slice: S) -> impl Iterator<Item = (String, D)> + 'a
    where
        D: TryFrom<ArrayData> + 'a,
        S: AsRef<[SelectInfoElem]> + 'a,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.keys().into_iter().flat_map(move |key| {
            let data = self.get_item_slice::<D, _>(&key, slice.as_ref()).unwrap()?;
            Some((key, data))
        })
    }

    fn drain<D>(&self) -> impl Iterator<Item = (String, D)>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.keys().into_iter().flat_map(|key| {
            let data = self.get_item::<D>(&key).unwrap()?;
            self.remove(&key).unwrap();
            Some((key, data))
        })
    }

    fn drain_slice<'a, D, S>(&'a self, slice: S) -> impl Iterator<Item = (String, D)> + 'a
    where
        D: TryFrom<ArrayData> + 'a,
        S: AsRef<[SelectInfoElem]> + 'a,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.keys().into_iter().flat_map(move |key| {
            let data = self.get_item_slice::<D, _>(&key, slice.as_ref()).unwrap()?;
            self.remove(&key).unwrap();
            Some((key, data))
        })
    }

    /// Gets a slice of the array data by key.
    fn get_item_slice<D, S>(&self, key: &str, slice: S) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get(key)
            .and_then(|x| x.slice(slice).transpose())
            .transpose()
    }

    /// Returns an iterator over the data.
    fn get_item_iter<D>(
        &self,
        key: &str,
        chunk_size: usize,
    ) -> Option<<Self::ArrayElem as ArrayElemOp>::ArrayIter<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: std::fmt::Debug,
    {
        self.get(key).map(|x| x.iter(chunk_size))
    }

    /// Adds array data by key.
    fn add<D: Into<ArrayData>>(&self, key: &str, data: D) -> Result<()>;

    /// Adds array data from an iterator by key.
    fn add_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk + Into<ArrayData>;

    /// Removes data by key.
    fn remove(&self, key: &str) -> Result<()>;
}

impl<B: Backend> AxisArraysOp for &AxisArrays<B> {
    type ArrayElem = ArrayElem<B>;

    fn keys(&self) -> Vec<String> {
        if self.is_empty() {
            Vec::new()
        } else {
            self.inner().keys().cloned().collect()
        }
    }

    fn get(&self, key: &str) -> Option<Self::ArrayElem> {
        self.lock().as_ref().and_then(|x| x.get(key).cloned())
    }

    fn add<D: Into<ArrayData>>(&self, key: &str, data: D) -> Result<()> {
        self.inner().add_data(key, data)
    }

    fn add_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk,
    {
        self.inner().add_data_from_iter(key, data)
    }

    fn remove(&self, key: &str) -> Result<()> {
        self.inner().remove_data(key)
    }
}

impl<B: Backend> AxisArraysOp for &StackedAxisArrays<B> {
    type ArrayElem = StackedArrayElem<B>;

    fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    fn get(&self, key: &str) -> Option<Self::ArrayElem> {
        self.data.get(key).cloned()
    }

    fn add<D: Into<ArrayData>>(&self, _key: &str, _data: D) -> Result<()> {
        todo!()
    }

    fn add_iter<I, D>(&self, _key: &str, _data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk,
    {
        todo!()
    }

    fn remove(&self, _key: &str) -> Result<()> {
        todo!()
    }
}

/// Trait for operations on array elements.
pub trait ArrayElemOp {
    type ArrayIter<D>: ExactSizeIterator<Item = (D, usize, usize)>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: std::fmt::Debug;

    /// Returns whether the element contains no data.
    fn is_none(&self) -> bool;

    /// Returns whether the data is currently cached in memory.
    fn is_cached(&self) -> bool;

    /// Enables caching of the array data.
    fn enable_cache(&self);

    /// Disables caching of the array data.
    fn disable_cache(&self);

    /// Return the data type of the array.
    fn dtype(&self) -> Option<DataType>;

    /// Returns the shape of the array.
    fn shape(&self) -> Option<Shape>;

    /// Gets the array data.
    fn get<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    /// Takes the array data, moving it out of the cache if present.
    fn take<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    /// Gets a slice of the array data.
    fn slice<D, S>(&self, slice: S) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    /// Gets a slice of the data along an axis.
    fn slice_axis<D, S>(&self, axis: usize, slice: S) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        S: AsRef<SelectInfoElem>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.shape()
            .and_then(|shape| {
                let full = SelectInfoElem::full();
                let slice: SmallVec<[SelectInfoElem; 3]> = slice
                    .as_ref()
                    .set_axis(axis, shape.ndim(), &full)
                    .into_iter()
                    .cloned()
                    .collect();
                self.slice(slice.as_slice()).transpose()
            })
            .transpose()
    }

    /// Returns an iterator over the data.
    fn iter<D>(&self, chunk_size: usize) -> Self::ArrayIter<D>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: std::fmt::Debug;
}

impl<B: Backend> ArrayElemOp for ArrayElem<B> {
    type ArrayIter<D>
        = ChunkedArrayElem<B, D>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: std::fmt::Debug;

    fn is_none(&self) -> bool {
        self.lock().is_none()
    }

    fn is_cached(&self) -> bool {
        self.lock().as_ref().is_some_and(|x| x.is_cached())
    }

    fn enable_cache(&self) {
        if let Some(elem) = self.lock().as_mut() {
            elem.enable_cache();
        }
    }

    fn disable_cache(&self) {
        if let Some(elem) = self.lock().as_mut() {
            elem.disable_cache();
        }
    }

    fn dtype(&self) -> Option<DataType> {
        self.lock().as_ref().map(|x| x.dtype())
    }

    fn shape(&self) -> Option<Shape> {
        self.lock().as_ref().map(|x| x.shape().clone())
    }

    fn get<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let mut lock = self.lock();
        if let Some(elem) = lock.as_mut() {
            elem.data()?.try_into().map_err(Into::into).map(Some)
        } else {
            Ok(None)
        }
    }

    fn take<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let mut lock = self.lock();
        if let Some(elem) = lock.as_mut() {
            elem.take()?.try_into().map_err(Into::into).map(Some)
        } else {
            Ok(None)
        }
    }

    fn slice<D, S>(&self, slice: S) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let mut lock = self.lock();
        if let Some(elem) = lock.as_mut() {
            elem.select(slice.as_ref())?
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
        self.chunked(chunk_size)
    }
}

impl<B: Backend> ArrayElemOp for StackedArrayElem<B> {
    type ArrayIter<D>
        = StackedChunkedArrayElem<B, D>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: std::fmt::Debug;

    fn is_none(&self) -> bool {
        self.0.is_none()
    }

    fn is_cached(&self) -> bool {
        false
    }

    fn enable_cache(&self) {}
    fn disable_cache(&self) {}

    fn dtype(&self) -> Option<DataType> {
        if self.is_none() {
            None
        } else {
            Some(self.elems[0].inner().dtype())
        }
    }

    fn shape(&self) -> Option<Shape> {
        self.shape.clone()
    }

    fn get<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.data()
    }

    fn take<D>(&self) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get()
    }

    fn slice<D, S>(&self, slice: S) -> Result<Option<D>>
    where
        D: TryFrom<ArrayData>,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.select(slice.as_ref())
    }

    fn iter<D>(&self, chunk_size: usize) -> Self::ArrayIter<D>
    where
        D: TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: std::fmt::Debug,
    {
        self.chunked(chunk_size)
    }
}
