use crate::{
    ArrayElem,
    anndata::{
        AnnData,
        elem_io::{open_df, open_obsm},
    },
    backend::{Backend, DataContainer, GroupOp, StoreOp},
    container::{Axis, Dim, Slot, StackedArrayElem, StackedAxisArrays, StackedDataFrame},
    data::*,
    traits::{AnnDataOp, ElemCollectionOp},
};

use anyhow::{Context, Result};
use itertools::Itertools;
use polars::{
    df,
    prelude::{Column, DataFrame},
};
use smallvec::SmallVec;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

pub struct AnnDataSet<B: Backend> {
    pub(crate) annotation: AnnData<B>,
    pub(crate) anndatas: Slot<StackedAnnData<B>>,
}

impl<B: Backend> std::fmt::Display for AnnDataSet<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AnnDataSet object with n_obs x n_vars = {} x {} backed at '{}'",
            self.n_obs(),
            self.n_vars(),
            self.annotation.filename().display(),
        )?;
        let adatas = self.anndatas.inner();
        if !adatas.is_empty() {
            write!(f, "\ncontains {} AnnData objects", adatas.len(),)?;
        }
        if let Some(obs) = self
            .annotation
            .obs
            .lock()
            .as_ref()
            .map(|x| x.get_column_names())
            && !obs.is_empty()
        {
            write!(f, "\n    obs: '{}'", obs.into_iter().join("', '"))?;
        }
        if let Some(var) = self
            .annotation
            .var
            .lock()
            .as_ref()
            .map(|x| x.get_column_names())
            && !var.is_empty()
        {
            write!(f, "\n    var: '{}'", var.into_iter().join("', '"))?;
        }
        if let Some(keys) = self
            .annotation
            .uns
            .lock()
            .as_ref()
            .map(|x| x.keys().join("', '"))
            && !keys.is_empty()
        {
            write!(f, "\n    uns: '{keys}'")?;
        }
        if let Some(keys) = self
            .annotation
            .obsm
            .lock()
            .as_ref()
            .map(|x| x.keys().join("', '"))
            && !keys.is_empty()
        {
            write!(f, "\n    obsm: '{keys}'")?;
        }
        if let Some(keys) = self
            .annotation
            .obsp
            .lock()
            .as_ref()
            .map(|x| x.keys().join("', '"))
            && !keys.is_empty()
        {
            write!(f, "\n    obsp: '{keys}'")?;
        }
        if let Some(keys) = self
            .annotation
            .varm
            .lock()
            .as_ref()
            .map(|x| x.keys().join("', '"))
            && !keys.is_empty()
        {
            write!(f, "\n    varm: '{keys}'")?;
        }
        if let Some(keys) = self
            .annotation
            .varp
            .lock()
            .as_ref()
            .map(|x| x.keys().join("', '"))
            && !keys.is_empty()
        {
            write!(f, "\n    varp: '{keys}'")?;
        }
        Ok(())
    }
}

impl<B: Backend> AnnDataSet<B> {
    pub fn adatas(&self) -> &Slot<StackedAnnData<B>> {
        &self.anndatas
    }

    pub fn get_anno(&self) -> &AnnData<B> {
        &self.annotation
    }

    pub fn new<T, S, P>(
        data: T,
        filename: P,
        add_key: &str,
        use_absolute_path: bool,
    ) -> Result<Self>
    where
        T: IntoIterator<Item = (S, AnnData<B>)>,
        S: ToString,
        P: AsRef<Path>,
    {
        let mut adatas = data.into_iter().peekable();
        let mut annotation = AnnData::new(filename)?;

        // Set VAR.
        let var_names = adatas.peek().expect("no AnnData objects").1.var_names();
        let n_vars = var_names.len();
        annotation.n_vars = Dim::new(n_vars);
        if !var_names.is_empty() {
            annotation.set_var_names(var_names.clone())?;
        }

        let mut n_obs = 0;
        let mut file_keys = Vec::new();
        let mut files = Vec::new();
        let mut add_keys = Vec::new();
        let mut xs = SmallVec::new();
        let mut obs_names = Vec::new();
        let mut obs = Vec::new();
        let mut obsm = Vec::new();
        let mut uns = Vec::new();

        for (key, adata) in adatas {
            assert!(
                adata.var_names() == var_names,
                "var_names mismatch when stacking AnnData objects"
            );

            n_obs += adata.n_obs();
            xs.push(adata.get_x().clone());
            adata
                .obs_names()
                .into_iter()
                .for_each(|x| obs_names.push(x.clone()));
            obs.push(adata.obs.clone());
            obsm.push(adata.obsm.clone());
            uns.push(adata.uns.clone());

            add_keys.extend(vec![key.to_string(); adata.n_obs()]);
            file_keys.push(key.to_string());
            files.push(adata.file);
        }

        // Set OBS.
        annotation.n_obs = Dim::new(n_obs);
        if !obs_names.is_empty() && obs_names.len() == n_obs {
            annotation.set_obs_names(obs_names.into())?;
        }
        annotation.set_obs(df!(add_key => add_keys)?)?;

        // Set UNS. UNS includes children anndata locations and shared elements.
        let filenames: Vec<_> = files
            .iter()
            .map(|x| {
                let mut filename = x.filename();
                if use_absolute_path {
                    filename = std::fs::canonicalize(filename).unwrap()
                }
                filename.display().to_string()
            })
            .collect();
        annotation.uns().add(
            "AnnDataSet",
            df!("keys" => file_keys, "file_path" => filenames)?,
        )?;

        // Add shared uns elements.
        let shared_keys: HashSet<String> = uns
            .iter()
            .map(|x| x.keys().into_iter().collect::<HashSet<_>>())
            .reduce(|a, b| a.intersection(&b).cloned().collect())
            .unwrap_or(HashSet::new());
        for key in shared_keys {
            if uns
                .iter()
                .map(|x| x.get_item::<Data>(&key).unwrap().unwrap())
                .all_equal()
            {
                annotation
                    .uns()
                    .add(&key, uns.first().unwrap().get_item::<Data>(&key)?.unwrap())?;
            }
        }

        let anndatas = StackedAnnData {
            files,
            n_obs,
            n_vars,
            x: StackedArrayElem::new(xs)?,
            obs: StackedDataFrame::new(obs)?,
            obsm: StackedAxisArrays::new(Axis::Row, obsm)?,
        };

        Ok(Self {
            annotation,
            anndatas: Slot::new(anndatas),
        })
    }

    pub fn open<P: AsRef<Path>>(
        file: B::Store,
        adata_files_update: Option<Result<HashMap<String, P>, P>>,
    ) -> Result<Self> {
        let annotation: AnnData<B> = AnnData::open(file)?;
        let file_path = annotation
            .filename()
            .read_link()
            .unwrap_or(annotation.filename())
            .to_path_buf();

        let adata_files = match adata_files_update {
            None => update_anndata_locations_by_map(&annotation, HashMap::<String, P>::new())?,
            Some(Ok(adata_files)) => update_anndata_locations_by_map(&annotation, adata_files)?,
            Some(Err(dir)) => update_anndata_location_dir(&annotation, dir)?,
        };

        let files: Vec<_> = adata_files
            .into_iter()
            .map(|(_, path)| {
                if path.is_absolute() {
                    B::open(path)
                } else {
                    B::open(file_path.parent().unwrap_or(Path::new("./")).join(path))
                }
            })
            .collect::<Result<_>>()?;

        let n_obs = annotation.n_obs();
        let n_vars = annotation.n_vars();
        Ok(Self {
            annotation,
            anndatas: Slot::new(StackedAnnData::new(files, n_obs, n_vars)?),
        })
    }

    /// Convert AnnDataSet to AnnData object
    pub fn to_adata<O: Backend, P: AsRef<Path>>(&self, out: P, copy_x: bool) -> Result<AnnData<O>> {
        self.annotation.write::<O, _>(&out, None, None)?;
        let adata = AnnData::open(O::open_rw(&out)?)?;
        if copy_x {
            adata.set_x_from_iter::<_, ArrayData>(
                self.anndatas.inner().x.chunked(500).map(|x| x.0),
            )?;
        }
        Ok(adata)
    }

    pub fn to_adata_select<O, P, S>(&self, select: S, out: P, copy_x: bool) -> Result<AnnData<O>>
    where
        O: Backend,
        P: AsRef<Path>,
        S: AsRef<[SelectInfoElem]>,
    {
        self.annotation.write_select::<O, _, _>(&select, &out)?;
        let adata = AnnData::open(O::open_rw(&out)?)?;
        if copy_x {
            let x: ArrayData = self.anndatas.inner().x.select(select.as_ref())?.unwrap();
            adata.set_x(x)?;
        }
        Ok(adata)
    }

    /// Convert AnnDataSet to AnnData object
    pub fn into_adata(self, copy_x: bool) -> Result<AnnData<B>> {
        if copy_x {
            self.annotation.set_x_from_iter::<_, ArrayData>(
                self.anndatas.inner().x.chunked(500).map(|x| x.0),
            )?;
        }
        for ann in self.anndatas.extract().unwrap().files.into_iter() {
            ann.close()?;
        }
        Ok(self.annotation)
    }

    pub fn close(self) -> Result<()> {
        self.annotation.close()?;
        for ann in self.anndatas.extract().unwrap().files.into_iter() {
            ann.close()?;
        }
        Ok(())
    }
}

/// Update the locations of AnnData files.
fn update_anndata_locations_by_map<B: Backend, P: AsRef<Path>>(
    ann: &AnnData<B>,
    new_locations: HashMap<String, P>,
) -> Result<Vec<(String, PathBuf)>> {
    let df: DataFrame = ann
        .uns()
        .get_item("AnnDataSet")?
        .context("key 'AnnDataSet' is not present")?;
    let keys = df.column("keys").unwrap();
    let filenames = as_str_vec(df.column("file_path")?);
    let new_files: Vec<_> = as_str_vec(keys)
        .into_iter()
        .zip(filenames)
        .map(|(k, v)| {
            let name = new_locations
                .get(&k)
                .map_or(PathBuf::from(v), |x| x.as_ref().to_path_buf());
            (k.to_string(), name)
        })
        .collect();
    let data = DataFrame::new_infer_height(vec![
        keys.clone(),
        Column::new(
            "file_path".into(),
            new_files
                .iter()
                .map(|x| x.1.to_str().unwrap().to_string())
                .collect::<Vec<_>>(),
        ),
    ])
    .unwrap();
    if !new_locations.is_empty() {
        ann.uns().add("AnnDataSet", data)?;
    }
    Ok(new_files)
}

fn update_anndata_location_dir<B: Backend, P: AsRef<Path>>(
    ann: &AnnData<B>,
    dir: P,
) -> Result<Vec<(String, PathBuf)>> {
    let df: DataFrame = ann
        .uns()
        .get_item("AnnDataSet")?
        .context("key 'AnnDataSet' is not present")?;
    let keys = df.column("keys").unwrap();
    let file_map: HashMap<String, PathBuf> = std::fs::read_dir(dir)?
        .map(|x| x.map(|entry| (entry.file_name().into_string().unwrap(), entry.path())))
        .collect::<Result<_, std::io::Error>>()?;
    let filenames = as_str_vec(df.column("file_path")?);
    let new_files: Vec<_> = as_str_vec(keys)
        .into_iter()
        .zip(filenames)
        .map(|(k, filename)| {
            let path = PathBuf::from(filename);
            let name = path.file_name().unwrap().to_str().unwrap();
            (
                k,
                file_map
                    .get(name)
                    .map_or(path, |x| std::fs::canonicalize(x).unwrap()),
            )
        })
        .collect();
    let data = DataFrame::new_infer_height(vec![
        keys.clone(),
        Column::new(
            "file_path".into(),
            new_files
                .iter()
                .map(|x| x.1.to_str().unwrap().to_string())
                .collect::<Vec<_>>(),
        ),
    ])
    .unwrap();
    ann.uns().add("AnnDataSet", data)?;
    Ok(new_files)
}

pub struct StackedAnnData<B: Backend> {
    files: Vec<B::Store>,
    pub(crate) n_obs: usize,
    pub(crate) n_vars: usize,
    pub(crate) x: StackedArrayElem<B>,
    pub(crate) obs: StackedDataFrame<B>,
    pub(crate) obsm: StackedAxisArrays<B>,
}

impl<B: Backend> std::fmt::Display for StackedAnnData<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stacked AnnData objects:")?;
        write!(
            f,
            "\n    obs: '{}'",
            self.obs.get_column_names().iter().join("', '")
        )?;
        write!(f, "\n    obsm: '{}'", self.obsm.keys().join("', '"))?;
        Ok(())
    }
}

impl<B: Backend> StackedAnnData<B> {
    fn new(files: Vec<B::Store>, n_obs: usize, n_vars: usize) -> Result<Self> {
        let mut xs = SmallVec::new();
        let mut obs = Vec::new();
        let mut obsm = Vec::new();
        for file in &files {
            let x = if file.exists("X")? {
                ArrayElem::try_from(DataContainer::open(file, "X")?)?
            } else {
                Slot::none()
            };
            xs.push(x);

            obs.push(open_df(file, "obs")?);
            obsm.push(open_obsm(file.open_group("obsm")?, None)?);
        }
        Ok(Self {
            files,
            n_obs,
            n_vars,
            x: StackedArrayElem::new(xs)?,
            obs: StackedDataFrame::new(obs)?,
            obsm: StackedAxisArrays::new(Axis::Row, obsm)?,
        })
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    pub fn get_x(&self) -> &StackedArrayElem<B> {
        &self.x
    }

    pub fn get_obs(&self) -> &StackedDataFrame<B> {
        &self.obs
    }

    pub fn get_obsm(&self) -> &StackedAxisArrays<B> {
        &self.obsm
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

fn as_str_vec(series: &Column) -> Vec<String> {
    if let Ok(s) = series.str() {
        s.iter()
            .map(|x| x.unwrap().to_string())
            .collect::<Vec<_>>()
    } else {
        series
            .cat32()
            .unwrap()
            .iter_str()
            .map(|x| x.unwrap().to_string())
            .collect::<Vec<_>>()
    }
}
