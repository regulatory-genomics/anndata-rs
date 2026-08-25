use crate::{
    DataFrameElem,
    backend::{Backend, DataContainer, GroupOp},
    container::{Axis, AxisArrays, Dim, Slot},
    data::MAPPING_ENCODING,
};

use anyhow::Result;

pub(crate) fn open_df<B: Backend>(root: &B::Store, name: &str) -> Result<DataFrameElem<B>> {
    let df = if root.exists(name)? {
        DataFrameElem::try_from(DataContainer::open(root, "obs")?)?
    } else {
        Slot::none()
    };
    Ok(df)
}

// Helper function to create a new observation matrix (obsm)
// Dataframes stored in obsm are indexed by the obs names.
pub(crate) fn open_obsm<B: Backend>(group: B::Group, n_obs: Option<&Dim>) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Row, n_obs, None, Some("obs"))
}

// Helper function to create a new pairwise observation matrix (obsp)
pub(crate) fn open_obsp<B: Backend>(group: B::Group, n_obs: Option<&Dim>) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Pairwise, n_obs, None, None)
}

// Helper function to create a new variable matrix (varm)
pub(crate) fn open_varm<B: Backend>(
    group: B::Group,
    n_vars: Option<&Dim>,
) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Row, n_vars, None, Some("var"))
}

// Helper function to create a new pairwise variable matrix (varp)
pub(crate) fn open_varp<B: Backend>(
    group: B::Group,
    n_vars: Option<&Dim>,
) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Pairwise, n_vars, None, None)
}

// Helper function to create new layers of data
pub(crate) fn open_layers<B: Backend>(
    group: B::Group,
    n_obs: Option<&Dim>,
    n_vars: Option<&Dim>,
) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::RowColumn, n_obs, n_vars, None)
}

pub(crate) fn new_mapping<G: GroupOp<B>, B: Backend>(store: &G, name: &str) -> Result<B::Group> {
    let mut g = store.new_group(name)?;
    MAPPING_ENCODING.save(&mut g)?;
    Ok(g)
}
