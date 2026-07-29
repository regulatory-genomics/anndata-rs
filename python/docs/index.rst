anndata-rs Python bindings
==========================

``anndata_rs`` is the Python package for the Rust ``anndata-rs`` backend.
It provides a backed AnnData API for reading, writing, subsetting, and
streaming AnnData objects without loading every field into memory.

The package follows the AnnData data model:

* ``X``: primary observations × variables matrix, dense or sparse.
* ``obs`` / ``var``: observation and variable annotations.
* ``obsm`` / ``varm``: multi-dimensional annotations.
* ``obsp`` / ``varp``: pairwise annotations.
* ``layers``: alternative matrices with the same shape as ``X``.
* ``uns``: unstructured metadata.

Key differences from Python ``anndata``
---------------------------------------

1. ``AnnData`` is opened in backed mode by default.
2. Elements are lazily loaded; arrays are read when requested.
3. Subsetting writes/copies data instead of creating AnnData views.
4. The current Python package is centered on HDF5/``.h5ad`` workflows. Use
   the Rust API directly for the full set of Rust backend features, including
   object-store Zarr workflows.

Package layout note
-------------------

The user-facing Python package is in the repository's ``python/`` directory and
is imported as ``anndata_rs``. The ``pyanndata/`` Rust crate contains the internal
PyO3 implementation used by the package.

For details about the AnnData specification, see
https://anndata.readthedocs.io/en/latest/.

.. toctree::
   :maxdepth: 3
   :hidden:

   install
   api
