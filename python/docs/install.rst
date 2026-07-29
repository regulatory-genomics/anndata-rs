Installation
============

From PyPI
---------

Install the user-facing Python package and import it as ``anndata_rs``:

::

    pip install anndata_rs

::

    import anndata_rs as ad

From source
-----------

You need a Rust toolchain to compile the extension module. Install Rust with
``rustup`` if it is not already available:

::

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

Clone the repository and install from the ``python`` package directory:

::

    git clone https://github.com/kaizhang/anndata-rs.git
    cd anndata-rs/python
    pip install .

Development installs can also use maturin:

::

    cd anndata-rs/python
    maturin develop
