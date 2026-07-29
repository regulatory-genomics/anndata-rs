"""Native ``anndata.AnnData`` reference numbers for in-memory subsetting.

Separates the cost of subsetting X from the surrounding AnnData object
machinery (obs/var index slicing), so the Rust ``select`` numbers can be
compared against the part that actually corresponds to them.

    uv run bench-anndata
"""

from __future__ import annotations

import time

import anndata as ad
import numpy as np
import pandas as pd

from .select_bench import bench_param, make_csr, time_it


def build_adata(rows: int, cols: int, per_row: int) -> ad.AnnData:
    csr = make_csr(rows, cols, per_row)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(np.repeat(["a", "b", "c", "d"], rows // 4 + 1)[:rows])},
        index=[f"cell_{i}" for i in range(rows)],
    )
    var = pd.DataFrame(
        {"highly_variable": np.arange(cols) % 3 == 0},
        index=[f"gene_{i}" for i in range(cols)],
    )
    return ad.AnnData(X=csr, obs=obs, var=var)


def run_case(label: str, rows: int, cols: int, per_row: int, repeats: int) -> None:
    adata = build_adata(rows, cols, per_row)
    print(
        f"\n=== {label}: {rows}x{cols}, nnz={adata.X.nnz}, "
        f"density={100.0 * adata.X.nnz / (rows * cols):.2f}% ==="
    )

    row_half = np.arange(rows // 2)
    col_half = np.arange(cols // 2)
    X = adata.X

    # X only: directly comparable to the Rust `select` benchmark.
    time_it("X only: rows", repeats, lambda: X[row_half, :])
    time_it("X only: rows + cols", repeats, lambda: X[row_half, :][:, col_half])

    # Full AnnData object: adds obs/var index slicing on top.
    time_it("AnnData: rows (copy)", repeats, lambda: adata[row_half, :].copy())
    time_it(
        "AnnData: rows + cols (copy)",
        repeats,
        lambda: adata[row_half, col_half].copy(),
    )

    # A view is lazy; included to show what anndata defers rather than does.
    time_it("AnnData: rows + cols (view)", repeats, lambda: adata[row_half, col_half])


def main() -> None:
    repeats = bench_param("ANNDATA_BENCH_REPEATS", 10)
    print(f"anndata {ad.__version__}, numpy {np.__version__}")
    run_case("small", 5_000, 2_000, 100, repeats)
    run_case("medium", 20_000, 10_000, 500, repeats)


if __name__ == "__main__":
    main()
