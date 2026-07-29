"""Native anndata reference numbers for h5ad write/read.

Mirrors ``anndata-test-utils/tests/io_bench.rs``: the same matrix, the same
compression settings. Run via:

    uv run bench-io
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from .select_bench import bench_param, make_csr


def run_case(label: str, compression: str | None, csr: sp.csr_matrix) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.h5ad"
        adata = ad.AnnData(X=csr)

        start = time.perf_counter()
        adata.write_h5ad(path, compression=compression)
        write_ms = (time.perf_counter() - start) * 1000.0

        size_mb = path.stat().st_size / (1024 * 1024)

        start = time.perf_counter()
        back = ad.read_h5ad(path)
        _ = back.X.nnz
        read_ms = (time.perf_counter() - start) * 1000.0

        print(
            f"{label:<26} write {write_ms:>8.1f} ms   "
            f"read {read_ms:>8.1f} ms   file {size_mb:>7.1f} MB"
        )


def main() -> None:
    rows = bench_param("ANNDATA_BENCH_ROWS", 20_000)
    cols = bench_param("ANNDATA_BENCH_COLS", 10_000)
    per_row = bench_param("ANNDATA_BENCH_PER_ROW", 500)

    csr = make_csr(rows, cols, per_row)
    density = 100.0 * csr.nnz / (rows * cols)
    print(f"anndata {ad.__version__}, numpy {np.__version__}")
    print(f"\n=== {rows}x{cols}, nnz={csr.nnz}, density={density:.2f}% ===")

    run_case("gzip-5", "gzip", csr)
    run_case("uncompressed (default)", None, csr)


if __name__ == "__main__":
    main()
