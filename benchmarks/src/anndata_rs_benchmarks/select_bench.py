"""scipy reference numbers for sparse subsetting.

Mirrors ``anndata-test-utils/tests/select_bench.rs`` exactly: the same matrix
construction (deterministic, no RNG), the same shapes, and the same four
selections. Run via:

    uv run bench-select
"""

from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np
import scipy
import scipy.sparse as sp


def make_csr(rows: int, cols: int, per_row: int) -> sp.csr_matrix:
    """Same deterministic pattern as ``make_csr`` in select_bench.rs."""
    step = 7919
    k = np.arange(per_row)
    # Row r gets columns (r + k*step) % cols, sorted and deduplicated.
    raw = (np.arange(rows)[:, None] + k[None, :] * step) % cols
    raw.sort(axis=1)

    indices_per_row = [np.unique(row) for row in raw]
    counts = np.fromiter((len(x) for x in indices_per_row), dtype=np.int64, count=rows)
    indptr = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.concatenate(indices_per_row).astype(np.int32)
    data = np.concatenate(
        [np.arange(c, dtype=np.float32) for c in counts]
    )
    return sp.csr_matrix((data, indices, indptr), shape=(rows, cols))


def bench_param(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def time_it(label: str, repeats: int, fn: Callable[[], object]) -> None:
    fn()  # warmup outside the measured loop
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = time.perf_counter() - start
    print(f"{label:<44} {elapsed * 1000.0 / repeats:>9.3f} ms/iter")


def run_case(label: str, rows: int, cols: int, per_row: int, repeats: int) -> None:
    csr = make_csr(rows, cols, per_row)
    density = 100.0 * csr.nnz / (rows * cols)
    print(f"\n=== {label}: {rows}x{cols}, nnz={csr.nnz}, density={density:.2f}% ===")

    row_half = np.arange(rows // 2)
    time_it("subset rows (index), all cols", repeats, lambda: csr[row_half, :])

    col_half = np.arange(cols // 2)
    time_it(
        "subset rows + cols (both index)",
        repeats,
        lambda: csr[row_half, :][:, col_half],
    )

    col_stride = np.arange(0, cols, 3)
    time_it(
        "subset rows + strided cols",
        repeats,
        lambda: csr[row_half, :][:, col_stride],
    )

    col_reversed = np.arange(cols // 2)[::-1]
    time_it(
        "subset rows + reversed cols (non-monotonic)",
        repeats,
        lambda: csr[row_half, :][:, col_reversed],
    )


def main() -> None:
    repeats = bench_param("ANNDATA_BENCH_REPEATS", 10)
    print(f"scipy {scipy.__version__}, numpy {np.__version__}")
    run_case("small", 5_000, 2_000, 100, repeats)
    run_case("medium", 20_000, 10_000, 500, repeats)


if __name__ == "__main__":
    main()
