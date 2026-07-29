use crate::data::{isinstance_of_csc, isinstance_of_csr};

use anndata::data::{
    CsrNonCanonical, DynArray, DynCsrNonCanonical, DynIndSparseMatrix, DynSparseMatrix,
};
use ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::{IntoPyObjectExt, exceptions::PyTypeError, prelude::*};
use sprs::CsMatI;

macro_rules! proc_py_numeric {
    ($dtype:expr, $data:expr, $ty_anno:ident $(, $($gen:tt),*)?) => {
        match $dtype {
            "int8" => {
                let x: $ty_anno<i8 $(, $($gen),*)?> = $data;
                x.into()
            }
            "int16" => {
                let x: $ty_anno<i16 $(, $($gen),*)?> = $data;
                x.into()
            }
            "int32" => {
                let x: $ty_anno<i32 $(, $($gen),*)?> = $data;
                x.into()
            }
            "int64" => {
                let x: $ty_anno<i64 $(, $($gen),*)?> = $data;
                x.into()
            }
            "uint8" => {
                let x: $ty_anno<u8 $(, $($gen),*)?> = $data;
                x.into()
            }
            "uint16" => {
                let x: $ty_anno<u16 $(, $($gen),*)?> = $data;
                x.into()
            }
            "uint32" => {
                let x: $ty_anno<u32 $(, $($gen),*)?> = $data;
                x.into()
            }
            "uint64" => {
                let x: $ty_anno<u64 $(, $($gen),*)?> = $data;
                x.into()
            }
            "float32" => {
                let x: $ty_anno<f32 $(, $($gen),*)?> = $data;
                x.into()
            }
            "float64" => {
                let x: $ty_anno<f64 $(, $($gen),*)?> = $data;
                x.into()
            }
            "bool" => {
                let x: $ty_anno<bool $(, $($gen),*)?> = $data;
                x.into()
            }
            other => panic!("converting python type '{}' is not supported", other),
        }
    };
}

pub(super) fn to_array(ob: &Bound<'_, PyAny>) -> PyResult<DynArray> {
    let py = ob.py();
    let dtype = ob.getattr("dtype")?.getattr("char")?;
    let dtype = dtype.extract::<&str>()?;
    let arr = if dtype == "U" || dtype == "S" {
        ob.getattr("astype")?
            .call1(("object",))?
            .extract::<PyReadonlyArrayDyn<Py<PyAny>>>()?
            .as_array()
            .map(|x| x.extract::<String>(py).unwrap())
            .into()
    } else if dtype == "O" {
        ob.extract::<PyReadonlyArrayDyn<Py<PyAny>>>()?
            .as_array()
            .map(|x| x.extract::<String>(py).unwrap())
            .into()
    } else {
        let ty = ob.getattr("dtype")?.getattr("name")?;
        let ty = ty.extract::<&str>()?;
        proc_py_numeric!(
            ty,
            ob.extract::<PyReadonlyArrayDyn<_>>()?.to_owned_array(),
            ArrayD
        )
    };
    Ok(arr)
}

fn extract_indices_as_i32(arr: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    arr.call_method1("astype", ("int32",))?
        .extract::<Vec<i32>>()
}

fn extract_indices_as_i64(arr: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    arr.call_method1("astype", ("int64",))?
        .extract::<Vec<i64>>()
}

fn extract_indptr_as_u64(arr: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    arr.call_method1("astype", ("uint64",))?
        .extract::<Vec<u64>>()
}

fn extract_indices_as_u64(arr: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    arr.call_method1("astype", ("uint64",))?
        .extract::<Vec<u64>>()
}

pub(super) fn to_csr(ob: &Bound<'_, PyAny>) -> PyResult<DynIndSparseMatrix> {
    if !isinstance_of_csr(ob)? {
        return Err(PyTypeError::new_err("not a csr matrix"));
    }

    let shape: Vec<usize> = ob.getattr("shape")?.extract()?;
    let indices_ob = ob.getattr("indices")?;
    let indptr = extract_indptr_as_u64(&ob.getattr("indptr")?)?;
    let data_ty_ob = ob.getattr("data")?.getattr("dtype")?.getattr("name")?;
    let ty = data_ty_ob.extract::<&str>()?;
    let indices_ty_ob = indices_ob.getattr("dtype")?.getattr("name")?;
    let indices_ty = indices_ty_ob.extract::<&str>()?;

    if indices_ty == "int32" {
        let indices = extract_indices_as_i32(&indices_ob)?;
        let csr = proc_py_numeric!(
            ty,
            CsMatI::new(
                (shape[0], shape[1]),
                indptr,
                indices,
                ob.getattr("data")?
                    .extract::<PyReadonlyArrayDyn<_>>()?
                    .to_vec()
                    .unwrap()
            ),
            CsMatI,
            i32,
            u64
        );
        Ok(DynIndSparseMatrix::I32(csr))
    } else {
        let indices = extract_indices_as_i64(&indices_ob)?;
        let csr = proc_py_numeric!(
            ty,
            CsMatI::new(
                (shape[0], shape[1]),
                indptr,
                indices,
                ob.getattr("data")?
                    .extract::<PyReadonlyArrayDyn<_>>()?
                    .to_vec()
                    .unwrap()
            ),
            CsMatI,
            i64,
            u64
        );
        Ok(DynIndSparseMatrix::I64(csr))
    }
}

pub(super) fn to_csr_noncanonical(ob: &Bound<'_, PyAny>) -> PyResult<DynCsrNonCanonical> {
    if !isinstance_of_csr(ob)? {
        return Err(PyTypeError::new_err("not a csr matrix"));
    }

    let shape: Vec<usize> = ob.getattr("shape")?.extract()?;
    let indices = extract_indices_as_u64(&ob.getattr("indices")?)?;
    let indptr = extract_indptr_as_u64(&ob.getattr("indptr")?)?;
    let ty_ob = ob.getattr("data")?.getattr("dtype")?.getattr("name")?;
    let ty = ty_ob.extract::<&str>()?;

    let csr = proc_py_numeric!(
        ty,
        CsrNonCanonical::from_csr_data(
            shape[0],
            shape[1],
            indptr,
            indices,
            ob.getattr("data")?
                .extract::<PyReadonlyArrayDyn<_>>()?
                .to_vec()
                .unwrap()
        ),
        CsrNonCanonical
    );
    Ok(csr)
}

pub(super) fn to_csc(ob: &Bound<'_, PyAny>) -> PyResult<DynIndSparseMatrix> {
    if !isinstance_of_csc(ob)? {
        return Err(PyTypeError::new_err("not a csc matrix"));
    }

    let shape: Vec<usize> = ob.getattr("shape")?.extract()?;
    let indices_ob = ob.getattr("indices")?;
    let indptr = extract_indptr_as_u64(&ob.getattr("indptr")?)?;
    let data_ty_ob = ob.getattr("data")?.getattr("dtype")?.getattr("name")?;
    let ty = data_ty_ob.extract::<&str>()?;
    let indices_ty_ob = indices_ob.getattr("dtype")?.getattr("name")?;
    let indices_ty = indices_ty_ob.extract::<&str>()?;

    if indices_ty == "int32" {
        let indices = extract_indices_as_i32(&indices_ob)?;
        let csc = proc_py_numeric!(
            ty,
            CsMatI::new_csc(
                (shape[0], shape[1]),
                indptr,
                indices,
                ob.getattr("data")?
                    .extract::<PyReadonlyArrayDyn<_>>()?
                    .to_vec()
                    .unwrap()
            ),
            CsMatI,
            i32,
            u64
        );
        Ok(DynIndSparseMatrix::I32(csc))
    } else {
        let indices = extract_indices_as_i64(&indices_ob)?;
        let csc = proc_py_numeric!(
            ty,
            CsMatI::new_csc(
                (shape[0], shape[1]),
                indptr,
                indices,
                ob.getattr("data")?
                    .extract::<PyReadonlyArrayDyn<_>>()?
                    .to_vec()
                    .unwrap()
            ),
            CsMatI,
            i64,
            u64
        );
        Ok(DynIndSparseMatrix::I64(csc))
    }
}

pub(super) fn arr_to_py<'py>(arr: DynArray, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let res = match arr {
        DynArray::I8(arr) => arr.into_pyarray(py).into_any(),
        DynArray::I16(arr) => arr.into_pyarray(py).into_any(),
        DynArray::I32(arr) => arr.into_pyarray(py).into_any(),
        DynArray::I64(arr) => arr.into_pyarray(py).into_any(),
        DynArray::U8(arr) => arr.into_pyarray(py).into_any(),
        DynArray::U16(arr) => arr.into_pyarray(py).into_any(),
        DynArray::U32(arr) => arr.into_pyarray(py).into_any(),
        DynArray::U64(arr) => arr.into_pyarray(py).into_any(),
        DynArray::F32(arr) => arr.into_pyarray(py).into_any(),
        DynArray::F64(arr) => arr.into_pyarray(py).into_any(),
        DynArray::Bool(arr) => arr.into_pyarray(py).into_any(),
        DynArray::String(arr) => arr
            .mapv(|x| x.into_py_any(py).unwrap())
            .into_pyarray(py)
            .into_any(),
    };
    Ok(res)
}

macro_rules! match_inner {
    ($csr:expr, $py:expr, $helper:ident) => {
        match $csr {
            DynSparseMatrix::I8(m) => $helper(m, $py),
            DynSparseMatrix::I16(m) => $helper(m, $py),
            DynSparseMatrix::I32(m) => $helper(m, $py),
            DynSparseMatrix::I64(m) => $helper(m, $py),
            DynSparseMatrix::U8(m) => $helper(m, $py),
            DynSparseMatrix::U16(m) => $helper(m, $py),
            DynSparseMatrix::U32(m) => $helper(m, $py),
            DynSparseMatrix::U64(m) => $helper(m, $py),
            DynSparseMatrix::F32(m) => $helper(m, $py),
            DynSparseMatrix::F64(m) => $helper(m, $py),
            DynSparseMatrix::Bool(m) => $helper(m, $py),
            DynSparseMatrix::String(_) => todo!(),
        }
    };
}

macro_rules! dyn_sparse_to_py {
    ($csr:expr, $py:expr, $helper:ident) => {
        match $csr {
            DynIndSparseMatrix::I16(csr) => match_inner!(csr, $py, $helper),
            DynIndSparseMatrix::I32(csr) => match_inner!(csr, $py, $helper),
            DynIndSparseMatrix::I64(csr) => match_inner!(csr, $py, $helper),
            DynIndSparseMatrix::U16(csr) => match_inner!(csr, $py, $helper),
            DynIndSparseMatrix::U32(csr) => match_inner!(csr, $py, $helper),
            DynIndSparseMatrix::U64(csr) => match_inner!(csr, $py, $helper),
        }
    };
}

pub(super) fn csr_to_py<'py>(
    csr: DynIndSparseMatrix,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    fn helper<'py, T: numpy::Element, Ix: numpy::Element + sprs::SpIndex>(
        csr: CsMatI<T, Ix, u64>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = csr.rows();
        let m = csr.cols();
        let (indptr, indices, data) = csr.into_raw_storage();
        let scipy = PyModule::import(py, "scipy.sparse")?;
        scipy.getattr("csr_matrix")?.call1((
            (
                data.into_pyarray(py),
                indices.into_pyarray(py),
                indptr.into_pyarray(py),
            ),
            (n, m),
        ))
    }
    dyn_sparse_to_py!(csr, py, helper)
}

pub(super) fn csr_noncanonical_to_py<'py>(
    csr: DynCsrNonCanonical,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    fn helper<'py, T: numpy::Element>(
        csr: CsrNonCanonical<T>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = csr.nrows();
        let m = csr.ncols();
        let (indptr, indices, data) = csr.disassemble();
        let scipy = PyModule::import(py, "scipy.sparse")?;
        scipy.getattr("csr_matrix")?.call1((
            (
                data.into_pyarray(py),
                indices.into_pyarray(py),
                indptr.into_pyarray(py),
            ),
            (n, m),
        ))
    }

    match csr {
        DynCsrNonCanonical::I8(csr) => helper(csr, py),
        DynCsrNonCanonical::I16(csr) => helper(csr, py),
        DynCsrNonCanonical::I32(csr) => helper(csr, py),
        DynCsrNonCanonical::I64(csr) => helper(csr, py),
        DynCsrNonCanonical::U8(csr) => helper(csr, py),
        DynCsrNonCanonical::U16(csr) => helper(csr, py),
        DynCsrNonCanonical::U32(csr) => helper(csr, py),
        DynCsrNonCanonical::U64(csr) => helper(csr, py),
        DynCsrNonCanonical::F32(csr) => helper(csr, py),
        DynCsrNonCanonical::F64(csr) => helper(csr, py),
        DynCsrNonCanonical::Bool(csr) => helper(csr, py),
        DynCsrNonCanonical::String(_) => todo!(),
    }
}

pub(super) fn csc_to_py<'py>(
    csc: DynIndSparseMatrix,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    fn helper<'py, T: numpy::Element, Ix: numpy::Element + sprs::SpIndex>(
        csc: CsMatI<T, Ix, u64>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = csc.rows();
        let m = csc.cols();
        let (indptr, indices, data) = csc.into_raw_storage();
        let scipy = PyModule::import(py, "scipy.sparse")?;
        scipy.getattr("csc_matrix")?.call1((
            (
                data.into_pyarray(py),
                indices.into_pyarray(py),
                indptr.into_pyarray(py),
            ),
            (n, m),
        ))
    }
    dyn_sparse_to_py!(csc, py, helper)
}
