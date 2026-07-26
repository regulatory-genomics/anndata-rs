use crate::data::{DynArray, DynCowArray, DynScalar};

use anyhow::{Result, bail};
use core::fmt::{Debug, Display, Formatter};
use ndarray::{ArrayD, CowArray, IxDyn};
use serde::{Deserialize, Serialize};

/// All data types that can be stored in an AnnData object.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum DataType {
    Array(ScalarType),
    CsrMatrix(ScalarType, ScalarType),
    CscMatrix(ScalarType, ScalarType),

    DataFrame,
    Mapping,
    Scalar(ScalarType),
    Categorical,
    NullableArray,
}

impl DataType {
    pub fn scalar_type(&self) -> Option<ScalarType> {
        match self {
            DataType::Array(t) => Some(*t),
            DataType::CsrMatrix(t, _) => Some(*t),
            DataType::CscMatrix(t, _) => Some(*t),
            DataType::Scalar(t) => Some(*t),
            _ => None,
        }
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::Array(t) => write!(f, "Array({t})"),
            DataType::Categorical => write!(f, "Categorical"),
            DataType::CsrMatrix(t, i) => write!(f, "CsrMatrix({t}, {i})"),
            DataType::CscMatrix(t, i) => write!(f, "CscMatrix({t}, {i})"),
            DataType::DataFrame => write!(f, "DataFrame"),
            DataType::Scalar(t) => write!(f, "Scalar({t})"),
            DataType::Mapping => write!(f, "Mapping"),
            DataType::NullableArray => write!(f, "Nullable array"),
        }
    }
}

/// All scalar types that are supported in an AnnData object.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ScalarType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
    String,
}

impl ScalarType {
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            ScalarType::I8
                | ScalarType::I16
                | ScalarType::I32
                | ScalarType::I64
                | ScalarType::U8
                | ScalarType::U16
                | ScalarType::U32
                | ScalarType::U64
                | ScalarType::F32
                | ScalarType::F64
        )
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            ScalarType::I8
                | ScalarType::I16
                | ScalarType::I32
                | ScalarType::I64
                | ScalarType::U8
                | ScalarType::U16
                | ScalarType::U32
                | ScalarType::U64
        )
    }

    pub fn is_floating(&self) -> bool {
        matches!(self, ScalarType::F32 | ScalarType::F64)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, ScalarType::Bool)
    }

    pub fn is_string(&self) -> bool {
        matches!(self, ScalarType::String)
    }
}

impl Display for ScalarType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarType::I8 => write!(f, "i8"),
            ScalarType::I16 => write!(f, "i16"),
            ScalarType::I32 => write!(f, "i32"),
            ScalarType::I64 => write!(f, "i64"),
            ScalarType::U8 => write!(f, "u8"),
            ScalarType::U16 => write!(f, "u16"),
            ScalarType::U32 => write!(f, "u32"),
            ScalarType::U64 => write!(f, "u64"),
            ScalarType::F32 => write!(f, "f32"),
            ScalarType::F64 => write!(f, "f64"),
            ScalarType::Bool => write!(f, "bool"),
            ScalarType::String => write!(f, "string"),
        }
    }
}

pub trait BackendData:
    Serialize + for<'a> Deserialize<'a> + Send + Sync + Clone + std::fmt::Debug + 'static
{
    /// Type of the data
    const DTYPE: ScalarType;

    /// Convert to opaque representation.
    fn as_dyn(&self) -> DynScalar;

    #[deprecated]
    #[allow(clippy::wrong_self_convention)]
    fn into_dyn(&self) -> DynScalar {
        self.as_dyn()
    }

    /// Convert to opaque array representation.
    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a>;

    /// Convert from opaque representation.
    fn from_dyn(x: DynScalar) -> Result<Self>;

    /// Convert from opaque array representation.
    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>>;
}

impl BackendData for i8 {
    const DTYPE: ScalarType = ScalarType::I8;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::I8(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::I8(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::I8(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i8")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::I8(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i8 array")
        }
    }
}

impl BackendData for i16 {
    const DTYPE: ScalarType = ScalarType::I16;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::I16(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::I16(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::I16(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i16")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::I16(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i16 array")
        }
    }
}

impl BackendData for i32 {
    const DTYPE: ScalarType = ScalarType::I32;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::I32(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::I32(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::I32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i32")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::I32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i32 array")
        }
    }
}

impl BackendData for i64 {
    const DTYPE: ScalarType = ScalarType::I64;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::I64(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::I64(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::I64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i64")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::I64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i64 array")
        }
    }
}

impl BackendData for u8 {
    const DTYPE: ScalarType = ScalarType::U8;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::U8(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::U8(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::U8(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u8")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::U8(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u8 array")
        }
    }
}

impl BackendData for u16 {
    const DTYPE: ScalarType = ScalarType::U16;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::U16(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::U16(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::U16(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u16")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::U16(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u16 array")
        }
    }
}

impl BackendData for u32 {
    const DTYPE: ScalarType = ScalarType::U32;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::U32(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::U32(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::U32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u32")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::U32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u32 array")
        }
    }
}

impl BackendData for u64 {
    const DTYPE: ScalarType = ScalarType::U64;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::U64(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::U64(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::U64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u64")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::U64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u64 array")
        }
    }
}

impl BackendData for f32 {
    const DTYPE: ScalarType = ScalarType::F32;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::F32(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::F32(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::F32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting f32")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::F32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting f32 array")
        }
    }
}

impl BackendData for f64 {
    const DTYPE: ScalarType = ScalarType::F64;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::F64(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::F64(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::F64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting f64")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::F64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting f64 array")
        }
    }
}

impl BackendData for String {
    const DTYPE: ScalarType = ScalarType::String;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::String(self.clone())
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::String(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::String(x) = x {
            Ok(x)
        } else {
            bail!("Expecting string")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::String(x) = x {
            Ok(x)
        } else {
            bail!("Expecting string array")
        }
    }
}

impl BackendData for bool {
    const DTYPE: ScalarType = ScalarType::Bool;

    fn as_dyn(&self) -> DynScalar {
        DynScalar::Bool(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::Bool(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::Bool(x) = x {
            Ok(x)
        } else {
            bail!("Expecting bool")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::Bool(x) = x {
            Ok(x)
        } else {
            bail!("Expecting bool array")
        }
    }
}
