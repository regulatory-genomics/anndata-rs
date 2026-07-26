use anyhow::{Result, bail};
use itertools::Itertools;
use ndarray::{Array1, Array2, IxDyn, Slice, SliceInfo, SliceInfoElem};
use serde_json::Value;
use smallvec::{SmallVec, smallvec};
use std::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo};

/// A structure that represents a shape, internally represented as a small vector.
///
/// # Examples
/// ```
/// use anndata::data::Shape;
///
/// let shape = Shape::from(vec![3, 4, 5]);
/// assert_eq!(shape.ndim(), 3);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(SmallVec<[usize; 3]>);

impl Shape {
    /// Returns the number of dimensions in the shape.
    pub fn ndim(&self) -> usize {
        self.0.len()
    }
}

impl From<Shape> for Value {
    fn from(val: Shape) -> Self {
        Value::Array(val.0.into_iter().map(|x| x.into()).collect())
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.0.as_slice().iter().map(|x| x.to_string()).join(" x ")
        )
    }
}

impl AsRef<[usize]> for Shape {
    /// Provides a reference to the underlying shape data as a slice of `usize`.
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}

impl AsMut<[usize]> for Shape {
    /// Provides a mutable reference to the underlying shape data as a slice of `usize`.
    fn as_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self(SmallVec::from_vec(shape))
    }
}

impl From<(usize, usize)> for Shape {
    fn from(shape: (usize, usize)) -> Self {
        Self(smallvec![shape.0, shape.1])
    }
}

impl From<&[usize]> for Shape {
    fn from(shape: &[usize]) -> Self {
        Self(SmallVec::from_slice(shape))
    }
}

impl From<usize> for Shape {
    fn from(shape: usize) -> Self {
        Self(smallvec![shape])
    }
}

impl FromIterator<usize> for Shape {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self(SmallVec::from_iter(iter))
    }
}

/// A structure that represents a multidimensional selection, used for reading and writing to a container.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SelectInfo(pub Vec<SelectInfoElem>);

impl AsRef<[SelectInfoElem]> for SelectInfo {
    fn as_ref(&self) -> &[SelectInfoElem] {
        &self.0
    }
}

impl FromIterator<SelectInfoElem> for SelectInfo {
    /// Constructs `SelectInfo` from an iterator of `SelectInfoElem`.
    fn from_iter<T: IntoIterator<Item = SelectInfoElem>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl FromIterator<Slice> for SelectInfo {
    /// Constructs `SelectInfo` from an iterator of slices.
    fn from_iter<T: IntoIterator<Item = Slice>>(iter: T) -> Self {
        Self(iter.into_iter().map(SelectInfoElem::Slice).collect())
    }
}

impl<'a> FromIterator<&'a Slice> for SelectInfo {
    fn from_iter<T: IntoIterator<Item = &'a Slice>>(iter: T) -> Self {
        Self(
            iter.into_iter()
                .map(|x| SelectInfoElem::Slice(*x))
                .collect(),
        )
    }
}

impl TryInto<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> for SelectInfo {
    type Error = anyhow::Error;

    /// Attempts to convert `SelectInfo` into a `SliceInfo` object. Returns an error if conversion fails.
    fn try_into(self) -> Result<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> {
        let elems: Result<Vec<_>> = self
            .0
            .into_iter()
            .map(|e| match e {
                SelectInfoElem::Slice(s) => Ok(SliceInfoElem::from(s)),
                _ => bail!("Cannot convert SelectInfo to SliceInfo"),
            })
            .collect();
        let slice = SliceInfo::try_from(elems?)?;
        Ok(slice)
    }
}

impl SelectInfo {
    /// Creates a new `SelectInfo` with a specified number of full slices.
    ///
    /// # Parameters
    /// - `n`: The number of full slices to include.
    ///
    /// # Returns
    /// `SelectInfo` with `n` full slices.
    pub fn full_slice(n: usize) -> Self {
        Self(vec![SelectInfoElem::Slice(SLICE_FULL); n])
    }
}

/// Enum representing different types of selection elements for indexing and slicing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectInfoElem {
    Index(Vec<usize>),
    Slice(Slice),
}

impl FromIterator<usize> for SelectInfoElem {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self::Index(iter.into_iter().collect())
    }
}

impl From<Slice> for SelectInfoElem {
    fn from(x: Slice) -> Self {
        Self::Slice(x)
    }
}

impl From<usize> for SelectInfoElem {
    fn from(x: usize) -> Self {
        Self::Index(vec![x])
    }
}

impl From<&[usize]> for SelectInfoElem {
    fn from(x: &[usize]) -> Self {
        Self::Index(x.to_vec())
    }
}

impl From<Vec<usize>> for SelectInfoElem {
    fn from(x: Vec<usize>) -> Self {
        Self::Index(x)
    }
}

impl From<&Vec<usize>> for SelectInfoElem {
    fn from(x: &Vec<usize>) -> Self {
        x.as_slice().into()
    }
}

impl From<Array1<usize>> for SelectInfoElem {
    fn from(x: Array1<usize>) -> Self {
        Self::Index(x.to_vec())
    }
}

impl From<&Array1<usize>> for SelectInfoElem {
    fn from(x: &Array1<usize>) -> Self {
        Self::Index(x.to_vec())
    }
}

impl From<Range<usize>> for SelectInfoElem {
    fn from(x: Range<usize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<Range<isize>> for SelectInfoElem {
    fn from(x: Range<isize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<Range<i32>> for SelectInfoElem {
    fn from(x: Range<i32>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeFull> for SelectInfoElem {
    fn from(x: RangeFull) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeFrom<usize>> for SelectInfoElem {
    fn from(x: RangeFrom<usize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeFrom<isize>> for SelectInfoElem {
    fn from(x: RangeFrom<isize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeFrom<i32>> for SelectInfoElem {
    fn from(x: RangeFrom<i32>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeTo<usize>> for SelectInfoElem {
    fn from(x: RangeTo<usize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeTo<isize>> for SelectInfoElem {
    fn from(x: RangeTo<isize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeTo<i32>> for SelectInfoElem {
    fn from(x: RangeTo<i32>) -> Self {
        Self::Slice(x.into())
    }
}

impl AsRef<SelectInfoElem> for SelectInfoElem {
    fn as_ref(&self) -> &SelectInfoElem {
        self
    }
}

impl SelectInfoElem {
    /// Will panic if the index is out of bounds.
    pub fn bound_check(&self, bound: usize) -> Result<()> {
        match self {
            SelectInfoElem::Index(index) => index.iter().try_for_each(|i| {
                if *i >= bound {
                    bail!("index out of bounds: {i} >= {bound}")
                } else {
                    Ok(())
                }
            }),
            SelectInfoElem::Slice(slice) => slice.end.map_or(Ok(()), |end| {
                if end > bound as isize {
                    bail!("slice end out of bounds: {end} >= {bound}")
                } else {
                    Ok(())
                }
            }),
        }
    }

    pub fn is_index(&self) -> bool {
        matches!(self, SelectInfoElem::Index(_))
    }

    pub fn is_slice(&self) -> bool {
        matches!(self, SelectInfoElem::Slice(_))
    }

    pub fn empty() -> Self {
        SelectInfoElem::Index(Vec::new())
    }

    pub fn full() -> Self {
        SelectInfoElem::Slice(Slice {
            start: 0,
            end: None,
            step: 1,
        })
    }

    pub(crate) fn set_axis<'a>(
        &'a self,
        axis: usize,
        ndim: usize,
        fill: &'a Self,
    ) -> SmallVec<[&'a Self; 3]> {
        let mut slice = smallvec![fill; ndim];
        slice[axis] = self;
        slice
    }

    pub fn is_full(&self) -> bool {
        matches!(
            self,
            SelectInfoElem::Slice(Slice {
                start: 0,
                end: None,
                step: 1
            })
        )
    }
}

/// `SelectInfoBounds` represents bounds-aware selection information.
/// It includes an input shape and a list of selection elements with bound information for each axis.
///
/// # Example
/// ```
/// use anndata::data::{Shape, SelectInfoElem, SelectInfoBounds};
/// use ndarray::Slice;
///
/// let shape = Shape::from(vec![5, 10]);
/// let select = SelectInfoBounds::new(&[SelectInfoElem::Slice(Slice { start: 0, end: Some(5), step: 1 })], &shape);
/// assert_eq!(select.in_shape(), shape);
/// ```
#[derive(Debug, Clone)]
pub struct SelectInfoBounds<'a> {
    input_shape: Shape,
    select: Vec<SelectInfoElemBounds<'a>>,
}

impl<'a> AsRef<[SelectInfoElemBounds<'a>]> for SelectInfoBounds<'a> {
    /// Provides a reference to the selection elements with bounds.
    fn as_ref(&self) -> &[SelectInfoElemBounds<'a>] {
        &self.select
    }
}

impl<'a> TryInto<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> for SelectInfoBounds<'a> {
    type Error = anyhow::Error;

    /// Attempts to convert `SelectInfoBounds` into a `SliceInfo`, retaining the bounded selection elements.
    fn try_into(self) -> Result<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> {
        let elems: Result<Vec<_>> = self
            .select
            .into_iter()
            .map(|e| match e {
                SelectInfoElemBounds::Slice(s) => Ok(s.into()),
                _ => bail!("Cannot convert SelectInfo to SliceInfo"),
            })
            .collect();
        let slice = SliceInfo::try_from(elems?)?;
        Ok(slice)
    }
}

impl<'a> SelectInfoBounds<'a> {
    /// Constructs a new `SelectInfoBounds` with a given selection and shape.
    ///
    /// # Parameters
    /// - `select`: Reference to a selection, which may include indices and slices.
    /// - `shape`: Shape of the container being indexed.
    pub fn new<S, E>(select: &'a S, shape: &Shape) -> Self
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem> + 'a,
    {
        let res: Vec<_> = select
            .as_ref()
            .iter()
            .zip(shape.as_ref())
            .map(|(sel, dim)| SelectInfoElemBounds::new(sel.as_ref(), *dim))
            .collect();
        Self {
            input_shape: shape.clone(),
            select: res,
        }
    }

    /// Returns the input shape.
    pub fn in_shape(&self) -> Shape {
        self.input_shape.clone()
    }

    /// Calculates and returns the output shape based on the selection bounds.
    pub fn out_shape(&self) -> Shape {
        Shape::from(self.select.iter().map(|x| x.len()).collect::<Vec<_>>())
    }

    /// Returns the total number of elements in the output shape.
    pub fn out_size(&self) -> usize {
        self.out_shape().as_ref().iter().product()
    }

    /// Returns the number of dimensions in the selection.
    pub fn ndim(&self) -> usize {
        self.select.len()
    }

    // Convert to a new slice that contains only the unique indices. A mapping for
    // getting back the original indices is also returned.
    /*
    pub fn to_unique(&self) -> (Self, Self) {
        let out_shape = self.out_shape();
        let (unique, mapping): (Vec<_>, Vec<_>) = self.select.iter().zip(out_shape.as_ref())
            .map(|(sel, dim)| match sel {
                SelectInfoElemBounds::Index(x) => {
                    let (unique, mapping) = unique_indices(x, *dim);
                    (unique.into(), mapping.into())
            }
            SelectInfoElemBounds::Slice(x) => todo!(),
        }).unzip();
        todo!()
    }
    */

    /// Attempts to convert the selection into indices. Fail if elements are all slices.
    ///
    /// # Returns
    /// An optional `Array2` of indices if conversion is possible.
    pub fn try_into_indices(&self) -> Option<Array2<usize>> {
        if self
            .select
            .iter()
            .all(|e| matches!(e, SelectInfoElemBounds::Slice(_)))
        {
            return None;
        }

        let shape = self.out_shape();
        let ncols = self.ndim();
        let nrows = self.out_size();

        let mut result: Array2<usize> = Array2::zeros((nrows, ncols));

        for c in 0..ncols {
            let n_repeat = shape.as_ref()[(c + 1)..].iter().product();
            match self.select[c] {
                SelectInfoElemBounds::Index(x) => {
                    let mut values = x
                        .iter()
                        .flat_map(|x| std::iter::repeat_n(x, n_repeat))
                        .cycle();
                    for r in 0..nrows {
                        result[[r, c]] = *values.next().unwrap();
                    }
                }
                SelectInfoElemBounds::Slice(SliceBounds { start, end, step }) => {
                    if step > 0 {
                        let mut values = (start..end)
                            .step_by(step as usize)
                            .flat_map(|x| std::iter::repeat_n(x, n_repeat))
                            .cycle();
                        for r in 0..nrows {
                            result[[r, c]] = values.next().unwrap();
                        }
                    } else {
                        let mut values = (start..end)
                            .step_by(step.unsigned_abs())
                            .rev()
                            .flat_map(|x| std::iter::repeat_n(x, n_repeat))
                            .cycle();
                        for r in 0..nrows {
                            result[[r, c]] = values.next().unwrap();
                        }
                    }
                }
            }
        }
        Some(result)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &SelectInfoElemBounds<'a>> + '_ {
        self.select.iter()
    }
}

/// Enum representing selection elements with bounds awareness.
#[derive(Debug, Clone)]
pub enum SelectInfoElemBounds<'a> {
    Index(&'a [usize]),
    Slice(SliceBounds),
}

impl<'a> SelectInfoElemBounds<'a> {
    /// Creates a new bound-aware selection element from a selection element and an axis bound.
    pub fn new<S: AsRef<SelectInfoElem>>(select: &'a S, bound: usize) -> Self {
        match select.as_ref() {
            SelectInfoElem::Index(idx) => Self::Index(idx.as_slice()),
            SelectInfoElem::Slice(slice) => Self::Slice(SliceBounds::new(slice, bound)),
        }
    }

    /// Returns the length of the selection element.
    pub fn len(&self) -> usize {
        match self {
            Self::Index(idx) => idx.len(),
            Self::Slice(slice) => slice.len(),
        }
    }

    /// Returns whether the selection element is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieves the index at the specified position.
    pub fn index(&self, i: usize) -> usize {
        match self {
            Self::Index(idx) => idx[i],
            Self::Slice(slice) => slice.index(i),
        }
    }

    /// Checks if the selection element represents a full slice of the axis.
    pub fn is_full(&self, bound: usize) -> bool {
        match self {
            Self::Slice(slice) => slice.start == 0 && slice.end == bound && slice.step == 1,
            Self::Index(indices) => {
                indices.len() == bound && indices.iter().enumerate().all(|(i, &x)| x == i)
            }
        }
    }

    /// Converts the selection element into a vector of indices.
    pub fn to_vec(&self) -> Vec<usize> {
        match self {
            Self::Index(idx) => idx.to_vec(),
            Self::Slice(slice) => {
                if slice.step > 0 {
                    (slice.start..slice.end)
                        .step_by(slice.step as usize)
                        .collect()
                } else {
                    (slice.start..slice.end)
                        .step_by(slice.step.unsigned_abs())
                        .rev()
                        .collect()
                }
            }
        }
    }

    /// Returns an iterator over the indices represented by the selection element.
    pub fn iter(&self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        match self {
            Self::Index(idx) => Box::new(idx.iter().copied()),
            Self::Slice(slice) => {
                if slice.step > 0 {
                    Box::new((slice.start..slice.end).step_by(slice.step as usize))
                } else {
                    Box::new(
                        (slice.start..slice.end)
                            .step_by(slice.step.unsigned_abs())
                            .rev(),
                    )
                }
            }
        }
    }
}

/// `SliceBounds` represents bounds-aware slicing information, holding the start, end, and step values.
#[derive(Debug, Copy, Clone)]
pub struct SliceBounds {
    pub start: usize,
    pub end: usize,
    pub step: isize,
}

impl From<SliceBounds> for SliceInfoElem {
    /// Converts `SliceBounds` into a `SliceInfoElem`.
    fn from(val: SliceBounds) -> Self {
        Slice {
            start: val.start as isize,
            end: Some(val.end as isize),
            step: val.step,
        }
        .into()
    }
}

impl SliceBounds {
    /// Constructs a new `SliceBounds` from a `Slice` and an axis bound.
    pub(crate) fn new(slice: &Slice, bound: usize) -> Self {
        fn convert(x: isize, d: usize) -> usize {
            if x < 0 {
                d.checked_add_signed(x).unwrap()
            } else {
                x as usize
            }
        }

        Self {
            start: convert(slice.start, bound),
            end: slice.end.map_or(bound, |x| convert(x, bound)),
            step: slice.step,
        }
    }

    pub(crate) fn len(&self) -> usize {
        let step = self.step.unsigned_abs();
        let span = self.end - self.start;
        if span == 0 { 0 } else { span.div_ceil(step) }
    }

    pub(crate) fn index(&self, i: usize) -> usize {
        if self.step > 0 {
            self.start + i * self.step as usize
        } else {
            self.end.checked_sub(1 + i * (-self.step) as usize).unwrap()
        }
    }
}

/// Constant for a full slice, covering all indices from 0 to the bound.
pub const SLICE_FULL: Slice = Slice {
    start: 0,
    end: None,
    step: 1,
};

/// find unique indices and return the mapping
///
/// Example:
///
/// (unique_idx, mapping) = unique_indices_sorted(ori_idx, upper_bound)
/// assert_eq!(ori_idx, mapping.iter().map(|x| unique_idx[*x]).collect::<Vec<usize>>())
fn _unique_indices_sorted(indices: &[usize], upper_bound: usize) -> (Vec<usize>, Vec<usize>) {
    let mut mask = vec![upper_bound; upper_bound];
    // Set the mask for the present indices
    for i in indices {
        mask[*i] = *i;
    }
    let unique = mask
        .iter()
        .filter(|x| **x != upper_bound)
        .copied()
        .collect();

    // Find the new order
    mask.iter_mut().fold(0, |acc, x| {
        if *x != upper_bound {
            *x = acc;
            acc + 1
        } else {
            acc
        }
    });

    // Get the mapping
    let mapping = indices.iter().map(|x| mask[*x]).collect();
    (unique, mapping)
}

/// Slice argument constructor.
///
/// `s![]` takes a list of ranges/slices/indices/new-axes, separated by comma,
/// with optional step sizes that are separated from the range by a semicolon.
///
/// # Example
///
/// # Negative *step*
///
/// The behavior of negative *step* arguments is most easily understood with
/// slicing as a two-step process:
///
/// 1. First, perform a slice with *range*.
///
/// 2. If *step* is positive, start with the front of the slice; if *step* is
///    negative, start with the back of the slice. Then, add *step* until
///    reaching the other end of the slice (inclusive).
///
/// An equivalent way to think about step 2 is, "If *step* is negative, reverse
/// the slice. Start at the front of the (possibly reversed) slice, and add
/// *step.abs()* until reaching the back of the slice (inclusive)."
///
/// For example,
///
/// ```
/// # use anndata::s;
/// #
/// # fn main() {
/// println!("{:?}", s![1..3 , ..]);
/// println!("{:?}", s![vec![1, 10, 3], ..]);
/// # }
/// ```
#[macro_export]
macro_rules! s{
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($crate::data::SelectInfoElem::from($x));
            )*
            $crate::data::SelectInfo(temp_vec)
        }

    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_indices(input: Vec<u16>) {
            let max = (*input.iter().max().unwrap_or(&0) as usize) + 1;
            let indices = input.into_iter().map(|x| x as usize).collect::<Vec<_>>();
            let sorted_expected = indices.iter().copied().unique().sorted().collect::<Vec<_>>();
            let (sorted, mapping) = _unique_indices_sorted(indices.as_slice(), max);
            assert_eq!(sorted, sorted_expected);
            assert_eq!(indices, mapping.iter().map(|x| sorted[*x]).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_basic() {
        assert_eq!(
            _unique_indices_sorted(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11),
            (
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
        );
    }
}
