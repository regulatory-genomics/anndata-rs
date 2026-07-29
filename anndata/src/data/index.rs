use crate::data::array::slice::{SelectInfoElem, SliceBounds};

use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::Slice;
use smallvec::SmallVec;
use std::borrow::Borrow;
use std::cmp::Eq;
use std::hash::Hash;
use std::ops::Deref;
use std::{collections::HashMap, ops::Range};

use super::SelectInfoElemBounds;

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Index {
    Intervals(NamedIntervals),
    List(List<String>),
    Range(Range<usize>),
}

impl std::cmp::PartialEq for Index {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Index::Intervals(a), Index::Intervals(b)) => a == b,
            (Index::List(a), Index::List(b)) => a == b,
            (Index::Range(a), Index::Range(b)) => a == b,
            _ => self.iter().zip(other.iter()).all(|(a, b)| a == b),
        }
    }
}

impl Index {
    pub fn empty() -> Self {
        Index::List(List::empty())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contain_duplicates(&self) -> bool {
        match self {
            Index::Intervals(_) => false,
            Index::List(list) => list.items.len() != list.index_map.len(),
            Index::Range(_) => false,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Index::Intervals(map) => map.len(),
            Index::List(list) => list.items.len(),
            Index::Range(range) => range.end - range.start,
        }
    }

    pub fn get_index(&self, key: &str) -> Option<usize> {
        match self {
            Index::Intervals(map) => {
                let query: SmallVec<[&str; 3]> = key.split(&['-', ':']).collect();
                map.get_index(
                    query[0],
                    (query[1].parse().unwrap(), query[2].parse().unwrap()),
                )
            }
            Index::List(list) => list.get_index(key),
            Index::Range(range) => {
                let i: usize = key.parse().unwrap();
                if i >= range.start && i < range.end {
                    Some(i - range.start)
                } else {
                    None
                }
            }
        }
    }

    pub fn select(&self, select: &SelectInfoElem) -> Self {
        match SelectInfoElemBounds::new(select, self.len()) {
            SelectInfoElemBounds::Slice(slice) => self.slice(slice.start, slice.end),
            SelectInfoElemBounds::Index(index) => {
                let vec = self.clone().into_vec();
                index.iter().map(|i| vec[*i].clone()).collect()
            }
        }
    }

    pub fn into_vec(self) -> Vec<String> {
        if let Index::List(list) = self {
            list.items
        } else {
            self.into_iter().collect()
        }
    }

    fn slice(&self, start: usize, end: usize) -> Self {
        match self {
            Index::Intervals(intervals) => {
                let (o1, i1) = intervals.accum_length.ix(&start);
                let (o2, i2) = intervals.accum_length.ix(&(end - 1));
                if o1 == o2 {
                    let (k, interval) = intervals.intervals.get_index(o1).unwrap();
                    std::iter::once((k.to_owned(), interval.slice(i1, i2 + 1))).collect()
                } else {
                    let (k1, interval1) = intervals.intervals.get_index(o1).unwrap();
                    let (k2, interval2) = intervals.intervals.get_index(o2).unwrap();
                    std::iter::once((k1.to_owned(), interval1.slice(i1, interval1.len())))
                        .chain(
                            intervals
                                .intervals
                                .iter()
                                .skip(o1 + 1)
                                .take(o2 - o1 - 1)
                                .map(|(k, v)| (k.to_owned(), v.clone())),
                        )
                        .chain(std::iter::once((k2.to_owned(), interval2.slice(0, i2 + 1))))
                        .collect()
                }
            }
            Index::List(list) => list.items[start..end].iter().cloned().collect(),
            Index::Range(r) => Index::Range(r.start + start..r.start + end),
        }
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = String> + '_> {
        match self {
            Index::List(list) => Box::new(list.items.iter().cloned()),
            _ => self.clone().into_iter(),
        }
    }
}

impl From<Vec<String>> for Index {
    fn from(names: Vec<String>) -> Self {
        names.into_iter().collect()
    }
}

impl From<usize> for Index {
    fn from(n: usize) -> Self {
        Index::Range(0..n)
    }
}

impl From<Range<usize>> for Index {
    fn from(range: Range<usize>) -> Self {
        Index::Range(range)
    }
}

impl FromIterator<String> for Index {
    fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
        let (items, index_map): (Vec<_>, HashMap<_, _>) = iter
            .into_iter()
            .enumerate()
            .map(|(i, x)| (x.clone(), (x, i)))
            .unzip();
        Self::List(List { items, index_map })
    }
}

impl<S: Into<String>> FromIterator<(S, Interval)> for Index {
    fn from_iter<T: IntoIterator<Item = (S, Interval)>>(iter: T) -> Self {
        Self::Intervals(NamedIntervals::from_iter(iter))
    }
}

impl IntoIterator for Index {
    type Item = String;
    type IntoIter = Box<dyn Iterator<Item = String>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Index::Intervals(map) => {
                Box::new(map.intervals.into_iter().flat_map(|(k, interval)| {
                    interval.map(move |(start, end)| format!("{k}:{start}-{end}"))
                }))
            }
            Index::List(list) => Box::new(list.items.into_iter()),
            Index::Range(range) => Box::new(range.map(|i| i.to_string())),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NamedIntervals {
    intervals: IndexMap<String, Interval>,
    accum_length: VecVecIndex,
}

impl Deref for NamedIntervals {
    type Target = IndexMap<String, Interval>;

    fn deref(&self) -> &Self::Target {
        &self.intervals
    }
}

impl NamedIntervals {
    pub fn get_index(&self, name: &str, interval: (usize, usize)) -> Option<usize> {
        self.intervals
            .get_full(name)
            .and_then(|(i, _, v)| v.get_index(interval).map(|x| x + self.accum_length.0[i]))
    }

    pub fn len(&self) -> usize {
        self.accum_length.len()
    }

    pub fn is_empty(&self) -> bool {
        self.accum_length.len() == 0
    }
}

impl<S: Into<String>> FromIterator<(S, Interval)> for NamedIntervals {
    fn from_iter<T: IntoIterator<Item = (S, Interval)>>(iter: T) -> Self {
        let mut intervals = IndexMap::new();
        let accum_length = iter
            .into_iter()
            .map(|(name, interval)| {
                let n = interval.len();
                intervals.insert(name.into(), interval);
                n
            })
            .collect();
        Self {
            intervals,
            accum_length,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Interval {
    pub start: usize,
    pub end: usize,
    pub size: usize,
    pub step: usize,
}

impl Interval {
    pub fn get_index(&self, interval: (usize, usize)) -> Option<usize> {
        if interval.0 >= self.start && interval.1 <= self.end {
            let (d, m) = num::integer::div_rem(interval.0 - self.start, self.step);
            if m != 0 {
                None
            } else if interval.1 - interval.0 == self.size || interval.1 == self.end {
                Some(d)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        num::integer::div_ceil(self.end - self.start, self.step)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    fn slice(&self, start: usize, end: usize) -> Self {
        let start = self.start + self.step * start;
        let end = self.start + self.step * end;
        Self {
            start: start.min(self.end),
            end: end.min(self.end),
            size: self.size,
            step: self.step,
        }
    }
}

impl Iterator for Interval {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let end = (self.start + self.size).min(self.end);
            let item = (self.start, end);
            self.start += self.step;
            Some(item)
        }
    }
}

#[derive(Clone, Debug)]
pub struct List<K> {
    pub items: Vec<K>,
    index_map: HashMap<K, usize>,
}

impl std::cmp::PartialEq for List<String> {
    fn eq(&self, other: &Self) -> bool {
        self.items == other.items
    }
}

impl<K: Hash + Eq> List<K> {
    pub fn empty() -> Self {
        Self {
            items: vec![],
            index_map: HashMap::new(),
        }
    }

    pub fn get_index<Q>(&self, item: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.index_map.get(item).cloned()
    }
}

/// This struct is used to perform index lookup for nested Vectors (vectors of vectors).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VecVecIndex(SmallVec<[usize; 96]>);

impl VecVecIndex {
    /// Find the outer and inner index for a given index corresponding to the
    /// flattened view.
    ///
    /// # Example
    ///
    /// let vec_of_vec = vec![vec![0, 1, 2], vec![3, 4], vec![5, 6]];
    /// let flatten_view = vec![0, 1, 2, 3, 4, 5, 6];
    /// let index = VecVecIndex::new(vec_of_vec);
    /// assert_eq!(index.ix(0), (0, 0));
    /// assert_eq!(index.ix(1), (0, 1));
    /// assert_eq!(index.ix(2), (0, 2));
    /// assert_eq!(index.ix(3), (1, 0));
    /// assert_eq!(index.ix(4), (1, 1));
    /// assert_eq!(index.ix(5), (2, 0));
    /// assert_eq!(index.ix(6), (2, 1));
    pub fn ix(&self, i: &usize) -> (usize, usize) {
        let j = self.outer_ix(i);
        (j, i - self.0[j])
    }

    /// The inverse of ix.
    pub fn _inv_ix(&self, idx: (usize, usize)) -> usize {
        self.0[idx.0] + idx.1
    }

    /// Find the outer index for a given index corresponding to the flattened view.
    pub fn outer_ix(&self, i: &usize) -> usize {
        match self.0.binary_search(i) {
            Ok(i_) => i_,
            Err(i_) => i_ - 1,
        }
    }

    pub fn split_slice(&self, slice: &Slice) -> HashMap<usize, SelectInfoElem> {
        let bounded = SliceBounds::new(slice, self.len());
        let (outer_start, inner_start) = self.ix(&bounded.start);
        let (outer_end, inner_end) = self.ix(&bounded.end);
        let mut res = HashMap::new();
        if outer_start == outer_end {
            res.insert(
                outer_start,
                Slice {
                    start: inner_start as isize,
                    end: Some(inner_end as isize),
                    step: slice.step,
                }
                .into(),
            );
        } else {
            res.insert(
                outer_start,
                Slice {
                    start: inner_start as isize,
                    end: None,
                    step: slice.step,
                }
                .into(),
            );
            res.insert(
                outer_end,
                Slice {
                    start: 0,
                    end: Some(inner_end as isize),
                    step: slice.step,
                }
                .into(),
            );
            for i in outer_start + 1..outer_end {
                res.insert(
                    i,
                    Slice {
                        start: 0,
                        end: None,
                        step: slice.step,
                    }
                    .into(),
                );
            }
        };
        res
    }

    fn split_indices(
        &self,
        indices: &[usize],
    ) -> (HashMap<usize, SelectInfoElem>, Option<Vec<usize>>) {
        let (new_indices, orders): (HashMap<usize, SelectInfoElem>, Vec<_>) = indices
            .iter()
            .map(|x| self.ix(x))
            .enumerate()
            .sorted_by_key(|(_, (x, _))| *x)
            .chunk_by(|(_, (x, _))| *x)
            .into_iter()
            .map(|(outer, inner)| {
                let (new_indices, order): (Vec<_>, Vec<_>) =
                    inner.map(|(i, (_, x))| (x, i)).unzip();
                ((outer, new_indices.into()), order)
            })
            .unzip();
        let order: Vec<_> = orders.into_iter().flatten().collect();
        if order.as_slice().windows(2).all(|w| w[1] - w[0] == 1) {
            (new_indices, None)
        } else {
            (new_indices, Some(order))
        }
    }

    /// Sort and split the indices.
    pub fn split_select(
        &self,
        select: &SelectInfoElem,
    ) -> (HashMap<usize, SelectInfoElem>, Option<Vec<usize>>) {
        match select {
            SelectInfoElem::Slice(slice) => (self.split_slice(slice), None),
            SelectInfoElem::Index(index) => self.split_indices(index.as_slice()),
        }
    }

    /// The total number of elements
    pub fn len(&self) -> usize {
        *self.0.last().unwrap_or(&0)
    }
}

impl FromIterator<usize> for VecVecIndex {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        let index: SmallVec<_> = std::iter::once(0)
            .chain(iter.into_iter().scan(0, |state, x| {
                *state += x;
                Some(*state)
            }))
            .collect();
        VecVecIndex(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::strategy::BoxedStrategy;

    fn index_strat(n: usize) -> BoxedStrategy<Index> {
        if n == 0 {
            Just(Index::empty()).boxed()
        } else {
            let list = (0..n).map(|i| format!("i_{i}")).collect();
            let range = n.into();
            let interval = (0..n).prop_flat_map(move |i| {
                (Just(i), (0..n - i)).prop_flat_map(move |(a, b)| {
                    let c = n - a - b;
                    [a, b, c]
                        .into_iter()
                        .filter(|x| *x != 0)
                        .map(interval_strat)
                        .collect::<Vec<_>>()
                        .prop_map(move |x| {
                            x.into_iter()
                                .enumerate()
                                .map(|(i, x)| (i.to_string(), x))
                                .collect::<Index>()
                        })
                })
            });
            prop_oneof![Just(list), Just(range), interval].boxed()
        }
    }

    fn interval_strat(n: usize) -> impl Strategy<Value = Interval> {
        (1_usize..100, 1_usize..100).prop_map(move |(size, step)| Interval {
            start: 0,
            end: n * step,
            size,
            step,
        })
    }

    #[test]
    fn test_example() {
        let index: Index = [
            (
                "0",
                Interval {
                    start: 0,
                    end: 2,
                    size: 1,
                    step: 1,
                },
            ),
            (
                "1",
                Interval {
                    start: 0,
                    end: 972,
                    size: 1,
                    step: 27,
                },
            ),
            (
                "2",
                Interval {
                    start: 0,
                    end: 1134,
                    size: 53,
                    step: 18,
                },
            ),
        ]
        .into_iter()
        .collect();

        assert_eq!(index.len(), 101);
        assert_eq!(index, index.select(&(0..101).into()));
        assert_eq!(
            [(
                "2",
                Interval {
                    start: 900,
                    end: 1062,
                    step: 18,
                    size: 53
                }
            )]
            .into_iter()
            .collect::<Index>(),
            index.select(&(88..97).into()),
        );
    }

    fn select_strat(n: usize) -> BoxedStrategy<SelectInfoElem> {
        if n == 0 {
            Just(Vec::new().into()).boxed()
        } else {
            let indices = proptest::collection::vec(0..n, 0..2 * n).prop_map(|i| i.into());
            let slice = (0..n).prop_flat_map(move |start| {
                (Just(start), (start + 1)..=n).prop_map(|(start, stop)| (start..stop).into())
            });
            prop_oneof![indices, slice,].boxed()
        }
    }

    #[test]
    fn test_index() {
        let index = (0_usize..500).prop_flat_map(|n| (Just(n), index_strat(n), select_strat(n)));
        proptest!(ProptestConfig::with_cases(256), |((n, i, slice) in index)| {
            prop_assert_eq!(i.len(), n);
            prop_assert_eq!(i.len(), i.clone().into_vec().len());

            prop_assert!(i.iter().enumerate().all(|(idx, x)| i.get_index(&x).unwrap() == idx));

            let out_len = SelectInfoElemBounds::new(&slice, n).len();
            let i_slice = i.select(&slice);
            prop_assert_eq!(i_slice.len(), out_len);
            prop_assert_eq!(i_slice.len(), i_slice.into_vec().len());
        });
    }
}
