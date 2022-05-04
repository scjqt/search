use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet, VecDeque},
    hash::Hash,
    marker::PhantomData,
    ops::Add,
};

struct Traverse<S, Q, AF, A, NF, N> {
    adjacent: AF,
    normalise: NF,
    states: Q,
    visited: HashSet<N>,
    _phantom: PhantomData<(S, A)>,
}

impl<S, Q, AF, A, NF, N> Iterator for Traverse<S, Q, AF, A, NF, N>
where
    Q: Collection<S>,
    AF: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    NF: FnMut(&S) -> N,
    N: Eq + Hash,
{
    type Item = S;

    fn next(&mut self) -> Option<Self::Item> {
        let mut current = self.states.pop()?;
        let mut normalised = (self.normalise)(&current);
        while self.visited.contains(&normalised) {
            current = self.states.pop()?;
            normalised = (self.normalise)(&current);
        }
        self.visited.insert(normalised);
        for state in (self.adjacent)(&current) {
            if !self.visited.contains(&(self.normalise)(&state)) {
                self.states.push(state);
            }
        }
        return Some(current);
    }
}

impl<S, Q, AF, A, NF, N> Traverse<S, Q, AF, A, NF, N>
where
    Q: Collection<S>,
    AF: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    NF: FnMut(&S) -> N,
    N: Eq + Hash,
{
    fn new(start: S, mut states: Q, adjacent: AF, normalise: NF) -> Traverse<S, Q, AF, A, NF, N> {
        states.push(start);
        Traverse {
            adjacent,
            normalise,
            states,
            visited: HashSet::new(),
            _phantom: PhantomData,
        }
    }
}

/// A function for traversing a graph with a breadth-first traversal, visiting each node once.
///
/// Requires:
/// * An initial state
/// * A function for generating all states adjacent to a given state
/// * A function for normalising a state, returning a unique identifier of the node it represents
///   * Must return a type that is [`Eq`] and [`Hash`]
///   * Should remove any search-specific data, such as total cost to reach the node
///
/// # Examples
///
/// ```
/// use search::bft;
/// use std::collections::HashMap;
///
/// // To traverse a graph of this form:
/// // a --- b --- c
/// // | \       / |
/// // |  \     /  |
/// // |   \   /   |
/// // d --- e     f
/// let graph = HashMap::from([
///     ('a', vec!['b', 'd', 'e']),
///     ('b', vec!['a', 'c']),
///     ('c', vec!['b', 'e', 'f']),
///     ('d', vec!['a', 'e']),
///     ('e', vec!['a', 'c', 'd']),
///     ('f', vec!['c']),
/// ]);
///
/// let mut traverse = bft('a', |x| (&graph[x]).clone(), |x| *x);
/// assert_eq!(traverse.next(), Some('a'));
/// assert_eq!(traverse.next(), Some('b'));
/// assert_eq!(traverse.next(), Some('d'));
/// assert_eq!(traverse.next(), Some('e'));
/// assert_eq!(traverse.next(), Some('c'));
/// assert_eq!(traverse.next(), Some('f'));
/// assert_eq!(traverse.next(), None);
///
/// // To find the length of the shortest path to a target node:
///
/// let length = bft(
///     ('a', 0),
///     |&(x, cost)| graph[&x].clone().into_iter().map(move |y| (y, cost + 1)),
///     |(x, _)| *x,
/// )
/// .find(|(x, _)| *x == 'f')
/// .unwrap()
/// .1;
/// assert_eq!(length, 3);
/// ```
pub fn bft<S, AF, A, NF, N>(start: S, adjacent: AF, normalise: NF) -> impl Iterator<Item = S>
where
    AF: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    NF: FnMut(&S) -> N,
    N: Eq + Hash,
{
    Traverse::new(start, Queue::new(), adjacent, normalise)
}

/// A function for traversing a graph with a depth-first traversal, visiting each node once.
///
/// Requires:
/// * An initial state
/// * A function for generating all states adjacent to a given state
/// * A function for normalising a state, returning a unique identifier of the node it represents
///   * Must return a type that is [`Eq`] and [`Hash`]
///   * Should remove any search-specific data, such as total cost to reach the node
///
/// # Examples
///
/// ```
/// use search::dft;
/// use std::collections::HashMap;
///
/// // To traverse a graph of this form:
/// // a --- b --- c
/// // | \       / |
/// // |  \     /  |
/// // |   \   /   |
/// // d --- e     f
/// let graph = HashMap::from([
///     ('a', vec!['b', 'd', 'e']),
///     ('b', vec!['a', 'c']),
///     ('c', vec!['b', 'e', 'f']),
///     ('d', vec!['a', 'e']),
///     ('e', vec!['a', 'c', 'd']),
///     ('f', vec!['c']),
/// ]);
///
/// let mut traverse = dft('a', |x| graph[x].clone(), |x| *x);
/// assert_eq!(traverse.next(), Some('a'));
/// assert_eq!(traverse.next(), Some('e'));
/// assert_eq!(traverse.next(), Some('d'));
/// assert_eq!(traverse.next(), Some('c'));
/// assert_eq!(traverse.next(), Some('f'));
/// assert_eq!(traverse.next(), Some('b'));
/// assert_eq!(traverse.next(), None);
/// ```
pub fn dft<S, AF, A, NF, N>(start: S, adjacent: AF, normalise: NF) -> impl Iterator<Item = S>
where
    AF: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    NF: FnMut(&S) -> N,
    N: Eq + Hash,
{
    Traverse::new(start, Stack::new(), adjacent, normalise)
}

/// A function for traversing a weighted graph using Dijkstra's algorithm, visiting each node once.
///
/// It is often useful to store the total cost required to reach the node within the state itself.
///
/// Requires:
/// * An initial state
///   * Usually starts with zero total cost
/// * A function for generating all states adjacent to a given state
///   * Should return states with updated total costs using the weightings of the graph
/// * A function for normalising a state, returning a unique identifier of the node it represents
///   * Must return a type that is [`Eq`] and [`Hash`]
///   * Should remove any search-specific data, such as total cost to reach the node
/// * A function that returns the total cost required to reach the node represented by a given state
///
/// # Examples
///
/// ```
/// use search::dijkstra;
/// use std::collections::HashMap;
///
/// // To traverse a graph of this form:
/// // a -6- b -3- c
/// // | \       / |
/// // 1  5     4  7
/// // |   \   /   |
/// // d -2- e     f
/// let graph = HashMap::from([
///     ('a', vec![('b', 6), ('d', 1), ('e', 5)]),
///     ('b', vec![('a', 6), ('c', 3)]),
///     ('c', vec![('b', 3), ('e', 4), ('f', 7)]),
///     ('d', vec![('a', 1), ('e', 2)]),
///     ('e', vec![('a', 5), ('c', 4), ('d', 2)]),
///     ('f', vec![('c', 7)]),
/// ]);
///
/// let mut traverse = dijkstra(
///     ('a', 0), // initial state
///     |&(x, cost)| { // adjacent states function
///         graph[&x]
///             .clone()
///             .into_iter()
///             .map(move |(y, w)| (y, cost + w))
///     },
///     |(x, _)| *x, // normalise function
///     |(_, cost)| *cost, // cost function
/// );
/// assert_eq!(traverse.next(), Some(('a', 0)));
/// assert_eq!(traverse.next(), Some(('d', 1)));
/// assert_eq!(traverse.next(), Some(('e', 3)));
/// assert_eq!(traverse.next(), Some(('b', 6)));
/// assert_eq!(traverse.next(), Some(('c', 7)));
/// assert_eq!(traverse.next(), Some(('f', 14)));
/// assert_eq!(traverse.next(), None);
///
/// // To search for a target node:
///
/// let cost = dijkstra(
///     ('a', 0),
///     |&(x, cost)| {
///         graph[&x]
///             .clone()
///             .into_iter()
///             .map(move |(y, w)| (y, cost + w))
///     },
///     |(x, _)| *x,
///     |(_, cost)| *cost,
/// )
/// .find(|(x, _)| *x == 'f')
/// .unwrap()
/// .1;
/// assert_eq!(cost, 14);
/// ```
pub fn dijkstra<S, AF, A, NF, N, CF, P>(
    start: S,
    adjacent: AF,
    normalise: NF,
    cost: CF,
) -> impl Iterator<Item = S>
where
    AF: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    NF: FnMut(&S) -> N,
    N: Eq + Hash,
    CF: FnMut(&S) -> P,
    P: Ord,
{
    Traverse::new(start, PriorityQueue::new(cost), adjacent, normalise)
}

/// A function for traversing a weighted graph using A* search algorithm, visiting each node once.
///
/// It is often useful to store the total cost required to reach the node within the state itself.
///
/// Requires:
/// * An initial state
///   * Usually starts with zero total cost
/// * A function for generating all states adjacent to a given state
///   * Should return states with updated total costs using the weightings of the graph
/// * A function for normalising a state, returning a unique identifier of the node it represents
///   * Must return a type that is [`Eq`] and [`Hash`]
///   * Should remove any search-specific data, such as total cost to reach the node
/// * A function that returns the total cost required to reach the node represented by a given state
/// * A heuristic function that returns an estimate of the total cost remaining to reach the goal
///   * Should be admissible, meaning it never overestimates the actual cost to reach the goal
///
/// # Examples
///
/// ```
/// use search::a_star;
/// use std::collections::HashSet;
///
/// // To quickly find the shortest distance in this grid from S to G,
/// // using Manhattan distance as a heuristic:
/// // #...G
/// // ..##.
/// // .....
/// // ....#
/// // S....
/// let obstacles = HashSet::from([(0, 0), (2, 1), (3, 1), (4, 3)]);
/// let goal = (4, 0);
///
/// let steps = a_star(
///     (0i8, 4i8, 0i8), // initial state
///     |&(x, y, cost)| { // adjacent states function
///         [(1, 0), (0, -1), (-1, 0), (0, 1)]
///             .into_iter()
///             .map(move |(x1, y1)| (x + x1, y + y1))
///             .filter(|pos| !obstacles.contains(pos))
///             .map(move |(x, y)| (x, y, cost + 1))
///     },
///     |(x, y, _)| (*x, *y), // normalise function
///     |(_, _, cost)| *cost, // cost function
///     |(x, y, _)| (goal.0 - *x).abs() + (goal.1 - *y).abs(), // heuristic function
/// )
/// .find(|(x, y, _)| (*x, *y) == goal)
/// .unwrap()
/// .2;
/// assert_eq!(steps, 8);
/// ```
pub fn a_star<S, AF, A, NF, N, CF, HF, P>(
    start: S,
    adjacent: AF,
    normalise: NF,
    mut cost: CF,
    mut heuristic: HF,
) -> impl Iterator<Item = S>
where
    AF: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    NF: FnMut(&S) -> N,
    N: Eq + Hash,
    CF: FnMut(&S) -> P,
    HF: FnMut(&S) -> P,
    P: Add,
    <P as Add>::Output: Ord,
{
    dijkstra(start, adjacent, normalise, move |s| cost(s) + heuristic(s))
}

trait Collection<S> {
    fn push(&mut self, state: S);

    fn pop(&mut self) -> Option<S>;
}

struct Stack<S> {
    stack: Vec<S>,
}

impl<S> Stack<S> {
    fn new() -> Stack<S> {
        Stack { stack: Vec::new() }
    }
}

impl<S> Collection<S> for Stack<S> {
    fn push(&mut self, state: S) {
        self.stack.push(state);
    }

    fn pop(&mut self) -> Option<S> {
        self.stack.pop()
    }
}

struct Queue<S> {
    queue: VecDeque<S>,
}

impl<S> Queue<S> {
    fn new() -> Queue<S> {
        Queue {
            queue: VecDeque::new(),
        }
    }
}

impl<S> Collection<S> for Queue<S> {
    fn push(&mut self, state: S) {
        self.queue.push_back(state);
    }

    fn pop(&mut self) -> Option<S> {
        self.queue.pop_front()
    }
}

struct PriorityQueue<S, PF, P> {
    heap: BinaryHeap<PriorityState<S, P>>,
    priority: PF,
}

impl<S, PF, P> PriorityQueue<S, PF, P>
where
    PF: FnMut(&S) -> P,
    P: Ord,
{
    fn new(priority: PF) -> PriorityQueue<S, PF, P> {
        PriorityQueue {
            heap: BinaryHeap::new(),
            priority,
        }
    }
}

impl<S, PF, P> Collection<S> for PriorityQueue<S, PF, P>
where
    PF: FnMut(&S) -> P,
    P: Ord,
{
    fn push(&mut self, state: S) {
        self.heap
            .push(PriorityState::new(state, &mut self.priority));
    }

    fn pop(&mut self) -> Option<S> {
        self.heap.pop().map(|p| p.state)
    }
}

struct PriorityState<S, P> {
    state: S,
    priority: P,
}

impl<S, P> PriorityState<S, P>
where
    P: Ord,
{
    fn new<C>(state: S, mut priority: C) -> PriorityState<S, P>
    where
        C: FnMut(&S) -> P,
    {
        let priority = priority(&state);
        PriorityState { state, priority }
    }
}

impl<S, P> Ord for PriorityState<S, P>
where
    P: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
    }
}

impl<S, P> PartialOrd for PriorityState<S, P>
where
    P: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<S, P> Eq for PriorityState<S, P> where P: Ord {}

impl<S, P> PartialEq for PriorityState<S, P>
where
    P: Ord,
{
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
