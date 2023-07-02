//! Graph traversal functions using common search algorithms.
//!
//! Abstracts away the details of the search, allowing you to focus solely on the problem-specific details.
//!
//! Traversals visit each node once.
//!
//! Works on infinite graphs.
//!
//! The functions return an iterator over states, in visit order.
//! A state should include the current node in the graph, along with any necessary search-specific data such as the total cost to reach the node.
//!
//! To turn a traversal into a search, use `.find(goal)`, where `goal` is a predicate specifying the target node.
//! You can also `.filter(goal)` to obtain an iterator over any states (or nodes) satisfying the predicate.
//!
//! Each traversal function requires at least the following:
//!
//! * An initial state
//! * A function for generating all states adjacent to a given state
//! * A function for normalising a state, returning a unique identifier of the node it represents
//!   * Must return a type that is [`Eq`] and [`Hash`]
//!   * Should remove any search-specific data, such as total cost to reach the node
//!
//! When the length, or total cost, of the path to a target node is desired, and when `dijkstra` or `a_star` are being used,
//! it is often useful to keep track of the total cost to reach the node within the state itself.
//!
//! Similarly, when the full path from start to finish is desired, it is often useful to keep track of the nodes previously visited within the state.
//!
//! Data such as this should be removed when a state is normalised.
//!
//! # Examples
//!
//! ```
//! use search::bft;
//! use std::collections::HashMap;
//!
//! // To traverse a graph of this form with a breadth-first search:
//! // a --- b --- c
//! // | \       / |
//! // |  \     /  |
//! // |   \   /   |
//! // d --- e     f
//! let graph = HashMap::from([
//!     ('a', vec!['b', 'd', 'e']),
//!     ('b', vec!['a', 'c']),
//!     ('c', vec!['b', 'e', 'f']),
//!     ('d', vec!['a', 'e']),
//!     ('e', vec!['a', 'c', 'd']),
//!     ('f', vec!['c']),
//! ]);
//!
//! struct State {
//!     node: char,
//!     path: Vec<char>,
//! }
//!
//! impl State {
//!     fn adjacent(&self, graph: &HashMap<char, Vec<char>>) -> Vec<State> {
//!         let mut states = Vec::new();
//!         for &node in &graph[&self.node] {
//!             let mut path = self.path.clone();
//!             path.push(node);
//!             states.push(State { node, path });
//!         }
//!         states
//!     }
//! }
//!
//! let mut traverse = bft(
//!     State { node: 'a', path: vec!['a'] },
//!     |s| s.adjacent(&graph),
//!     |s| s.node,
//! );
//! let goal = traverse.find(|s| s.node == 'f').unwrap();
//!
//! assert_eq!(goal.node, 'f');
//! assert_eq!(goal.path, vec!['a', 'b', 'c', 'f']);
//! ```
//!
//! ```
//! use search::dijkstra;
//! use std::collections::HashMap;
//!
//! // To traverse a graph of this form with Dijkstra's algorithm:
//! // a -6- b -3- c
//! // | \       / |
//! // 1  5     4  7
//! // |   \   /   |
//! // d -2- e     f
//! let graph = HashMap::from([
//!     ('a', vec![('b', 6), ('d', 1), ('e', 5)]),
//!     ('b', vec![('a', 6), ('c', 3)]),
//!     ('c', vec![('b', 3), ('e', 4), ('f', 7)]),
//!     ('d', vec![('a', 1), ('e', 2)]),
//!     ('e', vec![('a', 5), ('c', 4), ('d', 2)]),
//!     ('f', vec![('c', 7)]),
//! ]);
//!
//! struct State {
//!     node: char,
//!     cost: u8,
//!     path: Vec<char>,
//! }
//!
//! impl State {
//!     fn adjacent(&self, graph: &HashMap<char, Vec<(char, u8)>>) -> Vec<State> {
//!         let mut states = Vec::new();
//!         for &(node, weight) in &graph[&self.node] {
//!             let mut path = self.path.clone();
//!             path.push(node);
//!             states.push(State { node, cost: self.cost + weight, path });
//!         }
//!         states
//!     }
//! }
//!
//! let mut traverse = dijkstra(
//!     State { node: 'a', cost: 0, path: vec!['a'] },
//!     |s| s.adjacent(&graph),
//!     |s| s.node,
//!     |s| s.cost,
//! );
//! let goal = traverse.find(|s| s.node == 'f').unwrap();
//!
//! assert_eq!(goal.node, 'f');
//! assert_eq!(goal.cost, 14);
//! assert_eq!(goal.path, vec!['a', 'd', 'e', 'c', 'f']);
//! ```

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, VecDeque},
    hash::Hash,
    marker::PhantomData,
    ops::Add,
};

use rustc_hash::FxHashSet as HashSet;

struct Traverse<S, Q, Af, A, Nf, N> {
    adjacent: Af,
    normalise: Nf,
    states: Q,
    visited: HashSet<N>,
    _phantom: PhantomData<(S, A)>,
}

impl<S, Q, Af, A, Nf, N> Iterator for Traverse<S, Q, Af, A, Nf, N>
where
    Q: Collection<S>,
    Af: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    Nf: FnMut(&S) -> N,
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
        Some(current)
    }
}

impl<S, Q, Af, A, Nf, N> Traverse<S, Q, Af, A, Nf, N>
where
    Q: Collection<S>,
    Af: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    Nf: FnMut(&S) -> N,
    N: Eq + Hash,
{
    fn new(start: S, mut states: Q, adjacent: Af, normalise: Nf) -> Traverse<S, Q, Af, A, Nf, N> {
        states.push(start);
        Traverse {
            adjacent,
            normalise,
            states,
            visited: HashSet::default(),
            _phantom: PhantomData,
        }
    }
}

/// A function for traversing a graph with a [breadth-first](https://en.wikipedia.org/wiki/Breadth-first_search) traversal.
///
/// Returns an iterator over states, in visit order.
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
///
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
/// struct State {
///     node: char,
///     cost: u8,
/// }
///
/// impl State {
///     fn adjacent(&self, graph: &HashMap<char, Vec<char>>) -> Vec<State> {
///         let mut states = Vec::new();
///         for &node in &graph[&self.node] {
///             states.push(State { node, cost: self.cost + 1 });
///         }
///         states
///     }
/// }
///
/// let mut traverse = bft(
///     State { node: 'a', cost: 0 },
///     |s| s.adjacent(&graph),
///     |s| s.node,
/// );
/// let goal = traverse.find(|s| s.node == 'f').unwrap();
///
/// assert_eq!(goal.node, 'f');
/// assert_eq!(goal.cost, 3);
///
/// // To find the shortest path to a target node:
///
/// struct PathState {
///     node: char,
///     path: Vec<char>,
/// }
///
/// impl PathState {
///     fn adjacent(&self, graph: &HashMap<char, Vec<char>>) -> Vec<PathState> {
///         let mut states = Vec::new();
///         for &node in &graph[&self.node] {
///             let mut path = self.path.clone();
///             path.push(node);
///             states.push(PathState { node, path });
///         }
///         states
///     }
/// }
///
/// let mut traverse = bft(
///     PathState { node: 'a', path: vec!['a'] },
///     |s| s.adjacent(&graph),
///     |s| s.node,
/// );
/// let goal = traverse.find(|s| s.node == 'f').unwrap();
///
/// assert_eq!(goal.node, 'f');
/// assert_eq!(goal.path, vec!['a', 'b', 'c', 'f']);
/// ```
pub fn bft<S, Af, A, Nf, N>(start: S, adjacent: Af, normalise: Nf) -> impl Iterator<Item = S>
where
    Af: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    Nf: FnMut(&S) -> N,
    N: Eq + Hash,
{
    Traverse::new(start, VecDeque::new(), adjacent, normalise)
}

/// A function for traversing a graph with a [depth-first](https://en.wikipedia.org/wiki/Depth-first_search) traversal.
///
/// Returns an iterator over states, in visit order.
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
///
/// assert_eq!(traverse.next(), Some('a'));
/// assert_eq!(traverse.next(), Some('e'));
/// assert_eq!(traverse.next(), Some('d'));
/// assert_eq!(traverse.next(), Some('c'));
/// assert_eq!(traverse.next(), Some('f'));
/// assert_eq!(traverse.next(), Some('b'));
/// assert_eq!(traverse.next(), None);
///
/// // To find a path to a target node:
///
/// struct State {
///     node: char,
///     path: Vec<char>,
/// }
///
/// impl State {
///     fn adjacent(&self, graph: &HashMap<char, Vec<char>>) -> Vec<State> {
///         let mut states = Vec::new();
///         for &node in &graph[&self.node] {
///             let mut path = self.path.clone();
///             path.push(node);
///             states.push(State { node, path });
///         }
///         states
///     }
/// }
///
/// let mut traverse = dft(
///     State { node: 'a', path: vec!['a'] },
///     |s| s.adjacent(&graph),
///     |s| s.node,
/// );
/// let goal = traverse.find(|s| s.node == 'f').unwrap();
///
/// assert_eq!(goal.node, 'f');
/// assert_eq!(goal.path, vec!['a', 'e', 'c', 'f']);
/// ```
pub fn dft<S, Af, A, Nf, N>(start: S, adjacent: Af, normalise: Nf) -> impl Iterator<Item = S>
where
    Af: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    Nf: FnMut(&S) -> N,
    N: Eq + Hash,
{
    Traverse::new(start, Vec::new(), adjacent, normalise)
}

/// A function for traversing a weighted graph using [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm).
///
/// Returns an iterator over states, in visit order.
///
/// It is often useful to keep track of the total cost to reach the node within the state itself.
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
/// #[derive(PartialEq, Debug)]
/// struct State {
///     node: char,
///     cost: u8,
/// }
///
/// impl State {
///     fn adjacent(&self, graph: &HashMap<char, Vec<(char, u8)>>) -> Vec<State> {
///         let mut states = Vec::new();
///         for &(node, weight) in &graph[&self.node] {
///             states.push(State { node, cost: self.cost + weight });
///         }
///         states
///     }
/// }
///
/// let mut traverse = dijkstra(
///     State { node: 'a', cost: 0 },
///     |s| s.adjacent(&graph),
///     |s| s.node,
///     |s| s.cost,
/// );
///
/// assert_eq!(traverse.next(), Some(State { node: 'a', cost: 0 }));
/// assert_eq!(traverse.next(), Some(State { node: 'd', cost: 1 }));
/// assert_eq!(traverse.next(), Some(State { node: 'e', cost: 3 }));
/// assert_eq!(traverse.next(), Some(State { node: 'b', cost: 6 }));
/// assert_eq!(traverse.next(), Some(State { node: 'c', cost: 7 }));
/// assert_eq!(traverse.next(), Some(State { node: 'f', cost: 14 }));
/// assert_eq!(traverse.next(), None);
///
/// // To find the cost of the shortest path to a target node:
///
/// let mut traverse = dijkstra(
///     State { node: 'a', cost: 0 },
///     |s| s.adjacent(&graph),
///     |s| s.node,
///     |s| s.cost,
/// );
/// let goal = traverse.find(|s| s.node == 'f').unwrap();
///
/// assert_eq!(goal.node, 'f');
/// assert_eq!(goal.cost, 14);
///
/// // To find the shortest path to a target node:
///
/// struct PathState {
///     node: char,
///     cost: u8,
///     path: Vec<char>,
/// }
///
/// impl PathState {
///     fn adjacent(&self, graph: &HashMap<char, Vec<(char, u8)>>) -> Vec<PathState> {
///         let mut states = Vec::new();
///         for &(node, weight) in &graph[&self.node] {
///             let mut path = self.path.clone();
///             path.push(node);
///             states.push(PathState { node, cost: self.cost + weight, path });
///         }
///         states
///     }
/// }
///
/// let mut traverse = dijkstra(
///     PathState { node: 'a', cost: 0, path: vec!['a'] },
///     |s| s.adjacent(&graph),
///     |s| s.node,
///     |s| s.cost,
/// );
/// let goal = traverse.find(|s| s.node == 'f').unwrap();
///
/// assert_eq!(goal.node, 'f');
/// assert_eq!(goal.path, vec!['a', 'd', 'e', 'c', 'f']);
/// ```
pub fn dijkstra<S, Af, A, Nf, N, Cf, P>(
    start: S,
    adjacent: Af,
    normalise: Nf,
    cost: Cf,
) -> impl Iterator<Item = S>
where
    Af: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    Nf: FnMut(&S) -> N,
    N: Eq + Hash,
    Cf: FnMut(&S) -> P,
    P: Ord,
{
    Traverse::new(start, PriorityQueue::new(cost), adjacent, normalise)
}

/// A function for traversing a weighted graph using [A* search](https://en.wikipedia.org/wiki/A*_search_algorithm) algorithm.
///
/// Returns an iterator over states, in visit order.
///
/// It is often useful to keep track of the total cost to reach the node within the state itself.
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
/// struct State {
///     x: i8,
///     y: i8,
///     cost: u8,
/// }
///
/// impl State {
///     fn adjacent(&self, obstacles: &HashSet<(i8, i8)>) -> Vec<State> {
///         let mut states = Vec::new();
///         for offset in [(1, 0), (0, -1), (-1, 0), (0, 1)] {
///             let (x, y) = (self.x + offset.0, self.y + offset.1);
///             if !obstacles.contains(&(x, y)) {
///                 states.push(State { x, y, cost: self.cost + 1 });
///             }
///         }
///         states
///     }
///
///     fn heuristic(&self, goal: (i8, i8)) -> u8 {
///         self.x.abs_diff(goal.0) + self.y.abs_diff(goal.1)
///     }
/// }
///
/// let mut traverse = a_star(
///     State { x: 0, y: 4, cost: 0 },
///     |s| s.adjacent(&obstacles),
///     |s| (s.x, s.y),
///     |s| s.cost,
///     |s| s.heuristic(goal),
/// );
/// let goal = traverse.find(|s| (s.x, s.y) == goal).unwrap();
///
/// assert_eq!(goal.cost, 8);
/// ```
pub fn a_star<S, Af, A, Nf, N, Cf, Hf, P>(
    start: S,
    adjacent: Af,
    normalise: Nf,
    mut cost: Cf,
    mut heuristic: Hf,
) -> impl Iterator<Item = S>
where
    Af: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    Nf: FnMut(&S) -> N,
    N: Eq + Hash,
    Cf: FnMut(&S) -> P,
    Hf: FnMut(&S) -> P,
    P: Add,
    <P as Add>::Output: Ord,
{
    dijkstra(start, adjacent, normalise, move |s| cost(s) + heuristic(s))
}

trait Collection<S> {
    fn push(&mut self, state: S);

    fn pop(&mut self) -> Option<S>;
}

impl<S> Collection<S> for Vec<S> {
    fn push(&mut self, state: S) {
        self.push(state);
    }

    fn pop(&mut self) -> Option<S> {
        self.pop()
    }
}

impl<S> Collection<S> for VecDeque<S> {
    fn push(&mut self, state: S) {
        self.push_back(state);
    }

    fn pop(&mut self) -> Option<S> {
        self.pop_front()
    }
}

struct PriorityQueue<S, Pf, P> {
    heap: BinaryHeap<PriorityState<S, P>>,
    priority: Pf,
}

impl<S, Pf, P> PriorityQueue<S, Pf, P>
where
    Pf: FnMut(&S) -> P,
    P: Ord,
{
    fn new(priority: Pf) -> PriorityQueue<S, Pf, P> {
        PriorityQueue {
            heap: BinaryHeap::new(),
            priority,
        }
    }
}

impl<S, Pf, P> Collection<S> for PriorityQueue<S, Pf, P>
where
    Pf: FnMut(&S) -> P,
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
        Some(self.cmp(other))
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
