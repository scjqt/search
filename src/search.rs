use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Add;

pub struct Traversal<S, Q, A, I, F, N>
where
    Q: Collection<S>,
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
{
    adjacent: A,
    normalise: F,
    states: Q,
    visited: HashSet<N>,
    _phantom: PhantomData<S>,
}

impl<S, Q, A, I, F, N> Iterator for Traversal<S, Q, A, I, F, N>
where
    Q: Collection<S>,
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
{
    type Item = S;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(current) = self.states.pop() {
                let normalised = (self.normalise)(&current);
                if self.visited.contains(&normalised) {
                    continue;
                }
                self.visited.insert(normalised);
                for state in (self.adjacent)(&current) {
                    if !self.visited.contains(&(self.normalise)(&state)) {
                        self.states.push(state);
                    }
                }
                return Some(current);
            }
            return None;
        }
    }
}

pub fn bft<S, A, I, F, N>(start: S, adjacent: A, normalise: F) -> Traversal<S, Queue<S>, A, I, F, N>
where
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
{
    Traversal::new(Queue::new(start), adjacent, normalise)
}

pub fn dft<S, A, I, F, N>(start: S, adjacent: A, normalise: F) -> Traversal<S, Stack<S>, A, I, F, N>
where
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
{
    Traversal::new(Stack::new(start), adjacent, normalise)
}

pub fn dijkstra<S, A, I, F, N, C, P>(
    start: S,
    adjacent: A,
    normalise: F,
    cost: C,
) -> Traversal<S, PriorityQueue<S, P>, A, I, F, N>
where
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
    C: FnMut(&S) -> P + 'static,
    P: Ord,
{
    Traversal::new(PriorityQueue::new(start, cost), adjacent, normalise)
}

pub fn a_star<S, A, I, F, N, C, H, P>(
    start: S,
    adjacent: A,
    normalise: F,
    mut cost: C,
    mut heuristic: H,
) -> Traversal<S, PriorityQueue<S, P>, A, I, F, N>
where
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
    C: FnMut(&S) -> P + 'static,
    H: FnMut(&S) -> P + 'static,
    P: Ord + Add<Output = P>,
{
    Traversal::new(
        PriorityQueue::new(start, move |s| cost(s) + heuristic(s)),
        adjacent,
        normalise,
    )
}

impl<S, Q, A, I, F, N> Traversal<S, Q, A, I, F, N>
where
    Q: Collection<S>,
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
{
    fn new(states: Q, adjacent: A, normalise: F) -> Traversal<S, Q, A, I, F, N> {
        Traversal {
            adjacent,
            normalise,
            states,
            visited: HashSet::new(),
            _phantom: PhantomData,
        }
    }

    pub fn search<G>(mut self, mut goal: G) -> Option<S>
    where
        G: FnMut(&S) -> bool,
    {
        while let Some(state) = self.next() {
            if goal(&state) {
                return Some(state);
            }
        }
        None
    }
}

pub trait Collection<S> {
    fn push(&mut self, state: S);

    fn pop(&mut self) -> Option<S>;
}

pub struct Stack<S> {
    stack: Vec<S>,
}

impl<S> Stack<S> {
    fn new(start: S) -> Stack<S> {
        Stack { stack: vec![start] }
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

pub struct Queue<S> {
    queue: VecDeque<S>,
}

impl<S> Queue<S> {
    fn new(start: S) -> Queue<S> {
        Queue {
            queue: VecDeque::from([start]),
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

pub struct PriorityQueue<S, P>
where
    P: Ord,
{
    heap: BinaryHeap<PriorityState<S, P>>,
    priority: Box<dyn FnMut(&S) -> P>,
}

impl<S, P> PriorityQueue<S, P>
where
    P: Ord,
{
    fn new(start: S, mut priority: impl FnMut(&S) -> P + 'static) -> PriorityQueue<S, P> {
        PriorityQueue {
            heap: BinaryHeap::from([PriorityState::new(start, &mut priority)]),
            priority: Box::new(priority),
        }
    }
}

impl<S, P> Collection<S> for PriorityQueue<S, P>
where
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

struct PriorityState<S, P>
where
    P: Ord,
{
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
