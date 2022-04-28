use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet, VecDeque},
    hash::Hash,
    marker::PhantomData,
    ops::Add,
};

struct Traverse<S, Q, A, I, F, N> {
    adjacent: A,
    normalise: F,
    states: Q,
    visited: HashSet<N>,
    _phantom: PhantomData<(S, I)>,
}

impl<S, Q, A, I, F, N> Iterator for Traverse<S, Q, A, I, F, N>
where
    Q: Collection<S>,
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
{
    type Item = S;

    fn next(&mut self) -> Option<Self::Item> {
        let mut current = self.states.pop()?;
        while self.visited.contains(&(self.normalise)(&current)) {
            current = self.states.pop()?;
        }
        self.visited.insert((self.normalise)(&current));
        for state in (self.adjacent)(&current) {
            if !self.visited.contains(&(self.normalise)(&state)) {
                self.states.push(state);
            }
        }
        return Some(current);
    }
}

pub fn bft<S, A, I, F, N>(start: S, adjacent: A, normalise: F) -> impl Iterator<Item = S>
where
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
{
    Traverse::new(Queue::new(), start, adjacent, normalise)
}

pub fn dft<S, A, I, F, N>(start: S, adjacent: A, normalise: F) -> impl Iterator<Item = S>
where
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
{
    Traverse::new(Stack::new(), start, adjacent, normalise)
}

pub fn dijkstra<S, A, I, F, N, C, P>(
    start: S,
    adjacent: A,
    normalise: F,
    cost: C,
) -> impl Iterator<Item = S>
where
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
    C: FnMut(&S) -> P,
    P: Ord,
{
    Traverse::new(PriorityQueue::new(cost), start, adjacent, normalise)
}

pub fn a_star<S, A, I, F, N, C, H, P>(
    start: S,
    adjacent: A,
    normalise: F,
    mut cost: C,
    mut heuristic: H,
) -> impl Iterator<Item = S>
where
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
    C: FnMut(&S) -> P,
    H: FnMut(&S) -> P,
    P: Add,
    <P as Add>::Output: Ord,
{
    dijkstra(start, adjacent, normalise, move |s| cost(s) + heuristic(s))
}

impl<S, Q, A, I, F, N> Traverse<S, Q, A, I, F, N>
where
    Q: Collection<S>,
    A: FnMut(&S) -> I,
    I: IntoIterator<Item = S>,
    F: FnMut(&S) -> N,
    N: Hash + Eq,
{
    fn new(mut states: Q, start: S, adjacent: A, normalise: F) -> Traverse<S, Q, A, I, F, N> {
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

pub trait Collection<S> {
    fn push(&mut self, state: S);

    fn pop(&mut self) -> Option<S>;
}

pub struct Stack<S> {
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

pub struct Queue<S> {
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

pub struct PriorityQueue<S, C, P> {
    heap: BinaryHeap<PriorityState<S, P>>,
    priority: C,
}

impl<S, C, P> PriorityQueue<S, C, P>
where
    C: FnMut(&S) -> P,
    P: Ord,
{
    fn new(priority: C) -> PriorityQueue<S, C, P> {
        PriorityQueue {
            heap: BinaryHeap::new(),
            priority,
        }
    }
}

impl<S, C, P> Collection<S> for PriorityQueue<S, C, P>
where
    C: FnMut(&S) -> P,
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
