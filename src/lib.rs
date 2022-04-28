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
    N: Hash + Eq,
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

pub fn bft<S, AF, A, NF, N>(start: S, adjacent: AF, normalise: NF) -> impl Iterator<Item = S>
where
    AF: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    NF: FnMut(&S) -> N,
    N: Hash + Eq,
{
    Traverse::new(start, Queue::new(), adjacent, normalise)
}

pub fn dft<S, AF, A, NF, N>(start: S, adjacent: AF, normalise: NF) -> impl Iterator<Item = S>
where
    AF: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    NF: FnMut(&S) -> N,
    N: Hash + Eq,
{
    Traverse::new(start, Stack::new(), adjacent, normalise)
}

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
    N: Hash + Eq,
    CF: FnMut(&S) -> P,
    P: Ord,
{
    Traverse::new(start, PriorityQueue::new(cost), adjacent, normalise)
}

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
    N: Hash + Eq,
    CF: FnMut(&S) -> P,
    HF: FnMut(&S) -> P,
    P: Add,
    <P as Add>::Output: Ord,
{
    dijkstra(start, adjacent, normalise, move |s| cost(s) + heuristic(s))
}

impl<S, Q, AF, A, NF, N> Traverse<S, Q, AF, A, NF, N>
where
    Q: Collection<S>,
    AF: FnMut(&S) -> A,
    A: IntoIterator<Item = S>,
    NF: FnMut(&S) -> N,
    N: Hash + Eq,
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
