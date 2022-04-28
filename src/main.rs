use search::dijkstra;
use std::collections::HashMap;

fn main() {
    let edges = HashMap::from([
        ('a', vec![('b', 6), ('d', 1)]),
        ('b', vec![('d', 2), ('e', 2), ('c', 5)]),
        ('d', vec![('a', 1), ('b', 2), ('e', 1)]),
        ('e', vec![('d', 1), ('b', 2), ('c', 5)]),
        ('c', vec![('b', 2), ('e', 5)]),
    ]);
    let start = State { pos: 'a', cost: 0 };
    let state = dijkstra(
        start,
        |s| State::adjacent(s, &edges),
        State::normalise,
        State::cost,
    )
    .search(State::goal)
    .unwrap();
    println!("{}", state.pos);
    println!("{}", state.cost);

    let start = Pos(1, 1, 0);
    let end = dijkstra(start, Pos::adjacent, Pos::normalise, Pos::cost)
        .search(Pos::goal)
        .unwrap();
    println!("{:?}", end);
}

struct State {
    pos: char,
    cost: u32,
}

impl State {
    fn adjacent(&self, edges: &HashMap<char, Vec<(char, u32)>>) -> Vec<State> {
        edges[&self.pos]
            .iter()
            .map(|x| State {
                pos: x.0,
                cost: self.cost + x.1,
            })
            .collect()
    }

    fn normalise(&self) -> char {
        self.pos
    }

    fn cost(&self) -> u32 {
        self.cost
    }

    fn goal(&self) -> bool {
        self.pos == 'c'
    }
}

#[derive(Debug)]
struct Pos(i32, i32, u32);

impl Pos {
    fn adjacent(&self) -> Vec<Pos> {
        let &Pos(x, y, cost) = self;
        vec![
            Pos(x + 2, y + 1, cost + 1),
            Pos(x + 1, y + 2, cost + 1),
            Pos(x - 1, y + 2, cost + 1),
            Pos(x - 2, y + 1, cost + 1),
            Pos(x - 2, y - 1, cost + 1),
            Pos(x - 1, y - 2, cost + 1),
            Pos(x + 1, y - 2, cost + 1),
            Pos(x + 2, y - 1, cost + 1),
        ]
    }

    fn normalise(&self) -> (i32, i32) {
        (self.0, self.1)
    }

    fn cost(&self) -> u32 {
        self.2
    }

    fn goal(&self) -> bool {
        self.0 == 4 && self.1 == 6
    }
}
