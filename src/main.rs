#![allow(unused_imports, unused_macros, dead_code)]
use std::f64::*;
use std::cmp::*;
use std::collections::*;
use std::ops::*;

macro_rules! debug{
    ($($a:expr),*) => {
        #[cfg(debug_assertions)]
        eprintln!(
            concat!("{}:{}:{}: ",$(stringify!($a), " = {:?}, "),*),
            file!(), line!(), column!(), $($a),*
        );
        #[cfg(not(debug_assertions))]
        {};
    }
}

const N: usize = 100000;
const S: usize = 10000;
const C: usize = 10;
const H: usize = 10;
const U: usize = 10;

const NOTHING: usize = std::usize::MAX;
type Chamereon = usize;
type Color = usize;

fn solve<R: std::io::BufRead, W: std::io::Write>(reader: &mut R, writer: &mut W) {
    let mut buf = String::new();
    reader.read_line(&mut buf).unwrap();
    buf.clear();
    reader.read_line(&mut buf).unwrap();
    let stripe: Vec<usize> = buf.trim_right().chars().map(|c| c as usize - 'A' as usize).collect();
    buf.clear();
    reader.read_line(&mut buf).unwrap();
    let bottle: Vec<usize> = buf.trim_right().chars().map(|c| c as usize - 'A' as usize).collect();

    let mut cp = ColoredPath::new(stripe, bottle);

    while !cp.is_end() {
        let actions = one_step_greedy(&cp);
        for action in actions {
            if !cp.is_end() {
                cp.throw(action);
            } else {
                break;
            }
        }
    }

    for h in cp.history {
        writer.write_fmt(format_args!("{}\n", h)).unwrap();
    }
    eprintln!("{}", cp.state.score());
}

pub fn one_step_greedy(cp: &ColoredPath) -> Vec<Action> {
    let mut leftest_id = 0;
    for i in 1..U {
        if cp.state.position[leftest_id] > cp.state.position[i] {
            leftest_id = i;
        }
    }

    let mut max_next_pos = 0;
    let mut max_next_color = 0;
    for i in 0..H {
        let h = cp.hand[i];
        let next = cp.next_state(Action::new(leftest_id, h));
        if max_next_pos < next.position[leftest_id] {
            max_next_pos = next.position[leftest_id];
            max_next_color = h;
        }
    }
    vec![Action::new(leftest_id, max_next_color)]
}

pub struct ColoredPath {
    pub state: State,
    pub hand: Vec<Color>,
    bottle_queue: VecDeque<Color>,
    stripe: Vec<usize>,
    next: Vec<[usize; C]>,
    pub history: Vec<Action>,
}

impl ColoredPath {
    pub fn new(stripe: Vec<Color>, bottle: Vec<Color>) -> ColoredPath {
        let next = Self::build_next(&stripe);
        ColoredPath {
            state: State::new(),
            hand: bottle[0..H].to_vec(),
            bottle_queue: VecDeque::from(bottle[H..].to_vec()),
            stripe: stripe,
            next: next,
            history: vec![],
        }
    }

    pub fn is_end(&self) -> bool {
        self.bottle_queue.is_empty()
    }

    pub fn throw(&mut self, action: Action) {
        let mut pos = self.state.position[action.target];
        while self.state.occupied(pos) {
            pos += self.next[pos][action.color];
        }
        self.state.position[action.target] = pos;
        self.history.push(action);
        for i in 0..H {
            if self.hand[i] == action.color {
                self.hand[i] = self.bottle_queue.pop_front().unwrap();
                break;
            }
        }
    }

    pub fn next_state(&self, action: Action) -> State {
        let mut pos = self.state.position[action.target];
        while self.state.occupied(pos) {
            pos += self.next[pos][action.color];
        }
        let mut position: [usize; U] = self.state.position.clone();
        position[action.target] = pos;
        State { position: position }
    }

    fn build_next(vec: &Vec<usize>) -> Vec<[usize; C]> {
        let n = vec.len();
        let mut next = vec![[0x3f3f3f3f;C];2 * n];
        for i in (1..2 * n).rev() {
            for j in 0..C {
                if vec[i % n] == j {
                    next[i - 1][j] = 1;
                } else {
                    next[i - 1][j] = next[i][j] + 1;
                }
            }
        }
        next.truncate(n);
        next
    }
}

#[derive(Debug)]
pub struct State {
    pub position: [usize; U],
}

impl State {
    pub fn new() -> State {
        let mut pos = [0; U];
        for i in 0..U {
            pos[i] = i;
        }
        State { position: pos }
    }

    pub fn occupied(&self, pos: usize) -> bool {
        self.position.contains(&pos)
    }

    pub fn score(&self) -> usize {
        *self.position.iter().min().unwrap()
    }
}

#[derive(Debug,Clone,Copy)]
pub struct Action {
    target: Chamereon,
    color: Color,
}

impl Action {
    fn new(target: Chamereon, color: Color) -> Action {
        Action {
            target: target,
            color: color,
        }
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f,
               "{} {}",
               self.target,
               (self.color as u8 + 'A' as u8) as char)
    }
}

fn main() {
    use std::io::*;
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let stdin = stdin();
            let stdout = stdout();
            solve(&mut BufReader::new(stdin.lock()),
                  &mut BufWriter::new(stdout.lock()));
        })
        .unwrap()
        .join()
        .unwrap();
}
