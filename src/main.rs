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
    let mut cp = ColoredPath::new(stripe, &bottle);

    let mut rng = Random::new();

    let mut best_score = 0;
    let mut best_actions = vec![];
    let mut parameters = vec![];
    let probs = vec![2, 4, 8, 16, 32, 64];
    for &p in &probs {
        parameters.push((5, p));
        parameters.push((5, p + 1));
    }
    for &p in &probs {
        parameters.push((6, p));
    }
    for (step, prob) in parameters {

        while !cp.is_end() {
            let actions = greedy(&cp, min(step, cp.bottle_queue.len()), prob, &mut rng);
            for action in actions {
                if !cp.is_end() {
                    cp.throw(action);
                } else {
                    break;
                }
            }
        }
        let score = cp.state.score();
        eprintln!("score: {}, step: {}, prob: {}", score, step, prob);
        if best_score < score {
            best_score = score;
            best_actions = cp.history.clone();
        }
        cp.init_by_bottle(&bottle);
    }

    for h in best_actions {
        writer.write_fmt(format_args!("{}\n", h)).unwrap();
    }
    eprintln!("{}", best_score);
}

pub fn greedy(cp: &ColoredPath, step_size: usize, prob: usize, rng: &mut Random) -> Vec<Action> {
    let mut chamereons: Vec<usize> = (0..U).collect();
    chamereons.sort_by_key(|&i| cp.state.position[i]);
    chamereons.truncate(step_size);
    chamereons.sort();
    let mut best_score = 0;
    let mut best_actions = vec![];
    loop {
        let (st, actions) = cp.simulate(&chamereons, prob, rng);
        let score = st.position.iter().sum();
        if score > best_score && rng.usize(..prob) != 0 {
            best_score = score;
            best_actions = actions;
        }
        if !chamereons.next_permutation() {
            break;
        }
    }
    best_actions
}

pub struct ColoredPath {
    pub state: State,
    pub hand: Vec<Color>,
    pub bottle_queue: VecDeque<Color>,
    stripe: Vec<usize>,
    next: Vec<[usize; C]>,
    pub history: Vec<Action>,
}

impl ColoredPath {
    pub fn new(stripe: Vec<Color>, bottle: &Vec<Color>) -> ColoredPath {
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

    pub fn init_by_bottle(&mut self, bottle: &Vec<Color>) {
        self.state = State::new();
        self.hand = bottle[0..H].to_vec();
        self.bottle_queue = VecDeque::from(bottle[H..].to_vec());
        self.history.clear();
    }

    pub fn is_end(&self) -> bool {
        self.bottle_queue.is_empty()
    }

    pub fn throw(&mut self, action: Action) {
        let mut pos = self.state.position[action.target];
        while self.state.occupied(pos) {
            pos += self.next[pos % N][action.color];
        }
        self.state.position[action.target] = pos;
        self.history.push(action);
        for i in 0..H {
            if self.hand[i] == action.color {
                self.hand[i] = self.bottle_queue.pop_front().unwrap();
                return;
            }
        }
        unreachable!("faild to throw: color = {}, hand = {:?}",
                     action.color,
                     self.hand);
    }

    pub fn next_state(&self, action: Action) -> State {
        let mut pos = self.state.position[action.target];
        while self.state.occupied(pos) {
            pos += self.next[pos % N][action.color];
        }
        let mut position: [usize; U] = self.state.position.clone();
        position[action.target] = pos;
        State { position: position }
    }

    pub fn simulate(&self,
                    chamereons: &Vec<Chamereon>,
                    prob: usize,
                    rng: &mut Random)
                    -> (State, Vec<Action>) {
        let n = chamereons.len();
        let mut hand = self.hand.clone();
        let mut queue: VecDeque<Color> = self.bottle_queue.iter().map(|&x| x).take(n + 1).collect();
        let mut position = self.state.position.clone();
        let mut actions = Vec::with_capacity(n);
        for &target in chamereons {
            let mut max_pos = 0;
            let mut max_id = 0;
            for i in 0..H {
                let mut pos = position[target];
                while position.contains(&pos) {
                    pos += self.next[pos % N][hand[i]];
                }
                if max_pos < pos && rng.usize(..prob) != 0 {
                    max_pos = pos;
                    max_id = i;
                }
            }
            position[target] = max_pos;
            actions.push(Action::new(target, hand[max_id]));
            hand[max_id] = queue.pop_front().unwrap();
        }
        (State { position: position }, actions)
    }

    pub fn build_next(vec: &Vec<usize>) -> Vec<[usize; C]> {
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

pub trait Permutation {
    fn next_permutation(&mut self) -> bool;
}

impl<T: Ord> Permutation for Vec<T> {
    fn next_permutation(&mut self) -> bool {
        let n = self.len();
        let mut k = NOTHING;
        let mut l = NOTHING;
        for i in 1..n {
            if self[i - 1] < self[i] {
                k = i - 1;
                l = i;
            } else if k != NOTHING && self[k] < self[i] {
                l = i;
            }
        }
        if k == NOTHING {
            false
        } else {
            self.swap(k, l);
            self[k + 1..n].reverse();
            true
        }
    }
}

// xoroshiro128**
// http://vigna.di.unimi.it/xorshift/
pub struct Random {
    state0: u64,
    state1: u64,
}

impl Random {
    pub fn new() -> Random {
        Self::from_seed(123456789)
    }

    pub fn from_seed(seed: u64) -> Random {
        let mut sm = SplitMix64::new(seed);
        Random {
            state0: sm.next(),
            state1: sm.next(),
        }
    }

    fn next(&mut self) -> u64 {
        let s0 = self.state0;
        let mut s1 = self.state1;
        let res = (((s0.wrapping_mul(5)) << 7) | ((s0.wrapping_mul(5)) >> 57)).wrapping_mul(9);
        s1 ^= s0;
        self.state0 = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
        self.state1 = (s1 << 37) | (s1 >> 27);
        res
    }

    pub fn usize<T: U64Normalizer<Output = usize>>(&mut self, normalizer: T) -> usize {
        normalizer.normalize_u64(self.next())
    }

    pub fn f64(&mut self) -> f64 {
        let x = 0x3ff << 52 | self.next() >> 12;
        let f: f64 = unsafe { ::std::mem::transmute::<u64, f64>(x) };
        f - 1.0
    }

    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let x = self.f64();
        let y = self.f64();
        let z = (-2.0 * x.ln()).sqrt() * (2.0 * ::std::f64::consts::PI * y).cos();
        sigma * z + mu
    }

    pub fn shuffle<T>(&mut self, vec: &mut Vec<T>) {
        for i in (0..vec.len()).rev() {
            vec.swap(i, self.usize(i + 1));
        }
    }

    pub fn choice<T: Clone>(&mut self, vec: &Vec<T>) -> T {
        vec[self.usize(vec.len())].clone()
    }

    pub fn choices<T: Clone>(&mut self, vec: &Vec<T>, n: usize) -> Vec<T> {
        (0..n).map(|_| self.choice(vec)).collect()
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    pub fn next(&mut self) -> u64 {
        let mut z = self.state.wrapping_add(0x9e3779b97f4a7c15);
        self.state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

pub trait U64Normalizer {
    type Output;
    fn normalize_u64(&self, x: u64) -> Self::Output;
}

impl U64Normalizer for usize {
    type Output = usize;
    fn normalize_u64(&self, x: u64) -> usize {
        x as usize % self
    }
}

impl U64Normalizer for ::std::ops::Range<usize> {
    type Output = usize;
    fn normalize_u64(&self, x: u64) -> usize {
        self.start + x as usize % (self.end - self.start)
    }
}

impl U64Normalizer for ::std::ops::RangeTo<usize> {
    type Output = usize;
    fn normalize_u64(&self, x: u64) -> usize {
        x as usize % self.end
    }
}

impl U64Normalizer for ::std::ops::RangeFrom<usize> {
    type Output = usize;
    fn normalize_u64(&self, x: u64) -> usize {
        self.start + x as usize % (::std::usize::MAX - self.start)
    }
}

impl U64Normalizer for ::std::ops::RangeFull {
    type Output = usize;
    fn normalize_u64(&self, x: u64) -> usize {
        x as usize
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
