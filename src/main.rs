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

const BEAM_WIDTH: usize = 75;
const PUSH_RATE: usize = 4;

const N: usize = 100000;
const S: usize = 10000;
const C: usize = 10;
const H: usize = 10;
const U: usize = 10;

const NOTHING: usize = std::usize::MAX;

type Chamereon = usize;
type Color = usize;
type Position = usize;

fn solve<R: std::io::BufRead, W: std::io::Write>(reader: &mut R, writer: &mut W) {
    let clock = Clock::new();
    let mut buf = String::new();
    reader.read_line(&mut buf).unwrap();
    buf.clear();
    reader.read_line(&mut buf).unwrap();
    let stripe: Vec<usize> = buf.trim_right().chars().map(|c| c as usize - 'A' as usize).collect();
    buf.clear();
    reader.read_line(&mut buf).unwrap();
    let bottle: Vec<usize> = buf.trim_right().chars().map(|c| c as usize - 'A' as usize).collect();
    let hand: Vec<Color> = bottle[0..H].to_vec();
    let bottle: Vec<Color> = bottle[H..].to_vec();
    let cp = ColoredPath::new(stripe, bottle);

    let (final_state, history) = beam_search(&cp, State::new(&hand), BEAM_WIDTH);

    assert_eq!(history.len(), S);

    for h in history {
        writer.write_fmt(format_args!("{}\n", h)).unwrap();
    }

    eprintln!("-----------------------------------------------------");
    eprintln!("score: {}", cp.eval_state(final_state).score);
    eprintln!("result: {}", final_state.result());
    eprintln!("time: {} ms", clock.elapsed());
    eprintln!("-----------------------------------------------------");
}

pub fn beam_search(cp: &ColoredPath, state: State, width: usize) -> (State, Vec<Action>) {
    let mut heap = MaxHeapTopK::new(width);
    let mut dp: Vec<Vec<Step>> = vec![vec![];S+1];
    dp[0] = vec![Step {
                     state: state,
                     action: Action::dummy(),
                     prev_id: NOTHING,
                 }];
    for i in 1..S + 1 {
        heap.clear();
        for (j, &step) in dp[i - 1].iter().enumerate() {
            for candidate in cp.next_all_states(step.state) {
                heap.push((candidate, j));
            }
        }
        for (candidate, j) in heap.to_vec() {
            dp[i].push(Step {
                state: candidate.0.state,
                action: candidate.1,
                prev_id: j,
            })
        }
    }

    let mut best_id = 0;
    let mut best_result = 0;
    for (i, &step) in dp[S].iter().enumerate() {
        if best_result < step.state.result() {
            best_id = i;
            best_result = step.state.result();
        }
    }
    let best_step = dp[S][best_id];

    let mut history = vec![];
    history.push(best_step.action);
    let mut prev_id = best_step.prev_id;
    for i in (1..S).rev() {
        let step = dp[i][prev_id];
        prev_id = step.prev_id;
        history.push(step.action);
    }
    (best_step.state, history)
}

pub struct ColoredPath {
    pub bottle: Vec<Color>,
    pub stripe: Vec<Color>,
    pub next: Vec<[Position; C]>,
}

impl ColoredPath {
    pub fn new(stripe: Vec<Color>, bottle: Vec<Color>) -> ColoredPath {
        let next = Self::build_next(&stripe);
        ColoredPath {
            bottle: bottle,
            stripe: stripe,
            next: next,
        }
    }

    pub fn eval_state(&self, state: State) -> ScoredState {
        let mut pos = state.position.clone();
        pos.sort();
        let mut score = 0.0;
        for i in 0..U {
            score += pos[i] as f64 * 0.925f64.powi(i as i32);
        }
        ScoredState {
            score: score,
            state: state,
        }
    }

    pub fn next_all_states(&self, state: State) -> Vec<(ScoredState, Action)> {
        let mut res = Vec::new();
        for (target, &position) in state.position.iter().enumerate() {
            for (hand_id, &color) in state.hand.iter().enumerate() {
                let mut pos = position + self.next[position][color];
                while state.occupied(pos) {
                    pos += self.next[pos][color];
                }
                let mut new_position = state.position.clone();
                new_position[target] = pos;
                let mut new_hand = state.hand.clone();
                new_hand[hand_id] = self.bottle[state.time];
                let next = State {
                    time: state.time + 1,
                    position: new_position,
                    hand: new_hand,
                };

                let action = Action::new(target, color);
                res.push((self.eval_state(next), action));
            }
        }
        res
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

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct State {
    pub time: usize,
    pub position: [Position; U],
    pub hand: [Color; H],
}

impl State {
    pub fn new(hand: &Vec<Color>) -> State {
        let mut pos = [0; U];
        let mut hs = [0; H];
        for i in 0..U {
            pos[i] = i;
        }
        for i in 0..H {
            hs[i] = hand[i];
        }
        State {
            time: 0,
            position: pos,
            hand: hs,
        }
    }

    pub fn occupied(&self, pos: usize) -> bool {
        self.position.contains(&pos)
    }

    pub fn result(&self) -> usize {
        *self.position.iter().min().unwrap()
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct ScoredState {
    pub score: f64,
    pub state: State,
}

impl Eq for ScoredState {}

impl Ord for ScoredState {
    fn cmp(&self, other: &ScoredState) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

impl PartialOrd for ScoredState {
    fn partial_cmp(&self, other: &ScoredState) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

#[derive(Eq,PartialEq, Ord, PartialOrd,Debug,Clone,Copy)]
pub struct Action {
    target: Chamereon,
    color: Color,
}

impl Action {
    pub fn new(target: Chamereon, color: Color) -> Action {
        Action {
            target: target,
            color: color,
        }
    }
    pub fn dummy() -> Action {
        Action {
            target: NOTHING,
            color: NOTHING,
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

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Step {
    pub state: State,
    pub action: Action,
    pub prev_id: usize,
}

pub struct MaxHeapTopK<T> {
    k: usize,
    min_heap: MinHeap<T>,
    rng: Random,
}

impl<T> MaxHeapTopK<T> {
    pub fn new(k: usize) -> MaxHeapTopK<T> {
        MaxHeapTopK {
            k: k,
            min_heap: MinHeap::with_capacity(k + 1),
            rng: Random::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.min_heap.len()
    }

    pub fn clear(&mut self) {
        self.min_heap.data.clear()
    }
}
impl<T: Ord + Clone> MaxHeapTopK<T> {
    pub fn to_vec(&self) -> Vec<T> {
        self.min_heap.data.clone()
    }

    pub fn push(&mut self, x: T) {
        if self.len() < self.k {
            self.min_heap.push(x);
        } else if self.min_heap.data[0] < x && self.rng.usize(PUSH_RATE) == 0 {
            self.min_heap.data.push(x);
            self.min_heap.data.swap_remove(0);
            MinHeap::sift_down(&mut self.min_heap.data, 0..self.k);
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

#[derive(Clone)]
pub struct MinHeap<T> {
    pub data: Vec<T>,
}

impl<T> MinHeap<T> {
    pub fn new() -> MinHeap<T> {
        MinHeap { data: Vec::new() }
    }

    pub fn with_capacity(n: usize) -> MinHeap<T> {
        MinHeap { data: Vec::with_capacity(n) }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: ::std::cmp::PartialOrd + Clone> MinHeap<T> {
    pub fn from_vec(vec: &Vec<T>) -> MinHeap<T> {
        let mut data: Vec<T> = vec.clone();
        let n = data.len();
        for i in (0..n / 2).rev() {
            Self::sift_down(&mut data, i..n);
        }
        MinHeap { data: data }
    }

    pub fn to_vec(&self) -> Vec<T> {
        let mut h = self.clone();
        let mut res: Vec<T> = Vec::with_capacity(self.data.len());
        while let Some(x) = h.pop() {
            res.push(x);
        }
        res
    }

    pub fn peek(&self) -> Option<&T> {
        self.data.get(0)
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            None
        } else {
            let n = self.data.len();
            let popped = self.data.swap_remove(0);
            Self::sift_down(&mut self.data, 0..n - 1);
            Some(popped)
        }
    }

    pub fn push(&mut self, x: T) {
        let n = self.data.len();
        self.data.push(x);
        Self::sift_up(&mut self.data, 0..n + 1);
    }

    pub fn append(&mut self, other: &mut Self) {
        if self.len() < other.len() {
            ::std::mem::swap(self, other);
        }
        while let Some(x) = other.pop() {
            self.push(x);
        }
    }

    pub fn sift_down(vec: &mut Vec<T>, range: ::std::ops::Range<usize>) {
        let n = range.end;
        let mut parent = range.start;
        loop {
            let l_child = (parent << 1) + 1;
            let r_child = l_child + 1;
            if l_child >= n {
                break;
            }

            let mut cur = parent;
            if vec[cur] > vec[l_child] {
                cur = l_child;
            }

            if r_child < n && vec[cur] > vec[r_child] {
                cur = r_child;
            }

            if cur == parent {
                break;
            } else {
                vec.swap(parent, cur);
                parent = cur;
            }
        }
    }

    fn sift_up(vec: &mut Vec<T>, range: ::std::ops::Range<usize>) {
        let mut child = range.end - 1;
        while child > range.start {
            let parent = (child - 1) >> 1;
            if range.start <= parent && vec[parent] > vec[child] {
                vec.swap(parent, child);
                child = parent;
            } else {
                break;
            }
        }
    }
}

pub struct Clock {
    instant: ::std::time::Instant,
}

impl Clock {
    pub fn new() -> Clock {
        Clock { instant: ::std::time::Instant::now() }
    }

    pub fn elapsed(&self) -> u64 {
        self.elapsed_as_msec()
    }

    pub fn elapsed_as_msec(&self) -> u64 {
        let t = self.instant.elapsed();
        t.as_secs() * 1000 + t.subsec_millis() as u64
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
