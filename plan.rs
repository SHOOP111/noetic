//! Deliberate reasoning: neural-guided tree search that *learns from its own
//! search* (AlphaZero-style), with no pretrained anything.
//!
//! The language model is System 1: one fast forward pass per token.
//! This module is System 2: a policy/value network proposes, PUCT-guided
//! Monte-Carlo tree search disposes, and the search's own visit counts and
//! returns become the next round of supervised targets. Search makes the net
//! stronger; the stronger net makes search cheaper. That loop is the whole
//! point, and it needs no dataset - only the rules of the task.
//!
//! Task: the sliding 8-puzzle. Small enough to verify by hand, deep enough
//! that greedy policies fail (state space 181,440 reachable states, solutions
//! up to 31 moves).

use crate::autograd::{Graph, Nid};
use crate::nn::{init_std, Linear};
use crate::optim::AdamW;
use crate::rng::Rng;
use crate::tensor::{matvec_nt, silu, softmax_inplace};

pub const N_ACT: usize = 4;
pub const FEAT: usize = 81;
pub const STEP_COST: f32 = 0.02;

/// tiles[i] is the tile sitting at position i; 0 is the blank.
#[derive(Clone, Copy)]
pub struct Puzzle {
    pub tiles: [u8; 9],
    pub blank: usize,
}

impl Puzzle {
    pub fn solved() -> Puzzle {
        let mut tiles = [0u8; 9];
        for i in 0..9 {
            tiles[i] = i as u8;
        }
        Puzzle { tiles, blank: 0 }
    }

    pub fn is_solved(&self) -> bool {
        for i in 0..9 {
            if self.tiles[i] != i as u8 {
                return false;
            }
        }
        true
    }

    /// 0 = slide blank up, 1 = down, 2 = left, 3 = right
    pub fn legal(&self, a: usize) -> bool {
        let r = self.blank / 3;
        let c = self.blank % 3;
        match a {
            0 => r > 0,
            1 => r < 2,
            2 => c > 0,
            3 => c < 2,
            _ => false,
        }
    }

    fn target_of(&self, a: usize) -> usize {
        match a {
            0 => self.blank - 3,
            1 => self.blank + 3,
            2 => self.blank - 1,
            _ => self.blank + 1,
        }
    }

    /// Applies `a` if legal and returns the reward: -STEP_COST per move,
    /// +1 on reaching the goal. Illegal moves are no-ops with a small penalty,
    /// so the policy learns the action mask instead of being handed it.
    pub fn step(&mut self, a: usize) -> f32 {
        if !self.legal(a) {
            return -STEP_COST;
        }
        let t = self.target_of(a);
        let tmp = self.tiles[t];
        self.tiles[t] = self.tiles[self.blank];
        self.tiles[self.blank] = tmp;
        self.blank = t;
        if self.is_solved() {
            1.0 - STEP_COST
        } else {
            -STEP_COST
        }
    }

    /// Random walk from the goal: guarantees solvability (half of all
    /// permutations of the 8-puzzle are unreachable).
    pub fn scramble(&mut self, n: usize, rng: &mut Rng) {
        let mut last = usize::MAX;
        let mut done = 0usize;
        let mut guard = 0usize;
        while done < n && guard < n * 20 + 64 {
            guard += 1;
            let a = rng.below(N_ACT);
            if !self.legal(a) {
                continue;
            }
            // avoid immediately undoing the previous move
            let inverse = match a {
                0 => 1,
                1 => 0,
                2 => 3,
                _ => 2,
            };
            if last == inverse {
                continue;
            }
            self.step(a);
            last = a;
            done += 1;
        }
    }

    /// One-hot position x tile encoding: 9 positions * 9 tile ids.
    pub fn features(&self, out: &mut [f32]) {
        assert_eq!(out.len(), FEAT, "feature buffer has the wrong size");
        out.fill(0.0);
        for p in 0..9 {
            out[p * 9 + (self.tiles[p] as usize)] = 1.0;
        }
    }

    /// 36-bit packed state, used as the transposition key.
    pub fn key(&self) -> u64 {
        let mut k = 0u64;
        for i in 0..9 {
            k |= (self.tiles[i] as u64) << (i * 4);
        }
        k
    }

    /// Sum of tile displacements. Never used for training or search - only for
    /// reporting how hard a scrambled board is.
    pub fn manhattan(&self) -> usize {
        let mut d = 0usize;
        for p in 0..9 {
            let t = self.tiles[p] as usize;
            if t == 0 {
                continue;
            }
            let (pr, pc) = (p / 3, p % 3);
            let (tr, tc) = (t / 3, t % 3);
            let dr = if pr > tr { pr - tr } else { tr - pr };
            let dc = if pc > tc { pc - tc } else { tc - pc };
            d += dr + dc;
        }
        d
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        for r in 0..3 {
            for c in 0..3 {
                let t = self.tiles[r * 3 + c];
                if t == 0 {
                    s.push_str(" . ");
                } else {
                    s.push_str(&format!(" {} ", t));
                }
            }
            s.push('\n');
        }
        s
    }
}

pub fn action_name(a: usize) -> &'static str {
    match a {
        0 => "up",
        1 => "down",
        2 => "left",
        _ => "right",
    }
}

/// Two-headed MLP: shared trunk -> policy logits over 4 moves + scalar value.
/// Deliberately not a transformer, and deliberately tiny: the intelligence
/// comes from the search/learning loop, not from parameter count.
pub struct PvNet {
    pub l1: Linear,
    pub l2: Linear,
    pub ph: Linear,
    pub vh: Linear,
    pub hidden: usize,
}

struct EvalScratch {
    h1: Vec<f32>,
    h2: Vec<f32>,
    policy: [f32; N_ACT],
    value: [f32; 1],
}

impl EvalScratch {
    fn new(hidden: usize) -> EvalScratch {
        EvalScratch {
            h1: vec![0.0; hidden],
            h2: vec![0.0; hidden],
            policy: [0.0; N_ACT],
            value: [0.0; 1],
        }
    }
}

impl PvNet {
    pub fn new(g: &mut Graph, rng: &mut Rng, hidden: usize) -> PvNet {
        let l1 = Linear::new(g, rng, "pv.l1", FEAT, hidden, true, init_std(FEAT));
        let l2 = Linear::new(g, rng, "pv.l2", hidden, hidden, true, init_std(hidden));
        let ph = Linear::new(g, rng, "pv.policy", hidden, N_ACT, true, 0.05);
        let vh = Linear::new(g, rng, "pv.value", hidden, 1, true, 0.05);
        PvNet { l1, l2, ph, vh, hidden }
    }

    /// Tape-free single-state evaluation. The convenience path allocates one
    /// scratch set; MCTS uses the reusable internal path below.
    pub fn eval_one(&self, g: &Graph, feat: &[f32]) -> ([f32; N_ACT], f32) {
        let mut scratch = EvalScratch::new(self.hidden);
        self.eval_one_with_scratch(g, feat, &mut scratch)
    }

    fn eval_one_with_scratch(&self, g: &Graph, feat: &[f32], scratch: &mut EvalScratch) -> ([f32; N_ACT], f32) {
        assert_eq!(feat.len(), FEAT, "policy/value feature vector has the wrong width");
        assert_eq!(scratch.h1.len(), self.hidden, "policy/value scratch has the wrong width");
        assert_eq!(scratch.h2.len(), self.hidden, "policy/value scratch has the wrong width");
        let th = 1;
        let b1 = match self.l1.b {
            Some(b) => Some(&g.val[b][..]),
            None => None,
        };
        matvec_nt(&g.val[self.l1.w], b1, feat, &mut scratch.h1, self.hidden, FEAT, th);
        for i in 0..self.hidden {
            scratch.h1[i] = silu(scratch.h1[i]);
        }
        let b2 = match self.l2.b {
            Some(b) => Some(&g.val[b][..]),
            None => None,
        };
        matvec_nt(
            &g.val[self.l2.w],
            b2,
            &scratch.h1,
            &mut scratch.h2,
            self.hidden,
            self.hidden,
            th,
        );
        for i in 0..self.hidden {
            scratch.h2[i] = silu(scratch.h2[i]) + scratch.h1[i];
        }
        let bp = match self.ph.b {
            Some(b) => Some(&g.val[b][..]),
            None => None,
        };
        matvec_nt(
            &g.val[self.ph.w],
            bp,
            &scratch.h2,
            &mut scratch.policy,
            N_ACT,
            self.hidden,
            th,
        );
        softmax_inplace(&mut scratch.policy);
        let bv = match self.vh.b {
            Some(b) => Some(&g.val[b][..]),
            None => None,
        };
        matvec_nt(
            &g.val[self.vh.w],
            bv,
            &scratch.h2,
            &mut scratch.value,
            1,
            self.hidden,
            th,
        );
        (scratch.policy, scratch.value[0].tanh())
    }

    /// Batched, differentiable version used for training.
    /// Returns (policy logits [rows, N_ACT], value [rows, 1] in (-1, 1)).
    pub fn heads(&self, g: &mut Graph, x: Nid, rows: usize) -> (Nid, Nid) {
        let a1 = self.l1.forward(g, x, rows);
        let s1 = g.silu(a1);
        let a2 = self.l2.forward(g, s1, rows);
        let s2 = g.silu(a2);
        let h2 = g.add(s2, s1);
        let logits = self.ph.forward(g, h2, rows);
        let vraw = self.vh.forward(g, h2, rows);
        let v = g.tanh(vraw);
        (logits, v)
    }
}

// ---------------------------------------------------------------------------
// PUCT Monte-Carlo tree search
// ---------------------------------------------------------------------------

struct Node {
    key: u64,
    prior: [f32; N_ACT],
    n: [u32; N_ACT],
    w: [f32; N_ACT],
    child: [i32; N_ACT],
    legal: [bool; N_ACT],
    visits: u32,
    value: f32,
    terminal: bool,
}

/// Arena-allocated search tree. Nodes are indices, never pointers: no Rc, no
/// RefCell, no lifetime puzzles, and the whole tree frees in one drop.
pub struct Mcts {
    nodes: Vec<Node>,
    pub c_puct: f32,
    pub gamma: f32,
    pub max_depth: usize,
    pub evals: u64,
}

impl Mcts {
    pub fn new() -> Mcts {
        Mcts { nodes: Vec::new(), c_puct: 1.6, gamma: 0.99, max_depth: 48, evals: 0 }
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    fn expand(
        &mut self,
        env: &Puzzle,
        net: &PvNet,
        g: &Graph,
        feat: &mut [f32],
        scratch: &mut EvalScratch,
    ) -> usize {
        env.features(feat);
        let (p, v) = net.eval_one_with_scratch(g, feat, scratch);
        self.evals = self.evals.checked_add(1).expect("MCTS evaluation counter overflow");
        let mut legal = [false; N_ACT];
        let mut sum = 0.0f32;
        let mut n_legal = 0usize;
        for a in 0..N_ACT {
            legal[a] = env.legal(a);
            if legal[a] {
                sum += p[a];
                n_legal += 1;
            }
        }
        let mut prior = [0.0f32; N_ACT];
        if sum > 1e-9 {
            for a in 0..N_ACT {
                if legal[a] {
                    prior[a] = p[a] / sum;
                }
            }
        } else if n_legal > 0 {
            for a in 0..N_ACT {
                if legal[a] {
                    prior[a] = 1.0 / (n_legal as f32);
                }
            }
        }
        let terminal = env.is_solved();
        self.nodes.push(Node {
            key: env.key(),
            prior,
            n: [0; N_ACT],
            w: [0.0; N_ACT],
            child: [-1; N_ACT],
            legal,
            visits: 0,
            value: if terminal { 1.0 } else { v },
            terminal,
        });
        self.nodes.len() - 1
    }

    /// PUCT: argmax_a [ Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ].
    /// Unvisited actions inherit the parent value (first-play urgency), which
    /// stops the search from being forced to try every branch once.
    fn select(&self, idx: usize) -> usize {
        let node = &self.nodes[idx];
        let tot = node.visits as f32;
        let sqrt_tot = if tot > 1.0 { tot.sqrt() } else { 1.0 };
        let mut best = usize::MAX;
        let mut best_score = f32::NEG_INFINITY;
        for a in 0..N_ACT {
            if !node.legal[a] {
                continue;
            }
            let na = node.n[a] as f32;
            let q = if node.n[a] > 0 { node.w[a] / na } else { node.value };
            let u = self.c_puct * node.prior[a] * sqrt_tot / (1.0 + na);
            let s = q + u;
            if s > best_score {
                best_score = s;
                best = a;
            }
        }
        best
    }

    /// Returns the normalised root visit distribution - the search policy.
    pub fn run(&mut self, root: &Puzzle, sims: usize, net: &PvNet, g: &Graph, rng: &mut Rng, dir_alpha: f32) -> [f32; N_ACT] {
        assert!(self.c_puct.is_finite() && self.c_puct >= 0.0, "MCTS exploration constant must be finite and non-negative");
        assert!(self.gamma.is_finite() && self.gamma >= 0.0 && self.gamma <= 1.0, "MCTS discount must be in [0, 1]");
        assert!(self.max_depth > 0, "MCTS maximum depth must be positive");
        assert!(dir_alpha.is_finite() && dir_alpha >= 0.0, "Dirichlet alpha must be finite and non-negative");

        self.nodes.clear();
        let desired_capacity = sims.saturating_add(1);
        if desired_capacity > self.nodes.capacity() {
            self.nodes.reserve(desired_capacity - self.nodes.capacity());
        }
        let mut feat = vec![0.0f32; FEAT];
        let mut eval_scratch = EvalScratch::new(net.hidden);
        let r = self.expand(root, net, g, &mut feat, &mut eval_scratch);

        // Dirichlet exploration noise at the root: keeps self-play from
        // collapsing onto whatever the current net already prefers.
        if dir_alpha > 0.0 {
            let noise = rng.dirichlet(dir_alpha, N_ACT);
            let eps = 0.25f32;
            let mut sum = 0.0f32;
            for a in 0..N_ACT {
                if self.nodes[r].legal[a] {
                    let mixed = (1.0 - eps) * self.nodes[r].prior[a] + eps * noise[a];
                    self.nodes[r].prior[a] = mixed;
                    sum += mixed;
                }
            }
            if sum > 1e-9 {
                for a in 0..N_ACT {
                    self.nodes[r].prior[a] /= sum;
                }
            }
        }

        let path_capacity = self.max_depth.min(4096);
        let mut path: Vec<(usize, usize, f32)> = Vec::with_capacity(path_capacity);
        let mut seen_keys: Vec<u64> = Vec::with_capacity(path_capacity.saturating_add(1));
        for _ in 0..sims {
            let mut env = *root;
            let mut cur = r;
            path.clear();
            seen_keys.clear();
            seen_keys.push(self.nodes[r].key);
            let leaf;
            let mut depth = 0usize;
            loop {
                if self.nodes[cur].terminal {
                    leaf = 0.0;
                    break;
                }
                if depth >= self.max_depth {
                    leaf = self.nodes[cur].value;
                    break;
                }
                let action = self.select(cur);
                if action == usize::MAX {
                    leaf = self.nodes[cur].value;
                    break;
                }
                let reward = env.step(action);
                depth += 1;
                path.push((cur, action, reward));

                // Sliding-puzzle moves can revisit an ancestor. Treat that
                // loop as a neutral leaf; the accumulated step cost then makes
                // it less attractive without expanding duplicate cycle nodes.
                let next_key = env.key();
                if seen_keys.contains(&next_key) {
                    leaf = 0.0;
                    break;
                }
                seen_keys.push(next_key);

                let child = self.nodes[cur].child[action];
                if child < 0 {
                    let next = self.expand(&env, net, g, &mut feat, &mut eval_scratch);
                    assert!(next <= i32::MAX as usize, "MCTS arena exceeds the child-index limit");
                    self.nodes[cur].child[action] = next as i32;
                    leaf = if self.nodes[next].terminal { 0.0 } else { self.nodes[next].value };
                    break;
                }
                let next = child as usize;
                assert_eq!(self.nodes[next].key, next_key, "MCTS child state does not match the environment");
                cur = next;
            }

            // Discounted backup along the traversed path.
            let mut value = leaf;
            let mut i = path.len();
            while i > 0 {
                i -= 1;
                let node = path[i].0;
                let action = path[i].1;
                let reward = path[i].2;
                value = reward + self.gamma * value;
                self.nodes[node].w[action] += value;
                self.nodes[node].n[action] = self.nodes[node].n[action]
                    .checked_add(1)
                    .expect("MCTS edge visit count overflow");
                self.nodes[node].visits = self.nodes[node]
                    .visits
                    .checked_add(1)
                    .expect("MCTS node visit count overflow");
            }
        }

        let mut out = [0.0f32; N_ACT];
        let total = self.nodes[r].visits as f32;
        if total > 0.0 {
            for a in 0..N_ACT {
                out[a] = (self.nodes[r].n[a] as f32) / total;
            }
        } else {
            out = self.nodes[r].prior;
        }
        out
    }

    pub fn root_value(&self) -> f32 {
        if self.nodes.is_empty() {
            0.0
        } else {
            self.nodes[0].value
        }
    }
}

/// Play greedily w.r.t. search visits. Returns (solved, moves, nodes expanded).
pub fn solve(start: &Puzzle, net: &PvNet, g: &Graph, rng: &mut Rng, sims: usize, max_steps: usize) -> (bool, Vec<usize>, usize) {
    let mut env = *start;
    let mut moves: Vec<usize> = Vec::new();
    let mut mcts = Mcts::new();
    let mut expanded = 0usize;
    for _ in 0..max_steps {
        if env.is_solved() {
            return (true, moves, expanded);
        }
        let pi = mcts.run(&env, sims, net, g, rng, 0.0);
        expanded += mcts.size();
        let mut best = usize::MAX;
        let mut bv = -1.0f32;
        for a in 0..N_ACT {
            if env.legal(a) && pi[a] > bv {
                bv = pi[a];
                best = a;
            }
        }
        if best == usize::MAX {
            break;
        }
        env.step(best);
        moves.push(best);
    }
    (env.is_solved(), moves, expanded)
}

pub struct TrainStats {
    pub loss: f32,
    pub solve_rate: f32,
    pub avg_len: f32,
    pub samples: usize,
    pub evals: u64,
}

/// One self-play + learn cycle.
///
/// Targets:
/// * policy <- MCTS root visit distribution (a strictly better policy than the
///   net's own priors, because search corrected them)
/// * value  <- discounted return actually achieved from that state
///
/// Training on your own search output is what makes this self-improving: no
/// human games, no labels, no external model.
pub fn selfplay_iteration(
    g: &mut Graph,
    net: &PvNet,
    opt: &mut AdamW,
    rng: &mut Rng,
    games: usize,
    sims: usize,
    scramble: usize,
    batch: usize,
    lr: f32,
    max_steps: usize,
    temp_moves: usize,
) -> TrainStats {
    assert!(batch > 0, "self-play training batch must be positive");
    assert!(lr.is_finite() && lr >= 0.0, "self-play learning rate must be finite and non-negative");
    let gamma = 0.99f32;
    let mut feats: Vec<f32> = Vec::new();
    let mut pols: Vec<f32> = Vec::new();
    let mut vals: Vec<f32> = Vec::new();
    let mut solved_n = 0usize;
    let mut len_sum = 0usize;
    let mut mcts = Mcts::new();
    let mut feat = vec![0.0f32; FEAT];

    for _ in 0..games {
        let mut env = Puzzle::solved();
        env.scramble(scramble, rng);
        let mut g_feats: Vec<f32> = Vec::new();
        let mut g_pols: Vec<f32> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut solved = false;
        for step in 0..max_steps {
            if env.is_solved() {
                solved = true;
                break;
            }
            let pi = mcts.run(&env, sims, net, g, rng, 0.6);
            env.features(&mut feat);
            g_feats.extend_from_slice(&feat);
            g_pols.extend_from_slice(&pi);
            let a = if step < temp_moves {
                rng.categorical(&pi)
            } else {
                let mut best = 0usize;
                for i in 1..N_ACT {
                    if pi[i] > pi[best] {
                        best = i;
                    }
                }
                best
            };
            let r = env.step(a);
            rewards.push(r);
            if env.is_solved() {
                solved = true;
                break;
            }
        }
        if solved {
            solved_n += 1;
        }
        len_sum += rewards.len();
        // backward pass over the trajectory: discounted return per state
        let n = rewards.len();
        let mut acc = 0.0f32;
        let mut ret = vec![0.0f32; n];
        let mut i = n;
        while i > 0 {
            i -= 1;
            acc = rewards[i] + gamma * acc;
            ret[i] = if acc > 1.0 {
                1.0
            } else if acc < -1.0 {
                -1.0
            } else {
                acc
            };
        }
        feats.extend_from_slice(&g_feats);
        pols.extend_from_slice(&g_pols);
        vals.extend_from_slice(&ret);
    }

    // ---- supervised fit on the freshly generated targets ----
    let rows_total = vals.len();
    let mut weighted_loss = 0.0f64;
    let mut trained_rows = 0usize;
    if rows_total > 0 {
        let mut idx: Vec<usize> = (0..rows_total).collect();
        rng.shuffle_usize(&mut idx);
        let mut off = 0usize;
        while off < rows_total {
            let rows = batch.min(rows_total - off);
            g.reset();
            let mut xb = vec![0.0f32; rows.checked_mul(FEAT).expect("feature minibatch size overflow")];
            let mut pb = vec![0.0f32; rows.checked_mul(N_ACT).expect("policy minibatch size overflow")];
            let mut vb = vec![0.0f32; rows];
            for b in 0..rows {
                let s = idx[off + b];
                for j in 0..FEAT {
                    xb[b * FEAT + j] = feats[s * FEAT + j];
                }
                for j in 0..N_ACT {
                    pb[b * N_ACT + j] = pols[s * N_ACT + j];
                }
                vb[b] = vals[s];
            }
            let x = g.input(vec![rows, FEAT], xb);
            let (logits, v) = net.heads(g, x, rows);
            let lp = g.soft_ce(logits, rows, N_ACT, &pb);
            let lv = g.mse(v, &vb);
            let loss = g.add(lp, lv);
            g.zero_grad();
            g.backward(loss);
            g.clip_grad_norm(1.0);
            opt.step(g, lr);
            weighted_loss += (g.scalar(loss) as f64) * (rows as f64);
            off += rows;
            trained_rows += rows;
        }
        g.reset();
    }

    TrainStats {
        loss: if trained_rows > 0 { (weighted_loss / (trained_rows as f64)) as f32 } else { 0.0 },
        solve_rate: if games > 0 { (solved_n as f32) / (games as f32) } else { 0.0 },
        avg_len: if games > 0 { (len_sum as f32) / (games as f32) } else { 0.0 },
        samples: rows_total,
        evals: mcts.evals,
    }
}
