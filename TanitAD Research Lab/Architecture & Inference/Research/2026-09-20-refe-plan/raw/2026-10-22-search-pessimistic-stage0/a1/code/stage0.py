#!/usr/bin/env python3
"""SEARCH-PESSIMISTIC Stage 0: the head-level, session-exposure gap gate.  (CPU only, numpy only.  Run under CUDA_VISIBLE_DEVICES=-1.)

QUESTION.  Does P2's scorer head score tokens from a drive SESSION it never trained on worse than tokens from sessions it did train on?
DESIGN (FEASIBILITY.md section 4 and section 7, kept; every choice the memo left open is fixed in the constants below and listed in DRAFT_PREREG_STAGE0.md):
  * 5 SESSION folds over the 21 sessions of the 24-log table, greedy-balanced by tokens.  A session is the log name minus its last two '_' parts.
  * Inside each LOG a seeded hash splits the tokens into a FIT half and a SEARCH half.  Model h is refit on the FIT halves of every fold except h (init = P2's rows, rows
    NC, DAC, EP, TTC, C trainable, DDC frozen).  Search-half tokens of fold g are scored by OOF = model g (never saw g's sessions) and by IN = model (g+1) % 5 (saw g's logs,
    never g's search tokens).  Both trained on four folds' fit halves.
  * Three measures, each on the SEARCH-half tokens, in the QUALITY direction (positive gap = OOF is worse):
      M1  in-fan concordance of the composite NC.DAC.(5EP+5TTC+2C)/12 with the PDMS labels over the 64-proposal fan   gap = IN - OOF
      M2  A3 aggregate concordance on Step A (anchor, perturbation) pairs                                              gap = IN - OOF
      M3  regret of the argmax of the composite over the fan, in PDMS points (x100)                                    gap = regret(OOF) - regret(IN)
  * Estimator: session-cluster bootstrap over the sessions (B = 10,000, seed fixed below).  Noise floors: the model-swap null (IN_k - IN_(k+1) on the same tokens) and a
    training-seed replicate (H-ESTIM-SEED-1).
  * Controls that must read known values: C-STRUCT (exclusion is structural and fails closed), C-I (labels shuffled within token -> chance), C-II (a model scored on its own
    fit tokens beats OOF; and a PLANTED session effect on real features is recovered), C-III (delete the exclusion line -> the gap collapses to the null).

AMENDMENT A1 (2026-10-22, DRAFT_AMENDMENT_A1.md).  The first sealed read was INVALID on C-II-b because model-to-model training noise on the real features (swap-null nu 0.005-0.010) was as large as the Bonferroni
lower bound of the planted gap (0.002-0.007).  A1 replaces every fold model by the average, at the level of the COMPOSITE SCORE (never the weights), of S models that differ only in the minibatch seed
(primary seeds 0..S-1, replicate seeds 100..100+S-1).  Nothing else changes: the folds, halves, recipe, measures, Delta, alpha, the decision rule and every control with its bound are the sealed ones.  S = N_SEEDS = 1 is the
unamended pipeline bit for bit (golden digests in test_stage0.py); S is fixed from the calibration (code/calib_a1.py, real features with SYNTHETIC labels only) by the selection rule committed before the calibration ran.

NOTHING here reads a result by itself: `run` is the single entry point that reads the real tokens, and it refuses to start without a sealed pre-registration.
Evidence class of every number it will write: MEASURED (ours, SELECTION tier, seen-scene).  It is NOT a T1 driving number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ====================================================================================================================== PRE-REGISTERED CONSTANTS
# Anything in this block is fixed BEFORE the real read; `constants_sha256()` digests it and the sealed prereg must quote the digest.
N_FOLDS = 5
SPLIT_SEED = 20261022                    # fit/search half hash
BOOT_B = 10_000
BOOT_SEED = 20261022
N_SEEDS = 3                              # A1: S = the number of minibatch-seed models whose COMPOSITE SCORES are averaged into every fold model (OOF, IN, swap-null, replicate, planted, mutated, shuffled-label arms alike). 1 = the unamended pipeline; the A1 selection rule (DRAFT_AMENDMENT_A1.md) sets it from the calibration
REPL_SEED_BASE = 100                     # A1: the primary arm trains seeds 0..S-1, the training-seed replicate (H-ESTIM-SEED-1) seeds REPL_SEED_BASE..REPL_SEED_BASE+S-1: disjoint by construction and checked (seed_sets)
ROWS =(0, 1, 2, 3, 4)                   # NC, DAC, EP, TTC, C: the rows the search composite uses (stepA.py:35,126-131)
DDC = 5                                  # frozen: its logit is not in the composite
V1_W = (5.0, 5.0, 2.0)                   # composite weights, planner.py:520 / stepA.py V1_W
THR_PAIR = 0.05                          # a pair enters a concordance only if the TRUE value differs by >= this (stepA.THR_TRUE_PAIR)
PDMS_POINTS = 100.0                      # regret is reported in PDMS points = 100 x (pdms in [0,1])
N_MEAS = 3
ALPHA_FW = 0.025                         # family-wise one-sided level, Bonferroni over the three measures
ALPHA_M = ALPHA_FW / N_MEAS
DELTA = {"M1": 0.01, "M2": 0.02, "M3": 0.35}      # smallest gap of interest, in each measure's own units (JUDGEMENT; M3 = half of C1-C2 = 0.713 points)
REPL_FRAC = 0.5                          # the seed replicate must show >= this fraction of the primary gap
SHUF_SEEDS = tuple(range(9001, 9101))    # C-I (evaluation): R = 100 within-token label shuffles (the M3 statistics need that many to reach 4 x SE <= delta)
SHUF_WORLD_SEEDS = tuple(range(9101, 9111))   # C-I (full pipeline): 10 shuffled worlds, heads trained on the shuffled labels
Z_MAX = 4.0                              # C-I: |z| bound on the shuffled-label residuals (each has mean exactly 0 under the shuffle); the control must also have 4 x SE <= DELTA of its measure
PLANT_KAPPAS = (0.5, 1.0, 2.0, 4.0)      # C-II-b: planted session effects, in units of the std of P2's own logit (feature-only, no label read); the control passes if ANY of them is read and collapses; the table calibrates
PLANT_SEED = 7771
PLANT_SHARP = 1.0                        # C-II-b: label sharpness, p = sigmoid(PLANT_SHARP x true logit)
PLANT_MIN_M1 = 0.02                      # C-II-b: the planted gap must be read at >= this on M1
COLLAPSE_FRAC = 0.2                      # C-III: |gap| after deleting the exclusion must be <= max(its own swap-null, 2 x its own bootstrap SE, this fraction of the unmutated gap)
SEEN_MARGIN = 0.0                        # C-II-a / C-II-b: in the PLANTED world SEEN - OOF on M1 must exceed this (strictly); in the REAL world only a significant REVERSAL (token-level z < -Z_MAX) fails
MIN_FOLD_TOKENS = 100


@dataclass(frozen=True)
class Recipe:
    """The head recipe.  Adam, no weight decay; per row loss = pairwise logistic over Step A (anchor, perturbation) pairs of training Step A tokens (|d sub_r| >= 0.05)
    + lam_bce x pointwise BCE (soft target) over in-fan proposals, + prox x ||row - P2 row||^2.  Optimiser budget `steps` = L2-OFFPROP's chosen recipe (570/540 steps)."""
    lr: float = 1e-3
    steps: int = 600
    batch: int = 256
    lam_bce: float = 10.0
    prox: float = 0.0
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    def as_dict(self):
        d = dict(self.__dict__)
        d["betas"] = list(self.betas)
        return d


R0 = Recipe(prox=0.0)        # PRIMARY (gating): P2's own recipe is plain BCE on the sub-scores; no pull back to P2 so that the head CAN absorb a session
R1 = Recipe(prox=0.3)        # SECONDARY (descriptive, non-gating): L2-OFFPROP's chosen proximal weight


def seed_sets(n: int | None = None) -> tuple[tuple, tuple]:
    """A1: (primary seeds 0..S-1, replicate seeds REPL_SEED_BASE..REPL_SEED_BASE+S-1).  Read at CALL time from N_SEEDS / REPL_SEED_BASE (so a test can move them).  Fails closed if the sets overlap."""
    s = N_SEEDS if n is None else int(n)
    if s < 1:
        raise ValueError("N_SEEDS must be >= 1")
    prim, rep = tuple(range(s)), tuple(range(REPL_SEED_BASE, REPL_SEED_BASE + s))
    if not set(prim).isdisjoint(rep):
        raise RuntimeError(f"A1: the replicate seeds {rep} overlap the primary seeds {prim}: the seed replicate would not be independent")
    return prim, rep


def constants_dict() -> dict:
    return {
        "schema": "spess0-a1", "n_folds": N_FOLDS, "split_seed": SPLIT_SEED, "boot_B": BOOT_B, "boot_seed": BOOT_SEED, "n_seeds": N_SEEDS, "repl_seed_base": REPL_SEED_BASE,
        "primary_seeds": list(seed_sets()[0]), "replicate_seeds": list(seed_sets()[1]), "rows": list(ROWS),
        "ddc_row": DDC, "composite_w": list(V1_W), "thr_pair": THR_PAIR, "pdms_points": PDMS_POINTS, "n_meas": N_MEAS, "alpha_fw": ALPHA_FW, "delta": DELTA, "repl_frac": REPL_FRAC,
        "shuf_seeds": [SHUF_SEEDS[0], SHUF_SEEDS[-1], len(SHUF_SEEDS)], "informative_rule": "4*SE<=delta; trained-world control: M1,M2 only", "shuf_world_seeds": [SHUF_WORLD_SEEDS[0], SHUF_WORLD_SEEDS[-1], len(SHUF_WORLD_SEEDS)], "z_max": Z_MAX, "plant_kappas": list(PLANT_KAPPAS), "plant_seed": PLANT_SEED, "plant_sharp": PLANT_SHARP,
        "plant_min_m1": PLANT_MIN_M1, "collapse_frac": COLLAPSE_FRAC, "seen_margin": SEEN_MARGIN, "r0": R0.as_dict(), "r1": R1.as_dict(),
    }


def constants_sha256() -> str:
    return hashlib.sha256(json.dumps(constants_dict(), sort_keys=True).encode()).hexdigest()


# ====================================================================================================================== inputs
ARTIFACTS = os.environ.get("TANITAD_ARTIFACTS", "D:/Projects/TanitAD-artifacts")
INPUTS = {   # key -> (path relative to ARTIFACTS, sha256, bytes)   MEASURED 2026-10-22 (sha256sum), identical to the banked MANIFESTs
    "table": ("p2_heldout_table_full_2026-10-18/table_p2_heldout.npz", "8eacc19b35b12aee9b0ab047716c901399a711d22ddd3c25cd890ecf2f02004b", 13304426),
    "prehead": ("p2_heldout_table_full_2026-10-18/prehead_features_p2_heldout.npz", "e37b9c554bf2c2546e958ff763dca23b71e0b77036a271d6a003c8ba2672f522", 51442644),
    "feats": ("l2_offprop/feats.npz", "49f6a034464580ef6e14bae9dc397b898a360bc9d7051d6d0676acdba141e616", 67074052),
    "head": ("l2_offprop/p2_score_head.npz", "4b64c3feb89ca571228ed8d4cfa795f3ec5143e71d06545e287e0b9b13ff55d0", 7142),
    "bundle": ("l1_stepA_run/bundle/bundle.npz", "77e8584d8a02ee7a93b9239e1e33a16edade75689f36e1f32959960d0e2eb5c4", 8309154),
    "perts": ("l1_stepA_run/perts.npz", "40ecafc88d401463767d74af5d41809d73b36009083e19d62e197f4bfd6bb8d5", 18972106),
    "labels": ("l1_stepA_run/labels.npz", "7c986dcc736a6dcaf49dfaccd33f3ded9e9c4f2ec49b88f28bb728a71ebf9b27", 2151734),
}


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_file_lf(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def session_of(log: str) -> str:
    """a drive session = the log name minus its two trailing segment indices"""
    return str(log).rsplit("_", 2)[0]


# ====================================================================================================================== the world
@dataclass
class Labels:
    sub: np.ndarray       # [N,64,6] true sub-scores (nc, dac, ep, ttc, c, ddc)
    pdms: np.ndarray      # [N,64]
    valid: np.ndarray     # [N,64] bool
    p_sub: np.ndarray     # [Ns,Q,6] true sub-scores of the perturbed plans
    p_pdms: np.ndarray    # [Ns,Q]
    pair_ok: np.ndarray   # [Ns,Q] bool: the pair has finite labels and a valid anchor


@dataclass
class World:
    token: np.ndarray     # [N] str
    log: np.ndarray       # [N] str
    X: np.ndarray         # [N,64,D] float32 pre-head features
    labels: Labels
    sa_row: np.ndarray    # [Ns] row of each Step A token in the table
    Xp: np.ndarray        # [Ns,Q,D] float32 pre-head features of the perturbed sets at the anchor slot
    anchor: np.ndarray    # [Ns,Q] int
    W6: np.ndarray        # [6,D] P2 score_head
    b6: np.ndarray        # [6]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.N, self.K, self.D = self.X.shape
        self.Ns, self.Q = self.anchor.shape
        self.session = np.array([session_of(x) for x in self.log])
        self.sessions = sorted(set(self.session.tolist()))
        si = {s: i for i, s in enumerate(self.sessions)}
        self.sess_idx = np.array([si[s] for s in self.session], dtype=np.int64)
        self.S = len(self.sessions)
        self.sa_of_row = -np.ones(self.N, dtype=np.int64)          # row -> Step A index, or -1
        self.sa_of_row[self.sa_row] = np.arange(self.Ns)
        if self.W6.shape != (6, self.D):
            raise ValueError("head shape does not match the features")

    def with_labels(self, labels: Labels) -> "World":
        w = World(self.token, self.log, self.X, labels, self.sa_row, self.Xp, self.anchor, self.W6, self.b6, dict(self.meta))
        return w


def load_world(root: str = ARTIFACTS, verify_sha: bool = True) -> tuple[World, dict]:
    """Load the real inputs.  Checks sha256 + size of every file against the constants above (fail loud), checks alignment between the table and Step A."""
    paths, manifest = {}, {}
    for k, (rel, sha, nbytes) in INPUTS.items():
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            raise FileNotFoundError(f"input {k} missing: {p}  (set TANITAD_ARTIFACTS if the drive letter moved)")
        got = sha256_file(p) if verify_sha else None
        if verify_sha and (got != sha or os.path.getsize(p) != nbytes):
            raise RuntimeError(f"ABORT: input {k} ({p}) sha256/size does not match the sealed value: {got} vs {sha}")
        paths[k] = p
        manifest[k] = {"path": p, "sha256": got or sha, "bytes": os.path.getsize(p)}
    t = np.load(paths["table"], allow_pickle=False)
    h = np.load(paths["prehead"], allow_pickle=False)
    f = np.load(paths["feats"], allow_pickle=False)
    hd = np.load(paths["head"], allow_pickle=False)
    b = np.load(paths["bundle"], allow_pickle=False)
    pe = np.load(paths["perts"], allow_pickle=False)
    lb = np.load(paths["labels"], allow_pickle=False)
    tok = t["token"].astype(str)
    if not np.array_equal(h["token"].astype(str), tok) or not np.array_equal(h["log"].astype(str), t["log"].astype(str)):
        raise RuntimeError("ABORT: prehead_features and table disagree on token/log order")
    X = h["prehead_features"].astype(np.float32)
    ti = {x: i for i, x in enumerate(tok)}
    btok = b["token"].astype(str)
    sa_row = np.array([ti[x] for x in btok], dtype=np.int64)            # KeyError = a Step A token is missing from the table: abort
    if len(set(sa_row.tolist())) != len(sa_row):
        raise RuntimeError("ABORT: duplicate Step A tokens")
    if not np.array_equal(t["log"].astype(str)[sa_row], b["log"].astype(str)):
        raise RuntimeError("ABORT: Step A logs disagree with the table")
    if np.abs(t["logits"][sa_row] - b["logits"]).max() > 1e-5:
        raise RuntimeError("ABORT: Step A fan logits differ from the table's: the two sources are not the same fan")
    for src, key, other in ((f, "bundle_sha256", "bundle"), (f, "perts_sha256", "perts"), (pe, "bundle_sha256", "bundle"), (lb, "perts_sha256", "perts")):
        if key in src.files and str(src[key]) != manifest[other]["sha256"]:
            raise RuntimeError(f"ABORT: provenance chain broken: {key} recorded in an input does not match the {other} file")
    fan32 = f["infan_tok"].astype(np.float32)
    drift = float(np.abs(X[sa_row] - fan32).max())
    if drift > 0.02:
        raise RuntimeError(f"ABORT: fp16 table features and Step A float32 features differ by {drift}")
    X[sa_row] = fan32                                                   # one precision for a token's fan and its perturbations
    sub = t["sub"].astype(np.float64)
    pdms = t["pdms"].astype(np.float64)
    valid = t["valid"].astype(bool)
    if not (np.isfinite(sub).all() and np.isfinite(pdms).all() and np.isfinite(X).all() and np.isfinite(f["pert_tok"]).all()):
        raise RuntimeError("ABORT: non-finite entries in the table labels or in the features")
    if pdms.min() < -1e-9 or pdms.max() > 1 + 1e-9:
        raise RuntimeError(f"ABORT: pdms is not on the [0,1] scale ({pdms.min()}, {pdms.max()}): M3 is reported in PDMS points = 100 x pdms")
    if not (np.allclose(b["sub"], sub[sa_row], atol=1e-9, equal_nan=True) and np.allclose(b["pdms"], pdms[sa_row], atol=1e-9, equal_nan=True)):
        raise RuntimeError("ABORT: Step A anchor labels differ from the table's labels for the same proposals")
    anchor = pe["anchor"].astype(np.int64)
    p_sub = lb["sub"].astype(np.float64)
    p_pdms = lb["pdms"].astype(np.float64)
    pair_ok = np.isfinite(p_sub).all(-1) & np.isfinite(p_pdms) & np.take_along_axis(valid[sa_row], anchor, 1)
    p_sub = np.where(pair_ok[..., None], p_sub, 0.0)
    p_pdms = np.where(pair_ok, p_pdms, 0.0)
    lab = Labels(sub, pdms, valid, p_sub, p_pdms, pair_ok)
    w = World(tok, t["log"].astype(str), X, lab, sa_row, f["pert_tok"].astype(np.float32), anchor, hd["weight"].astype(np.float64), hd["bias"].astype(np.float64),
              meta={"kind": "real", "stepA_fp16_drift": drift, "n_pairs_dropped_nonfinite": int((~pair_ok).sum())})
    return w, manifest


# ====================================================================================================================== design: session folds and token halves
@dataclass
class Design:
    fold_of_session: np.ndarray   # [S]
    fold: np.ndarray              # [N] fold of each token
    is_search: np.ndarray         # [N] bool
    info: dict


def make_folds(counts: np.ndarray, names: list[str], k: int = N_FOLDS) -> np.ndarray:
    """greedy balance by tokens: sessions by (-count, name); each goes to the fold with the fewest tokens so far (ties -> lowest fold id).  Deterministic."""
    order = sorted(range(len(names)), key=lambda i: (-int(counts[i]), names[i]))
    load = np.zeros(k, dtype=np.int64)
    out = -np.ones(len(names), dtype=np.int64)
    for i in order:
        f = int(np.argmin(load))
        out[i] = f
        load[f] += int(counts[i])
    return out


def split_halves(token: np.ndarray, log: np.ndarray, seed: int = SPLIT_SEED) -> np.ndarray:
    """Inside every LOG, order the tokens by sha256(seed|token); even ranks are FIT, odd ranks are SEARCH (search = floor(n/2)).  Strict: a token is in exactly one half."""
    is_search = np.zeros(len(token), dtype=bool)
    for lg in sorted(set(log.tolist())):
        idx = np.nonzero(log == lg)[0]
        keys = [hashlib.sha256(f"{seed}|{token[i]}".encode()).hexdigest() for i in idx]
        order = idx[np.argsort(np.array(keys))]
        is_search[order[1::2]] = True
    return is_search


def make_design(w: World, split_seed: int = SPLIT_SEED) -> Design:
    counts = np.bincount(w.sess_idx, minlength=w.S)
    fos = make_folds(counts, w.sessions)
    fold = fos[w.sess_idx]
    is_search = split_halves(w.token, w.log, split_seed)
    info = {
        "fold_tokens": [int((fold == g).sum()) for g in range(N_FOLDS)],
        "fold_search_tokens": [int(((fold == g) & is_search).sum()) for g in range(N_FOLDS)],
        "fold_fit_tokens": [int(((fold == g) & ~is_search).sum()) for g in range(N_FOLDS)],
        "fold_sessions": [int((fos == g).sum()) for g in range(N_FOLDS)],
        "fold_search_stepA_tokens": [int(((fold == g) & is_search & (w.sa_of_row >= 0)).sum()) for g in range(N_FOLDS)],
        "fold_fit_stepA_tokens": [int(((fold == g) & ~is_search & (w.sa_of_row >= 0)).sum()) for g in range(N_FOLDS)],
        "n_sessions": w.S, "n_tokens": w.N, "n_search": int(is_search.sum()), "n_fit": int((~is_search).sum()),
        "session_fold": {s: int(fos[i]) for i, s in enumerate(w.sessions)}, "session_tokens": {s: int(counts[i]) for i, s in enumerate(w.sessions)},
    }
    return Design(fos, fold, is_search, info)


# ====================================================================================================================== the exclusion (THE line that C-III deletes) and its structural check
def train_tokens_for_model(w: World, d: Design, h: int) -> np.ndarray:
    """Tokens model h trains on: the FIT halves of every fold except h."""
    train_folds = [f for f in range(N_FOLDS) if f != h]      # EXCLUDE-LINE: model h never sees fold h
    return np.nonzero(np.isin(d.fold, train_folds) & ~d.is_search)[0]


def check_exclusion(w: World, d: Design, h: int, train_idx: np.ndarray) -> dict:
    """C-STRUCT.  Fails closed.  (a) no training token belongs to fold h; (b) none is a search-half token; (c) no SESSION of fold h appears in the training tokens."""
    tr = np.asarray(train_idx, np.int64)
    in_fold = int((d.fold[tr] == h).sum())
    in_search = int(d.is_search[tr].sum())
    sess_h = set(w.sess_idx[d.fold == h].tolist())
    sess_overlap = len(sess_h & set(w.sess_idx[tr].tolist()))
    if in_fold or in_search or sess_overlap:
        raise RuntimeError(f"C-LEAK: model {h} trains on {in_fold} tokens of its own fold, {in_search} search-half tokens, {sess_overlap} sessions of its own fold")
    return {"model": h, "n_train": int(len(tr)), "own_fold_tokens": 0, "search_tokens": 0, "own_fold_sessions": 0}


# ====================================================================================================================== head training (numpy; analytic gradients, pinned by test_stage0.py against finite differences and torch Adam)
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def softplus(x):
    return np.logaddexp(0.0, x)


def loss_grad(W, b, W0, b0, xf, tf, xa, xp, y, rec: Recipe, touch_ddc: bool = False):
    """Loss and gradient over the full [6,D] head.  Rows in ROWS are trained; row DDC never enters the loss, so its gradient is EXACTLY 0.0.
    xf [B,D], tf [B,5] soft targets, xa/xp [B2,D] pair features, y [B2,5] in {-1,0,+1} (0 = inactive for that row).  `touch_ddc` is the RED mutation of G-LIVE."""
    R = list(ROWS)
    gW = np.zeros_like(W)
    gb = np.zeros_like(b)
    zf = xf @ W[R].T + b[R]
    pf = sigmoid(zf)
    bce = float(np.sum(np.mean(np.logaddexp(0.0, zf) - tf * zf, axis=0)))
    gz = rec.lam_bce * (pf - tf) / len(xf)
    gW[R] += gz.T @ xf
    gb[R] += gz.sum(0)
    pair = 0.0
    if len(xa):
        za = xa @ W[R].T + b[R]
        zp = xp @ W[R].T + b[R]
        act = (y != 0).astype(np.float64)
        d = y * (zp - za)
        n_act = np.maximum(act.sum(0), 1.0)
        pair = float(np.sum((softplus(-d) * act).sum(0) / n_act))
        s = -y * sigmoid(-d) * act / n_act                       # dL/dzp ; dL/dza = -s ; the bias cancels in zp - za
        gW[R] += s.T @ (xp - xa)
    tot = rec.lam_bce * bce + pair
    if rec.prox:
        dW, db = W[R] - W0[R], b[R] - b0[R]
        tot += rec.prox * float((dW ** 2).sum() + (db ** 2).sum())
        gW[R] += 2 * rec.prox * dW
        gb[R] += 2 * rec.prox * db
    if touch_ddc:                                                # RED mutation: a term through the frozen row
        z5 = xf @ W[DDC] + b[DDC]
        tot += 0.1 * float(np.mean(z5 ** 2))
        gW[DDC] += 0.2 * (z5 @ xf) / len(xf)
        gb[DDC] += 0.2 * float(np.mean(z5))
    return tot, bce, pair, gW, gb


def adam_state(W, b) -> dict:
    return {"mW": np.zeros_like(W), "vW": np.zeros_like(W), "mb": np.zeros_like(b), "vb": np.zeros_like(b)}


def adam_update(W, b, gW, gb, st: dict, step: int, rec: Recipe) -> None:
    """one Adam step in place (torch.optim.Adam's update: bias-corrected moments, eps outside the square root, no weight decay); pinned against torch by test_stage0.py"""
    b1, b2 = rec.betas
    st["mW"][:] = b1 * st["mW"] + (1 - b1) * gW
    st["vW"][:] = b2 * st["vW"] + (1 - b2) * gW * gW
    st["mb"][:] = b1 * st["mb"] + (1 - b1) * gb
    st["vb"][:] = b2 * st["vb"] + (1 - b2) * gb * gb
    c1, c2 = 1 - b1 ** step, 1 - b2 ** step
    W -= rec.lr * (st["mW"] / c1) / (np.sqrt(st["vW"] / c2) + rec.eps)
    b -= rec.lr * (st["mb"] / c1) / (np.sqrt(st["vb"] / c2) + rec.eps)


def fit_model(w: World, train_idx: np.ndarray, rec: Recipe, seed: int, touch_ddc: bool = False, check_frozen: bool = True) -> dict:
    """Refit rows NC, DAC, EP, TTC, C on the fit-half tokens `train_idx` (init = P2's head).  Returns {W [6,D], b [6], audit}."""
    L = w.labels
    tr = np.asarray(train_idx, np.int64)
    R = list(ROWS)
    xf = w.X[tr].reshape(-1, w.D)
    ok = L.valid[tr].reshape(-1)
    xf = xf[ok].astype(np.float64)
    tf = L.sub[tr][..., R].reshape(-1, len(R))[ok]
    sa = w.sa_of_row[tr]
    sa = sa[sa >= 0]
    if len(sa):
        rows = w.sa_row[sa]
        a_slot = w.anchor[sa]                                                   # [m,Q]
        xa = np.take_along_axis(w.X[rows], a_slot[..., None], 1)                # [m,Q,D]
        xp = w.Xp[sa]
        sa_a = np.take_along_axis(L.sub[rows], a_slot[..., None], 1)[..., R]    # [m,Q,5]
        dif = L.p_sub[sa][..., R] - sa_a
        y = np.sign(dif) * (np.abs(dif) >= THR_PAIR - 1e-12) * L.pair_ok[sa][..., None]
        keep = (y != 0).any(-1).reshape(-1)
        xa, xp, y = xa.reshape(-1, w.D)[keep].astype(np.float64), xp.reshape(-1, w.D)[keep].astype(np.float64), y.reshape(-1, len(R))[keep]
    else:
        xa = xp = np.zeros((0, w.D))
        y = np.zeros((0, len(R)))
    W0, b0 = w.W6.copy(), w.b6.copy()
    W, b = W0.copy(), b0.copy()
    opt = adam_state(W, b)
    rng = np.random.default_rng(int(seed))
    F, M = len(xf), len(xa)
    if F == 0:
        raise RuntimeError("no training fan points")
    gmin_ep, trace = float("inf"), []
    for step in range(1, rec.steps + 1):
        fi = rng.integers(0, F, rec.batch)
        if M:
            pi = rng.integers(0, M, rec.batch)
            tot, bce, pair, gW, gb = loss_grad(W, b, W0, b0, xf[fi], tf[fi], xa[pi], xp[pi], y[pi], rec, touch_ddc)
        else:
            tot, bce, pair, gW, gb = loss_grad(W, b, W0, b0, xf[fi], tf[fi], xa[:0], xp[:0], y[:0], rec, touch_ddc)
        if not math.isfinite(tot):
            raise RuntimeError("G-LIVE: a loss term is not finite")
        if check_frozen and (np.abs(gW[DDC]).max() != 0.0 or abs(gb[DDC]) != 0.0):
            raise RuntimeError("G-LIVE: the frozen DDC row received a non-zero gradient")
        gmin_ep = min(gmin_ep, float(np.abs(gW[2]).sum()))
        adam_update(W, b, gW, gb, opt, step, rec)
        if step % 100 == 1:
            trace.append((step, float(bce), float(pair)))
    if check_frozen and not (np.array_equal(W[DDC], W0[DDC]) and b[DDC] == b0[DDC]):
        raise RuntimeError("G-LIVE: the frozen DDC row changed")
    return {"W": W, "b": b, "audit": {"n_train_tokens": int(len(tr)), "n_fan_points": int(F), "n_pairs": int(M), "pairs_active_by_row": [int((y[:, r] != 0).sum()) for r in range(len(R))],
                                      "steps": rec.steps, "min_ep_row_grad_abs_sum": gmin_ep, "loss_trace": trace}}


def fit_all(w: World, d: Design, rec: Recipe, seed: int, touch_ddc: bool = False) -> dict:
    """The five fold models.  The exclusion + its structural check run for every model BEFORE any optimiser step."""
    models, audits = [], []
    for h in range(N_FOLDS):
        tr = train_tokens_for_model(w, d, h)
        chk = check_exclusion(w, d, h, tr)                       # CHECK-LINE: structural C-STRUCT, fails closed
        m = fit_model(w, tr, rec, seed * 1_000_003 + 7919 * h + 17, touch_ddc)
        models.append((m["W"], m["b"]))
        audits.append({**chk, **m["audit"]})
    return {"models": models, "audit": audits, "recipe": rec.as_dict(), "seed": seed}


def _as_seeds(seeds) -> tuple:
    """an int (the unamended single-seed API) or an iterable of ints -> a tuple of distinct ints"""
    out = (int(seeds),) if np.isscalar(seeds) else tuple(int(s) for s in seeds)
    if not out or len(set(out)) != len(out):
        raise ValueError(f"A1: need >= 1 distinct training seeds, got {seeds!r}")
    return out


def fit_all_seeds(w: World, d: Design, rec: Recipe, seeds, touch_ddc: bool = False) -> dict:
    """A1.  For every fold h, S models that differ ONLY in the minibatch seed (same data, same init = P2's head, same recipe): models[h] = [(W, b) per seed].  Their COMPOSITE SCORES are averaged at scoring time
    (`ens_score`), never their weights.  Each seed is one unamended `fit_all` call, so the exclusion line and its structural check run for every one of the N_FOLDS x S models before any optimiser step.
    `audit` is the N_FOLDS x S per-model audit rows (each carries its seed).  S = 1 is bit for bit the unamended `fit_all`."""
    seeds = _as_seeds(seeds)
    per = [fit_all(w, d, rec, s, touch_ddc) for s in seeds]
    models = [[per[j]["models"][h] for j in range(len(seeds))] for h in range(N_FOLDS)]
    audit = [{**a, "seed": s} for j, s in enumerate(seeds) for a in per[j]["audit"]]
    return {"models": models, "audit": audit, "recipe": rec.as_dict(), "seeds": list(seeds)}


def p2_models(w: World) -> list:
    return [(w.W6.copy(), w.b6.copy()) for _ in range(N_FOLDS)]


# ====================================================================================================================== measures
def aggregate(z: np.ndarray) -> np.ndarray:
    """NC x DAC x (5 EP + 5 TTC + 2 C) / 12 over sigmoid probabilities; [...,6] -> [...] (== stepA.aggregate, pinned by a test)"""
    p = sigmoid(np.asarray(z, np.float64))
    wv = np.asarray(V1_W, np.float64)
    return p[..., 0] * p[..., 1] * (p[..., 2] * wv[0] + p[..., 3] * wv[1] + p[..., 4] * wv[2]) / wv.sum()


def head_logits(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(X, np.float64) @ W.T + b


def _members(m) -> list:
    """a fold model is ONE head (W, b) (the unamended form, and P2's) or a LIST of the S seed heads of that fold (A1)"""
    return [m] if isinstance(m, tuple) else list(m)


def ens_score(X: np.ndarray, members: list) -> np.ndarray:
    """A1: the composite score of a fold model = the MEAN, over its S seed heads, of each head's composite score aggregate(head_logits(X, W_s, b_s)).  The composite is a product of sigmoids, so this is NOT the
    score of the averaged weights.  One member returns that member's score untouched (bit for bit the unamended path)."""
    acc = None
    for W, b in members:
        a = aggregate(head_logits(X, W, b))
        acc = a if acc is None else acc + a
    return acc if len(members) == 1 else acc / len(members)


def pair_mask(pdms: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[n,64,64] bool: proposal i is better than proposal j by >= THR_PAIR in TRUE pdms (both valid)"""
    dp = pdms[:, :, None] - pdms[:, None, :]
    return (dp >= THR_PAIR - 1e-12) & valid[:, :, None] & valid[:, None, :]


def infan_terms(agg: np.ndarray, mask: np.ndarray):
    """M1 per-token (num, den): concordant weight (ties 0.5) and number of qualifying pairs"""
    ds = agg[:, :, None] - agg[:, None, :]
    conc = (ds > 0).astype(np.float64) + 0.5 * (ds == 0)
    return (mask * conc).sum((1, 2)), mask.sum((1, 2)).astype(np.float64)


def regret_points(agg: np.ndarray, pdms: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """M3 per-token regret of the argmax of the composite over the valid fan, in PDMS points"""
    a = np.where(valid, agg, -np.inf)
    pick = np.argmax(a, axis=1)
    best = np.where(valid, pdms, -np.inf).max(1)
    return PDMS_POINTS * (best - np.take_along_axis(pdms, pick[:, None], 1)[:, 0])


def a3_terms(ag, pg, pa, pp, ok):
    """M2 per-token (num, den) over a token's (anchor, perturbation) pairs: composite vs TRUE pdms, pairs with |dpdms| >= THR (stepA.concordance_counts)"""
    m = (np.abs(pp - pa) >= THR_PAIR - 1e-12) & ok
    ds, dt = np.sign(pg - ag), np.sign(pp - pa)
    conc = np.where(ds == 0, 0.5, (ds == dt).astype(np.float64))
    return (conc * m).sum(1), m.sum(1).astype(np.float64)


@dataclass
class Cache:
    """Scores of every (fold g, role k) cell: tokens of fold g (one half) scored by model (g+k) % 5.  Labels are applied later, so a control can re-measure the SAME scores."""
    half: str
    idx: list           # per fold: token rows
    agg: dict           # (g,k) -> [n,64]
    sa_loc: list        # per fold: positions (within idx[g]) of Step A tokens
    agg_a: dict         # (g,k) -> [m,Q]
    agg_p: dict         # (g,k) -> [m,Q]


def build_cache(w: World, d: Design, models: list, half: str = "search", roles=range(N_FOLDS)) -> Cache:
    sel = d.is_search if half == "search" else ~d.is_search
    idx, sa_loc = [], []
    for g in range(N_FOLDS):
        ii = np.nonzero((d.fold == g) & sel)[0]
        if (half == "search" and not d.is_search[ii].all()) or (half == "fit" and d.is_search[ii].any()):
            raise RuntimeError("C-LEAK: the tokens judged do not belong to the half they are reported as")
        idx.append(ii)
        sa_loc.append(np.nonzero(w.sa_of_row[ii] >= 0)[0])
    agg, agg_a, agg_p = {}, {}, {}
    for g in range(N_FOLDS):
        for k in roles:
            mem = _members(models[(g + k) % N_FOLDS])
            a = ens_score(w.X[idx[g]], mem)
            agg[(g, k)] = a
            sl = sa_loc[g]
            if len(sl):
                s_idx = w.sa_of_row[idx[g][sl]]
                agg_a[(g, k)] = np.take_along_axis(a[sl], w.anchor[s_idx], 1)
                agg_p[(g, k)] = ens_score(w.Xp[s_idx], mem)
            else:
                agg_a[(g, k)] = np.zeros((0, w.Q))
                agg_p[(g, k)] = np.zeros((0, w.Q))
    return Cache(half, idx, agg, sa_loc, agg_a, agg_p)


@dataclass
class Reads:
    """Per-token measure terms, all folds concatenated in fold order.  roles k = 0 (OOF), 1 (IN), 2..4 (further IN models)."""
    roles: list
    tok: np.ndarray                 # [n] token rows
    sess: np.ndarray                # [n] session index
    m1_num: dict                    # k -> [n]
    m1_den: np.ndarray              # [n]
    m3_reg: dict                    # k -> [n]
    sa_tok_pos: np.ndarray          # [m] position (within tok) of each Step A token
    m2_num: dict                    # k -> [m]
    m2_den: np.ndarray              # [m]
    m3_expected_random: np.ndarray  # [n] 100 x (max - mean) of the token's valid fan labels: the exact expectation of a RANDOM pick's regret


def measure(w: World, d: Design, cache: Cache, labels: Labels | None = None) -> Reads:
    L = labels if labels is not None else w.labels
    roles = sorted({k for (_, k) in cache.agg})
    toks, sess, m1d, sapos = [], [], [], []
    m1n = {k: [] for k in roles}
    m3 = {k: [] for k in roles}
    m2n = {k: [] for k in roles}
    m2d, exp_rand = [], []
    off = 0
    for g in range(N_FOLDS):
        ii = cache.idx[g]
        if len(ii) == 0:
            continue
        pd, va = L.pdms[ii], L.valid[ii]
        mask = pair_mask(pd, va)
        toks.append(ii)
        sess.append(w.sess_idx[ii])
        den = mask.sum((1, 2)).astype(np.float64)
        m1d.append(den)
        mean_pd = (pd * va).sum(1) / np.maximum(va.sum(1), 1)
        exp_rand.append(PDMS_POINTS * (np.where(va, pd, -np.inf).max(1) - mean_pd))
        sl = cache.sa_loc[g]
        if len(sl):
            sidx = w.sa_of_row[ii[sl]]
            pa = np.take_along_axis(pd[sl], w.anchor[sidx], 1)
            pp, ok = L.p_pdms[sidx], L.pair_ok[sidx]
            sapos.append(sl + off)
        for k in roles:
            a = cache.agg[(g, k)]
            num, _ = infan_terms(a, mask)
            m1n[k].append(num)
            m3[k].append(regret_points(a, pd, va))
            if len(sl):
                n2, dn2 = a3_terms(cache.agg_a[(g, k)], cache.agg_p[(g, k)], pa, pp, ok)
                m2n[k].append(n2)
                if k == roles[0]:
                    m2d.append(dn2)
        off += len(ii)
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    return Reads(roles, np.concatenate(toks), np.concatenate(sess), {k: cat(m1n[k]) for k in roles}, cat(m1d), {k: cat(m3[k]) for k in roles},
                 cat(sapos).astype(np.int64) if sapos else np.zeros(0, np.int64), {k: cat(m2n[k]) for k in roles}, cat(m2d), cat(exp_rand))


# ====================================================================================================================== the estimator: session-cluster bootstrap
def boot_indices(S: int, B: int = BOOT_B, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, S, size=(B, S))


def cluster_gap(sess: np.ndarray, S: int, A: np.ndarray, B_: np.ndarray, den: np.ndarray, idx: np.ndarray) -> dict:
    """gap = (sum A - sum B) / sum den as a ratio of per-SESSION sums; resamples WHOLE SESSIONS (idx = [B,S]).  A, B_, den are per-token (or per-Step-A-token) arrays with `sess`.
    It answers 'would another draw of SESSIONS say this?' and nothing else: not another training run (the seed replicate), not another split (descriptive)."""
    sa = np.bincount(sess, weights=np.asarray(A, np.float64), minlength=S)
    sb = np.bincount(sess, weights=np.asarray(B_, np.float64), minlength=S)
    sd = np.bincount(sess, weights=np.asarray(den, np.float64), minlength=S)
    if sd.sum() <= 0:
        return {"point": float("nan"), "n_den": 0.0}
    point = float((sa.sum() - sb.sum()) / sd.sum())
    n, dd = (sa - sb)[idx].sum(1), sd[idx].sum(1)
    ok = dd > 0
    r = n[ok] / dd[ok]
    loso = []
    for s in range(S):
        if sd.sum() - sd[s] > 0:
            loso.append(float(((sa.sum() - sa[s]) - (sb.sum() - sb[s])) / (sd.sum() - sd[s])))
    q = lambda p: float(np.quantile(r, p))
    return {"point": point, "se": float(r.std(ddof=1)), "lo_bonf": q(ALPHA_M), "hi_bonf": q(1 - ALPHA_M), "lo95": q(0.025), "hi95": q(0.975), "n_valid_resamples": int(ok.sum()),
            "loso_min": float(min(loso)), "loso_max": float(max(loso)), "n_den": float(sd.sum()), "n_sessions_contributing": int((sd > 0).sum())}


def gaps_of(rd: Reads, S: int, idx: np.ndarray, a_role: int, b_role: int) -> dict:
    """quality-direction gaps between two roles scored on the same tokens:  M1, M2: (num_a - num_b)/den  ;  M3: (regret_b - regret_a) per token.
    So with a_role = IN and b_role = OOF a positive gap means OOF is worse, on every measure."""
    out = {"M1": cluster_gap(rd.sess, S, rd.m1_num[a_role], rd.m1_num[b_role], rd.m1_den, idx)}
    if len(rd.sa_tok_pos):
        out["M2"] = cluster_gap(rd.sess[rd.sa_tok_pos], S, rd.m2_num[a_role], rd.m2_num[b_role], rd.m2_den, idx)
    else:
        out["M2"] = {"point": float("nan"), "n_den": 0.0}
    out["M3"] = cluster_gap(rd.sess, S, rd.m3_reg[b_role], rd.m3_reg[a_role], np.ones(len(rd.tok)), idx)
    return out


def level_of(rd: Reads, S: int, idx: np.ndarray, k: int) -> dict:
    """absolute level of role k (descriptive): M1, M2 concordance, M3 mean regret"""
    sd1 = np.bincount(rd.sess, weights=rd.m1_den, minlength=S)
    sn1 = np.bincount(rd.sess, weights=rd.m1_num[k], minlength=S)
    out = {"M1": float(sn1.sum() / sd1.sum()), "M3": float(rd.m3_reg[k].mean())}
    if len(rd.sa_tok_pos):
        s2 = rd.sess[rd.sa_tok_pos]
        out["M2"] = float(np.bincount(s2, weights=rd.m2_num[k], minlength=S).sum() / np.bincount(s2, weights=rd.m2_den, minlength=S).sum())
    return out


def swap_null(rd: Reads, S: int, idx: np.ndarray) -> dict:
    """the model-swap null: two IN models on the SAME tokens.  Chained pairs (k, k+1) for k = 1..3 (models g+1, g+2, g+3, g+4)."""
    pairs = [(1, 2), (2, 3), (3, 4)]
    per = [gaps_of(rd, S, idx, a, b) for (a, b) in pairs]
    out = {"pairs": [list(p) for p in pairs], "gaps": per}
    for m in ("M1", "M2", "M3"):
        v = [abs(x[m]["point"]) for x in per if not math.isnan(x[m]["point"])]
        out[f"nu_{m}"] = float(max(v)) if v else float("nan")
    return out


# ====================================================================================================================== the arm: everything one (world, recipe, seed) produces
def run_arm(w: World, d: Design, rec: Recipe, seed, idx: np.ndarray, touch_ddc: bool = False, fit=None) -> dict:
    """`seed`: an int (one model per fold, the unamended form) or a tuple of S seeds (A1: S seed models per fold, composite scores averaged)"""
    t0 = time.time()
    fa = fit if fit is not None else fit_all_seeds(w, d, rec, seed, touch_ddc)
    cache = build_cache(w, d, fa["models"], "search")
    rd = measure(w, d, cache)
    sg = gaps_of(rd, w.S, idx, 1, 0)
    sn = swap_null(rd, w.S, idx)
    seen_cache = build_cache(w, d, fa["models"], "fit", roles=(0, 1))
    seen = measure(w, d, seen_cache)
    seen_gap = {"M1": cluster_gap(seen.sess, w.S, seen.m1_num[1], seen.m1_num[0], seen.m1_den, idx), "M1_token": _stat(seen.m1_num[1] - seen.m1_num[0], seen.m1_den.sum()), "M3": cluster_gap(seen.sess, w.S, seen.m3_reg[0], seen.m3_reg[1], np.ones(len(seen.tok)), idx)}
    return {"fit": fa, "cache": cache, "reads": rd, "gaps": sg, "swap_null": sn, "seen_minus_oof": seen_gap, "levels": {"OOF": level_of(rd, w.S, idx, 0), "IN": level_of(rd, w.S, idx, 1)},
            "seconds": time.time() - t0}


# ====================================================================================================================== controls
def shuffled_labels(w: World, seed: int) -> Labels:
    """C-I: within-token label shuffle.  A token WITHOUT Step A perturbations: its 64 labels (sub, pdms, valid together) are permuted.  A Step A token: its 64 fan labels and its 96 perturbation labels are POOLED
    and re-dealt at random (64 to the fan slots, 96 to the perturbation slots).  Pooling is what makes an (anchor, perturbation) pair EXCHANGEABLE: the perturbed plans' labels are distributed differently from the
    fan's (a perturbation is usually worse), so shuffling the two sets separately would NOT put A3 at 0.5 even for a blind score.  Under either deal, any two positions of the same token hold exchangeable labels, so every
    concordance is at 0.5 in expectation, every IN - OOF gap is 0, and a random pick's expected regret is max - mean of the labels the fan received.  Anchors keep pointing at the (re-dealt) fan."""
    L = w.labels
    rng = np.random.default_rng(seed)
    pf = np.argsort(rng.random((w.N, w.K)), axis=1)
    sub = np.take_along_axis(L.sub, pf[..., None], 1)
    pdms = np.take_along_axis(L.pdms, pf, 1)
    valid = np.take_along_axis(L.valid, pf, 1)
    p_sub, p_pdms, p_ok = L.p_sub.copy(), L.p_pdms.copy(), L.pair_ok.copy()
    if w.Ns:
        pu = np.argsort(rng.random((w.Ns, w.K + w.Q)), axis=1)
        U_sub = np.concatenate([L.sub[w.sa_row], L.p_sub], axis=1)
        U_pdms = np.concatenate([L.pdms[w.sa_row], L.p_pdms], axis=1)
        U_ok = np.concatenate([L.valid[w.sa_row], L.pair_ok], axis=1)
        d_sub, d_pdms, d_ok = np.take_along_axis(U_sub, pu[..., None], 1), np.take_along_axis(U_pdms, pu, 1), np.take_along_axis(U_ok, pu, 1)
        sub[w.sa_row], pdms[w.sa_row], valid[w.sa_row] = d_sub[:, :w.K], d_pdms[:, :w.K], d_ok[:, :w.K]
        p_sub, p_pdms, p_ok = d_sub[:, w.K:], d_pdms[:, w.K:], d_ok[:, w.K:]
    return Labels(sub, pdms, valid, p_sub, p_pdms, p_ok)


def _stat(r: np.ndarray, den: float) -> dict:
    """value = sum(r) / den, se = its standard error over the independent (token, shuffle) draws, z = value / se.  Every residual used below has mean EXACTLY 0 under a within-token label shuffle."""
    r = np.asarray(r, np.float64)
    n = len(r)
    if n < 2 or den <= 0:
        return {"value": float("nan"), "se": float("nan"), "z": float("nan"), "n": n}
    se = float(r.std(ddof=1) * math.sqrt(n) / den)
    val = float(r.sum() / den)
    return {"value": val, "se": se, "z": (val / se if se > 0 else (0.0 if abs(val) < 1e-12 else float("inf"))), "n": n}


def chance_check(w: World, d: Design, pairs: list, informative_for=("M1", "M2")) -> dict:
    """C-I.  pairs = [(Cache, Labels), ...]: scores and the (shuffled) labels they are judged against.  Under a within-token label shuffle every concordance is at 0.5, every IN - OOF gap is 0, and the
    regret equals the random-pick expectation 100 x (max - mean) of the token's labels.  Each statistic must (a) satisfy |z| <= Z_MAX and (b) be INFORMATIVE: 4 x SE <= the smallest gap of interest
    DELTA of that measure, i.e. the control could have seen a bias that would matter.  For M1 and M2 an uninformative control FAILS (fix: more shuffles).  M3's regret is so noisy under a shuffle that its
    informativeness is REPORTED (out['informative']) and decides whether M3 is CERTIFIED to take part in the GAP / POWERED rule; it never voids the read by itself."""
    R = {k: [] for k in ("m1c", "m1g", "m2c", "m2g", "m3r", "m3g")}
    den1 = den2 = 0.0
    n3 = 0
    for cache, L in pairs:
        rd = measure(w, d, cache, L)
        R["m1c"].append(rd.m1_num[0] - 0.5 * rd.m1_den)
        R["m1g"].append(rd.m1_num[1] - rd.m1_num[0])
        R["m3r"].append(rd.m3_reg[0] - rd.m3_expected_random)
        R["m3g"].append(rd.m3_reg[0] - rd.m3_reg[1])
        den1 += rd.m1_den.sum()
        n3 += len(rd.tok)
        if len(rd.sa_tok_pos):
            R["m2c"].append(rd.m2_num[0] - 0.5 * rd.m2_den)
            R["m2g"].append(rd.m2_num[1] - rd.m2_num[0])
            den2 += rd.m2_den.sum()
    cat = np.concatenate
    stats = {"M1_concordance_minus_half": (_stat(cat(R["m1c"]), den1), "M1"), "M1_gap": (_stat(cat(R["m1g"]), den1), "M1"),
             "M3_regret_minus_random_pick": (_stat(cat(R["m3r"]), n3), "M3"), "M3_gap": (_stat(cat(R["m3g"]), n3), "M3")}
    if R["m2c"]:
        stats["M2_concordance_minus_half"] = (_stat(cat(R["m2c"]), den2), "M2")
        stats["M2_gap"] = (_stat(cat(R["m2g"]), den2), "M2")
    out, chk, uninformative, inf_by_measure = {"n_label_sets": len(pairs)}, {}, [], {"M1": True, "M2": True, "M3": True}
    for name, (st, m) in stats.items():
        out[name] = st
        at_chance = bool(abs(st["z"]) <= Z_MAX)
        informative = bool(4 * st["se"] <= DELTA[m])
        chk[name] = at_chance and (informative or m not in informative_for)
        if not informative:
            uninformative.append(name)
            inf_by_measure[m] = False
    out["checks"] = chk
    out["uninformative"] = uninformative
    out["informative"] = inf_by_measure
    out["pass"] = bool(all(chk.values()))
    return out


def control_shuffle(w: World, d: Design, cache: Cache, seeds=SHUF_SEEDS) -> dict:
    """C-I (evaluation): the trained models, R = 100 within-token shuffles of the labels they are judged against"""
    return chance_check(w, d, [(cache, shuffled_labels(w, s)) for s in seeds])


def control_shuffle_world(w: World, d: Design, rec: Recipe, idx: np.ndarray, seeds=SHUF_WORLD_SEEDS, train_seeds=None) -> dict:
    """C-I (full pipeline): the heads are TRAINED on shuffled labels and judged on the same shuffled labels; one pipeline per permutation seed, pooled.  A1: each fold model is the S-seed average (train_seeds
    defaults to the primary seeds 0..S-1), so the control tests the pipeline that is read."""
    pairs = []
    tseeds = seed_sets()[0] if train_seeds is None else train_seeds
    for sd in seeds:
        ws = w.with_labels(shuffled_labels(w, sd))
        fa = fit_all_seeds(ws, d, rec, tseeds)
        pairs.append((build_cache(ws, d, fa["models"], "search"), ws.labels))
    out = chance_check(w, d, pairs)
    out["trained_on_shuffled_labels"] = True
    return out


def control_seen(arm: dict) -> dict:
    """C-II-a on the REAL tokens: a model scored on tokens it TRAINED on (SEEN = model g+1 on fold g's fit half) against the model that did not (OOF = model g), on M1.  The token-level overfit of a 257-parameter
    row on 44,000 points is of the order of the swap-null floor (MEASURED on synthetic null worlds: SEEN - OOF from -0.001 to +0.0125; 1 of 36 worlds read <= 0), so the strict rule 'SEEN > OOF' would void about
    one valid read in thirty.  The REAL-world gate is therefore NON-REVERSAL: the token-level paired z of SEEN - OOF must exceed -Z_MAX.  The strict rule 'SEEN beats OOF' is gated where a gap exists by construction:
    in the PLANTED world (`control_planted`).  `strict_positive` is reported."""
    g = arm["seen_minus_oof"]["M1"]
    t = arm["seen_minus_oof"]["M1_token"]
    return {"M1_seen_minus_oof": g["point"], "ci95_session_cluster": [g["lo95"], g["hi95"]], "token_level_z": t["z"], "token_level_se": t["se"], "strict_positive": bool(g["point"] > SEEN_MARGIN),
            "rule": "non-reversal: token-level z > -Z_MAX", "pass": bool(t["z"] > -Z_MAX)}


def control_collapse(arm: dict, arm_mut: dict, gating=("M1",)) -> dict:
    """C-III: with the exclusion line deleted the IN - OOF gap must collapse to the NULL.  The null scale is the rig's own: the larger of the unmutated run's swap-null (ν) and the mutated run's, and twice the
    mutated gap's bootstrap SE.  pass iff |gap_mut| <= max(ν, ν_mut, 2 x SE_mut, COLLAPSE_FRAC x |gap|), i.e. indistinguishable from zero on the rig's noise scale, or at most a fifth of the unmutated gap.
    GATING measure: M1 only (the precise one).  M2 (201 tokens) and M3 (regret flips with every pick) are computed and reported with the same bound but are descriptive: on a null world their
    run-to-run noise alone exceeds a bound this tight about once in ten reads (MEASURED on synthetic null worlds, raw/rehearsal), and a control that fails on noise voids a valid read."""
    out = {}
    for m in ("M1", "M2", "M3"):
        g, gm = arm["gaps"][m]["point"], arm_mut["gaps"][m]["point"]
        if math.isnan(g) or math.isnan(gm):
            out[m] = {"available": False, "pass": True}
            continue
        nu, nu_m, se_m = arm["swap_null"][f"nu_{m}"], arm_mut["swap_null"][f"nu_{m}"], arm_mut["gaps"][m]["se"]
        bound = max(nu, nu_m, 2.0 * se_m, COLLAPSE_FRAC * abs(g))
        out[m] = {"gap": g, "gap_mutated": gm, "swap_null": nu, "swap_null_mutated": nu_m, "se_mutated": se_m, "bound": bound, "pass": bool(abs(gm) <= bound), "gating": m in gating}
    out["pass"] = bool(all(out[m]["pass"] for m in gating))
    return out


# ---------------------------------------------------------------------------------------------------------------------- C-II-b: a PLANTED session effect on real features
def plant_labels(w: World, kappa: float = PLANT_KAPPAS[-1], seed: int = PLANT_SEED, noise_ep: float = 0.05, sharp: float = PLANT_SHARP) -> Labels:
    """SEMI-SYNTHETIC labels: REAL features, SYNTHETIC labels.  True logit of row r at feature x in session s = P2's own logit(x) + kappa x std(P2 logit_r) x (u_{s,r} . x), where u_{s,r} is a
    random direction scaled so that (u . x) has unit std over the world's fan points.  P2 knows the common part; only a model that trained on session s can learn the session part.
    NC/DAC/TTC/C ~ Bernoulli(sigmoid(true logit)); EP = sigmoid(true logit) + N(0, noise_ep) clipped; DDC = 1; pdms = NC.DAC.(5EP+5TTC+2C)/12.  No real label is read."""
    rng = np.random.default_rng(seed)
    X = w.X.reshape(-1, w.D).astype(np.float64)
    zc_all = X @ w.W6.T + w.b6                                           # [NK,6]
    sig = zc_all.std(0)
    sub = np.ones((w.N, w.K, 6))
    p_sub = np.ones((w.Ns, w.Q, 6))
    sess_sa = w.sess_idx[w.sa_row]
    sharp_s = np.broadcast_to(np.asarray(sharp, np.float64), (w.S,))          # a scalar, or one sharpness per session (the power simulation uses heterogeneous sessions)
    for r in ROWS:
        U = rng.standard_normal((w.S, w.D))
        for s in range(w.S):
            u = U[s]
            u = u / max(float((X @ u).std()), 1e-12)                     # (u . x) has unit std over the fan points
            m = w.sess_idx == s
            if m.any():
                z = w.X[m].astype(np.float64) @ w.W6[r] + w.b6[r] + kappa * sig[r] * (w.X[m].astype(np.float64) @ u)
                p = sigmoid(sharp_s[s] * z)
                sub[m, :, r] = _draw(rng, p, r, noise_ep)
            ms = sess_sa == s
            if ms.any():
                zp = w.Xp[ms].astype(np.float64) @ w.W6[r] + w.b6[r] + kappa * sig[r] * (w.Xp[ms].astype(np.float64) @ u)
                p_sub[ms, :, r] = _draw(rng, sigmoid(sharp_s[s] * zp), r, noise_ep)
    sub[..., DDC] = 1.0
    p_sub[..., DDC] = 1.0
    wv = np.asarray(V1_W)
    comp = lambda s: s[..., 0] * s[..., 1] * s[..., 5] * (wv[0] * s[..., 2] + wv[1] * s[..., 3] + wv[2] * s[..., 4]) / wv.sum()
    return Labels(sub, comp(sub), np.ones((w.N, w.K), bool), p_sub, comp(p_sub), np.ones((w.Ns, w.Q), bool))


def _draw(rng, p, r, noise_ep):
    if r == 2:
        return np.clip(p + noise_ep * rng.standard_normal(p.shape), 0.0, 1.0)
    return (rng.random(p.shape) < p).astype(np.float64)


def control_planted(arm: dict, arm_mut: dict) -> dict:
    """C-II-a (strict) + C-II-b + C-III on the PLANTED world: the normal pipeline must read the planted gap (M1 >= PLANT_MIN_M1, lower bound above its own swap-null), a model scored on the tokens it trained on
    must beat the model that did not (SEEN - OOF > SEEN_MARGIN: here a gap exists by construction), and deleting the exclusion must collapse the gap"""
    g = arm["gaps"]["M1"]
    nu = arm["swap_null"]["nu_M1"]
    seen_pt = arm["seen_minus_oof"]["M1"]["point"]
    read = bool(g["point"] >= PLANT_MIN_M1 and g["lo_bonf"] > nu)
    coll = control_collapse(arm, arm_mut)
    return {"M1_gap": g["point"], "M1_lo_bonf": g["lo_bonf"], "swap_null_M1": nu, "planted_gap_read": read, "min_required": PLANT_MIN_M1, "seen_minus_oof_planted": seen_pt, "seen_beats_oof_planted": bool(seen_pt > SEEN_MARGIN),
            "collapse": coll, "pass": bool(read and coll["pass"] and seen_pt > SEEN_MARGIN)}


def control_planted_grid(planted: dict) -> dict:
    """planted = {kappa: (arm, arm_exclusion_deleted)}.  The control PASSES iff at least one planted strength is read AND collapses when the exclusion is deleted (the instrument can see a gap, and the
    exclusion line is what produces it).  The table of which kappa are read calibrates the instrument; a larger kappa is not always easier (labels saturate, the swap null grows)."""
    return planted_grid_from_per({kp: control_planted(a, m) for kp, (a, m) in planted.items()})


def planted_grid_from_per(per: dict) -> dict:
    """per = {kappa: control_planted() dict}.  The grid verdict; split out so the A1 calibration (which computes the planted worlds in separate processes) applies the SAME rule."""
    ok = [kp for kp in sorted(per) if per[kp]["pass"]]
    read = [kp for kp in sorted(per) if per[kp]["planted_gap_read"]]
    return {"per_kappa": {f"{kp:g}": v for kp, v in per.items()}, "kappas_read": read, "kappas_passing": ok, "smallest_kappa_read": (read[0] if read else None), "pass": bool(ok)}


# ====================================================================================================================== the decision rule (a pure function of numbers; pinned by tests)
def z_inv(p: float) -> float:
    return statistics.NormalDist().inv_cdf(p)


MDE_FACTOR = z_inv(1 - ALPHA_M) + z_inv(0.8)      # achieved MDE(80 %) = MDE_FACTOR x SE


def decide(primary: dict, replicate: dict, swap: dict, controls: dict, certified: dict | None = None) -> dict:
    """primary / replicate: {M: cluster_gap dict} of seed 0 / seed 1.  swap: swap_null() of the primary.  controls: {name: {'pass': bool}}.
    certified: {M: bool}; a measure whose shuffle control was not informative (M3, typically) is reported but cannot trigger GAP and cannot make a NULL POWERED.
    Order: INVALID (a control failed) > GAP > NULL (POWERED or UNDERPOWERED)."""
    failed = [k for k, v in controls.items() if not v.get("pass", False)]
    cert = {"M1": True, "M2": True, "M3": True, **(certified or {})}
    per, gap_any = {}, False
    for m in ("M1", "M2", "M3"):
        g = primary[m]
        if math.isnan(g.get("point", float("nan"))):
            per[m] = {"available": False, "gap": False}
            continue
        nu = swap[f"nu_{m}"]
        r = replicate[m]["point"]
        S = bool(g["lo_bonf"] > nu)
        Mg = bool(g["point"] >= DELTA[m])
        Rp = bool(r > 0 and r >= REPL_FRAC * g["point"])
        Ls = bool(g["loso_min"] > 0)
        gap_m = S and Mg and Rp and Ls and cert[m]
        per[m] = {"available": True, "certified": cert[m], "gap": gap_m, "reverse_gap": bool(g["hi_bonf"] < -nu), "separated_above_swap_null": S, "magnitude_ge_delta": Mg, "seed_replicate_agrees": Rp, "leave_one_session_out_stable": Ls,
                  "point": g["point"], "lo_bonf": g["lo_bonf"], "hi_bonf": g["hi_bonf"], "swap_null_nu": nu, "seed_replicate_point": r, "delta": DELTA[m],
                  "achieved_se": g["se"], "achieved_mde80": MDE_FACTOR * g["se"]}
        gap_any = gap_any or gap_m
    if failed:
        return {"outcome": "INVALID", "failed_controls": failed, "per_measure": per, "routing": "fix the failed control(s) (see DRAFT_PREREG_STAGE0.md section 7) and re-seal; do not read the gap"}
    if gap_any:
        return {"outcome": "GAP", "gap_measures": [m for m, v in per.items() if v.get("gap")], "per_measure": per, "routing": "Stage 1 (FEASIBILITY.md section 7): search + knobs K1-K4, ~30 CPU-h"}
    powered = all(per[m]["available"] and per[m]["certified"] and per[m]["hi_bonf"] < DELTA[m] for m in ("M1", "M3"))
    return {"outcome": "NULL", "sub": "POWERED" if powered else "UNDERPOWERED", "per_measure": per,
            "routing": ("pessimism cannot be tested on P2; the lever is D-SESSION-RESERVE (PI)" if powered else
                        "no gap detected but gaps of size >= delta are NOT excluded: the claim 'pessimism cannot be tested on P2' is not licensed; MM decides")}


# ====================================================================================================================== the run
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return o


def arm_summary(a: dict) -> dict:
    return {"gaps": a["gaps"], "swap_null": a["swap_null"], "seen_minus_oof": a["seen_minus_oof"], "levels": a["levels"], "fit_audit": a["fit"]["audit"], "seconds": a["seconds"]}


def run_all(w: World, d: Design, manifest: dict | None = None, log=print) -> dict:
    """Everything the real read does, on any World (real or synthetic).  Returns the RESULT dict."""
    t0 = time.time()
    idx = boot_indices(w.S)
    prim, repl = seed_sets()
    res = {"schema": "spess0-result-a1", "constants_sha256": constants_sha256(), "n_seeds": N_SEEDS, "primary_seeds": list(prim), "replicate_seeds": list(repl), "design": d.info, "world": w.meta, "inputs": manifest}
    log(f"[1/7] primary arm  R0 seeds {prim[0]}..{prim[-1]} (S = {N_SEEDS}, composite scores averaged)")
    a0 = run_arm(w, d, R0, prim, idx)
    log(f"[2/7] replicate    R0 seeds {repl[0]}..{repl[-1]}")
    a1 = run_arm(w, d, R0, repl, idx)
    log("[3/7] secondary    R1 (prox 0.3), descriptive")
    ar1 = run_arm(w, d, R1, prim, idx)
    log("[4/7] C-III real   exclusion line deleted")
    am = run_arm(w, d, R0, prim, idx, fit=fit_all_unexcluded(w, d, R0, prim))
    log("[5/7] C-I          shuffled labels (evaluation x100, and full pipeline x10)")
    c_shuf = control_shuffle(w, d, a0["cache"])
    c_shuf_world = control_shuffle_world(w, d, R0, idx, train_seeds=prim)
    log("[6/7] C-II-b      planted session effect on real features")
    planted = {}
    for kp in PLANT_KAPPAS:
        wp = w.with_labels(plant_labels(w, kp))
        planted[kp] = (run_arm(wp, d, R0, prim, idx), run_arm(wp, d, R0, prim, idx, fit=fit_all_unexcluded(wp, d, R0, prim)))
    c_plant = control_planted_grid(planted)
    log("[7/7] decision")
    controls = {
        "C-STRUCT": {"pass": all(x["own_fold_tokens"] == 0 and x["search_tokens"] == 0 and x["own_fold_sessions"] == 0 for x in a0["fit"]["audit"] + a1["fit"]["audit"] + ar1["fit"]["audit"]),
                     "note": "check_exclusion ran before every fit; a violation raises"},
        "C-I-eval": {"pass": c_shuf["pass"], **c_shuf},
        "C-I-trained": {"pass": c_shuf_world["pass"], **c_shuf_world},
        "C-II-a": control_seen(a0),
        "C-II-b": c_plant,
        "C-III-real": {"pass": control_collapse(a0, am)["pass"], **control_collapse(a0, am)},
    }
    dec = decide(a0["gaps"], a1["gaps"], a0["swap_null"], controls, certified={"M3": bool(c_shuf["informative"]["M3"])})
    res.update({"arms": {"R0_seed0": arm_summary(a0), "R0_seed1": arm_summary(a1), "R1_seed0_descriptive": arm_summary(ar1), "R0_seed0_exclusion_deleted": arm_summary(am),
                         **{f"PLANTED_kappa{kp:g}_R0_seed0": arm_summary(planted[kp][0]) for kp in PLANT_KAPPAS},
                         **{f"PLANTED_kappa{kp:g}_R0_seed0_exclusion_deleted": arm_summary(planted[kp][1]) for kp in PLANT_KAPPAS}},
                "controls": controls, "decision": dec, "seconds": time.time() - t0})
    p2 = run_arm(w, d, R0, 0, idx, fit={"models": p2_models(w), "audit": [], "recipe": None, "seed": None})   # P2's own head: single member, no training
    res["p2_reference_search_half"] = {"levels_OOF_role_is_P2": p2["levels"]["OOF"], "note": "P2's own banked head on the same search-half tokens; context only, never a gap"}
    return res


CRLF, LF = chr(13) + chr(10), chr(10)


def load_mutated(replace_markers: dict, name: str = "stage0_mutated"):
    """Re-import THIS file with the line carrying each marker replaced by the given line (textual mutation: the C-III arm and the tests run the REAL pipeline minus those lines).
    replace_markers: {marker comment: replacement code line (without indentation)}.  Each marker must occur on exactly one line."""
    import types
    lines = open(os.path.abspath(__file__), encoding="utf-8").read().replace(CRLF, LF).split(LF)
    for mk, new in replace_markers.items():
        hit = [i for i, ln in enumerate(lines) if ln.rstrip().endswith(mk) and not ln.lstrip().startswith(("#", '"'))]
        if len(hit) != 1:
            raise RuntimeError(f"mutation marker {mk!r} matched {len(hit)} lines, need exactly 1")
        ind = lines[hit[0]][:len(lines[hit[0]]) - len(lines[hit[0]].lstrip())]
        lines[hit[0]] = ind + new
    mod = types.ModuleType(name)
    mod.__file__ = os.path.abspath(__file__)
    sys.modules[name] = mod
    exec(compile(LF.join(lines), mod.__file__ + f"<{name}>", "exec"), mod.__dict__)
    return mod


MUT_NO_EXCLUDE = {"# EXCLUDE-LINE: model h never sees fold h": "train_folds = list(range(N_FOLDS))"}
MUT_NO_CHECK = {"# CHECK-LINE: structural C-STRUCT, fails closed": "chk = {'model': h, 'n_train': int(len(tr)), 'own_fold_tokens': -1, 'search_tokens': 0, 'own_fold_sessions': -1}"}


def fit_all_unexcluded(w: World, d: Design, rec: Recipe, seed) -> dict:
    """C-III arm: the REAL `fit_all` with the exclusion line deleted (and the structural check stubbed, because the arm exists to show what the exclusion buys).
    Controls only; never the primary arm.  Its audit rows carry own_fold_tokens = -1 so the C-STRUCT summary can never count it.  A1: `seed` may be the tuple of S seeds, averaged exactly like the primary arm."""
    mod = load_mutated({**MUT_NO_EXCLUDE, **MUT_NO_CHECK}, "stage0_no_exclusion")
    return mod.fit_all_seeds(w, d, rec, seed)


def check_prereg(path: str, sha_expected: str) -> dict:
    got = sha256_file(path)
    if got != sha_expected.strip().lower():
        raise RuntimeError(f"ABORT: the prereg file's sha256 {got} is not the sealed value {sha_expected}")
    txt = open(path, "r", encoding="utf-8").read()
    if constants_sha256() not in txt:
        raise RuntimeError(f"ABORT: the sealed prereg does not quote this code's constants_sha256 {constants_sha256()}: the code and the prereg disagree")
    return {"prereg_path": path, "prereg_sha256": got}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the real read (needs a sealed prereg)")
    r.add_argument("--prereg", required=True)
    r.add_argument("--prereg-sha256", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--artifacts", default=ARTIFACTS)
    sub.add_parser("constants", help="print the constants digest")
    c = sub.add_parser("check-inputs", help="shapes, counts and sha256 of the inputs; computes no statistic")
    c.add_argument("--artifacts", default=ARTIFACTS)
    a = ap.parse_args(argv)
    if a.cmd == "constants":
        print(json.dumps(constants_dict(), indent=1))
        print("constants_sha256", constants_sha256())
        return 0
    if a.cmd == "check-inputs":
        w, man = load_world(a.artifacts)
        d = make_design(w)
        print(json.dumps(jsonable({"shapes": {"X": w.X.shape, "Xp": w.Xp.shape, "sessions": w.S}, "design": d.info, "meta": w.meta, "inputs": man}), indent=1))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("ABORT: run under CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    seal = check_prereg(a.prereg, a.prereg_sha256)
    w, man = load_world(a.artifacts)
    d = make_design(w)
    if min(d.info["fold_search_tokens"]) < MIN_FOLD_TOKENS // 2:
        raise RuntimeError("ABORT: a fold has too few search tokens")
    res = run_all(w, d, man)
    res["seal"] = seal
    res["code_sha256"] = {fn: sha256_file_lf(os.path.join(HERE, fn)) for fn in ("stage0.py",)}
    res["env"] = {"python": sys.version.split()[0], "numpy": np.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(jsonable(res), open(a.out, "w"), indent=1)
    print("OUTCOME", res["decision"]["outcome"], res["decision"].get("sub", ""), "->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
