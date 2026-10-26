#!/usr/bin/env python3
"""v8f_kscore.py -- PREREG_V8F_KSCORE built exactly as written: a small scorer that READS THE SCENE and picks one of O's 7 longitudinal hypotheses.

B = the banked v8F-mini M2b checkpoint (seed 0 / seed 1), FROZEN. Only a ~0.2 M-parameter head trains. Nothing here was run on real data when it was written.

STAGES (run in this order, one python process at a time; every stage is resumable / idempotent)
  hyps  --seed S --split train|eval   O's 7 longitudinal hypotheses tau_k (T's own code with its longitudinal slice replaced by the one-hot of k, through
                                      ``forward_diag(code_override=...)``, as khyp.py) PLUS the scorer's other inputs (T's longitudinal softmax, B's adapted
                                      scene tokens at NOW, the ego token) for every window, STREAMED clip by clip into resumable shards (no merged file, no
                                      cache copy). train = the TRAINER's own window list (4,512 clips, 306,566 windows); eval = the eval-141 grid (2,399).
  fit   --seed S [--perm-labels]      the scorer: cross-entropy to k* = argmin_k ADE6(tau_k, GT) on the full training set; AdamW lr 1e-3, 3 epochs, batch 256.
                                      SEL-CONST (the single class with the lowest mean ADE6 over the TRAIN windows) is chosen here, per seed, before any eval.
                                      THE SCORER LAUNCH GATE runs inside fit, before the first optimiser step, and fit refuses to train without its PASS token.
  eval  --seeds 0 1                   the plan of each selector on eval-141 (KSCORE, the permuted-label scorer, the scorer FORCED to T's argmax) -> picks.
  read                                PRIMARY-B, PRIMARY-SEL, SECONDARY, beside; the PREREG outcome table applied verbatim. rc 2 on a failed stop-control or on
                                      ANY state the table does not map.
  mkjob / cmdline / assert-gate       box-2 helpers used by chain_v8f_kscore.sh (the job .cmd, the exact command line, the chain's gate assertions).

THE SCORER (PREREG): inputs, all from B's frozen forward at inference (vision + ego dynamics + nav; NO situation-classifier output and NO label -- the keys read are
asserted to be a subset of ``v8f_mini.INFERENCE_INPUT_KEYS``): the 7 plans tau_k (0.2 s knots, ego frame), T's longitudinal softmax, B's adapted scene tokens at NOW
(mean over the 40 tokens of each of the 4 frames), the ego token. Output: a softmax over k. Deployed plan: tau_{argmax g}.

DECLARED INTERPRETATIONS (each a choice the PREREG left open, fixed HERE, before any KSCORE number existed):
  1. "ADE6" = the mean over the 30 knots of the displacement to the GT (the eval's own ``ade6``); k* ties go to the lowest index (np.argmin); a SEL-CONST tie too.
  2. "the trainer's own windows" = ``D.train_windows`` over the TRAIN manifest minus ``--exclude 540:600`` (the RA-1 holdout), the SAME call ``v8f_mini_train.build_data``
     makes (n_windows_all_clips); B's ages are the eval's deterministic cycle (a_t = i mod 3, a_s = a_t + (i div 3) mod 5, i = global window index) on BOTH splits
     (the trainer samples ages at random; the cycle is the eval's own and khyp's).
  3. scene input = ``V8FMini._scene`` tokens at NOW (the tokens O reads, time embedding included) averaged over the n_tok tokens of each of the ``hist`` frames,
     [hist, d] = 4 x 256 = 1024 numbers; a frame whose ``frame_ok`` is False contributes 0. Ego input = ``_ego`` of O's tick (NOW), [d] = 256.
  4. Architecture (fixed): a shared plan encoder Linear(60 -> 128) + a hypothesis embedding + T's softmax weight per hypothesis; a context = GELU(scene_proj(scene) +
     ego_proj(ego) + b) with BIAS-FREE scene_proj / ego_proj (so a zeroed scene input kills the scene path EXACTLY); logit_k = MLP([h_k ; ctx]). 205,825 parameters.
     Inputs are standardised with TRAIN-window statistics (per-feature mean / std, stored in the scorer's buffers).
  5. Optimiser: AdamW lr 1e-3 constant, weight_decay 0.01 (torch's AdamW default; the PREREG fixes lr / epochs / batch only), 3 epochs, batch 256, the last batch of an
     epoch kept; epoch order = np.random.default_rng((seed, epoch)).permutation. Seeds 0 / 1 follow B's.
  6. Control 2 is ONE fixed permutation (seed 0) of the training k* across windows, the scorer trained with the same seed. Controls 1 and 3 and the harness identity
     are STOP-controls (rc 2); control 2 failing is FLAGGED (``controls_all_pass`` false, the outcome marked NOT ADMISSIBLE) and the table is still printed, as khyp.py.
  7. PRIMARY-B passes at a seed iff the UNROUNDED point mean(ADE6 KSCORE) - mean(ADE6 B) <= -F and the paired episode-cluster CI excludes 0; PRIMARY-SEL iff the point < 0
     and the CI excludes 0 (KSCORE - SEL-CONST). SECONDARY "not separated-worse" = ``v8f_beta_eval.paired_row``'s ``worse``; CONSISTENCY(a) 'any' is the per-window
     ``viol_any`` of the eval's own ``evaluate``; LEAD-VIOL is the AT-RISK share (``lead_metrics.paired_lead_viol(...).at_risk``, the key the SPEC bars, PI ruling
     2026-10-11) on the K = 30 B1 lead block, through ``v8f_dk_read.join_block``.
  8. The outcome table (PREREG + Amendment 1, 2026-10-25, tip ff58be9d), applied in this order: PRIMARY-B fails (at either seed) -> NOT PREDICTABLE; else PRIMARY-SEL fails ->
     BIAS, NOT SELECTION; else (both primaries pass) CONSISTENCY(a) 'any' or LEAD-VIOL separated-worse at either seed -> COSTS SAFETY (takes PRECEDENCE); else speed MAE or accel
     MAE separated-worse at either seed (those two only) -> COSTS DYNAMICS; else SELECTION PAYS. Every boolean state maps to exactly one outcome; a reading that is UNAVAILABLE
     (None) or a seed set other than {0, 1} is UNMAPPED, rc 2 (the read refuses rather than pick a label).
  9. A SEL-ARGMAX / SEL-CONST / forced-scorer plan is a VERTEX (one of the 7 hypotheses): ORACLE <= it on every window (hard control, as khyp's control 4).

Every number: MEASURED (ours) at the artifact it is written to; ORACLE = T0 diagnostic (never a result), selectors = T1-class open loop; estimator = paired episode-cluster
bootstrap, B = 2,000, seed 0, 141 episodes (``taniteval/ci.py``) -- it answers "would another draw of EPISODES say this?" and nothing about training runs, so both seeds ride
side by side with F. Hypothesis hyps/ code adapted from ``raw/2026-10-25-khyp/khyp.py`` (``lon_onehot_code``, ``enumerate_batch``, ``ade6_per_hyp``, ``pick``,
``arm_pw``, ``prow``, ``single``): the bytes of those functions are pinned against khyp's by test_v8f_kscore.py.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# commit charge: numpy's OpenBLAS reserves ~30 MB x logical CPUs at import (0.74 GB on a 24-thread box, 0.02 GB at 1 thread). BLAS is not the hot path (torch is).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
for _p in (REPO / "stack" / "scripts", REPO / "taniteval", REPO / "stack"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

PREREG = "PREREG_V8F_KSCORE"
#: the model module's own admissibility probe (binding rule 2026-08-03: the goal path and the situation path are information-disjoint at inference); fetched by name so that THIS file's
#: code names no situation classifier (test_v8f_kscore scans it with the model module's own scanner)
admissibility_check = getattr(M, "assert_no_situation_classifier_input")
K = M.N_LON                                   # 7 longitudinal classes
LON0 = M.N_GOAL + M.N_LAT                     # first longitudinal slot of T's code
DT = 0.2
N_KNOT = M.K30                                # 30 knots at 0.2 s
SEEDS = (0, 1)
F_PREREG = 0.1207                             # the PREREG's printed seed floor
B_BOOT = 2000
BANKED_B_ADE6 = {0: 2.1753, 1: 2.0546}        # the PREREG's control 3 (B's own-code ADE6, 4 dp)
CKPT_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
PINNED = {"n_clips": 4512, "n_excluded": 60, "n_windows": 306566, "eval_windows": 2399, "eval_episodes": 141}      # B's banked summary / the committed grid
FIT = {"lr": 1e-3, "epochs": 3, "batch": 256, "wd": 0.01, "hid": 128, "n_params": (100_000, 300_000)}             # PREREG: fixed in advance
TIER = {"ORACLE": "T0 diagnostic (reads the GT; headroom, never a result)", "selectors": "T1-class open loop (own held outputs, no GT read at inference)"}
ESTIMATOR = "paired episode-cluster bootstrap, B=2000, seed 0, 141 episodes (taniteval/ci.py)"
ARMS = ("B", "ARGMAX", "CONST", "KSCORE", "PERM", "FORCED")

# the PREREG outcome table: names are NAMED CONSTANTS so a test can swap two of them (a mutation) and see the mapping check go red
O_PAYS = "SELECTION PAYS"
O_BIAS = "BIAS, NOT SELECTION"
O_NOTPRED = "NOT PREDICTABLE"
O_COSTS = "COSTS SAFETY"
O_COSTS_DYN = "COSTS DYNAMICS"                                            # Amendment 1 (2026-10-25, before any KSCORE data)
NEXT = {  # the PREREG's committed next step, verbatim
    O_PAYS: "the scorer joins `forward_infer` as a K-hypothesis planner, and the dominance row is re-read on it against P10 and FLAT",
    O_BIAS: "the gain is a constant correction of T's longitudinal decision. Next: re-train T's longitudinal head with class-balanced loss (a T-side arm)",
    O_NOTPRED: "the best hypothesis is not predictable from these inputs at this scale. The longitudinal lever moves to O's speed profile (OACC's line)",
    O_COSTS: "reported; not deployed",
    O_COSTS_DYN: "reported; not deployed as is. Next: the same scorer re-read on top of OACC's base once OACC exists (descriptive until its own read)",
}


class Refused(SystemExit):
    """A guard fired: the stage refuses (rc 3 for the launch gate, rc 2 for the read)."""


def jdump(o, path):
    def enc(x):
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (set, tuple)):
            return list(x)
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(o, fh, indent=1, default=enc)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(o) -> str:
    return sha256_bytes(json.dumps(o, sort_keys=True, default=str).encode("utf-8"))


# ================================================================================================ 1. the hypotheses (adapted from khyp.py)
def lon_onehot_code(code0: torch.Tensor, k: int) -> torch.Tensor:
    """T's own code with the longitudinal slice replaced by the one-hot of class ``k``; goal, lateral and constraint slices stay T's own (khyp.py)."""
    c = code0.clone()
    c[:, LON0:LON0 + K] = 0.0
    c[:, LON0 + k] = 1.0
    return c


def scene_at_now(model, adapt: torch.Tensor, frame_ok: torch.Tensor) -> torch.Tensor:
    """B's adapted scene tokens at NOW (the tokens O reads), mean over each frame's tokens -> [B, hist, d]; a frame with ``frame_ok`` False contributes 0 (declared 3)."""
    cfg = model.cfg
    B = adapt.shape[0]
    last = torch.full((B,), adapt.shape[1] - 1, dtype=torch.long, device=adapt.device)
    toks, pad = model._scene(adapt, frame_ok, last)
    sc = toks.reshape(B, cfg.hist, cfg.n_tok, -1).mean(2)
    fpad = pad.reshape(B, cfg.hist, cfg.n_tok)[:, :, 0]
    return sc.masked_fill(fpad[..., None], 0.0)


def _featurize_t(model, b) -> dict:
    """One collated batch -> TORCH tensors (graph-connected iff called with grad enabled): 1 ``forward_infer`` (B's own plan and T's code), 7 ``forward_diag`` (tau_k),
    1 ``forward_diag`` with T's UNCHANGED code (the harness identity), and the scorer's scene / ego inputs. Reads only INFERENCE_INPUT_KEYS."""
    o0 = model.forward_infer(b)
    code0 = o0["code"]
    taus = []
    for k in range(K):
        ov = lon_onehot_code(code0, k)
        ok = model.forward_diag(b, ov)
        if not torch.equal(ok["code"], ov):
            raise AssertionError("O did not read the override: out['code'] != code_override")
        taus.append(ok["traj30"].float())
    idn = model.forward_diag(b, code0)
    guard = M._KeyGuard(b)                                              # the extras are read through a guard too
    last = torch.full((code0.shape[0],), o0["adapt"].shape[1] - 1, dtype=torch.long, device=code0.device)
    scene = scene_at_now(model, o0["adapt"], guard["frame_ok"])
    ego = model._ego(model._at(guard["p10f"], last))
    keys = set(o0["_keys_read"]) | guard.read
    bad = keys - M.INFERENCE_INPUT_KEYS
    if bad:
        raise AssertionError(f"the scorer's inputs read keys outside the admissible set: {sorted(bad)}")
    return {"tau": torch.stack(taus, 1), "own": o0["traj30"].float(), "p30": o0["p10"].float(), "p_lon": code0[:, LON0:LON0 + K].float(),
            "scene": scene.float(), "ego": ego.float(), "own_diag_maxabs": (idn["traj30"] - o0["traj30"]).abs().flatten(1).max(1).values,
            "keys_read": sorted(keys)}


featurize_t = _featurize_t                                               # the gate's positive control calls this one with grad enabled


def featurize_batch(model, b) -> dict:
    """The production path: no autograd graph is ever built, so B's parameters cannot receive a gradient (G-LIVE reads exactly 0.0)."""
    with torch.no_grad():
        o = featurize_t(model, b)
    return o


def _np(x, dtype=None):
    a = x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def ade6_per_hyp(tau, gt) -> np.ndarray:
    """[N, 7] per-window ADE6 of each hypothesis (khyp.py)."""
    return np.linalg.norm(np.asarray(tau, np.float64) - np.asarray(gt, np.float64)[:, None], axis=-1).mean(2)


def kstar_of(ade_k) -> np.ndarray:
    """k* = argmin_k ADE6 (ties -> the lowest index)."""
    return np.asarray(ade_k).argmin(1)


def pick(tau, k):
    return np.asarray(tau, np.float64)[np.arange(len(k)), np.asarray(k)]


def permute_labels(y, seed=0):
    """CONTROL 2: the labels permuted across windows (same multiset, features and labels decoupled)."""
    return np.asarray(y)[np.random.default_rng(seed).permutation(len(y))]


def hypotheses_distinct(tau, tol=1e-4) -> bool:
    tau = np.asarray(tau, np.float64)
    return all(np.abs(tau[:, i] - tau[:, j]).max() > tol for i in range(K) for j in range(i + 1, K))


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ck


def train_clip_list(cache, exclude: str) -> tuple:
    """The trainer's clip list (``v8f_mini_train.build_data``): the TRAIN manifest minus ``--exclude A:B`` of the sha12 order. -> (clips_all, info)."""
    man = D.read_manifest(cache, "train")
    clips_all = [c for c, _ in man]
    excluded, n_decl = [], 0
    if exclude:
        xlo, xhi = (int(x) for x in exclude.split(":"))
        n_decl = xhi - xlo
        excluded = sorted(clips_all, key=D.sha12)[xlo:xhi]
        ex = set(excluded)
        clips_all = [c for c in clips_all if c not in ex]
    info = {"n_clips": len(clips_all), "n_excluded": len(excluded), "n_excluded_declared": n_decl, "excluded_in_train": len(set(excluded) & set(clips_all)),
            "excluded_sha256": sha256_bytes(chr(10).join(excluded).encode("utf-8")) if excluded else None,
            "train_sha256": sha256_bytes(chr(10).join(sorted(clips_all, key=D.sha12)).encode("utf-8"))}
    return clips_all, info


def open_split(a, split, cfg, ck):
    """-> (wins, ds, info): the window list (the trainer's for train, the committed grid for eval) and the dataset the batches are cut from."""
    tracks = D.load_native_tracks(a.tracks) if cfg.kin_mode == "yaw" else None
    cache = a.eval_cache if split == "eval" else a.train_cache
    J = D.load_join(a.eval_join if split == "eval" else a.train_join)
    store = D.ClipStore(cache, split, lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode=cfg.kin_mode, tracks=tracks)
    if split == "eval":
        wins, info = D.eval_grid_windows(a.eval_grid, J), {}
    else:
        clips_all, info = train_clip_list(cache, a.exclude)
        wins = D.train_windows(store, J, clips_all, hist=cfg.hist, start=cfg.window_start)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    info["n_windows"] = len(wins)
    info["windows_sha256"] = sha256_bytes(np.asarray([(w[0], w[1]) for w in wins], np.int64).tobytes())
    return wins, ds, info


def batch_items(ds, wins, i0, i1):
    items = []
    for i in range(i0, i1):
        ci, t, _, _ = wins[i]
        a_t = i % 3                                              # the eval's own deterministic cycle, on the global window index (declared 2)
        a_s = a_t + (i // 3) % 5
        items.append(ds.item(ci, t, a_t, a_s))
    return items


def todev(b, dev):
    return {k: ({kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict) else v.to(dev)) for k, v in b.items()}


def enumerate_window_batch(model, ds, wins, i0, i1, dev, feat_dtype) -> dict:
    b = todev(D.collate(batch_items(ds, wins, i0, i1)), dev)
    f = featurize_batch(model, b)
    out = {"tau": _np(f["tau"], np.float32), "own": _np(f["own"], np.float32), "p30": _np(f["p30"], np.float32), "p_lon": _np(f["p_lon"], np.float32),
           "scene": _np(f["scene"], feat_dtype), "ego": _np(f["ego"], feat_dtype), "gt": _np(b["gt"], np.float32),
           "own_diag_maxabs": _np(f["own_diag_maxabs"], np.float32)}
    for k in ("ci", "t", "a_t", "a_s"):
        out[k] = b[k].cpu().numpy().astype(np.int32)
    out["_keys_read"] = f["keys_read"]
    return out


# ================================================================================================ shard store
def split_tag(seed, split):
    return f"M2b_s{seed}_{split}"


def hyps_paths(run_dir, seed, split):
    tag = split_tag(seed, split)
    h = Path(run_dir) / "hyps"
    return h / f"{tag}_shards", h / f"{tag}.meta.json"


def cmd_hyps(a):
    t0 = time.time()
    torch.set_num_threads(max(1, min(a.threads, 6)))
    dev = torch.device(a.device)
    ckpt = a.ckpt or str(Path(a.b_dir) / f"M2b_s{a.seed}" / "ckpt_last.pt")
    ck_sha = sha256_file(ckpt)
    if not a.allow_any_ckpt and ck_sha != CKPT_SHA[a.seed]:
        raise Refused(f"REFUSED: checkpoint sha256 {ck_sha} differs from the banked M2b_s{a.seed} checkpoint {CKPT_SHA[a.seed]}")
    model, cfg, ck = load_model(ckpt, dev)
    wins, ds, info = open_split(a, a.split, cfg, ck)
    N = len(wins)
    sdir, meta_path = hyps_paths(a.run_dir, a.seed, a.split)
    sdir.mkdir(parents=True, exist_ok=True)
    shard = max(a.batch, (a.shard // a.batch) * a.batch)
    fdt = np.float16 if a.feat_dtype == "float16" else np.float32
    print(f"[hyps {split_tag(a.seed, a.split)}] windows {N} batch {a.batch} shard {shard} threads {torch.get_num_threads()} device {dev} lru {a.lru}", flush=True)
    keys_seen = set()
    for s0 in range(0, N, shard):
        path = sdir / f"shard_{s0 // shard:05d}.npz"
        if path.exists():
            continue
        parts = []
        for b0 in range(s0, min(s0 + shard, N), a.batch):
            p = enumerate_window_batch(model, ds, wins, b0, min(b0 + a.batch, N, s0 + shard), dev, fdt)
            keys_seen |= set(p.pop("_keys_read"))
            parts.append(p)
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        tmp = sdir / f"tmp_{path.name}"
        np.savez(tmp, **cat)
        os.replace(tmp, path)
        print(f"[hyps {split_tag(a.seed, a.split)}] {min(s0 + shard, N)}/{N}  {time.time() - t0:.0f}s", flush=True)
    files = sorted(sdir.glob("shard_*.npz"))
    n_expect = math.ceil(N / shard)
    if len(files) != n_expect:
        raise Refused(f"REFUSED: {len(files)} shards on disk, {n_expect} expected")
    shards, n_tot, ident = [], 0, 0.0
    for f in files:
        with np.load(f) as z:
            n = len(z["ci"])
            ident = max(ident, float(z["own_diag_maxabs"].max()))
        shards.append({"name": f.name, "n": n, "sha256": sha256_file(f)})
        n_tot += n
    if n_tot != N:
        raise Refused(f"REFUSED: shards hold {n_tot} windows, the window list has {N}")
    peak = None
    try:
        import psutil
        pm = psutil.Process().memory_info()
        peak = {"peak_private_commit_gb": round(getattr(pm, "peak_pagefile", 0) / 2**30, 3), "peak_working_set_gb": round(getattr(pm, "peak_wset", 0) / 2**30, 3)}
    except Exception:
        pass
    meta = {"prereg": PREREG, "tag": split_tag(a.seed, a.split), "seed": a.seed, "split": a.split, "n_windows": N, "shard": shard, "n_shards": len(shards), "shards": shards,
            "ckpt": str(ckpt), "ckpt_sha256": ck_sha, "feat_dtype": a.feat_dtype, "device": str(dev), "info": info, "seconds": time.time() - t0, "torch_threads": torch.get_num_threads(),
            "ages": "cycle (a_t = i%3, a_s = a_t + (i//3)%5, i = global window index)", "tier": TIER["selectors"], "own_diag_maxabs_over_windows": ident,
            "keys_read": sorted(keys_seen), "kin_mode": cfg.kin_mode, "hist": cfg.hist, "d": cfg.d, "n_tok": cfg.n_tok, "memory": peak}
    jdump(meta, meta_path)                                                # written LAST: no meta, no data (fit / eval refuse)
    print(f"[hyps {meta['tag']}] DONE {time.time() - t0:.0f}s own_diag_maxabs {ident}", flush=True)
    return 0


def load_meta(run_dir, seed, split) -> dict:
    sdir, meta_path = hyps_paths(run_dir, seed, split)
    if not meta_path.exists():
        raise Refused(f"REFUSED: {meta_path} is missing -- the hyps stage has not finished for {split_tag(seed, split)}")
    return json.load(open(meta_path, encoding="utf-8"))


def load_split(run_dir, seed, split, keys=None, verify=True) -> tuple:
    """-> (arrays, meta). Concatenates the shards in order; ``verify`` re-hashes every shard (CONTENT, never presence)."""
    meta = load_meta(run_dir, seed, split)
    sdir, _ = hyps_paths(run_dir, seed, split)
    parts = []
    for s in meta["shards"]:
        p = sdir / s["name"]
        if verify and sha256_file(p) != s["sha256"]:
            raise Refused(f"REFUSED: shard {p} does not match the sha256 recorded in the meta")
        with np.load(p) as z:
            parts.append({k: z[k] for k in (keys or z.files)})
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if len(cat["ci"] if "ci" in cat else next(iter(cat.values()))) != meta["n_windows"]:
        raise Refused("REFUSED: the loaded shards do not hold the meta's window count")
    return cat, meta


# ================================================================================================ 2. the scorer
class KScorer(nn.Module):
    """g: (7 plans, T's longitudinal softmax, scene, ego) -> 7 logits. ``mode='forced_argmax'`` is PREREG control 1: the scorer FORCED to T's argmax (logits = T's softmax)."""

    def __init__(self, d=256, hist=4, hid=128, n_hyp=K, n_knot=N_KNOT, mode="learned"):
        super().__init__()
        if mode not in ("learned", "forced_argmax"):
            raise ValueError(f"mode must be learned|forced_argmax, got {mode!r}")
        self.d, self.hist, self.hid, self.n_hyp, self.n_knot, self.mode = int(d), int(hist), int(hid), int(n_hyp), int(n_knot), mode
        self.plan_proj = nn.Linear(2 * n_knot, hid)
        self.hyp_emb = nn.Parameter(torch.randn(n_hyp, hid) * 0.02)
        self.lon_proj = nn.Linear(1, hid, bias=False)
        self.scene_proj = nn.Linear(hist * d, hid, bias=False)        # bias-free: a zeroed scene input contributes EXACTLY 0 and its weights get EXACTLY 0 gradient
        self.ego_proj = nn.Linear(d, hid, bias=False)
        self.ctx_bias = nn.Parameter(torch.zeros(hid))
        self.mix = nn.Linear(2 * hid, hid)
        self.out = nn.Linear(hid, 1)
        for nm, n in (("tau", 2 * n_knot), ("scene", hist * d), ("ego", d)):
            self.register_buffer(f"{nm}_mu", torch.zeros(n))
            self.register_buffer(f"{nm}_sd", torch.ones(n))

    def set_stats(self, st: dict):
        for nm in ("tau", "scene", "ego"):
            getattr(self, f"{nm}_mu").copy_(torch.as_tensor(st[f"{nm}_mu"], dtype=torch.float32))
            getattr(self, f"{nm}_sd").copy_(torch.as_tensor(st[f"{nm}_sd"], dtype=torch.float32))

    def standardise(self, tau, scene, ego):
        B = tau.shape[0]
        return ((tau.reshape(B, self.n_hyp, -1).float() - self.tau_mu) / self.tau_sd, (scene.reshape(B, -1).float() - self.scene_mu) / self.scene_sd,
                (ego.reshape(B, -1).float() - self.ego_mu) / self.ego_sd)

    def forward_std(self, tau_s, p_lon, scene_s, ego_s):
        """The core on STANDARDISED inputs (tau_s [B,7,60], scene_s [B,hist*d], ego_s [B,d]); the scene-path regression zeroes ``scene_s`` here."""
        if self.mode == "forced_argmax":
            return p_lon.float()
        h = F.gelu(self.plan_proj(tau_s)) + self.hyp_emb[None] + self.lon_proj(p_lon.float()[..., None])
        ctx = F.gelu(self.scene_proj(scene_s) + self.ego_proj(ego_s) + self.ctx_bias)
        z = torch.cat([h, ctx[:, None].expand(-1, self.n_hyp, -1)], -1)
        return self.out(F.gelu(self.mix(z))).squeeze(-1)

    def forward(self, tau, p_lon, scene, ego):
        tau_s, scene_s, ego_s = self.standardise(tau, scene, ego)
        return self.forward_std(tau_s, p_lon, scene_s, ego_s)

    def built(self) -> dict:
        return {"mode": self.mode, "hid": self.hid, "d": self.d, "hist": self.hist, "n_hyp": self.n_hyp,
                "n_params": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
                "scene_proj_bias": self.scene_proj.bias is not None, "ego_proj_bias": self.ego_proj.bias is not None}


def build_scorer(hid, mode, d=256, hist=4):
    return KScorer(d=d, hist=hist, hid=hid, mode=mode)


@torch.no_grad()
def scorer_logits(scorer, arr, bs=4096) -> np.ndarray:
    scorer.eval()
    out = []
    for i in range(0, len(arr["tau"]), bs):
        sl = slice(i, i + bs)
        out.append(scorer(torch.as_tensor(arr["tau"][sl]), torch.as_tensor(arr["p_lon"][sl]), torch.as_tensor(arr["scene"][sl]), torch.as_tensor(arr["ego"][sl])).numpy())
    return np.concatenate(out).astype(np.float32)


def picks_from_logits(logits) -> np.ndarray:
    """The deployed selection k-hat = argmax g (ties -> the lowest index); the ONE function every selector's plan goes through."""
    return np.asarray(logits).argmax(1)


class Stats:
    """Streaming per-feature mean / std of the scorer's standardised inputs over the TRAIN windows."""

    def __init__(self, hist, d):
        self.n = 0
        self.s = {nm: np.zeros(n, np.float64) for nm, n in (("tau", 2 * N_KNOT), ("scene", hist * d), ("ego", d))}
        self.q = {nm: np.zeros(n, np.float64) for nm, n in (("tau", 2 * N_KNOT), ("scene", hist * d), ("ego", d))}
        self.nn = {"tau": 0, "scene": 0, "ego": 0}

    def add(self, tau, scene, ego):
        for nm, x in (("tau", np.asarray(tau).reshape(-1, 2 * N_KNOT)), ("scene", np.asarray(scene).reshape(len(scene), -1)), ("ego", np.asarray(ego).reshape(len(ego), -1))):
            x = x.astype(np.float64)
            self.s[nm] += x.sum(0)
            self.q[nm] += (x * x).sum(0)
            self.nn[nm] += len(x)

    def result(self) -> dict:
        out = {}
        for nm in ("tau", "scene", "ego"):
            mu = self.s[nm] / max(self.nn[nm], 1)
            sd = np.sqrt(np.maximum(self.q[nm] / max(self.nn[nm], 1) - mu * mu, 0.0))
            out[f"{nm}_mu"] = mu.astype(np.float32)
            out[f"{nm}_sd"] = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
        return out


# ================================================================================================ 3. the launch gate (scorer form)
#: every `fit` flag has an entry; a flag without one makes the gate REFUSE the launch (a new lever means a new check). Kinds: run = no effect on the scorer; the others name
#: the BUILT object the flag is compared with.
FLAG_CHECKS = {
    "run_dir": "run", "seed": "run:seed", "out": "run", "ckpt": "input:ckpt_sha256", "b_dir": "run", "train_cache": "run", "train_join": "run", "eval_cache": "run", "eval_join": "run",
    "eval_grid": "run", "eval_kin": "run", "tracks": "run", "kin_mode": "input:kin_mode", "exclude": "input:n_excluded", "lru": "run", "threads": "run", "device": "built:device",
    "lr": "built:lr", "epochs": "built:steps", "batch": "built:steps", "wd": "built:weight_decay", "hid": "built:hid", "scorer_mode": "built:mode", "perm_labels": "built:labels",
    "perm_seed": "built:labels", "log_every": "run", "no_resume": "run", "gate_only": "run", "require_token": "run", "token": "run", "allow_nonprereg": "run:pins",
    "commit": "run:token", "probe_n": "built:probe", "input_tol": "built:probe", "expect_clips": "input:n_clips", "expect_windows": "input:n_windows", "allow_any_ckpt": "run:pins",
}


def argv_canonical(a) -> dict:
    drop = {"gate_only", "require_token", "token", "commit", "threads", "log_every", "no_resume"}                 # run-mode / bookkeeping flags: the gate and the fit share one sha
    return {k: v for k, v in sorted(vars(a).items()) if k not in drop and not callable(v)}


def argv_sha(a) -> str:
    return sha256_json(argv_canonical(a))[:16]


def key_files() -> dict:
    fs = {"stack/scripts/v8f_kscore.py": HERE, "stack/tanitad/models/v8f_mini.py": REPO / "stack" / "tanitad" / "models" / "v8f_mini.py",
          "stack/tanitad/data/v8f_mini_data.py": REPO / "stack" / "tanitad" / "data" / "v8f_mini_data.py"}
    return {k: (sha256_file(v) if Path(v).exists() else None) for k, v in fs.items()}


def find_commit(a) -> str | None:
    if getattr(a, "commit", None):
        return a.commit
    if os.environ.get("KSCORE_COMMIT"):
        return os.environ["KSCORE_COMMIT"]
    tip = REPO / "TIP_V8FKSCORE.txt"
    if tip.exists():
        return tip.read_text(encoding="utf-8").strip().split()[0]
    try:
        r = subprocess.run(["git", "-C", str(REPO), "-c", "safe.directory=*", "rev-parse", "HEAD"], capture_output=True, text=True, encoding="utf-8", timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return None


def declared_vs_built(a, scorer, opt, plan, info, labels_true, labels_built) -> list:
    """G-DVB: every flag against what was BUILT. Returns the mismatches (empty = pass)."""
    bad = []
    dests = {x.dest for x in build_parser()._subparsers._group_actions[0].choices["fit"]._actions if x.dest != "help"}
    miss = sorted(dests - set(FLAG_CHECKS))
    if miss:
        bad.append(f"flags without a declared-vs-built entry (the gate refuses them): {miss}")
    stale = sorted(set(FLAG_CHECKS) - dests)
    if stale:
        bad.append(f"FLAG_CHECKS entries for flags the parser does not have: {stale}")
    bt = scorer.built()
    if bt["mode"] != a.scorer_mode:
        bad.append(f"scorer_mode: declared {a.scorer_mode} but built {bt['mode']}")
    if bt["hid"] != a.hid:
        bad.append(f"hid: declared {a.hid} but built {bt['hid']}")
    g = opt.param_groups[0]
    if type(opt).__name__ != "AdamW":
        bad.append(f"optimiser: AdamW required, built {type(opt).__name__}")
    if abs(g["lr"] - a.lr) > 1e-15:
        bad.append(f"lr: declared {a.lr} but built {g['lr']}")
    if abs(g["weight_decay"] - a.wd) > 1e-15:
        bad.append(f"wd: declared {a.wd} but built {g['weight_decay']}")
    held = {id(p) for gr in opt.param_groups for p in gr["params"]}
    mine = {id(p) for p in scorer.parameters() if p.requires_grad}
    if held != mine:
        bad.append(f"the optimiser holds {len(held)} tensors, the scorer has {len(mine)} trainable: the optimiser must hold EXACTLY the scorer's parameters (no B parameter)")
    dev = next(scorer.parameters()).device.type
    if dev != torch.device(a.device).type:
        bad.append(f"device: declared {a.device} but built {dev}")
    steps_exp = a.epochs * math.ceil(info["n_windows"] / a.batch)
    if plan["steps_total"] != steps_exp or plan["epochs"] != a.epochs or plan["batch"] != a.batch:
        bad.append(f"epochs / batch: declared {a.epochs} x {a.batch} -> {steps_exp} steps, built plan {plan}")
    same = bool(np.array_equal(labels_built, labels_true))
    if a.perm_labels:
        if not np.array_equal(labels_built, permute_labels(labels_true, a.perm_seed)):
            bad.append(f"perm_seed {a.perm_seed}: the built labels are not permute_labels(k*, {a.perm_seed})")
        if same:
            bad.append("perm_labels declared but the built labels ARE the true k*: a lever that is declared and does nothing")
        if not np.array_equal(np.sort(labels_built), np.sort(labels_true)):
            bad.append("perm_labels: the permuted labels are not a permutation of k* (different multiset)")
    elif not same:
        bad.append("perm_labels NOT declared but the built labels differ from the true k*")
    if info.get("n_probe") != min(a.probe_n, info["n_windows"]):
        bad.append(f"probe_n: declared {a.probe_n} but the probe holds {info.get('n_probe')} windows")
    if info.get("kin_mode") != a.kin_mode:
        bad.append(f"kin_mode: declared {a.kin_mode} but the hyps were built {info.get('kin_mode')}")
    if info.get("n_excluded") != info.get("n_excluded_declared"):
        bad.append(f"exclude {a.exclude!r}: {info.get('n_excluded')} excluded, {info.get('n_excluded_declared')} declared")
    if info.get("excluded_in_train"):
        bad.append(f"exclude {a.exclude!r}: {info['excluded_in_train']} excluded clips are still in the TRAIN set")
    if a.expect_clips and info.get("n_clips") != a.expect_clips:
        bad.append(f"expect_clips: declared {a.expect_clips} but the hyps cover {info.get('n_clips')}")
    if a.expect_windows and info["n_windows"] != a.expect_windows:
        bad.append(f"expect_windows: declared {a.expect_windows} but the hyps cover {info['n_windows']}")
    if info.get("ckpt_sha256") != info.get("ckpt_file_sha256"):
        bad.append(f"ckpt: the shards were built from {info.get('ckpt_sha256')}, --ckpt is {info.get('ckpt_file_sha256')}")
    if not a.allow_nonprereg:                                              # the PREREG pins: not a lever, a refusal
        pins = [("lr", a.lr, FIT["lr"]), ("epochs", a.epochs, FIT["epochs"]), ("batch", a.batch, FIT["batch"]), ("wd", a.wd, FIT["wd"]), ("hid", a.hid, FIT["hid"])]
        bad += [f"PREREG pin: {n} is {v}, the PREREG fixes {w}" for n, v, w in pins if v != w]
        lo, hi = FIT["n_params"]
        if scorer.mode == "learned" and not lo <= bt["n_params"] <= hi:
            bad.append(f"PREREG pin: {bt['n_params']} parameters outside the PREREG's 0.1-0.3 M")
        if info.get("n_clips") != PINNED["n_clips"] or info["n_windows"] != PINNED["n_windows"] or info.get("n_excluded") != PINNED["n_excluded"]:
            bad.append(f"PREREG pin: clips / excluded / windows {info.get('n_clips')} / {info.get('n_excluded')} / {info['n_windows']} != "
                       f"{PINNED['n_clips']} / {PINNED['n_excluded']} / {PINNED['n_windows']} (B's banked window set)")
        if info.get("ckpt_sha256") != CKPT_SHA[a.seed]:
            bad.append(f"PREREG pin: the checkpoint is not the banked M2b_s{a.seed}")
        if bt["scene_proj_bias"] or bt["ego_proj_bias"]:
            bad.append("the scene / ego projections carry a bias: a zeroed scene would no longer kill the scene path")
    return bad


def live_check(scorer, probe_raw, labels, a, B_model=None, probe_batch=None, feat=None, steps=3) -> dict:
    """G-LIVE on a COPY of the scorer, on REAL windows recomputed through B: CE finite; every scorer parameter receives a non-zero gradient; the SCENE-ZEROED control reads exactly
    0.0 on the scene path while the real scene reads non-zero; and B's parameters receive EXACTLY 0.0 gradient when the CE is back-propagated through the featuriser's own output
    (against a positive control: the same probe through a leaky, grad-enabled featuriser must read non-zero, so the zero is not a dead probe). ``feat`` = the production
    featuriser's output on ``probe_batch`` (TENSORS, so a featuriser that kept B's graph is seen)."""
    res = {"steps": steps}
    y = torch.as_tensor(np.asarray(labels, np.int64))
    inp = [torch.as_tensor(probe_raw[k]) for k in ("tau", "p_lon", "scene", "ego")]
    m = copy.deepcopy(scorer)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    acc = {n: 0.0 for n, p in m.named_parameters() if p.requires_grad}
    losses = []
    for s in range(steps):
        opt.zero_grad(set_to_none=False)
        loss = F.cross_entropy(m(*inp), y)
        if not torch.isfinite(loss):
            raise AssertionError(f"G-LIVE: cross-entropy is not finite at step {s}: {float(loss)}")
        losses.append(float(loss.detach()))
        loss.backward()
        for n, p in m.named_parameters():
            if p.grad is not None:
                acc[n] += float(p.grad.abs().sum())
        opt.step()
    dead = sorted(n for n, g in acc.items() if g == 0.0)
    res.update({"ce": losses, "dead_params": dead, "grad_abs_sum": acc})
    # the scene-zeroed control (a fresh copy, ONE backward): the scene projection reads exactly 0.0, every other parameter is alive
    m0 = copy.deepcopy(scorer)
    m0.zero_grad(set_to_none=False)
    tau_s, scene_s, ego_s = m0.standardise(inp[0], inp[2], inp[3])
    F.cross_entropy(m0.forward_std(tau_s, inp[1], torch.zeros_like(scene_s), ego_s), y).backward()
    g0 = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.0 for n, p in m0.named_parameters()}
    res["scene_zero_control_scene_proj_grad"] = g0["scene_proj.weight"]
    res["scene_zero_control_others_alive"] = bool(all(v > 0.0 for n, v in g0.items() if n != "scene_proj.weight"))
    m1 = copy.deepcopy(scorer)
    m1.zero_grad(set_to_none=False)
    F.cross_entropy(m1(*inp), y).backward()
    res["scene_real_scene_proj_grad"] = float(m1.scene_proj.weight.grad.abs().sum())
    ok = (not dead and res["scene_zero_control_scene_proj_grad"] == 0.0 and res["scene_real_scene_proj_grad"] > 0.0 and res["scene_zero_control_others_alive"])
    if B_model is not None:
        flags = [(p, p.requires_grad) for p in B_model.parameters()]
        digest = lambda: sha256_json({k: sha256_bytes(v.detach().cpu().numpy().tobytes()) for k, v in B_model.state_dict().items()})        # noqa: E731
        sd0 = digest()
        for p, _ in flags:
            p.requires_grad_(True)                         # the probe is NON-VACUOUS only if B's parameters COULD receive a gradient
        try:
            B_model.zero_grad(set_to_none=True)
            f = feat if feat is not None else featurize_batch(B_model, probe_batch)
            m1.zero_grad(set_to_none=False)
            F.cross_entropy(m1(f["tau"], f["p_lon"], f["scene"], f["ego"]), y).backward()          # tensors AS PRODUCED: a leaked graph would reach B's parameters here
            gB = sum(float(p.grad.abs().sum()) for p in B_model.parameters() if p.grad is not None)
            res["scorer_grad_on_B_inputs"] = {"scene_proj": float(m1.scene_proj.weight.grad.abs().sum()), "ego_proj": float(m1.ego_proj.weight.grad.abs().sum()),
                                              "plan_proj": float(m1.plan_proj.weight.grad.abs().sum())}
            B_model.zero_grad(set_to_none=True)
            m1.zero_grad(set_to_none=False)
            leaky = featurize_t(B_model, probe_batch)                                                 # POSITIVE CONTROL: grad enabled, graph connected to B
            F.cross_entropy(m1(leaky["tau"], leaky["p_lon"], leaky["scene"], leaky["ego"]), y).backward()
            gB_leaky = sum(float(p.grad.abs().sum()) for p in B_model.parameters() if p.grad is not None)
            B_model.zero_grad(set_to_none=True)
        finally:
            for p, rg in flags:
                p.requires_grad_(rg)
        res["B_grad_abs_sum"] = gB
        res["B_grad_positive_control_abs_sum"] = gB_leaky
        res["B_state_unchanged"] = bool(sd0 == digest())
        ok = ok and gB == 0.0 and gB_leaky > 0.0 and res["B_state_unchanged"] and res["scorer_grad_on_B_inputs"]["scene_proj"] > 0.0
    res["pass"] = bool(ok)
    return res


def input_identity(rec, stored, tol_tau=1e-3, tol_feat=5e-3, feat_dtype="float32") -> dict:
    """G-INPUTS: the features B produces NOW for the probe windows equal the shard rows (the stored arrays are what B makes).

    scene / ego are STORED in ``feat_dtype``: float16 rounds a value to a relative 2**-11, so B's float32 output differs from its own stored row by up to that much. The
    comparison therefore allows ``tol_feat + 2 * 2**-11 * |stored|`` on scene / ego when the shards are float16 (two ulp: a batch-composition change may flip one rounding
    boundary), and ``tol_feat`` when they are float32. tau and p_lon are always stored float32 and keep their absolute tolerances."""
    def diff(k):
        a = np.asarray(rec[k], np.float64).reshape(len(rec[k]), -1)
        b = np.asarray(stored[k], np.float64).reshape(len(stored[k]), -1)
        return a, b
    mx = lambda k: float(np.abs(np.subtract(*diff(k))).max())      # noqa: E731
    rtol = 2.0 * 2.0 ** -11 if feat_dtype == "float16" else 0.0
    def excess(k):
        a, b = diff(k)
        return float((np.abs(a - b) - rtol * np.abs(b)).max())
    r = {"tau_maxabs": mx("tau"), "scene_maxabs": mx("scene"), "ego_maxabs": mx("ego"), "p_lon_maxabs": mx("p_lon"), "feat_dtype": feat_dtype, "feat_rtol": rtol,
         "scene_excess": excess("scene"), "ego_excess": excess("ego")}
    r["pass"] = bool(r["tau_maxabs"] <= tol_tau and r["scene_excess"] <= tol_feat and r["ego_excess"] <= tol_feat and r["p_lon_maxabs"] <= 1e-4)
    return r


def token_expect(a, info, labels_built) -> dict:
    return {"prereg": PREREG, "gate": "PASS", "commit": find_commit(a), "argv_sha": argv_sha(a), "code_sha256": key_files(),
            "inputs": {"hyps_train_meta_sha256": info["meta_sha256"], "ckpt_sha256": info["ckpt_sha256"], "train_sha256": info.get("train_sha256"),
                       "windows_sha256": info.get("windows_sha256"), "n_windows": info["n_windows"], "labels_sha256": sha256_bytes(np.asarray(labels_built, np.int64).tobytes())},
            "seed": a.seed, "perm_labels": bool(a.perm_labels), "prereg_pinned": not a.allow_nonprereg}


def verify_token(path, expect: dict, allow_nonprereg=False):
    """The PASS token must exist and match the commit, the argv sha, the code and the input manifests it was written for; else REFUSE (rc 3)."""
    p = Path(path)
    if not p.exists():
        raise Refused(f"ZZDVB-REFUSEDZZ no launch-gate PASS token at {p}: fit will not train without one")
    tok = json.load(open(p, encoding="utf-8"))
    bad = []
    if tok.get("gate") != "PASS":
        bad.append(f"gate is {tok.get('gate')!r}")
    for k in ("prereg", "commit", "argv_sha", "code_sha256", "inputs", "seed", "perm_labels", "prereg_pinned"):
        if tok.get(k) != expect.get(k):
            bad.append(f"{k}: token {tok.get(k)!r} != expected {expect.get(k)!r}")
    if expect.get("commit") is None and not allow_nonprereg:
        bad.append("no commit could be established to bind the token to")
    if bad:
        raise Refused("ZZDVB-REFUSEDZZ the gate token does not match this run: " + "; ".join(bad))
    return tok


def run_gate(a, scorer, opt, plan, info, labels_true, labels_built, B_model, probe_batch, probe_stored, out) -> dict:
    """The scorer launch gate: G-DVB, G-LIVE, G-INPUTS, admissibility. Writes selfcheck.json always; the PASS token only on PASS."""
    sc = {"prereg": PREREG, "seed": a.seed, "perm_labels": bool(a.perm_labels), "levers": scorer.built(), "plan": plan, "steps_total": plan["steps_total"], "info": dict(info)}      # steps_total: v8f_guard.sh reads it top-level
    bad = declared_vs_built(a, scorer, opt, plan, info, labels_true, labels_built)
    sc["declared_vs_built"] = "PASS" if not bad else list(bad)
    if not bad:
        try:
            feat = featurize_batch(B_model, probe_batch)
            rec = {k: _np(feat[k], np.float32) for k in ("tau", "p_lon", "scene", "ego")}
            sc["inputs_identity"] = input_identity(rec, probe_stored, tol_tau=a.input_tol, feat_dtype=info.get("feat_dtype", "float32"))
            cpu_scorer = copy.deepcopy(scorer).cpu()
            sc["live"] = live_check(cpu_scorer, rec, labels_built[:len(rec["tau"])], a, B_model, probe_batch, feat)
            sc["admissibility"] = (admissibility_check(B_model.cpu().eval(), probe_batch) if device_is_cpu(B_model) else {"read": feat["keys_read"]})
            sc["keys_read"] = feat["keys_read"]
            sc["own_diag_maxabs"] = float(_np(feat["own_diag_maxabs"]).max())
        except AssertionError as e:
            sc["live"] = {"pass": False, "error": str(e)[:1500]}
        if not sc.get("inputs_identity", {}).get("pass"):
            bad.append(f"G-INPUTS: the stored arrays are not what B produces now: {sc.get('inputs_identity')}")
        if not sc.get("live", {}).get("pass"):
            bad.append(f"G-LIVE failed: {json.dumps({k: v for k, v in sc.get('live', {}).items() if k != 'grad_abs_sum'}, default=str)[:900]}")
        if sc.get("own_diag_maxabs", 1.0) > identity_tol(next(B_model.parameters()).device.type):
            bad.append(f"harness identity broken: forward_diag(T's own code) != forward_infer, max abs {sc.get('own_diag_maxabs')}")
    sc["bad"] = [str(b) for b in bad]
    sc["gate"] = "PASS" if not bad else "FAIL"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jdump(sc, out / "selfcheck.json")
    if bad:
        raise Refused("ZZDVB-REFUSEDZZ launch gate FAILED: " + "; ".join(str(b) for b in bad)[:2500])
    tok = token_expect(a, info, labels_built)
    tok["selfcheck_sha256"] = sha256_file(out / "selfcheck.json")
    jdump(tok, a.token)
    return sc


def identity_tol(device) -> float:
    """forward_diag(T's own code) == forward_infer: EXACTLY 0.0 on the CPU (khyp.py MEASURED 0.0 on 11,981 windows); 1e-4 m on a GPU, whose kernels need not be bitwise repeatable."""
    return 0.0 if str(device).startswith("cpu") else 1e-4


def device_is_cpu(model):
    return next(model.parameters()).device.type == "cpu"


# ================================================================================================ 4. fit
def load_train_arrays(a, meta) -> tuple:
    """Streams the train shards into preallocated arrays (verified by sha256), accumulating the TRAIN statistics and the per-class ADE sums for SEL-CONST."""
    N, hist, d = meta["n_windows"], meta["hist"], meta["d"]
    sdir, _ = hyps_paths(a.run_dir, a.seed, "train")
    fdt = np.float16 if meta["feat_dtype"] == "float16" else np.float32
    arr = {"tau": np.empty((N, K, N_KNOT, 2), np.float32), "p_lon": np.empty((N, K), np.float32), "scene": np.empty((N, hist * d), fdt), "ego": np.empty((N, d), fdt)}
    kstar = np.empty(N, np.int64)
    ade_sum, ade_n = np.zeros(K, np.float64), 0
    st = Stats(hist, d)
    i = 0
    for s in meta["shards"]:
        p = sdir / s["name"]
        if sha256_file(p) != s["sha256"]:
            raise Refused(f"REFUSED: shard {p} does not match the sha256 recorded in the meta")
        with np.load(p) as z:
            n = len(z["ci"])
            ade = ade6_per_hyp(z["tau"], z["gt"])
            kstar[i:i + n] = kstar_of(ade)
            ade_sum += ade.sum(0)
            ade_n += n
            arr["tau"][i:i + n], arr["p_lon"][i:i + n] = z["tau"], z["p_lon"]
            arr["scene"][i:i + n], arr["ego"][i:i + n] = z["scene"].reshape(n, -1), z["ego"]
            st.add(z["tau"], z["scene"], z["ego"])
            i += n
    if i != N:
        raise Refused(f"REFUSED: loaded {i} windows, the meta says {N}")
    mean_ade = ade_sum / ade_n
    return arr, kstar, st.result(), {"class_mean_ade6": mean_ade.tolist(), "n_windows": int(ade_n), "chosen_class": int(np.argmin(mean_ade))}


def fit_out_dir(a) -> Path:
    return Path(a.out) if a.out else Path(a.run_dir) / "fit" / f"s{a.seed}{'_perm' if a.perm_labels else ''}"


def cmd_fit(a):
    t0 = time.time()
    if a.scorer_mode != "learned":
        raise Refused("REFUSED: scorer_mode forced_argmax has no parameters to train; `eval` builds the forced scorer (CONTROL 1) itself")
    torch.set_num_threads(max(1, min(a.threads, 6)))
    out = fit_out_dir(a)
    out.mkdir(parents=True, exist_ok=True)
    if not a.token:
        a.token = str(out / "gate_token.json")
    meta = load_meta(a.run_dir, a.seed, "train")
    arr, kstar, stats, sel = load_train_arrays(a, meta)
    labels = permute_labels(kstar, a.perm_seed) if a.perm_labels else kstar
    N = meta["n_windows"]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device)
    scorer = build_scorer(a.hid, a.scorer_mode, d=meta["d"], hist=meta["hist"])
    scorer.set_stats(stats)
    scorer.to(dev)
    opt = torch.optim.AdamW(scorer.parameters(), lr=a.lr, weight_decay=a.wd)
    S = math.ceil(N / a.batch)
    plan = {"epochs": a.epochs, "batch": a.batch, "steps_per_epoch": S, "steps_total": a.epochs * S}
    ckpt = a.ckpt or str(Path(a.b_dir) / f"M2b_s{a.seed}" / "ckpt_last.pt")
    info = dict(meta["info"])
    info.update({"n_windows": N, "kin_mode": meta["kin_mode"], "ckpt_sha256": meta["ckpt_sha256"], "ckpt_file_sha256": sha256_file(ckpt),
                 "meta_sha256": sha256_file(hyps_paths(a.run_dir, a.seed, "train")[1]), "n_probe": min(a.probe_n, N), "feat_dtype": meta["feat_dtype"]})
    # the probe windows are RECOMPUTED through B (the same ``batch_items`` / ``featurize_batch`` as the hyps stage), so the gate sees the features B makes now, B's frozen-ness
    # and the scene path on REAL windows; they are compared with the shard rows of the same windows (G-INPUTS)
    B_model, cfg, ck = load_model(ckpt, a.device)
    wins, ds, _ = open_split(a, "train", cfg, ck)
    n_probe = info["n_probe"]
    pb = todev(D.collate(batch_items(ds, wins, 0, n_probe)), next(B_model.parameters()).device)
    probe_stored = {k: np.asarray(arr[k][:n_probe], np.float32) for k in ("tau", "p_lon", "scene", "ego")}
    a_probe = copy.copy(a)
    a_probe.probe_n = n_probe
    arrays = {k: arr[k] for k in ("tau", "p_lon", "scene", "ego")}
    jdump({"chosen_on": "train", "n_windows": sel["n_windows"], "chosen_class": sel["chosen_class"], "class_mean_ade6": sel["class_mean_ade6"], "kstar_counts": np.bincount(kstar, minlength=K).tolist(),
           "rule": "argmin_k mean over the TRAIN windows of ADE6(tau_k, GT); ties -> lowest index", "seed": a.seed}, out / "sel_const.json")
    # ---- the launch gate: runs BEFORE the first optimiser step; writes the token only on PASS (--require-token: the chain ran the gate already, this run only checks it)
    if not a.require_token:
        run_gate(a_probe, scorer, opt, plan, info, kstar, labels, B_model, pb, probe_stored, out)
    tok = verify_token(a.token, token_expect(a_probe, info, labels), a.allow_nonprereg)
    del B_model
    if a.gate_only:
        print("SELFCHECK PASS", json.dumps({"steps_total": plan["steps_total"], "n_windows": N, "n_params": scorer.built()["n_params"], "token": a.token}))
        return 0
    return train_loop(a, scorer, opt, arrays, labels, kstar, plan, out, tok, t0, dev)


def train_loop(a, scorer, opt, arrays, labels, kstar, plan, out, tok, t0, dev):
    """The training loop. ``tok`` is a VERIFIED token (verify_token returned it): a caller that has none cannot reach this line."""
    if not isinstance(tok, dict) or tok.get("gate") != "PASS":
        raise Refused("ZZDVB-REFUSEDZZ train_loop needs a verified PASS token")
    N = len(labels)
    y_all = torch.as_tensor(np.asarray(labels, np.int64))
    ck_path = out / "ckpt_last.pt"
    argsha = argv_sha(a)
    step, ep, bi = 0, 0, 0
    if ck_path.exists() and not a.no_resume:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if ck["argv_sha"] != argsha:
            raise Refused("ZZRESUME-REFUSEDZZ the checkpoint was written by a different argv")
        scorer.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, ep, bi = ck["step"], ck["epoch"], ck["batch_in_epoch"]
        print(f"[resume] step {step} epoch {ep} batch {bi}", flush=True)
    jdump({"argv": argv_canonical(a), "sha": argsha, "token_sha256": sha256_file(a.token)}, out / "argv.json")

    def save(final=False):
        torch.save({"model": scorer.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": ep, "batch_in_epoch": bi, "argv_sha": argsha, "cfg": scorer.built(),
                    "stats": {nm: getattr(scorer, nm).cpu().numpy() for nm in ("tau_mu", "tau_sd", "scene_mu", "scene_sd", "ego_mu", "ego_sd")}, "final": final}, str(ck_path) + ".tmp")
        os.replace(str(ck_path) + ".tmp", ck_path)

    logf = open(out / "train_log.jsonl", "a", encoding="utf-8", newline="\n")
    scorer.train()
    run_loss, run_acc, run_n = 0.0, 0.0, 0
    ts = time.time()
    while ep < a.epochs:
        perm = np.random.default_rng((a.seed, ep)).permutation(N)
        for b0 in range(bi * a.batch, N, a.batch):
            idx = np.sort(perm[b0:b0 + a.batch])
            y = y_all[idx]
            lg = scorer(torch.as_tensor(arrays["tau"][idx]).to(dev), torch.as_tensor(arrays["p_lon"][idx]).to(dev), torch.as_tensor(arrays["scene"][idx]).to(dev),
                        torch.as_tensor(arrays["ego"][idx]).to(dev))
            loss = F.cross_entropy(lg, y.to(dev))
            if not torch.isfinite(loss):
                raise AssertionError(f"cross-entropy is not finite at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step, bi = step + 1, bi + 1
            run_loss += float(loss.detach()) * len(idx)
            run_acc += float((lg.argmax(1).cpu() == y).float().sum())
            run_n += len(idx)
            if step % a.log_every == 0:
                logf.write(json.dumps({"step": step, "sec": time.time() - ts, "loss": run_loss / run_n, "acc": run_acc / run_n, "epoch": ep}) + "\n")
                logf.flush()
                run_loss, run_acc, run_n = 0.0, 0.0, 0
        ep, bi = ep + 1, 0
        save()
    if step != plan["steps_total"]:
        raise AssertionError(f"ran {step} steps, the plan says {plan['steps_total']}")
    save(final=True)
    scorer.eval()
    lg = scorer_logits(scorer.cpu(), {k: arrays[k] for k in ("tau", "p_lon", "scene", "ego")})
    kh = picks_from_logits(lg)
    summary = {"done": True, "prereg": PREREG, "seed": a.seed, "perm_labels": bool(a.perm_labels), "steps": step, "seconds": time.time() - t0, "argv_sha": argsha,
               "n_params": scorer.built()["n_params"], "train_acc_vs_labels": float((kh == np.asarray(labels)).mean()), "train_acc_vs_kstar": float((kh == kstar).mean()),
               "majority_class_acc_train": float((kstar == np.bincount(kstar, minlength=K).argmax()).mean()), "pick_counts_train": np.bincount(kh, minlength=K).tolist(),
               "token_sha256": sha256_file(a.token), "ckpt_sha256": sha256_file(ck_path), "gate": "PASS"}
    jdump(summary, out / "summary.json")                                  # LAST: no summary, no done (a supervisor that finds none restarts the run)
    print("FIT DONE", json.dumps({k: summary[k] for k in ("steps", "train_acc_vs_kstar", "majority_class_acc_train", "pick_counts_train")}), flush=True)
    return 0


# ================================================================================================ 5. eval
def load_scorer(run_dir, seed, perm, d, hist, mode="learned"):
    out = Path(run_dir) / "fit" / f"s{seed}{'_perm' if perm else ''}"
    sm = out / "summary.json"
    if not sm.exists() or not json.load(open(sm, encoding="utf-8")).get("done"):
        raise Refused(f"REFUSED: {sm} is missing / not done: the scorer did not finish training")
    ck = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    s = build_scorer(ck["cfg"]["hid"], mode, d=d, hist=hist)
    s.load_state_dict(ck["model"], strict=(mode == "learned"))
    return s.eval(), json.load(open(sm, encoding="utf-8")), out


def cmd_eval(a):
    torch.set_num_threads(max(1, min(a.threads, 6)))
    for seed in a.seeds:
        ev, meta = load_split(a.run_dir, seed, "eval")
        arr = {"tau": ev["tau"], "p_lon": ev["p_lon"], "scene": ev["scene"].reshape(len(ev["ci"]), -1), "ego": ev["ego"]}
        sc, summ, out = load_scorer(a.run_dir, seed, False, meta["d"], meta["hist"])
        pm, summ_p, _ = load_scorer(a.run_dir, seed, True, meta["d"], meta["hist"])
        fo = build_scorer(sc.hid, "forced_argmax", d=meta["d"], hist=meta["hist"]).eval()                 # CONTROL 1: the scorer FORCED to T's argmax
        sel = json.load(open(out / "sel_const.json", encoding="utf-8"))
        if sel.get("chosen_on") != "train":
            raise Refused("REFUSED: SEL-CONST was not chosen on the TRAIN windows")
        lg, lgp, lgf = scorer_logits(sc, arr), scorer_logits(pm, arr), scorer_logits(fo, arr)
        d = Path(a.run_dir) / "eval"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"kscore_s{seed}.npz", logits=lg, logits_perm=lgp, logits_forced=lgf, k_const=np.full(len(lg), sel["chosen_class"], np.int64),
                 ci=ev["ci"], t=ev["t"])
        jdump({"seed": seed, "n_windows": len(lg), "scorer_summary": summ, "perm_summary": summ_p, "const_class": sel["chosen_class"], "const_chosen_on": sel["chosen_on"],
               "scorer_ckpt_sha256": sha256_file(out / "ckpt_last.pt"), "eval_meta_sha256": sha256_file(hyps_paths(a.run_dir, seed, "eval")[1])}, d / f"kscore_s{seed}.json")
        print(f"[eval s{seed}] {len(lg)} windows; scorer picks {np.bincount(picks_from_logits(lg), minlength=K).tolist()}; const class {sel['chosen_class']}", flush=True)
    return 0


# ================================================================================================ 6. the read
class Stop(Exception):
    """A stop-control failed: the read refuses to interpret anything."""


class Unmapped(Exception):
    """A state the PREREG outcome table does not map."""


def decide(per_seed: dict) -> dict:
    """The PREREG outcome table, verbatim. ``per_seed[s]`` = {pb, ps, sec_lon_ok, sec_cl_ok} (booleans; ``None`` = the reading could not be made -> UNMAPPED)."""
    for s, r in per_seed.items():
        for k in ("pb", "ps", "sec_lon_ok", "sec_cl_ok"):
            if r.get(k) is None:
                raise Unmapped(f"seed {s}: the {k} reading is undefined (UNAVAILABLE): the table maps booleans only")
    if set(per_seed) != set(SEEDS):
        raise Unmapped(f"seeds {sorted(per_seed)} != {list(SEEDS)}: every bar is 'at BOTH seeds'")
    pb = all(r["pb"] for r in per_seed.values())
    ps = all(r["ps"] for r in per_seed.values())
    lon = all(r["sec_lon_ok"] for r in per_seed.values())
    cl = all(r["sec_cl_ok"] for r in per_seed.values())
    if not pb:
        o = O_NOTPRED
    elif not ps:
        o = O_BIAS
    elif not cl:
        o = O_COSTS                                                       # CONSISTENCY 'any' or LEAD-VIOL separated-worse at either seed: takes precedence over the dynamics row
    elif not lon:
        o = O_COSTS_DYN                                                   # Amendment 1: speed or accel MAE separated-worse ONLY (CONSISTENCY and LEAD-VIOL are not)
    else:
        o = O_PAYS
    return {"outcome": o, "PRIMARY_B_per_seed": {s: bool(r["pb"]) for s, r in per_seed.items()}, "PRIMARY_SEL_per_seed": {s: bool(r["ps"]) for s, r in per_seed.items()},
            "SECONDARY_lon_per_seed": {s: bool(r["sec_lon_ok"]) for s, r in per_seed.items()}, "SECONDARY_consistency_leadviol_per_seed": {s: bool(r["sec_cl_ok"]) for s, r in per_seed.items()},
            "F": F_PREREG, "bar_primary_b": -F_PREREG, "table": "PREREG_V8F_KSCORE.md + Amendment 1 (precedence COSTS SAFETY > COSTS DYNAMICS)", "committed_next_step_quoted": NEXT[o]}


def _eval_modules():
    import v8f_mini_eval as EV
    from taniteval import v8f_beta_eval as BE
    return EV, BE


def arm_pw(pred, gt, BE):
    """Per-window scalars of one arm: ADE / FDE at 1..6 s and the family components (taniteval's own ``family_windows``) (khyp.py)."""
    pred = np.asarray(pred, np.float64)
    d = np.linalg.norm(pred - gt, axis=-1)
    pw = {}
    for h in range(1, 7):
        pw[f"ade{h}"] = d[:, :5 * h].mean(1)
        pw[f"fde{h}"] = d[:, 5 * h - 1]
    for k, v in BE.family_windows(pred, gt, DT).items():
        pw["fw_" + k] = v
    return pw


LOWER_IS_BETTER = {"tac_lat_acc": False, "tac_lon_acc": False}


def prow(a, b, eid, BE, nb, key):
    r = BE.paired_row(a, b, eid, nb, lower_is_better=LOWER_IS_BETTER.get(key.replace("fw_", ""), True))
    if "mean" in r:
        ok = np.isfinite(a) & np.isfinite(b)
        r["point_unrounded"] = float(np.mean(np.asarray(a)[ok]) - np.mean(np.asarray(b)[ok]))
    return r


def single(v, eid, nb):
    from taniteval.ci import episode_cluster_bootstrap
    v, eid = np.asarray(v, np.float64), np.asarray(eid)
    ok = np.isfinite(v)
    r = episode_cluster_bootstrap(v[ok], eid[ok], n_boot=nb, seed=0)
    return {"mean": float(r["mean"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "n_windows": int(ok.sum())}


def _dk_reader(a):
    p = Path(a.dk_reader) if a.dk_reader else Path(_eval_modules()[0].PKG) / "code" / "v8f_dk_read.py"
    if not p.exists():
        raise Stop(f"the programme's distance-keeping reader {p} is missing: LEAD-VIOL cannot be read (the table maps no state without it)")
    sp = importlib.util.spec_from_file_location("v8f_dk_read", str(p))
    m = importlib.util.module_from_spec(sp)
    sys.modules["v8f_dk_read"] = m
    sp.loader.exec_module(m)
    return m


def lead_context(a, X, grid):
    """The K = 30 B1 lead block joined to the eval windows by the programme's own reader (speed proof and grid-coverage guards included)."""
    dk = _dk_reader(a)
    path = a.lead_block or str(dk.LEAD_BLOCK_K30)
    if not os.path.exists(path):
        raise Stop(f"the K = 30 lead block {path} is missing: LEAD-VIOL cannot be read")
    try:
        blk = dk.load_block(path)
        order = list(grid["order"])
        J = dk.join_block(blk, np.array([order[e] for e in X["eid"]]), 2 * np.asarray(X["t"], np.int64), np.asarray(X["kin"][:, 0], np.float64), k=N_KNOT, dt=DT)
    except dk.Refusal as e:
        raise Stop(f"lead block refused: {e}")
    return dk, blk, J


def lead_pack(plan, gt, J, eid, nb):
    from taniteval import lead_metrics as lm
    dk_plan = lm.distance_keeping(np.asarray(plan, np.float64)[:, :N_KNOT], J["leads"], J["lead_lens"], J["speeds"], DT)
    return dk_plan


def score_seed(a, s, X, grid, EV, BE, nb, lead):
    from taniteval import lead_metrics as lm
    ev, meta = load_split(a.run_dir, s, "eval")
    kz = np.load(Path(a.run_dir) / "eval" / f"kscore_s{s}.npz")
    kj = json.load(open(Path(a.run_dir) / "eval" / f"kscore_s{s}.json", encoding="utf-8"))
    sel = json.load(open(Path(a.run_dir) / "fit" / f"s{s}" / "sel_const.json", encoding="utf-8"))
    ctl = {}
    gt = X["gt"].astype(np.float64)
    eid = X["eid"]
    al = {"gt_maxabs": float(np.abs(ev["gt"].astype(np.float64) - gt).max()) if len(ev["gt"]) == len(gt) else None, "n": int(len(ev["gt"])), "n_expected": int(len(gt))}
    ctl["alignment"] = al
    if not (al["gt_maxabs"] is not None and al["gt_maxabs"] < 1e-4 and al["n"] == al["n_expected"]):
        raise Stop(f"alignment of the stored windows with the eval context failed: {al}")
    if not (np.array_equal(kz["ci"], ev["ci"]) and np.array_equal(kz["t"], ev["t"])):
        raise Stop("the eval picks were made on other windows than the stored hypotheses")
    if meta["info"].get("windows_sha256") is None:
        raise Stop("the eval meta has no window digest")
    ctl["harness_identity_maxabs_m"] = float(ev["own_diag_maxabs"].max())
    ctl["harness_identity_tol_m"] = identity_tol(meta.get("device", "cpu"))
    if ctl["harness_identity_maxabs_m"] > ctl["harness_identity_tol_m"]:
        raise Stop(f"harness identity broken: forward_diag(T's own code) != forward_infer, max abs {ctl['harness_identity_maxabs_m']}")
    B = ev["own"].astype(np.float64)
    d6_own = np.linalg.norm(B - gt, axis=-1).mean(1)
    ctl["control3_B_reproduces_banked"] = {"ade6_mean_mine": float(d6_own.mean()), "ade6_banked_4dp": BANKED_B_ADE6[s], "pass": bool(round(float(d6_own.mean()), 4) == BANKED_B_ADE6[s])}
    if a.no_banked:
        ctl["control3_B_reproduces_banked"] = {"skipped": "smoke (--no-banked)", "ade6_mean_mine": float(d6_own.mean())}
    elif not ctl["control3_B_reproduces_banked"]["pass"]:
        raise Stop(f"CONTROL 3 failed: B's own-code ADE6 {d6_own.mean():.7f} does not round to the PREREG's {BANKED_B_ADE6[s]}")
    if a.banked_own:
        z = np.load(a.banked_own.format(seed=s))
        r = {"traj30_maxabs_vs_banked_pred_m": float(np.abs(ev["own"].astype(np.float64) - z["pred"].astype(np.float64)).max()), "stop_bar_m": 1e-3}
        r["pass"] = bool(r["traj30_maxabs_vs_banked_pred_m"] <= 1e-3)
        ctl["control3_per_window"] = r
        if not r["pass"]:
            raise Stop(f"CONTROL 3 (per window) failed: {r}")
    tau = ev["tau"]
    ade_k = ade6_per_hyp(tau, gt)
    kev = kstar_of(ade_k)
    kart = np.asarray(ev["p_lon"]).argmax(1)                                      # SEL-ARGMAX, an expression independent of the scorer's pick path
    k_sc, k_pm, k_fo = (picks_from_logits(kz[n]) for n in ("logits", "logits_perm", "logits_forced"))
    k_const = np.full(len(kart), int(sel["chosen_class"]), np.int64)
    if not np.array_equal(kz["k_const"], k_const):
        raise Stop("the eval's SEL-CONST class is not the one fit chose on the train windows")
    if int(np.argmin(sel["class_mean_ade6"])) != int(sel["chosen_class"]) or sel.get("chosen_on") != "train":
        raise Stop("sel_const.json: the chosen class is not the argmin of its own train class means / was not chosen on train")
    ora = ade_k[np.arange(len(kev)), kev]
    c4 = {n: int((ora > ade_k[np.arange(len(k)), k] + 1e-12).sum()) for n, k in (("ARGMAX", kart), ("CONST", k_const), ("KSCORE", k_sc), ("PERM", k_pm), ("FORCED", k_fo))}
    ctl["control4_oracle_le_vertex_selectors"] = {"n_violations": c4, "pass": bool(all(v == 0 for v in c4.values()))}
    if not ctl["control4_oracle_le_vertex_selectors"]["pass"]:
        raise Stop(f"CONTROL 4 failed: ORACLE is worse than a vertex-picking selector: {c4}")
    ctl["hypotheses_distinct"] = bool(hypotheses_distinct(tau))
    plans = {"B": B, "ARGMAX": pick(tau, kart), "CONST": pick(tau, k_const), "KSCORE": pick(tau, k_sc), "PERM": pick(tau, k_pm), "FORCED": pick(tau, k_fo), "ORACLE": pick(tau, kev)}
    # CONTROL 1: the scorer forced to T's argmax IS SEL-ARGMAX: identical picks, bit-identical plans, a paired delta of exactly 0.0
    c1 = {"picks_equal": bool(np.array_equal(k_fo, kart)), "plans_bit_equal": bool(np.array_equal(plans["FORCED"], plans["ARGMAX"]))}
    ctl["control1_forced_is_sel_argmax"] = c1
    PW, R = {}, {"arms": {}, "paired_vs_B": {}, "paired_vs_CONST": {}, "ctl": ctl}
    for name, pl in plans.items():
        PW[name] = arm_pw(pl, gt, BE)
    for name in ("B", "ARGMAX", "CONST", "KSCORE", "PERM", "FORCED"):
        res, pw = EV.evaluate(plans[name], X, None, nb, with_families=(name in ("B", "KSCORE", "CONST", "ARGMAX")))
        R["arms"][name] = {"evaluate": res, "ade6_single": single(PW[name]["ade6"], eid, nb)}
        PW[name]["viol_any"] = pw["viol_any"].astype(np.float64)
    c1["paired_ade6_forced_minus_argmax"] = prow(PW["FORCED"]["ade6"], PW["ARGMAX"]["ade6"], eid, BE, nb, "ade6")
    c1["pass"] = bool(c1["picks_equal"] and c1["plans_bit_equal"] and c1["paired_ade6_forced_minus_argmax"].get("point_unrounded") == 0.0
                      and c1["paired_ade6_forced_minus_argmax"].get("lo") == 0.0 and c1["paired_ade6_forced_minus_argmax"].get("hi") == 0.0)
    if not c1["pass"]:
        raise Stop(f"CONTROL 1 failed: the scorer forced to T's argmax does not reproduce SEL-ARGMAX exactly: {c1}")
    # LEAD-VIOL on the K = 30 block
    dk, blk, J = lead
    hdk = lm.distance_keeping(gt[:, :N_KNOT], J["leads"], J["lead_lens"], J["speeds"], DT)
    DKS, LVS = {}, {}
    for name in ("B", "ARGMAX", "CONST", "KSCORE", "PERM", "FORCED"):
        if not a.smoke:
            dk.check_pred(name, plans[name], gt, np.asarray(X["kin"][:, 0], np.float64), N_KNOT, None)
        DKS[name] = lm.distance_keeping(plans[name][:, :N_KNOT], J["leads"], J["lead_lens"], J["speeds"], DT)
        lv = lm.lead_viol(DKS[name], hdk, eid, lead_valid=J["is_lead"], n_boot=nb, seed=0, plan_arc=lm.path_arc_length(plans[name][:, :N_KNOT]), human_arc=lm.path_arc_length(gt[:, :N_KNOT]))
        pwl = lv.pop("_per_window")
        LVS[name] = {"_per_window": pwl, **lv}
    for name in ("ARGMAX", "CONST", "KSCORE", "PERM"):
        R["paired_vs_B"][name] = {k: prow(PW[name][k], PW["B"][k], eid, BE, nb, k) for k in PW[name]}
    for name in ("KSCORE", "PERM", "ARGMAX", "B"):
        R["paired_vs_CONST"][name] = {k: prow(PW[name][k], PW["CONST"][k], eid, BE, nb, k) for k in ("ade6", "ade2")}
    R["paired_vs_B"]["ORACLE"] = {k: prow(PW["ORACLE"][k], PW["B"][k], eid, BE, nb, k) for k in ("ade6", "ade2")}
    pl_perm = lm.paired_lead_viol(LVS["PERM"], LVS["CONST"], eid, names=("PERM", "CONST"), n_boot=nb, seed=0)
    c2 = R["paired_vs_CONST"]["PERM"]["ade6"]
    ctl["control2_perm_scorer_vs_sel_const"] = {"perm_minus_const_ade6": {k: c2.get(k) for k in ("point_unrounded", "mean", "lo", "hi", "separated")}, "pass": bool(not c2.get("separated", True)),
                                                "bar": "a scorer trained on permuted k* must NOT separate from SEL-CONST"}
    plv = lm.paired_lead_viol(LVS["KSCORE"], LVS["B"], eid, names=("KSCORE", "B"), n_boot=nb, seed=0)
    plv_sel = lm.paired_lead_viol(LVS["KSCORE"], LVS["CONST"], eid, names=("KSCORE", "CONST"), n_boot=nb, seed=0)
    R["lead_viol_kscore_minus_B"] = {"all_lead_windows": {k: plv.get(k) for k in ("n_lead_windows", "n_viol_a", "n_viol_b", "delta", "lo", "hi", "separated")}, "at_risk": plv.get("at_risk"),
                                     "progress": plv.get("progress")}
    R["lead_viol_kscore_minus_const"] = {"at_risk": plv_sel.get("at_risk")}
    R["lead_viol_perm_minus_const"] = {"at_risk": pl_perm.get("at_risk")}
    dkp = lm.paired_distance_keeping(DKS["KSCORE"], DKS["B"], eid, names=("KSCORE", "B"), n_boot=nb, seed=0)["metrics"]
    R["distance_keeping_kscore_minus_B"] = dkp
    # the readings
    pb_row, ps_row = prow(PW["KSCORE"]["ade6"], PW["B"]["ade6"], eid, BE, nb, "ade6"), prow(PW["KSCORE"]["ade6"], PW["CONST"]["ade6"], eid, BE, nb, "ade6")
    sp, ac = R["paired_vs_B"]["KSCORE"]["fw_lon_speed_mae"], R["paired_vs_B"]["KSCORE"]["fw_lon_accel_mae"]
    cons = R["paired_vs_B"]["KSCORE"]["viol_any"]
    ar = plv.get("at_risk") or {}
    lead_row = {k: ar.get(k) for k in ("status", "delta", "lo", "hi", "separated", "n_at_risk", "n_viol_a", "n_viol_b")}
    lead_row["worse"] = bool(ar.get("status") == "OK" and ar.get("separated") and ar.get("delta", 0) > 0) if ar.get("status") == "OK" else None
    R["PRIMARY_B"] = {"read": "ADE6 KSCORE - B", "bar": f"<= -{F_PREREG} and separated", **{k: pb_row.get(k) for k in ("point_unrounded", "mean", "lo", "hi", "separated")}}
    R["PRIMARY_SEL"] = {"read": "ADE6 KSCORE - SEL-CONST", "bar": "< 0 and separated", **{k: ps_row.get(k) for k in ("point_unrounded", "mean", "lo", "hi", "separated")}}
    R["SECONDARY"] = {"speed_mae": {k: sp.get(k) for k in ("point_unrounded", "mean", "lo", "hi", "separated", "worse")}, "accel_mae": {k: ac.get(k) for k in ("point_unrounded", "mean", "lo", "hi", "separated", "worse")},
                      "consistency_a_any": {k: cons.get(k) for k in ("point_unrounded", "mean", "lo", "hi", "separated", "worse")}, "lead_viol_at_risk": lead_row}
    fin = lambda r: r.get("point_unrounded") is not None and np.isfinite(r["point_unrounded"])        # noqa: E731
    R["seed_row"] = {"pb": (bool(pb_row["point_unrounded"] <= -F_PREREG and pb_row["separated"]) if fin(pb_row) else None),
                     "ps": (bool(ps_row["point_unrounded"] < 0 and ps_row["separated"]) if fin(ps_row) else None),
                     "sec_lon_ok": (bool((not sp.get("worse", True)) and (not ac.get("worse", True))) if fin(sp) and fin(ac) else None),
                     "sec_cl_ok": (bool((not cons.get("worse", True)) and (lead_row["worse"] is False)) if fin(cons) and lead_row["worse"] is not None else None)}
    kmaj = int(np.argmax(sel["kstar_counts"]))                                    # the train-majority class of k*
    R["selector_accuracy"] = {"KSCORE_vs_kstar": float((k_sc == kev).mean()), "PERM_vs_kstar": float((k_pm == kev).mean()), "ARGMAX_vs_kstar": float((kart == kev).mean()),
                              "CONST_vs_kstar": float((k_const == kev).mean()), "majority_train_class": kmaj, "majority_train_class_acc_on_eval": float((kev == kmaj).mean()),
                              "kstar_eval_counts": np.bincount(kev, minlength=K).tolist(), "kstar_train_counts": sel["kstar_counts"], "pick_counts_KSCORE": np.bincount(k_sc, minlength=K).tolist(),
                              "pick_counts_ARGMAX": np.bincount(kart, minlength=K).tolist(), "const_class": int(sel["chosen_class"]), "n_eval": int(len(kev)),
                              "train_acc_vs_kstar": kj["scorer_summary"].get("train_acc_vs_kstar"), "train_majority_acc": kj["scorer_summary"].get("majority_class_acc_train")}
    R["B_ade6_mean"] = float(d6_own.mean())
    R["stack_H"] = stack_h(a, s, ev, plans, X, EV, BE, nb, eid, gt)
    R["stack_HS"] = {"status": "NOT-INSTRUMENTED", "reason": "HS is PREREG_V8F_HSPEC's coefficient-shrunk operator; this harness imports no HSPEC code. Descriptive row only: a work item, not a pass."}
    return R, plans


def stack_h(a, s, ev, plans, X, EV, BE, nb, eid, gt):
    """Descriptive: H (HBLEND's banked per-step shrinkage, P + w * (plan - P), P = the checkpoint's own frozen P10) on B and on KSCORE. Needs --h-weights."""
    if not a.h_weights:
        return {"status": "NOT-RUN", "reason": "no --h-weights given (descriptive row)"}
    try:
        w = json.load(open(a.h_weights.format(seed=s), encoding="utf-8"))
        W = np.stack([np.asarray(w["w_along_by_step"], np.float64), np.asarray(w["w_cross_by_step"], np.float64)], 1)
        P = ev["p30"].astype(np.float64)
        hb, hk = P + W * (plans["B"] - P), P + W * (plans["KSCORE"] - P)
        r = {k: prow(arm_pw(hk, gt, BE)[k], arm_pw(hb, gt, BE)[k], eid, BE, nb, k) for k in ("ade2", "ade6")}
        return {"status": "OK", "H_KSCORE_minus_H_B": r}
    except Exception as e:                                                         # recorded, not swallowed into a pass
        return {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}


def fmt(r, nd=4):
    if r is None or r.get("mean") is None:
        return "n/a"
    return f"{r['mean']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]" + (" *" if r.get("separated") else "")


def write_md(path, READ, RES):
    L = []
    o = READ["decision"]
    L.append("# KSCORE_RESULT: PREREG_V8F_KSCORE executed exactly as written\n")
    L.append(f"**Outcome (the PREREG table applied verbatim): {o['outcome']}**{'' if READ['controls_all_pass'] else ' -- CONTROL 2 FAILED: NOT ADMISSIBLE, the Master Mind adjudicates'}. "
             f"PRIMARY-B per seed {o['PRIMARY_B_per_seed']}; PRIMARY-SEL per seed {o['PRIMARY_SEL_per_seed']}; SECONDARY (speed/accel) per seed {o['SECONDARY_lon_per_seed']}; "
             f"SECONDARY (CONSISTENCY, LEAD-VIOL) per seed {o['SECONDARY_consistency_leadviol_per_seed']}.\n")
    L.append(f"**Committed next step, quoted from the PREREG:** *\"{o['committed_next_step_quoted']}\"*\n")
    L.append(f"Tiers: ORACLE = {TIER['ORACLE']}; selectors = {TIER['selectors']}. Estimator: {ESTIMATOR}; answers 'another draw of EPISODES' only, so both seeds are shown. "
             f"F = {F_PREREG} m (PREREG), exact |ADE6(s0)-ADE6(s1)| of B = {READ['F_exact']:.5f} m.\n")
    for s in SEEDS:
        R = RES[s]
        L.append(f"\n## Seed {s}\n")
        L.append("| read | value (m) | CI | bar |\n|---|---|---|---|")
        L.append(f"| PRIMARY-B ADE6 KSCORE - B | {R['PRIMARY_B']['point_unrounded']:+.4f} | {fmt(R['PRIMARY_B'])} | <= -{F_PREREG} and separated |")
        L.append(f"| PRIMARY-SEL ADE6 KSCORE - SEL-CONST | {R['PRIMARY_SEL']['point_unrounded']:+.4f} | {fmt(R['PRIMARY_SEL'])} | < 0 and separated |")
        for k, nm in (("speed_mae", "speed MAE"), ("accel_mae", "accel MAE"), ("consistency_a_any", "CONSISTENCY(a) 'any'")):
            x = R["SECONDARY"][k]
            L.append(f"| SECONDARY {nm} KSCORE - B | {x['point_unrounded']:+.4f} | {fmt(x)} | not separated-worse (worse = {x.get('worse')}) |")
        x = R["SECONDARY"]["lead_viol_at_risk"]
        L.append(f"| SECONDARY LEAD-VIOL (at-risk share) KSCORE - B | {x.get('delta')} | {fmt({'mean': x.get('delta'), 'lo': x.get('lo'), 'hi': x.get('hi'), 'separated': x.get('separated')})} | not separated-worse (worse = {x.get('worse')}) |")
        L.append("\n| arm (ADE6, m) | level [CI] | arm - B (paired) | arm - SEL-CONST (paired) |\n|---|---|---|---|")
        for nm in ("B", "ARGMAX", "CONST", "KSCORE", "PERM", "FORCED"):
            lv = R["arms"][nm]["ade6_single"]
            pv, pc = R["paired_vs_B"].get(nm), R["paired_vs_CONST"].get(nm)
            L.append(f"| {nm} | {lv['mean']:.4f} [{lv['lo']:.4f}, {lv['hi']:.4f}] | {fmt(pv['ade6']) if pv else '-'} | {fmt(pc['ade6']) if pc else '-'} |")
        L.append("\nFour families, KSCORE - B (paired; lower is better except the two tactical-agreement rows):\n")
        L.append("| family | metric | KSCORE - B |\n|---|---|---|")
        for key, fam, nm in (("fw_lon_speed_mae", "LONGITUDINAL", "speed MAE"), ("fw_lon_accel_mae", "LONGITUDINAL", "accel MAE"), ("fw_lon_along_mae", "LONGITUDINAL", "along MAE"),
                             ("fw_lat_cross_mae", "LATERAL", "cross-track MAE"), ("fw_lat_heading_mae_deg", "LATERAL", "heading MAE (deg)"),
                             ("fw_lat_yaw_rate_mae_degps", "LATERAL", "yaw-rate MAE (deg/s)"), ("fw_lat_curvature_mae_1pm", "LATERAL", "curvature MAE (1/m)"),
                             ("fw_tac_lat_acc", "TACTICAL (plan-derived)", "lateral decision agreement"), ("fw_tac_lon_acc", "TACTICAL (plan-derived)", "longitudinal decision agreement")):
            L.append(f"| {fam} | {nm} | {fmt(R['paired_vs_B']['KSCORE'].get(key), 4)} |")
        L.append("| STRATEGIC | - | NOT PAUSED for v8F (PI-V8F-2026-10-17; R72): S is B's own in every arm here (the scorer changes only which of O's hypothesis plans is deployed), so the family is identical across arms by construction; not a pass |")
        sa = R["selector_accuracy"]
        L.append(f"\nSelector accuracy vs k*: KSCORE {sa['KSCORE_vs_kstar']:.4f}, PERM {sa['PERM_vs_kstar']:.4f}, ARGMAX {sa['ARGMAX_vs_kstar']:.4f}, SEL-CONST (class {sa['const_class']}) "
                 f"{sa['CONST_vs_kstar']:.4f}, majority train class {sa['majority_train_class']} {sa['majority_train_class_acc_on_eval']:.4f}. k* counts (eval) {sa['kstar_eval_counts']}; "
                 f"KSCORE picks {sa['pick_counts_KSCORE']}.")
        L.append(f"\nControls (seed {s}): {json.dumps(R['ctl'], default=lambda x: x.item() if hasattr(x, 'item') else str(x))[:3500]}")
        L.append(f"\nStack: H {json.dumps(R['stack_H'], default=str)[:600]}; HS {R['stack_HS']['status']} ({R['stack_HS']['reason']})")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def cmd_read(a):
    torch.set_num_threads(max(1, min(a.threads, 6)))
    EV, BE = _eval_modules()
    grid = json.load(open(a.eval_grid))
    J = D.load_join(a.eval_join)
    X = D.eval_context(grid, a.eval_kin, J, kin_mode=a.kin_mode)
    run = Path(a.run_dir)
    rc_path = run / "READ.json"
    RES, PLANS, stops = {}, {}, {}
    try:
        n_win, n_ep = len(X["gt"]), len(set(np.asarray(X["eid"]).tolist()))
        if not a.smoke and (n_win, n_ep) != (PINNED["eval_windows"], PINNED["eval_episodes"]):
            raise Stop(f"the eval grid is {n_win} windows / {n_ep} episodes, the PREREG's is {PINNED['eval_windows']} / {PINNED['eval_episodes']}")
        lead = lead_context(a, X, grid)
    except Stop as e:
        jdump({"status": "STOPPED", "stops": {"setup": str(e)}}, rc_path)
        print("STOPPED:", e)
        return 2
    for s in SEEDS:
        try:
            RES[s], PLANS[s] = score_seed(a, s, X, grid, EV, BE, a.n_boot, lead)
        except Stop as e:
            stops[s] = str(e)
    if stops:
        jdump({"status": "STOPPED", "stops": stops}, rc_path)
        print("STOPPED:", stops)
        return 2
    try:
        decision = decide({s: RES[s]["seed_row"] for s in SEEDS})
    except Unmapped as e:
        jdump({"status": "UNMAPPED", "reason": str(e), "seed_rows": {s: RES[s]["seed_row"] for s in SEEDS}}, rc_path)
        print("UNMAPPED:", e)
        return 2
    f_exact = abs(RES[0]["B_ade6_mean"] - RES[1]["B_ade6_mean"])
    ctl2 = all(RES[s]["ctl"]["control2_perm_scorer_vs_sel_const"]["pass"] for s in SEEDS)
    READ = {"prereg": PREREG, "decision": decision, "controls_all_pass": bool(ctl2), "F_exact": f_exact, "evidence_class": "MEASURED (ours), artifacts under " + str(run), "tier": TIER,
            "estimator": ESTIMATOR, "strategic_family": "NOT PAUSED for v8F (PI-V8F-2026-10-17; R72): identical across arms by construction (S is B's own)",
            "per_seed": {f"s{s}": {k: RES[s][k] for k in ("PRIMARY_B", "PRIMARY_SEL", "SECONDARY", "seed_row", "selector_accuracy", "B_ade6_mean", "stack_H", "stack_HS", "lead_viol_kscore_minus_B",
                                                          "distance_keeping_kscore_minus_B")} | {"controls": RES[s]["ctl"]} for s in SEEDS}}
    jdump(READ, rc_path)
    jdump({f"s{s}": RES[s] for s in SEEDS}, run / "kscore_numbers.json")
    write_md(run / "KSCORE_RESULT.md", READ, RES)
    print(json.dumps({"outcome": decision["outcome"], "PRIMARY_B": decision["PRIMARY_B_per_seed"], "PRIMARY_SEL": decision["PRIMARY_SEL_per_seed"],
                      "SECONDARY_lon": decision["SECONDARY_lon_per_seed"], "SECONDARY_cl": decision["SECONDARY_consistency_leadviol_per_seed"], "controls_all_pass": READ["controls_all_pass"]}, indent=1))
    return 0


# ================================================================================================ 7. box-2 helpers
BOX2 = {"REPO": "D:\\tanitad\\repo", "PY": "D:\\tanitad\\venv\\Scripts\\python.exe", "RUN": "D:\\tanitad\\runs\\v8f_kscore", "B": "D:\\tanitad\\runs\\v8f_mini_full",
        "CACHE": "D:\\tanitad\\cache_full", "JOIN": "D:\\tanitad\\runs\\v8f_mini_full\\join_train.npz", "TRACKS": "D:\\tanitad\\cache_full\\native_tracks.pt"}


def box2_cmdline(stage_args: list, env=None) -> str:
    """The ONE command string a box-2 job runs for a stage (the chain's --dry-run prints exactly this, the job .cmd embeds exactly this)."""
    e = {**BOX2, **{k: v for k, v in (env or os.environ).items() if k.startswith("KSCORE_B2_") and v}}
    g = lambda k: e.get(f"KSCORE_B2_{k}", BOX2[k])                                    # noqa: E731
    st = stage_args[0]
    common = f'--run-dir "{g("RUN")}" --threads 4'
    data = f'--train-cache "{g("CACHE")}" --eval-cache "{g("CACHE")}" --train-join "{g("JOIN")}" --tracks "{g("TRACKS")}" --b-dir "{g("B")}"'
    if st == "hyps":
        tail = f'{data} --exclude 540:600 --device cuda --batch 16 --shard 640 --lru 2 --feat-dtype float16'
    elif st == "fit":
        tail = f'{data} --exclude 540:600 --device cpu --expect-clips {PINNED["n_clips"]} --expect-windows {PINNED["n_windows"]}'
    elif st == "eval":
        tail = ""
    elif st == "read":
        tail = f'--n-boot {B_BOOT}'
    else:
        raise SystemExit(f"cmdline: unknown stage {st}")
    parts = [f'"{g("PY")}"', f'"{g("REPO")}\\stack\\scripts\\v8f_kscore.py"', st, *stage_args[1:], common, tail]
    return " ".join(p for p in parts if p)


def mkjob_text(name: str, stage_args: list, env=None) -> str:
    """The box-2 job .cmd (CRLF): env.cmd (safe mode: 4 threads, AVX2, TEMP), PYTHONPATH, one %RUNSAFE% python call, the ZZJOB-END marker with the rc. Every backslash is chr(92)."""
    e = {**BOX2, **{k: v for k, v in (env or os.environ).items() if k.startswith("KSCORE_B2_") and v}}
    g = lambda k: e.get(f"KSCORE_B2_{k}", BOX2[k])                                    # noqa: E731
    BS = chr(92)
    lines = ["@echo off", "setlocal enabledelayedexpansion", "call D:" + BS + "tanitad" + BS + "env.cmd", f'set "REPO={g("REPO")}"',
             'set "PYTHONPATH=%REPO%' + BS + 'stack;%REPO%' + BS + 'taniteval;%REPO%' + BS + 'stack' + BS + 'scripts"', 'set "PYTHONIOENCODING=utf-8"',
             f"echo ===== V8FKSCORE {name} {' '.join(stage_args)} %DATE% %TIME%", "%RUNSAFE% " + box2_cmdline(stage_args, env), "set RC=!ERRORLEVEL!",
             "echo ===== ZZJOB-END %DATE% %TIME% rc=!RC!"]
    return "\r\n".join(lines) + "\r\n"


def cmd_assert_gate(a):
    """The chain's assertions on a gate's selfcheck.json + token (run HERE, after the box-2 gate): PASS, the PREREG pins, B's frozen-ness, the scene path alive, the token bound."""
    sc = json.load(open(a.selfcheck, encoding="utf-8"))
    tok = json.load(open(a.token, encoding="utf-8"))
    bad = []
    if sc.get("gate") != "PASS" or sc.get("declared_vs_built") != "PASS":
        bad.append(f"gate {sc.get('gate')} / declared_vs_built {sc.get('declared_vs_built')}")
    lv = sc.get("live", {})
    if not lv.get("pass"):
        bad.append("G-LIVE did not pass")
    if lv.get("B_grad_abs_sum") != 0.0:
        bad.append(f"B's parameters received a gradient: {lv.get('B_grad_abs_sum')}")
    if not (lv.get("B_grad_positive_control_abs_sum") or 0.0) > 0.0:
        bad.append("the B-frozen probe has no positive control (it could not have seen a leak)")
    if lv.get("scene_zero_control_scene_proj_grad") != 0.0 or not (lv.get("scene_real_scene_proj_grad") or 0.0) > 0.0:
        bad.append("the scene path is not alive / the scene-zero control is not exactly 0.0")
    if (sc.get("own_diag_maxabs") if sc.get("own_diag_maxabs") is not None else 1.0) > 0.0:
        bad.append(f"harness identity {sc.get('own_diag_maxabs')}: the chain's gate runs B on the CPU, where it must be EXACTLY 0.0")
    if not sc.get("inputs_identity", {}).get("pass"):
        bad.append("G-INPUTS did not pass")
    i = sc.get("info", {})
    if (i.get("n_clips"), i.get("n_excluded"), i.get("n_windows")) != (PINNED["n_clips"], PINNED["n_excluded"], PINNED["n_windows"]):
        bad.append(f"clips / excluded / windows {(i.get('n_clips'), i.get('n_excluded'), i.get('n_windows'))} != {tuple(PINNED[k] for k in ('n_clips', 'n_excluded', 'n_windows'))}")
    if i.get("ckpt_sha256") != CKPT_SHA[a.seed]:
        bad.append("the hyps were not built from the banked M2b checkpoint")
    if tok.get("gate") != "PASS" or not tok.get("prereg_pinned") or not tok.get("commit"):
        bad.append(f"token: gate {tok.get('gate')} pinned {tok.get('prereg_pinned')} commit {tok.get('commit')}")
    if tok.get("selfcheck_sha256") != sha256_file(a.selfcheck):
        bad.append("the token was written for another selfcheck.json")
    if tok.get("seed") != a.seed or bool(tok.get("perm_labels")) != bool(a.perm):
        bad.append(f"token seed / perm {tok.get('seed')} / {tok.get('perm_labels')}")
    n_params = sc.get("levers", {}).get("n_params")
    print("KSCORE gate:", "PASS" if not bad else bad, "| params", n_params, "| windows", i.get("n_windows"), "| commit", tok.get("commit"))
    return 1 if bad else 0


# ================================================================================================ main
def add_data_args(p):
    p.add_argument("--run-dir", required=True)
    p.add_argument("--b-dir", default=os.environ.get("KSCORE_B_DIR", "D:/tanitad/runs/v8f_mini_full"), help="holds M2b_s<seed>/ckpt_last.pt (the banked checkpoints)")
    p.add_argument("--train-cache", default="D:/tanitad/cache_full")
    p.add_argument("--eval-cache", default="D:/tanitad/cache_full")
    p.add_argument("--train-join", default="D:/tanitad/runs/v8f_mini_full/join_train.npz")
    p.add_argument("--tracks", default="D:/tanitad/cache_full/native_tracks.pt")
    p.add_argument("--kin-mode", default="yaw", choices=("yaw", "steer"))
    p.add_argument("--exclude", default="540:600", help="A:B -- the RA-1 holdout of the TRAIN manifest's sha12 order (the trainer's own flag)")
    p.add_argument("--lru", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    ER = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "raw"
    p.add_argument("--eval-grid", default=str(ER / "eval_grid_windows.json"))
    p.add_argument("--eval-kin", default=str(ER / "eval141_kin.npz"))
    p.add_argument("--eval-join", default=str(ER / "join_eval141.npz"))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("hyps")
    add_data_args(p)
    p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    p.add_argument("--split", required=True, choices=("train", "eval"))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--shard", type=int, default=640)
    p.add_argument("--feat-dtype", default="float16", choices=("float16", "float32"))
    p.add_argument("--allow-any-ckpt", action="store_true", help="TESTS ONLY: skip the banked-checkpoint sha256 assertion")
    p = sub.add_parser("fit")
    add_data_args(p)
    p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    p.add_argument("--out", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--lr", type=float, default=FIT["lr"])
    p.add_argument("--epochs", type=int, default=FIT["epochs"])
    p.add_argument("--batch", type=int, default=FIT["batch"])
    p.add_argument("--wd", type=float, default=FIT["wd"])
    p.add_argument("--hid", type=int, default=FIT["hid"])
    p.add_argument("--scorer-mode", default="learned", choices=("learned", "forced_argmax"))
    p.add_argument("--perm-labels", action="store_true", help="CONTROL 2: train on the permuted k*")
    p.add_argument("--perm-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--gate-only", action="store_true", help="run the launch gate, write the PASS token, exit")
    p.add_argument("--require-token", action="store_true", help="do NOT issue a token: train only against an existing one (the chain runs the gate first)")
    p.add_argument("--token", default=None)
    p.add_argument("--commit", default=None, help="the commit the token is bound to (default: $KSCORE_COMMIT, TIP_V8FKSCORE.txt, git HEAD)")
    p.add_argument("--probe-n", type=int, default=256)
    p.add_argument("--input-tol", type=float, default=1e-3)
    p.add_argument("--expect-clips", type=int, default=0)
    p.add_argument("--expect-windows", type=int, default=0)
    p.add_argument("--allow-nonprereg", action="store_true", help="TESTS ONLY: do not enforce the PREREG pins (the chain asserts the token says prereg_pinned)")
    p.add_argument("--allow-any-ckpt", action="store_true", help="TESTS ONLY")
    p = sub.add_parser("eval")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("read")
    add_data_args(p)
    p.add_argument("--n-boot", type=int, default=B_BOOT)
    p.add_argument("--lead-block", default=None)
    p.add_argument("--dk-reader", default=None)
    p.add_argument("--banked-own", default=None, help="optional: eval_M2b_s{seed}-kind-own.npz template for a per-window check of B's plan")
    p.add_argument("--h-weights", default=None, help="optional: HBLEND weights_M2b_s{seed}.json template (descriptive H stack)")
    p.add_argument("--no-banked", action="store_true", help="smoke only: skip the PREREG's banked ADE6 of B")
    p.add_argument("--smoke", action="store_true", help="smoke only: do not pin the eval grid to 2,399 / 141 and skip the dk plan guards")
    p = sub.add_parser("mkjob")
    p.add_argument("name")
    p.add_argument("stage_args", nargs=argparse.REMAINDER)
    p = sub.add_parser("cmdline")
    p.add_argument("stage_args", nargs=argparse.REMAINDER)
    p = sub.add_parser("assert-gate")
    p.add_argument("--selfcheck", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--perm", action="store_true")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        if a.mode == "hyps":
            return cmd_hyps(a)
        if a.mode == "fit":
            return cmd_fit(a)
        if a.mode == "eval":
            return cmd_eval(a)
        if a.mode == "read":
            return cmd_read(a)
    except Refused as e:
        print(str(e), file=sys.stderr)
        return 3
    if a.mode == "mkjob":
        sys.stdout.buffer.write(mkjob_text(a.name, a.stage_args).encode("utf-8"))          # bytes: a text-mode stdout would turn CRLF into CRCRLF on Windows
        return 0
    if a.mode == "cmdline":
        print(box2_cmdline(a.stage_args))
        return 0
    return cmd_assert_gate(a)


if __name__ == "__main__":
    sys.exit(main())
