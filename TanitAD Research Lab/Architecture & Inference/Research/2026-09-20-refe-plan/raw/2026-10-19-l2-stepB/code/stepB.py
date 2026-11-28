#!/usr/bin/env python3
"""L2 Step B runner: L1's CEM search (`tts_cem.cem_search`, landed, frozen) driven on P2's scorer with the EP head row replaced by an L2 head file (`l2_scorer.py`).

THIS IS AN INFERENCE-TIME SEARCH, NOT A TRAINING RUN (MM ruling E7 / the draft's section on gating): no optimiser, no loss, no gradient.  It is GATED BY `g_reach_stepb.py` (G-REACH-STEPB), whose PASS stamp
binds the exact (head sha256, checkpoint sha256, config sha256, code sha256, L1 code pins) of the run; `search` REFUSES without a matching stamp and re-runs the cheap reachability assertion (d) itself at the
start, so a hand-written stamp cannot skip it.

NOTHING OF L1 IS RE-IMPLEMENTED.  The search is `tts_cem.cem_search`; the scorer is `real_scorer.make_score_fn` with `load_scorer` wrapped to install the head; the endpoint machinery is `tts_cem.delta_ci` /
`inert_verdict` / `seed_spread`.  This file adds only: the frozen C8 config (widths and clips by a stated rule), the token / seed loop, the shards, and the refusals.

C8 (the placeholders the L1 amendment left open): `c8_from_grid` derives them from the MEASURED GRID of Step A (speed 0.85-1.15, offset +-0.3 m) and from nothing else.  Never tuned on an outcome.

    CUDA_VISIBLE_DEVICES=-1 python stepB.py search --arm L2 --head <final_head_s0.npz> --head-sha256 <hex> --bundle <bundle.npz> --ctx <ctx.npy> --ckpt <model_final.pt> --ckpt-sha256 <hex> \
        --refe-dir <refe> --config stepB_config.json --config-sha256 <hex> --gate <G-REACH-STEPB_PASS_*.json> --out-dir <dir> [--start 0 --end 400]
    python stepB.py merge --out-dir <dir> --out search_out.npz
    python stepB.py make-config --out stepB_config.json
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, fields

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ------------------------------------------------------------------------------------------------------------------------------ L1's code, pinned (LF sha256 of the LANDED files, d8b77c1 / tip d2f4a06)
L1_PINS = {"stepA.py": "05d182c7cf1b9e44665b5f9928a35dfa2dd218e89bf7846bdee8a230456313f8",
           "tts_cem.py": "9dea96038d18ae0b7f7a5d522939cb50c5424b372a6de4282ab49c02977bbedb",
           "real_scorer.py": "05c4b4fb771ac924cbbf291b4e019f854411c318dc57e80ae09b02f3312ffbfa",
           "label_perturbations.py": "9e2caca122d1230ff920a3d4722bbb9591d39426134d03af01684fc130c471e5"}


def _l1_code_dir() -> str:
    cands = [os.environ.get("L1_CODE"),
             os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "2026-10-18-ep-rl-lit", "l1_stepA")),     # the repo layout: .../Research/2026-10-18-ep-rl-lit/l1_stepA
             "C:/Users/Admin/par18/l1run/code"]                                                                     # the run copy (byte-identical to the landed files, LF-hash equal)
    for c in cands:
        if c and os.path.exists(os.path.join(c, "tts_cem.py")):
            return c
    raise ImportError("L1's tts_cem.py not found: set L1_CODE (no silent fallback)")


L1_CODE = _l1_code_dir()
if L1_CODE not in sys.path:
    sys.path.insert(0, L1_CODE)


def sha256_lf(path: str) -> str:
    """sha256 of a text file with CRLF folded to LF (a Windows checkout must not make the same source look like different code)"""
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def verify_l1_code(l1_dir: str = None, pins: dict = None) -> dict:
    """the four L1 files Step B stands on are the LANDED ones, byte for byte (LF).  Raises on any mismatch: a Step B run on edited search machinery certifies nothing."""
    d, p = l1_dir or L1_CODE, pins or L1_PINS
    got = {fn: sha256_lf(os.path.join(d, fn)) for fn in p}
    bad = {fn: got[fn][:12] + " != " + p[fn][:12] for fn in p if got[fn] != p[fn]}
    if bad:
        raise RuntimeError(f"L1 code is not the landed code: {bad}")
    return got


import stepA as A  # noqa: E402  (L1's reference implementation, imported unchanged)
import tts_cem as T  # noqa: E402

import l2_scorer  # noqa: E402

# ------------------------------------------------------------------------------------------------------------------------------ C8: the search space and widths, by a stated rule
SEARCH_SEEDS = tuple(T.SEARCH_SEEDS)


@dataclass(frozen=True)
class StepBConfig:
    """Every field is REQUIRED (no defaults): a lever that falls back to a default is the 'built, tested, unreachable from its caller' class (CLAUDE.md)."""
    k_iters: int
    m_pop: int
    n_elite: int
    sigma0_u: float
    sigma0_d: float
    u_lo: float
    u_hi: float
    d_lo: float
    d_hi: float
    widen: float
    profile: str
    seeds: tuple

    @property
    def sigma0(self) -> tuple:
        return (self.sigma0_u, self.sigma0_d)

    def as_json(self) -> dict:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        return d

    @staticmethod
    def from_json(path: str) -> "StepBConfig":
        d = json.load(open(path, encoding="utf-8"))
        d = d.get("config", d)
        known = {f.name for f in fields(StepBConfig)}
        extra = sorted(set(d) - known)
        missing = sorted(known - set(d))
        if extra or missing:
            raise ValueError(f"config {path!r}: unknown fields {extra}, missing fields {missing} (every lever is explicit; an unread key is the 'unreachable' class)")
        d = dict(d)
        d["seeds"] = tuple(int(s) for s in d["seeds"])
        return StepBConfig(**d)


def c8_from_grid(speed_scales=A.SPEED_SCALES, lat_offsets=A.LAT_OFFSETS, k_iters=T.K_ITERS, m_pop=T.M_POP, n_elite=T.N_ELITE, seeds=SEARCH_SEEDS, profile=A.LAT_PROFILE, widen: float = 1.0) -> StepBConfig:
    """THE C8 RULE (stated before any Step B outcome exists; uses ONLY the perturbation grid Step A measured, never an outcome):

      clips   = the closed HULL of the measured grid:  u in [ln min(scales), ln max(scales)] = [ln 0.85, ln 1.15],  d in [min(offsets), max(offsets)] = [-0.3, +0.3] m
                (the search cannot leave the region the admission test covered: tts_cem's own docstring intent; the placeholders, x0.70-1.30 and +-0.6 m, left it by ~2x)
      sigma0  = (hull half-width) / 2 per axis, half-width = max(|lo|, |hi|):  the grid's extreme values sit at 2 sigma0 of the warm start (the pick = the grid's centre)
      K = 5, M = 64, elites M/8, 3 search seeds, profile 'ramp', widen 1.0: exactly L1 prereg section 4 / amendment C3.
    """
    u = [math.log(s) for s in speed_scales]
    d = [float(x) for x in lat_offsets]
    u_lo, u_hi, d_lo, d_hi = min(u), max(u), min(d), max(d)
    return StepBConfig(k_iters=int(k_iters), m_pop=int(m_pop), n_elite=int(n_elite), sigma0_u=max(abs(u_lo), abs(u_hi)) / 2.0, sigma0_d=max(abs(d_lo), abs(d_hi)) / 2.0,
                       u_lo=u_lo, u_hi=u_hi, d_lo=d_lo, d_hi=d_hi, widen=float(widen), profile=str(profile), seeds=tuple(int(s) for s in seeds))


def validate_config(cfg: StepBConfig) -> None:
    """a config is admissible iff it IS the C8 rule's output for the Step A grid (for its own seeds / widen / M), with L1's fixed K = 5 and elites = M/8.  The O4 arm (M = 256, widen = 2) is therefore
    expressible, a hand-edited clip or width is not."""
    if cfg.k_iters != 5 or cfg.n_elite * 8 != cfg.m_pop:
        raise SystemExit(f"config violates L1 prereg section 4 (K = 5, elites = M/8): K={cfg.k_iters}, M={cfg.m_pop}, elites={cfg.n_elite}")
    if cfg != c8_from_grid(k_iters=cfg.k_iters, m_pop=cfg.m_pop, n_elite=cfg.n_elite, seeds=cfg.seeds, widen=cfg.widen):
        raise SystemExit("config is not the C8 rule's output for the Step A grid (and its stated M / widen / seeds): refusing a hand-edited config")


@contextlib.contextmanager
def clips(Tmod, cfg: StepBConfig):
    """install the C8 clips for the duration of one search (tts_cem reads them as module globals at call time); restored afterwards"""
    old = (Tmod.U_CLIP, Tmod.D_CLIP)
    Tmod.U_CLIP = (cfg.u_lo, cfg.u_hi)                                                             # @CLIPS
    Tmod.D_CLIP = (cfg.d_lo, cfg.d_hi)                                                             # @CLIPS
    try:
        yield
    finally:
        Tmod.U_CLIP, Tmod.D_CLIP = old


def search_token(score_fn, fan_raw, pick: int, seed: int, cfg: StepBConfig, Tmod=None) -> dict:
    """ONE token, ONE search seed: the actual caller of `tts_cem.cem_search`.  Every lever is passed EXPLICITLY (no cem_search default is relied on except the ones the config states)."""
    Tm = Tmod or T
    with clips(Tm, cfg):
        return Tm.cem_search(score_fn, fan_raw, int(pick), n_iters=cfg.k_iters, m=cfg.m_pop, n_elite=cfg.n_elite, seed=int(seed),
                             sigma0=cfg.sigma0,                                                    # @SIGMA
                             widen=cfg.widen, profile=cfg.profile)


# ------------------------------------------------------------------------------------------------------------------------------ the DATA CLAUSE (Revision 1: Step B runs on the sealed carve and on nothing else)
SEALED_CARVE_SHA256 = "0629811ff5d05879eb62abc5726005ccc4924f03539423003503a099ba29577c"      # carve B, seed 20261019, 400 tokens (PREREG_L2_FRESH.md section 8.3, sealed)


def tokens_sha256(tokens) -> str:
    """the carve's token-list sha256 = sha256 of the tokens joined by newlines, IN THE BUNDLE'S ORDER.  It is the very definition of `fresh_carve.sha_tokens` (the carve file) and of `make_stepA_bundle.py`'s
    ctx.json `tokens_sha256` (the bundle's tokens are `sorted(sample)`, and the carve is sorted too: MEASURED, the sorted and unsorted joins of the sealed carve agree)."""
    return hashlib.sha256("\n".join(str(t) for t in tokens).encode()).hexdigest()


def check_data_clause(bundle, expect: str | None = None) -> dict:
    """REFUSE (SystemExit) unless the bundle's token list IS the sealed carve.  `expect=None` means the sealed carve B sha256 (the ONLY value any CLI can reach: there is no flag that overrides it; a Python
    caller, i.e. a test, can pass a planted carve's sha256 explicitly, and then says so)."""
    want = SEALED_CARVE_SHA256 if expect is None else str(expect)
    toks = [str(t) for t in np.asarray(bundle["token"])]
    got = tokens_sha256(toks)
    if got != want:
        raise SystemExit(f"DATA CLAUSE REFUSED: the bundle's token-list sha256 {got[:12]} (n={len(toks)}) != the sealed carve's {want[:12]}: Step B runs on the sealed carve and on nothing else")
    return {"carve_sha256": want, "bundle_tokens_sha256": got, "n_tokens": len(toks), "sealed_default": expect is None}


# ------------------------------------------------------------------------------------------------------------------------------ the scorer (ONE builder, used by the launch AND by the gate)
def _real_scorer_module():
    import real_scorer
    return real_scorer


def build_scorer(args, scorer_mod=None, head_mod=None):
    """argv -> score_fn.  The ONLY place a Step B scorer is built: `search` and `g_reach_stepb.py` both call it, so the gate proves the launch's path, not a look-alike."""
    L2S = head_mod or l2_scorer
    SM = scorer_mod or _real_scorer_module()
    head = L2S.read_head_file(args.head, args.head_sha256) if args.arm == "L2" else None          # @WIRE
    b = A.load_npz(args.bundle)
    return L2S.build_score_fn(SM, args.ckpt, args.ctx, np.asarray(b["token"]), head, refe_dir=args.refe_dir, device="cpu", autocast=getattr(args, "autocast", "none"))


# ------------------------------------------------------------------------------------------------------------------------------ the loop
def run_search(score_fn, bundle: dict, token_idx, cfg: StepBConfig, seeds=None, progress=None, Tmod=None) -> dict:
    """tokens x seeds -> arrays.  `score_fn.bind(n)` selects the token's cached visual_ctx.  Returns the shard dict."""
    seeds = tuple(cfg.seeds if seeds is None else seeds)
    idx = [int(i) for i in token_idx]
    n, S = len(idx), len(seeds)
    out = {"token": np.asarray(bundle["token"])[idx], "bundle_index": np.asarray(idx, np.int64), "pick": np.asarray(bundle["pick"])[idx].astype(np.int64), "seeds": np.asarray(seeds, np.int64),
           "exec": np.zeros((n, S, 20, 3), np.float32), "scorer_in": np.zeros((n, S, 20, 3), np.float32), "theta": np.zeros((n, S, 2)), "moved": np.zeros((n, S), bool),
           "hist_best_agg": np.zeros((n, S, cfg.k_iters)), "hist_mu": np.zeros((n, S, cfg.k_iters, 2)), "pred_logit_out": np.zeros((n, S, 6)), "pred_logit_pick": np.zeros((n, 6)),
           "arm_argmax": np.zeros(n, np.int64), "pick_exec": np.zeros((n, 20, 3), np.float32), "t_search_s": np.zeros((n, S)), "t_token_s": np.zeros(n)}
    for i, bi in enumerate(idx):
        t0 = time.time()
        score_fn.bind(bi)
        fan = np.asarray(bundle["traj_raw"][bi], np.float32)
        pick = int(bundle["pick"][bi])
        base = np.asarray(score_fn(np.array(fan, copy=True)), np.float64)
        out["pred_logit_pick"][i] = base[pick]
        out["arm_argmax"][i] = int(np.argmax(A.aggregate(base)))
        out["pick_exec"][i] = A.repair_last_heading_np(fan[pick])
        for s, seed in enumerate(seeds):
            t1 = time.time()
            r = search_token(score_fn, fan, pick, seed, cfg, Tmod)
            out["exec"][i, s], out["scorer_in"][i, s] = r["exec"], r["raw"]
            out["theta"][i, s] = r["theta"]
            out["moved"][i, s] = bool(r["moved"])
            for k, h in enumerate(r["history"]):
                out["hist_best_agg"][i, s, k], out["hist_mu"][i, s, k] = h["best_agg"], h["mu"]
            out["pred_logit_out"][i, s] = A.score_replaced(score_fn, fan, pick, r["raw"])[pick] if r["moved"] else base[pick]
            out["t_search_s"][i, s] = time.time() - t1
        out["t_token_s"][i] = time.time() - t0
        if progress:
            progress(i + 1, n, out["t_token_s"][i])
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the gate stamp (written by g_reach_stepb.py, checked here)
CODE_FILES = ("stepB.py", "l2_scorer.py", "g_reach_stepb.py")


def code_shas() -> dict:
    return {fn: sha256_lf(os.path.join(HERE, fn)) for fn in CODE_FILES if os.path.exists(os.path.join(HERE, fn))}


def check_stamp(stamp_path: str, *, arm: str, head_sha256: str | None, ckpt_sha256: str, config_sha256: str, bundle_sha256: str | None = None) -> dict:
    """REFUSE (RuntimeError) unless the G-REACH-STEPB stamp is a PASS earned on exactly this head / checkpoint / config / code / L1 code.  A stamp for another head certifies nothing about this one."""
    if not stamp_path or not os.path.isfile(stamp_path):
        raise RuntimeError(f"no G-REACH-STEPB stamp at {stamp_path!r}: run g_reach_stepb.py first (no search without a PASS)")
    st = json.load(open(stamp_path, encoding="utf-8"))
    bad = []
    if st.get("schema") != "tanitad.g_reach_stepb/1":
        bad.append("schema")
    if st.get("verdict") != "PASS":
        bad.append(f"verdict {st.get('verdict')!r}")
    if arm == "L2" and st.get("head_sha256") != head_sha256:
        bad.append("head_sha256")
    if st.get("ckpt_sha256") != ckpt_sha256:
        bad.append("ckpt_sha256")
    if st.get("config_sha256") != config_sha256:
        bad.append("config_sha256")
    if bundle_sha256 is not None and st.get("bundle_sha256") != bundle_sha256:
        bad.append("bundle_sha256")
    now = code_shas()
    for fn, sh in now.items():
        if (st.get("code_sha256") or {}).get(fn) != sh:
            bad.append(f"code:{fn}")
    for fn, sh in (st.get("l1_pins") or {}).items():
        if L1_PINS.get(fn) != sh:
            bad.append(f"l1:{fn}")
    if bad:
        raise RuntimeError(f"G-REACH-STEPB stamp {stamp_path!r} does not bind this launch: {bad}")
    return st


def preflight_reach(score_fn_arm, score_fn_p2, bundle: dict, n0: int, expect: str) -> dict:
    """assertion (d), run INSIDE the launch: on one real set, the arm's logits vs plain P2's.  expect='differs' -> EP column differs somewhere (> 0.0) and every other column is EXACTLY 0.0; expect='identical' -> all exactly 0.0."""
    fan = np.asarray(bundle["traj_raw"][n0], np.float32)
    score_fn_arm.bind(n0)
    score_fn_p2.bind(n0)
    la = np.asarray(score_fn_arm(np.array(fan, copy=True)), np.float64)
    lp = np.asarray(score_fn_p2(np.array(fan, copy=True)), np.float64)
    d = np.abs(la - lp)
    others = [c for c in range(6) if c != l2_scorer.EP]
    rec = {"token_index": int(n0), "max_abs_ep_logit_diff": float(d[:, l2_scorer.EP].max()), "max_abs_other_logit_diff": float(d[:, others].max()), "expect": expect}
    rec["ok"] = bool(rec["max_abs_other_logit_diff"] == 0.0 and ((rec["max_abs_ep_logit_diff"] > 0.0) if expect == "differs" else (rec["max_abs_ep_logit_diff"] == 0.0)))
    return rec


# ------------------------------------------------------------------------------------------------------------------------------ CLI
def build_search_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="stepB.py search")
    ap.add_argument("--arm", required=True, choices=("L2", "P2HEAD"), help="L2: P2 + the L2 EP head file.  P2HEAD: plain P2 (the descriptive control arm; no head args allowed)")
    ap.add_argument("--head", default=None)
    ap.add_argument("--head-sha256", default=None)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-sha256", required=True)
    ap.add_argument("--refe-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--config-sha256", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--shard", type=int, default=20)
    ap.add_argument("--autocast", default="none", choices=("none", "bf16"))
    ap.add_argument("--threads", type=int, default=4)
    return ap


def validate_search_args(a) -> None:
    if a.arm == "L2" and not (a.head and a.head_sha256):
        raise SystemExit("--arm L2 needs --head and --head-sha256 (a head is never loaded without its expected sha256)")
    if a.arm == "P2HEAD" and (a.head or a.head_sha256):
        raise SystemExit("--arm P2HEAD is plain P2: --head / --head-sha256 are refused (a flag that is silently ignored is the 'unreachable' class)")


def run_search_cli(a, enforce_stamp: bool = True, scorer_mod=None, carve_sha256: str | None = None) -> int:
    """the whole launch, from a parsed namespace.  `enforce_stamp=False` exists ONLY for g_reach_stepb.py's end-to-end check (f), which has to run this very function BEFORE a stamp can exist; every other
    refusal (L1 pins, config rule, checkpoint sha, head sha256, the DATA CLAUSE, the in-launch reachability assertion) stays on.  `carve_sha256=None` is the sealed carve B; no CLI flag reaches the parameter."""
    validate_search_args(a)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU only: Step B never uses the GPU (set CUDA_VISIBLE_DEVICES=-1)"
    import torch
    torch.set_num_threads(int(a.threads))
    l1 = verify_l1_code()
    cfg_sha = sha256_lf(a.config)
    if cfg_sha != a.config_sha256:
        raise SystemExit(f"config sha256 {cfg_sha[:12]} != --config-sha256 {a.config_sha256[:12]}")
    cfg = StepBConfig.from_json(a.config)
    validate_config(cfg)
    ck_sha = l2_scorer.sha256_file(a.ckpt)
    if ck_sha != a.ckpt_sha256:
        raise SystemExit(f"checkpoint sha256 {ck_sha[:12]} != --ckpt-sha256 {a.ckpt_sha256[:12]}")
    b = A.load_npz(a.bundle)
    clause = check_data_clause(b, carve_sha256)                                                         # @DATACLAUSE
    bundle_sha = l2_scorer.sha256_file(a.bundle)
    if enforce_stamp:
        check_stamp(a.gate, arm=a.arm, head_sha256=a.head_sha256, ckpt_sha256=ck_sha, config_sha256=cfg_sha, bundle_sha256=bundle_sha)
    n_all = len(b["token"])
    end = n_all if a.end is None else min(a.end, n_all)
    os.makedirs(a.out_dir, exist_ok=True)
    fn = build_scorer(a, scorer_mod=scorer_mod)                                                         # @CLIFN
    if a.arm == "L2":                                              # (d) INSIDE the launch: the head must change the EP column and nothing else
        p2 = build_scorer(argparse.Namespace(**{**vars(a), "arm": "P2HEAD", "head": None, "head_sha256": None}), scorer_mod=scorer_mod)
        pre = preflight_reach(fn, p2, b, a.start, "differs")
        if not pre["ok"]:
            raise SystemExit(f"PREFLIGHT REACH FAILED (the L2 head did not reach the logits the search sees): {pre}")
        del p2
    else:
        pre = None
    t0 = time.time()
    for lo in range(a.start, end, a.shard):
        hi = min(lo + a.shard, end)
        p = os.path.join(a.out_dir, f"shard_{a.arm}_{lo:04d}.npz")
        if os.path.exists(p):
            print("ZZSKIP", lo, flush=True)
            continue
        out = run_search(fn, b, range(lo, hi), cfg)
        meta = {"arm": a.arm, "head_sha256": a.head_sha256, "ckpt_sha256": ck_sha, "config_sha256": cfg_sha, "bundle_sha256": bundle_sha, "gate_stamp": os.path.abspath(a.gate) if enforce_stamp else None,
                "gate_stamp_sha256": l2_scorer.sha256_file(a.gate) if enforce_stamp else None, "l1_code_sha256": l1, "code_sha256": code_shas(), "preflight": pre, "data_clause": clause,
                "scorer_info": {k: v for k, v in fn.info.items() if k != "ckpt"}, "lo": lo, "hi": hi, "seconds": time.time() - t0}
        tmp = p + ".tmp.npz"
        np.savez(tmp, meta=np.array(json.dumps(meta)), **out)
        os.replace(tmp, p)
        print(f"ZZSHARD {a.arm} {lo}-{hi} {time.time() - t0:.0f}s tokens={hi - lo} t/token={out['t_token_s'].mean():.2f}s", flush=True)
    print("ZZSEARCH_DONE", flush=True)
    return 0


def cmd_search(argv) -> int:
    return run_search_cli(build_search_parser().parse_args(argv), enforce_stamp=True)


def cmd_merge(argv) -> int:
    ap = argparse.ArgumentParser(prog="stepB.py merge")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", required=True, choices=("L2", "P2HEAD"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None, help="expected number of tokens (a merge of fewer is refused)")
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.out_dir, f"shard_{a.arm}_*.npz")))
    if not files:
        raise SystemExit("no shards")
    Z = [np.load(f, allow_pickle=False) for f in files]
    metas = [json.loads(str(z["meta"])) for z in Z]
    for k in ("head_sha256", "ckpt_sha256", "config_sha256", "bundle_sha256", "arm", "data_clause"):
        if len({json.dumps(m.get(k), sort_keys=True) for m in metas}) != 1:
            raise SystemExit(f"shards disagree on {k}: refusing to merge runs that are not one run")
    keys = [k for k in Z[0].files if k != "meta"]
    cat = {}
    for k in keys:
        if k == "seeds":
            cat[k] = Z[0][k]
        else:
            cat[k] = np.concatenate([z[k] for z in Z], 0)
    n = len(cat["token"])
    if len(set(cat["bundle_index"].tolist())) != n:
        raise SystemExit("duplicate tokens across shards")
    if a.n is not None and n != a.n:
        raise SystemExit(f"merged {n} tokens, expected {a.n}")
    cat["meta"] = np.array(json.dumps({**metas[0], "merged_from": [os.path.basename(f) for f in files], "n_tokens": n}))
    np.savez(a.out, **cat)
    print("ZZMERGE_OK", n, flush=True)
    return 0


def cmd_make_config(argv) -> int:
    ap = argparse.ArgumentParser(prog="stepB.py make-config")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = c8_from_grid()
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg.as_json(), f, indent=1)
        f.write("\n")
    print("ZZCONFIG", sha256_lf(a.out), json.dumps(cfg.as_json()))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"search": cmd_search, "merge": cmd_merge, "make-config": cmd_make_config}
    if not argv or argv[0] not in cmds:
        print(__doc__)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
