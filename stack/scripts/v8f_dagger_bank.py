#!/usr/bin/env python3
"""PREREG_V8F_DAGGER section 1: the own-state bank generator. FROZEN M2b, ONE pass, ticks 1 and 2 of the decisive T1-kin configuration, TRAIN clips only, clip-sharded.

    gen    one shard (or the whole bank with --n-shards 1): rolls every TRAIN window of the shard's clips for two forwards (ticks 0 and 1; tick 2's state is reached by committing tick 1's plan, so no third
           forward), stores the own state at ticks 1 and 2 with its matched route point (``tanitad.data.v8f_dagger.match_route``), writes ``shard_KKKofNNN.npz`` + ``shard_KKKofNNN.provenance.json``;
           with --n-shards 1 it then merges.
    merge  verifies that the shards are all there, disjoint, from one generator configuration and TRAIN-only, and writes ``bank.json`` (the manifest; ITS sha256 is the bank's sha256, the arm's
           ``--dagger-bank-sha256``) and ``PROVENANCE.json``.

REUSE, NOT A FORK. The rollout is ``v8f_t1_eval.rollout`` itself (its ``hook`` seam, the decisive flags --replan-steps 10 --limit causal --nav deadreckon --scene logged --schedule periodic, executor
``plan``, no post-map = PLAIN); the states are ``v8f_t1_eval.WindowState`` objects; the items are ``v8f_t1_eval.build_item``. This script adds a hook that snapshots the state at ticks 1 and 2, and a planner
wrapper that runs the model for ticks 0 and 1 only (the third plan would be unused). Nothing in ``v8f_t1_eval.py`` is edited.

CONTROLS THAT RUN HERE (section 6). K-B0 on EVERY window: the tick-0 own state is the logged item, bit for bit. K-B1: a seeded fraction of the CHUNKS is re-run and must reproduce its stored rows bit for bit
(a chunk, not a single rollout: a batched forward is not invariant to the batch's composition, so a lone window re-run is a different computation). K-B3: the TRAIN clips are disjoint from eval-141 (refused
before any rollout). COVERAGE is computed and reported (the eligible fraction of TRAIN windows; below 0.5 the outcome is BLOCKED). Every cut is counted by reason, per tick.
K-B4 (the matcher's six synthetic cases) is RUN at generation time, in every shard and again at the merge, by ``tanitad.data.v8f_dagger.kb4_cases`` -- the function the unit test calls. K-B5 (re-anchoring) is
MEASURED on the records the shard is made of (``Kb5Accumulator``: the re-anchored own NOW against the logged pose at r', and the held-frame relative transforms before/after, both gated at 1e-4 m). Both are written to
each shard's provenance (``controls``) and, merged, to ``PROVENANCE.json`` ``controls`` (the records ``dag_read.py`` refuses without); PASS is DERIVED from the numbers. A failed control writes PASS false and exits
non-zero (6): at the merge it writes ``PROVENANCE.json`` and NO ``bank.json``, so a bank with a failed control never looks complete.

EVERY PATH IS AN ARGUMENT (it must run on box 2): --ckpt, --m2b-argv, --cache-dir, --tracks, --join-train, --eval-grid, --out-dir. --dry-run applies every guard and writes nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data import v8f_dagger as G  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_t1_eval as T1  # noqa: E402

H = D.HORIZON
ROLL = {k: v for k, v in G.ROLLOUT.items() if k != "post_map"}        # the decisive T1-kin flags, as v8f_t1_eval.rollout takes them
TOOL_FILES = ("stack/tanitad/data/v8f_dagger.py", "stack/scripts/v8f_dagger_bank.py", "stack/scripts/v8f_t1_eval.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py")


class _Done(Exception):
    """Raised by the hook once the last bank tick has been captured: the rollout is not carried to its end."""


# ------------------------------------------------------------------------------------------------ the snapshot of one own state
def snapshot(st, tick: int, hmax: int, pose_grid) -> dict:
    """The bank record of ``WindowState`` ``st`` at its current clock row j: the own history rows j - hmax + 1 ... j (NaN before the clip), the own kin3 at j, and the match against the LOGGED grid."""
    j = int(st.j)
    idx = np.arange(j - hmax + 1, j + 1)
    real = idx >= 0
    idc = np.clip(idx, 0, st.T - 1)
    p10f = np.array(st.d["p10f"][idc], np.float32)
    p10f[~real] = np.nan
    pose = np.array(st.d["pose"][idc][:, :3], np.float64)
    pose[~real] = np.nan
    seg = np.hypot(*np.diff(pose[:, :2], axis=0).T)                                  # NaN across a non-existent row
    arc = np.concatenate([[0.0], np.cumsum(np.where(real[1:] & real[:-1], seg, 0.0))])
    arc[~real] = np.nan
    m = G.match_route(pose_grid, st.t0, pose[-1], j)
    log_now = np.array(np.asarray(pose_grid)[m.r][:3], np.float64) if m.r >= 0 else np.full(3, np.nan)       # the LOGGED pose at r' (K-B5 compares the re-anchored own NOW with it); NOT a bank field (RECORD_FIELDS), so the bank's bytes do not move
    return {"ci": int(st.ci), "t0": int(st.t0), "tick": int(tick), "j": j, "p10f": p10f, "pose": pose, "arc": arc, "kin3": np.array(st.d["kin3"][j], np.float64),
            "r": m.r, "e_lat": m.e_lat, "dpsi": m.dpsi_deg, "reason": m.reason, "log_now": log_now}


class TwoTickPlanner:
    """``v8f_t1_eval.rollout``'s planner seam: the model's plan for the first ``n_forward`` calls of a chunk, and NO forward afterwards (the state at tick 2 is all the bank needs of tick 2)."""

    def __init__(self, inner, n_forward: int = 2):
        self.inner, self.n_forward, self.calls = inner, n_forward, 0
        self.keys_read = set()

    def __call__(self, batch):
        self.calls += 1
        if self.calls > self.n_forward:
            return np.zeros((int(batch["p10f"].shape[0]), H, 2), np.float32)
        plan = self.inner(batch)
        self.keys_read = self.inner.keys_read
        return plan


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))


class Capture:
    """The ``hook`` of ``v8f_t1_eval.rollout``: runs BEFORE the tick's plan is committed, so ``st.j`` and ``st.d`` are the tick's executed state."""

    def __init__(self, *, store, J, ds_log, ds_own, proxy, hmax, tick0_check=True):
        self.store, self.J, self.ds_log, self.ds_own, self.proxy, self.hmax, self.tick0_check = store, J, ds_log, ds_own, proxy, hmax, tick0_check
        self.records, self.tick0 = [], {"checked": 0, "mismatch": 0}

    def __call__(self, n, batch, plan, states):
        if n == 0 and self.tick0_check:
            for st in states:
                self._tick0(st)
        if n in G.TICKS:
            for w, st in enumerate(states):
                rec = snapshot(st, n, self.hmax, self.store.get(str(self.J["clip"][st.ci]))["pose"])
                rec["_win"] = w
                self.records.append(rec)
        if n == max(G.TICKS):
            raise _Done

    def _tick0(self, st):
        """K-B0: the own state at tick 0 reproduces the logged item's ego tensors bit for bit (REBUILT + pose_now)."""
        own = T1.build_item(self.ds_own, self.proxy, st, 0, 0, 0, "logged", "logged")
        log = self.ds_log.item(st.ci, st.t0, 0, 0)
        bad = [k for k in G.REBUILT + ("pose_now",) if not _same(own[k], log[k])]
        self.tick0["checked"] += 1
        if bad:
            self.tick0["mismatch"] += 1
            raise G.BankRefused(f"K-B0: the tick-0 own state of window (clip {self.J['clip'][st.ci]}, t0 {st.t0}) differs from the logged item in {bad}")


def make_states(store, tracks, J, chunk, nav_mode="deadreckon", state_cls=None):
    if state_cls is None:
        return T1.make_states(store, tracks, J, chunk, nav_mode)
    out = []
    for ci, t0, _, _ in chunk:
        clip = str(J["clip"][ci])
        o, n = int(J["off"][ci]), int(J["n"][ci])
        out.append(state_cls(store.get(clip), tracks[clip], J["nav"][o:o + n], ci, t0, nav_mode))
    return out


def rollout_chunk(chunk, *, planner, store, tracks, J, ds_log, ds_own, proxy, hmax, device="cpu", tick0_check=True, state_cls=None):
    """(records, K-B0 totals) of one chunk of windows: ``v8f_t1_eval.rollout`` (the decisive configuration) with the capture hook; records ordered (window, tick)."""
    states = make_states(store, tracks, J, chunk, ROLL["nav"], state_cls)
    cap = Capture(store=store, J=J, ds_log=ds_log, ds_own=ds_own, proxy=proxy, hmax=hmax, tick0_check=tick0_check)
    try:
        T1.rollout(TwoTickPlanner(planner), states, ds_own, proxy, m=ROLL["replan_steps"], schedule=ROLL["schedule"], scene=ROLL["scene"], limit=ROLL["limit"], device=device,
                   executor=ROLL["executor"], hook=cap)
    except _Done:
        pass
    else:
        raise RuntimeError("the rollout ended without reaching the last bank tick")
    recs = sorted(cap.records, key=lambda r: (r["_win"], r["tick"]))
    return recs, cap.tick0



def records_to_arrays(recs: list) -> dict:
    return {k: np.stack([np.asarray(r[k]) for r in recs]).astype(dt) for k, dt in G.RECORD_FIELDS.items()}


def arrays_equal(a: dict, b: dict) -> bool:
    return all(_same(a[k], b[k]) for k in G.RECORD_FIELDS)


def run_windows(wins, *, planner_factory, store, tracks, J, ds_log, ds_own, proxy, hmax, batch=32, device="cpu", replay_chunks=(), tick0_check=True, state_cls=None, log=None, kb5=None):
    """All windows -> (the record arrays, the K-B0 totals, the K-B1 totals). ``replay_chunks`` = chunk numbers re-run (K-B1). ``kb5`` = a ``G.Kb5Accumulator``, fed the records of every chunk (K-B5 is measured ON
    the records the bank is made of; the replayed chunk is not fed twice)."""
    parts, replay, k0 = [], {"chunks": 0, "windows": 0, "mismatch_chunks": 0}, {"checked": 0, "mismatch": 0}
    t_start, n_chunks = time.time(), math.ceil(len(wins) / batch)
    kw = dict(store=store, tracks=tracks, J=J, ds_log=ds_log, ds_own=ds_own, proxy=proxy, hmax=hmax, device=device, tick0_check=tick0_check, state_cls=state_cls)
    for c in range(n_chunks):
        chunk = wins[c * batch:(c + 1) * batch]
        recs, t0c = rollout_chunk(chunk, planner=planner_factory(), **kw)
        k0["checked"] += t0c["checked"]
        k0["mismatch"] += t0c["mismatch"]
        if kb5 is not None:
            kb5.update(recs)                                      # K-B5, on this chunk's own records
        arr = records_to_arrays(recs)
        if c in replay_chunks:                                    # K-B1: the SAME chunk (same batch composition) re-run from scratch must reproduce its rows bit for bit
            again, _ = rollout_chunk(chunk, planner=planner_factory(), **kw)
            replay["chunks"] += 1
            replay["windows"] += len(chunk)
            replay["mismatch_chunks"] += int(not arrays_equal(arr, records_to_arrays(again)))
        parts.append(arr)
        if log and ((c + 1) % 25 == 0 or c + 1 == n_chunks):
            log(f"[{min((c + 1) * batch, len(wins))}/{len(wins)} windows] {time.time() - t_start:.0f}s")
    if not parts:
        shp = {"p10f": (hmax, 8), "pose": (hmax, 3), "arc": (hmax,), "kin3": (3,)}
        return {k: np.zeros((0,) + shp.get(k, ()), dt) for k, dt in G.RECORD_FIELDS.items()}, k0, replay
    return {k: np.concatenate([p[k] for p in parts]) for k in G.RECORD_FIELDS}, k0, replay


# ------------------------------------------------------------------------------------------------ inputs
def load_ckpt(path, device):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    if cfg.kin_mode != "yaw":
        raise SystemExit("REFUSED: the T1-kin rollout rebuilds the ego state from the executed pose track; a --kin-mode steer checkpoint has no own-motion equivalent.")
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ck


def check_readable_rows(cfg):
    """The rows a swapped sample can READ (T's and S's last hist frames at their ages) must be real kinematic rows (cache step >= 1) for the earliest bank state j = window_start + 10."""
    j_min = D.effective_window_start(cfg.hist, cfg.window_start) + G.REPLAN_STEPS * min(G.TICKS)
    first = j_min - cfg.max_age_s - (cfg.hist - 1)
    if first < 1:
        raise SystemExit(f"REFUSED: at j = {j_min} a held S frame ({cfg.max_age_s} steps old, {cfg.hist} frames) reaches row {first} < 1: a bank row a reader could index would not exist")
    return {"j_min": j_min, "first_row_read": first}


def train_list(a):
    """The arm's TRAIN list exactly as the trainer builds it from M2b's argv (``build_data``): (args namespace, store, J, wins, clips, info)."""
    import v8f_mini_train as TR
    argv = json.load(open(a.m2b_argv, encoding="utf-8"))["argv"]
    ns = TR.build_parser().parse_args(argv)
    if ns.synthetic:
        raise SystemExit("REFUSED: the M2b argv is a --synthetic run; the bank is built from the real TRAIN list")
    for k in ("cache_dir", "tracks", "join_train"):
        if getattr(a, k):
            setattr(ns, k, getattr(a, k))
    return ns, TR


def input_digest(path) -> dict:
    if not path or not os.path.exists(path):
        return {"path": str(path), "sha256": None, "bytes": None}
    return {"path": str(path), "sha256": G.sha256_file(path), "bytes": os.path.getsize(path)}


# ------------------------------------------------------------------------------------------------ gen
def cmd_gen(a, log=print):
    import torch
    t_all = time.time()
    device = torch.device(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if a.threads:
        torch.set_num_threads(a.threads)
    if not (0 <= a.shard < a.n_shards):
        raise SystemExit(f"REFUSED: --shard {a.shard} is not in [0, {a.n_shards})")
    model, cfg, ck = load_ckpt(a.ckpt, device)
    rows = check_readable_rows(cfg)
    ns, TR = train_list(a)
    store, J, wins, clips, info = TR.build_data(ns, cfg)
    grid = json.load(open(a.eval_grid, encoding="utf-8"))
    eval_clips = list(grid["order"])
    try:                                                                                # K-B3, before any rollout: TRAIN is disjoint from eval-141 (by clip id and by stable sid)
        G.check_train_only(bank_clips=clips, train_clips=clips, eval_clips=eval_clips, complete=False)
    except G.BankRefused as e:
        raise SystemExit(f"REFUSED: {e}")
    if a.expect_clips and len(clips) != a.expect_clips:
        raise SystemExit(f"REFUSED: the TRAIN list has {len(clips)} clips, --expect-clips {a.expect_clips}")
    if a.expect_windows and len(wins) != a.expect_windows:
        raise SystemExit(f"REFUSED: the TRAIN list has {len(wins)} windows, --expect-windows {a.expect_windows}")
    wsha = G.windows_sha256(J["clip"], wins)
    sel_clips = clips[:a.limit_clips] if a.limit_clips else list(clips)
    positions = [i for i in range(len(sel_clips)) if i % a.n_shards == a.shard]
    ci_of = {c: i for i, c in enumerate(J["clip"].tolist())}
    shard_ci = {ci_of[sel_clips[i]] for i in positions}
    wins_shard = [w for w in wins if w[0] in shard_ci]
    n_chunks = math.ceil(len(wins_shard) / a.batch)
    rng = np.random.default_rng(a.replay_seed)
    replay_chunks = set(rng.choice(n_chunks, min(n_chunks, max(1, math.ceil(a.replay_frac * n_chunks))), replace=False).tolist()) if (a.replay_frac > 0 and n_chunks) else set()
    plan = {"mode": "dry-run" if a.dry_run else "gen", "shard": [a.shard, a.n_shards], "n_train_clips": len(clips), "n_train_windows": len(wins), "window_list_sha256": wsha,
            "train_sha256": info["train_sha256"], "limit_clips": a.limit_clips, "partial": bool(a.limit_clips), "shard_clips": len(positions), "shard_windows": len(wins_shard),
            "chunks": n_chunks, "batch": a.batch, "est_records": 2 * len(wins_shard), "est_shard_bytes": 2 * len(wins_shard) * (cfg.hmax * (8 * 4 + 3 * 8 + 8) + 3 * 8 + 48),
            "replay_chunks": len(replay_chunks), "eval_clips": len(eval_clips), "eval_overlap": 0, "ckpt_sha256": G.sha256_file(a.ckpt) if a.dry_run else None, "hmax": cfg.hmax,
            "readable_rows": rows, "device": str(device)}
    if a.dry_run:
        log(json.dumps(plan, indent=1))
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    ds_proxy = T1._OneClipStore()
    kw = dict(hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    ds_own, ds_log = D.V8FMiniWindows(ds_proxy, J, [], window_start=cfg.window_start, **kw), D.V8FMiniWindows(store, J, [], window_start=cfg.window_start, **kw)     # spelled out: test_v8f_window_start's static guard rejects window_start hidden in **kw
    tracks = D.load_native_tracks(ns.tracks)
    ckpt_sha = G.sha256_file(a.ckpt)
    t_roll = time.time()
    kb5 = G.Kb5Accumulator()                                                            # K-B5, measured on the records this run makes
    arrays, k0, replay = run_windows(wins_shard, planner_factory=lambda: T1.model_planner(model, device), store=store, tracks=tracks, J=J, ds_log=ds_log, ds_own=ds_own, proxy=ds_proxy,
                                     hmax=cfg.hmax, batch=a.batch, device=device, replay_chunks=replay_chunks, log=log, kb5=kb5)
    t_roll = time.time() - t_roll
    controls = {"K-B4": G.kb4_record(), "K-B5": kb5.record()}                           # PREREG section 6: the matcher's synthetic cases and the re-anchoring, RUN HERE and written to this shard's provenance
    names = {int(c): str(J["clip"][c]) for c in sorted(shard_ci)}
    inputs = {"ckpt": input_digest(a.ckpt), "m2b_argv": input_digest(a.m2b_argv), "tracks": input_digest(ns.tracks), "join_train": input_digest(ns.join_train), "eval_grid": input_digest(a.eval_grid)}
    thr = a.thresholds_file or os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini", "raw", "2026-10-26-dagger-draft", "probe", "THRESHOLDS_FIXED.txt")
    header = {"schema": G.SCHEMA, "ckpt_sha256": ckpt_sha, "cfg": {k: getattr(cfg, k) for k in ("hist", "hmax", "max_age_t", "max_age_s", "window_start", "kin_mode")},
              "rollout": G.ROLLOUT, "constants": G.constants(), "constants_sha256": G.constants_sha256(), "thresholds_file_sha256": G.sha256_file(thr) if os.path.exists(thr) else None,
              "train": {"n_clips": len(clips), "n_windows": len(wins), "train_sha256": info["train_sha256"], "exclude": ns.exclude, "n_excluded": info["n_excluded"]},
              "window_list_sha256": wsha, "eval": {"n_clips": len(eval_clips), "clips": sorted(eval_clips), "grid_sha256": inputs["eval_grid"]["sha256"], "overlap": 0},
              "inputs_sha256": {k: v["sha256"] for k, v in inputs.items() if k != "ckpt"}, "limit_clips": a.limit_clips, "partial": bool(a.limit_clips),
              "shard": {"k": a.shard, "n": a.n_shards, "positions": positions, "n_windows": len(wins_shard)}, "clips": {str(k): v for k, v in names.items()}}
    bank = G.DaggerBank(arrays, header)
    cov = bank.coverage(wins_shard)
    by_tick = {str(t): {G.REASONS[i]: int(((arrays["tick"] == t) & (arrays["reason"] == i)).sum()) for i in range(len(G.REASONS))} for t in G.TICKS}
    warnings = []
    if by_tick and any(v["nonfinite"] for v in by_tick.values()):
        warnings.append(f"NONFINITE own states: {sum(v['nonfinite'] for v in by_tick.values())} (a non-finite own pose or a non-finite logged row; counted, never matched)")
    if replay["mismatch_chunks"]:
        warnings.append(f"K-B1 FAILED: {replay['mismatch_chunks']} of {replay['chunks']} replayed chunks differ from their stored rows")
    stem = os.path.join(a.out_dir, f"shard_{a.shard:03d}of{a.n_shards:03d}")
    G.save_shard(stem + ".npz", arrays, header)
    prov = {"schema": G.SCHEMA, "kind": "shard", "created_unix": time.time(), "argv": sys.argv, "plan": plan, "inputs": inputs, "tool_sha256": {f: G.sha256_file(os.path.join(REPO, f)) for f in TOOL_FILES if os.path.exists(os.path.join(REPO, f))},
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}, "device": str(device), "threads": torch.get_num_threads(),
            "n_records": int(len(arrays["ci"])), "reasons_by_tick": by_tick, "coverage": cov, "kb0_tick0_identity": k0, "kb1_replay": {**replay, "frac": a.replay_frac, "seed": a.replay_seed},
            "kb3_eval_disjoint": {"n_eval_clips": len(eval_clips), "overlap": 0}, "controls": controls, "causal_bucket_stats": T1.CAUSAL_STATS, "file_sha256": G.sha256_file(stem + ".npz"),
            "content_sha256": G.content_sha256(arrays), "seconds": {"rollout": t_roll, "total": time.time() - t_all}, "warnings": warnings}
    json.dump(prov, open(stem + ".provenance.json", "w", encoding="utf-8"), indent=1, default=G._jsonable)
    log(f"wrote {stem}.npz  records {prov['n_records']}  coverage {cov['fraction']:.4f}  K-B0 {k0}  K-B1 {replay}")
    for w in warnings:
        log("WARNING " + w)
    if replay["mismatch_chunks"]:
        return 5
    failed = [k for k, v in controls.items() if v.get("PASS") is not True]
    for k in failed:
        log(f"CONTROL {k} FAILED: {json.dumps(controls[k], default=G._jsonable)}")
    if a.n_shards == 1:                                                                 # the merge reads the shard's controls back: with a failed control it writes PROVENANCE.json (PASS false) and NO bank.json
        return cmd_merge(argparse.Namespace(out_dir=a.out_dir), log=log) or (6 if failed else 0)
    return 6 if failed else 0                                                           # the shard IS written (the evidence), with PASS false; the merge of the shards will refuse to write a bank


# ------------------------------------------------------------------------------------------------ merge
def cmd_merge(a, log=print):
    files = sorted(glob.glob(os.path.join(a.out_dir, "shard_*of*.npz")))
    if not files:
        raise SystemExit(f"REFUSED: no shard in {a.out_dir}")
    loaded = [(f, *G.load_shard(f)) for f in files]
    n = loaded[0][2]["shard"]["n"]
    ks = sorted(h["shard"]["k"] for _, _, h in loaded)
    if ks != list(range(n)) or any(h["shard"]["n"] != n for _, _, h in loaded):
        raise SystemExit(f"REFUSED: the shards present are {ks}, a complete bank of {n} shards needs 0..{n - 1}")
    strip = lambda h: {k: v for k, v in h.items() if k not in ("shard", "clips", "n_records")}
    common = strip(loaded[0][2])
    for f, _, h in loaded[1:]:
        if G.canon(strip(h)) != G.canon(common):
            raise SystemExit(f"REFUSED: {os.path.basename(f)} was generated under a different configuration than {os.path.basename(loaded[0][0])} (checkpoint, rollout, constants, TRAIN list or inputs differ)")
    pos = [p for _, _, h in loaded for p in h["shard"]["positions"]]
    want = common["limit_clips"] or common["train"]["n_clips"]
    if len(pos) != len(set(pos)) or sorted(pos) != list(range(want)):
        raise SystemExit(f"REFUSED: the shards' clip positions are not a disjoint cover of 0..{want - 1}")
    n_win = sum(h["shard"]["n_windows"] for _, _, h in loaded)
    if not common["partial"] and n_win != common["train"]["n_windows"]:
        raise SystemExit(f"REFUSED: the shards hold {n_win} windows, the TRAIN list has {common['train']['n_windows']}")
    clips = {}
    for _, _, h in loaded:
        clips.update(h["clips"])
    try:
        G.check_train_only(bank_clips=clips.values(), train_clips=clips.values(), eval_clips=common["eval"]["clips"], complete=False)
    except G.BankRefused as e:
        raise SystemExit(f"REFUSED: {e}")
    provs = [json.load(open(f[:-4] + ".provenance.json", encoding="utf-8")) for f in files]
    if any(p["kb1_replay"]["mismatch_chunks"] for p in provs):
        raise SystemExit("REFUSED: K-B1 failed in at least one shard")
    no_ctl = [os.path.basename(f)[:-4] + ".provenance.json" for f, p in zip(files, provs) if not all(isinstance((p.get("controls") or {}).get(k), dict) for k in ("K-B4", "K-B5"))]
    if no_ctl:
        raise SystemExit(f"REFUSED: {no_ctl} carry no controls.K-B4 / controls.K-B5 (a shard from a generator that did not run them): regenerate the shard(s)")
    controls = {"K-B4": G.kb4_combine([p["controls"]["K-B4"] for p in provs]),          # PREREG section 6: the six matcher cases (re-run now, AND every shard's) ...
                "K-B5": G.kb5_combine([p["controls"]["K-B5"] for p in provs])}          # ... and the re-anchoring errors measured on the shards' own records (PASS recomputed from the numbers, never from a shard's flag)
    failed_controls = [k for k, v in controls.items() if v["PASS"] is not True]
    shards = [{"file": os.path.basename(f), "k": h["shard"]["k"], "n_records": h["n_records"], "file_sha256": G.sha256_file(f), "content_sha256": G.content_sha256(arr)}
              for f, arr, h in ((f, x, h) for f, x, h in loaded)]
    cov_n = sum(p["coverage"]["n_windows"] for p in provs)
    cov_e = sum(p["coverage"]["n_eligible"] for p in provs)
    by_tick = {t: {r: sum(p["reasons_by_tick"][t][r] for p in provs) for r in G.REASONS} for t in map(str, G.TICKS)}
    cov = {"n_windows": cov_n, "n_eligible": cov_e, "fraction": cov_e / max(cov_n, 1), "min": G.COVERAGE_MIN, "blocked": bool(cov_e / max(cov_n, 1) < G.COVERAGE_MIN)}
    man = {"schema": G.SCHEMA, "n_shards": n, "shards": sorted(shards, key=lambda s: s["k"]), "common": common, "partial": common["partial"], "n_records": sum(s["n_records"] for s in shards),
           "reasons_by_tick": by_tick, "coverage": cov}
    prov_path = os.path.join(a.out_dir, "PROVENANCE.json")
    prov_common = {"schema": G.SCHEMA, "kind": "bank", "created_unix": time.time(), "shard_provenance": [os.path.basename(f)[:-4] + ".provenance.json" for f in files],
                   "n_records": man["n_records"], "coverage": cov, "reasons_by_tick": by_tick, "partial": man["partial"], "controls": controls,
                   "warnings": [w for p in provs for w in p["warnings"]]}
    if failed_controls:                                                                 # a failed control: PROVENANCE.json records PASS false, and NO bank.json is written (a stale one cannot match bank_sha256 = null)
        with open(prov_path, "w", encoding="utf-8") as fh:
            json.dump({**prov_common, "bank_sha256": None, "manifest": None, "complete": False, "failed_controls": failed_controls,
                       "warnings": prov_common["warnings"] + [f"CONTROL {k} FAILED: no bank was written" for k in failed_controls]}, fh, indent=1, default=G._jsonable)
        log(f"REFUSED: control(s) {failed_controls} FAILED; wrote {prov_path} with PASS false and NO bank.json")
        return 6
    mp = os.path.join(a.out_dir, "bank.json")
    with open(mp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(man, sort_keys=True, indent=1, default=G._jsonable) + "\n")
    sha = G.sha256_file(mp)
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump({**prov_common, "bank_sha256": sha, "manifest": "bank.json"}, fh, indent=1, default=G._jsonable)
    log(f"wrote {mp}  bank sha256 {sha}  records {man['n_records']}  coverage {cov['fraction']:.4f}{'  (BLOCKED: below ' + str(G.COVERAGE_MIN) + ')' if cov['blocked'] else ''}  K-B4 {controls['K-B4']['PASS']}  K-B5 {controls['K-B5']['PASS']}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--ckpt", required=True, help="the frozen M2b_s{seed}/ckpt_last.pt")
    g.add_argument("--m2b-argv", required=True, help="M2b's argv.json: the TRAIN list is rebuilt from it by the trainer's build_data")
    g.add_argument("--cache-dir", default="", help="override M2b's --cache-dir (the TRAIN S4 cache)")
    g.add_argument("--tracks", default="", help="override M2b's --tracks (native_tracks.pt)")
    g.add_argument("--join-train", default="", help="override M2b's --join-train (join_train.npz)")
    g.add_argument("--eval-grid", required=True, help="eval_grid_windows.json: the eval-141 clip list (the bank must be disjoint from it)")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--n-shards", type=int, default=1)
    g.add_argument("--limit-clips", type=int, default=0, help="smoke: only the first N TRAIN clips (the bank is then marked PARTIAL)")
    g.add_argument("--batch", type=int, default=32)
    g.add_argument("--threads", type=int, default=0)
    g.add_argument("--device", default="auto")
    g.add_argument("--replay-frac", type=float, default=0.005, help="K-B1: the seeded fraction of chunks re-run (0 = off)")
    g.add_argument("--replay-seed", type=int, default=0)
    g.add_argument("--expect-clips", type=int, default=0, help="refuse unless the TRAIN list has this many clips (M2b: 4512)")
    g.add_argument("--expect-windows", type=int, default=0, help="refuse unless the TRAIN list has this many windows (M2b: 306566)")
    g.add_argument("--thresholds-file", default="", help="probe/THRESHOLDS_FIXED.txt: its sha256 is recorded in the header")
    g.add_argument("--dry-run", action="store_true", help="apply every guard, print the plan, write nothing")
    m = sub.add_parser("merge")
    m.add_argument("--out-dir", required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return {"gen": cmd_gen, "merge": cmd_merge}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
