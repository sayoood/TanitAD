"""D1 -- the per-window LABEL / INPUT COVERAGE census of refcv7-r101-s0 (refcv8 data audit, 2026-10-04).

For every TRAINING window (746,946 expected) and every EVAL window (23,772 expected): which supervision /
input channel carries a value, built through the trainer's OWN dataset objects (the launch tree
`fec3a0d`), plus label-side ego-future geometry from the clip's own poses.

HOW THE "NO FRAMES" PATH IS TAKEN (and why it is not a re-implementation)
-------------------------------------------------------------------------
The dataset is built EXACTLY as `refc_v3_train.train()` builds it (every block cites the trainer line it
mirrors): `build_v2_providers` (lazy: poses resident, JPEG/PNG payloads never read), `V3Dataset`,
`enable_clip_clock` + `assert_label_clock_true` (G3), `enable_max_speed_v6`, `enable_nav_from_v7`,
`enable_agent_join`, `enable_map_hires`, `enable_join3d`. Then, per window, the label block of
`V3Dataset.__getitem__` (refc_v3_train.py:3836-3938) is evaluated WITHOUT the frame decode of
`super().__getitem__` -- it calls the SAME functions (`ds._now_s`, `v7l.tactical_class_ids`,
`v7l.tactical_goal_targets`, `ds._nav_by_sid`, `ds._max_speed_by_sid`, `ds.agent_join.lookup`,
`_agent_cuboid.zh_for_frame`, the map store's n_frames). Control C2 then checks the direct values against the
FULL `ds[i]` (frames decoded) on a sampled set of windows -- the `label_clock_table.py` approach.

DEPARTURES FROM train() (recorded in the output record):
  * JoinFileReader is built `with_rates=False` (the rates are loss targets; they never change which
    (clip, frame) carry a label or how many agents there are -- config.json's `n_windows_labelled` 719,739 is
    the control that proves it);
  * no model attributes (`_tac_goal_pos_weight` ...), no VIS-1 sidecar, no calibration loader (none of them
    touches a label presence);
  * `--max-clips N` (sample mode) builds the dataset over a seeded subset of clips (listed by sha12).

Run: PYTHONPATH is set by this script from --repo. On Thor: --repo /home/nvidia/refcv7_run/fec3a0dccf
--kit /home/nvidia. CPU only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

T_PROC0 = time.time()


def log(*a):
    print(f"[d1 {time.time() - T_PROC0:7.1f}s]", *a, flush=True)


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:12]


def md5_file(p, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def wrap_pi(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


# -------------------------------------------------------------------------------------------------- #
# environment                                                                                         #
# -------------------------------------------------------------------------------------------------- #
def setup_env(repo: str, kit: str):
    os.environ["REFCV6_REPO"] = repo
    os.environ["REFCV6_KIT"] = kit
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for p in (f"{repo}/stack", f"{repo}/taniteval", f"{repo}/stack/scripts", f"{repo}/taniteval/tools"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tanitad
    here = os.path.normcase(os.path.abspath(tanitad.__file__))
    want = os.path.normcase(os.path.abspath(f"{repo}/stack"))
    if not here.startswith(want):
        raise SystemExit(f"tanitad imported from {here}, not {want}")
    log("tanitad from", tanitad.__file__)


def blob_report(repo: str, paths):
    """git blob id of each launch-tree file, computed WITHOUT filters (LF-normalised if CRLF present)."""
    out = {}
    for p in paths:
        raw = Path(repo, p).read_bytes()
        crlf = raw.count(b"\r\n")
        lf = raw.replace(b"\r\n", b"\n")
        h = hashlib.sha1(b"blob %d\0" % len(lf) + lf).hexdigest()
        out[p] = {"git_blob_raw_bytes": hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest(),
                  "git_blob_lf_normalised": h, "n_crlf": crlf, "bytes": len(raw)}
    return out


# -------------------------------------------------------------------------------------------------- #
# config -> args -> cfg (the trainer's own pin; only window / channels are read from it)              #
# -------------------------------------------------------------------------------------------------- #
def make_args_cfg(config: dict, L, tr):
    args, argv, arec = L.parse_args(config)
    from tanitad.refs import refc_v3 as v3
    rec = {"argv_remap": arec}
    try:
        tr._check_nav_from_v7_args(args)
        tr._check_max_speed_args(args)
        tr._check_goal_point_args(args)
        tr.check_effective_weights(args)
        art = tr._read_anchor_artifact(args)
        cfg = tr._pin_trainer_cfg(
            v3.refc_v3_smoke_config(args.arm == "hier") if args.smoke else
            v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
        rec["cfg"] = "trainer _pin_trainer_cfg"
    except SystemExit as e:                        # pragma: no cover - departure, recorded
        raise SystemExit(f"cfg pin refused: {e}")
    return args, cfg, rec


# -------------------------------------------------------------------------------------------------- #
# raw label records (nav_30s etc.) -- read straight from the blob, keyed like the trainer             #
# -------------------------------------------------------------------------------------------------- #
def _f(x):
    return float("nan") if x is None else float(x)


def read_raw_records(path: str):
    from tanitad.data.v2_dataset import stable_episode_id
    by_sid = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            keep = {
                "t0_s": float(r.get("t0_s")),
                "bands": r.get("bands"),
                "tac_lat": (r.get("a_tac") or {}).get("lat"),
                "tac_lon": (r.get("a_tac") or {}).get("lon"),
                "nav_token": (r.get("nav_command") or {}).get("token"),
                "nav_args": (r.get("nav_command") or {}).get("args"),
                "nav_entries": [
                    {"t_start_s": _f(e.get("t_start_s")), "t_end_s": _f(e.get("t_end_s")),
                     "dyaw_deg": _f(e.get("dyaw_deg")), "token": e.get("token"),
                     "distance_m": _f(e.get("distance_m"))}
                    for e in ((r.get("nav_30s") or {}).get("entries") or [])],
                "v_hi_ms_band": (((r.get("g_tac") or {}).get("goals") or {}).get("SPEED_BAND") or {}).get("v_hi_ms"),
                "strata": r.get("strata") or {},
                "alp_lat_agree": ((r.get("alpamayo") or {}).get("lateral") or {}).get("agree"),
            }
            by_sid[stable_episode_id(r["clip_id"])] = (sha12(r["clip_id"]), keep)
    return by_sid


# -------------------------------------------------------------------------------------------------- #
# dataset builders                                                                                    #
# -------------------------------------------------------------------------------------------------- #
def pick_subset(eps, man_clip_ids, n, seed):
    """seeded subset of providers, by sha12 order -> returns sorted list of provider indices."""
    order = sorted(range(len(eps)), key=lambda i: sha12(man_clip_ids[i]))
    rng = np.random.default_rng(seed)
    pick = sorted(int(x) for x in rng.choice(len(order), size=min(n, len(order)), replace=False))
    return [order[k] for k in pick]


def build_train(tr, L, cfg, args, config, max_clips, seed, kit):
    """train():8336-8700, label-relevant blocks only (line refs = fec3a0d)."""
    from tanitad.data import v7_labels as v7l
    from tanitad.data.v2_dataset import (build_v2_providers, load_or_build_manifest, stable_episode_id)
    rec = {"departures": []}
    # 8336-8346: lazy providers (poses resident, payloads never read)
    eps = build_v2_providers(args.v2_cache, lru_size=args.v2_lru)
    _dirs = [args.v2_cache] if isinstance(args.v2_cache, (str, os.PathLike)) else list(args.v2_cache)
    assert len(_dirs) == 1, f"one v2 cache dir expected, got {_dirs}"
    man = load_or_build_manifest(_dirs[0], verbose=False)
    cids = list(man["clip_id"])
    assert len(cids) == len(eps)
    sub = None
    if max_clips and max_clips < len(eps):
        sub = pick_subset(eps, cids, max_clips, seed)
        eps = [eps[i] for i in sub]
        rec["departures"].append(f"SAMPLE MODE: {len(eps)} of {len(cids)} clips (seed {seed}), by sha12")
    # 8348-ish: the trainer's own train/eval overlap refusal (episode ids; eval cache manifest)
    try:
        _ev_man = load_or_build_manifest(args.eval_cache, verbose=False)
        _ov = {int(e.episode_id) for e in eps} & {int(u) for u in _ev_man["episode_uid"]}
        rec["train_eval_episode_overlap"] = len(_ov)
        if _ov:
            raise SystemExit(f"{len(_ov)} episodes appear in both the train and the eval cache")
    except FileNotFoundError:
        rec["train_eval_episode_overlap"] = "eval cache not readable on this host"
    kw = dict(window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    ds = tr.V3Dataset(eps, **kw)                                   # 8377
    ds.u8_frames = bool(getattr(args, "u8_batches", False))        # 8384
    # 8392-8398: labels, clock, G3
    labels, manifest = v7l.load_v7_labels(args.v7_labels, allow_oracle_nav=True)
    geom_train_state = frozenset(v7l._MEASURED_GEOMETRY_TOKENS)
    ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in labels}
    ds.v7_dt = 0.1
    rec["clock"] = ds.enable_clip_clock(getattr(args, "clip_clock_sidecar", None))
    rec["clock"]["g3"] = ds.assert_label_clock_true(
        getattr(args, "clip_clock_sidecar", None), split="train",
        synthetic=bool(getattr(args, "synth_episodes", 0)),
        max_unverified_frac=getattr(args, "label_clock_max_unverified", None))
    rec["labels"] = manifest.to_dict()
    ds.ego_history = bool(getattr(args, "ego_history", False))
    ds.r7_agent_future = float(getattr(args, "w_r7_scorer", 0.0) or 0.0) > 0.0
    if (float(getattr(args, "w_tac_goal", 0.0) or 0.0) > 0.0
            or float(getattr(args, "w_tac_v6", 0.0) or 0.0) > 0.0):         # 8466-8470
        ds.tac_goal_targets = True
        ds.tac_goal_negatives = str(getattr(args, "tac_goal_negatives", "measured"))
    # 8559-8566: the v6 ceiling and the nav source
    rec["max_speed_v6"] = ds.enable_max_speed_v6(str(args.speed_max_sidecar_v6), manifest)
    rec["nav"] = ds.enable_nav_from_v7(manifest)
    # 8586-8610: the 2-D agent join (departure: with_rates=False; track ids kept for the 3-D join)
    from train_p8_occupancy import JoinFileReader
    t_j = time.time()
    rd = JoinFileReader(args.agent_join, episode_ids={int(e.episode_id) for e in eps}, with_rates=False,
                        with_track_ids=bool(getattr(args, "join3d", None)))
    log(f"agent join loaded: {rd.n_records} records / {rd.n_clips} clips in {time.time() - t_j:.1f}s")
    rec["departures"].append("JoinFileReader with_rates=False (rates are loss targets; no label presence effect)")
    rec["agent_join"] = ds.enable_agent_join(rd, pad=int(getattr(args, "agent_pad", 0)),
                                             allow_legacy_ids=bool(getattr(args, "agent_join_allow_legacy_ids", False)))
    # 8618-8680: map + 3-D join
    clip_of_ep, n_stack = tr._clip_table_for_caches(args.v2_cache)
    if os.environ.get("D1_SMOKE_MAP_MIN_COVERAGE") is not None:      # SMOKE ONLY (dev box has no train SAM3 GT)
        args.map_min_coverage = float(os.environ["D1_SMOKE_MAP_MIN_COVERAGE"])
        rec["departures"].append(f"SMOKE: --map-min-coverage forced to {args.map_min_coverage}")
    store = tr._sem_fine.FineMapGTStore(Path(args.map_gt_root), max_open=int(getattr(args, "map_lru", 4) or 4),
                                        extent=tr._mhr.declared_extent(args))
    n_frames_of = {}
    _orig_open = store.open

    def _open_rec(cid):
        gt = _orig_open(cid)
        n_frames_of[sha12(cid)] = int(gt.n_frames)
        return gt
    store.open = _open_rec                                          # records n_frames during the trainer's own census
    t_m = time.time()
    rec["map_fine"] = ds.enable_map_hires(store, clip_of_ep, n_stack,
                                          min_coverage=getattr(args, "map_min_coverage", None))
    log(f"map census {time.time() - t_m:.1f}s over {len(n_frames_of)} clips")
    if getattr(args, "join3d", None):
        if ds.map_clip_of_ep is None:
            ds.map_clip_of_ep, ds.map_n_stack = clip_of_ep, n_stack
        t_j = time.time()
        rec["join3d"] = ds.enable_join3d(tr.require_join3d(
            tr._agent_cuboid.open_join3d(args.join3d, clips=set(clip_of_ep.values())),
            args.join3d, len(set(clip_of_ep.values()))))
        log(f"join3d loaded {time.time() - t_j:.1f}s")
    # eval blob is loaded AFTER the train blob in train() (8736) and the module-level negative policy
    # (`_MEASURED_GEOMETRY_TOKENS`) is overwritten by it; the DataLoader (8957) is created afterwards.
    ev_labels, ev_man = v7l.load_v7_labels(args.eval_labels, allow_oracle_nav=True)
    geom_eval_state = frozenset(v7l._MEASURED_GEOMETRY_TOKENS)
    rec["geom_tokens"] = {"train_blob_state": sorted(geom_train_state), "eval_blob_state_as_in_workers": sorted(geom_eval_state)}
    return ds, rec, n_frames_of, geom_train_state, geom_eval_state, sub


def build_eval(tr, L, cfg, args, config):
    """the launch tree's own eval-dataset builder (refcv7_loader.build_eval_dataset == train():8724-8895)."""
    from tanitad.data import v7_labels as v7l
    ds, eps, rec = L.build_eval_dataset(None, cfg, args, config, with_perception_targets=True)
    geom_eval_state = frozenset(v7l._MEASURED_GEOMETRY_TOKENS)
    store = ds.map_fine_store
    n_frames_of = {}
    for cid in set(ds.map_clip_of_ep.values()):
        try:
            n_frames_of[sha12(cid)] = int(store.open(cid).n_frames)
        except Exception as e:                                     # recorded, not swallowed
            n_frames_of[sha12(cid)] = -1
            rec.setdefault("map_open_errors", []).append(f"{sha12(cid)}: {type(e).__name__}")
    return ds, rec, n_frames_of, geom_eval_state, geom_eval_state, None


# -------------------------------------------------------------------------------------------------- #
# the per-window census                                                                               #
# -------------------------------------------------------------------------------------------------- #
SLOTS = (5, 10, 15, 20, 30, 40, 50, 60)       # the V3 waypoint slots in ticks (cfg.horizons)
IGN = -100


def census(ds, tr, args, raw_by_sid, n_frames_of, geom_state_A, geom_state_B, rec_out):
    from tanitad.data import v7_labels as v7l
    from tanitad.refs import refcv6_max_speed as v6ms
    agent_cuboid = tr._agent_cuboid
    w = int(ds.window)
    n_tok = len(v7l.TAC_GOAL_TOKENS)
    N = len(ds.index)
    log(f"census over {N} windows, {len(ds.episodes)} clips, window {w}")
    cols = {}

    def col(name, dtype, fill=0):
        a = np.full(N, fill, dtype=dtype)
        cols[name] = a
        return a
    c_clip = col("clip_ix", np.int32)
    c_t = col("t", np.int16)
    c_now = col("t_now_s", np.float64, np.nan)
    c_hasrec = col("has_record", np.bool_)
    c_inband = col("in_band", np.bool_)
    c_lat = col("lat_v7", np.int8, IGN)
    c_lon = col("lon_v7", np.int8, IGN)
    c_nsA = col("n_goal_scored_censusstate", np.int8)
    c_nsB = col("n_goal_scored", np.int8)
    c_npos = col("n_goal_pos", np.int8)
    c_yb = col("goal_y_bits", np.uint32)
    c_wbA = col("goal_w_bits_censusstate", np.uint32)
    c_wbB = col("goal_w_bits", np.uint32)
    c_nav = col("nav_cmd", np.int8)
    c_navv = col("nav_valid", np.bool_)
    c_vms = col("vmax_ms", np.float32, np.nan)
    c_vok = col("vmax_valid", np.bool_)
    c_vbin = col("vmax_bin", np.int8, -1)
    c_ag = col("agent_labelled", np.bool_)
    c_nag = col("n_agents", np.int16, -1)
    c_b3 = col("box3d_labelled", np.bool_)
    c_nb3 = col("n_box3d", np.int16, -1)
    c_map = col("map_label", np.bool_)
    c_nfv = col("n_future_valid", np.int16)
    c_full6 = col("full6", np.bool_)
    gnames = ["v0", "v2", "v4", "v6", "dyaw6_deg", "lat6_m", "fwd6_m", "vmax26", "vmin26", "vmin06",
              "theta_slot_deg", "path_len_slot_m", "ttn_s", "ttn_dyaw_deg", "turn_now_dyaw_deg"]
    for g in gnames:
        col(g, np.float32, np.nan)
    c_stop = col("stop_0_6", np.bool_)
    c_inturn = col("in_turn_now", np.bool_)
    c_ttnside = col("ttn_side", np.int8)          # +1 left, -1 right, 0 none
    c_nturn_ahead = col("n_turns_in_labels", np.int8)

    # ---- clip table ----------------------------------------------------------------------------- #
    clips = []
    sid_of_e = [int(e.episode_id) for e in ds.episodes]
    cid_of_sid = ds.map_clip_of_ep
    # contiguity of ds.index per clip
    first_row = {}
    last_e = -1
    for r, (e_i, t) in enumerate(ds.index):
        if e_i != last_e:
            assert e_i not in first_row, "ds.index is not clip-contiguous"
            first_row[e_i] = r
            last_e = e_i
    ladder_kmh = v6ms.SPEED_MAX_STEPS_KMH_V6
    t_loop = time.time()
    for e_i, ep in enumerate(ds.episodes):
        sid = sid_of_e[e_i]
        r0 = first_row[e_i]
        # rows for this clip
        n_w = 0
        while r0 + n_w < N and ds.index[r0 + n_w][0] == e_i:
            n_w += 1
        ts = np.array([ds.index[r0 + k][1] for k in range(n_w)], dtype=np.int64)
        rr = slice(r0, r0 + n_w)
        c_clip[rr] = e_i
        c_t[rr] = ts
        lab = ds.v7_by_sid.get(sid)
        if sid in ds.tactical_excluded_sids:
            lab = None
        cid = cid_of_sid[sid]
        s12 = sha12(cid)
        poses = ep.poses.numpy().astype(np.float64)
        T = poses.shape[0]
        g0, dt, src = ds._clock_for(ep)
        now_rows = ts + (w - 1)
        # ---- label clock + tactical block (refc_v3_train.py:3836-3874) ----
        tn = np.array([ds._now_s(ep, int(t)) for t in ts], dtype=np.float64)
        c_now[rr] = tn
        c_hasrec[rr] = lab is not None
        if lab is not None:
            ib = np.array([v7l.window_in_band(lab, float(x)) for x in tn], dtype=bool)
            c_inband[rr] = ib
            la = np.full(n_w, IGN, np.int8)
            lo = np.full(n_w, IGN, np.int8)
            nsA = np.zeros(n_w, np.int8); nsB = np.zeros(n_w, np.int8); npos = np.zeros(n_w, np.int8)
            yb = np.zeros(n_w, np.uint32); wA = np.zeros(n_w, np.uint32); wB = np.zeros(n_w, np.uint32)
            cacheA = cacheB = None
            for k in np.nonzero(ib)[0]:
                a, b = v7l.tactical_class_ids(lab, float(tn[k]))
                la[k], lo[k] = a, b
                if cacheB is None:
                    v7l._MEASURED_GEOMETRY_TOKENS = geom_state_A
                    yA, wA_ = v7l.tactical_goal_targets(lab, float(tn[k]), negatives=ds.tac_goal_negatives,
                                                        sidecar=ds.cot_negative_sidecar)
                    v7l._MEASURED_GEOMETRY_TOKENS = geom_state_B
                    yB, wB_ = v7l.tactical_goal_targets(lab, float(tn[k]), negatives=ds.tac_goal_negatives,
                                                        sidecar=ds.cot_negative_sidecar)
                    assert yA == yB
                    cacheA, cacheB = (yA, wA_), (yB, wB_)
                y_, wa_ = cacheA
                wb_ = cacheB[1]
                nsA[k] = sum(1 for x in wa_ if x > 0)
                nsB[k] = sum(1 for x in wb_ if x > 0)
                npos[k] = sum(1 for x, ww in zip(y_, wb_) if x > 0 and ww > 0)
                yb[k] = sum(1 << i for i, x in enumerate(y_) if x > 0)
                wA[k] = sum(1 << i for i, x in enumerate(wa_) if x > 0)
                wB[k] = sum(1 << i for i, x in enumerate(wb_) if x > 0)
            c_lat[rr], c_lon[rr] = la, lo
            c_nsA[rr], c_nsB[rr], c_npos[rr] = nsA, nsB, npos
            c_yb[rr], c_wbA[rr], c_wbB[rr] = yb, wA, wB
        # ---- nav + ceiling (3888-3925) ----
        nav_idx = ds._nav_by_sid.get(sid)
        c_nav[rr] = 0 if nav_idx is None else nav_idx
        c_navv[rr] = nav_idx is not None
        raw = (ds._max_speed_by_sid or {}).get(sid)
        v_ms, ok = (0.0, 0.0) if raw is None else raw
        c_vms[rr] = v_ms
        c_vok[rr] = ok > 0.5
        if ok > 0.5:
            c_vbin[rr] = v6ms.speed_max_bin(float(v_ms))[0]
        # ---- agents / 3-D / map (3939-3978, _agent_item 3260-3300) ----
        rd = ds.agent_join
        n_raw = np.full(n_w, -1, np.int16)
        has = np.zeros(n_w, np.bool_)
        n3 = np.full(n_w, -1, np.int16)
        for k in range(n_w):
            f = int(now_rows[k])
            ag = rd.lookup(sid, f)
            if ag is None:
                continue
            has[k] = True
            n_raw[k] = int(ag.shape[0])
            if ds.join3d is not None:
                tids = rd.lookup_track_ids(sid, f)
                if tids is None or ag.shape[0] == 0:
                    n3[k] = 0
                else:
                    _cz, _h, _m = agent_cuboid.zh_for_frame(cid, f + int(ds.map_n_stack) - 1, list(tids),
                                                            join3d=ds.join3d)
                    n3[k] = int(np.asarray(_m).sum())
        c_ag[rr], c_nag[rr], c_nb3[rr] = has, n_raw, n3
        c_b3[rr] = n3 > 0
        nf = n_frames_of.get(s12, -1)
        raw_f = now_rows + (int(ds.map_n_stack) - 1)
        c_map[rr] = (nf > 0) & (raw_f >= 0) & (raw_f < nf)
        # ---- ego-future geometry from the clip's own poses (label side) ----
        x, y, yaw, v = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
        nfv = np.clip(T - 1 - now_rows, 0, 60)
        c_nfv[rr] = nfv
        k2, k4, k6 = int(round(2.0 / dt)), int(round(4.0 / dt)), int(round(6.0 / dt))
        full6 = (now_rows + k6) <= (T - 1)
        c_full6[rr] = full6
        v0 = v[now_rows]
        cols["v0"][rr] = v0

        def at(k):
            idx = now_rows + k
            out = np.full(n_w, np.nan)
            okk = idx <= (T - 1)
            out[okk] = v[idx[okk]]
            return out, okk
        cols["v2"][rr] = at(k2)[0]
        cols["v4"][rr] = at(k4)[0]
        cols["v6"][rr] = at(k6)[0]
        # heading / lateral at +6 s in the NOW ego frame
        idx6 = np.minimum(now_rows + k6, T - 1)
        dy_ = np.degrees(wrap_pi(yaw[idx6] - yaw[now_rows]))
        dx_w, dy_w = x[idx6] - x[now_rows], y[idx6] - y[now_rows]
        c0, s0 = np.cos(yaw[now_rows]), np.sin(yaw[now_rows])
        fwd = c0 * dx_w + s0 * dy_w
        lat = -s0 * dx_w + c0 * dy_w
        cols["dyaw6_deg"][rr] = np.where(full6, dy_, np.nan)
        cols["fwd6_m"][rr] = np.where(full6, fwd, np.nan)
        cols["lat6_m"][rr] = np.where(full6, lat, np.nan)
        # speed extremes over [+2,+6] and [0,+6] (rows now+k2 .. now+k6)
        vmax = np.full(n_w, np.nan); vmin = np.full(n_w, np.nan); vmin0 = np.full(n_w, np.nan)
        for k in np.nonzero(full6)[0]:
            a = int(now_rows[k])
            seg = v[a + k2:a + k6 + 1]
            vmax[k], vmin[k] = seg.max(), seg.min()
            vmin0[k] = v[a:a + k6 + 1].min()
        cols["vmax26"][rr], cols["vmin26"][rr], cols["vmin06"][rr] = vmax, vmin, vmin0
        c_stop[rr] = np.where(full6, vmin0 < 0.5, False)
        # the route-following package's own terminal-heading definition (SPEC sec. 3): slots in TICKS
        okslot = (now_rows + 60) <= (T - 1)
        P = []
        for h in SLOTS:
            ii = np.minimum(now_rows + h, T - 1)
            ddx, ddy = x[ii] - x[now_rows], y[ii] - y[now_rows]
            P.append(np.stack([c0 * ddx + s0 * ddy, -s0 * ddx + c0 * ddy], -1))
        P = np.stack(P, 1)                                          # [n_w, 8, 2]
        d = P[:, -1] - P[:, -2]
        th = np.arctan2(d[:, 1], d[:, 0])
        th = np.where(np.hypot(d[:, 0], d[:, 1]) < 0.05, 0.0, th)
        Q = np.concatenate([np.zeros((n_w, 1, 2)), P], 1)
        plen = np.linalg.norm(np.diff(Q, axis=1), axis=-1).sum(-1)
        cols["theta_slot_deg"][rr] = np.where(okslot, np.degrees(th), np.nan)
        cols["path_len_slot_m"][rr] = np.where(okslot, plen, np.nan)
        # ---- time to the next turn start, from the record's nav_30s.entries -------------------------
        sid_raw = raw_by_sid.get(sid)
        if sid_raw is not None:
            rec_ = sid_raw[1]
            t0 = float(rec_["t0_s"])
            # nav_30s.entries also carries NAV_FOLLOW_ROAD pseudo-entries (t 0..30, dyaw None = "no commanded turn
            # within the horizon"); only NAV_TURN_L / NAV_TURN_R are turns
            ent = [e for e in rec_["nav_entries"] if e["token"] in ("NAV_TURN_L", "NAV_TURN_R")]
            nturn = len(ent)
            c_nturn_ahead[rr] = nturn
            if nturn:
                st = np.array([t0 + e["t_start_s"] for e in ent])
                en = np.array([t0 + e["t_end_s"] for e in ent])
                sd = np.array([1 if e["token"] == "NAV_TURN_L" else (-1 if e["token"] == "NAV_TURN_R" else 0) for e in ent])
                dyd = np.array([e["dyaw_deg"] for e in ent])
                ttn = np.full(n_w, np.nan); ttn_dy = np.full(n_w, np.nan); side = np.zeros(n_w, np.int8)
                inturn = np.zeros(n_w, bool); tdy = np.full(n_w, np.nan)
                for k in range(n_w):
                    up = np.nonzero(st >= tn[k])[0]
                    if up.size:
                        j = up[np.argmin(st[up] - tn[k])]
                        ttn[k] = st[j] - tn[k]; ttn_dy[k] = dyd[j]; side[k] = sd[j]
                    ong = np.nonzero((st <= tn[k]) & (en >= tn[k]))[0]
                    if ong.size:
                        inturn[k] = True; tdy[k] = dyd[ong[0]]
                cols["ttn_s"][rr] = ttn; cols["ttn_dyaw_deg"][rr] = ttn_dy; c_ttnside[rr] = side
                c_inturn[rr] = inturn; cols["turn_now_dyaw_deg"][rr] = tdy
        clips.append({"clip_ix": e_i, "sha12": s12, "T": int(T), "n_windows": int(n_w), "dt_s": float(dt),
                      "grid_start_s": float(g0), "clock_src": src,
                      "map_n_frames": int(nf), "has_record": lab is not None,
                      "tac_lat": None if sid_raw is None else sid_raw[1]["tac_lat"],
                      "tac_lon": None if sid_raw is None else sid_raw[1]["tac_lon"],
                      "nav_token": None if sid_raw is None else sid_raw[1]["nav_token"],
                      "nav_cmd": None if nav_idx is None else int(nav_idx),
                      "nav_args": None if sid_raw is None else sid_raw[1]["nav_args"],
                      "n_nav_entries": 0 if sid_raw is None else len(sid_raw[1]["nav_entries"]),
                      "t0_s": None if sid_raw is None else sid_raw[1]["t0_s"],
                      "tactical_band_s": None if sid_raw is None else (sid_raw[1]["bands"] or {}).get("tactical_s"),
                      "v_hi_ms_band": None if sid_raw is None else sid_raw[1]["v_hi_ms_band"],
                      "vmax_ms_fed": float(v_ms), "strata": None if sid_raw is None else sid_raw[1]["strata"]})
        if (e_i + 1) % 250 == 0:
            el = time.time() - t_loop
            log(f"  clip {e_i + 1}/{len(ds.episodes)}  {el:.0f}s  ETA {(len(ds.episodes) - e_i - 1) * el / (e_i + 1):.0f}s")
    return cols, clips


# -------------------------------------------------------------------------------------------------- #
# C2: direct == full __getitem__ (frames decoded)                                                     #
# -------------------------------------------------------------------------------------------------- #
def control_c2(ds, cols, n_win=24, n_clips_wanted=6, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    ncl = int(cols["clip_ix"].max()) + 1
    clip_pick = rng.choice(ncl, size=min(n_clips_wanted, ncl), replace=False)
    rows = []
    for c in clip_pick:
        idx = np.nonzero(cols["clip_ix"] == c)[0]
        inb = idx[cols["in_band"][idx]]
        outb = idx[~cols["in_band"][idx]]
        take = []
        if inb.size:
            take += list(rng.choice(inb, size=min(2, inb.size), replace=False))
        take += list(rng.choice(outb, size=min(2, outb.size), replace=False))
        rows += [int(x) for x in take]
    rows = rows[:max(n_win, len(rows))]
    mism = []
    for r in rows:
        it = ds[r]
        chk = {
            "lat": (int(it["lat_v7"]), int(cols["lat_v7"][r])),
            "lon": (int(it["lon_v7"]), int(cols["lon_v7"][r])),
            "nav_cmd": (int(it["nav_cmd"]), int(cols["nav_cmd"][r])),
            "nav_valid": (bool(it["nav_valid"]), bool(cols["nav_valid"][r])),
            "v_max_valid": (bool(float(it["v_max_valid"]) > 0.5), bool(cols["vmax_valid"][r])),
            "agent_label": (bool(it["agent_label"]), bool(cols["agent_labelled"][r])),
            "agent_n_raw": (int(it["agent_n_raw"]), int(cols["n_agents"][r]) if cols["agent_labelled"][r] else 0),
            "map_fine_label": (bool(it["map_fine_label"]), bool(cols["map_label"][r])),
        }
        v_it = float(it["v_max_ms"])
        v_col = float(cols["vmax_ms"][r])
        if not (abs(v_it - v_col) < 1e-5):
            mism.append({"row": r, "field": "v_max_ms", "got": v_it, "want": v_col})
        if "agent_zh_mask" in it:
            chk["box3d_any"] = (bool(it["agent_zh_mask"].any()), bool(cols["box3d_labelled"][r]))
        gy = it["tac_goal_y"].numpy(); gw = it["tac_goal_w"].numpy()
        ybits = int(sum(1 << i for i, x in enumerate(gy) if x > 0))
        wbits = int(sum(1 << i for i, x in enumerate(gw) if x > 0))
        chk["goal_y_bits"] = (ybits, int(cols["goal_y_bits"][r]))
        chk["goal_w_bits"] = (wbits, int(cols["goal_w_bits"][r]))
        for k, (a, b) in chk.items():
            if a != b:
                mism.append({"row": r, "field": k, "got": a, "want": b})
    return {"n_windows": len(rows), "n_clips": int(len({int(cols["clip_ix"][r]) for r in rows})),
            "n_in_band": int(sum(1 for r in rows if cols["in_band"][r])),
            "mismatches": mism, "pass": bool(len(rows) >= 20 and not mism)}


# -------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------- #
# pose-side controls: the sign convention, the clock and the speed units, against the LABEL FILE       #
# -------------------------------------------------------------------------------------------------- #
def pose_controls(ds, raw_by_sid):
    """Independent of the census columns: for every nav_30s entry that lies fully inside the cached pose
    range, compare the label's own `dyaw_deg` (built by the label builder from the 100 Hz egomotion log)
    with the heading change of the CACHED poses between the entry's raw start and end times. Left is
    positive in both iff the pose yaw is CCW-positive and the clock is right. Also: pose speed channel
    against the finite-difference of pose xy (units: m/s)."""
    lab_dy, pose_dy, tok_side, shas = [], [], [], []
    ratios = []
    for e_i, ep in enumerate(ds.episodes):
        sid = int(ep.episode_id)
        poses = ep.poses.numpy().astype(np.float64)
        T = poses.shape[0]
        g0, dt, src = ds._clock_for(ep)
        off = ds._raw_offset(ep)
        yaw = np.unwrap(poses[:, 2])
        disp = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1])) / dt
        vm = 0.5 * (poses[1:, 3] + poses[:-1, 3])
        mv = vm > 2.0
        if mv.any():
            ratios.append(float(np.median(vm[mv] / np.maximum(disp[mv], 1e-9))))
        sr = raw_by_sid.get(sid)
        if sr is None or src != "sidecar":
            continue
        t0 = float(sr[1]["t0_s"])
        grid = np.arange(T, dtype=np.float64)
        for e in sr[1]["nav_entries"]:
            if e["token"] not in ("NAV_TURN_L", "NAV_TURN_R"):
                continue
            r_s = (t0 + e["t_start_s"] - g0) / dt - off
            r_e = (t0 + e["t_end_s"] - g0) / dt - off
            if r_s >= 0 and r_e <= T - 1:
                lab_dy.append(e["dyaw_deg"])
                pose_dy.append(float(np.degrees(np.interp(r_e, grid, yaw) - np.interp(r_s, grid, yaw))))
                tok_side.append(1 if e["token"] == "NAV_TURN_L" else (-1 if e["token"] == "NAV_TURN_R" else 0))
                shas.append(sha12(ds.map_clip_of_ep[sid]) if ds.map_clip_of_ep else str(e_i))
    lab_dy, pose_dy, tok_side = map(np.array, (lab_dy, pose_dy, tok_side))
    out = {"n_ratio_clips": len(ratios),
           "speed_over_finite_diff_median": float(np.median(ratios)) if ratios else None,
           "speed_over_finite_diff_p05_p95": [float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))] if ratios else None,
           "n_entries_inside_pose_range": int(lab_dy.size)}
    if lab_dy.size:
        big = np.abs(lab_dy) >= 20.0
        out.update({
            "label_token_L_has_positive_dyaw_deg": float(np.mean(lab_dy[tok_side == 1] > 0)) if (tok_side == 1).any() else None,
            "label_token_R_has_negative_dyaw_deg": float(np.mean(lab_dy[tok_side == -1] < 0)) if (tok_side == -1).any() else None,
            "n_big": int(big.sum()),
            "sign_agree_big": float(np.mean(np.sign(pose_dy[big]) == np.sign(lab_dy[big]))) if big.any() else None,
            "median_abs_diff_deg": float(np.median(np.abs(pose_dy - lab_dy))),
            "p90_abs_diff_deg": float(np.percentile(np.abs(pose_dy - lab_dy), 90)),
            "pearson_r": float(np.corrcoef(pose_dy, lab_dy)[0, 1]) if lab_dy.size > 2 else None,
            "mean_pose_dyaw_deg_by_token": {"L": float(pose_dy[tok_side == 1].mean()) if (tok_side == 1).any() else None,
                                            "R": float(pose_dy[tok_side == -1].mean()) if (tok_side == -1).any() else None}})
    return out, {"label_dyaw_deg": lab_dy, "pose_dyaw_deg": pose_dy, "token_side": tok_side}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "eval"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-c2", action="store_true")
    a = ap.parse_args()
    tag = a.tag or a.split
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_env(a.repo, a.kit)
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    from tanitad.eval import refcv7_loader as L
    L.bootstrap()
    tr = L.trainer()
    config = L.load_config(a.config)
    rec = {"tool": "d1_census.py", "split": a.split, "tag": tag, "repo": a.repo, "kit": a.kit, "config": a.config,
           "config_md5": md5_file(a.config), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "blobs": blob_report(a.repo, ["stack/scripts/refc_v3_train.py", "stack/tanitad/data/v7_labels.py",
                                          "stack/tanitad/data/v2_dataset.py"])}
    log("blobs", json.dumps(rec["blobs"]))
    args, cfg, prec = make_args_cfg(config, L, tr)
    rec["pin"] = prec
    from tanitad.data import v7_labels as _v7l
    from tanitad.refs import refcv6_max_speed as _v6ms
    from tanitad.refs import refb as _refb
    rec["vocab"] = {"goal_tokens": list(_v7l.TAC_GOAL_TOKENS), "lat_classes": list(_v7l.HEADS["tac_lat"]),
                    "lon_classes": list(_v7l.HEADS["tac_lon"]), "nav_commands": list(_refb.NAV_COMMANDS),
                    "speed_ladder_kmh": list(_v6ms.SPEED_MAX_STEPS_KMH_V6)}
    rec["window"] = int(cfg.core.window)
    rec["channels"] = int(cfg.core.encoder.in_channels)
    raw_path_pre = args.v7_labels if a.split == "train" else args.eval_labels
    raw_by_sid = read_raw_records(raw_path_pre)
    n_none = sum(1 for _, (sh, k) in raw_by_sid.items() for e in k["nav_entries"]
                 if e["token"] in ("NAV_TURN_L", "NAV_TURN_R")
                 and not (math.isfinite(e["t_start_s"]) and math.isfinite(e["t_end_s"]) and math.isfinite(e["dyaw_deg"])))
    n_follow = sum(1 for _, (sh, k) in raw_by_sid.items() for e in k["nav_entries"] if e["token"] == "NAV_FOLLOW_ROAD")
    n_turn = sum(1 for _, (sh, k) in raw_by_sid.items() for e in k["nav_entries"] if e["token"] in ("NAV_TURN_L", "NAV_TURN_R"))
    rec["raw_entries"] = {"turn_entries": n_turn, "follow_road_pseudo_entries": n_follow, "turn_entries_with_missing_field": n_none}
    log(f"raw label records read: {len(raw_by_sid)}; turn entries {n_turn}, follow-road pseudo entries {n_follow}, turn entries with a missing field {n_none}")
    t_b = time.time()
    if a.split == "train":
        ds, brec, n_frames_of, gA, gB, sub = build_train(tr, L, cfg, args, config, a.max_clips, a.seed, a.kit)
        raw_path = args.v7_labels
    else:
        ds, brec, n_frames_of, gA, gB, sub = build_eval(tr, L, cfg, args, config)
        raw_path = args.eval_labels
    rec["build"] = brec
    rec["build_s"] = round(time.time() - t_b, 1)
    log(f"dataset built: {len(ds.episodes)} clips, {len(ds)} windows in {rec['build_s']}s")
    rec["raw_labels"] = {"path": raw_path, "md5": md5_file(raw_path), "n": len(raw_by_sid)}
    t_c = time.time()
    cols, clips = census(ds, tr, args, raw_by_sid, n_frames_of, gA, gB, rec)
    rec["census_s"] = round(time.time() - t_c, 1)
    log(f"census done {rec['census_s']}s")
    if not a.no_c2:
        t_x = time.time()
        rec["C2_direct_equals_getitem"] = control_c2(ds, cols)
        rec["C2_s"] = round(time.time() - t_x, 1)
        log("C2", json.dumps({k: v for k, v in rec["C2_direct_equals_getitem"].items() if k != "mismatches"}),
            "mismatches:", rec["C2_direct_equals_getitem"]["mismatches"][:5])
    pc, pc_arr = pose_controls(ds, raw_by_sid)
    rec["pose_controls"] = pc
    log("pose_controls", json.dumps(pc))
    np.savez_compressed(out / f"pose_controls_{tag}.npz", **pc_arr)
    np.savez_compressed(out / f"windows_{tag}.npz", **cols)
    json.dump(clips, open(out / f"clips_{tag}.json", "w", encoding="utf-8"), indent=0, default=str)
    if sub is not None:
        json.dump({"seed": a.seed, "n_clips": len(clips), "sha12": [c["sha12"] for c in clips]},
                  open(out / f"sample_clips_{tag}.json", "w"), indent=0)
    rec["n_windows"] = int(len(ds))
    rec["n_clips"] = int(len(ds.episodes))
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["wall_s"] = round(time.time() - T_PROC0, 1)
    json.dump(rec, open(out / f"record_{tag}.json", "w", encoding="utf-8"), indent=1, default=str)
    log("wrote", out, "wall", rec["wall_s"])
    if not a.no_c2 and not rec["C2_direct_equals_getitem"]["pass"]:
        log("C2 FAILED -> exit 3")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
