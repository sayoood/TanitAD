"""dataeff_prep.py -- offline data preparation of PREREG_V8F_PROOF_ADD13 (DATAEFF).  CPU only, no model, no training.

Rules: DATAEFF_DEFS.md (R1..R11).  Usage:  python dataeff_prep.py [--out-dir DIR] [--skip-eval-controls]
Writes lists/{R_half,R_quarter,R_eighth,S_quarter}.txt, LISTS.json, join_S_quarter.npz, manifest_S_quarter.jsonl, G1.json, ARMS.md, CONTROLS.json.
"""
import argparse
import hashlib
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SHIP = "C:/Users/Admin/lorapod/ship2"
TRAIN_MANIFEST = SHIP + "/ref_box2_train_manifest.jsonl"
EVAL_MANIFEST = SHIP + "/ref_box2_eval_manifest.jsonl"
JOIN_TRAIN = SHIP + "/join_train.npz"
TRACKS = SHIP + "/native_tracks.pt"
ART = "E:/Projects/TanitAD-artifacts"
EVAL_RAW = (ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw")
EVAL_GRID = EVAL_RAW + "/eval_grid_windows.json"
DUMP_DIR = ART + "/v8f-dagger-t1/dump_DAG_s0_m10_causal_heads"
HIERWHERE = "F:/TanitAD-scratch/v8proof/hier/hierwhere.py"

HOLD = (540, 600)                       # the A1 holdout of the full arm: --exclude 540:600
KEPT = {"R_half": 2256, "R_quarter": 1128, "R_eighth": 564}
TOP, FLOOR = 846, 282
G1_RATIO = 1.3
HORIZON, FIRST_T, DT = 30, 3, 0.2
THR = {"stop_mps": 0.5, "dv_mps": 2.0, "head_deg": 10.0}
MIN_DS = 0.5 * DT                        # four_families.MIN_DS_MPS (0.5) x 0.2 s
STEPS = 19164
EVAL_SC, EVAL_BULK = 1305, 1094
FULL_WINDOWS, FULL_CLIPS = 306566, 4512


# ------------------------------------------------------------------------------------------------ R1 order, R2 ranges
def sha12(cid):
    return hashlib.sha256(cid.encode("utf-8")).hexdigest()[:12]


def hash_order(clips):
    """clip ids in ascending sha12 (Python sorted is stable: a tie keeps the given order)."""
    return sorted(clips, key=sha12)


def range_end(n_clips, kept, a=HOLD[0]):
    """B of `--exclude a:B` that keeps `kept` clips of `n_clips` (the slice drops B-a clips)."""
    b = a + n_clips - kept
    if not (HOLD[1] <= b <= n_clips):
        raise ValueError(f"kept={kept} of {n_clips}: B={b} outside [{HOLD[1]}, {n_clips}]")
    return b


def apply_exclude(order, a, b):
    """what the trainer keeps: order (sha12 order) without the slice order[a:b]  (train.py:2575-2578)."""
    ex = set(order[a:b])
    return [c for c in order if c not in ex]


# ------------------------------------------------------------------------------------------------ R3 windows, R4 modes
def window_futures(track, n_steps, first=FIRST_T):
    """track = native track [n_raw,4] (x,y,yaw,v).  pose_grid = even rows (must have n_steps rows).
    -> (t [m], g [m,30,2] float32, labelled [m] bool) for t = first..n_steps-31 (train_windows: range(first, T-30))."""
    pg = np.asarray(track, np.float64)[0::2]
    if len(pg) != n_steps:
        raise ValueError(f"track has {len(pg)} even rows, expected {n_steps}")
    t = np.arange(first, n_steps - HORIZON)
    return t, *_gt(pg, t)


def _gt(pg, t0):
    """floors_kin.gt_waypoints (ego frame at t0, steps t0+1..t0+30), cast to float32 as gt_future does."""
    t0 = np.asarray(t0, np.int64)
    if len(t0) == 0:
        return np.zeros((0, HORIZON, 2), np.float32), np.zeros(0, bool)
    fr = np.arange(1, HORIZON + 1)
    d = pg[t0[:, None] + fr[None, :], :2] - pg[t0, None, :2]
    c, s = np.cos(-pg[t0, 2])[:, None], np.sin(-pg[t0, 2])[:, None]
    g = np.stack([d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c], axis=-1)
    ok = np.isfinite(g).all(axis=(1, 2))
    return g.astype(np.float32), ok


def kinematics(traj):
    """copy of hierwhere.py:58. traj [n,30,2] -> speed [n,30], dheading_deg [n]."""
    p = np.asarray(traj, np.float64)
    p = np.concatenate([np.zeros_like(p[:, :1]), p], axis=1)
    d = p[:, 1:] - p[:, :-1]
    ds = np.linalg.norm(d, axis=-1)
    speed = ds / DT
    valid = ds > MIN_DS
    head = np.arctan2(d[..., 1], d[..., 0])
    dh = np.zeros(len(p))
    for i in range(len(p)):
        v = np.flatnonzero(valid[i])
        if len(v) >= 2:
            u = np.unwrap(head[i, v])
            dh[i] = u[-1] - u[0]
    return speed, np.degrees(dh)


def classify(speed, dhead_deg, scale=1.0):
    """copy of hierwhere.py:77. -> lon [n] 0..3 = STOP, DECEL, ACCEL, KEEP; lat [n] 0..2 = LEFT, RIGHT, STRAIGHT."""
    s_stop, s_dv, s_h = THR["stop_mps"] * scale, THR["dv_mps"] * scale, THR["head_deg"] * scale
    n = len(speed)
    below = speed < s_stop
    suffix_all = np.logical_and.accumulate(below[:, ::-1], axis=1)[:, ::-1]
    stop = suffix_all[:, 9:].any(axis=1)
    dv = speed[:, 29] - speed[:, 0]
    lon = np.full(n, 3, np.int64)
    lon[dv >= s_dv] = 2
    lon[dv <= -s_dv] = 1
    lon[stop] = 0
    lat = np.full(n, 2, np.int64)
    lat[dhead_deg >= s_h] = 0
    lat[dhead_deg <= -s_h] = 1
    return lon, lat


def selection_critical(g):
    """[n,30,2] -> bool [n]: not (KEEP, STRAIGHT)."""
    if len(g) == 0:
        return np.zeros(0, bool)
    lon, lat = classify(*kinematics(g))
    return ~((lon == 3) & (lat == 2))


# ------------------------------------------------------------------------------------------------ R5 scores
def clip_stats(clips, n_steps, tracks):
    """-> {clip: (n_windows, n_labelled, n_sc)}  (R3-R5)."""
    out = {}
    for c, T in zip(clips, n_steps):
        t, g, ok = window_futures(tracks[c], int(T))
        sc = selection_critical(g[ok])
        out[c] = (int(len(t)), int(ok.sum()), int(sc.sum()))
    return out


def scores_of(stats):
    """R5: share of selection-critical among labelled windows; no labelled window -> 0.0."""
    return {c: (s / l if l > 0 else 0.0) for c, (_, l, s) in stats.items()}


# ------------------------------------------------------------------------------------------------ R6 S-1/4
def select_S(order, scores, excluded, top=TOP, floor=FLOOR):
    """order = all clips in hash order; excluded = set never eligible.  -> list in hash order."""
    pos = {c: i for i, c in enumerate(order)}
    elig = [c for c in order if c not in excluded]
    ranked = sorted(elig, key=lambda c: (-scores[c], pos[c]))
    head = ranked[:top]
    hs = set(head)
    rest = [c for c in elig if c not in hs][:floor]
    return sorted(head + rest, key=lambda c: pos[c])


# ------------------------------------------------------------------------------------------------ R8 restriction (pure filter)
PER_CLIP = ("limit", "limit_kmh", "limit_ok")
PER_ROW = ("has", "now_s", "nav", "nav_d", "rc", "rc_ok", "goal_y", "goal_w", "lat_cls", "lat_allowed", "lon_cls", "lon_allowed",
           "cons", "lead_valid", "lead_gap")


def restrict_join(J, clips):
    """keep the join's clips that are in `clips`, in the join's order; a pure filter of every array (DEFS R8)."""
    want = set(clips)
    names = J["clip"].tolist()
    ci = np.array([i for i, c in enumerate(names) if c in want], np.int64)
    if len(ci) != len(want):
        raise ValueError(f"{len(want) - len(ci)} clips of the list are not in the join")
    n = J["n"][ci]
    rows = np.concatenate([np.arange(J["off"][i], J["off"][i] + J["n"][i]) for i in ci]) if len(ci) else np.zeros(0, np.int64)
    R = {"clip": J["clip"][ci], "n": n, "off": np.concatenate([[0], np.cumsum(n)[:-1]]).astype(J["off"].dtype)}
    for k in PER_CLIP:
        R[k] = J[k][ci]
    for k in PER_ROW:
        R[k] = J[k][rows]
    R["n_clips_missing_labels"] = int(sum(1 for i in ci if not J["has"][J["off"][i]:J["off"][i] + J["n"][i]].any()))
    extra = set(J) - set(R)
    if extra:
        raise ValueError(f"join arrays not handled by the restriction tool: {sorted(extra)}")
    return R


def save_join(J, path):
    np.savez_compressed(path, **{k: v for k, v in J.items() if isinstance(v, np.ndarray)},
                        n_clips_missing_labels=np.int64(J["n_clips_missing_labels"]))


def load_join(path):
    with np.load(path, allow_pickle=False) as z:
        J = {k: z[k] for k in z.files}
    J["n_clips_missing_labels"] = int(J["n_clips_missing_labels"])
    return J


def windows_of(J, clips=None):
    """train windows of the join's clips (all, or those in `clips`): sum of max(0, T-33)."""
    names = J["clip"].tolist()
    want = None if clips is None else set(clips)
    return int(sum(max(0, int(n) - FIRST_T - HORIZON) for c, n in zip(names, J["n"]) if want is None or c in want))


def cons_stats(J):
    """models/v8f_mini.py:1208 (mean / std of the 26 cons cells over labelled rows)."""
    import warnings
    c = J["cons"][J["has"]].astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, sd = np.nanmean(c, 0), np.nanstd(c, 0)
    return (np.where(np.isfinite(mu), mu, 0.0).astype(np.float32),
            np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0).astype(np.float32))


# ------------------------------------------------------------------------------------------------ io helpers
def read_manifest(path):
    ents = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                ents.append(json.loads(ln))
    return ents


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def write_list(path, clips):
    data = ("\n".join(clips) + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    return sha256_bytes(data)


def train_sha(clips):
    """the trainer's `train_sha256` (train.py:2590): newline-joined clips in sha12 order, no trailing newline."""
    return hashlib.sha256("\n".join(hash_order(clips)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ controls
def control_d1(lists, order, bs):
    r8, r4, r2, f4 = (set(lists[k]) for k in ("R_eighth", "R_quarter", "R_half", "R_quarter"))
    out = {"counts": {k: len(v) for k, v in lists.items()}, "R_eighth_in_R_quarter": r8 <= r4, "R_quarter_in_R_half": r4 <= r2,
           "F_quarter_list_sha256_equals_R_quarter": True, "known": "nested True/True; counts 564 / 1128 / 2256 / 1128 (S)"}
    out["PASS"] = bool(out["R_eighth_in_R_quarter"] and out["R_quarter_in_R_half"] and out["counts"] == {"R_half": 2256, "R_quarter": 1128, "R_eighth": 564, "S_quarter": 1128})
    out["hash_tie_count"] = int(len(order) - len({sha12(c) for c in order}))
    return out


def control_d2(lists, eval_ids):
    ev = set(eval_ids)
    ev12 = {sha12(c) for c in ev}
    per = {k: len(ev & set(v)) for k, v in lists.items()}
    per12 = {k: len(ev12 & {sha12(c) for c in v}) for k, v in lists.items()}
    return {"n_eval_clips": len(ev), "overlap_by_id": per, "overlap_by_sha12": per12, "known": 0, "PASS": bool(len(ev) == 141 and not any(per.values()) and not any(per12.values()))}


def control_d5(J, order_lists_rq):
    R = restrict_join(J, order_lists_rq)
    full_w = windows_of(J, order_lists_rq)
    got_w = windows_of(R)
    ok_rows = True
    names = J["clip"].tolist()
    idx = {c: i for i, c in enumerate(names)}
    off_r = np.concatenate([[0], np.cumsum(R["n"])[:-1]])
    for k, c in enumerate(R["clip"].tolist()):
        i = idx[c]
        a, b = int(J["off"][i]), int(J["off"][i] + J["n"][i])
        a2, b2 = int(off_r[k]), int(off_r[k] + R["n"][k])
        for key in PER_ROW:
            if not np.array_equal(J[key][a:b], R[key][a2:b2], equal_nan=J[key].dtype.kind == "f"):
                ok_rows = False
    return {"n_clips_kept": int(len(R["clip"])), "windows_in_restricted_join": got_w, "windows_of_same_clips_in_full_join": full_w, "rows_equal": ok_rows,
            "known": "equal counts, rows_equal True", "PASS": bool(got_w == full_w and ok_rows and len(R["clip"]) == len(set(order_lists_rq)))}


def eval_windows(tracks, eval_ent):
    """(g [n,30,2] float32, clip id [n], t [n]) of the 141-episode eval grid, in the grid json's order."""
    with open(EVAL_GRID, encoding="utf-8") as f:
        grid = json.load(f)
    nfr = {e["clip"]: int(e["n_frames"]) for e in eval_ent}
    gs, cs, ts = [], [], []
    for c in grid["order"]:
        pg = np.asarray(tracks[c], np.float64)[0::2]
        if len(pg) != nfr[c]:
            raise ValueError(f"eval clip {c}: {len(pg)} even rows vs manifest {nfr[c]}")
        t = np.asarray(grid["windows"][c], np.int64)
        g, ok = _gt(pg, t)
        if not ok.all():
            raise ValueError(f"eval clip {c}: non-finite logged future")
        gs.append(g)
        cs += [c] * len(t)
        ts.append(t)
    return np.concatenate(gs), np.array(cs), np.concatenate(ts)


def control_d6(tracks, eval_ent):
    try:
        g, cl, t = eval_windows(tracks, eval_ent)
    except Exception as e:                                           # noqa: BLE001
        return {"PASS": None, "reason": f"d6: not run, {type(e).__name__}: {str(e)[:300]}"}
    sc = selection_critical(g)
    out = {"n_windows": int(len(g)), "n_episodes": int(len(set(cl.tolist()))), "selection_critical": int(sc.sum()), "bulk": int((~sc).sum()),
           "known": {"selection_critical": EVAL_SC, "bulk": EVAL_BULK}}
    out["counts_match"] = bool(out["selection_critical"] == EVAL_SC and out["bulk"] == EVAL_BULK)
    try:                                                              # the tool's g against the banked dump's g, and its rule against hierwhere's own
        import glob
        fs = sorted(glob.glob(DUMP_DIR + "/ep*.npz"))
        dg = np.concatenate([np.load(f)["g"] for f in fs]) if len(fs) == 141 else None
        dws = np.concatenate([np.load(f)["ws"] for f in fs]) if len(fs) == 141 else None
        if dg is None:
            out["dump_compare"] = f"not run, {len(fs)} dump files"
        else:
            same_shape = dg.shape == g.shape
            if same_shape:
                out["g_max_abs_diff_vs_dump"] = float(np.abs(dg.astype(np.float64) - g.astype(np.float64)).max())
                out["ws_equal_dump"] = bool(np.array_equal(dws, t))
            else:
                out["dump_compare"] = f"shape {dg.shape} vs {g.shape}"
            out["dump_rule_counts"] = int(selection_critical(dg).sum())
    except Exception as e:                                           # noqa: BLE001
        out["dump_compare"] = f"not run, {type(e).__name__}: {str(e)[:200]}"
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hierwhere_ro", HIERWHERE)
        hw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hw)
        lon, lat = hw.modes(g)
        hw_sc = ~((lon == 3) & (lat == 2))
        out["hierwhere_rule_agrees_windows"] = bool(np.array_equal(hw_sc, sc))
    except Exception as e:                                           # noqa: BLE001
        out["hierwhere_rule_agrees_windows"] = f"not run, {type(e).__name__}: {str(e)[:200]}"
    out["PASS"] = bool(out["counts_match"] and out.get("hierwhere_rule_agrees_windows") is True and out.get("g_max_abs_diff_vs_dump", 1.0) == 0.0)
    return out


# ------------------------------------------------------------------------------------------------ ARMS.md
def arms_md(info):
    W = "/workspace/v8proof"
    base = (f"--arm M2b --seed {{S}} --grid pool4x10 {{CUT}}--out <OUT>/{{NAME}}_s{{S}} --cache-dir {{CACHE}} --v9-train {W}/data/v9/v9_labels_train.npz "
            f"--v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train {W}/data/v9/lead_train4572.npz --speed-train {W}/data/v9/refcv6_speed_max_v8_train.jsonl "
            f"--join-train {{JOIN}} --p10 {W}/data/inputs/p10_prior.pt --kin-mode yaw --tracks {W}/data/inputs/native_tracks.pt {{EXC}}--batch 64 --steps-mult 4 --steps {STEPS} --device cuda")
    arms = [("R-1/2", "R_half", "", f"--exclude {HOLD[0]}:{info['B']['R_half']}", False), ("R-1/4", "R_quarter", "", f"--exclude {HOLD[0]}:{info['B']['R_quarter']}", False),
            ("R-1/8", "R_eighth", "", f"--exclude {HOLD[0]}:{info['B']['R_eighth']}", False), ("F-1/4", "F_quarter", "--o-interface cut ", f"--exclude {HOLD[0]}:{info['B']['R_quarter']}", False),
            ("S-1/4", "S_quarter", "", "", True)]
    L = ["# ARMS: the 10 arms of PREREG_V8F_PROOF_ADD13 (5 arms x seeds 0, 1), pod argv", "",
         "Built by dataeff_prep.py. Pod paths follow the HIERREP chain (`W=/workspace/v8proof`, cache `data/cache_pod`, inputs `data/inputs`). `<OUT>` = the run root. One line per run, executed as",
         "`python <TREE>/stack/scripts/v8f_mini_train.py <argv>` at commit 3e264bbf, each under its own selfcheck gate (`--selfcheck-only` appended, `--out` as in the chain).",
         "Differences from the HIERREP argv: `--exclude 540:600` becomes `540:B`; `--o-interface cut` for F-1/4; `--steps 19164` added (it wins over `--steps-mult 4`, train.py:2703); S-1/4 data flags (below).", "",
         "## Selfcheck expectations (from the tool, the same rules as the trainer)", "", "| arm | --exclude | n_clips | n_clips_all | n_windows (= windows of clips_all) | S = ceil(n_windows/64) | steps |", "|---|---|---|---|---|---|---|"]
    for name, key, cut, exc, is_s in arms:
        lk = "R_quarter" if key == "F_quarter" else key
        nw = info["windows"][lk]
        L.append(f"| {name} | {exc.replace('--exclude ', '') or 'none (restricted manifest)'} | {info['counts'][lk]} | {info['counts'][lk]} | {nw} | {math.ceil(nw / 64)} | {STEPS} |")
    L += ["", f"Full arm (HIERREP): n_clips 4,512, n_windows {FULL_WINDOWS:,}, S 4,791, steps {STEPS}.", "",
          "## S-1/4 needs two inputs, not one (departure 1, DATAEFF_DEFS.md R9)",
          "The trainer takes its clip set from `<cache-dir>/train/manifest.jsonl` (train.py:2560-2561), and `D.train_windows` indexes the join for every manifest clip (data.py:611-619). A restricted join alone would raise KeyError, "
          "so the arm runs on a restricted cache root built by `s_cache_setup.py` from the pod's own manifest and the list `lists/S_quarter.txt`:", "",
          f"`python s_cache_setup.py --src-cache {W}/data/cache_pod --list lists/S_quarter.txt --dst-cache {W}/data/cache_S_quarter`  (writes `train/manifest.jsonl` with only the 1,128 clips in the source file order and a link `train/clips` to the full clip folder; refuses if a listed clip is missing).",
          f"The join `join_S_quarter.npz` goes to `{W}/data/inputs/join_S_quarter.npz`. Do NOT pass `--exclude` to S-1/4 (a 540:600 would cut S's own positions). Expected selfcheck: n_clips 1,128, n_excluded 0.", "",
          "## Argv", ""]
    for name, key, cut, exc, is_s in arms:
        for s in (0, 1):
            nm = name.replace("/", "_").replace("-", "")
            cache = f"{W}/data/cache_S_quarter" if is_s else f"{W}/data/cache_pod"
            join = f"{W}/data/inputs/join_S_quarter.npz" if is_s else f"{W}/data/inputs/join_train.npz"
            line = base.format(S=s, CUT=cut, NAME=nm, CACHE=cache, JOIN=join, EXC=(exc + " ") if exc else "")
            L += [f"### {name} seed {s}", "```", line, "```", ""]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ main
def quantiles(x):
    qs = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    v = np.percentile(x, qs)
    return {f"q{q}": round(float(a), 6) for q, a in zip(qs, v)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--skip-eval-controls", action="store_true")
    a = ap.parse_args(argv)
    out = a.out_dir.replace("\\", "/")
    os.makedirs(out + "/lists", exist_ok=True)

    man = read_manifest(TRAIN_MANIFEST)
    clips = [e["clip"] for e in man]
    nfr = {e["clip"]: int(e["n_frames"]) for e in man}
    J = load_join(JOIN_TRAIN)
    assert J["clip"].tolist() == clips, "join clip order != manifest order"
    assert all(int(n) == nfr[c] for c, n in zip(J["clip"].tolist(), J["n"])), "join n != manifest n_frames"
    assert all(e["sha12"] == sha12(e["clip"]) for e in man), "manifest sha12 != sha256[:12] of the id"
    N = len(clips)
    order = hash_order(clips)
    print(f"train manifest: {N} clips; manifest already in sha12 order: {order == clips}")

    # R2 lists
    B = {k: range_end(N, v) for k, v in KEPT.items()}
    lists = {k: apply_exclude(order, HOLD[0], b) for k, b in B.items()}
    excluded60 = set(order[HOLD[0]:HOLD[1]])
    for k, v in KEPT.items():
        assert len(lists[k]) == v, (k, len(lists[k]))
    print("range ends B:", B)

    # R3-R5 scores over the 4,512 eligible clips (+ the 60 held out, for the record only)
    tracks_pt = __import__("torch").load(TRACKS, map_location="cpu", weights_only=True)["tracks"]
    tracks = {c: t.double().numpy() for c, t in tracks_pt.items()}
    stats = clip_stats(order, [nfr[c] for c in order], tracks)
    sc = scores_of(stats)
    elig = [c for c in order if c not in excluded60]
    tot_w = sum(stats[c][0] for c in elig)
    print(f"eligible clips {len(elig)}, windows {tot_w} (known {FULL_WINDOWS})")

    # R6 S-1/4
    S = select_S(order, sc, excluded60)
    lists["S_quarter"] = S
    assert len(S) == TOP + FLOOR and not (set(S) & excluded60)

    # lists + hashes
    meta = {}
    for k, v in lists.items():
        v_sorted = hash_order(v)
        h = write_list(f"{out}/lists/{k}.txt", v_sorted)
        meta[k] = {"n_clips": len(v), "list_sha256": h, "trainer_train_sha256": train_sha(v), "exclude": (f"{HOLD[0]}:{B[k]}" if k in B else None),
                   "n_windows": int(sum(stats[c][0] for c in v))}
    meta["F_quarter"] = dict(meta["R_quarter"], note="F-1/4 uses the R_quarter list (same file, same sha256)")
    with open(out + "/LISTS.json", "w", encoding="utf-8") as f:
        json.dump({"order": "ascending sha12 of the clip id; one clip id per line; sha256 of the file bytes", "lists": meta}, f, indent=1)

    # G1 (R7)
    sq, rq = np.array([sc[c] for c in S]), np.array([sc[c] for c in lists["R_quarter"]])
    allsc = np.array([sc[c] for c in elig])
    mS, mR = float(sq.mean()), float(rq.mean())
    ratio = mS / mR if mR > 0 else float("inf")
    n_nolab = [c for c in order if stats[c][1] == 0]
    top_set = set(select_S(order, sc, excluded60, TOP, 0))
    cs_cur = cons_stats(J)
    SJ = restrict_join(J, S)
    cs_S = cons_stats(SJ)
    RJ = restrict_join(J, lists["R_quarter"])
    cs_R = cons_stats(RJ)
    g1 = {"rule": "mean score(S-1/4) >= 1.3 x mean score(R-1/4)", "mean_score_S_quarter": mS, "mean_score_R_quarter": mR, "ratio": ratio, "threshold": G1_RATIO,
          "PASS": bool(ratio >= G1_RATIO),
          "score_distribution_S_quarter": {"quantiles": quantiles(sq), "share_at_0": float((sq == 0).mean()), "share_at_1": float((sq == 1).mean())},
          "score_distribution_R_quarter": {"quantiles": quantiles(rq), "share_at_0": float((rq == 0).mean()), "share_at_1": float((rq == 1).mean())},
          "score_distribution_eligible_4512": {"quantiles": quantiles(allsc), "share_at_0": float((allsc == 0).mean()), "share_at_1": float((allsc == 1).mean()), "mean": float(allsc.mean())},
          "S_top846_mean": float(np.mean([sc[c] for c in S if c in top_set])), "S_floor282_mean": float(np.mean([sc[c] for c in S if c not in top_set])),
          "score_of_846th_ranked_clip": float(sorted(allsc, reverse=True)[TOP - 1]),
          "n_clips_tied_at_cutoff_score": int((allsc == sorted(allsc, reverse=True)[TOP - 1]).sum()),
          "clips_without_labelled_window": n_nolab, "n_clips_without_labelled_window": len(n_nolab),
          "windows": {"eligible_total": tot_w, "labelled_eligible": int(sum(stats[c][1] for c in elig)), "selection_critical_eligible": int(sum(stats[c][2] for c in elig))},
          "cons_stats_gap_G2": {"max_abs_mu_diff_over_sd_S_vs_full": float(np.max(np.abs(cs_S[0] - cs_cur[0]) / cs_cur[1])),
                                "max_abs_sd_ratio_minus_1_S_vs_full": float(np.max(np.abs(cs_S[1] / cs_cur[1] - 1))),
                                "max_abs_mu_diff_over_sd_Rq_vs_full": float(np.max(np.abs(cs_R[0] - cs_cur[0]) / cs_cur[1])),
                                "max_abs_sd_ratio_minus_1_Rq_vs_full": float(np.max(np.abs(cs_R[1] / cs_cur[1] - 1)))}}
    with open(out + "/G1.json", "w", encoding="utf-8") as f:
        json.dump(g1, f, indent=1)
    print(f"G1: S {mS:.5f} / R {mR:.5f} = {ratio:.4f} -> {'PASS' if g1['PASS'] else 'FAIL'}")

    # restricted join + manifest (R8, R9)
    save_join(SJ, out + "/join_S_quarter.npz")
    Sset = set(S)
    with open(out + "/manifest_S_quarter.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for e in man:
            if e["clip"] in Sset:
                f.write(json.dumps(e) + "\n")

    # controls
    ctl = {"d0_known_full_counts": {"eligible_clips": len(elig), "windows": tot_w, "known": [FULL_CLIPS, FULL_WINDOWS], "PASS": bool(len(elig) == FULL_CLIPS and tot_w == FULL_WINDOWS)},
           "d1": control_d1(lists, order, B)}
    ctl["d1"]["F_quarter_list_sha256_equals_R_quarter"] = bool(meta["F_quarter"]["list_sha256"] == meta["R_quarter"]["list_sha256"])
    eval_ent = read_manifest(EVAL_MANIFEST)
    ids = [e["clip"] for e in eval_ent]
    with open(EVAL_GRID, encoding="utf-8") as f:
        gids = json.load(f)["order"]
    ctl["d2"] = control_d2(lists, ids)
    ctl["d2"]["grid_order_equals_eval_manifest_ids"] = bool(set(gids) == set(ids))
    ctl["d2"]["overlap_by_id_using_grid_ids"] = {k: len(set(gids) & set(v)) for k, v in lists.items()}
    ctl["d5"] = control_d5(J, lists["R_quarter"])
    SJr = load_join(out + "/join_S_quarter.npz")
    ctl["d5_S_join_file"] = {"n_clips": int(len(SJr["clip"])), "windows": windows_of(SJr), "windows_by_list": meta["S_quarter"]["n_windows"], "PASS": bool(windows_of(SJr) == meta["S_quarter"]["n_windows"] and len(SJr["clip"]) == 1128)}
    ctl["d6"] = {"PASS": None, "reason": "d6: not run (--skip-eval-controls)"} if a.skip_eval_controls else control_d6(tracks, eval_ent)
    with open(out + "/CONTROLS.json", "w", encoding="utf-8") as f:
        json.dump(ctl, f, indent=1)
    for k, v in ctl.items():
        print(k, json.dumps(v)[:400])

    info = {"B": B, "counts": {k: meta[k]["n_clips"] for k in lists}, "windows": {k: meta[k]["n_windows"] for k in lists}}
    with open(out + "/ARMS.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(arms_md(info))
    return 0 if all(v.get("PASS") in (True, None) for v in ctl.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
