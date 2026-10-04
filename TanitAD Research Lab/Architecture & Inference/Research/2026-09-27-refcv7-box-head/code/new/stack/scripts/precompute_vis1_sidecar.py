#!/usr/bin/env python3
"""precompute_vis1_sidecar.py -- SPEC_REFCV7 §14 (A9) R3: the VIS-1 visibility sidecar, computed ONCE.

For every clip of the refcv6/refcv7 corpus (TRAIN 4,369 + EVAL 139) and every NOW frame ``f`` the 2-D agent
join labels (``frame_idx >= 0``), this runs the box-head audit's exact per-pixel ray/cuboid z-buffer
(``tanitad.data.vis1_zbuffer``, vendored verbatim) on the trainer's OWN target block for that frame and stores
``(n_full, n_img, n_vis, vis_rows)`` per row inside :data:`tanitad.data.vis1.STORE_SCOPE`, keyed by
``(clip sha12, f, row)`` with the row's ``track_id`` and float32 centre beside it for verification.

⛔ THE TARGET BLOCK IS THE TRAINER'S, NOT A RE-DERIVATION. Each frame's rows come from
``refc_v3_train.V3Dataset._agent_item(ep, f)`` called on a bare dataset object holding the SAME readers the
trainer attaches (``JoinFileReader`` on ``--agent-join``, ``AgentJoin3D`` on ``--join3d``, the manifest's
``n_stack``) -- so the float32 boxes, the 3-D frame offset (``f + n_stack - 1``) and the row order are the ones
the loss will see, by construction.

⛔ OCCLUDERS = every valid row of the frame with a 3-D label (targets AND rows the trainer's filter removes),
exactly as the audit (``vis_zbuf.py:324-327``).

Modes:
  --mode control   recompute the audit's BANKED windows (raw/visibility/vis_{train,clipgrid,inrun,gtval}_boxes
                   .json) and require EXACT equality of n_full / n_img / n_vis / vis_rows, the class and the
                   range, row by row. This is the control the build must pass before the sidecar is trusted.
  --mode full      the sidecar itself: every clip of --train-cache and --eval-cache.

Parallelism: the two joins are split ONCE by clip into shard files (a streaming pass, no agent JSON parsed),
then a forked pool processes one shard per task (each worker loads only its shard). ``--nice`` (default 19) is
applied before the pool starts; thread pools are pinned to 1 per worker.

sha12 only in every output; no raw clip id is written or printed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import math
import os
import re
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve()
TREE = HERE.parents[2]                      # <tree>/stack/scripts/this.py -> <tree>
STACK = TREE / "stack"
SCRIPTS = STACK / "scripts"
for _p in (str(STACK), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

CLIP_RE = re.compile(r'^\{\s*"clip_id"\s*:\s*"([^"]+)"')
CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer", "person", "rider", "stroller",
           "animal", "protruding_object")                        # vis_zbuf.py:41-42, the audit's own list


def sha12(s) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def md5_file(p, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[vis1-pre {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------------------- #
# the trainer, by path (it is a script)                                                                      #
# --------------------------------------------------------------------------------------------------------- #
_TR = None


def trainer():
    global _TR
    if _TR is None:
        import tanitad
        want = os.path.normcase(str(STACK))
        got = os.path.normcase(os.path.abspath(tanitad.__file__))
        if not got.startswith(want):
            raise SystemExit(f"[vis1-pre] tanitad imported from {got}, not from this tree {want}")
        name = "refc_v3_train_for_vis1"
        spec = importlib.util.spec_from_file_location(name, str(SCRIPTS / "refc_v3_train.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _TR = mod
    return _TR


# --------------------------------------------------------------------------------------------------------- #
# shards                                                                                                     #
# --------------------------------------------------------------------------------------------------------- #
def split_join(src: str, out_dir: Path, shard_of: dict, n_shards: int, tag: str) -> dict:
    """Stream ``src`` (.jsonl / .jsonl.xz) once; write each wanted clip's lines to its shard file.

    Lines are routed by the clip id read with a regex (no agent JSON parsed); a clip's lines stay contiguous
    and in file order, which is all the readers need."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    fhs = [open(out_dir / f"{tag}_{k:03d}.jsonl", "w", encoding="utf-8", newline="\n") for k in range(n_shards)]
    opener = lzma.open if src.endswith(".xz") else open
    n_in = n_kept = 0
    seen = set()
    with opener(src, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n_in += 1
            m = CLIP_RE.match(line)
            if m is None:
                raise SystemExit(f"[vis1-pre] {tag}: line {n_in} does not start with a clip_id field")
            k = shard_of.get(m.group(1))
            if k is None:
                continue
            fhs[k].write(line if line.endswith("\n") else line + "\n")
            n_kept += 1
            seen.add(m.group(1))
    for f in fhs:
        f.close()
    log(f"split {tag}: {n_in} lines read, {n_kept} kept, {len(seen)} clips, {time.time() - t0:.0f} s")
    return {"lines_read": n_in, "lines_kept": n_kept, "clips_seen": len(seen), "wall_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------------------------------------- #
# per-frame work (the SAME function for control and full)                                                   #
# --------------------------------------------------------------------------------------------------------- #
class ShardCtx:
    """The readers for one shard, attached to a bare ``V3Dataset`` the way the trainer attaches them."""

    def __init__(self, path2d: str, path3d: str, clip_of_uid: dict, n_stack: int):
        tr = trainer()
        from train_p8_occupancy import JoinFileReader
        from tanitad.data import agent_cuboid_gt as ACG
        uids = set(clip_of_uid)
        self.reader = JoinFileReader(path2d, episode_ids=uids, with_rates=False, with_track_ids=True)
        self.j3 = ACG.AgentJoin3D.open(path3d, clips=set(clip_of_uid.values()))
        ds = tr.V3Dataset.__new__(tr.V3Dataset)          # class defaults, no __init__ (no frames touched)
        ds.agent_join = self.reader
        ds.agent_pad = int(self.reader.max_agents_per_frame)
        ds.join3d = self.j3
        ds.map_clip_of_ep = dict(clip_of_uid)
        ds.map_n_stack = int(n_stack)
        self.ds = ds
        self.item_fn = tr.V3Dataset._agent_item
        self.frames_by_clip: dict = {}
        for (cid, fi) in self.reader._by_clip.keys():
            self.frames_by_clip.setdefault(cid, []).append(int(fi))
        for v in self.frames_by_clip.values():
            v.sort()

    def rows(self, uid: int, f: int):
        """The trainer's valid target rows at (episode uid, NOW frame f), plus the join's track ids."""
        import types
        item = self.item_fn(self.ds, types.SimpleNamespace(episode_id=int(uid)), int(f))
        if not bool(item["agent_label"]):
            return None
        v = item["agent_valid"].bool()
        n = int(v.sum())
        tids = self.reader.lookup_track_ids(int(uid), int(f))
        tids = [] if tids is None else [str(x) for x in list(tids)[:n]]
        if int(item["agent_n_truncated"]) != 0:
            raise SystemExit("[vis1-pre] truncation inside a shard reader -- pad < n_raw cannot happen here")
        return {"box": item["agent_box"][v], "yaw": item["agent_yaw"][v], "cls": item["agent_cls"][v],
                "cz": item["agent_cz"][v], "h": item["agent_h"][v], "zh": item["agent_zh_mask"][v].bool(),
                "tids": tids, "n": n}


def frame_zbuffer(rows: dict, cam) -> list:
    """Every valid row WITH a 3-D label, in block order, with the audit's per-box z-buffer numbers.

    ⛔ The float inputs are the float32 tensor values ``.tolist()``-ed, exactly as vis_zbuf.py:236-240 read the
    audit's dumped blocks, and the box array is built by the same expression (vis_zbuf.py:325-326)."""
    from tanitad.data.vis1_zbuffer import zbuffer
    R, t, obs = cam
    gt = []
    for j in range(rows["n"]):
        b = rows["box"][j].tolist()
        gt.append({"j": j, "cx": b[0], "cy": b[1], "l": b[2], "w": b[3], "yaw": float(rows["yaw"][j]),
                   "cz": float(rows["cz"][j]), "h": float(rows["h"][j]), "zh": bool(rows["zh"][j]),
                   "cls_i": int(rows["cls"][j]),
                   "tid": rows["tids"][j] if j < len(rows["tids"]) else ""})
    gt_z = [g for g in gt if g["zh"]]
    bx = np.array([[g["cx"], g["cy"], g["cz"], g["l"], g["w"], g["h"], g["yaw"]] for g in gt_z]) \
        if gt_z else np.zeros((0, 7))
    nf, ni, nv, vr, _ = zbuffer(bx, R, t, obs)
    out = []
    for i, g in enumerate(gt_z):
        out.append({"j": g["j"], "tid": g["tid"], "cx": g["cx"], "cy": g["cy"], "cls_i": g["cls_i"],
                    "n_full": int(nf[i]), "n_img": int(ni[i]), "n_vis": int(nv[i]), "vis_rows": int(vr[i])})
    return out


def camera_for(cid: str, table: dict, calib_dirs):
    from tanitad.data.vis1_zbuffer import clip_camera
    R, t, obs, why, src = clip_camera(cid, table[cid], calib_dirs)
    return (R, t, obs), why, src


# --------------------------------------------------------------------------------------------------------- #
# pool tasks                                                                                                 #
# --------------------------------------------------------------------------------------------------------- #
_P: dict = {}                                    # set in the parent before the fork


def run_shard(k: int):
    """Process one shard's clips (full mode). Returns per-clip compact arrays, in the shard's clip order."""
    from tanitad.data.vis1 import in_store_scope
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    P = _P
    clips = P["shard_clips"][k]
    if not clips:
        return {"k": k, "clips": [], "wall_s": 0.0}
    ctx = ShardCtx(str(P["shard_dir"] / f"j2d_{k:03d}.jsonl"), str(P["shard_dir"] / f"j3d_{k:03d}.jsonl"),
                   {P["uid_of"][c]: c for c in clips}, P["n_stack"])
    t_load = time.time() - t0
    res = []
    for cid in clips:
        cam, why, src = camera_for(cid, P["table"], P["calib_dirs"])
        frames = [f for f in ctx.frames_by_clip.get(cid, []) if f >= 0]
        fr_f, fr_nzh, fr_nv, fr_cnt = [], [], [], []
        R = {k2: [] for k2 in ("row", "track", "n_full", "n_img", "n_vis", "vis_rows", "cx", "cy")}
        for f in frames:
            rows = ctx.rows(P["uid_of"][cid], f)
            if rows is None:
                # ⛔ the reader HOLDS this (clip, frame) -- a None here means the episode id the manifest
                # carries does not resolve to this clip in the join, and every frame would vanish silently
                raise SystemExit(f"[vis1-pre] clip {sha12(cid)} frame {f}: in the join but _agent_item "
                                 f"returned NO_LABEL -- the manifest's episode uid does not resolve")
            zr = frame_zbuffer(rows, cam)
            keep = in_store_scope([r["cx"] for r in zr], [r["cy"] for r in zr]) if zr else np.zeros(0, bool)
            cnt = 0
            for r, kk in zip(zr, keep):
                if not kk:
                    continue
                if not str(r["tid"]).isdigit():
                    raise SystemExit(f"[vis1-pre] clip {sha12(cid)} frame {f}: non-numeric track id")
                R["row"].append(r["j"])
                R["track"].append(int(r["tid"]))
                R["n_full"].append(r["n_full"])
                R["n_img"].append(r["n_img"])
                R["n_vis"].append(r["n_vis"])
                R["vis_rows"].append(r["vis_rows"])
                R["cx"].append(r["cx"])
                R["cy"].append(r["cy"])
                cnt += 1
            fr_f.append(int(f))
            fr_nzh.append(len(zr))
            fr_nv.append(int(rows["n"]))
            fr_cnt.append(cnt)
        res.append({"sha12": sha12(cid), "split": P["split_of"][cid], "mask_ftheta": why.startswith("f-theta"),
                    "frame_f": np.asarray(fr_f, np.int32), "frame_n_zh": np.asarray(fr_nzh, np.int32),
                    "frame_n_valid": np.asarray(fr_nv, np.int32), "frame_cnt": np.asarray(fr_cnt, np.int64),
                    "row": np.asarray(R["row"], np.int16), "track": np.asarray(R["track"], np.int64),
                    "n_full": np.asarray(R["n_full"], np.int32), "n_img": np.asarray(R["n_img"], np.int32),
                    "n_vis": np.asarray(R["n_vis"], np.int32), "vis_rows": np.asarray(R["vis_rows"], np.int32),
                    "cx": np.asarray(R["cx"], np.float32), "cy": np.asarray(R["cy"], np.float32),
                    "intrinsics_file": (None if src is None else Path(src).name)})
    wall = time.time() - t0
    n_fr = sum(len(r["frame_f"]) for r in res)
    return {"k": k, "clips": res, "wall_s": round(wall, 1), "load_s": round(t_load, 1), "n_frames": n_fr}


def run_control_shard(k: int):
    """Control mode: every banked window of the shard's clips, ALL rows with a 3-D label, compared exactly."""
    import torch
    torch.set_num_threads(1)
    P = _P
    clips = P["shard_clips"][k]
    if not clips:
        return {"k": k, "n_windows": 0, "n_rows": 0, "mismatch": [], "n_mismatch": 0}
    ctx = ShardCtx(str(P["shard_dir"] / f"j2d_{k:03d}.jsonl"), str(P["shard_dir"] / f"j3d_{k:03d}.jsonl"),
                   {P["uid_of"][c]: c for c in clips}, P["n_stack"])
    W = P["window"]
    mism, n_w, n_r = [], 0, 0
    for cid in clips:
        s12 = sha12(cid)
        cam, why, _src = camera_for(cid, P["table"], P["calib_dirs"])
        for (set_name, w), banked in P["banked"].get(s12, {}).items():
            n_w += 1
            rows = ctx.rows(P["uid_of"][cid], int(w) + W - 1)
            zr = [] if rows is None else frame_zbuffer(rows, cam)
            if len(zr) != len(banked):
                mism.append({"set": set_name, "sha12": s12, "w": int(w), "why": "row count",
                             "got": len(zr), "banked": len(banked)})
                continue
            for i, (g, b) in enumerate(zip(zr, banked)):
                n_r += 1
                cls = CLASSES[g["cls_i"]] if 0 <= g["cls_i"] < len(CLASSES) else "unknown"
                rng = math.hypot(g["cx"], g["cy"])
                bad = [key for key in ("n_full", "n_img", "n_vis", "vis_rows") if int(g[key]) != int(b[key])]
                if cls != b["cls"]:
                    bad.append("cls")
                if rng != float(b["range"]):
                    bad.append("range")
                vf = (g["n_vis"] / g["n_full"]) if g["n_full"] else float("nan")
                bvf = float(b["vis_frac"]) if b["vis_frac"] is not None else float("nan")
                if not ((vf != vf and bvf != bvf) or vf == bvf):
                    bad.append("vis_frac")
                if bad:
                    mism.append({"set": set_name, "sha12": s12, "w": int(w), "row": i, "fields": bad,
                                 "got": {kk: g[kk] for kk in ("n_full", "n_img", "n_vis", "vis_rows")},
                                 "banked": {kk: b[kk] for kk in ("n_full", "n_img", "n_vis", "vis_rows")}})
            if why != "f-theta observed mask":
                mism.append({"set": set_name, "sha12": s12, "w": int(w), "why": f"mask source {why!r}"})
    return {"k": k, "n_windows": n_w, "n_rows": n_r, "mismatch": mism[:200], "n_mismatch": len(mism)}


# --------------------------------------------------------------------------------------------------------- #
# main                                                                                                       #
# --------------------------------------------------------------------------------------------------------- #
def load_split_clips(cache: str):
    from tanitad.data.v2_dataset import load_or_build_manifest
    man = load_or_build_manifest(cache, verbose=False)
    cids = [str(c) for c in (man.get("clip_id") or [])]
    uids = [int(u) for u in (man.get("episode_uid") or [])]
    ns = sorted({int(x) for x in (man.get("n_stack") or [])})
    if len(cids) != len(uids) or not cids:
        raise SystemExit(f"[vis1-pre] {cache}: manifest has {len(cids)} clip ids / {len(uids)} uids")
    return cids, uids, ns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=("control", "full"), required=True)
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--agent-join", required=True)
    ap.add_argument("--join3d", required=True)
    ap.add_argument("--extrinsics", required=True, help="the per-clip table the trainer's --agent-rig-extrinsics "
                                                         "reads (refcv6_train_eval139_extrinsics.json)")
    ap.add_argument("--calib-dir", action="append", required=True,
                    help="intrinsics roots, searched in order (repeatable)")
    ap.add_argument("--window", type=int, default=8, help="W; the NOW frame of window t is t + W - 1")
    ap.add_argument("--banked-dir", default=None, help="--mode control: the audit's raw/visibility/")
    ap.add_argument("--banked-sets", default="train,clipgrid,inrun,gtval")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", default=None, help="--mode full: the sidecar .npz")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shards", type=int, default=24)
    ap.add_argument("--nice", type=int, default=19)
    ap.add_argument("--max-clips", type=int, default=0, help="smoke: first N clips (sha12 order) per split")
    ap.add_argument("--allow-bounds-only", action="store_true",
                    help="accept clips with no intrinsics file (the audit's image-bounds fallback)")
    a = ap.parse_args(argv)
    t_all = time.time()
    if a.nice:
        os.nice(int(a.nice))
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rec = {"tool": "precompute_vis1_sidecar.py", "mode": a.mode, "argv": sys.argv[1:],
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tree": str(TREE),
           "tool_md5": md5_file(HERE)}
    import tanitad
    rec["tanitad_file"] = tanitad.__file__
    from tanitad.data import vis1 as V
    from tanitad.data import vis1_zbuffer as VZ
    rec["zbuffer"] = {"module_md5": md5_file(VZ.__file__), "audit_source": VZ.AUDIT_SOURCE,
                      "function_sha256": {k: VZ.function_sha256(Path(VZ.__file__).read_text(encoding="utf-8"), k)
                                          for k in VZ.AUDIT_FUNCTION_SHA256}}
    if rec["zbuffer"]["function_sha256"] != VZ.AUDIT_FUNCTION_SHA256:
        raise SystemExit("[vis1-pre] the vendored z-buffer no longer matches the audit's source digests")
    trainer()                                           # import once in the parent; the fork inherits it
    table = json.load(open(a.extrinsics, encoding="utf-8"))
    tr_c, tr_u, tr_ns = load_split_clips(a.train_cache)
    ev_c, ev_u, ev_ns = load_split_clips(a.eval_cache)
    ns = sorted(set(tr_ns) | set(ev_ns))
    if len(ns) != 1:
        raise SystemExit(f"[vis1-pre] caches disagree on n_stack {ns}")
    n_stack = ns[0]
    overlap = set(tr_c) & set(ev_c)
    if overlap:
        raise SystemExit(f"[vis1-pre] {len(overlap)} clips are in BOTH the train and eval caches")
    uid_of = {c: u for c, u in zip(tr_c + ev_c, tr_u + ev_u)}
    split_of = {**{c: "train" for c in tr_c}, **{c: "eval" for c in ev_c}}
    miss_ex = [c for c in uid_of if c not in table]
    if miss_ex:
        raise SystemExit(f"[vis1-pre] {len(miss_ex)} clips have no extrinsics row (e.g. "
                         f"{[sha12(c) for c in miss_ex[:4]]})")
    rec["splits"] = {"train": {"cache": a.train_cache, "n_clips": len(tr_c),
                               "clip_sha12_sha256": hashlib.sha256("\n".join(sorted(sha12(c) for c in tr_c)).encode()).hexdigest()},
                     "eval": {"cache": a.eval_cache, "n_clips": len(ev_c),
                              "clip_sha12_sha256": hashlib.sha256("\n".join(sorted(sha12(c) for c in ev_c)).encode()).hexdigest()}}
    rec["n_stack"] = int(n_stack)
    rec["window"] = int(a.window)

    # ---- the clip set for this mode ----------------------------------------------------------------------- #
    banked = {}
    if a.mode == "control":
        if not a.banked_dir:
            raise SystemExit("[vis1-pre] --mode control needs --banked-dir")
        by12 = {sha12(c): c for c in uid_of}
        for nm in [s for s in a.banked_sets.split(",") if s]:
            p = Path(a.banked_dir) / f"vis_{nm}_boxes.json"
            rows = json.load(open(p, encoding="utf-8"))
            rec.setdefault("banked", {})[nm] = {"file": p.name, "md5": md5_file(p), "n_rows": len(rows)}
            for r in rows:
                banked.setdefault(r["sha12"], {}).setdefault((nm, int(r["w"])), []).append(r)
        unknown = sorted(s for s in banked if s not in by12)
        if unknown:
            raise SystemExit(f"[vis1-pre] {len(unknown)} banked clips are in neither cache: {unknown[:4]}")
        clips = sorted((by12[s] for s in banked), key=sha12)
    else:
        tr_sel = sorted(tr_c, key=sha12)
        ev_sel = sorted(ev_c, key=sha12)
        if a.max_clips:
            tr_sel, ev_sel = tr_sel[:a.max_clips], ev_sel[:a.max_clips]
        clips = sorted(tr_sel + ev_sel, key=sha12)
    n_sh = max(1, min(int(a.shards), len(clips)))
    shard_clips = [clips[k::n_sh] for k in range(n_sh)]
    shard_of = {c: k for k, cs in enumerate(shard_clips) for c in cs}
    rec["n_clips_run"] = len(clips)
    rec["n_shards"] = n_sh
    log(f"{a.mode}: {len(clips)} clips -> {n_sh} shards, n_stack {n_stack}, W {a.window}")

    # ---- inputs, by content -------------------------------------------------------------------------------- #
    rec["sources"] = {"agent_join": {"path": a.agent_join, "md5": md5_file(a.agent_join)},
                      "join3d": {"path": a.join3d, "md5": md5_file(a.join3d)},
                      "extrinsics": {"path": a.extrinsics, "md5": md5_file(a.extrinsics)}}
    chunks = sorted({int(table[c].get("chunk", -1)) for c in clips})
    calib = {}
    for ch in chunks:
        hit = None
        for d in a.calib_dir:
            for pat in VZ.INTRINSICS_GLOBS:
                q = Path(pat.format(calib_dir=d, ch=ch))
                if q.exists():
                    hit = q
                    break
            if hit:
                break
        calib[ch] = None if hit is None else md5_file(hit)
    no_int = [ch for ch, v in calib.items() if v is None]
    rec["sources"]["intrinsics"] = {"calib_dirs": list(a.calib_dir), "n_chunks": len(chunks),
                                    "n_missing": len(no_int),
                                    "md5_of_sorted_chunk_md5s": hashlib.md5(json.dumps(sorted(calib.items()),
                                                                                       default=str).encode()).hexdigest()}
    if no_int and not a.allow_bounds_only:
        raise SystemExit(f"[vis1-pre] {len(no_int)} chunk(s) have no intrinsics file (e.g. {no_int[:6]}); "
                         f"their clips would be z-buffered with image bounds only. Ship the parquet(s) or pass "
                         f"--allow-bounds-only BY NAME.")

    # ---- split the joins -------------------------------------------------------------------------------- #
    sdir = work / "shards"
    rec["split_2d"] = split_join(a.agent_join, sdir, shard_of, n_sh, "j2d")
    rec["split_3d"] = split_join(a.join3d, sdir, shard_of, n_sh, "j3d")

    _P.update({"shard_clips": shard_clips, "shard_dir": sdir, "uid_of": uid_of, "split_of": split_of,
               "n_stack": n_stack, "table": table, "calib_dirs": list(a.calib_dir), "window": int(a.window),
               "banked": banked})
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    t_pool = time.time()
    results = []
    fn = run_control_shard if a.mode == "control" else run_shard
    with ctx.Pool(processes=max(1, min(int(a.workers), n_sh))) as pool:
        for r in pool.imap_unordered(fn, range(n_sh)):
            results.append(r)
            done = len(results)
            log(f"shard {r['k']:03d} done ({done}/{n_sh}) wall {r.get('wall_s', '?')} s"
                + (f", load {r.get('load_s')} s, {r.get('n_frames')} frames" if a.mode == "full" else
                   f", {r['n_windows']} windows, {r['n_rows']} rows, {r['n_mismatch']} MISMATCH"))
    rec["pool_wall_s"] = round(time.time() - t_pool, 1)

    if a.mode == "control":
        n_w = sum(r["n_windows"] for r in results)
        n_r = sum(r["n_rows"] for r in results)
        n_m = sum(r["n_mismatch"] for r in results)
        exp_w = sum(len(v) for v in banked.values())
        exp_r = sum(len(rows) for v in banked.values() for rows in v.values())
        rec["control"] = {"n_windows": n_w, "n_windows_banked": exp_w, "n_rows_compared": n_r,
                          "n_rows_banked": exp_r, "n_mismatch": n_m,
                          "mismatch_sample": [m for r in results for m in r["mismatch"]][:50],
                          "PASS": bool(n_m == 0 and n_w == exp_w and n_r == exp_r)}
        rec["wall_s"] = round(time.time() - t_all, 1)
        (work / "control_record.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        log(f"CONTROL {'PASS' if rec['control']['PASS'] else 'FAIL'}: windows {n_w}/{exp_w}, rows {n_r}/{exp_r}, "
            f"mismatches {n_m}")
        return 0 if rec["control"]["PASS"] else 3

    # ---- assemble the sidecar (clips in sha12 order) ------------------------------------------------------- #
    per_clip = {c["sha12"]: c for r in results for c in r["clips"]}
    order = sorted(per_clip)
    A = {k: [] for k in ("row", "track", "n_full", "n_img", "n_vis", "vis_rows", "cx", "cy")}
    fr = {k: [] for k in ("frame_clip", "frame_f", "frame_n_zh", "frame_n_valid")}
    frame_ptr = [0]
    clip_frame_ptr = [0]
    n_rows = 0
    for ci, s12 in enumerate(order):
        c = per_clip[s12]
        for k in A:
            A[k].append(c[k])
        nf = len(c["frame_f"])
        fr["frame_clip"].append(np.full(nf, ci, np.int32))
        fr["frame_f"].append(c["frame_f"])
        fr["frame_n_zh"].append(c["frame_n_zh"])
        fr["frame_n_valid"].append(c["frame_n_valid"])
        for cnt in c["frame_cnt"].tolist():
            n_rows += int(cnt)
            frame_ptr.append(n_rows)
        clip_frame_ptr.append(clip_frame_ptr[-1] + nf)
    arrays = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in A.items()}
    arrays["row"] = arrays["row"].astype(np.int16)
    arrays["track"] = arrays["track"].astype(np.int64)
    for k in ("n_full", "n_img", "n_vis", "vis_rows"):
        arrays[k] = arrays[k].astype(np.int32)
    for k in ("cx", "cy"):
        arrays[k] = arrays[k].astype(np.float32)
    for k, v in fr.items():
        arrays[k] = np.concatenate(v).astype(np.int32) if v else np.zeros(0, np.int32)
    arrays["frame_ptr"] = np.asarray(frame_ptr, np.int64)
    arrays["clip_frame_ptr"] = np.asarray(clip_frame_ptr, np.int64)
    arrays["clip_sha12"] = np.asarray(order, dtype="<U12")
    arrays["clip_mask_ftheta"] = np.asarray([bool(per_clip[s]["mask_ftheta"]) for s in order], np.bool_)
    arrays["clip_split"] = np.asarray([per_clip[s]["split"] for s in order], dtype="<U5")
    if int(arrays["frame_ptr"][-1]) != len(arrays["row"]):
        raise SystemExit("[vis1-pre] internal: frame pointers do not close over the rows")
    content = hashlib.sha256()
    for k in sorted(arrays):
        content.update(k.encode())
        content.update(np.ascontiguousarray(arrays[k]).tobytes())
    meta = {"splits": rec["splits"], "sources": rec["sources"], "zbuffer": rec["zbuffer"],
            "store_scope": V.STORE_SCOPE, "window": int(a.window), "n_stack": int(n_stack),
            "frame_key": "the 2-D agent join's frame_idx (V3Dataset._agent_item's f = t + W - 1)",
            "row_key": "the agent's position in the join record (the padded block's row, pre-truncation)",
            "occluders": "every valid row with a 3-D label (targets AND filter-removed rows)",
            "n_clips": len(order), "n_frames": int(len(arrays["frame_f"])), "n_rows": int(len(arrays["row"])),
            "n_clips_bounds_only": int((~arrays["clip_mask_ftheta"]).sum()),
            "allow_bounds_only_mask": bool(a.allow_bounds_only),
            "max_clips_smoke": int(a.max_clips), "content_sha256_of_arrays": content.hexdigest()}
    out = Path(a.out or (work / "vis1_sidecar.npz"))
    dig = V.write_sidecar(out, arrays, meta)
    rec["sidecar"] = {"path": str(out), **dig, "n_clips": meta["n_clips"], "n_frames": meta["n_frames"],
                      "n_rows": meta["n_rows"], "content_sha256_of_arrays": meta["content_sha256_of_arrays"]}
    # a read-back through the reader the trainer uses -- a sidecar the reader refuses is not a sidecar
    sc = V.VIS1Sidecar(out)
    rec["readback"] = sc.stamp()
    per_shard = sorted(((r["k"], r["wall_s"], r.get("load_s"), r.get("n_frames")) for r in results))
    rec["per_shard"] = per_shard
    tot_frames = sum(x[3] or 0 for x in per_shard)
    rec["cpu_s_per_frame"] = round(sum(x[1] for x in per_shard) / max(tot_frames, 1), 4)
    rec["wall_s"] = round(time.time() - t_all, 1)
    (work / "full_record.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    log(f"SIDECAR {out.name}: {meta['n_clips']} clips, {meta['n_frames']} frames, {meta['n_rows']} rows, "
        f"sha256 {dig['sha256'][:16]}..., wall {rec['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
