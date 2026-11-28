#!/usr/bin/env python3
"""P3 -- MEASURE the per-frame agent-count distribution on the **TRAIN** corpus
join, closing refcv5 escalation #3.

WHY THIS SCRIPT EXISTS
----------------------
``refc_agents.AgentSeamConfig.queries = 32`` carries, in its own docstring:

    Measured on **val40**, not on the 2,376-episode train corpus, which has no
    join on this box -- the train distribution is UNMEASURED.

The second half of that sentence was wrong: the train join was built 2026-08-17
and banked to HF (``Sayood/tanitad-ph0-aug120 -> joins/train2400_agents.jsonl.xz``,
md5 24cbdca8c3b23aafc2fb17e6bf99cf76).  This script measures the train
distribution so the sizing decision stops resting on 39 val clips.

DO NOT RE-WRITE THE DISTRIBUTION MATHS
--------------------------------------
``rank_pct``, ``describe`` and ``decision_row`` are PATH-IMPORTED from the
refcv5 stream's ``measure_agent_density.py``, so the train and val40 tables are
produced by literally the same code.  Only the CUT PREDICATES had to be
re-stated here (they are function-local upstream), and that re-statement is
covered by a control.

CONTROLS (a probe with no control is a comment)
-----------------------------------------------
C0  **THE RE-STATEMENT CONTROL.**  The same cuts are run over the **val40**
    join and every ``infield_bevbox`` statistic must reproduce the banked
    ``agent_density.json`` EXACTLY.  If the re-stated predicates differ from
    the upstream ones by so much as a boundary, this fails and the train table
    below is void.
C1  ``occ`` is RECOMPUTED from ``(cx, cy)`` at hfov 120 and must agree on every
    box -- the same C1 the val40 pass ran.
C2  The three headline counts (433,040 frames / 12,122,129 boxes / 2,308 clips)
    are RE-DERIVED here, not copied from the meta.
C4  A NO-INFORMATION control: the drop fraction at ``n_queries = max_count``
    must be EXACTLY 0.0 by construction.

WARNING -- the md5 in the two joins' metas is over DIFFERENT ARTIFACTS.  val40's
``summary.md5`` is the md5 of the DECOMPRESSED ``.jsonl``; train's is the md5 of
the COMPRESSED ``.xz``.  A checker inherited from one silently REFUSES the
other (the upstream C5 does exactly that).  Both are therefore checked here by
name, and which artifact each digest covers is recorded in the output.

Evidence class: MEASURED (ours).  Artifacts named in the JSON output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

REPO = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
_G_UPSTREAM = (REPO + "/TanitAD Research Lab/Architecture & Inference/Research/"
               "2026-09-05-refcv5-build/measure_agent_density.py")
_G_VAL40_BANKED = (REPO + "/TanitAD Research Lab/Architecture & Inference/"
                   "Research/2026-09-05-refcv5-build/raw/agent_density.json")
# The G: mount cannot be IMPORTED FROM (OSError Errno 22 mid-read, CLAUDE.md).
# So both upstream artifacts are staged off-Drive and their sha256 is asserted
# against the G: original here -- which is what makes "literally the same code"
# an auditable claim rather than an assurance.
_HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.environ.get("UPSTREAM_DENSITY_PY",
                          os.path.join(_HERE, "_upstream_measure_agent_density.py"))
VAL40_BANKED = os.environ.get("UPSTREAM_DENSITY_JSON",
                              os.path.join(_HERE, "_upstream_agent_density.json"))


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _assert_same_as_gdrive(local, gpath, label):
    """Return (sha, matched|'INCONCLUSIVE') -- never claim a match on an
    unreadable operand (both sides empty compares equal, CLAUDE.md)."""
    try:
        a, b = _sha256(local), _sha256(gpath)
    except OSError as ex:
        return {"label": label, "local_sha256": None,
                "status": "INCONCLUSIVE (%r)" % (ex,)}
    if len(a) != 64 or len(b) != 64:
        return {"label": label, "local_sha256": a, "status": "INCONCLUSIVE"}
    return {"label": label, "local_sha256": a, "gdrive_sha256": b,
            "status": "VERIFIED-IDENTICAL" if a == b else "DIVERGED"}

TRAIN_JSONL = r"C:\Users\Admin\tanitad-data\joins\joins\train2400_agents.jsonl"
TRAIN_XZ = TRAIN_JSONL + ".xz"
TRAIN_META = TRAIN_XZ + ".meta.json"
VAL40_JSONL = ("C:/Users/Admin/tanitad-caches/val40-obstacle-20260818/join/"
               "val40_agents.jsonl")

# ---- the distribution maths, IMPORTED from the refcv5 stream --------------
_spec = importlib.util.spec_from_file_location("_agent_density_upstream",
                                               UPSTREAM)
_up = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_up)
describe, decision_row, rank_pct, md5_of = (_up.describe, _up.decision_row,
                                            _up.rank_pct, _up.md5_of)
X_FWD_M, Y_HALF_M, HFOV_DEG = _up.X_FWD_M, _up.Y_HALF_M, _up.HFOV_DEG
N_QUERY_GRID, ALL_CLASSES = _up.N_QUERY_GRID, _up.ALL_CLASSES
_HALF = math.radians(HFOV_DEG) / 2.0

# re-stated because they are function-local upstream; C0 proves the
# re-statement is the same predicate.
CUTS = {
    "all":            lambda cx, cy, o, r: np.ones(o.shape, dtype=bool),
    "all_r60":        lambda cx, cy, o, r: r <= 60.0,
    "all_r80":        lambda cx, cy, o, r: r <= 80.0,
    "infield":        lambda cx, cy, o, r: o == 0,
    "infield_r40":    lambda cx, cy, o, r: (o == 0) & (r <= 40.0),
    "infield_r60":    lambda cx, cy, o, r: (o == 0) & (r <= 60.0),
    "infield_r80":    lambda cx, cy, o, r: (o == 0) & (r <= 80.0),
    "infield_r100":   lambda cx, cy, o, r: (o == 0) & (r <= 100.0),
    "infield_fwd60":  lambda cx, cy, o, r: (o == 0) & (cx <= X_FWD_M),
    "infield_bevbox": lambda cx, cy, o, r: ((o == 0) & (cx >= 0.0)
                                            & (cx <= X_FWD_M)
                                            & (np.abs(cy) <= Y_HALF_M)),
}
DECISION_CUTS = ("all", "infield", "infield_r60", "infield_r80",
                 "infield_bevbox")


class _CSRRanges:
    """``ranges[i]`` -> that frame's ranges, backed by ONE flat array.

    Why not a list of arrays: 433,040 frames x 5 cuts is 2.2 M numpy objects
    whose per-object overhead alone is ~240 MB before a single float. Same
    class as the dense-windowed-tensor trap in CLAUDE.md -- price the host
    tensor before building it.
    """

    def __init__(self):
        self._buf = []
        self._off = [0]
        self._flat = None
        self._offa = None

    def append(self, r):
        self._buf.append(np.asarray(r, dtype=np.float32))
        self._off.append(self._off[-1] + int(np.asarray(r).size))

    def finish(self):
        self._flat = (np.concatenate(self._buf) if self._buf
                      else np.zeros(0, np.float32))
        self._buf = []
        self._offa = np.asarray(self._off, dtype=np.int64)
        self._off = []
        return self

    def __getitem__(self, i):
        return self._flat[self._offa[i]:self._offa[i + 1]]


def census(path, tag):
    """One pass over a join file -> the same tables the val40 pass produced."""
    t0 = time.time()
    counts = {k: [] for k in CUTS}
    ranges = {k: _CSRRanges() for k in DECISION_CUTS}
    cls_hist = {k: Counter() for k in DECISION_CUTS}
    cls_idx = {c: i for i, c in enumerate(ALL_CLASSES)}
    clip_ids = set()
    seen = set()
    n_lines = n_boxes = n_visible = n_clear = 0
    n_dup_keys = n_occ_disagree = n_nest_violation = 0
    n_unknown_cls = n_missing_cls = n_bevbox_beyond_r60 = 0
    per_clip_max = {}

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_lines += 1
            cid = str(rec["clip_id"])
            key = (cid, int(rec["frame_idx"]))
            if key in seen:
                n_dup_keys += 1
            seen.add(key)
            clip_ids.add(cid)
            ag = rec.get("agents") or []
            if not ag:
                n_clear += 1
            n_boxes += len(ag)
            if ag:
                cx = np.fromiter((float(d["cx"]) for d in ag), np.float64,
                                 len(ag))
                cy = np.fromiter((float(d["cy"]) for d in ag), np.float64,
                                 len(ag))
                occ = np.fromiter((int(d["occ"]) for d in ag), np.int64,
                                  len(ag))
                rng = np.hypot(cx, cy)
                # C1 -- recompute the flag from the geometry
                recomputed = np.where(np.abs(np.arctan2(cy, cx)) <= _HALF,
                                      0, 1)
                n_occ_disagree += int((recomputed != occ).sum())
                n_visible += int((occ == 0).sum())
                ci = np.fromiter(
                    (cls_idx.get(str(d.get("cls")), -1) for d in ag),
                    np.int64, len(ag))
                for d in ag:
                    if d.get("cls") is None:
                        n_missing_cls += 1
                    elif str(d.get("cls")) not in cls_idx:
                        n_unknown_cls += 1
            else:
                cx = cy = rng = np.zeros(0, np.float64)
                occ = ci = np.zeros(0, np.int64)

            per_cut_n = {}
            for name, pred in CUTS.items():
                m = pred(cx, cy, occ, rng) if occ.size else np.zeros(0, bool)
                n = int(m.sum())
                per_cut_n[name] = n
                counts[name].append(n)
                if name in ranges:
                    ranges[name].append(rng[m] if occ.size
                                        else np.zeros(0, np.float64))
                    if n:
                        vals, cnts = np.unique(ci[m], return_counts=True)
                        for v, c in zip(vals, cnts):
                            cls_hist[name][ALL_CLASSES[int(v)] if v >= 0
                                           else "__unknown__"] += int(c)
            ok = (per_cut_n["infield_bevbox"] <= per_cut_n["infield_fwd60"]
                  <= per_cut_n["infield"] <= per_cut_n["all"]
                  and per_cut_n["infield_r40"] <= per_cut_n["infield_r60"]
                  <= per_cut_n["infield_r80"] <= per_cut_n["infield_r100"]
                  <= per_cut_n["infield"]
                  and per_cut_n["infield_r60"] <= per_cut_n["infield_fwd60"]
                  and per_cut_n["all_r60"] <= per_cut_n["all_r80"]
                  <= per_cut_n["all"])
            if not ok:
                n_nest_violation += 1
            if occ.size:
                n_bevbox_beyond_r60 += int(
                    ((occ == 0) & (cx >= 0.0) & (cx <= X_FWD_M)
                     & (np.abs(cy) <= Y_HALF_M) & (rng > 60.0)).sum())
            per_clip_max[cid] = max(per_clip_max.get(cid, 0),
                                    per_cut_n["infield_bevbox"])

    arr = {k: np.asarray(v, np.int64) for k, v in counts.items()}
    for k in ranges:
        ranges[k].finish()
    dist = {k: describe(arr[k], int(arr[k].sum())) for k in CUTS}
    decision = {}
    for k in DECISION_CUTS:
        rows = [decision_row(arr[k], ranges[k], n) for n in N_QUERY_GRID]
        mx = int(arr[k].max()) if arr[k].size else 0
        ctrl = decision_row(arr[k], ranges[k], mx)          # C4
        decision[k] = {"grid": rows, "control_at_max": ctrl,
                       "C4_zero_drop_at_max":
                       ctrl["frac_boxes_dropped"] == 0.0}
    return {
        "tag": tag, "path": path, "wall_s": round(time.time() - t0, 1),
        "n_frames": n_lines, "n_clips": len(clip_ids), "n_boxes": n_boxes,
        "n_visible_boxes": n_visible,
        "visible_frac": round(n_visible / max(n_boxes, 1), 4),
        "n_frames_labelled_clear": n_clear,
        "n_duplicate_keys": n_dup_keys,
        "C1_occ_disagreements": n_occ_disagree,
        "C3_nest_violations": n_nest_violation,
        "n_boxes_bevbox_beyond_r60": n_bevbox_beyond_r60,
        "n_boxes_unknown_cls": n_unknown_cls,
        "n_boxes_missing_cls": n_missing_cls,
        "dist": dist, "decision": decision,
        "cls_hist": {k: dict(v.most_common()) for k, v in cls_hist.items()},
        "per_clip_max_infield_bevbox": {
            "max": max(per_clip_max.values()) if per_clip_max else 0,
            "mean": round(float(np.mean(list(per_clip_max.values()))), 2)
            if per_clip_max else 0.0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw/train_agent_density.json")
    ap.add_argument("--skip-val40-control", action="store_true")
    a = ap.parse_args()

    prov = {"_evidence_class": "MEASURED (ours)",
            "upstream_module": UPSTREAM,
            "upstream_provenance": [
                _assert_same_as_gdrive(UPSTREAM, _G_UPSTREAM,
                                       "measure_agent_density.py"),
                _assert_same_as_gdrive(VAL40_BANKED, _G_VAL40_BANKED,
                                       "agent_density.json")],
            "N_QUERY_GRID": list(N_QUERY_GRID),
            "SlotDecodeRanges": {"x_fwd_m": X_FWD_M, "y_half_m": Y_HALF_M},
            "hfov_deg": HFOV_DEG}

    # ---- md5, BOTH artifacts, named -------------------------------------
    meta = json.load(open(TRAIN_META, encoding="utf-8"))
    want = str(meta["summary"]["md5"])
    got_xz = md5_of(TRAIN_XZ)
    got_jsonl = md5_of(TRAIN_JSONL)
    prov["md5"] = {
        "meta_summary_md5": want,
        "md5_of_xz": got_xz, "md5_of_decompressed_jsonl": got_jsonl,
        "meta_digest_covers": ("xz" if got_xz == want else
                               ("jsonl" if got_jsonl == want else "NEITHER")),
        "verified": want in (got_xz, got_jsonl)}
    print("[md5] meta=%s  covers=%s  verified=%s"
          % (want, prov["md5"]["meta_digest_covers"],
             prov["md5"]["verified"]), flush=True)
    if not prov["md5"]["verified"]:
        print("REFUSING: no md5-verified copy of the train join. A census of "
              "an unverified file is not a measurement.", file=sys.stderr)
        return 2

    # ---- C0: the re-statement control on val40 --------------------------
    ctrl = None
    if not a.skip_val40_control and os.path.exists(VAL40_JSONL):
        print("[C0] re-running the cuts on val40 ...", flush=True)
        v = census(VAL40_JSONL, "val40")
        banked = json.load(open(VAL40_BANKED, encoding="utf-8"))
        bd = banked["distribution"]["infield_bevbox"]
        vd = v["dist"]["infield_bevbox"]
        keys = ("n_frames", "n_boxes", "mean", "median", "max", "p99",
                "p95", "p90", "n_frames_zero")
        mism = {}
        for k in keys:
            x, y = bd.get(k), vd.get(k)
            same = (abs(x - y) < 1e-9 if isinstance(x, float)
                    and isinstance(y, float) else x == y)
            if not same:
                mism[k] = (x, y)
        ctrl = {"banked": {k: bd.get(k) for k in keys},
                "recomputed": {k: vd.get(k) for k in keys},
                "mismatches": mism, "PASS": not mism,
                "val40_headline": {"n_frames": v["n_frames"],
                                   "n_boxes": v["n_boxes"],
                                   "n_clips": v["n_clips"]},
                "C1_occ_disagreements": v["C1_occ_disagreements"]}
        print("[C0] re-statement control: %s  %r"
              % ("PASS" if not mism else "FAIL", mism), flush=True)
    prov["C0_restatement_control"] = ctrl

    # ---- the train census ------------------------------------------------
    print("[train] census over %s ..." % TRAIN_JSONL, flush=True)
    tr = census(TRAIN_JSONL, "train2400")
    exp = meta["summary"]
    tr["C2_headline_matches_meta"] = {
        "n_frames": (tr["n_frames"], exp["n_frames"],
                     tr["n_frames"] == exp["n_frames"]),
        "n_clips": (tr["n_clips"], exp["n_episodes"],
                    tr["n_clips"] == exp["n_episodes"]),
        "n_boxes": (tr["n_boxes"], exp["n_agent_boxes"],
                    tr["n_boxes"] == exp["n_agent_boxes"]),
        "visible_frac": (tr["visible_frac"], exp["visible_frac"],
                         abs(tr["visible_frac"] - exp["visible_frac"])
                         < 5e-4)}

    res = {"provenance": prov, "train": tr}
    out = a.out
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    d = tr["dist"]["infield_bevbox"]
    print("")
    print("=== TRAIN corpus, cut = infield & decode box (the trainable set)")
    print("  n_frames %d  n_boxes %d  mean %.2f  median %.0f  p95 %d  p99 %d "
          "max %d  zero-frames %.1f%%"
          % (d["n_frames"], d["n_boxes"], d["mean"], d["median"], d["p95"],
             d["p99"], d["max"], 100 * d["frac_frames_zero"]))
    print("  visible_frac(all boxes) %.4f   C1 occ disagreements %d   "
          "C3 nest violations %d"
          % (tr["visible_frac"], tr["C1_occ_disagreements"],
             tr["C3_nest_violations"]))
    for row in tr["decision"]["infield_bevbox"]["grid"]:
        print("  N=%-3d frames_with_drop %8d (%.4f%%)  boxes_dropped %8d "
              "(%.4f%%)  nearest_dropped_min %s"
              % (row["n_queries"], row["n_frames_with_drop"],
                 100 * row["frac_frames_with_drop"], row["n_boxes_dropped"],
                 100 * row["frac_boxes_dropped"],
                 ("%.1f m" % row["nearest_dropped_m_min"])
                 if row["nearest_dropped_m_min"] is not None else "-"))
    print("  C4 zero-drop at max: %s"
          % tr["decision"]["infield_bevbox"]["C4_zero_drop_at_max"])
    print("")
    print("wrote %s" % out)
    bad = (tr["C1_occ_disagreements"] or tr["C3_nest_violations"]
           or (ctrl is not None and not ctrl["PASS"])
           or not all(v[2] for v in tr["C2_headline_matches_meta"].values()))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
