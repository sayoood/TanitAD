#!/usr/bin/env python3
"""MEASURE the per-frame agent-count distribution of the ``obstacle.offline``
join -- the number ``agent_slots.N_QUERIES_DEFAULT = 16`` is a placeholder for.

WHY THIS SCRIPT EXISTS
----------------------
``stack/tanitad/models/agent_slots.py`` says, in its own docstring:

    n_queries must be >= the per-frame agent count. When it is not, the
    FARTHEST targets are dropped (by range) and the drop is COUNTED ...
    The right n_queries is the join's measured per-frame agent-count
    distribution; that distribution is UNMEASURED on this box ...
    N_QUERIES_DEFAULT is a declared placeholder, not a fitted value.

A join file DOES now exist on this box (2026-08-18 val40 lead-join package), so
the distribution is measurable. This script measures it.

WHAT IT DOES NOT CLAIM
----------------------
* The join is the **val40** corpus (39 of 40 val episodes -- ep_00037 has no
  ``obstacle.offline``).  It is NOT the 2,376-episode canonical TRAIN corpus.
  The training-corpus distribution is a different quantity and is UNMEASURED;
  everything here is stamped ``corpus = physicalai-val-0c5f7dac3b11``.
* ``occ`` is flagged at the SENSOR's 120 deg, not at whatever sub-frame the
  encoder is actually fed (build_obstacle_join.py's own
  ``encoder_frame_rule`` warning).  The in-field set measured here is therefore
  an UPPER bound on what a narrower encoder frame can see.
* No tier claim is made: this is a LABEL-SIDE census, not a model evaluation.

THE CUTS, AND WHY EACH ONE EXISTS
---------------------------------
* ``all``            -- every cuboid the join carries, at any azimuth and any
                        range (the builder applies NO range filter; boxes at
                        -83 m appear in line 1 of the file).
* ``infield``        -- ``occ == 0``: agent centre azimuth within +-60 deg.
                        A front-camera monocular head cannot see the rest, so a
                        query trained against them asks for a hallucination.
* ``*_rNN``          -- Euclidean range ``sqrt(cx^2 + cy^2) <= NN`` metres.
* ``infield_fwd60``  -- in-field and ``cx <= 60``: SlotDecodeRanges.x_fwd_m.
* ``infield_bevbox`` -- in-field and inside GRID_DEFAULT's box
                        (``0 <= cx <= 60`` and ``|cy| <= 16``): the set the
                        decode constants were sized for.

CONTROLS (a probe with no control is a comment)
-----------------------------------------------
C1  ``occ`` is RECOMPUTED from ``(cx, cy)`` at hfov 120 and must agree on
    every box.  If it does not, the flag does not mean what the doc says and
    every in-field number below is void.
C2  The three inherited headline counts (7,400 frames / 195,805 boxes /
    39 clips) are re-derived, not copied.
C3  The cuts must NEST along the chains that ARE nested, per frame.
    ⚠️ ``infield_bevbox`` is NOT a subset of ``infield_r60``: the BEV box's far
    corner is at ``sqrt(60^2 + 16^2) = 62.08 m``, outside a 60 m radius.  The
    first version of this control asserted that chain and FAILED on 85 frames
    -- the control caught a mis-written predicate, which is what it is for.
    The count of boxes living in that corner is reported below.
C4  A NO-INFORMATION control: the "drop fraction" at ``n_queries = max_count``
    must be EXACTLY 0.0 by construction.  A table that does not read 0 there is
    computing something other than what it says.
C5  Both source copies of the join (durable local jsonl + the repo's .xz) are
    md5-checked against the meta's recorded digest before anything is counted.

Evidence class: MEASURED (ours).  Artifacts named in the JSON output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np

# --------------------------------------------------------------------------
# constants imported from the label/model side, never re-listed here
# --------------------------------------------------------------------------
try:
    from tanitad.data.bev_raster import ALL_CLASSES, GRID_DEFAULT
    from tanitad.models.agent_slots import N_QUERIES_DEFAULT
    _IMPORT_OK = True
except Exception as exc:                                  # pragma: no cover
    print(f"[warn] stack import failed ({exc!r}) -- falling back to literals; "
          f"the assertions below then cannot fire", file=sys.stderr)
    ALL_CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle",
                   "trailer", "person", "rider", "stroller", "animal",
                   "protruding_object")
    GRID_DEFAULT = None
    N_QUERIES_DEFAULT = 16
    _IMPORT_OK = False

X_FWD_M = 60.0 if GRID_DEFAULT is None else float(GRID_DEFAULT.x_fwd_m)
Y_HALF_M = 16.0 if GRID_DEFAULT is None else float(GRID_DEFAULT.y_half_m)
HFOV_DEG = 120.0                       # the join's own flag angle (meta)

JOIN_LOCAL = ("C:/Users/Admin/tanitad-caches/val40-obstacle-20260818/join/"
              "val40_agents.jsonl")
REPO = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
JOIN_XZ = (f"{REPO}/TanitAD Research Lab/Benchmarks & Evals/Implementation/"
           f"incoming/2026-08-18-val40-lead-join/raw/val40_agents.jsonl.xz")
JOIN_META = JOIN_XZ + ".meta.json"

N_QUERY_GRID = (8, 12, 16, 20, 24, 32, 48, 64)
PCTS = (50.0, 75.0, 90.0, 95.0, 99.0, 99.9)


def md5_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# distribution helpers
# --------------------------------------------------------------------------
def rank_pct(counts: np.ndarray, q: float) -> int:
    """INVERSE-CDF percentile: the smallest integer N with P(count <= N) >= q%.

    This -- not numpy's interpolating percentile -- is the sizing-relevant
    definition: a buffer of size N covers a frame iff count <= N, and a
    fractional query count is not a thing one can allocate.
    """
    if counts.size == 0:
        return 0
    s = np.sort(counts)
    k = int(math.ceil(q / 100.0 * s.size)) - 1
    return int(s[min(max(k, 0), s.size - 1)])


def describe(counts: np.ndarray, n_boxes: int) -> dict:
    c = np.asarray(counts, dtype=np.int64)
    out = {
        "n_frames": int(c.size),
        "n_boxes": int(n_boxes),
        "n_boxes_check": int(c.sum()),
        "mean": float(c.mean()) if c.size else 0.0,
        "std": float(c.std(ddof=1)) if c.size > 1 else 0.0,
        "median": float(np.median(c)) if c.size else 0.0,
        "min": int(c.min()) if c.size else 0,
        "max": int(c.max()) if c.size else 0,
        "n_frames_zero": int((c == 0).sum()),
        "frac_frames_zero": float((c == 0).mean()) if c.size else 0.0,
    }
    for q in PCTS:
        tag = ("p%g" % q).replace(".", "_")
        out[tag] = rank_pct(c, q)
        out[tag + "_linear"] = float(np.percentile(c, q)) if c.size else 0.0
    return out


def decision_row(counts: np.ndarray, ranges: list[np.ndarray],
                 n_q: int) -> dict:
    """What ``match_slots`` would do at ``n_queries = n_q`` on this cut.

    ``ranges[i]`` = the per-box ``sqrt(cx^2+cy^2)`` for frame i, ASCENDING-
    sortable; match_slots keeps the ``n_q`` NEAREST and drops the rest, so the
    dropped set is exactly ``sorted(range)[n_q:]``.
    """
    c = np.asarray(counts, dtype=np.int64)
    over = np.maximum(c - n_q, 0)
    n_dropped = int(over.sum())
    n_boxes = int(c.sum())
    nearest_dropped = []
    for i in np.nonzero(over)[0]:
        r = np.sort(ranges[int(i)])
        nearest_dropped.append(float(r[n_q]))       # the closest dropped box
    nd = np.asarray(nearest_dropped, dtype=np.float64)
    return {
        "n_queries": int(n_q),
        "n_frames_with_drop": int((over > 0).sum()),
        "frac_frames_with_drop": float((over > 0).mean()) if c.size else 0.0,
        "n_boxes_dropped": n_dropped,
        "frac_boxes_dropped": (float(n_dropped / n_boxes) if n_boxes else 0.0),
        "max_dropped_in_one_frame": int(over.max()) if c.size else 0,
        # how NEAR does the closest sacrificed target get?  If the answer is
        # "70 m" the drop is clutter; if it is "8 m" the head stops being
        # scored on an agent it is about to hit.
        "nearest_dropped_m_min": float(nd.min()) if nd.size else None,
        "nearest_dropped_m_p5": float(np.percentile(nd, 5)) if nd.size
        else None,
        "nearest_dropped_m_median": float(np.median(nd)) if nd.size else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-xz-check", action="store_true",
                    help="skip decompressing the repo .xz cross-check (the "
                         "G: mount flaps; the local copy is still md5-checked)")
    a = ap.parse_args()

    prov = {"_evidence_class": "MEASURED (ours)"}

    # ---- C5: the sources, md5-checked before a single box is counted -------
    meta = json.load(open(JOIN_META, encoding="utf-8"))
    want_md5 = str(meta["summary"]["md5"])
    prov["meta_path"] = JOIN_META
    prov["expected_md5"] = want_md5
    prov["corpus_key"] = meta.get("corpus_key")
    prov["hfov_deg_used_by_join"] = meta["p4_predicate_identity"][
        "hfov_deg_used"]
    prov["hfov_is_sensor_default"] = meta["p4_predicate_identity"][
        "hfov_is_sensor_default"]
    prov["inherited_summary"] = {
        "n_episodes": meta["summary"]["n_episodes"],
        "n_frames": meta["summary"]["n_frames"],
        "n_agent_boxes": meta["summary"]["n_agent_boxes"],
        "visible_frac": meta["summary"]["visible_frac"],
    }

    src = None
    if os.path.exists(JOIN_LOCAL):
        got = md5_of(JOIN_LOCAL)
        prov["local_copy"] = {"path": JOIN_LOCAL, "md5": got,
                              "matches_meta": got == want_md5}
        if got == want_md5:
            src = JOIN_LOCAL
    if src is None or not a.skip_xz_check:
        tmpd = tempfile.mkdtemp(prefix="agentjoin_")
        tmp = os.path.join(tmpd, "val40_agents.jsonl")
        with lzma.open(JOIN_XZ, "rb") as fin, open(tmp, "wb") as fout:
            while True:
                b = fin.read(1 << 20)
                if not b:
                    break
                fout.write(b)
        got_xz = md5_of(tmp)
        prov["repo_xz_copy"] = {"path": JOIN_XZ, "decompressed_to": tmp,
                                "md5": got_xz,
                                "matches_meta": got_xz == want_md5}
        if src is None and got_xz == want_md5:
            src = tmp
    if src is None:
        print("REFUSING: no md5-verified copy of the join is available. "
              "A census of an unverified file is not a measurement.",
              file=sys.stderr)
        return 2
    prov["source_used"] = src
    prov["stack_import_ok"] = _IMPORT_OK
    prov["N_QUERIES_DEFAULT_at_measure_time"] = int(N_QUERIES_DEFAULT)
    prov["SlotDecodeRanges"] = {"x_fwd_m": X_FWD_M, "y_half_m": Y_HALF_M}
    prov["AGENT_CLASSES"] = list(ALL_CLASSES)

    # ---- parse ------------------------------------------------------------
    half = math.radians(HFOV_DEG) / 2.0
    cuts = {
        "all":             lambda cx, cy, o, r: np.ones(o.shape, dtype=bool),
        "all_r60":         lambda cx, cy, o, r: r <= 60.0,
        "all_r80":         lambda cx, cy, o, r: r <= 80.0,
        "infield":         lambda cx, cy, o, r: o == 0,
        "infield_r40":     lambda cx, cy, o, r: (o == 0) & (r <= 40.0),
        "infield_r60":     lambda cx, cy, o, r: (o == 0) & (r <= 60.0),
        "infield_r80":     lambda cx, cy, o, r: (o == 0) & (r <= 80.0),
        "infield_r100":    lambda cx, cy, o, r: (o == 0) & (r <= 100.0),
        "infield_fwd60":   lambda cx, cy, o, r: (o == 0) & (cx <= X_FWD_M),
        "infield_bevbox":  lambda cx, cy, o, r: ((o == 0) & (cx >= 0.0)
                                                 & (cx <= X_FWD_M)
                                                 & (np.abs(cy) <= Y_HALF_M)),
    }
    counts = {k: [] for k in cuts}
    ranges_by_cut = {k: [] for k in cuts}
    cls_hist = {k: Counter() for k in cuts}
    cls_frames = {k: Counter() for k in cuts}
    cls_clips = {k: defaultdict(set) for k in cuts}

    clip_ids = set()
    seen_keys = set()
    n_dup_keys = 0
    n_lines = 0
    n_boxes_total = 0
    n_visible_total = 0
    n_occ_disagree = 0            # C1
    n_nest_violation = 0          # C3
    n_bevbox_beyond_r60 = 0       # the 60x16 corner, outside a 60 m radius
    n_frames_bevbox_gt_r60 = 0
    n_clear_frames = 0            # labelled CLEAR (agents == [])
    n_missing_cls = 0
    n_unknown_cls = 0
    n_dup_track_in_frame = 0
    per_clip_max = {}
    per_clip_frames = Counter()
    per_clip_boxes = Counter()
    frame_index = []              # (clip_id, frame_idx) parallel to counts
    occ_all: list = []            # per-frame occ, aligned to ranges["all"]
    inbev_all: list = []          # per-frame "inside the decode box" bool

    with open(src, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_lines += 1
            cid = str(rec["clip_id"])
            fidx = int(rec["frame_idx"])
            clip_ids.add(cid)
            per_clip_frames[cid] += 1
            key = (cid, fidx)
            if key in seen_keys:
                n_dup_keys += 1
            seen_keys.add(key)
            frame_index.append(key)

            ag = rec.get("agents") or []
            if not ag:
                n_clear_frames += 1
            n_boxes_total += len(ag)
            per_clip_boxes[cid] += len(ag)

            if ag:
                cx = np.fromiter((float(d["cx"]) for d in ag), np.float64,
                                 len(ag))
                cy = np.fromiter((float(d["cy"]) for d in ag), np.float64,
                                 len(ag))
                occ = np.fromiter((int(d["occ"]) for d in ag), np.int64,
                                  len(ag))
                rng = np.hypot(cx, cy)
                cl = [d.get("cls") for d in ag]
                tids = [d.get("track_id") for d in ag]
                if len(set(tids)) != len(tids):
                    n_dup_track_in_frame += 1
                # C1: recompute the flag
                recomputed = np.where(np.abs(np.arctan2(cy, cx)) <= half, 0, 1)
                n_occ_disagree += int((recomputed != occ).sum())
                n_visible_total += int((occ == 0).sum())
            else:
                cx = cy = rng = np.zeros(0, dtype=np.float64)
                occ = np.zeros(0, dtype=np.int64)
                cl = []
            occ_all.append(occ)
            inbev_all.append(((occ == 0) & (cx >= 0.0) & (cx <= X_FWD_M)
                              & (np.abs(cy) <= Y_HALF_M)) if occ.size
                             else np.zeros(0, dtype=bool))

            per_cut_n = {}
            for name, pred in cuts.items():
                m = pred(cx, cy, occ, rng) if occ.size else np.zeros(
                    0, dtype=bool)
                n = int(m.sum())
                per_cut_n[name] = n
                counts[name].append(n)
                ranges_by_cut[name].append(rng[m] if occ.size
                                           else np.zeros(0))
                if n:
                    seen_here = set()
                    for j in np.nonzero(m)[0]:
                        c = cl[int(j)]
                        if c is None:
                            k = "__missing__"
                            if name == "all":
                                n_missing_cls += 1
                        else:
                            k = str(c)
                            if k not in ALL_CLASSES and name == "all":
                                n_unknown_cls += 1
                        cls_hist[name][k] += 1
                        seen_here.add(k)
                        cls_clips[name][k].add(cid)
                    for k in seen_here:
                        cls_frames[name][k] += 1
            # C3: nesting, along the chains that ARE nested
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
            if per_cut_n["infield_bevbox"] > per_cut_n["infield_r60"]:
                n_frames_bevbox_gt_r60 += 1
            if occ.size:
                corner = ((occ == 0) & (cx >= 0.0) & (cx <= X_FWD_M)
                          & (np.abs(cy) <= Y_HALF_M) & (rng > 60.0))
                n_bevbox_beyond_r60 += int(corner.sum())
            per_clip_max[cid] = max(per_clip_max.get(cid, 0),
                                    per_cut_n["all"])

    arr = {k: np.asarray(v, dtype=np.int64) for k, v in counts.items()}

    # ---- the tables -------------------------------------------------------
    dist = {k: describe(arr[k], int(arr[k].sum())) for k in cuts}

    decision = {}
    for k in ("all", "infield", "infield_r60", "infield_r80",
              "infield_bevbox"):
        rows = [decision_row(arr[k], ranges_by_cut[k], n)
                for n in N_QUERY_GRID]
        # C4: the no-information control -- at n = max the drop must be 0
        mx = int(arr[k].max()) if arr[k].size else 0
        ctrl = decision_row(arr[k], ranges_by_cut[k], mx)
        rows.append({**ctrl, "_control": "n_queries == max_count; "
                                         "frac_frames_with_drop MUST be 0.0"})
        decision[k] = rows

    # ---- the NAIVE-PIPELINE audit ----------------------------------------
    # `targets_from_join` sets valid=True for EVERY agent in the record -- it
    # applies no azimuth and no range filter.  So the pipeline as written today
    # feeds the `all` cut, and `match_slots` then keeps the n_queries NEAREST.
    # THAT IS THE DEFECT n_queries alone cannot fix: the nearest agents include
    # ones BEHIND the ego, so a slot is spent asking a front camera to regress
    # a car it cannot see.  This table prices it.
    naive = []
    for n_q in N_QUERY_GRID:
        kept = kept_occ1 = kept_out_box = 0
        for r, o, b in zip(ranges_by_cut["all"], occ_all, inbev_all):
            if r.size == 0:
                continue
            sel = np.argsort(r)[:n_q]
            kept += int(sel.size)
            kept_occ1 += int((o[sel] == 1).sum())
            kept_out_box += int((~b[sel]).sum())
        naive.append({
            "n_queries": int(n_q),
            "n_targets_kept": kept,
            "n_kept_occ1_out_of_field": kept_occ1,
            "frac_kept_out_of_field": (kept_occ1 / kept) if kept else 0.0,
            "n_kept_outside_decode_box": kept_out_box,
            "frac_kept_outside_decode_box": ((kept_out_box / kept)
                                             if kept else 0.0),
        })

    # ---- TAIL CONCENTRATION ----------------------------------------------
    # A p99 pooled over 7,400 frames from 39 episodes can be ONE episode's
    # traffic jam.  The programme's estimator is an episode-cluster bootstrap
    # for exactly this reason, so the tail is reported at the CLIP grain: a
    # threshold exceeded in 1-2 clips is a property of those clips, not of the
    # corpus, and must not be quoted as "the 99th percentile of driving".
    clip_of = np.array([c for c, _ in frame_index], dtype=object)
    tail = {}
    for k in ("all", "infield", "infield_r60", "infield_bevbox"):
        rows = []
        for n_q in N_QUERY_GRID:
            over = arr[k] > n_q
            cc = Counter(clip_of[over].tolist())
            tot = int(over.sum())
            top = cc.most_common(3)
            rows.append({
                "n_queries": int(n_q),
                "n_frames_over": tot,
                "n_clips_contributing": len(cc),
                "top3_clip_share": (round(sum(v for _, v in top) / tot, 4)
                                    if tot else None),
                "top3": [[c[:8], v] for c, v in top],
            })
        tail[k] = rows

    classes = {}
    for k in ("all", "infield", "infield_r60", "infield_bevbox"):
        tot = sum(cls_hist[k].values())
        classes[k] = {
            "n_boxes": tot,
            "by_class": {c: {"n_boxes": cls_hist[k].get(c, 0),
                             "frac": (cls_hist[k].get(c, 0) / tot
                                      if tot else 0.0),
                             "n_frames_present": cls_frames[k].get(c, 0),
                             "n_clips_present": len(cls_clips[k].get(c, ()))}
                         for c in list(ALL_CLASSES)
                         + sorted(set(cls_hist[k]) - set(ALL_CLASSES))},
        }

    controls = {
        "C1_occ_recomputed_disagreements": n_occ_disagree,
        "C1_verdict": ("PASS -- occ IS abs(atan2(cy,cx)) <= 60 deg on every "
                       "box" if n_occ_disagree == 0 else
                       "FAIL -- the in-field numbers are VOID"),
        "C2_n_frames": n_lines,
        "C2_n_boxes": n_boxes_total,
        "C2_n_clips": len(clip_ids),
        "C2_visible_frac": (round(n_visible_total / n_boxes_total, 4)
                            if n_boxes_total else None),
        "C2_matches_inherited": {
            "n_frames": n_lines == meta["summary"]["n_frames"],
            "n_boxes": n_boxes_total == meta["summary"]["n_agent_boxes"],
            "n_clips": len(clip_ids) == meta["summary"]["n_episodes"],
            "visible_frac": (abs((n_visible_total / n_boxes_total)
                                 - meta["summary"]["visible_frac"]) < 5e-5),
        },
        "C3_nesting_violations": n_nest_violation,
        "C3_note": ("infield_bevbox is deliberately NOT nested inside "
                    "infield_r60: the decode box's far corner is at "
                    "sqrt(60^2+16^2) = 62.08 m."),
        "C3_frames_bevbox_gt_r60": n_frames_bevbox_gt_r60,
        "C3_boxes_in_decode_box_beyond_60m_radius": n_bevbox_beyond_r60,
        "C4_see_last_row_of_each_decision_table": True,
        "duplicate_clip_frame_keys": n_dup_keys,
        "frames_with_duplicate_track_id": n_dup_track_in_frame,
        "labelled_CLEAR_frames_agents_empty": n_clear_frames,
        "boxes_missing_cls_field": n_missing_cls,
        "boxes_with_cls_outside_ALL_CLASSES": n_unknown_cls,
    }

    per_clip = {c: {"n_labelled_frames": per_clip_frames[c],
                    "n_boxes": per_clip_boxes[c],
                    "max_agents_all": per_clip_max[c]}
                for c in sorted(clip_ids)}
    # NO_LABEL accounting from the join meta's own per-clip block (the join
    # emits NO LINE for an unlabelled frame, so it cannot be counted from the
    # jsonl alone -- a frame absent is a different state from agents: []).
    no_label = None
    try:
        pc = meta.get("per_clip")
        if isinstance(pc, list):
            tot_f = sum(int(d["n_frames"]) for d in pc)
            tot_l = sum(int(d["n_labelled"]) for d in pc)
            no_label = {"n_frames_in_joined_episodes": tot_f,
                        "n_labelled": tot_l,
                        "n_NO_LABEL_within_joined_episodes": tot_f - tot_l,
                        "n_episodes_with_no_join": len(
                            meta["summary"].get("skipped", {})
                            .get("no_obstacle", []))}
    except Exception as exc:                              # pragma: no cover
        no_label = {"error": repr(exc)}

    out = {
        "task": "per-frame agent-count distribution of the obstacle.offline "
                "join -- the fit for agent_slots.N_QUERIES_DEFAULT",
        "provenance": prov,
        "controls": controls,
        "no_label_accounting": no_label,
        "cut_definitions": {
            "all": "every cuboid in the join, any azimuth, any range",
            "all_r60": "sqrt(cx^2+cy^2) <= 60 m",
            "all_r80": "sqrt(cx^2+cy^2) <= 80 m",
            "infield": "occ == 0 (centre azimuth within +-60 deg)",
            "infield_r40": "occ == 0 and range <= 40 m",
            "infield_r60": "occ == 0 and range <= 60 m",
            "infield_r80": "occ == 0 and range <= 80 m",
            "infield_r100": "occ == 0 and range <= 100 m",
            "infield_fwd60": f"occ == 0 and cx <= {X_FWD_M} m "
                             f"(SlotDecodeRanges.x_fwd_m)",
            "infield_bevbox": f"occ == 0 and 0 <= cx <= {X_FWD_M} and "
                              f"|cy| <= {Y_HALF_M} (GRID_DEFAULT box)",
        },
        "distribution": dist,
        "decision_table": decision,
        "naive_pipeline_audit": {
            "what": "targets_from_join marks EVERY agent valid (no azimuth, "
                    "no range filter), so match_slots keeps the n_queries "
                    "NEAREST of the `all` cut. This prices what those slots "
                    "are actually spent on.",
            "rows": naive,
        },
        "tail_concentration": tail,
        "class_histogram": classes,
        "per_clip": per_clip,
    }

    dest = a.out
    if dest is None:
        here = os.path.dirname(os.path.abspath(__file__))
        dest = os.path.join(here, "raw", "agent_density.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
    print(f"[out] {dest}")

    # ---- console summary --------------------------------------------------
    print(f"\nsource {src}\nmd5 OK vs meta ({want_md5})")
    print(f"C1 occ recompute disagreements: {n_occ_disagree}  "
          f"C3 nesting violations: {n_nest_violation}")
    print(f"frames {n_lines}  boxes {n_boxes_total}  clips {len(clip_ids)}  "
          f"visible_frac {n_visible_total / n_boxes_total:.4f}")
    hdr = ("cut", "n_frames", "n_boxes", "mean", "p50", "p75", "p90", "p95",
           "p99", "p99.9", "max")
    print("\n%-16s %8s %9s %7s %5s %5s %5s %5s %5s %6s %5s" % hdr)
    for k in cuts:
        d = dist[k]
        print("%-16s %8d %9d %7.2f %5d %5d %5d %5d %5d %6d %5d" % (
            k, d["n_frames"], d["n_boxes"], d["mean"], d["p50"], d["p75"],
            d["p90"], d["p95"], d["p99"], d["p99_9"], d["max"]))
    for k, rows in decision.items():
        print(f"\n-- drop table, cut = {k}")
        print("%5s %10s %12s %10s %12s %14s" % (
            "N", "fr_w_drop", "frac_frames", "box_drop", "frac_boxes",
            "nearest_drop_m"))
        for r in rows:
            nd = r["nearest_dropped_m_min"]
            print("%5d %10d %12.4f %10d %12.5f %14s%s" % (
                r["n_queries"], r["n_frames_with_drop"],
                r["frac_frames_with_drop"], r["n_boxes_dropped"],
                r["frac_boxes_dropped"],
                ("%.2f" % nd) if nd is not None else "-",
                "   <- CONTROL" if "_control" in r else ""))
    for k in ("infield", "infield_r60", "infield_bevbox"):
        print(f"\n-- tail concentration, cut = {k} "
              f"(how many CLIPS drive the over-N frames)")
        print("%5s %12s %14s %14s  %s" % ("N", "frames_over", "n_clips",
                                          "top3_share", "top3"))
        for r in tail[k]:
            print("%5d %12d %14d %14s  %s" % (
                r["n_queries"], r["n_frames_over"],
                r["n_clips_contributing"],
                ("%.3f" % r["top3_clip_share"]) if r["top3_clip_share"]
                is not None else "-", r["top3"]))
    print("\n-- NAIVE pipeline (targets_from_join = ALL agents, "
          "match_slots keeps the N nearest)")
    print("%5s %10s %14s %10s %16s %10s" % (
        "N", "kept", "kept_occ1", "frac_occ1", "kept_out_box", "frac_out"))
    for r in naive:
        print("%5d %10d %14d %10.4f %16d %10.4f" % (
            r["n_queries"], r["n_targets_kept"],
            r["n_kept_occ1_out_of_field"], r["frac_kept_out_of_field"],
            r["n_kept_outside_decode_box"],
            r["frac_kept_outside_decode_box"]))
    print("\n-- class histogram (all / infield)")
    for c in list(ALL_CLASSES) + sorted(
            set(classes["all"]["by_class"]) - set(ALL_CLASSES)):
        a_ = classes["all"]["by_class"].get(c, {})
        i_ = classes["infield"]["by_class"].get(c, {})
        print("%-20s %8d (%5.2f%%) %6d clips | infield %8d (%5.2f%%) "
              "%6d clips" % (
                  c, a_.get("n_boxes", 0), 100 * a_.get("frac", 0.0),
                  a_.get("n_clips_present", 0), i_.get("n_boxes", 0),
                  100 * i_.get("frac", 0.0), i_.get("n_clips_present", 0)))
    bad = (n_occ_disagree or n_nest_violation or n_dup_keys
           or not all(controls["C2_matches_inherited"].values()))
    print("\nCONTROLS: " + ("FAIL" if bad else "PASS"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
