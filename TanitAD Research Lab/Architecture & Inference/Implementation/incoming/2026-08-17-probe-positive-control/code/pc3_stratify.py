"""PC3 — TASK 2: is the F-18 failure a REPRESENTATION problem or a RESOLUTION
problem? Stratify the EXISTING probe results by the GT lead's RANGE and by its
APPARENT PIXEL WIDTH.

⛔ NO NEW PROBE. This file imports ``sp2_probe`` (byte-identical to the parity
run's, md5 aabbee36fce5f164d47a555fad369cbd) and calls ITS ``evaluate`` on the
head that run BANKED. The split, the C-SHUF / C-SHUF-XEP permutations, the
C-CONST / C-EPMEAN construction and the paired window set are rebuilt by the
same code paths, and the rebuilt HEADLINE is asserted against the banked
``results_*.json`` before any stratum is printed. If it does not reproduce, the
run FAILS LOUD and prints nothing.

⚠️ WHAT THE LABEL ACTUALLY CONTAINS — stated because the brief assumed more.
``lead130_agents.jsonl`` records are ``{cx, cy, yaw, l, w, occ, track_id, cls}``
— **EGO-FRAME 3-D GEOMETRY ONLY. There is NO image-space box in this join.**
So an image extent cannot be read off; it must be DERIVED from the camera
geometry, and it is therefore ESTIMATED, not MEASURED:

    apparent width (px)  ~=  f_ref * w_obj / cx

with ``f_ref = 305.5774907364391`` px, the value the cache meta records for the
256x640 120-deg cylindrical field this trunk was trained on (small-angle: an
object of physical width ``w`` at distance ``cx`` subtends ``w/cx`` rad, and a
cylindrical projection maps angle to column linearly at exactly ``f_ref`` px/rad).
The ViT patch is **16 px** (``token_grid`` [16, 40] over 256x640), so the
patch-fraction strata below are cut at 8 px (half a patch) and 16 px (one patch).

⚠️ HEIGHT IS NOT IN THE JOIN, so pixel AREA is not derivable and is not
reported. Width is, and it is the axis the "less than one patch" worry is about.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp2_probe as SP                                          # noqa: E402
from taniteval.ci import (episode_cluster_bootstrap,             # noqa: E402
                          paired_episode_cluster_bootstrap)
from tanitad.models.agent_slots import (AgentSlotDecoder,        # noqa: E402
                                        SlotDecodeRanges)

F_REF_PX = 305.5774907364391      # cache meta `geometry_binding.frame.f_ref`
PATCH_PX = 16                     # token_grid [16,40] over 256x640


def rebuild(cache, head_pt, split_json, n_queries, seed, device):
    """Reproduce sp2's eval state exactly, from the BANKED head."""
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    decl = json.loads(Path(split_json).read_text("utf-8"))
    eval_clips, train_clips = set(decl["eval_clips"]), set(decl["train_clips"])
    ev_idx = [i for i, r in enumerate(rows) if r["clip_id"] in eval_clips]
    tr_idx = [i for i, r in enumerate(rows) if r["clip_id"] in train_clips]
    ev_eid = np.array([rows[i]["episode_uid"] for i in ev_idx])

    gt_ev = np.array([SP.gt_lead_gap(rows[i]["agents"])
                      if SP.gt_lead_gap(rows[i]["agents"]) is not None
                      else np.nan for i in ev_idx])
    gt_tr = np.array([g for g in (SP.gt_lead_gap(rows[i]["agents"])
                                  for i in tr_idx) if g is not None])
    has_gt = ~np.isnan(gt_ev)
    const_m = float(np.median(gt_tr))

    # --- the two permutations, reproduced with sp2's own RNG discipline ------
    rng = np.random.default_rng(seed + 7)
    by_ep: dict[int, list[int]] = {}
    for pos, i in enumerate(ev_idx):
        by_ep.setdefault(rows[i]["episode_uid"], []).append(pos)
    shuf_pos = np.arange(len(ev_idx))
    for _ep, poss in by_ep.items():
        perm = list(poss)
        if len(perm) > 1:
            p = rng.permutation(len(perm))
            for k in range(len(perm)):
                if p[k] == k and len(perm) > 1:
                    p[k], p[(k + 1) % len(perm)] = p[(k + 1) % len(perm)], p[k]
            for k, src in enumerate(p):
                shuf_pos[poss[k]] = poss[src]
    shuf_idx = [ev_idx[j] for j in shuf_pos]
    ep_keys = sorted(by_ep)
    xep_pos = np.arange(len(ev_idx))
    if len(ep_keys) > 1:
        for n_e, ep in enumerate(ep_keys):
            src_poss = by_ep[ep_keys[(n_e + 1) % len(ep_keys)]]
            for k, pos in enumerate(by_ep[ep]):
                xep_pos[pos] = src_poss[k % len(src_poss)]
    xep_idx = [ev_idx[j] for j in xep_pos]

    # --- the banked head ----------------------------------------------------
    d_mem = int(rows[0]["cells"].shape[-1]); n_mem = int(rows[0]["cells"].shape[0])
    head = AgentSlotDecoder(d_mem, n_mem, n_queries=int(n_queries),
                            d_model=256, depth=3, n_heads=8,
                            ranges=SlotDecodeRanges(),
                            enforce_band=False).to(device)
    sd = torch.load(head_pt, map_location="cpu", weights_only=False)
    head.load_state_dict(sd["head"])

    g, e, _p, _v, _o = SP.evaluate(head, rows, ev_idx, "cells", device)
    gs, es, *_ = SP.evaluate(head, rows, ev_idx, "cells", device,
                             shuffle_within_ep=shuf_idx)
    gx, ex, *_ = SP.evaluate(head, rows, ev_idx, "cells", device,
                             shuffle_within_ep=xep_idx)

    ep_of = np.array([rows[i]["clip_id"] for i in ev_idx])
    epmean = np.full(len(ev_idx), const_m, dtype=np.float64)
    for _c in np.unique(ep_of):
        m = (ep_of == _c) & (~np.isnan(gt_ev))
        pos = np.nonzero(m)[0]
        if pos.size == 0:
            continue
        tot = float(np.sum(gt_ev[pos]))
        for k in pos:
            epmean[k] = ((tot - gt_ev[k]) / (pos.size - 1)) if pos.size > 1 \
                else const_m

    pred = {"cells": g, "cells__C-SHUF": gs, "cells__C-SHUF-XEP": gx,
            "C-CONST": np.full(len(ev_idx), const_m), "C-EPMEAN": epmean}
    emit = {"cells": e, "cells__C-SHUF": es, "cells__C-SHUF-XEP": ex,
            "C-CONST": np.ones(len(ev_idx), bool),
            "C-EPMEAN": np.ones(len(ev_idx), bool)}
    common = has_gt.copy()
    for k in pred:
        common &= emit[k].astype(bool)
    return dict(rows=rows, meta=meta, ev_idx=ev_idx, ev_eid=ev_eid,
                gt_ev=gt_ev, common=common, pred=pred, const_m=const_m,
                n_gt=int(has_gt.sum()))


def lead_geometry(rows, ev_idx, common):
    """Per-scored-window (cx, apparent px width, class) of the GT LEAD."""
    cxs, pxw, cls = [], [], []
    for k, i in enumerate(ev_idx):
        if not common[k]:
            continue
        r = rows[i]
        j = SP.gt_lead_row(r["agents"])
        ag = r["agents"][j]
        cx = float(ag[0]); w = float(ag[4])
        cxs.append(cx)
        pxw.append(F_REF_PX * w / max(cx, 1e-6))
        c = r["classes"][j] if r["classes"] is not None else "?"
        cls.append(c)
    return np.asarray(cxs), np.asarray(pxw), np.asarray(cls, dtype=object)


def stratum_row(err_arm, err_ctl, eid, name, sel, n_boot,
                pred_arm=None, gt=None):
    n = int(sel.sum())
    if n == 0:
        return {"stratum": name, "n_windows": 0, "n_clusters": 0,
                "note": "EMPTY — no claim"}
    a, b, e = err_arm[sel], err_ctl[sel], eid[sel]
    nc = int(len(np.unique(e)))
    extra = {}
    if pred_arm is not None and gt is not None:
        # ⭐ THE DIAGNOSTIC THE HEADLINE HIDES: a head that emits a near-constant
        # value looks "wrong by X m" without that being a perception statement.
        extra = {"pred_mean_m": round(float(pred_arm[sel].mean()), 3),
                 "pred_sd_m": round(float(pred_arm[sel].std()), 3),
                 "gt_mean_m": round(float(gt[sel].mean()), 3),
                 "gt_sd_m": round(float(gt[sel].std()), 3)}
    arm_ci = episode_cluster_bootstrap(a, e, n_boot=n_boot)
    ctl_ci = episode_cluster_bootstrap(b, e, n_boot=n_boot)
    d = paired_episode_cluster_bootstrap(a, b, e, n_boot=n_boot)
    return {"stratum": name, "n_windows": n, "n_clusters": nc,
            "arm_err_m": round(float(a.mean()), 4),
            "arm_ci": [arm_ci["lo"], arm_ci["hi"]],
            "ctl_err_m": round(float(b.mean()), 4),
            "ctl_ci": [ctl_ci["lo"], ctl_ci["hi"]],
            "K1_delta": d["delta"], "K1_lo": d["lo"], "K1_hi": d["hi"],
            "K1_separated": d["separated"],
            **extra,
            "_weak_n": bool(nc < 10)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--expect-json", required=True,
                    help="the BANKED sp2 result this must reproduce")
    ap.add_argument("--n-queries", type=int, default=74)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="v6F-SW-30k@11250")
    a = ap.parse_args(argv)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    st = rebuild(a.cache, a.head, a.split_json, a.n_queries, a.seed, device)
    rows, ev_idx, common = st["rows"], st["ev_idx"], st["common"]
    eid = st["ev_eid"][common]
    gt = st["gt_ev"][common]
    err = {k: np.abs(v[common] - gt) for k, v in st["pred"].items()}

    # ---- ⛔ THE REPRODUCTION GATE ------------------------------------------
    exp = json.loads(Path(a.expect_json).read_text("utf-8"))
    got = {"n_scored_windows": int(common.sum()),
           "n_bootstrap_clusters": int(len(np.unique(eid))),
           "arm_err": round(float(err["cells"].mean()), 4),
           "c_const": round(float(err["C-CONST"].mean()), 4)}
    want = {"n_scored_windows": exp["n_scored_windows"],
            "n_bootstrap_clusters": exp["n_bootstrap_clusters"],
            "arm_err": round(float(
                exp["per_arm"]["cells"]["lead_gap_abs_err_m"]["mean"]), 4),
            "c_const": round(float(
                exp["per_arm"]["C-CONST"]["lead_gap_abs_err_m"]["mean"]), 4)}
    if got != want:
        raise SystemExit(f"[pc3] ⛔ REPRODUCTION FAILED — refusing to stratify.\n"
                         f"  rebuilt {got}\n  banked  {want}")
    print(f"[pc3] reproduction gate PASSED: {got}", flush=True)

    cx, pxw, cls = lead_geometry(rows, ev_idx, common)
    assert cx.size == common.sum()

    out = {"_evidence_class": "MEASURED (ours; re-evaluation of the BANKED "
                              "parity head on the BANKED parity cache)",
           "eval_tier": "T0-DIAGNOSTIC",
           "arm": a.label, "seed": a.seed,
           "reproduction_gate": {"rebuilt": got, "banked": want,
                                 "expect_json": str(a.expect_json)},
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
           "n_boot": a.n_boot,
           "forbidden": "overlapping_holdout_se",
           "px_width_model": {
               "formula": "f_ref * w_obj / cx", "f_ref_px": F_REF_PX,
               "patch_px": PATCH_PX,
               "_evidence_class": "ESTIMATED (geometric derivation — the join "
                                  "carries NO image-space box; height is absent "
                                  "so pixel AREA is not derivable)"},
           "lead_geometry": {
               "cx_m": {"mean": round(float(cx.mean()), 3),
                        "median": round(float(np.median(cx)), 3),
                        "p10": round(float(np.percentile(cx, 10)), 3),
                        "p90": round(float(np.percentile(cx, 90)), 3)},
               "px_width": {"mean": round(float(pxw.mean()), 3),
                            "median": round(float(np.median(pxw)), 3),
                            "p10": round(float(np.percentile(pxw, 10)), 3),
                            "p90": round(float(np.percentile(pxw, 90)), 3),
                            "frac_below_one_patch":
                                round(float((pxw < PATCH_PX).mean()), 4),
                            "frac_below_half_patch":
                                round(float((pxw < PATCH_PX / 2).mean()), 4)},
               "class_counts": {c: int((cls == c).sum())
                                for c in sorted(set(cls.tolist()))}},
           "strata": {}}

    cuts_cx = [("cx<10m", cx < 10), ("10<=cx<20m", (cx >= 10) & (cx < 20)),
               ("20<=cx<=30m", cx >= 20)]
    cuts_px = [("pxw<8 (<0.5 patch)", pxw < 8),
               ("8<=pxw<16 (0.5-1 patch)", (pxw >= 8) & (pxw < 16)),
               ("pxw>=16 (>=1 patch)", pxw >= 16)]
    for fam, cuts in (("range", cuts_cx), ("apparent_px_width", cuts_px)):
        out["strata"][fam] = {
            "vs_C-CONST": [stratum_row(err["cells"], err["C-CONST"], eid,
                                       nm, sel, a.n_boot,
                                       st["pred"]["cells"][common], gt)
                           for nm, sel in cuts],
            "vs_C-EPMEAN": [stratum_row(err["cells"], err["C-EPMEAN"], eid,
                                        nm, sel, a.n_boot)
                            for nm, sel in cuts]}
    Path(a.out).write_text(json.dumps(out, indent=1), "utf-8")
    print(json.dumps(out["strata"]["range"]["vs_C-CONST"], indent=1))
    print(json.dumps(out["strata"]["apparent_px_width"]["vs_C-CONST"], indent=1))
    print(f"[pc3] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
