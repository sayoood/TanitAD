#!/usr/bin/env python3
"""t1_summary.py — combine several ``t1_eval.py`` arm records into ONE decision
record, with the PAIRED episode-cluster bootstrap ACROSS arms.

WHY THIS IS A SEPARATE TOOL. ``t1_eval.py`` carries the §1.12 byte-close gate;
its roll and its analysis must not grow a cross-arm mode that could disturb
either. This is a pure post-processor: it reads the per-arm JSONs (and, for the
paired contrast, their dumps) and writes ``t1_summary.json``. Nothing here can
change a per-arm number — it can only join, compare, or REFUSE to.

WHAT IT EMITS
-------------
* ``arms`` — per arm, per ARM-KEY (cl/ol/ha): tier stamp, the FULL-SET pooled
  headline metrics with their episode-cluster bootstrap intervals (copied from
  the arm record, not recomputed), the S-reproduction rate, the response lag,
  the event-response ratios, and which of the four families are UNAVAILABLE
  (with their reasons — a missing family is a WORK ITEM, never a pass).
* ``within_arm_T0_vs_T1`` — each record's own ``paired_decision_grade`` block
  (cl − ol on the same windows): the action-echo read that made §1.12
  admissible.
* ``cross_arm_paired`` — ⭐ the thing no single record can produce: for two arms
  rolled on the SAME grid, the PAIRED episode-cluster bootstrap of arm B minus
  arm A per window (``taniteval.ci.paired_episode_cluster_bootstrap``). Paired,
  never two intervals combined in quadrature (CLAUDE.md).

⛔ THE GRID CHECK IS A REFUSAL, NOT A WARNING. Two dumps are joinable only if
they have the same episode files, the same per-episode window counts AND
bit-identical GT (``g``) arrays. A positional join across different grids
produces a plausible number, which is worse than an error.

Usage::

  python3 t1_summary.py \
      --arm v5f-30k=/workspace/experiments/t1-v58f/t1_v5f_30k.json \
      --arm stage-a-repaired=/workspace/experiments/t1-v58f/t1_stage_a.json \
      --paired stage-a-repaired,v5f-30k \
      --out /workspace/experiments/t1-v58f/t1_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # <repo>/taniteval/tools
_TE_PARENT = os.path.dirname(_HERE)                     # <repo>/taniteval
_REPO = os.path.dirname(_TE_PARENT)                     # <repo>
for _pth in (os.path.join(_REPO, "stack"), _TE_PARENT):
    if os.path.isdir(_pth) and _pth not in sys.path:
        sys.path.insert(0, _pth)

#: metrics carried per arm-key from each record's own interval block.
HEADLINE = ("ade_dense_m", "fde_last_m", "LON_speed_mae_mps",
            "LON_along_mae_m", "LAT_cross_mae_m", "LAT_heading_mae_deg")

_ESTIMATOR = ("point estimates are the arm records' FULL-SET pooled means; "
              "intervals are the episode-cluster bootstrap (taniteval/ci.py) "
              "and cross-arm deltas the PAIRED version on the same windows. "
              "⛔ overlapping_holdout_se is used nowhere — it biases the POINT "
              "ESTIMATE, not only the interval.")


def _p(*a):
    print(*a, flush=True)


def load_record(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def dump_files_of(rec: dict, path: str) -> list:
    """Sorted ``ep*.npz`` of the record's dump dir (the paired join's input)."""
    d = rec.get("dump_dir")
    if not d or not os.path.isdir(d):
        raise SystemExit(
            f"[t1-summary] {path} names dump_dir={d!r}, which is not a "
            f"directory on this host — the cross-arm PAIRED bootstrap needs "
            f"the per-window dumps. Run --paired where the dumps live, or drop "
            f"--paired (the per-arm intervals still summarise).")
    files = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    if not files:
        raise SystemExit(f"[t1-summary] no ep*.npz under {d}")
    return files


def per_window_components(files, arm_key: str, dt: float):
    """``(ade [N], speed_mae [N], eids [N], gt_hash)`` for one arm key.

    Geometry comes from ``taniteval.four_families._seq_geometry`` — the SAME
    machinery ``t1_eval.analyze`` scores with, imported rather than re-derived
    (the eval_four_families rule)."""
    import torch

    from taniteval import four_families as ff
    ade, sp, eids, gts = [], [], [], []
    for f in files:
        with np.load(f) as d:
            if arm_key not in d.files:
                raise SystemExit(
                    f"[t1-summary] dump {f} has no arm {arm_key!r} "
                    f"(keys {list(d.files)}) — refusing to pair an arm that "
                    f"was not rolled.")
            G = d["g"][..., :2].astype(np.float64)
            P = d[arm_key][..., :2].astype(np.float64)
        if P.shape != G.shape:
            raise SystemExit(f"[t1-summary] {f}: arm {arm_key} {P.shape} != "
                             f"GT {G.shape}")
        gt_t = torch.as_tensor(G).float()
        pr_t = torch.as_tensor(P).float()
        ade.append(torch.linalg.norm(pr_t - gt_t, dim=-1).mean(1).numpy())
        sp.append((ff._seq_geometry(pr_t, dt)["speed"]
                   - ff._seq_geometry(gt_t, dt)["speed"]).abs().mean(1).numpy())
        eids += [os.path.splitext(os.path.basename(f))[0]] * G.shape[0]
        gts.append(G.astype(np.float32))
    return (np.concatenate(ade), np.concatenate(sp), eids,
            np.concatenate(gts))


def arm_block(rec: dict) -> dict:
    """Per-arm-key headline rows, copied (never recomputed) from the record."""
    out = {}
    for key, blk in rec.get("arms", {}).items():
        fam = blk.get("four_families", {})
        ints = blk.get("intervals", {}).get("metrics", {})
        s = blk.get("s_curve", {}).get("masked", {})
        resp = blk.get("response", {})
        out[key] = {
            "tier": blk.get("tier"),
            "tier_note": blk.get("tier_note"),
            "metrics": {m: ints[m] for m in HEADLINE if m in ints},
            "estimator": blk.get("intervals", {}).get("estimator"),
            "n_windows": blk.get("intervals", {}).get("n"),
            "s_curve_masked": ({"rate": s.get("rate"), "n_s": s.get("n_s_windows"),
                                "ci": s.get("ci")} if "rate" in s
                               else {"status": s.get("status"),
                                     "reason": s.get("reason")}),
            "lag_accel_s": blk.get("lag", {}).get("lag_accel_s_mean",
                                                  blk.get("lag", {})
                                                  .get("status")),
            "response_ratio": {
                kind: (resp.get(kind, {}).get("response_ratio")
                       if "response_ratio" in resp.get(kind, {})
                       else resp.get(kind, {}).get("status"))
                for kind in ("decel", "accel")},
            "families_unavailable": fam.get("_families_unavailable"),
            "families_unavailable_reasons": {
                fk: fam[fk].get("reason") for fk in
                ("longitudinal", "lateral", "tactical", "strategic")
                if isinstance(fam.get(fk), dict)
                and fam[fk].get("status") == "UNAVAILABLE"},
            "sel_gap": blk.get("sel_gap", {}).get(
                "gap", blk.get("sel_gap", {}).get("status")),
        }
    return out


def cross_arm_paired(recs: dict, pairs, arm_keys, *, n_boot: int, seed: int,
                     dt: float) -> dict:
    """Paired episode-cluster bootstrap of ``b - a`` per arm key, on the dumps.

    Refuses any join whose grids differ — same episode count, same per-episode
    window counts, bit-identical GT."""
    from taniteval import ci as _ci
    out = {}
    cache = {}
    for b_name, a_name in pairs:
        for nm in (a_name, b_name):
            if nm not in recs:
                raise SystemExit(f"[t1-summary] --paired names {nm!r}, which "
                                 f"is not among --arm {sorted(recs)}")
        key = f"{b_name}_minus_{a_name}"
        out[key] = {"direction": f"{b_name} - {a_name}",
                    "estimator": "paired_episode_cluster_bootstrap",
                    "n_boot": n_boot, "seed": seed}
        for ak in arm_keys:
            try:
                comps = {}
                for nm in (a_name, b_name):
                    ck = (nm, ak)
                    if ck not in cache:
                        cache[ck] = per_window_components(
                            dump_files_of(recs[nm], nm), ak, dt)
                    comps[nm] = cache[ck]
            except SystemExit as ex:
                out[key][ak] = {"status": "UNAVAILABLE", "reason": str(ex)}
                continue
            ad_a, sp_a, eid_a, g_a = comps[a_name]
            ad_b, sp_b, eid_b, g_b = comps[b_name]
            if eid_a != eid_b or not np.array_equal(g_a, g_b):
                out[key][ak] = {
                    "status": "REFUSED",
                    "reason": ("the two dumps are NOT on the same grid "
                               f"(n_windows {len(eid_a)} vs {len(eid_b)}, GT "
                               f"identical: {np.array_equal(g_a, g_b)}) — a "
                               "positional join across grids scores one arm "
                               "against another's traffic. Re-roll both on the "
                               "same corpus/episodes/stride; do NOT truncate."),
                    "n_a": len(eid_a), "n_b": len(eid_b)}
                continue
            out[key][ak] = {
                "tier": (recs[b_name]["arms"][ak].get("tier")),
                "n_windows": int(ad_a.size),
                "n_episodes": int(len(set(eid_a))),
                "ade_dense_m": _ci.paired_episode_cluster_bootstrap(
                    ad_b, ad_a, eid_a, n_boot=n_boot, seed=seed),
                "LON_speed_mae_mps": _ci.paired_episode_cluster_bootstrap(
                    sp_b, sp_a, eid_a, n_boot=n_boot, seed=seed),
                "_read": ("negative delta = the FIRST-named arm is better "
                          "(lower error) on these windows"),
            }
    return out


def build_summary(recs: dict, paths: dict, *, pairs=(), n_boot=2000, seed=0,
                  dt=0.1, arm_keys=("cl", "ol", "ha")) -> dict:
    present = sorted({k for r in recs.values() for k in r.get("arms", {})})
    keys = [k for k in arm_keys if k in present]
    rec = {
        "tool": "taniteval/tools/t1_summary.py",
        "inputs": {nm: {"json": paths[nm], "ckpt": recs[nm].get("ckpt"),
                        "dump_dir": recs[nm].get("dump_dir"),
                        "corpus_key": recs[nm].get("corpus_key"),
                        "decoder": recs[nm].get("decoder"),
                        "rollout_path": recs[nm].get("rollout_path"),
                        "n_episodes": recs[nm].get("n_episodes"),
                        "n_windows": recs[nm].get("n_windows"),
                        "grid": (recs[nm].get("rollout_provenance") or {})
                        .get("grid")}
                   for nm in recs},
        "arm_keys_present": present,
        "_tier_doctrine": ("EVAL_DOCTRINE.md — T1 (action-closed loop, 'cl'/"
                           "'ha') is the PRIMARY tier; T0 ('ol', teacher-"
                           "forced) supports prediction/attribution claims "
                           "ONLY and is NEVER quotable as driving performance."),
        "_estimator": _ESTIMATOR,
        "_binding_four_families": (
            "Sayed 2026-08-02 — LONGITUDINAL + LATERAL + TACTICAL + STRATEGIC "
            "in ADDITION to ADE, per-family. A T1 dump traverses no decision "
            "heads, so TACTICAL/STRATEGIC report UNAVAILABLE with their reason "
            "and n: a WORK ITEM, not a pass. Distance-keeping needs a lead "
            "block (tools/build_lead_block.py) attached at t1_eval time."),
        "_evidence_class": "MEASURED (ours; artifacts = the input JSONs+dumps)",
        "arms": {nm: arm_block(recs[nm]) for nm in recs},
        "within_arm_T0_vs_T1": {
            nm: recs[nm].get("paired_decision_grade", {}) for nm in recs},
    }
    if pairs:
        rec["cross_arm_paired"] = cross_arm_paired(
            recs, pairs, keys, n_boot=n_boot, seed=seed, dt=dt)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=JSON",
                    help="an arm record written by t1_eval.py (repeatable)")
    ap.add_argument("--paired", action="append", default=[], metavar="B,A",
                    help="cross-arm paired bootstrap of B minus A (repeatable)")
    ap.add_argument("--out", required=True, help="t1_summary.json (a FILE)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.1)
    a = ap.parse_args(argv)

    paths, recs = {}, {}
    for spec in a.arm:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"[t1-summary] --arm expects NAME=JSON, got {spec!r}")
        if not os.path.exists(path):
            sys.exit(f"[t1-summary] no such arm record: {path}")
        paths[name] = path
        recs[name] = load_record(path)
    pairs = []
    for spec in a.paired:
        b, _, aa = spec.partition(",")
        if not aa:
            sys.exit(f"[t1-summary] --paired expects B,A, got {spec!r}")
        pairs.append((b.strip(), aa.strip()))

    rec = build_summary(recs, paths, pairs=pairs, n_boot=a.n_boot,
                        seed=a.seed, dt=a.dt)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    _p(f"[out] {a.out}")
    for nm, blk in rec["arms"].items():
        for ak, row in blk.items():
            m = row["metrics"].get("ade_dense_m", {})
            _p(f"  {nm:20s} {ak:3s} tier={row['tier']}  "
               f"ade_dense={m.get('mean')} [{m.get('lo')}, {m.get('hi')}]  "
               f"S(masked)={row['s_curve_masked'].get('rate', 'UNAVAILABLE')}")
    for key, blk in rec.get("cross_arm_paired", {}).items():
        for ak in ("cl", "ol", "ha"):
            row = blk.get(ak)
            if isinstance(row, dict) and "ade_dense_m" in row:
                d = row["ade_dense_m"]
                _p(f"  PAIRED {key} [{ak}] ade delta {d['delta']} "
                   f"[{d['lo']}, {d['hi']}] separated={d['separated']}")
            elif isinstance(row, dict):
                _p(f"  PAIRED {key} [{ak}] {row.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
