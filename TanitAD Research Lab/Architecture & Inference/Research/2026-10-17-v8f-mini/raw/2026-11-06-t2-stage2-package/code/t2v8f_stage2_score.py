"""Dev-box scorer of the stage-2 pull (AGENT I): the pooled T2 score of `t2v8f_run.py score` PLUS every row per START CLASS (launch / moving), never pooled across the classes.

    python -I t2v8f_stage2_score.py --root PULL/rollouts --out PULL/SCORE2.json [--sources stage2/SCENE_SOURCES.json] [--tracks IDX=path ...] [--md PULL/SCORE2.md]

Start class (the Master Mind's ruling, 2026-11-06): LAUNCH = the logged speed at f0 is < 0.5 m/s, MOVING = >= 0.5 m/s. The speed is measured from the LOGGED track in each cell's `scene_<idx>.npz`
(backward difference of the 10 Hz rig positions at f0: the policy's own `set_t0` spelling), and is cross-checked against the record's own `v0` (they must agree to 1e-6).
Why per class (AUDIT_NUREC_SCORE section 5): a lateral / heading mean over launch and moving starts (10 deg and 61 deg for P10 on the two stage-1 starts) describes no start;
launch starts carry an unstable executed heading (a stress test of the executor, not a vision reading). Every row is reported per class with its n_valid; the pooled file is for the card's M1 contrast.
Per class and per seed: the paired scene-cluster bootstrap X1 = C0(T2) - P10(T2) and X2 = C0(T2) - C0(RR) on the LATERAL rows (the reader's own `paired`, B 2000, seed 0).
Evidence class of anything this writes: MEASURED (ours, T2 re-perception rollouts of the stage-2 pull); a class with n < 5 starts is a description, not an estimate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
LAUNCH_V0_MPS = 0.5
CLASSES = ("launch", "moving")
LATERAL_LABELS = ("heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)")
MAIN_LABELS = ("ADE6", "speed MAE", "M1 crawl share") + LATERAL_LABELS


def logged_v0(gt_T, f0):
    """Logged speed at tick f0, m/s: the backward difference over the logged rows <= f0, |p[f0] - p[f0-1]| / 0.1 (the policy's `set_t0` seeds v0 exactly so: it reads nothing after f0)."""
    import numpy as np
    a, b = np.asarray(gt_T[f0 - 1], np.float64), np.asarray(gt_T[f0], np.float64)
    return float(np.linalg.norm(b[:2, 3] - a[:2, 3]) / 0.1)


def start_class(v0):
    return "launch" if v0 < LAUNCH_V0_MPS else "moving"


def load_groups(root):
    """-> (groups {(arm, tcond, agents): [record]}, scene_npz {idx: path})."""
    groups, npz = {}, {}
    for f in sorted(glob.glob(f"{root}/**/records.jsonl", recursive=True)):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            groups.setdefault((r["arm"], r["tcond"], r["agents"]), []).append(r)
        for z in glob.glob(os.path.dirname(f) + "/scene_*.npz"):
            npz[int(os.path.basename(z)[6:-4])] = z
    return groups, npz


def tracks_map(sources, tracks):
    """{scene idx: sequence_tracks.json path} from SCENE_SOURCES.json (dev F: locations) and/or explicit IDX=path arguments."""
    out = {}
    if sources:
        for s in json.load(open(sources, encoding="utf-8")):
            if s["rel"] == "extracted/sequence_tracks.json":
                out[int(s["idx"])] = s["source"]
    for t in tracks:
        k, p = t.split("=", 1)
        out[int(k)] = p
    return out


def score(root, out, sources=None, tracks=(), md=None, pooled_out=None):
    import numpy as np
    import t2v8f_lead as L
    import t2v8f_run as R
    import t2v8f_score as S
    tr = tracks_map(sources, tracks)
    pooled = pooled_out or out.replace(".json", "_POOLED.json")
    pargs = ["--root", root, "--out", pooled]
    for k, v in tr.items():
        pargs += ["--tracks", f"{k}={v}"]
    R.score_main(pargs)                                                                           # the pooled T2 score, exactly as stage 1's MM ran it
    groups, npz = load_groups(root)
    scenes = {i: L.SceneLog(np.load(z)["gt_T"], np.load(z)["gt_ts_us"], L.load_tracks(tr[i]) if i in tr else None) for i, z in npz.items()}
    gts = {i: np.load(z)["gt_T"] for i, z in npz.items()}
    res = {"tier": S.TIER, "purpose": "stage-2 deciding rollouts, scored per start class", "rule": f"launch = logged v0 at f0 < {LAUNCH_V0_MPS} m/s, moving = >= {LAUNCH_V0_MPS} m/s",
           "pooled_file": os.path.basename(pooled), "classes": {}, "starts": {}, "checks": {}}
    max_gap, by_cls = 0.0, {c: {} for c in CLASSES}
    for key, recs in sorted(groups.items()):
        recs.sort(key=lambda r: (r["scene"], r["f0"]))
        cl = []
        for r in recs:
            v0 = logged_v0(gts[int(r["scene"])], int(r["f0"]))
            max_gap = max(max_gap, abs(v0 - float(r["v0"])))
            cl.append(start_class(v0))
            res["starts"][f"{r['scene_id'][:8]}|f0={r['f0']}"] = {"scene_idx": int(r["scene"]), "v0_mps": round(v0, 4), "class": cl[-1]}
        for c in CLASSES:
            sub = [r for r, k in zip(recs, cl) if k == c]
            if not sub:
                continue
            rows, missing, info = L.rows_for_records(sub, scenes, "|".join(key) + "|" + c)
            by_cls[c][key] = (rows, info)
            res["classes"].setdefault(c, {})["|".join(key)] = {"n_starts": len(sub), "n_scenes": len({int(r["scene"]) for r in sub}),
                                                              "mean": {k: (float(np.nanmean(v)) if np.isfinite(v).any() else None) for k, v in rows.items()},
                                                              "n_valid": S.n_valid(rows), "missing": missing, "family": {k: S.FAMILY.get(k) for k in rows}}
    res["checks"] = {"max_abs_v0_logged_minus_record_mps": max_gap, "ok": max_gap < 1e-6}
    for c in CLASSES:
        for s in (0, 1):
            kt, kp, kr = (f"C0_s{s}", "T2", "objects"), ("P10", "T2", "objects"), (f"C0_s{s}", "RR", "objects")
            if kt in by_cls[c] and kp in by_cls[c] and kr in by_cls[c]:
                out_c = {}
                try:
                    S.pair_check(by_cls[c][kt][1]["keys"], by_cls[c][kp][1]["keys"])
                    S.pair_check(by_cls[c][kt][1]["keys"], by_cls[c][kr][1]["keys"])
                    eid = by_cls[c][kt][1]["eid"]
                    for lab in LATERAL_LABELS + ("M1 crawl share",):
                        out_c[lab] = {"X1": R.json_safe(S.scene_contrast(by_cls[c][kt][0][lab], by_cls[c][kp][0][lab], eid)),
                                      "X2": R.json_safe(S.scene_contrast(by_cls[c][kt][0][lab], by_cls[c][kr][0][lab], eid))}
                except Exception as e:                                                          # noqa: BLE001
                    out_c = {"error": repr(e)[:300]}
                res.setdefault("contrasts", {}).setdefault(c, {})[f"seed{s}"] = out_c
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    if md:
        write_md(res, md)
    return res


def write_md(res, path):
    L = ["# Stage-2 score per start class", "", f"Rule: {res['rule']}. Tier: {res['tier']}. Never pooled across classes. A class with n < 5 starts is a description, not an estimate.", ""]
    n = {c: max((g["n_starts"] for g in res["classes"].get(c, {}).values()), default=0) for c in CLASSES}
    for c in CLASSES:
        L += [f"## {c} (n starts per group: {n[c]})", "", "| group | n | " + " | ".join(MAIN_LABELS) + " |", "|---|---|" + "---|" * len(MAIN_LABELS)]
        for k, g in sorted(res["classes"].get(c, {}).items()):
            L.append(f"| {k.replace('|', '/')} | {g['n_starts']} | " + " | ".join("n/a" if g["mean"].get(lab) is None else f"{g['mean'][lab]:.3f}" for lab in MAIN_LABELS) + " |")
        L.append("")
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default=None)
    ap.add_argument("--tracks", action="append", default=[])
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    r = score(a.root, a.out, a.sources, a.tracks, a.md)
    print(json.dumps({"classes": {c: len(g) for c, g in r["classes"].items()}, "starts": len(r["starts"]), "checks": r["checks"]}))
    return 0 if r["checks"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
