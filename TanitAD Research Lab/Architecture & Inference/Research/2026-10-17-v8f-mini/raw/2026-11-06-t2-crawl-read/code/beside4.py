"""Beside table (NOT deciding): every row of the T1 reader, C0_s{0,1}(T2) - P10(T2), paired scene-cluster bootstrap (t2v8f_score.scene_contrast, B 2000 seed 0), over all included launch scenes."""
import json, sys
import numpy as np
T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f"
sys.path.insert(0, T + "/code")
import t2v8f_lead as L, t2v8f_score as S, t2v8f_stage2_score as SC
root, src = T + "/stage2/pull_box2_run2/pull2/rollouts", T + "/stage2/SCENE_SOURCES.json"
tr = SC.tracks_map(src, ())
groups, npz = SC.load_groups(root)
scenes = {i: L.SceneLog(np.load(z)["gt_T"], np.load(z)["gt_ts_us"], L.load_tracks(tr[i]) if i in tr else None) for i, z in npz.items()}
built = {}
for k in [("P10", "T2", "objects"), ("C0_s0", "T2", "objects"), ("C0_s1", "T2", "objects")]:
    recs = sorted(groups[k], key=lambda r: (r["scene"], r["f0"]))
    built[k[0]] = L.rows_for_records(recs, scenes, "|".join(k))
out = {"tier": S.TIER, "purpose": "beside, NOT deciding (PREREG_T2_CRAWL section 3: the four families of Read A beside the M1 read)", "estimator": "paired scene-cluster bootstrap, B 2000 seed 0 (t2v8f_score.scene_contrast)", "rows": {}}
rp, _, ip = built["P10"]
for s in (0, 1):
    rc, _, ic = built[f"C0_s{s}"]
    S.pair_check(ic["keys"], ip["keys"])
    for lab in rc:
        if lab not in rp:
            continue
        c = S.scene_contrast(rc[lab], rp[lab], ic["eid"])
        out["rows"].setdefault(lab, {"family": S.FAMILY.get(lab), "P10_mean": float(np.nanmean(rp[lab])) if np.isfinite(rp[lab]).any() else None})[f"seed{s}"] = {k: c.get(k) for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")} | {"C0_mean": float(np.nanmean(rc[lab])) if np.isfinite(rc[lab]).any() else None}
json.dump(out, open(T + "/stage2/BESIDE4_C0_vs_P10.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
for lab, v in out["rows"].items():
    f = lambda d: f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{' *' if d['separated'] else ''}" if d.get("delta") is not None else "n/a"
    print(f"{lab:28s} {str(v['family']):22s} P10={v['P10_mean'] if v['P10_mean'] is None else round(v['P10_mean'],3)} | s0 {f(v['seed0'])} | s1 {f(v['seed1'])} | n={v['seed0'].get('n_windows')}/{v['seed0'].get('n_episodes')}")
