"""collect_smoke_r8.py -- distil the CPU smoke runs of the refcv8 battery kit into ONE small record (no raw clip ids).

    python collect_smoke_r8.py <smoke root E:/refcv8_eval_kit/smoke_battery> <out json>

Reads `smoke_unc2` (standalone clock, banked baselines), `smoke_cor` (corrected clock, no baselines, os-only at every seed) and
`perc8` (the perception pass) and records WHAT RAN, how long it took, and the assertions that matter: GT / model-free arms
bit-identical to refcv7-50,400 on the standalone clock; pose-sync attached in the corrected child; the corrected GT differs.
CPU smoke = a code-path check on 2 clips x 3 windows: NO number in it is a result.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])


def load_panel(d):
    man = json.load(open(d / "manifest.json", encoding="utf-8"))
    rows = {}
    for e in man["episodes"]:
        z = np.load(d / f"ep{e['file_index']:03d}.npz")
        for i, w in enumerate(z["ws"]):
            rows[(e["episode_index"], int(w))] = {k: z[k][i] for k in ("g", "v0", "ha0_ext", "os") if k in z.files}
    return man, rows


rec = {"tool": "collect_smoke_r8.py", "what": "CPU smoke of the refcv8 battery kit: 2 clips x 3 windows, ckpt_5000, --smoke-cpu "
       "(no GPU, no lock); a code-path check, not a result", "runs": {}}
for tag in ("smoke_unc2", "smoke_cor"):
    d = root / tag
    sp = d / "battery_summary.json"
    if not sp.exists():
        rec["runs"][tag] = {"status": "ABSENT"}
        continue
    j = json.load(open(sp, encoding="utf-8"))
    st = j.get("stages") or {}
    r = {"complete": bool(j.get("bars") and j.get("finished")), "started": j.get("started"), "finished": j.get("finished"),
         "r8": {k: (j.get("r8") or {}).get(k) for k in ("pose_sync_policy", "no_baselines", "os_only_all", "g0_nonblocking", "wall_s")},
         "n_windows_S2": (j.get("S2") or {}).get("n_windows"), "stages_present": sorted(st),
         "roll_wall_s": {s: (st.get(f"roll_{s}") or {}).get("wall_s") for s in ("s0", "s1")},
         "roll_arm_set": {s: (st.get(f"roll_{s}") or {}).get("arm_set") for s in ("s0", "s1")},
         "files_written": sorted(p.name for p in d.iterdir() if p.is_file()),
         "baselines_used": sorted((j.get("baselines_used") or {})),
         "child_rewrites": len((j.get("r8") or {}).get("patch", {}).get("child_rewrites") or [])}
    pairing = {}
    for s in (0, 1):
        pr = (st.get(f"panel_s{s}") or {}).get("pairing") or {}
        pairing[f"s{s}"] = {lab: {k: p.get(k) for k in ("n_windows_paired", "g_bit_equal_windows", "g_max_abs_diff",
                                                         "model_free_bit_equal", "model_free_max_abs_diff")} for lab, p in pr.items()}
    r["panel_pairing"] = pairing
    r["void_gates_3_4"] = {s: (st.get(f"panel_s{s}") or {}).get("void_gates_3_4") for s in (0, 1)}
    rec["runs"][tag] = r
# the two clocks on the SAME windows
if (root / "smoke_unc2/panel_s0/manifest.json").exists() and (root / "smoke_cor/panel_s0/manifest.json").exists():
    _, a = load_panel(root / "smoke_unc2/panel_s0")
    _, b = load_panel(root / "smoke_cor/panel_s0")
    common = sorted(set(a) & set(b))
    rec["clock_difference_same_windows"] = {
        "n_common_windows": len(common),
        "max_abs_diff_g_m": max(float(np.abs(a[k]["g"] - b[k]["g"]).max()) for k in common),
        "max_abs_diff_v0_mps": max(float(abs(a[k]["v0"] - b[k]["v0"])) for k in common),
        "max_abs_diff_ha0_ext_m": max(float(np.abs(a[k]["ha0_ext"] - b[k]["ha0_ext"]).max()) for k in common),
        "reading": "non-zero on the model-free arm and the GT = the corrected child really carries the pose shift; the standalone "
                   "child has bit-identical GT to refcv7 (panel_pairing above)"}
pj = root / "perc8" / "perc.json"
rec["perception_pass"] = json.load(open(pj, encoding="utf-8")) if pj.exists() else {"status": "ABSENT (no perc.json)"}
if isinstance(rec["perception_pass"], dict) and len(json.dumps(rec["perception_pass"], default=str)) > 6000:
    pp = rec["perception_pass"]
    rec["perception_pass"] = {k: (v if len(json.dumps(v, default=str)) < 400 else "<elided>") for k, v in pp.items()}
rec["pair_read"] = {}
for tag in ("smoke_unc2", "smoke_cor"):
    f = root / tag / f"PAIRED_READ_{tag}.json"
    if f.exists():
        p = json.load(open(f, encoding="utf-8"))
        rec["pair_read"][tag] = {"clock": p.get("clock"), "bars": {k: (v.get("verdict") or v.get("per_family")) for k, v in p["bars"].items()},
                                 "controls": {k: (v.get("status"), v.get("all_bit_identical")) for k, v in (p.get("controls") or {}).items()}}
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(rec, open(out, "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps({"runs": {k: v.get("complete") for k, v in rec["runs"].items()}, "clock": rec.get("clock_difference_same_windows")}, indent=1))
