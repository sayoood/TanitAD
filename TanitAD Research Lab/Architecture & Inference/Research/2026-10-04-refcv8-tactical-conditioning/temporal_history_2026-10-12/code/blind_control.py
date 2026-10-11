"""Deliberate-regression arm E (PREREG s5): constant-grey images in all 10 slots, first 12 episodes of G3.
Must be DEGRADED on P1 (S6 ADE, paired vs REAL on the same windows) or no verdict of this experiment stands.
usage: python blind_control.py <out_root> <real_s0_dump> <blind_dump>"""
import json, os, shutil, sys
from pathlib import Path
out_root = Path(sys.argv[1]); real, blind = sys.argv[2], sys.argv[3]
BAT = Path(os.environ.get("BATT_CODE", "E:/hold_temporal/work/battery")); sys.path.insert(0, str(BAT))
sys.stdout.reconfigure(encoding="utf-8")
import r7_panel as P  # noqa: E402
P.TIERS["os_blind"] = "T1"
mb = json.load(open(blind + "/manifest.json", encoding="utf-8"))
clips = [e["clip_id"] for e in mb["episodes"]]
mr = json.load(open(real + "/manifest.json", encoding="utf-8"))
sub = out_root / "real_subset"
(sub / "decisions").mkdir(parents=True, exist_ok=True)
keep = [e for e in mr["episodes"] if e["clip_id"] in set(clips)]
assert len(keep) == len(clips), (len(keep), len(clips))
for e in keep:
    fi = e["file_index"]
    shutil.copyfile(f"{real}/ep{fi:03d}.npz", sub / f"ep{fi:03d}.npz")
    shutil.copyfile(f"{real}/decisions/ep{fi:03d}.npz", sub / "decisions" / f"ep{fi:03d}.npz")
m2 = dict(mr)
m2["episodes"] = keep
json.dump(m2, open(sub / "manifest.json", "w", encoding="utf-8"), default=str)
s6 = str(out_root / "s6_blind")
rec = P.build_s6_dump(str(sub), s6, baselines={"os_blind": blind})
cp = P.cross_paired(s6, [("os", "os_blind", "os_blind_minus_os")], n_boot=2000)
json.dump({"build": rec, "cp": cp}, open(out_root / "blind_control.json", "w", encoding="utf-8"), indent=1, default=str)
a = cp["pairs"]["os_blind_minus_os"]["families"]["ADE"]["ade_m"]
print("BLIND vs REAL S6 ADE paired:", a["delta"], a["lo"], a["hi"], "n", a["n_windows"], a["n_episodes"],
      "RED(DEGRADED)" if a["lo"] > 0 else "NOT-RED: instrument blind -> no verdict stands")
print("ZZBLIND_DONE")
