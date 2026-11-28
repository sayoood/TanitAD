"""rehearsal_real_path.py -- REAL-path rehearsal (plumbing only, no S1c number): S1's H1-s1 head_pred stands in under the arm name H1u10k-s0, S1's H1-s0 is the comparator.
Needs the dry run's out/dryrun/forward_cpu/ (head_pred_H1-s0/s1.npz).  Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I rehearsal_real_path.py
Runs the harness `perception` (navtest) with the S1c names and nat_p1 with the contrast, then checks the harness head_fit_pairs entry against nat_p1's C3 contrast (what Job.finish asserts)."""
import json
import subprocess
import sys
from pathlib import Path

PKR = Path(__file__).resolve().parent
sys.path.insert(0, str(PKR))
import s1c_autoread as A  # noqa: E402

S1R = A.S1R
out = PKR / "out" / "rehearsal"
out.mkdir(parents=True, exist_ok=True)
fw = PKR / "out/dryrun/forward_cpu"
preds = {"H1u10k-s0": str(fw / "head_pred_H1-s1.npz"), "H1-s0": str(fw / "head_pred_H1-s0.npz")}
fl = S1R.floor_inputs(A.FLOOR_DIR, str(A.PK))
inits = {a: fl[A.arm_seed(a)] for a in preds}
nav_json = out / "perception_navtest.json"
cmd = S1R.cmd_perception_navtest(A.PY_EVAL, str(A.EVAL), str(A.PK), preds, inits, str(nav_json))
r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
print("harness rc", r.returncode, r.stdout.strip().splitlines()[-1:] , r.stderr.strip().splitlines()[-2:])
nat_json = out / "nat.json"
cmd = A.nat_cmd(A.PY_EVAL, preds, [("H1u10k-s0", "H1-s0")], nat_json)
r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
print("nat rc", r.returncode, r.stdout.strip().splitlines()[-1:], r.stderr.strip().splitlines()[-2:])
nav = json.load(open(nav_json))
nat = json.load(open(nat_json))
pair = "H1u10k-s0-minus-H1-s0"
h = nav["head_fit_pairs"][pair]["iou_0-40"]
c = nat["contrasts"][pair]["c3"]
print("harness pair keys:", sorted(nav["head_fit_pairs"][pair])[:8])
print("harness iou_0-40:", h["est"], h["ci95"])
print("nat_p1  c3      :", c["est"], c["ci95"])
print("abs diff est", abs(h["est"] - c["est"]), "ci", max(abs(h["ci95"][0] - c["ci95"][0]), abs(h["ci95"][1] - c["ci95"][1])))
print("harness arm p1:", nav["arms"]["H1u10k-s0"]["p1"]["all"]["0-40"]["est"], "nat c3:", nat["arms"]["H1u10k-s0"]["c3"]["est"])
print("nat contrast natural:", nat["contrasts"][pair]["natural"]["est"], nat["contrasts"][pair]["natural"]["ci95"])
print("keys in nav arms:", sorted(nav["arms"]))

# ---- step 3: the descriptive selection read with the S1c arm NAME (stand-in scores: S1's H1-s0 dump) beside the three N arms and the comparator
pins = json.load(open(A.BANKED_SEL))["inputs"]["scores_sha256"]
sc = {a: str(Path(A.S1_CACHE) / "arms" / a / "dump" / "scores_update400_B.npz") for a in ("H1-s0", "N-s0", "N-s1", "N-s2")}
sc["H1u10k-s0"] = sc["H1-s0"]
pdm = str(Path(A.S1_CACHE) / "labels/x/heldout_pdm")
dj = out / "desc.json"
r = subprocess.run(A.desc_cmd(A.PY_EVAL, sc, pdm, dj), capture_output=True, text=True, encoding="utf-8")
print("desc rc", r.returncode, r.stdout.strip().splitlines()[-1:], r.stderr.strip().splitlines()[-2:])
d = json.load(open(dj))
print("desc arms:", {a: round(s["PDMS_x100"], 6) for a, s in d["arms"].items()}, "stamp:", d["stamp"][:40])
