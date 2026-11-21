"""Final audit of the staged deliverables: readability, leakage, and that the
report's headline numbers recompute from the staged raw files with NO GPU."""
import json
import re
from pathlib import Path

import numpy as np
import torch

D = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Hub/"
         r"Architecture & Inference/Implementation/incoming/2026-07-27-t3-and-lambda-tau")

print("=== 1. every staged artifact loads ===")
for f in sorted((D / "raw").glob("*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    print(f"  {f.name:28} {len(d)} keys")
for f in sorted((D / "raw").glob("*.json")):
    json.load(open(f))
    print(f"  {f.name:28} ok")

print("\n=== 2. leakage scan (uuid / hf token / absolute clip ids) ===")
uuid = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
tok = re.compile(r"hf_[A-Za-z0-9]{20,}")
bad = []
for f in list(D.rglob("*.md")) + list(D.rglob("*.py")) + list(D.rglob("*.json")):
    t = f.read_text(encoding="utf-8", errors="replace")
    if uuid.search(t):
        bad.append((f.name, "UUID"))
    if tok.search(t):
        bad.append((f.name, "HF TOKEN"))
for f in (D / "raw").glob("*.pt"):
    b = f.read_bytes()
    if uuid.search(b.decode("latin-1")):
        bad.append((f.name, "UUID in tensor file"))
print("  leaks:", bad if bad else "NONE")

print("\n=== 3. T3 bars recompute from raw/t3_windows.pt, no GPU ===")
W = torch.load(D / "raw" / "t3_windows.pt", map_location="cpu", weights_only=False)
tr = W["has_tracks"].numpy().astype(bool)
for arm in ("as_trained", "as_trained_clipped", "ce", "bce_rule"):
    p = W[f"{arm}|pdms"].numpy()[tr]
    c = W[f"{arm}|collision"].numpy()[tr]
    a = W[f"{arm}|ade"].numpy()
    print(f"  {arm:20} pdms={p.mean():.4f} coll={c.mean():.4f} "
          f"ade(881)={a.mean():.4f}")
R = json.load(open(D / "raw" / "t3_result.json"))
print("  json says:", {k: v["pdms_lite"] for k, v in R["point_estimates"].items()})
print("  VERDICT:", R["VERDICT"]["verdict"])

print("\n=== 4. lambda/tau: tau=1 row + argmin + admissible set ===")
S = json.load(open(D / "raw" / "eh2_sweep.json"))["sheet_deployable"]
cells = {(c["lambda"], c["tau"]): c for c in S["cells"]}
row = [cells[(l, 1.0)]["ade_0_2s"] for l in (0, .25, .5, 1., 2., 4., 8.)]
print("  tau=1 row:", row, "-> argmin lambda =",
      [0, .25, .5, 1., 2., 4., 8.][int(np.argmin(row))])
print("  optimum:", S["optimum"]["verdict"], S["optimum"]["argmin"],
      "admissible", len(S["optimum"]["admissible_set"]))
print("  graft alone =", round(cells[(0., 1.)]["ade_0_2s"]
                               - cells[(1., 1.)]["ade_0_2s"], 4),
      " cv =", round(0.8781 - cells[(0., 1.)]["ade_0_2s"], 4))

print("\n=== 5. labels: rates + gate ===")
L = json.load(open(D / "raw" / "t3_labels.json"))
print("  ", L["positive_rates_on_v4_fan"])
print("  ", L["ground_truth_self_label_control"])
print("   clock gate:", L["clock_gate"]["n_episodes_pass"], "pass /",
      L["clock_gate"]["n_episodes_fail"], "fail; rms_max",
      L["clock_fit_rms_xy_m_max"], "k", L["index_offset_k_measured"])

print("\n=== 6. goalflow ===")
G = json.load(open(D / "raw" / "eh3_goalflow.json"))
print("  ", G["arms"], G["VERDICT"]["verdict"], G["VERDICT"]["best_signal"])
