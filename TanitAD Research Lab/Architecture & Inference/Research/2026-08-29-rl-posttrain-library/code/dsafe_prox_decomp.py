"""Post-hoc PROXIMITY decomposition for D-SAFE-CAL — TRAIN-C6 recurred.

⛔ WHY THIS SCRIPT HAD TO EXIST AT ALL. `R4_component_means` is computed with the
DEFAULT reward spec so arms stay comparable across the campaign — and the DEFAULT
spec does not contain `proximity`. So the decomposition, which the
pre-registration calls "the PRIMARY diagnostic", is STRUCTURALLY BLIND to the one
term the experiment varies. That is TRAIN-C6 exactly, recurring in the successor
experiment written to avoid it: I fixed exit 3's measurability by adding R5 and
did not check the decomposition for the same defect.

⚠️ THIS IS DESCRIPTIVE, NOT A VERDICT. The pre-registered exit was selected
mechanically and returned VOID. Nothing here may be used to reach a different
exit — a decomposition computed after seeing the outcome cannot license a branch
the committed rules did not select. It is published so the campaign's data is not
wasted and so the successor pre-registration can be written against real numbers.

Costs zero GPU beyond a readout: every arm saved `ckpt_after.pt` (the TRAIN-C5
fix, which has now rescued three separate analyses).
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, "stack")
_s = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(_s); _s.loader.exec_module(P)
from tanitad.rl import RewardSpec, rewards as RW  # noqa: E402

O = pathlib.Path(r"C:/Users/Admin/tanitad-data/rl-pilot")
dev = "cuda"
ARMS = ["dsafe-A-w1-d2", "dsafe-B-w10-d2", "dsafe-C-w1-d5",
        "dsafe-D-floor-d2", "dsafe-REG-hackable"]

src = P.WindowSource(
    r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
    f"{O}/pilot_val_agents.jsonl", seed=0, lru=60)

# a spec that CONTAINS proximity, used only for this diagnostic
w = dict(RW.DEFAULT_WEIGHTS); w["proximity"] = 0.5
spec_prox = RewardSpec(weights=w, dt=P.DT_TRAJ)

model = P.load_model(r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt", dev)
cold_sd = {k: v.clone() for k, v in model.state_dict().items()}
from tanitad.rl import PostTrainConfig  # noqa: E402
cfg = PostTrainConfig(method="grpo", group_size=4, steps=1, batch=1, lr=1e-5,
                      seed=0, dt=P.DT_TRAJ, decoder_steps=2)

base = P.readout(model, src, spec_prox, cfg, dev)
print(f"{'arm':<22}{'prox BEFORE':>13}{'prox AFTER':>13}{'delta':>11}"
      f"{'weighted':>11}")
print(f"{'(cold start)':<22}{base['R4_component_means']['proximity']:>13.4f}")
rows = {}
for a in ARMS:
    ck = O / a / "ckpt_after.pt"
    if not ck.exists():
        print(f"{a:<22} no ckpt_after — skipped"); continue
    model.load_state_dict(torch.load(ck, map_location="cpu",
                                     weights_only=False)["model"])
    model.to(dev)
    r = P.readout(model, src, spec_prox, cfg, dev)
    b4 = base["R4_component_means"]["proximity"]
    af = r["R4_component_means"]["proximity"]
    rows[a] = {"before": b4, "after": af, "delta": af - b4,
               "weighted": (af - b4) * 0.5}
    print(f"{a:<22}{b4:>13.4f}{af:>13.4f}{af-b4:>+11.4f}{(af-b4)*0.5:>+11.4f}")
    model.load_state_dict(cold_sd)

json.dump(rows, open(O / "dsafe_prox_decomp.json", "w"), indent=1)
print(f"\n-> {O}/dsafe_prox_decomp.json")
print("\n⚠️ DESCRIPTIVE ONLY. The committed exit was VOID and stays VOID.")
