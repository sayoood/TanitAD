"""D-LAN-PF probe — WHAT IS `route` IN THE v3 PREFLIGHT, AND CAN `--goal-str` EVER MOVE IT?

Pre-registered question (both outcomes committed before the run):

  H1  `losses["route"]` is the v2.1 NAV-derived route CE and is INDEPENDENT of `lan`.
      SUPPORTED  if route is bit-identical with and without --graft-lan/--goal-str
                 AND the mask that zeroes it is nav_valid, not a lan field.
      REFUTED    if any lan flag changes route.

  H2  `preflight()` structurally cannot exercise the lan pathway.
      SUPPORTED  if the dataset preflight builds carries no "lan" key while the
                 train path's dataset does.
      REFUTED    if preflight's batch carries "lan".

  H3  The lan pathway, when actually fed, produces a FINITE NON-ZERO goal_str.
      SUPPORTED  if a lan-carrying batch yields a finite goal_str > 0.
      REFUTED    if it is absent, zero, or non-finite.

Run:  PYTHONPATH=<stack> python probe_route_vs_lan.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
STACK = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/tanitad-wt/stack")
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))

import refc_v3_train as T                                    # noqa: E402
from tanitad.data.lan import LanConfig as DataLanConfig      # noqa: E402
from tanitad.refs import refc, refc_v3 as v3                 # noqa: E402

OUT: dict = {"probe": "D-LAN-PF", "evidence_class": "MEASURED"}

cfg = v3.refc_v3_smoke_config(True)
eps = T._synth_episodes(2, cfg.core, seed=0)

# ---- H2: what does preflight's own dataset carry? ---------------------------
ds_pf = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                    channels=cfg.core.encoder.in_channels)
batch_pf = torch.utils.data.default_collate([ds_pf[0], ds_pf[1]])

lan_cls = T.lan_dataset_class(T.V3Dataset)
ds_lan = lan_cls(eps, window=cfg.core.window, max_horizon=20,
                 channels=cfg.core.encoder.in_channels,
                 lan_cfg=DataLanConfig(arclengths_m=(20.0, 40.0, 80.0, 160.0),
                                       min_lead_m=5.0))
batch_lan = torch.utils.data.default_collate([ds_lan[0], ds_lan[1]])

OUT["H2_preflight_dataset_class"] = type(ds_pf).__name__
OUT["H2_lan_dataset_class"] = type(ds_lan).__name__
OUT["H2_preflight_batch_has_lan"] = "lan" in batch_pf
OUT["H2_lan_batch_has_lan"] = "lan" in batch_lan
OUT["H2_lan_shape"] = (list(batch_lan["lan"].shape)
                       if "lan" in batch_lan else None)
OUT["H2_verdict"] = ("SUPPORTED" if (not OUT["H2_preflight_batch_has_lan"]
                                     and OUT["H2_lan_batch_has_lan"])
                     else "REFUTED")

# ---- the mask that actually zeroes `route` ----------------------------------
nv = batch_pf["nav_valid"]
rt = batch_pf["route_target"]
OUT["nav_valid"] = [bool(x) for x in nv]
OUT["nav_valid_any"] = bool(nv.any())
OUT["route_target"] = [int(x) for x in rt]
OUT["ROUTE_UNKNOWN_is"] = 3

# ---- H1 + H3: run the loss on both batches ----------------------------------
torch.manual_seed(0)
cfg_lan = v3.refc_v3_smoke_config(True)
cfg_lan.core.lan = refc.LanConfig(k=4)
model_lan = v3.RefCV3Model(cfg_lan)

torch.manual_seed(0)
model_pf = v3.RefCV3Model(v3.refc_v3_smoke_config(True))


def scalars(d):
    return {k: round(float(t.detach()), 6)
            for k, t in d.items() if torch.is_tensor(t) and t.ndim == 0}


l_pf = scalars(T.compute_losses_v3(model_pf, batch_pf, "cpu", mode="diffusion"))
l_lan = scalars(T.compute_losses_v3(model_lan, batch_lan, "cpu",
                                    mode="diffusion"))

OUT["losses_preflight_path"] = l_pf
OUT["losses_lan_path"] = l_lan
OUT["H1_route_preflight"] = l_pf.get("route")
OUT["H1_route_lan"] = l_lan.get("route")
OUT["H1_route_identical"] = l_pf.get("route") == l_lan.get("route")
OUT["H1_verdict"] = ("SUPPORTED"
                     if (l_pf.get("route") == l_lan.get("route") == 0.0
                         and not OUT["nav_valid_any"])
                     else "REFUTED")

gs = l_lan.get("goal_str")
OUT["H3_goal_str_present_in_lan_path"] = gs is not None
OUT["H3_goal_str_present_in_preflight_path"] = l_pf.get("goal_str") is not None
OUT["H3_goal_str_value"] = gs
OUT["H3_verdict"] = ("SUPPORTED"
                     if (gs is not None and gs > 0.0
                         and float(gs) == float(gs))     # finite
                     else "REFUTED")

# ---- the leak guard: does lan carry any live anchor at all? -----------------
if "lan" in batch_lan:
    lan = batch_lan["lan"].reshape(batch_lan["lan"].shape[0], -1, 4)
    OUT["lan_valid_bits"] = lan[..., 3].tolist()
    OUT["lan_valid_frac"] = float(lan[..., 3].mean())

print(json.dumps(OUT, indent=1))
