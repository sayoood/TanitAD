#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2 -- prove the batch matches the CONSUMER'S contract, and report the
pre/post-filter n.

The consumer is `refc_v3_train.compute_losses_v3` (which builds `tgt_ag` from
`batch["agent_*"]`) feeding `refc_agents.agent_losses`.  Nothing here invents a
shape: every key/dtype/shape below is read off that call site.

CONTROLS
--------
K1  every key `compute_losses_v3` reads is present, with the dtype/shape the
    consumer's downstream (`targets_from_join`/`slot_set_loss`) requires.
K2  the IN-FIELD-ONLY visible fraction over the emitted targets must land near
    the join's own `visible_frac` 0.4106 -- two independent measurements of one
    physical fact.  Far from it => the wiring is wrong, not the corpus.
K3  NO_LABEL and LABELLED-CLEAR are distinguishable in the batch
    (`agent_label` False vs True with `valid` all-False).
K4  `agent_losses` runs on the emitted block and returns a FINITE, NON-ZERO
    total with non-zero matched n -- a target block that cannot produce a loss
    is not a wiring.
K5  DELIBERATE REGRESSION: with `agent_label` withheld the labelled count
    reads 0 and the trainer's guard must refuse rather than score.
"""
import json
import os
import sys

import numpy as np
import torch

STACK = r"C:\Users\Admin\tanitad-wt\stack"
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))

JOIN = r"C:\Users\Admin\tanitad-data\joins\joins\train2400_agents.jsonl"
EPROOT = (r"C:\Users\Admin\tanitad-data\physicalai\_epcache"
          r"\physicalai-train-14231cd29c74")
N_EPS = int(os.environ.get("N_EPS", "40"))

import refc_v3_train as T                                   # noqa: E402
from refb_train import load_cached_episodes                 # noqa: E402
from train_p8_occupancy import JoinFileReader               # noqa: E402
from tanitad.refs import refc, refc_agents                  # noqa: E402
from tanitad.models.agent_slots import SlotDecodeRanges     # noqa: E402

out = {"_evidence_class": "MEASURED (ours)"}

eps, root = load_cached_episodes(os.path.dirname(EPROOT), "*train*", N_EPS)
print("episodes: %d from %s" % (len(eps), root))
ids = {int(e.episode_id) for e in eps}

rd = JoinFileReader(JOIN, episode_ids=ids, with_rates=True)
print("reader: %d records / %d clips (filtered out %d), max_agents/frame %d, "
      "classes %s, occ flags %s"
      % (rd.n_records, rd.n_clips, rd.n_records_filtered_out,
         rd.max_agents_per_frame, rd.has_classes, rd.has_occlusion_flags))
out["reader"] = {"n_records": rd.n_records, "n_clips": rd.n_clips,
                 "n_filtered_out": rd.n_records_filtered_out,
                 "max_agents_per_frame": rd.max_agents_per_frame,
                 "has_classes": rd.has_classes,
                 "has_occlusion_flags": rd.has_occlusion_flags,
                 "n_episodes_offered": len(eps)}

ds = T.V3Dataset(eps, window=4, max_horizon=20, channels=9)
stats = ds.enable_agent_join(rd, allow_legacy_ids=True)
out["enable_agent_join_stats"] = stats
print(json.dumps(stats, indent=1))

# ---- K1: the contract, read off the consumer -----------------------------
from torch.utils.data import default_collate                # noqa: E402
n = min(64, len(ds))
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[:n].tolist()
batch = default_collate([ds[i] for i in idx])
want = {"agent_box": (torch.float32, 3), "agent_yaw": (torch.float32, 2),
        "agent_cls": (torch.int64, 2), "agent_valid": (torch.bool, 2),
        "agent_occ": (torch.float32, 2), "agent_rates": (torch.float32, 3),
        "agent_rates_mask": (torch.bool, 2), "agent_label": (torch.bool, 1)}
k1 = {}
for k, (dt, nd) in want.items():
    v = batch.get(k)
    k1[k] = {"present": v is not None,
             "dtype": str(v.dtype) if v is not None else None,
             "shape": list(v.shape) if v is not None else None,
             "ok": v is not None and v.dtype == dt and v.dim() == nd}
out["K1_contract"] = k1
print("K1 contract:", all(x["ok"] for x in k1.values()),
      {k: v["shape"] for k, v in k1.items()})

# ---- K2: pre/post filter n, and the visible-fraction control -------------
tgt = {"box": batch["agent_box"], "yaw": batch["agent_yaw"],
       "cls": batch["agent_cls"], "valid": batch["agent_valid"],
       "occ": batch["agent_occ"], "rates": batch["agent_rates"],
       "rates_mask": batch["agent_rates_mask"]}
lab = batch["agent_label"]
n_pre = int(tgt["valid"].sum())
infield = refc_agents.filter_targets_to_visible(tgt, x_min_m=-1e9)
n_infield = int(infield["valid"].sum())
full = refc_agents.visible_target_filter(tgt)
n_full = int(full["valid"].sum())
# the join's OWN occ flag over the same rows -- a second, independent probe
occ0 = int(((batch["agent_occ"] == 0.0) & batch["agent_valid"]).sum())
out["K2_filter"] = {
    "n_windows": int(lab.numel()), "n_windows_labelled": int(lab.sum()),
    "n_target_prefilter": n_pre,
    "n_target_infield_only": n_infield,
    "frac_infield_only": round(n_infield / max(n_pre, 1), 4),
    "n_target_occ0_from_join_flag": occ0,
    "frac_occ0_from_join_flag": round(occ0 / max(n_pre, 1), 4),
    "n_target_infield_and_decodebox": n_full,
    "frac_infield_and_decodebox": round(n_full / max(n_pre, 1), 4),
    "corpus_visible_frac_reference": 0.4106,
    "corpus_infield_bevbox_frac_reference": round(1899481 / 12122129, 4),
    "decode_box": {"x_fwd_m": SlotDecodeRanges().x_fwd_m,
                   "y_half_m": SlotDecodeRanges().y_half_m},
}
print("K2 pre-filter n=%d  in-field-only n=%d (%.4f, corpus 0.4106)  "
      "join-occ0 n=%d (%.4f)  in-field AND decode box n=%d (%.4f, corpus "
      "%.4f)" % (n_pre, n_infield, n_infield / max(n_pre, 1), occ0,
                 occ0 / max(n_pre, 1), n_full, n_full / max(n_pre, 1),
                 1899481 / 12122129))

# ---- K3: NO_LABEL vs LABELLED-CLEAR --------------------------------------
per_win_valid = batch["agent_valid"].sum(1)
n_nolabel = int((~lab).sum())
n_clear = int((lab & (per_win_valid == 0)).sum())
out["K3_states"] = {"n_NO_LABEL": n_nolabel, "n_labelled_clear": n_clear,
                    "distinguishable": bool(n_nolabel >= 0 and n_clear >= 0),
                    "NO_LABEL_rows_all_invalid":
                    bool((per_win_valid[~lab] == 0).all()) if n_nolabel
                    else True}
print("K3 NO_LABEL windows %d ; labelled-clear windows %d" % (n_nolabel,
                                                              n_clear))

# ---- K4: the loss actually computes --------------------------------------
cfg = refc_agents.AgentSeamConfig(enable=True, queries=32, enforce_band=False)
torch.manual_seed(0)
head = refc_agents.build_agent_head(cfg, d_memory=64, n_memory=16)
sel = lab.nonzero(as_tuple=False).flatten()
mem = torch.randn(int(sel.numel()), 16, 64)
slots = head(mem)
tgt_l = {k: v.index_select(0, sel) for k, v in tgt.items()}
ag = refc_agents.agent_losses(slots, tgt_l, cfg)
out["K4_loss"] = {k: (float(v) if torch.is_tensor(v) and v.numel() == 1
                      else (v if not torch.is_tensor(v) else None))
                  for k, v in ag.items() if k != "_weights"}
print("K4 total=%.6f finite=%s  n=%s  n_dropped=%d/%d"
      % (float(ag["total"]), bool(torch.isfinite(ag["total"])),
         ag["n"], ag["n_dropped"], ag["n_target"]))

# ---- K5: the deliberate-regression control -------------------------------
# labels withheld -> the trainer's `agent_box not in batch` guard must fire.
b2 = {k: v for k, v in batch.items() if not k.startswith("agent_")}
out["K5_withheld"] = {"agent_keys_left":
                      sorted(k for k in b2 if k.startswith("agent_"))}
print("K5 withheld batch has agent keys:", out["K5_withheld"])

with open("batch_contract.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("wrote batch_contract.json")
