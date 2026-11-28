"""P3 reference numbers on the C3 set (1,058 navtest tokens x 64 proposals), CPU only, from BANKED inputs. Not an S1 read: it measures the instrument.

  * scorer     : the u400 scorer's own DAC logit (table.npz logits[..., 1]) as the DAC = 1 ranking score -- 'the current scorer's DAC AUC on the same set, computed'.
  * truth-fed  : a head that outputs the TRUE block-mean SDF at the head's 2 m grid (AUX.build_targets of the navtest label) -> the best a head of this representation
                 can do through the P3 statistic (min predicted SDF along the footprint corners, 40 steps x 4 corners).
  * native     : the same statistic read from the stored 0.5 m label directly (field_loss.sample_sdf) -- the resolution ceiling of the label store itself.
DAC = 1 / 0 is NAVSIM's, from table.npz sub[..., drivable_area_compliance] (a controller-simulated path, not the raw proposal footprint: that is why even the
native label does not reach 1.0).  Run (driverl-eval venv): python p3_reference.py  -> p3_reference.json
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PKG, "eval"))
import aux_s1_readout as R  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402

C3 = os.path.join(PKG, "raw", "2026-10-11-c3-selection-headroom")
tab, toks = np.load(os.path.join(C3, "table.npz")), json.load(open(os.path.join(C3, "tokens.json"), encoding="utf-8"))
names = [str(x) for x in tab["sub_names"]]
di = names.index("drivable_area_compliance")
tok_list = [str(t) for t in tab["token"]]
tok_logs = [toks["token_log"][t] for t in tok_list]
src = R.open_labels("navtest")
t2k = {k[1]: k for k in src.all_keys()}
lab, valid = src.labels([t2k[t] for t in tok_list])
assert bool(valid.all())
t = AUX.build_targets(lab, valid)
dac, vp = tab["sub"][..., di], tab["valid"]
scorer = tab["logits"][..., 1]
out = {"n_tokens": len(tok_list), "n_logs": len(set(tok_logs)), "n_proposals_valid": int(vp.sum()), "n_dac1": int(((dac == 1.0) & vp).sum()),
       "n_dac0": int(((dac == 0.0) & vp).sum()), "c3_table_sha256": R.sha256_file(os.path.join(C3, "table.npz")),
       "navtest_store_sha256": src.manifest["store_sha256"], "n_boot": R.N_BOOT, "seed": R.SEED}
r2 = R.p3_read(t.sdf_m.numpy(), tab["proposals"], dac, vp, scorer, tok_logs)
out["scorer_dac_auc_pooled"] = r2["pooled"]["scorer"]
out["scorer_dac_auc_within_set_mean"] = r2["within_set"]["scorer"]
out["truth_fed_2m_pooled"] = r2["pooled"]["head"]
out["truth_fed_2m_within_set_mean"] = r2["within_set"]["head"]
out["truth_fed_2m_minus_scorer_pooled"] = r2["pooled"]["head_minus_scorer"]
out["truth_fed_2m_decision"] = r2["decision"]
N, K = dac.shape
rel = R.ND.at_10hz(np.asarray(tab["proposals"], np.float64))[..., 1:, :]
corners = AUX.FL.footprint_corners(torch.from_numpy(rel).float()).reshape(N, K * 160, 2)
s, _inb = AUX.FL.sample_sdf(lab, corners)
nat = s.reshape(N, K, 160).min(-1).values.numpy()
ul = sorted(set(tok_logs))
pos = {l: i for i, l in enumerate(ul)}
cl = np.repeat(np.asarray([pos[l] for l in tok_logs])[:, None], K, 1)
S, Nm = R.auc_cluster_stats(nat[vp], dac[vp] == 1.0, cl[vp], len(ul))
out["native_0p5m_label_pooled"] = R.auc_read(S, Nm, R.boot_weights(len(ul)))
json.dump(out, open(os.path.join(HERE, "p3_reference.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True, default=float)
print(json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a in ("est", "ci95", "decision")}) for k, v in out.items()}, indent=1, default=float))
