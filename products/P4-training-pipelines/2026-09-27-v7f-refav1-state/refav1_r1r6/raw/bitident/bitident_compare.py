"""Compare two bitident dumps. Tensors: torch.equal (bitwise, same dtype + shape). Everything
else: ==. `D.*.config.cfg_common` is compared after dropping the keys the R1-R6 build ADDS to
RefAV1Config (the new fields, all at their OFF defaults) -- those are printed, never hidden."""
import json
import sys

import torch

NEW_CFG_FIELDS = {"strategic_off", "vmax_input", "w_goal", "w_speed_band", "w_traj",
                  "traj_steps", "traj_grid", "traj_regions", "traj_d_region", "traj_hidden",
                  "traj_prior", "traj_prior_units", "traj_kappa_source",
                  "traj_huber_beta_m"}
a = torch.load(sys.argv[1], weights_only=False)
b = torch.load(sys.argv[2], weights_only=False)
ka, kb = set(a) - {"tanitad_file"}, set(b) - {"tanitad_file"}
print("base:", a["tanitad_file"])
print("new :", b["tanitad_file"])
only_a, only_b = sorted(ka - kb), sorted(kb - ka)
n_t = n_o = 0
diffs = []
added_cfg = {}
for k in sorted(ka & kb):
    x, y = a[k], b[k]
    if k.endswith("config.cfg_common"):
        xd, yd = json.loads(x), json.loads(y)
        extra = {kk: yd[kk] for kk in set(yd) - set(xd)}
        added_cfg[k] = extra
        if set(extra) - NEW_CFG_FIELDS:
            diffs.append((k, f"unexpected new cfg keys {sorted(set(extra) - NEW_CFG_FIELDS)}"))
        if xd != {kk: vv for kk, vv in yd.items() if kk in xd}:
            diffs.append((k, "shared cfg fields differ"))
        n_o += 1
        continue
    if torch.is_tensor(x) or torch.is_tensor(y):
        n_t += 1
        ok = (torch.is_tensor(x) and torch.is_tensor(y) and x.dtype == y.dtype
              and x.shape == y.shape and torch.equal(x, y))
        if not ok:
            diffs.append((k, "tensor differs"))
    else:
        n_o += 1
        if x != y:
            diffs.append((k, f"{str(x)[:80]!r} != {str(y)[:80]!r}"))
print(f"compared: {n_t} tensors (bitwise) + {n_o} other values; "
      f"only-in-base {len(only_a)}, only-in-new {len(only_b)}")
print("cfg fields ADDED by R1-R6 (must all be at their OFF defaults):",
      json.dumps(next(iter(added_cfg.values()), {}), default=str))
for k, why in diffs[:50]:
    print("DIFF", k, why)
if only_a or only_b:
    print("ONLY-IN:", only_a[:10], only_b[:10])
verdict = "BIT-IDENTICAL" if not diffs and not only_a and not only_b else "NOT IDENTICAL"
print("VERDICT:", verdict, f"({len(diffs)} differing entries)")
sys.exit(0 if verdict == "BIT-IDENTICAL" else 1)
