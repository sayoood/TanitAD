"""Q2: from a --save-cands base dump, write plan VARIANTS as ordinary eval dumps (traj replaced; every other array and
the record copied) so ladder_score scores them with its own definitions.
  oraclebase -- the base-fan row (index < n_base) with the lowest ADE over valid horizons vs GT (label-side upper bound)
  oracleall  -- the same over the whole fan (only written when the fan has extra rows)
  randombase -- a uniform base-fan row per window (numpy default_rng(0)): the control for any oracle gap
usage: python oracle_variants.py <arm dir with base_s{0,1}.npz>"""
import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
for s in (0, 1):
    src = d / f"base_s{s}.npz"
    z = dict(np.load(src, allow_pickle=False))
    rec = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
    c, gt, gv = z["cands"], z["gt"], z["gt_valid"].astype(bool)
    nb, nf = int(z["n_base"][0]), int(z["n_fan"][0])
    assert c.shape[1] == nf, (c.shape, nf)
    err = np.linalg.norm(c - gt[:, None], axis=-1)                                  # [W, N, H]
    w = gv[:, None, :].astype(float)
    ade = (err * w).sum(-1) / np.maximum(w.sum(-1), 1.0)                            # [W, N]
    sel = np.abs(c - z["traj"][:, None]).reshape(c.shape[0], nf, -1).max(-1).argmin(1)
    out = {"oraclebase": ade[:, :nb].argmin(1),
           "randombase": np.random.default_rng(0).integers(0, nb, len(c))}
    if nf > nb:
        out["oracleall"] = ade.argmin(1)
    ar = np.arange(len(c))
    print(f"{d.name} s{s}: fan {nf} (base {nb}); selected extra-row frac {(sel >= nb).mean():.3f}; "
          f"mean ADE sel {ade[ar, sel].mean():.3f} oracle-base {ade[ar, out['oraclebase']].mean():.3f}"
          + (f" oracle-all {ade[ar, out['oracleall']].mean():.3f}" if 'oracleall' in out else "")
          + f" random-base {ade[ar, out['randombase']].mean():.3f}")
    for nm, idx in out.items():
        v = dict(z)
        v["traj"] = c[ar, idx]
        v["variant_idx"] = idx
        p = d / f"{nm}_s{s}"
        np.savez_compressed(str(p) + ".npz", **v)
        r = dict(rec, row=nm, variant_of="base", variant_rule=nm)
        Path(str(p) + ".json").write_text(json.dumps(r, indent=1), encoding="utf-8")
