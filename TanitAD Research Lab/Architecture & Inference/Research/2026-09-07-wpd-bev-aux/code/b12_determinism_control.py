# -*- coding: utf-8 -*-
"""WP-D step 12 -- WHAT IS THE A3 FLOOR ACTUALLY MADE OF?

A3's floor is |AP(D0b) - AP(D0)| and its whole purpose is to isolate TRAINING-run
variance (H-ESTIM-SEED-1).  That reading is only valid if the MEASUREMENT CHAIN
itself is deterministic.  If it is not, part of the floor is the INSTRUMENT and
the number answers a third question nobody asked -- exactly the family CLAUDE.md
records for the stochastic planner ("would another INFERENCE RUN say this?").

D0 has now been put through the ENTIRE chain TWICE, from the same trunk file:
  bank/      -- the 2026-09-08 panel
  bank_rep/  -- this run
so the chain's own reproducibility is MEASURABLE rather than assumed, in two
separable halves:

  1. BANK   : are the encoder features bit-identical?   (tok_D0 vs tok_D0)
  2. PROBE  : is the trained head's AP identical?       (pt_cart_tok_D0 APs)

⛔ And the DISCRIMINATING half, without which a positive assertion is blind:
D0b must NOT be bit-identical to D0 -- otherwise the "replicate" is an identity
and the floor is not measuring anything.
"""
import io, json, os, sys
import numpy as np

SNAP = r"C:\Users\Admin\wpd-probe\snap"
sys.path.insert(0, SNAP)
from tanitad.refs.refc_bev_aux import _average_precision as AP

A = r"C:\Users\Admin\wpd-probe\bank"          # the 2026-09-08 panel
B = r"C:\Users\Admin\wpd-probe\bank_rep"      # this run
out = {}

ia = json.load(io.open(os.path.join(A, "idx.json"), encoding="utf-8"))
ib = json.load(io.open(os.path.join(B, "idx.json"), encoding="utf-8"))
out["clips_equal"] = ia["clips"] == ib["clips"]
out["plan_equal"] = [list(p) for p in ia["plan"]] == [list(p) for p in ib["plan"]]
out["N_equal"] = ia["N"] == ib["N"]
out["N"] = ib["N"]
out["trunk_md5_D0_equal"] = ia["trunks"]["D0"]["md5"] == ib["trunks"]["D0"]["md5"]
out["trunks_rep_md5"] = {k: v["md5"] for k, v in ib["trunks"].items()}
print(f"[control] clips_equal={out['clips_equal']} plan_equal={out['plan_equal']} "
      f"N={out['N']} trunk_D0_md5_equal={out['trunk_md5_D0_equal']}", flush=True)

for name in ("y_cart.npy", "y_pol.npy", "m_pol.npy", "pix.npy"):
    xa = np.load(os.path.join(A, name), mmap_mode="r")
    xb = np.load(os.path.join(B, name), mmap_mode="r")
    eq = xa.shape == xb.shape and bool(np.array_equal(np.asarray(xa), np.asarray(xb)))
    out[f"{name}_bit_equal"] = eq
    print(f"[control] {name:12s} bit-equal={eq}", flush=True)

# ---- HALF 1: the BANK -------------------------------------------------------
xa = np.load(os.path.join(A, "tok_D0.npy"), mmap_mode="r")
xb = np.load(os.path.join(B, "tok_D0.npy"), mmap_mode="r")
assert xa.shape == xb.shape, (xa.shape, xb.shape)
ndiff, maxabs, tot = 0, 0.0, 0
for s in range(0, xa.shape[0], 512):
    ca = np.asarray(xa[s:s + 512]); cb = np.asarray(xb[s:s + 512])
    d = ca != cb
    ndiff += int(d.sum()); tot += ca.size
    if d.any():
        maxabs = max(maxabs, float(np.abs(ca[d].astype(np.float32)
                                          - cb[d].astype(np.float32)).max()))
out["bank_tok_D0_cells"] = tot
out["bank_tok_D0_differing_cells"] = ndiff
out["bank_is_deterministic"] = ndiff == 0
out["bank_tok_D0_max_abs_diff"] = maxabs
print(f"[HALF 1 BANK ] tok_D0 re-bank: {ndiff}/{tot} differing cells "
      f"({100.0*ndiff/tot:.4f} %), max |diff| {maxabs:.6g}  "
      f"DETERMINISTIC={ndiff == 0}", flush=True)

# discriminating control: the replicate must be a REAL second run
xc = np.load(os.path.join(B, "tok_D0b.npy"), mmap_mode="r")
nd2 = 0
for s in range(0, xa.shape[0], 512):
    nd2 += int((np.asarray(xb[s:s + 512]) != np.asarray(xc[s:s + 512])).sum())
out["tok_D0b_vs_D0_differing_cells"] = nd2
out["tok_D0b_is_distinct_from_D0"] = nd2 > 0
print(f"[control    ] tok_D0b vs tok_D0: {nd2}/{tot} differing "
      f"({100.0*nd2/tot:.3f} %)  DISTINCT={nd2 > 0}", flush=True)

# ---- HALF 2: the PROBE ------------------------------------------------------
# Two independent probe runs of the SAME arm on the SAME split. Their AP gap is
# the INSTRUMENT floor: what the chain reports as "different" when NOTHING is.
import math
from tanitad.data.bev_raster import BEVGrid, cell_centers_xy, fov_mask
grid = BEVGrid(x_fwd_m=float(ib["x_fwd_m"]), y_half_m=float(ib["y_half_m"]),
               cell_m=float(ib["cell_m"]))
X_m, Y_m = cell_centers_xy(grid)
az_full = np.arctan2(Y_m, X_m)
ok = fov_mask(grid, math.radians(60.0)) & (np.abs(az_full) <= math.radians(60.0) - 1e-9)
for sg in (-1, 1):
    cc = (320.0 + sg * 305.5774907364391 * az_full) / 32.0 - 0.5
    ok &= (cc >= 0) & (cc <= 19)
vidx = np.flatnonzero(ok.reshape(-1))

plan = np.asarray(ib["plan"], dtype=np.int64)
rng = np.random.default_rng(0)
order = rng.permutation(len(ib["clips"]))
te_c = set(order[:30].tolist())
ti = np.where(np.isin(plan[:, 0], list(te_c)))[0]
Y = np.asarray(np.load(os.path.join(B, "y_cart.npy"), mmap_mode="r"))[ti][:, vidx]
yflat = (Y.reshape(-1) > 0)
n_pos, n_sc = int(yflat.sum()), int(yflat.size)
out["probe_n_scored_cells"] = n_sc
out["probe_n_pos"] = n_pos
out["probe_base_rate_from_integer_counts"] = n_pos / n_sc

probe = {}
for tag, root in (("panel_2026_09_08", A), ("replicate_run", B)):
    f = os.path.join(root, "pt_cart_tok_D0.npy")
    if not os.path.exists(f):
        probe[tag] = None; continue
    P = np.load(f).reshape(-1).astype(np.float64)
    probe[tag] = AP(P, yflat.astype(np.float64))
    print(f"[HALF 2 PROBE] AP(tok_D0) {tag:18s} = {probe[tag]:.9f}", flush=True)
out["probe_ap_tok_D0"] = probe
if probe.get("panel_2026_09_08") is not None and probe.get("replicate_run") is not None:
    inst = abs(probe["panel_2026_09_08"] - probe["replicate_run"])
    out["instrument_floor_same_arm_same_code"] = inst
    out["probe_is_deterministic"] = inst == 0.0
    print(f"[HALF 2 PROBE] INSTRUMENT FLOOR |AP(D0)_panel - AP(D0)_replicate| "
          f"= {inst:.9f}   DETERMINISTIC={inst == 0.0}", flush=True)

out["_evidence_class"] = "MEASURED (ours; artifact = this file + both banks)"
out["_reading"] = (
    "HALF 1 answers whether the ENCODER pass is reproducible; HALF 2 answers "
    "whether the PROBE HEAD's training is. A3's floor |AP(D0b)-AP(D0)| is a pure "
    "TRAINING-run measurement only if BOTH are deterministic. Where HALF 2 is "
    "non-zero, that value is an INSTRUMENT floor which every arm gap in this "
    "panel -- including the +0.00173 lever gap A3 is tested against -- inherits, "
    "and no difference below it is a difference.")
json.dump(out, io.open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print("WROTE", sys.argv[1])
