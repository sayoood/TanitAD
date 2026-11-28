"""Score the (metric, W_JERK, W_KAPPA) sweep.

SHAPE PROBES FIRST (they are what detects planning):
  * fraction of windows with kappa IDENTICALLY 0
  * number of DISTINCT plans emitted (bit-exact)
  * triviality by CONTENT (== zeros / == the decel_1.5 block), NOT by label
Then, ONLY for settings that actually plan, the four families vs ha0 and ha,
paired episode-cluster bootstrap on the same windows.
"""
import glob, json, os, sys
import numpy as np
from collections import Counter

SC = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad"
D = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\full_dump"
REPO = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\repo"
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "refav1_arm", os.path.join(REPO, "taniteval", "tools", "refav1_arm.py"))
arm = importlib.util.module_from_spec(spec); sys.modules["refav1_arm"] = arm
spec.loader.exec_module(arm)
import torch

SWEEP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SC, "cost_sweep_main.json")
J = json.load(open(SWEEP))
rows = J["rows"]
setts = {s["label"]: s for s in J["settings"]}
DT = arm.DT
K = 10
print(f"sweep: {J['n_windows']} windows of the {J['eval_grid_windows']}-window "
      f"eval grid (every {J['every']}th), {len(setts)} settings, "
      f"wallclock {J['wallclock_s']/3600:.2f} h")
print(f"plan_cfg: {J['plan_cfg']}")

# ---- banked reference arms for the SAME windows --------------------------- #
files = sorted(glob.glob(os.path.join(D, "ep*.npz")))
bank = {}
for f in files:
    z = np.load(f)
    ep = int(os.path.basename(f)[2:5])
    for j, t in enumerate(z["ws"]):
        bank[(ep, int(t))] = {k: z[k][j] for k in ("g", "cl", "ha", "ha0", "ol")}
        bank[(ep, int(t))]["v0"] = float(z["v0"][j])
print(f"banked windows available: {len(bank)}")

keys = [(r["ep"], r["t"]) for r in rows]
missing = [k for k in keys if k not in bank]
if missing:
    print(f"⛔ {len(missing)} swept windows are not in the banked dump: {missing[:5]}")
    sys.exit(1)
print(f"✅ all {len(keys)} swept windows found in the banked dump (join by ep,t)")

EID = np.array([k[0] for k in keys])
G = np.stack([bank[k]["g"] for k in keys])
REF = {a: np.stack([bank[k][a] for k in keys]) for a in ("cl", "ha", "ha0", "ol")}
V0 = np.array([bank[k]["v0"] for k in keys])
v0_sweep = np.array([r["v0"] for r in rows])
print(f"v0 cross-check dump vs sweep: max |diff| = {np.abs(V0-v0_sweep).max():.3g}")

ZER = np.zeros((J["plan_cfg"]["horizon"], 2), dtype=np.float32)
DEC = ZER.copy(); DEC[:, 0] = -1.5

print("\n" + "="*112)
print("SHAPE PROBES — what the planner actually EMITS  (n = %d windows / %d "
      "episode clusters)" % (len(rows), len(set(EID))))
print("="*112)
print(f"{'setting':18s} {'metric':6s} {'W_JERK':>8s} {'W_KAPPA':>9s} | "
      f"{'kappa==0':>9s} {'distinct':>8s} {'==cv':>6s} {'==decel':>8s} "
      f"{'NON-TRIVIAL':>12s} | {'label:cem':>10s} {'label!=cem':>11s}")
summary = {}
for lbl in [s["label"] for s in J["settings"]]:
    C = np.stack([np.array(r["plans"][lbl]["controls"], dtype=np.float32)
                  for r in rows])
    src = [r["plans"][lbl]["source"] for r in rows]
    k0 = float(np.all(C[..., 1] == 0.0, axis=1).mean())
    uniq = len(np.unique(C.reshape(len(rows), -1), axis=0))
    is_z = np.all(C == ZER, axis=(1, 2)); is_d = np.all(C == DEC, axis=(1, 2))
    triv = is_z | is_d
    s = setts[lbl]
    print(f"{lbl:18s} {s['cost_metric']:6s} {s['W_JERK']:8.3g} "
          f"{s['W_KAPPA']:9.3g} | {k0*100:8.1f}% {uniq:8d} {int(is_z.sum()):6d} "
          f"{int(is_d.sum()):8d} {int((~triv).sum()):12d} | "
          f"{sum(1 for x in src if x=='cem'):10d} "
          f"{sum(1 for x in src if x!='cem'):11d}")
    summary[lbl] = {"controls": C, "k0": k0, "uniq": uniq, "triv": triv,
                    "src": src}

# THE REPRODUCTION CONTROL: S0 is the SHIPPED setting, so its plan must be
# bit-identical to the banked one on every swept window.
BC = np.stack([np.array(bank[k]['cl'], dtype=np.float32) for k in keys])
if 'S0_shipped' in summary:
    C0 = summary['S0_shipped']['controls']
    P0 = np.concatenate([arm.paths_from_controls(torch.as_tensor(C0[i]),
                                                 float(V0[i]), DT, K)
                         .float().numpy() for i in range(len(rows))], axis=0)
    same = np.all(np.abs(P0 - BC) < 1e-6, axis=(1, 2))
    print(f"REPRODUCTION CONTROL - S0_shipped vs the BANKED cl path: "
          f"identical on {int(same.sum())}/{len(rows)} windows "
          f"(max |diff| = {np.abs(P0 - BC).max():.3g} m)")
    if same.sum() != len(rows):
        print('   REFUSE: S0 DOES NOT REPRODUCE THE BANKED RECORD')

print("\n⛔ `baseline_won_frac` (label-based) vs the TRUTH (content-based):")
for lbl in summary:
    C, src, triv = summary[lbl]["controls"], summary[lbl]["src"], summary[lbl]["triv"]
    lab = float(np.mean([x != "cem" for x in src]))
    print(f"  {lbl:18s} label baseline_won_frac {lab:.4f}   "
          f"CONTENT trivial-baseline frac {triv.mean():.4f}   "
          f"cem-labelled but bit-exact a baseline: "
          f"{int(sum(1 for x, t in zip(src, triv) if x=='cem' and t))}"
          f"/{sum(1 for x in src if x=='cem')}")

# the tie mechanism: is the 'cem' plan's cost strictly below every baseline's?
print("\nMECHANISM of the mislabel — for windows labelled `cem` whose controls "
      "ARE a baseline:")
for lbl in summary:
    n_lt, n_eq, deltas = 0, 0, []
    for r in rows:
        p = r["plans"][lbl]
        C = np.array(p["controls"], dtype=np.float32)
        if p["source"] != "cem" or not (np.all(C == ZER) or np.all(C == DEC)):
            continue
        bc = p["baseline_costs"]
        if not bc:
            continue
        mb = min(bc.values())
        deltas.append(p["cost"] - mb)
        if p["cost"] < mb:
            n_lt += 1
        elif p["cost"] == mb:
            n_eq += 1
    if deltas:
        d = np.array(deltas)
        print(f"  {lbl:18s} n={len(d):3d}  cost < min(baseline_costs) on {n_lt}"
              f"  (== on {n_eq})   median delta {np.median(d):+.4g}  "
              f"= {np.median(d)/2**-24:+.2f} float32 ulp at 1.0")

# ---- four families, only for settings that actually plan ------------------- #
print("\n" + "="*112)
print("FOUR FAMILIES — only for settings whose plan is NOT a trivial baseline "
      "on every window")
print("="*112)
comps, tiers = {}, {}
for a in ("ha", "ha0", "ol"):
    comps[a] = arm._components(REF[a], G, DT)
    tiers[a] = "open-loop"
comps["cl_banked"] = arm._components(REF["cl"], G, DT); tiers["cl_banked"] = "open-loop"
for lbl in summary:
    C = summary[lbl]["controls"]
    P = np.concatenate([arm.paths_from_controls(torch.as_tensor(C[i]),
                                               float(V0[i]), DT, K)
                        .float().numpy()
                        for i in range(len(rows))], axis=0)   # [1,K,2] each
    comps[lbl] = arm._components(P, G, DT); tiers[lbl] = "open-loop"

live = [lbl for lbl in summary if summary[lbl]["triv"].mean() < 1.0]
print(f"settings that emit at least one NON-TRIVIAL plan: {live or 'NONE'}")
print(f"(a setting whose plan is a trivial baseline on 100 % of windows has "
      f"nothing to score — its families are identical to ha0/decel by "
      f"construction)\n")

FAM_ORDER = ["ADE", "longitudinal", "lateral", "tactical"]
MK = [k for k in arm._FAMILY_OF]
from taniteval.ci import episode_cluster_bootstrap as _ecb
print("ABSOLUTE per-arm family values (episode-cluster bootstrap, "
      f"n={len(rows)} windows / {len(set(EID))} clusters) -- the floors are "
      "ha0 (constant velocity) and ha (held action); ol replays the RECORDED "
      "actions and g is the GT")
arms_abs = ["ha0", "ha", "ol", "cl_banked"] + live
hdr = "  " + "metric".ljust(26) + "".join(a.rjust(20) for a in arms_abs)
print(hdr)
for fam in FAM_ORDER:
    for mk in [k for k in MK if arm._FAMILY_OF[k] == fam]:
        cells = []
        for a_ in arms_abs:
            v = comps[a_][mk]
            keep = np.isfinite(v)
            if keep.sum() == 0:
                cells.append("n/a".rjust(20)); continue
            r = _ecb(v[keep], EID[keep], reduce="mean", n_boot=4000, seed=0, dp=6)
            cells.append(f"{r[chr(39)+chr(39)] if False else r['mean']:.4f}".rjust(20))
        print(f"  {fam[:3]+chr(58)+mk:26s}" + "".join(cells))
print()


for lbl in ["cl_banked"] + live:
    for ref in ("ha0", "ha"):
        r = arm._paired_families(comps, ref, lbl, EID, tiers, 4000, 0)
        print(f"--- {lbl}  minus  {ref}   (paired episode-cluster bootstrap, "
              f"n={len(rows)} windows / {len(set(EID))} clusters) ---")
        for fam in FAM_ORDER:
            for mk, rr in r["families"].get(fam, {}).items():
                if "delta" not in rr:
                    print(f"  {fam:13s} {mk:26s} REFUSED: {rr.get('reason')}")
                    continue
                flag = "SEPARATED" if rr["separated"] else "straddles 0"
                if rr.get("degenerate"):
                    flag = "DEGENERATE"
                print(f"  {fam:13s} {mk:26s} {rr['delta']:+11.6f} "
                      f"[{rr['lo']:+11.6f}, {rr['hi']:+11.6f}] "
                      f"{flag:11s} n={rr['n_windows']}/"
                      f"{rr['n_episodes']}cl  p(>0)={rr['p_delta_gt0']}")
        print()
