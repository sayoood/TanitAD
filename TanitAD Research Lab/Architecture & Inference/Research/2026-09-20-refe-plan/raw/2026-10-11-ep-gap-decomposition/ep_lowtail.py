"""Characterise u400's low-EP tail and decompose the PDMS gap to the human (0 GPU, banked hooks)."""
import json, numpy as np
S = "E:/Projects/TanitAD/data/refe_navtest/score"
def load(d):
    h = json.load(open(f"{S}/{d}/{d}_hooks.json", encoding="utf-8"))["pdm_score_calls"]
    return {c["token"]: c for c in h}
U, H, T = load("refe_stage2_sft4_B_u400_navtest_full"), load("refe_p4_human_navtest"), load("refe_p4_teacher_navtest")
toks = sorted(U)
def f(D, k): return np.array([D[t]["row"][k] for t in toks], float)
def plen(D):
    out = []
    for t in toks:
        p = np.array(D[t]["agent_poses"], float)[:, :2]
        p = np.vstack([[0, 0], p]); out.append(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    return np.array(out)
def end_heading(D): return np.array([np.array(D[t]["agent_poses"], float)[-1, 2] for t in toks])
def lat_end(D): return np.array([np.array(D[t]["agent_poses"], float)[-1, 1] for t in toks])
v0 = np.array([U[t]["v0_mps"] for t in toks]); ref = np.array([U[t]["max_compliant_progress_m"] for t in toks])
pu = np.array([U[t]["progress_raw_m"][-1] for t in toks]); ph = np.array([H[t]["progress_raw_m"][-1] for t in toks])
Lu, Lh = plen(U), plen(H)
# --- gap decomposition vs human (exact, sequential: gate first, then EP/TTC/C on the agent's gate)
def parts(D):
    nc, dac, ep, ttc, c = (f(D, k) for k in ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                                              "time_to_collision_within_bound", "comfort"))
    return nc * dac, ep, ttc, c
gu, epu, ttu, cu = parts(U); gh, eph, tth, ch = parts(H)
wu = (5 * epu + 5 * ttu + 2 * cu) / 12; wh = (5 * eph + 5 * tth + 2 * ch) / 12
tot = (gh * wh - gu * wu).mean() * 100
gate_term = ((gh - gu) * wh).mean() * 100
ep_term = (gu * 5 * (eph - epu) / 12).mean() * 100; ttc_term = (gu * 5 * (tth - ttu) / 12).mean() * 100
c_term = (gu * 2 * (ch - cu) / 12).mean() * 100
print(f"human - u400 = {tot:.2f} PDMS = gate(NC*DAC zeros) {gate_term:.2f} + EP {ep_term:.2f} + TTC {ttc_term:.2f} + C {c_term:.2f}")
nc_u, dac_u = f(U, "no_at_fault_collisions"), f(U, "drivable_area_compliance")
print(f"u400 zeros: DAC=0 on {np.mean(dac_u==0)*100:.2f}%  NC=0 on {np.mean(nc_u==0)*100:.2f}%  NC=0.5 on {np.mean(nc_u==0.5)*100:.2f}%  TTC=0 on {np.mean(ttu==0)*100:.2f}% (human TTC=0 {np.mean(tth==0)*100:.2f}%)")
# --- low-EP tail
low = epu < 0.5
print(f"\nlow-EP tokens (EP<0.5): {low.sum()} | gated-zero among them {np.mean(gu[low]==0)*100:.1f}% | v0 median {np.median(v0[low]):.2f} "
      f"| ref median {np.median(ref[low]):.1f} m | u400 progress median {np.median(pu[low]):.1f} | human progress median {np.median(ph[low]):.1f}")
print(f"   plan path length median u400 {np.median(Lu[low]):.1f} m vs human {np.median(Lh[low]):.1f} m; "
      f"u400 path >= 0.8*human path on {np.mean(Lu[low] >= 0.8*Lh[low])*100:.1f}% (i.e. it MOVES, but not along the route)")
ratio = pu / np.maximum(Lu, 1e-3)
print(f"   progress/path-length median: low-EP {np.median(ratio[low]):.2f} vs rest {np.median(ratio[~low & (Lu>5)]):.2f}")
hd = np.abs(np.degrees(end_heading(U) - end_heading(H))); hd = np.minimum(hd, 360 - hd)
print(f"   |final heading u400 - human| median {np.median(hd[low]):.1f} deg (rest {np.median(hd[~low]):.1f}); "
      f">30 deg on {np.mean(hd[low] > 30)*100:.1f}% (rest {np.mean(hd[~low] > 30)*100:.1f}%)")
ly = np.abs(lat_end(U) - lat_end(H))
print(f"   |final lateral u400 - human| median {np.median(ly[low]):.1f} m (rest {np.median(ly[~low]):.1f})")
# --- moderate deficit band, gated tokens only, EP in [0.5, 0.95): slow or short?
mid = (epu >= 0.5) & (epu < 0.95) & (gu > 0)
print(f"\nmid band (0.5<=EP<0.95, not zeroed): {mid.sum()} | median u400/human progress {np.median(pu[mid]/np.maximum(ph[mid],1e-3)):.3f} "
      f"| median u400/human path length {np.median(Lu[mid]/np.maximum(Lh[mid],1e-3)):.3f} | human EP there {eph[mid].mean()*100:.1f} vs u400 {epu[mid].mean()*100:.1f}")
# EP points by band, gated (the term that enters PDMS)
for name, m in (("EP<0.5", low), ("0.5-0.95", (epu >= .5) & (epu < .95)), (">=0.95", epu >= .95)):
    print(f"   {name:9s} PDMS-weighted EP deficit vs human: {(gu*5*(eph-epu)/12)[m].sum()/len(toks)*100:+.2f}")
json.dump({"low_ep_tokens": [t for t, m in zip(toks, low) if m]}, open("low_ep_tokens.json", "w"))
