"""D6 -- build the pre-registered token sets of SPEC_P1P2.md from the part-2 outputs (deterministic; seeds literal)."""
import hashlib, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
D = pd.read_csv(os.path.join(RAW, "d6_scene_geometry_step30000.csv")).set_index("token")
assert len(D) == 5912
sets = {}
dz = D[D.A1_DAC == 0]
assert len(dz) == 1563 and dz.resc_ok.all()
sets["S1_dac_cleanpath"] = sorted(dz[dz.ref_dac == 1].index)
sets["S1b_dac_ref_fails"] = sorted(dz[dz.ref_dac == 0].index)
nz = D[D.A1_NC == 0]
sets["S2_nc"] = sorted(nz.index)
ok = D[(D.A1_DAC == 1) & (D.A1_NC == 1) & (D.A1_DDC == 1) & (D.A1_TLC == 1)]   # all four multipliers EXACTLY 1
rng = np.random.default_rng(20261004)
ctrl = sorted(rng.choice(sorted(ok.index), 200, replace=False))
sets["Cpass"] = ctrl
sets["N_pass_population"] = [str(len(ok))]
P1 = sorted(set(sets["S1_dac_cleanpath"]) | set(sets["S1b_dac_ref_fails"]) | set(sets["S2_nc"]) | set(ctrl))
sets["P1_all"] = P1
lr = D[D.cmd.isin(["LEFT", "RIGHT"])]
sets["P2_all"] = sorted(lr.index)
sets["P2_S_prem"] = sorted(lr[~lr.route_turn].index)
sets["P2_S_turn"] = sorted(lr[lr.route_turn].index)
rng2 = np.random.default_rng(20261005)
sets["P2_K1"] = sorted(rng2.choice(sets["P2_all"], 300, replace=False))
sets["P1_K1_fail"] = sorted(rng2.choice(sets["S1_dac_cleanpath"], 100, replace=False))
out = {}
for k, v in sets.items():
    p = os.path.join(RAW, f"spec_tokens_{k}.txt")
    open(p, "w", newline="\n").write("\n".join(v) + "\n")
    out[k] = {"n": len(v), "sha256": hashlib.sha256(("\n".join(v) + "\n").encode()).hexdigest()}
json.dump(out, open(os.path.join(RAW, "spec_token_sets.json"), "w"), indent=1)
for k, v in out.items():
    print(k, v["n"], v["sha256"][:16])
