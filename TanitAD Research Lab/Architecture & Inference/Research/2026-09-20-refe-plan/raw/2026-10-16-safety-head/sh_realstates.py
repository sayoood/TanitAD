#!/usr/bin/env python3
"""PREREG_SAFEHEAD_A2_REALSTATES.md: the REAL banked R0 / R0b / C0 scorer states (A and B) read through the safety-head selections on the 364 cached sets.  Zero training.  -> realstates.json"""
import json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sh_common as SH
import sh_summarize as S
PC = SH.PC
t0 = time.time()
SF, a, holder, cfg, tr, ev = SH.setup(int(os.environ.get("SH_THREADS", "4")))
PV = json.load(open(os.path.join(SH.PHD, "proxy_validity.json")))["states"]
STATES = [("r0", 250), ("r0", 500), ("r0", 1000), ("r0b", 250), ("r0b", 500), ("c0", 250), ("c0", 500), ("c0", 1000)]
out = {"tool": "sh_realstates/1", "states": {}}


def state_theta(ss, which):
    for m in SF.SCORER:
        getattr(tr.armB, m).load_state_dict(ss[which][m], strict=True)
    return [p.detach().clone() for p in tr.armB.parameters()]


def keep(px):
    return {m: {"pooled": px[m]["pooled"], "ci": px[m]["pooled_ci"], "train": px[m].get("train"), "val": px[m].get("val")} for m in px}


for run, u in STATES:
    ss = torch.load(os.path.join(PC.AC.DL, f"rls1-{run}", "scorer_states", f"scorer_states_u{u}.pt"), map_location="cpu", weights_only=False)
    thA, thB = state_theta(ss, "A"), state_theta(ss, "B")
    E = {pn: SH.eval_views(e, tr, thB, thA, ["own", "SH_A", "SH_F", "A_own", "SH_F_ctrl"]) for pn, e in ev.items()}
    V = lambda v: {pn: E[pn][v] for pn in S.POOLS}
    full, sha, shf = S.prox(V("own"), V("A_own")), S.prox(V("SH_A"), V("A_own")), S.prox(V("SH_F"), V("SH_F_ctrl"))
    dF = np.concatenate([V("own")[p]["J"] - V("A_own")[p]["J"] for p in S.POOLS]).mean()
    rec = {"FULL": keep(full), "SH_A_sel": keep(sha), "SH_F_sel": keep(shf),
           "f_sel_SH_A": float(np.concatenate([V("SH_A")[p]["J"] - V("A_own")[p]["J"] for p in S.POOLS]).mean() / dF),
           "f_sel_SH_F": float(np.concatenate([V("SH_F")[p]["J"] - V("SH_F_ctrl")[p]["J"] for p in S.POOLS]).mean() / dF),
           "dJ_FULL": float(dF),
           "identity_vs_phead_P2_TTC_x100": abs(full["P2_TTC"]["pooled"] - PV[f"{run}_u{u}"]["P2_pi_TTC_B_minus_A_x100"]["mean"])}
    out["states"][f"{run}_u{u}"] = rec
    print(run, u, "P2 TTC FULL %+.2f SH_A %+.2f SH_F %+.2f | P1 TTC FULL %+.2f SH_A %+.2f SH_F %+.2f | f_sel A %.2f F %.2f | id %.1e | %ds" % (
        full["P2_TTC"]["pooled"], sha["P2_TTC"]["pooled"], shf["P2_TTC"]["pooled"], full["P1_TTC"]["pooled"], sha["P1_TTC"]["pooled"], shf["P1_TTC"]["pooled"],
        rec["f_sel_SH_A"], rec["f_sel_SH_F"], rec["identity_vs_phead_P2_TTC_x100"], time.time() - t0), flush=True)
r0 = [out["states"][f"r0_u{u}"] for u in (250, 500, 1000)]
c0 = [out["states"][f"c0_u{u}"] for u in (250, 500, 1000)]
floor_ok = all(abs(s[v]["P2_TTC"]["pooled"]) < 0.5 for s in c0 for v in ("FULL", "SH_A_sel", "SH_F_sel"))
ident_ok = all(s["identity_vs_phead_P2_TTC_x100"] <= 1e-5 for s in out["states"].values())


def row(view):
    n = sum(s[view]["P2_TTC"]["pooled"] >= -0.5 for s in r0)
    return ("A2-SAFETY-ROWS" if n >= 2 else "A2-PROGRESS-ROWS") + f" ({n}/3 R0 states with P2 TTC >= -0.5)"


out["verdict"] = {"C0_floor_ok": floor_ok, "identity_ok": ident_ok, "row_SH_A": row("SH_A_sel") if floor_ok else "INVALID (C0 floor)", "row_SH_F": row("SH_F_sel") if floor_ok else "INVALID (C0 floor)"}
json.dump(out, open(os.path.join(HERE, "realstates.json"), "w"), indent=1)
print("ZZREAL", out["verdict"], f"{time.time()-t0:.0f}s", flush=True)
