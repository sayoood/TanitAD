#!/usr/bin/env python3
"""PREREG_SELTTC.md: a TTC-heavier DEPLOYED selection on the REAL banked R0 / R0b / C0 states (A and B), zero training.  -> selttc.json
Rules replace sel_score in BOTH the policy pi (J, P2) and the pick (P1), for BOTH arms, on each arm's own logits.  Controls (i) identity, (ii) reach, (iii) mutation run on R0 u250
before the verdict; if any fails, the verdict is CONTROLS_FAILED and no row is drawn."""
import json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sh_common as SH
import sh_summarize as S
PC = SH.PC
t0 = time.time()
SF, a, holder, cfg, tr, ev = SH.setup(int(os.environ.get("SH_THREADS", "6")))
SEL = ev["train"].sel                                    # the trainer's own sel_score closure (scorer_finetune.py:750-752)
STATES = [("r0", 250), ("r0", 500), ("r0", 1000), ("r0b", 250), ("r0b", 500), ("c0", 250), ("c0", 500), ("c0", 1000)]
CANDS = {"S1_lin": ("lin", 5.0), "S2_gate": ("gate", 1.0), "S3_veto": ("veto", 0.5)}
IDENT = {"S1_lin": ("lin", 0.0), "S2_gate": ("gate", 0.0), "S3_veto": ("veto", 0.0)}
X = 0.5


def make_rule(kind, par):
    def rule(p):
        s = SEL(p)
        if os.environ.get("SELTTC_MUTATE") == "1":
            return s                                     # <- MUTATION (iii): the TTC emphasis is dropped
        ttc = p[..., 3]
        if kind == "lin":                                # <- THE EMPHASIS LINES
            lane = p[..., 6] if (a.lane_head and p.shape[-1] == 7) else 1.0
            return s + p[..., 0] * p[..., 1] * (par * ttc) / 12.0 * lane
        if kind == "gate":
            return s * ttc ** par
        if kind == "veto":
            m = (ttc >= par).to(s.dtype)
            return torch.where(m.sum(-1, keepdim=True) > 0, s * m, s)
        raise KeyError(kind)
    return rule


RULES = {"orig": SEL, **{f"{k}@id": make_rule(*v) for k, v in IDENT.items()}, **{k: make_rule(*v) for k, v in CANDS.items()}}
KEYS = ("J", "pi_comp", "pick_comp", "pick_pdms", "pick_idx")


@torch.no_grad()
def read_state(thA, thB, rules):
    """-> {arm: {rule: {pool: arrays}}}"""
    SH._set(tr.armA, thA); SH._set(tr.armB, thB)
    out = {arm: {r: {} for r in rules} for arm in ("A", "B")}
    for pn, e in ev.items():
        acc = {arm: {r: {k: [] for k in KEYS} for r in rules} for arm in ("A", "B")}
        for ids in e.batches:
            cxy, _c, sctx, _ = e.pool.micro(ids)
            R, tg6 = e.R[ids], e.tg6[ids]
            pdms_all = SF.agg_v1(tg6)
            ar = torch.arange(len(ids))
            for arm, mod in (("A", tr.armA), ("B", tr.armB)):
                sx = mod(cxy, sctx).float()
                for rn, rf in rules.items():
                    pi = SF.epo_policy(sx, rf, a.epo_temp).exp()
                    pick = rf(torch.sigmoid(sx)).argmax(1)
                    o = acc[arm][rn]
                    o["J"].append((pi * R).sum(1)); o["pi_comp"].append((pi[..., None] * tg6).sum(1))
                    o["pick_comp"].append(tg6[ar, pick]); o["pick_pdms"].append(pdms_all[ar, pick]); o["pick_idx"].append(pick.float())
        for arm in acc:
            for rn in acc[arm]:
                out[arm][rn][pn] = {k: torch.cat(v).numpy().astype(np.float64) for k, v in acc[arm][rn].items()}
    return out


def theta(ss, which):
    for m in SF.SCORER:
        getattr(tr.armB, m).load_state_dict(ss[which][m], strict=True)
    return [p.detach().clone() for p in tr.armB.parameters()]


def cat(d, key, col=None):
    return np.concatenate([(d[p][key][:, col] if col is not None else d[p][key]) for p in S.POOLS])


def diff(d1, d0, key, col=None, scale=100.0):
    x = (cat(d1, key, col) - cat(d0, key, col)) * scale
    b = PC.boot_ratio(S.LOGS_ALL, x)
    return {"pooled": b["mean_num"], "ci": b["ci_num"]}


res = {"tool": "sh_selttc/1", "prereg_sha256": open(os.path.join(HERE, "PREREG_SELTTC.sha256")).read().split()[0], "states": {}, "controls": {}}
for si, (run, u) in enumerate(STATES):
    ss = torch.load(os.path.join(PC.AC.DL, f"rls1-{run}", "scorer_states", f"scorer_states_u{u}.pt"), map_location="cpu", weights_only=False)
    thA, thB = theta(ss, "A"), theta(ss, "B")
    E = read_state(thA, thB, RULES)
    # (i) identity on every state, both arms
    res["controls"][f"identity_{run}_u{u}"] = all(np.array_equal(E[arm][f"{c}@id"][p][k], E[arm]["orig"][p][k]) for arm in ("A", "B") for c in CANDS for p in S.POOLS for k in KEYS)
    if si == 0:
        # (ii) reach and (iii) mutation, on R0 u250 arm B
        res["controls"]["reach_sets_changed"] = {c: int((cat(E["B"][c], "pick_idx") != cat(E["B"]["orig"], "pick_idx")).sum()) for c in CANDS}
        os.environ["SELTTC_MUTATE"] = "1"
        try:
            Em = read_state(thA, thB, {"orig": SEL, **{c: make_rule(*CANDS[c]) for c in CANDS}})
        finally:
            os.environ.pop("SELTTC_MUTATE", None)
        res["controls"]["mutation_sets_changed"] = {c: int((cat(Em["B"][c], "pick_idx") != cat(Em["B"]["orig"], "pick_idx")).sum()) for c in CANDS}
        print("controls", res["controls"], f"{time.time()-t0:.0f}s", flush=True)
    A0, B0 = E["A"]["orig"], E["B"]["orig"]
    dJ0 = float(cat(B0, "J").mean() - cat(A0, "J").mean())
    rec = {"dJ_orig": dJ0, "orig": {"P2_TTC_BmA": diff(B0, A0, "pi_comp", 3)}}
    for c in CANDS:
        An, Bn = E["A"][c], E["B"][c]
        rec[c] = {"P2_TTC_g5": diff(Bn, An, "pi_comp", 3), "P1_TTC_g5": diff(Bn, An, "pick_comp", 3), "P2_NC_g5": diff(Bn, An, "pi_comp", 0), "P2_DAC_g5": diff(Bn, An, "pi_comp", 1),
                  "P2_TTC_vs_Aorig": diff(Bn, A0, "pi_comp", 3), "P2_TTC_Anew_vs_Aorig": diff(An, A0, "pi_comp", 3),
                  "PDMS_pick_g5": diff(Bn, An, "pick_pdms"), "PDMS_pick_vs_Aorig": diff(Bn, A0, "pick_pdms"), "PDMS_pick_Bnew_vs_Borig": diff(Bn, B0, "pick_pdms"),
                  "EP_pick_g5": diff(Bn, An, "pick_comp", 2),
                  "rho": float((cat(Bn, "J").mean() - cat(An, "J").mean()) / dJ0), "rho_dep": float((cat(Bn, "J").mean() - cat(A0, "J").mean()) / dJ0)}
    res["states"][f"{run}_u{u}"] = rec
    print(run, u, "orig P2 TTC %+.2f |" % rec["orig"]["P2_TTC_BmA"]["pooled"],
          " | ".join(f"{c}: TTC g5 {rec[c]['P2_TTC_g5']['pooled']:+.2f} rho {rec[c]['rho']:.2f} PDMS g5 {rec[c]['PDMS_pick_g5']['pooled']:+.2f}" for c in CANDS), f"{time.time()-t0:.0f}s", flush=True)

ctrl_ok = all(v for k, v in res["controls"].items() if k.startswith("identity_")) and all(n >= 1 for n in res["controls"]["reach_sets_changed"].values()) \
    and all(n == 0 for n in res["controls"]["mutation_sets_changed"].values())
r0 = [res["states"][f"r0_u{u}"] for u in (250, 500, 1000)]
c0 = [res["states"][f"c0_u{u}"] for u in (250, 500, 1000)]
verdict = {"controls_ok": ctrl_ok}
for c in CANDS:
    nt = sum(s[c]["P2_TTC_g5"]["pooled"] >= -0.5 for s in r0)
    nr = sum(s[c]["rho"] >= X for s in r0)
    nr2 = sum(s[c]["rho"] >= 0.2 for s in r0)
    floor = all(abs(s[c]["P2_TTC_g5"]["pooled"]) < 0.5 for s in c0)
    if not floor:
        row = "INVALID (C0 floor)"
    elif nt >= 2 and nr >= 2:
        row = "SEL-GO"
    elif (nt >= 2 and nr2 >= 2) or (nr >= 2 and nt == 1):
        row = "SEL-PARTIAL"
    else:
        row = "SEL-NO"
    verdict[c] = {"ttc_bar_states": nt, "rho_ge_X_states": nr, "rho_ge_0.2_states": nr2, "c0_floor": floor, "row": row, "min_rho_r0": min(s[c]["rho"] for s in r0)}
gos = [c for c in CANDS if verdict[c]["row"] == "SEL-GO"]
verdict["programme_row"] = "CONTROLS_FAILED" if not ctrl_ok else ("SEL-GO: " + max(gos, key=lambda c: (verdict[c]["min_rho_r0"], -list(CANDS).index(c))) if gos else
                                                                  ("SEL-PARTIAL" if any(verdict[c]["row"] == "SEL-PARTIAL" for c in CANDS) else "SEL-NO"))
res["verdict"] = verdict
json.dump(res, open(os.path.join(HERE, "selttc.json"), "w"), indent=1)
print("ZZSELTTC", json.dumps(verdict), f"{time.time()-t0:.0f}s", flush=True)
