#!/usr/bin/env python3
"""read_pess_a1.py -- the read of PREREG_PESS_P6_A1 (frozen sha256 e2a4452c...): arm LCB1ND = mean - 1 SD on NC and DAC only.
Reuses read_pess.pess_read unchanged; adds the LCB1ND combine rule, captures the shared bootstrap for bar E, and decides A1-1/A1-2/A1-3.
CPU only.  python read_pess_a1.py --out DIR [--no-ledger] --member NAME=DIR ... (the same eight members as the PESS read)

Fail-closed checks: the A1 card's sha256; ENS9 and LCB1 logits must be bit-identical (sha256) to the PESS read's (so only the rule
changed and their ledger keys add nothing); deliberate regression: LCB1ND with TTC put back must equal LCB1 bit for bit."""
import argparse, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_pess as RP  # noqa: E402

R = RP.R
A1_CARD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "PREREG_PESS_P6_A1.md")
A1_SHA = "e2a4452c1b81858bd39a8f09b6fdc16c91dfcc117e3feaa90e9fcb809c5173f1"
PESS_READ_JSON = os.path.join(HERE, "real", "pess_read.json")
NC, DAC, TTC = RP.HEAD["NC"], RP.HEAD["DAC"], RP.HEAD["TTC"]
_orig_combine = RP.combine


def lcb_on(members, cols):
    S = np.stack(RP._stack(members), axis=0)
    mean = np.mean(S, axis=0, dtype=np.float32)
    out = mean.copy()
    sd = np.std(S[..., cols].astype(np.float64), axis=0, ddof=0)
    out[..., cols] = (mean[..., cols].astype(np.float64) - 1.0 * sd).astype(np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def combine(members, rule):
    if rule == "LCB1ND":
        return lcb_on(members, [NC, DAC])
    return _orig_combine(members, rule)


def decide_a1(p1, p2, m, e, guards):
    labels = []
    if p1 and p2 and m and guards:
        labels.append("A1-1")
    if e and not p1:
        labels.append("A1-2")
    if (e is False) and not p1:
        labels.append("A1-3")
    names = {"A1-1": "ADOPT", "A1-2": "EP RECOVERED, BELOW BAR", "A1-3": "TTC NOT THE EP COST"}
    prim = labels[0] if labels else "Unnamed"
    return {"label": prim, "name": names.get(prim, "Unnamed"), "matches": labels or ["Unnamed"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--p6", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default=RP.DEFAULT_LEDGER)
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args(argv)
    act = hashlib.sha256(open(A1_CARD, "rb").read()).hexdigest()
    if act != A1_SHA:
        print(f"ZZA1_REFUSED card sha256 {act} != frozen {A1_SHA}"); return 2
    prev = json.load(open(PESS_READ_JSON, encoding="utf-8"))
    members = RP.parse_members(a.member)
    if [n for n, _ in members] != prev.get("members"):
        print("ZZA1_REFUSED member list differs from the PESS read's"); return 2
    RP.combine = combine
    cap = {}
    _BS = R.BootSet

    def BS(*x, **k):  # keep the E3 bootstrap (built on the 12,146-token log array), not the later E2 ones
        b = _BS(*x, **k)
        if len(x[0] if x else k["log"]) == 12146 and "B3" not in cap:
            cap["B3"] = b
        return b
    R.BootSet = BS
    res = RP.pess_read(members, a.p6, arms=("ENS9", "LCB1", "LCB1ND"), ledger_path=None if a.no_ledger else a.ledger)
    if res.get("status") != "READ":
        print("ZZA1_STOPPED", res.get("why")); return 2
    sh = res["arm_logits_sha256"]
    chk = {"ENS9_sha_equals_PESS_read": sh["ENS9"] == prev["arm_logits_sha256"]["ENS9"],
           "LCB1_sha_equals_PESS_read": sh["LCB1"] == prev["arm_logits_sha256"]["LCB1"],
           "RED_LCB1ND_plus_TTC_equals_LCB1_bits": bool(np.array_equal(lcb_on(res["_mem_logits"], [NC, DAC, TTC]), res["_comb"]["LCB1"])),
           "LCB1ND_differs_from_LCB1_picks": int((res["_pick"]["LCB1ND"] != res["_pick"]["LCB1"]).sum())}
    if not (chk["ENS9_sha_equals_PESS_read"] and chk["LCB1_sha_equals_PESS_read"] and chk["RED_LCB1ND_plus_TTC_equals_LCB1_bits"]):
        print("ZZA1_CHECK_FAIL", json.dumps(chk)); return 2
    B3, truth = cap["B3"], res["_truth"]
    _, s_nd = truth.lookup(res["_pick"]["LCB1ND"]); _, s_l1 = truth.lookup(res["_pick"]["LCB1"])
    pt, rr, n = B3.diff(s_nd[:, RP.EP_COL] * 100.0, s_l1[:, RP.EP_COL] * 100.0)
    E = R.rec(pt, rr, n); E["clears"] = bool(E["lo"] > 0)
    o = res["arms"]["LCB1ND"]
    p1, p2, m, g = o["P1_vs_P6"]["clears"], o["P2_vs_ENS9"]["clears"], o["M_zero_pick_share_minus_ENS9"]["clears"], o["G"]["ok"]
    res["a1"] = {"card": {"path": A1_CARD, "sha256": act}, "checks": chk, "E_EP_x100_LCB1ND_minus_LCB1": E,
                 "outcome": decide_a1(p1, p2, m, E["clears"], g), "inputs": {"P1": p1, "P2": p2, "M": m, "E": E["clears"], "guards": g},
                 "goal_status_LCB1ND": R.goal_status(o["PDMS_x100_E3"]["point"], o["PDMS_x100_E3"]["hi"])}
    os.makedirs(a.out, exist_ok=True)
    pub = {k: v for k, v in res.items() if not k.startswith("_")}
    with open(os.path.join(a.out, "pess_a1_read.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(pub, f, indent=1, default=R.jdefault)
    L = ["# PESS A1 read (LCB1ND)", "", f"- Card A1 sha256 `{act}`; base card `{RP.CARD_SHA256}`", f"- Members ({len(members)}): {', '.join(n for n, _ in members)}",
         f"- Checks: {json.dumps(chk)}", "", "| arm | PDMS | minus P6 | EP x100 minus P6 | zero-pick share | P2 vs ENS9 | M vs ENS9 | guards | adverse |", "|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda r: f"{r['point']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
    for nm in ("ENS9", "LCB1", "LCB1ND"):
        q = res["arms"][nm]; lv = q["PDMS_x100_E3"]
        L.append(f"| {nm} | {lv['point']:.3f} [{lv['lo']:.3f}, {lv['hi']:.3f}] | {f2(q['minus_P6'])} | {f2(q['EP_x100_E3']['minus_P6'])} | {q['zero_pick_share']:.5f} | "
                 + (f"{f2(q['P2_vs_ENS9'])} {q['P2_vs_ENS9']['clears']}" if "P2_vs_ENS9" in q else "-") + " | "
                 + (f"{f2(q['M_zero_pick_share_minus_ENS9'])} {q['M_zero_pick_share_minus_ENS9']['clears']}" if "M_zero_pick_share_minus_ENS9" in q else "-")
                 + f" | {q['G']['ok']} | {q['adverse_components']} |")
    L += ["", f"- P1 LCB1ND vs P6: {f2(o['P1_vs_P6'])}, floor 0.4971, clears {p1}", f"- E: EP x100 LCB1ND - LCB1 {f2(E)}, clears {E['clears']}",
          f"- **Outcome: {res['a1']['outcome']['label']} {res['a1']['outcome']['name']}** (matches {res['a1']['outcome']['matches']}); goal status {res['a1']['goal_status_LCB1ND']}",
          f"- K ledger: {json.dumps({k: v for k, v in res['K_ledger'].items() if k in ('K', 'keys', 'this_read_added')})}",
          "- STRATEGIC: PAUSED (PI 2026-09-19, PI_DECISION_QUEUE item 25); not a pass, not omitted.",
          "- Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927 (another draw of logs only; fixed members, deterministic inference). Tier: T1-class open loop, full navtest E3."]
    with open(os.path.join(a.out, "pess_a1_read.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("ZZA1_DONE", res["a1"]["outcome"]["label"], "K", res["K_ledger"].get("K"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
