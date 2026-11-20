#!/usr/bin/env python3
"""P5-C gap decomposition (C1 vs human, C0 vs human, P7/base vs human, C1 vs C0), full navtest, tracked harness CSVs.

SAME EXACT METHOD as D-P6-GAP-1 (raw/2026-11-09-p6-navtest-read/gap_to_human/decomp_p6.py, itself = D-REFE-GAP-1b ep_lowtail.py):
  PDMS = g * r,  g = NC*DAC,  r = (5 EP + 5 TTC + 2 C) / 12   (NAVSIM v1.1; asserted per token against the CSV score)
  ORDER A (the banked order): gate term on the REFERENCE's r, then EP/TTC/C on the ARM's gate
      gate = mean((gR - gA) * rR); ep = mean(gA * 5 (epR - epA) / 12); ttc, comfort alike
  ORDER B (the other order): gate on the ARM's r, sub-terms on the REFERENCE's gate
CONTROLS (both must hold or exit 1): u400 reproduces 6.30 = 4.15 / 0.86 / 1.31 / -0.01 with 529 zeros (banked in decomp_p6.py),
  and P6 reproduces 3.35 = 3.04 / -1.20 / 1.53 / -0.02 with 384 zeros (banked p6_gap_to_human.json).
Point decomposition, T1-class (ego pseudo-sim of the open-loop plan, NAVSIM tracker), one training seed, no interval.
Extras: the gate term split by cause (DAC=0 only, NC=0, NC=0.5, both) and C1's zero tokens split by whether C0 is also zero.
Usage: gap_decomp_p5c.py OUT.json   (CPU, reads the banked harness CSVs only)."""
import csv, glob, json, sys
import numpy as np

R = "E:/Projects/TanitAD/data/refe_navtest/score/"
KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "score")


def load(name):
    fs = [f for f in glob.glob(R + name + "/*.csv") if "_logs" not in f]
    assert len(fs) == 1, (name, fs)
    out = {}
    for r in csv.DictReader(open(fs[0], newline="", encoding="utf-8")):
        if r["valid"] == "True" and len(r["token"]) == 16:
            out[r["token"]] = [float(r[k]) for k in KEYS]
    return fs[0].replace("\\", "/"), out


def parts(M):
    g = M[:, 0] * M[:, 1]
    r = (5 * M[:, 2] + 5 * M[:, 3] + 2 * M[:, 4]) / 12
    assert np.abs(g * r - M[:, 5]).max() < 1e-9
    return g, r


def decomp(Ref, A):
    gR, rR = parts(Ref)
    gA, rA = parts(A)
    gap = (Ref[:, 5] - A[:, 5]).mean() * 100
    oa = dict(gate=((gR - gA) * rR).mean() * 100,
              ep=(gA * 5 * (Ref[:, 2] - A[:, 2]) / 12).mean() * 100,
              ttc=(gA * 5 * (Ref[:, 3] - A[:, 3]) / 12).mean() * 100,
              comfort=(gA * 2 * (Ref[:, 4] - A[:, 4]) / 12).mean() * 100)
    ob = dict(gate=((gR - gA) * rA).mean() * 100,
              ep=(gR * 5 * (Ref[:, 2] - A[:, 2]) / 12).mean() * 100,
              ttc=(gR * 5 * (Ref[:, 3] - A[:, 3]) / 12).mean() * 100,
              comfort=(gR * 2 * (Ref[:, 4] - A[:, 4]) / 12).mean() * 100)
    # gate term (order A) split by cause on the ARM's side
    nc, dac = A[:, 0], A[:, 1]
    cause = {"dac0_nc1": (dac == 0) & (nc == 1), "nc0_dac1": (nc == 0) & (dac == 1), "both0": (nc == 0) & (dac == 0),
             "nc05_dac1": (nc == 0.5) & (dac == 1), "nc05_dac0": (nc == 0.5) & (dac == 0)}
    gt = (gR - gA) * rR
    split = {k: dict(n=int(m.sum()), term=float(gt[m].sum() / len(gt) * 100)) for k, m in cause.items()}
    split["other_gate_diff"] = dict(n=int(((gR != gA) & ~np.any(np.stack(list(cause.values())), 0)).sum()),
                                    term=float(gt[~np.any(np.stack(list(cause.values())), 0)].sum() / len(gt) * 100))
    gated = gA > 0
    return dict(arm=A[:, 5].mean() * 100, ref=Ref[:, 5].mean() * 100, gap=gap,
                order_A=dict(oa, closure=sum(oa.values()) - gap), order_B=dict(ob, closure=sum(ob.values()) - gap),
                gate_split_orderA=split,
                zeros=int((A[:, 5] == 0).sum()), zero_pct=float((A[:, 5] == 0).mean() * 100),
                dac0_pct=float((dac == 0).mean() * 100), nc0_pct=float((nc == 0).mean() * 100),
                nc05_pct=float((nc == 0.5).mean() * 100), ttc0_pct=float((A[:, 3] == 0).mean() * 100),
                ttc0_on_gated_pct=float((A[gated, 3] == 0).mean() * 100), ref_ttc0_pct=float((Ref[:, 3] == 0).mean() * 100),
                ep_on_gated=float(A[gated, 2].mean() * 100), ref_ep_same_tokens=float(Ref[gated, 2].mean() * 100),
                ref_zeros=int((Ref[:, 5] == 0).sum()))


names = {"human": "refe_p4_human_navtest", "u400": "refe_stage2_sft4_B_u400_navtest_full", "P6": "refe_p6final",
         "P7_base": "refe_stage2_p7_full", "C1": "refe_stage2_p5c_c1", "C0": "refe_stage2_p5c_c0"}
D, csvs = {}, {}
for k, n in names.items():
    csvs[k], D[k] = load(n)
toks = sorted(D["human"])
assert len(toks) == 12146
for k in D:
    assert set(D[k]) == set(toks), (k, len(D[k]))
M = {k: np.array([D[k][t] for t in toks]) for k in D}

res = {"method": __doc__.split("\n")[2].strip(), "csv": csvs, "n": len(toks)}
for arm in ("u400", "P6", "P7_base", "C0", "C1"):
    res[arm + "_vs_human"] = decomp(M["human"], M[arm])
# C1 vs C0: REFERENCE = C1 (the better tracked arm), ARM = C0, so gap = +1.17 and each term is what C1 gained over C0
res["C1_over_C0"] = decomp(M["C1"], M["C0"])
res["C1_over_P7_base"] = decomp(M["C1"], M["P7_base"])
# zero-token overlap
z1, z0, zb = M["C1"][:, 5] == 0, M["C0"][:, 5] == 0, M["P7_base"][:, 5] == 0
res["zero_overlap"] = dict(C1=int(z1.sum()), C0=int(z0.sum()), P7_base=int(zb.sum()), C1_and_C0=int((z1 & z0).sum()),
                           C1_only=int((z1 & ~z0).sum()), C0_only=int((z0 & ~z1).sum()),
                           C1_or_C0=int((z1 | z0).sum()), C1_and_C0_and_base=int((z1 & z0 & zb).sum()))

u, p = res["u400_vs_human"], res["P6_vs_human"]
ua, pa = u["order_A"], p["order_A"]
ok_u = abs(u["gap"] - 6.30) < 0.006 and abs(ua["gate"] - 4.15) < 0.006 and abs(ua["ep"] - 0.86) < 0.006 and abs(ua["ttc"] - 1.31) < 0.006 and abs(ua["comfort"] + 0.01) < 0.006 and u["zeros"] == 529
ok_p = abs(p["gap"] - 3.35) < 0.006 and abs(pa["gate"] - 3.04) < 0.006 and abs(pa["ep"] + 1.20) < 0.006 and abs(pa["ttc"] - 1.53) < 0.006 and abs(pa["comfort"] + 0.02) < 0.006 and p["zeros"] == 384
res["control_u400_reproduces_banked"] = bool(ok_u)
res["control_P6_reproduces_banked"] = bool(ok_p)
json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
for k in ("u400_vs_human", "P6_vs_human", "P7_base_vs_human", "C0_vs_human", "C1_vs_human", "C1_over_C0", "C1_over_P7_base"):
    v = res[k]
    a, b = v["order_A"], v["order_B"]
    print(f"{k:18s} gap {v['gap']:+.2f} | A gate {a['gate']:+.2f} ep {a['ep']:+.2f} ttc {a['ttc']:+.2f} c {a['comfort']:+.2f} | "
          f"B gate {b['gate']:+.2f} ep {b['ep']:+.2f} ttc {b['ttc']:+.2f} c {b['comfort']:+.2f} | zeros {v['zeros']} dac0 {v['dac0_pct']:.2f}% "
          f"nc0 {v['nc0_pct']:.2f}% ttc0 {v['ttc0_pct']:.2f}% ep_gated {v['ep_on_gated']:.2f} ref_same {v['ref_ep_same_tokens']:.2f}")
print("gate split C1 vs human:", json.dumps(res["C1_vs_human"]["gate_split_orderA"]))
print("gate split C0 vs human:", json.dumps(res["C0_vs_human"]["gate_split_orderA"]))
print("zero overlap:", res["zero_overlap"])
print("controls u400/P6:", ok_u, ok_p)
sys.exit(0 if (ok_u and ok_p) else 1)
