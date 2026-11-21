"""summarise_full.py -- print the decision tables of out/READ_FULL.json (read_otband.py output); reads, writes nothing.
Usage: python summarise_full.py [out/READ_FULL.json]"""
import json
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "out/READ_FULL.json"
d = json.load(open(P, encoding="utf-8"))
OP, TAC = "operative_0_2s", "tactical_2_6s"
ROWS = ("ade_m", "fde_m", "speed_mae_mps", "accel_mae_mps2", "along_mae_m", "dk_hw_abs_err_m", "dk_tg_abs_err_s",
        "cross_mae_m", "heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm")


def f(r):
    if not r or r.get("status") != "OK":
        return "n/a"
    return f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{'*' if r.get('separated') else ''}"


def m(r):
    return "n/a" if not r or r.get("status") != "OK" else f"{r['mean']:.4f}"


print("parity", d["parity"], "| outcome:", d.get("outcome"))
for s in ("s0", "s1"):
    sd = d["seeds"].get(s, {})
    if "bars" not in sd:
        print(s, sd.get("status"), sd.get("reason"))
        continue
    b = sd["bars"]
    print(f"\n== {s} label {b['label']} | hold {b['hold']} | B2 worse {b['B2_separated_worse']} | B6 {b['B6_exceeds_floor']}")
    print("   B3", b["B3"], "\n   B5", b["B5"])
    for band in (OP, TAC):
        rows = sd["bands"][band]["rows"]
        print(f"  -- {band}: BAND-CTL | CTL-DAG | CTL mean / BAND / DAG")
        for k in ROWS:
            if k in rows:
                r = rows[k]
                print(f"   {k:22s} {f(r['BAND_minus_CTL']):34s} {f(r['CTL_minus_DAG']):34s} {m(r['OB-CTL'])} / {m(r['OB-BAND'])} / {m(r['DAG-OACC'])}")
        print(f"  -- {band} ladder (beside): INIT | DAG0 | DAG | DAG-OACC means; step1 sched, step2 DAgger rows, step3 w_acc")
        for k in ROWS:
            r = rows.get(k, {})
            L, B = r.get("ladder_post_hoc"), r.get("beside_post_hoc")
            if L and B:
                print(f"   {k:22s} {m(B['INIT-M2b'])} | {m(L['DAG0'])} | {m(L['DAG'])} | {m(r['DAG-OACC'])} ; "
                      f"{f(L['step1_schedule_DAG0_minus_INIT'])} ; {f(L['step2_dagger_rows_DAG_minus_DAG0'])} ; {f(L['step3_w_acc_DAGOACC_minus_DAG'])}"
                      f" ; CTL-DAG0 {f(L['CTL_minus_DAG0'])} ; CTL-INIT {f(B['CTL_minus_INIT'])}")
    for nm, t in sd["TACTICAL_plan_reads"].items():
        print(f"  {nm}: CTL {m(t['OB-CTL'])} BAND {m(t['OB-BAND'])} DAG {m(t['DAG-OACC'])} BAND-CTL {f(t['BAND_minus_CTL'])}", end="")
        L = t.get("ladder_post_hoc")
        if L:
            print(f" | INIT {m(L['INIT-M2b'])} DAG0 {m(L['DAG0'])} DAG {m(L['DAG'])} steps {f(L['step1'])} {f(L['step2'])} {f(L['step3'])}")
        else:
            print()
rep = d.get("replicate")
if rep:
    print("\n== replicate REP - BAND_s0 (floor = |delta|)")
    for band in (OP, TAC):
        print("  ", band, {k: f(rep["rows_REP_minus_BAND_s0"][band][k]) for k in ("ade_m", "fde_m", "speed_mae_mps", "along_mae_m", "yaw_rate_mae_degps")})
