"""AMEND_HWMPLAN_7 SCORE readout (sections 6-7): AMEND_5's score() on the amend7 folders (RA7 arms, A0 base, MEAN-L1 reference, HOLD), plus Delta_7 = RA7-L1 - RA6-L1, the readout-B floor F_T,score, Floor_k = max(F_I, F_T),
the separated-better / WORSE rules recomputed with Floor_k and the readout-B point (card 6.3), CEIL-ORACLE7 and the section 7 outcome. Runs only after R7-FIT-PASS.
  python amend7_read.py train --chassis k   AMEND_5's TRAIN-stage report (beside, in-sample; C-SHUF7 sign and C-PLANT7 ii) on the amend7 TRAIN arms
  python amend7_read.py score               -> amend7/readout/RESULT_AMEND7.json + .md  (AMEND_5's full output stays beside as RESULT_AMEND7_full.*)
Tier T0. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, 994 windows in 71 episodes; it answers another draw of EPISODES only (F_I: another inference run; F_T: another readout-training run)."""
import amend7_env  # noqa: F401
import json, os, shutil, sys
sys.dont_write_bytecode = True
from amend7_env import ROOT, ST, A5, A6
import amend5_read as RD

STAMP = dict(tier="T0", evidence="MEASURED", estimator="paired_episode_cluster_bootstrap B 2000 seed 0 (another draw of EPISODES); F_I another inference run; F_T another readout-training run")
CELLS = (("LONGITUDINAL", "0-2s", "acc_mae"), ("LATERAL", "0-2s", "xtrack"))


def sep_better7(row, floor, seed1_delta, readout_b_delta):
    """Separated-better beyond Floor_k: the CI wholly below 0, |delta| > Floor_k, and both the seed-1 point and the readout-B point below 0 (card 6.3)."""
    return bool(RD.sep_better(dict(row), floor, max(seed1_delta, readout_b_delta)))


def worse7(row, floor):
    return bool(RD.worse(dict(row), floor))


def lever_input7(js_base, js_mean, js_76, js_bx, D):
    """The AMEND_5 label inputs for RA7-L1 with Floor_k = max(F_I,k, F_T,k) per row and the readout-B point folded into the seed-1 condition. js_base: RA7-L1 vs (A0, HOLD); js_mean: vs MEAN-L1; js_76: vs RA6-L1 (its reps = RA7 and RA6
    inference seeds, so F_I,k = max(|RA7-L1 s1 - s0|, |RA6-L1 s1 - s0|)); js_bx: RA7-L1-B (a) vs RA7-L1 (b) at inference seed 0 (F_T,k = |a - b|)."""
    def row_inputs(fam, band, met):
        c = RD.cell(js_base, fam, band, met)
        FI = RD.cell(js_76, fam, band, met)["F"]
        bx = RD.find_row(js_bx, "contrast", fam, band, met, "A", "B")["delta"]
        return c, max(FI, abs(bx)), bx, FI, abs(bx)
    c0, F0, bx0, FI0, FT0 = row_inputs(*CELLS[0])
    cg, Fg, bxg, _, _ = row_inputs(*CELLS[1])
    cm = RD.cell(js_mean, *CELLS[0])
    return dict(dH=c0["dH"], dH_F=F0, dH_seed1=max(c0["dH_seed1"], c0["dH"]["delta"] + bx0),
                dB=c0["d0"], dB_F=F0, dB_seed1=max(c0["d0_seed1"], c0["d0"]["delta"] + bx0),
                dM=cm["d0"], dM_F=F0, dM_seed1=max(cm["d0"]["delta"] + cm["repX"] - cm["repB"], cm["d0"]["delta"] + bx0),
                gx=cg["d0"], gx_F=Fg, D=D, base="A0", F_I=FI0, F_T=FT0, readout_b_minus_a_acc=bx0)


def delta7(js_76, js_bx):
    """Delta_7 = RA7-L1 - RA6-L1 on 0-2 s acc_mae with Floor_k."""
    c = RD.cell(js_76, *CELLS[0])
    bx = RD.find_row(js_bx, "contrast", *CELLS[0], "A", "B")["delta"]
    F = max(c["F"], abs(bx))
    row = c["d0"]
    return dict(delta=row["delta"], lo=row["lo"], hi=row["hi"], F_I=c["F"], F_T=abs(bx), Floor=F, seed1=c["d0_seed1"], readout_b_point=row["delta"] + bx,
                better=sep_better7(row, F, c["d0_seed1"], row["delta"] + bx), worse=worse7(row, F))


def outcome7(effective, dB_better, d7_better, d7_worse, invalid=False):
    """effective: AMEND_5's combined effective label for RA7-L1 under Floor_k (CEILING / NARROWS / NULL / HARM). dB_better / d7_better / d7_worse: {chassis: bool}. -> the section 7 row."""
    if invalid:
        return "INVALID"
    if effective == "HARM":
        return "HARM"
    if effective == "CEILING":
        return "R7-CEILING"
    if any(d7_worse.values()):
        return "R7-WORSE"
    if d7_better and all(d7_better.values()) and all(dB_better.values()):
        return "R7-PRECISION"
    return "R7-SAME"


NEXT = {
    "R7-CEILING": "AMEND_5's CEILING row with RA7: RA7-L1 is the candidate level-1 longitudinal cost; the 2-6 s band needs a K = 30 readout and its own pull; the T1-kin read is blocked on a goal head (Q2a)",
    "R7-PRECISION": "the readout's function class was a lever: (1) Option B by amendment, reading through RA7's local gradient (an MM ruling before Option B's pilot); (2) the pull of the remaining 3,512 eligible clips (box 2 read only; MM / PI under P3); CEIL-ORACLE7 sets the order",
    "R7-SAME": "the nonlinear readout is ELIMINATED as a lever at N = 1,000 (logged); with R6-SAME, readout size and readout class are both measured not to move RA-L1. Next: ACTRSP Option B, handed section 5.2 and CEIL-ORACLE7",
    "R7-WORSE": "C-GRAD7, C-RA7-SHA, the K-control and C-SEL7 rerun first (wiring), then the R7-SAME row",
    "HARM": "as AMEND_5 section 6: first C-OFF5, C-YAW-ID, C-GRAD7, C-PLANT7 and C-RA7-SHA are re-run on the SCORE argv (wiring first), then the R7-SAME row",
    "INVALID": "a control is RED: nothing is read as a result; the control is fixed and the stage rerun",
}


def _contrast(Bd, a, b, out, ra, rb, n_boot):
    f = lambda d, s=0: f"{d}/plans_hwm_oracle_s{s}.npz"
    return RD.run_contrast(Bd, f(a), f(b), out, f(a, 1) if ra else None, f(b, 1) if rb else None, n_boot)


def score(n_boot=2000):
    RD.score(ROOT, n_boot)
    full = json.load(open(f"{ROOT}/readout/RESULT_AMEND5.json", encoding="utf-8"))
    for ext in ("json", "md"):
        shutil.move(f"{ROOT}/readout/RESULT_AMEND5.{ext}", f"{ROOT}/readout/RESULT_AMEND7_full.{ext}")
    ro = f"{ROOT}/readout"
    li, oracle_in, d7, extra_rows = {}, {}, {}, {}
    for k in ("s0", "s1"):
        Bd = ST.base_dir("score", "A0", k)
        d_x = {a: ST.out_dir("score", a, "A0", k, ROOT) for a in ("RAL1", "RAL1B", "ORACLE")}
        d6 = {a: f"{A6}/score/{a}_A0_{k}" for a in ("RAL1", "ORACLE")}
        d5 = f"{A5}/score/RAL1_A0_{k}"
        js76 = _contrast(Bd, d_x["RAL1"], d6["RAL1"], f"{ro}/c_RAL1_RA6_{k}.json", True, True, n_boot)
        jsbx = _contrast(Bd, d_x["RAL1B"], d_x["RAL1"], f"{ro}/c_RAL1B_RAL1_{k}.json", False, False, n_boot)
        js75 = _contrast(Bd, d_x["RAL1"], d5, f"{ro}/c_RAL1_RA5_{k}.json", True, True, n_boot)
        jsoo = _contrast(Bd, d_x["ORACLE"], d6["ORACLE"], f"{ro}/c_ORACLE_RA6ORACLE_{k}.json", True, True, n_boot)
        jsb = json.load(open(f"{ro}/c_RAL1_BASE_{k}.json", encoding="utf-8"))
        jsm = json.load(open(f"{ro}/c_RAL1_MEANL1_{k}.json", encoding="utf-8"))
        jso = json.load(open(f"{ro}/c_ORACLE_BASE_{k}.json", encoding="utf-8"))
        D = full["chassis"][k]["D"]["RAL1"]
        li[k] = lever_input7(jsb, jsm, js76, jsbx, D)
        cO = RD.cell(jso, *CELLS[0])
        oracle_in[k] = dict(row=cO["dH"], F=cO["F"], seed1=cO["dH_seed1"])
        d7[k] = delta7(js76, jsbx)
        extra_rows[k] = dict(RA7_vs_RA5_L1=RD.cell(js75, *CELLS[0])["d0"], RA7_ORACLE_vs_RA6_ORACLE=RD.cell(jsoo, *CELLS[0])["d0"])
    invalid = (not full["controls_ok"]) or ("INVALID" in full["labels"])
    labels = RD.finalize({"RAL1": li}, oracle_in, not invalid)
    arm = labels["arms"]["RAL1"]
    dB = {k: bool(arm["per_chassis"][k] in ("NARROWS", "CEILING")) for k in li}
    out = dict(kind="RESULT_AMEND7", **dict(STAMP, estimator=STAMP["estimator"].replace("B 2000", f"B {n_boot}")), spec_sha256=ST.SPEC_SHA, amend5_rules_label_floor_k=arm["effective_name"], per_chassis=arm["per_chassis"], qualifier=arm["qualifier"], CEIL_ORACLE7=labels["CEIL_ORACLE"],
               CEIL_ORACLE7_per_chassis=labels["CEIL_ORACLE_per_chassis"], CEIL_ORACLE7_note="floor = AMEND_5's oracle floor (the ORACLE arm has inference seeds 0 and 1; no readout-B ORACLE arm exists)",
               delta_7=d7, floors={k: dict(F_I=li[k]["F_I"], F_T=li[k]["F_T"], Floor=li[k]["dH_F"], Floor_xtrack=li[k]["gx_F"]) for k in li}, cells={k: {n: li[k][n] for n in ("dH", "dB", "dM", "gx")} for k in li}, D={k: li[k]["D"] for k in li},
               beside=extra_rows, controls_ok=(not invalid), outcome=outcome7(arm["effective"], dB, {k: v["better"] for k, v in d7.items()}, {k: v["worse"] for k, v in d7.items()}, invalid), full="RESULT_AMEND7_full.json")
    out["next_lever"] = NEXT[out["outcome"]]
    json.dump(out, open(f"{ro}/RESULT_AMEND7.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    f_ = lambda v: f"{v:+.4f}"
    L = ["# RESULT_AMEND7 (generated; the MM writes the reading)", "",
         f"Tier T0, evidence MEASURED. Estimator: {out['estimator']}.", "",
         f"**Outcome (section 7): {out['outcome']}.** AMEND_5 label rules on RA7-L1 under Floor_k: {arm['effective_name']} (per chassis {arm['per_chassis']}, qualifier {arm['qualifier']}). CEIL-ORACLE7: {out['CEIL_ORACLE7']} (per chassis {out['CEIL_ORACLE7_per_chassis']}).", "",
         f"Next lever: {out['next_lever']}", "", "Delta_7 = RA7-L1 minus RA6-L1, 0-2 s acc_mae, T0 (Floor_k = max(F_I, F_T)):", ""]
    L += [f"* {k}: {f_(v['delta'])} [{f_(v['lo'])}, {f_(v['hi'])}]; F_I {v['F_I']:.4f}, F_T {v['F_T']:.4f}; seed-1 {f_(v['seed1'])}; readout-B point {f_(v['readout_b_point'])}; separated-better {v['better']}; worse {v['worse']}" for k, v in d7.items()]
    L += ["", "Deciding cells (0-2 s acc_mae unless named):", ""]
    for k in li:
        c = li[k]
        L.append(f"* {k}: RA7-L1 - HOLD {f_(c['dH']['delta'])} [{f_(c['dH']['lo'])}, {f_(c['dH']['hi'])}]; - A0 {f_(c['dB']['delta'])} [{f_(c['dB']['lo'])}, {f_(c['dB']['hi'])}]; - MEAN-L1 {f_(c['dM']['delta'])}; xtrack guard {f_(c['gx']['delta'])}; Floor {c['dH_F']:.4f}; D {c['D']:.3f}")
    L += ["", "Beside (no bar): " + json.dumps({k: {n: round(v["delta"], 4) for n, v in e.items()} for k, e in extra_rows.items()}), "",
          "All four families, every band, the mechanism rows and DIR-ACC: `RESULT_AMEND7_full.md` (AMEND_5's generator on the amend7 folders). STRATEGIC is reported in full there, with n per cell."]
    open(f"{ro}/RESULT_AMEND7.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("OUTCOME", out["outcome"])
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score(int(os.environ.get("AMEND7_NBOOT", "2000")))                  # the card's B is 2,000; the build's smoke lowers it through the environment
    elif len(sys.argv) > 3 and sys.argv[1] == "train":
        r = RD.train_stage(ROOT, sys.argv[3])
        print("C-SHUF_train_sign", r["C-SHUF_train_sign"], "C-PLANT_ii", r["C-PLANT_ii"])
        sys.exit(0 if r["checks_ok"] else 2)
