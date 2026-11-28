"""AMEND_HWMPLAN_6 readout: AMEND_5's score() on the amend6 folders (RA6 arms, A0 base, MEAN-L1, HOLD, ...), plus the new contrast Delta_R = RA6-L1 - RA-L1 (AMEND_5) and the section 6 outcome.
  python amend6_read.py train --chassis k   AMEND_5's TRAIN-stage report (beside, in-sample; C-SHUF sign and C-PLANT ii) on the amend6 TRAIN arms
  python amend6_read.py score               -> amend6/readout/RESULT_AMEND6.json + .md  (AMEND_5's full output stays beside as RESULT_AMEND6_full.*)"""
import amend6_env  # noqa: F401
import json, os, shutil, sys
from amend6_env import ROOT, ST, A5
import amend5_read as RD

A5SCORE = A5 + "/score"


def outcome6(effective, dB_better, dR_better, dR_worse):
    """effective: AMEND_5's combined effective label for RA6-L1 (CEILING / NARROWS / NULL / HARM). dB_better / dR_better / dR_worse: {chassis: bool}. -> the section 6 row."""
    if effective == "HARM":
        return "HARM"
    if effective == "CEILING":
        return "R6-CEILING"
    if any(dR_worse.values()):
        return "R6-WORSE"
    if dR_better and all(dR_better.values()) and all(dB_better.values()):
        return "R6-PRECISION"
    return "R6-SAME"


def delta_r(k, n_boot=2000):
    f = lambda d, s=0: f"{d}/plans_hwm_oracle_s{s}.npz"
    Bd = ST.base_dir("score", "A0", k)
    d6, d5 = ST.out_dir("score", "RAL1", "A0", k, ROOT), f"{A5SCORE}/RAL1_A0_{k}"
    for p in (f(d5), f(d5, 1)):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    js = RD.run_contrast(Bd, f(d6), f(d5), f"{ROOT}/readout/c_RA6_RA5_{k}.json", f(d6, 1), f(d5, 1), n_boot)
    c = RD.cell(js, "LONGITUDINAL", "0-2s", "acc_mae")
    row = c["d0"]
    return dict(delta=row["delta"], lo=row["lo"], hi=row["hi"], F=c["F"], seed1=c["d0_seed1"], better=RD.sep_better(dict(row), c["F"], c["d0_seed1"]), worse=RD.worse(dict(row), c["F"]))


def score():
    RD.score(ROOT)
    res = json.load(open(f"{ROOT}/readout/RESULT_AMEND5.json", encoding="utf-8"))
    for ext in ("json", "md"):
        shutil.move(f"{ROOT}/readout/RESULT_AMEND5.{ext}", f"{ROOT}/readout/RESULT_AMEND6_full.{ext}")
    arm = res["labels"]["arms"]["RAL1"]
    dr = {k: delta_r(k) for k in ("s0", "s1")}
    dB = {k: bool(arm["per_chassis"][k] in ("NARROWS", "CEILING")) for k in ("s0", "s1")}
    out = dict(kind="RESULT_AMEND6", tier="T0", evidence="MEASURED", estimator="paired_episode_cluster_bootstrap B 2000 seed 0", amend5_rules_label=arm["effective_name"], per_chassis=arm["per_chassis"],
               qualifier=arm["qualifier"], CEIL_ORACLE=res["labels"]["CEIL_ORACLE"], delta_R=dr, controls_ok=res["controls_ok"],
               outcome=outcome6(arm["effective"], dB, {k: v["better"] for k, v in dr.items()}, {k: v["worse"] for k, v in dr.items()}), full="RESULT_AMEND6_full.json")
    if not res["controls_ok"]:
        out["outcome"] = "INVALID"
    json.dump(out, open(f"{ROOT}/readout/RESULT_AMEND6.json", "w", encoding="utf-8", newline="\n"), indent=1)
    L = ["# RESULT_AMEND6 (generated; the MM writes the reading)", "",
         f"Outcome (section 6): **{out['outcome']}**. AMEND_5 label rules on RA6-L1: {arm['effective_name']} (per chassis {arm['per_chassis']}, qualifier {arm['qualifier']}). CEIL-ORACLE6: {out['CEIL_ORACLE']}.", "",
         "Delta_R = RA6-L1 minus RA-L1 (AMEND_5), 0-2 s acc_mae, T0:", ""]
    L += [f"* {k}: {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]; floor {v['F']:.4f}; seed-1 {v['seed1']:+.4f}; separated-better {v['better']}; worse {v['worse']}" for k, v in dr.items()]
    L += ["", "Every other row: `RESULT_AMEND6_full.md` (AMEND_5's generator on the amend6 folders)."]
    open(f"{ROOT}/readout/RESULT_AMEND6.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("OUTCOME", out["outcome"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score()
    elif len(sys.argv) > 3 and sys.argv[1] == "train":
        RD.train_stage(ROOT, sys.argv[3])
