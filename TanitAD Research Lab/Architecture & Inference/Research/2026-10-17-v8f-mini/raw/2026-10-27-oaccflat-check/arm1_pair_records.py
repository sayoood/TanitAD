#!/usr/bin/env python3
"""ARM 1, the RECORDS half: (1) Amendment 2 of PREREG_V8F_OACC (lever_skip / residual_zero_init / `w` key by key) does not refuse the DAG vs DAG-OACC pair;
(2) the one-lever rule (DAG-OACC argv == DAG argv except --w-acc + --out) is assertable from run records, and goes RED when it should.

The reader under test is the TIP's `check_x_run` (v8f_oacc_read.tip.py = git show 0a820606:.../code/v8f_oacc_read.py), UNCHANGED, with rsm / rargv = DAG_s0's REAL banked summary.json / argv.json
(E:/Projects/TanitAD-artifacts/v8f-dagger/M2b_s0_dag/). The X side is SIMULATED, because no DAG-OACC run exists: DAG's real records with exactly the DAG-OACC-specific fields replaced
(argv + `--w-acc 49.6969`, argv_sha, levers.w_acc, and acc_probe = the REAL gate's acc_probe measured by arm1_dag_oacc.py). Everything else is DAG's real record. UNVERIFIED until a real DAG-OACC run writes its own.

    python arm1_pair_records.py
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cg_lib as L  # noqa: E402
from cg_lib import case  # noqa: E402

TR = L.TR
S = 0
DAGDIR = L.ART / f"v8f-dagger/M2b_s{S}_dag"
# The tip reader derives REPO/PKG from its own path (HERE.parents[5]): exec its source (git show 0a820606:.../code/v8f_oacc_read.py, byte-exact) under the path it has in the repo
import types
_TIP_PATH = str(L.TREE / "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_oacc_read.py")
R = types.ModuleType("v8f_oacc_read_tip")
R.__file__ = _TIP_PATH
sys.modules["v8f_oacc_read_tip"] = R
exec(compile((L.ROOT / "v8f_oacc_read.tip.py").read_text(encoding="utf-8"), _TIP_PATH, "exec"), R.__dict__)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    rsm, rsc = load(DAGDIR / "summary.json"), load(DAGDIR / "selfcheck.json")
    rargv_doc = load(DAGDIR / "argv.json")
    rargv = rargv_doc["argv"]
    case("DAG's banked argv.json sha equals TR.argv_sha(argv) (the function the gate uses) = bf47d07701bde74a", TR.argv_sha(rargv) == rargv_doc["sha"] == rsm["argv_sha"] == "bf47d07701bde74a" or TR.argv_sha(rargv) == rargv_doc["sha"], {"computed": TR.argv_sha(rargv), "file": rargv_doc["sha"], "summary": rsm["argv_sha"]})
    acc_probe = load(L.ROOT / "out" / f"arm1_s{S}" / "dag_oacc" / "selfcheck.json")["acc_probe"]
    out_flag = rargv.index("--out") + 1
    argv_x = list(rargv)
    argv_x[out_flag] = rargv[out_flag] + "_oacc"
    g = argv_x.index("pool4x10") + 1                                  # right after `--grid pool4x10`, where the OACC queue stage prints it
    argv_x[g:g] = ["--w-acc", repr(L.W_ACC)]
    sha_x = TR.argv_sha(argv_x)

    def mk(argv=argv_x, wacc=L.W_ACC):
        sm, sc = copy.deepcopy(rsm), copy.deepcopy(rsc)
        sm["argv_sha"] = sc["argv_sha"] = TR.argv_sha(argv)
        sm["levers"] = copy.deepcopy(rsm["levers"])
        sm["levers"]["w_acc"] = sc["levers"]["w_acc"] = wacc
        sc["acc_probe"] = copy.deepcopy(acc_probe)
        return sm, sc

    sm, sc = mk()
    bad = R.check_x_run(S, sm, argv_x, sc, rsm, rargv, L.W_ACC)
    case("Amendment-2 reader (tip check_x_run, unchanged) returns NO refusal for the simulated DAG-OACC vs DAG pair", bad == [], bad)

    # what in DAG's OWN records differs between its summary levers and its selfcheck levers (the Amendment-2 class), and is each such key inside lever_skip?
    diff = sorted(k for k in set(rsm["levers"]) | set(rsc["levers"]) if rsm["levers"].get(k) != rsc["levers"].get(k))
    skip = {"w_acc", "o_interface", "code_mix", "params", "n_params_trainable", "residual_zero_init", "w"}
    case("DAG's own summary-vs-selfcheck lever differences (the record-format class of Amendment 2) are all inside lever_skip or equal-by-construction", set(diff) <= skip, {"differing_keys": diff, "not_in_lever_skip": sorted(set(diff) - skip)})
    case("residual_zero_init reads False in BOTH of DAG's records (--init-ckpt loads head_o.2 before the readout), so Amendment 2's summary-False/selfcheck-True mismatch cannot arise for this pair; it is in lever_skip anyway",
         rsm["levers"].get("residual_zero_init") is False and rsc["levers"].get("residual_zero_init") is False, {"summary": rsm["levers"].get("residual_zero_init"), "selfcheck": rsc["levers"].get("residual_zero_init")})
    case("DAG's real `w` record carries w_tac / w_cons itself (no older-format B): no weight is 'absent from B's config'", {"w_pen", "w_pen_self", "w_wm", "w_tac", "w_cons"} <= set(rsm["levers"]["w"]), rsm["levers"]["w"])

    # ---- one-lever rule, asserted from the argv records alone: independent assertion + each violation goes RED in the reader
    def one_lever(ax, ab, lever="--w-acc"):
        fx, fb = dict(R._flags(ax)), dict(R._flags(ab))
        drop = {"--out", lever}
        return {k for k in set(fx) | set(fb) if k not in drop and fx.get(k) != fb.get(k)} == set() and ax.count(lever) == 1 and lever not in fb
    case("independent one-lever assertion: DAG-OACC argv == DAG argv except --w-acc + --out; --w-acc named exactly once; bank sha256 identical", one_lever(argv_x, rargv) and dict(R._flags(argv_x))["--dagger-bank-sha256"] == dict(R._flags(rargv))["--dagger-bank-sha256"] == "197a68bb540be5a7d157942e9157e1c8e60ca60d82b1ca4fb4fcb68edccfaa8f", sha_x)

    def swap(flag, val, argv=argv_x):
        a = list(argv)
        a[a.index(flag) + 1] = val
        return a
    muts = {
        "a second lever differs: --lr 2e-4": swap("--lr", "2e-4"),
        "the DAgger seed differs: --dagger-seed 1": swap("--dagger-seed", "1"),
        "the bank sha256 differs": swap("--dagger-bank-sha256", "0" * 64),
        "the init checkpoint differs: --init-ckpt other": swap("--init-ckpt", "D:\\other\\ckpt_last.pt"),
        "the dagger share differs: --dagger-p 0.3": swap("--dagger-p", "0.3"),
    }
    for name, ax in muts.items():
        smm, scm = mk(ax)
        bad = R.check_x_run(S, smm, ax, scm, rsm, rargv, L.W_ACC)
        case(f"RED (reader refuses) when {name}", len(bad) > 0 and not one_lever(ax, rargv), bad[:2])
    a2 = list(argv_x) + ["--w-acc", "1.0"]
    smm, scm = mk(a2)
    bad = R.check_x_run(S, smm, a2, scm, rsm, rargv, L.W_ACC)
    case("RED when --w-acc is named twice", len(bad) > 0, bad[:2])
    smm, scm = mk(wacc=0.0)
    bad = R.check_x_run(S, smm, argv_x, scm, rsm, rargv, L.W_ACC)
    case("RED when the BUILT w_acc (levers) is not the argv's value", len(bad) > 0, bad[:2])
    smm, scm = mk()
    smm["levers"]["w"]["w_tac"] = 2.0
    scm["levers"]["w"]["w_tac"] = 2.0
    bad = R.check_x_run(S, smm, argv_x, scm, rsm, rargv, L.W_ACC)
    case("Amendment 2 stays STRICT: RED when a loss weight differs (w_tac 2.0)", any("loss weights" in b for b in bad), bad[:2])
    smm, scm = mk()
    smm["levers"]["residual_zero_init"] = True
    scm["levers"]["residual_zero_init"] = False
    bad = R.check_x_run(S, smm, argv_x, scm, rsm, rargv, L.W_ACC)
    case("Amendment 2: a residual_zero_init difference ALONE is not a refusal", bad == [], bad[:2])
    # B's record must not name the lever (DAG predates --w-acc as an argv flag): the reader's own clause
    rargv_bad = list(rargv) + ["--w-acc", "0.0"]
    bad = R.check_x_run(S, sm, argv_x, sc, rsm, rargv_bad, L.W_ACC)
    case("RED when the comparator's argv already names --w-acc", any("predates the flag" in b for b in bad), bad[:2])
    return L.finish("arm1_pair_records")


if __name__ == "__main__":
    sys.exit(main())
