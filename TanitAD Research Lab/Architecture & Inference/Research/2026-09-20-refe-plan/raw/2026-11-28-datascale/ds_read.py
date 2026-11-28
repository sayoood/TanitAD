#!/usr/bin/env python3
"""DATA-SCALE navtest TABLE READ (PREREG_DATASCALE.md 4.4, 5, 6): `l2next_read.py` extended with `--lever {ds_sh,ds_ep}`.  CPU only.

`l2next_read.py` is FROZEN (its card's CODE.sha256) and is NOT edited.  This file IMPORTS it, registers the two DATA-SCALE levers in its `LEVERS` table, and replaces exactly two of its
functions for them (module-level rebinding in this process only):
  * `_run_head`  = the original, except the head-select check wants `cv5m` (the original wants `cv5` for a CV lever and would flag a cv5m head)
  * `_ctrl_cv`   = CTRL-CV of PREREG_DATASCALE 6: (a) built.json folds == sorted-name index mod 5 over ALL N navtrain logs; (b) cv_select.json's token / log / fold counts equal the sealed train
                   index's (no hard-coded 2,272 / 35: N is whatever the sealed index holds; --n-train-tokens / --n-train-logs, if given, must equal it too); (c) the final fit used every token;
                   (d) the chosen step is the same in run.json, cv_select.json and the decision; **(e) the chosen step equals `cv5m` RECOMPUTED from cv_select.json's per-log OOF table**
                   (safehead_cvm.recheck: the margin and guard rule, the CIs bit-equal), the recorded rule is the REGISTERED rule, built.json carries the same rule, and the recorded no-margin
                   argmax equals the argmax of the per-step OOF PDMS.
Everything else (the bars P1, P2, G, R, the controls, the four families, the outcome precedence) is `l2next_read.py`'s, unchanged.  Usage is `l2next_read.py`'s with `--lever ds_sh|ds_ep`:
    python ds_read.py read --lever ds_ep --table ... --run-s0 ... --run-s1 ... --run-red ... [--refe <PK/refe>] ...        python ds_read.py confirm --read READ_DSEP.json ...
UNVERIFIED (no real data, no synthetic end-to-end read fixture was built for these two levers): the full `read` / `confirm` path under ds_sh / ds_ep.  What IS tested (test_ds_read.py) is the
lever table, the `_ctrl_cv` replacement incl. its mutations, and the `_run_head` select check.  The report's `card` string still names the L2-NEXT folder (a label in l2next_read.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-11-27-l2next"))
import l2next_read as L2  # noqa: E402

DS_NEXT = {"NULL-CV": "the learning curve (M6) picks: best-step OOF gain rising monotonically in n and higher at N than at 35 -> 7.1 DATA-SCALE-2 (a PI decision on download and storage); "
                      "otherwise 7.5 Level 2 (a PI decision on storage and GPU)",
           "ADVERSE": "7.5 Level 2; if the navtrain-eligible gain was separated and still did not transfer, the fit data, not its size, is the defect (navtrain-to-navtest shift)",
           "ADOPT": "7.2 the DSH+DEP composition if the other lever also adopts, else 7.3 (own card)",
           "SEPARATED-SMALL": "the same lever with 3 seeds",
           "NULL": "as NULL-CV: the learning curve picks 7.1 or 7.5"}
DS_LEVERS = {
    "ds_sh": {"tag": "DSSH", "rows": L2.ROWS, "prefix": "refe_ds_sh", "arms": ("DSH0", "DSH1"), "select_file": "cv_select.json", "cv": True, "mlp": False,
              "schema": "safehead.head/1", "null": "NULL-CV", "card": "PREREG_DATASCALE.md", "next": dict(DS_NEXT)},
    "ds_ep": {"tag": "DSEP", "rows": (2,), "prefix": "refe_ds_ep", "arms": ("DEP0", "DEP1"), "select_file": "cv_select.json", "cv": True, "mlp": False,
              "schema": "safehead.head/1", "null": "NULL-CV", "card": "PREREG_DATASCALE.md", "next": dict(DS_NEXT)},
}
_ORIG_RUN_HEAD = L2._run_head
_ORIG_CTRL_CV = L2._ctrl_cv


def _run_head_ds(D, run: Path, arm: str, lever: str = "shmlp") -> dict:
    r = _ORIG_RUN_HEAD(D, run, arm, lever)
    if lever in DS_LEVERS:
        r["problems"] = [p for p in r["problems"] if "head select" not in p]
        if r["head"]["meta"].get("select") != "cv5m":
            r["problems"].append(f"{arm}: head select {r['head']['meta'].get('select')!r}, lever {lever} needs 'cv5m'")
    return r


def _ctrl_cv_ds(D, run: Path, tc: list, chosen_step: int, a) -> dict:
    import safehead_cvm as CVM
    built = json.loads((run / "built.json").read_text(encoding="utf-8"))
    cvs = json.loads((run / "cv_select.json").read_text(encoding="utf-8"))
    rj = json.loads((run / "run.json").read_text(encoding="utf-8"))
    want = D.cv_folds([r["log"] for r in tc], L2.CV_FOLDS)
    got = built.get("cv_fold_of_log") or {}
    a_ok = sorted(got) == sorted(want) and all(list(got[lg]) == [want[lg]] for lg in want)
    want_folds = {str(f): sorted(lg for lg in want if want[lg] == f) for f in range(L2.CV_FOLDS)}
    n_tok, n_log = len(tc), len(want)
    given = [x for x in (getattr(a, "n_train_tokens", None), getattr(a, "n_train_logs", None)) if x is not None]
    b_ok = (int(cvs.get("n_tokens", -1)) == n_tok and int(cvs.get("n_logs", -1)) == n_log and len(cvs.get("folds", {})) == L2.CV_FOLDS
            and {k: sorted(v) for k, v in cvs.get("folds", {}).items()} == want_folds and sorted(cvs.get("per_log", {})) == sorted(want)
            and all(int(cvs["per_log"][lg]["n"]) == sum(1 for r in tc if r["log"] == lg) for lg in want)
            and (not given or (getattr(a, "n_train_tokens", None) in (None, n_tok) and getattr(a, "n_train_logs", None) in (None, n_log))))
    c_ok = int(built.get("n_train", -1)) == n_tok
    d_ok = int(rj.get("chosen_step", -1)) == int(cvs.get("chosen_step", -2)) == chosen_step
    rc = CVM.recheck(cvs)
    ps = cvs["per_step"]
    best = max(v["oof_pdms"] for v in ps.values())
    argmax = min(int(s) for s, v in ps.items() if v["oof_pdms"] == best)
    reg = json.loads(json.dumps(CVM.REGISTERED_RULE))
    e_ok = bool(rc["match"] and rc["ci_bit_equal"] and json.loads(json.dumps(cvs.get("margin_rule"))) == reg and json.loads(json.dumps(built.get("cv_rule"))) == reg
                and int(cvs.get("no_margin_argmax_step", -1)) == argmax and cvs.get("select") == "cv5m")
    return {"pass": bool(a_ok and b_ok and c_ok and d_ok and e_ok), "a_folds_ok": bool(a_ok), "b_counts_ok": bool(b_ok), "c_all_tokens_fit": bool(c_ok), "d_chosen_consistent": bool(d_ok),
            "e_chosen_equals_cv5m_recomputed": bool(e_ok), "recheck": rc, "chosen_step": chosen_step, "no_margin_argmax_step": argmax, "n_tokens": n_tok, "n_logs": n_log}


def install() -> None:
    """Register the DATA-SCALE levers and rebind the two functions in l2next_read (idempotent)."""
    L2.LEVERS.update(DS_LEVERS)
    L2.STEP_SEL.update({k: "cv_select.json" for k in DS_LEVERS})
    L2._run_head = _run_head_ds
    L2._ctrl_cv = _ctrl_cv_ds
    L2.N_TRAIN_TOKENS = L2.N_TRAIN_LOGS = None                    # the parser defaults 2272 / 35 are the PARENT's; N comes from the sealed index (CTRL-CV b)


def main(argv=None) -> int:
    install()
    return L2.main(argv)


if __name__ == "__main__":
    sys.exit(main())
