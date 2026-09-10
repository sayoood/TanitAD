"""Does `tac_goal_tok_head` finally appear in the effective-weights stamp?

The 2026-09-07 census established WHY the stamp was blind: it enumerates
DECLARED LOSS WEIGHTS, and this head had none, so it produced **no row at all**.
The row is therefore not packaging -- it is the half of the fix that makes the
head visible to the instrument that is supposed to see it.

Also exercises the REFUSAL: a weight whose term can never be reached must be
rejected AT LAUNCH, in milliseconds, not after the corpus mounts.

Run:  python probe_stamp.py <tree>/stack <label>
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

STACK = os.path.abspath(sys.argv[1])
LABEL = sys.argv[2]
SCRIPTS = os.path.join(STACK, "scripts")
for p in (STACK, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_gradient import LIVE_ARGV  # noqa: E402


def trainer():
    spec = importlib.util.spec_from_file_location(
        "refc_v3_train_stamp", os.path.join(SCRIPTS, "refc_v3_train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    T = trainer()
    out = {"label": LABEL, "cases": {}}

    def stamp_for(extra, argv=None):
        base = list(argv if argv is not None else LIVE_ARGV)
        args = T.build_parser().parse_args(base + list(extra))
        args._ew_parser = T.build_parser()
        args._ew_argv = base + list(extra)
        return T.effective_weights_stamp_v3(args, echo=False)

    # 1. the ON arm: the row must exist and be non-zero
    try:
        st = stamp_for(["--w-tac-goal", "1.0"])
        rows = st.get("rows", st)
        out["cases"]["ON"] = {"stamp": st}
    except SystemExit as e:
        out["cases"]["ON"] = {"REFUSED": str(e)}
    except Exception as e:                                   # noqa: BLE001
        out["cases"]["ON"] = {"ERROR": f"{type(e).__name__}: {e}"}

    # 2. the default arm: the row must exist and read 0.0
    try:
        out["cases"]["OFF"] = {"stamp": stamp_for([])}
    except SystemExit as e:
        out["cases"]["OFF"] = {"REFUSED": str(e)}
    except Exception as e:                                   # noqa: BLE001
        out["cases"]["OFF"] = {"ERROR": f"{type(e).__name__}: {e}"}

    # 3. THE REFUSAL: a weight with no head to spend it on.
    no_head = [a for a in LIVE_ARGV if a != "--tac-goal-tok-head"]
    try:
        args = T.build_parser().parse_args(no_head + ["--w-tac-goal", "1.0"])
        args._ew_parser = T.build_parser()
        args._ew_argv = no_head + ["--w-tac-goal", "1.0"]
        T.check_effective_weights(args)
        out["cases"]["W_WITHOUT_HEAD"] = {"ACCEPTED": "NO REFUSAL -- defect"}
    except SystemExit as e:
        out["cases"]["W_WITHOUT_HEAD"] = {"REFUSED": str(e)}
    except Exception as e:                                   # noqa: BLE001
        out["cases"]["W_WITHOUT_HEAD"] = {"ERROR": f"{type(e).__name__}: {e}"}

    # 4. the ON arm must NOT be refused.
    try:
        args = T.build_parser().parse_args(LIVE_ARGV + ["--w-tac-goal", "1.0"])
        args._ew_parser = T.build_parser()
        args._ew_argv = LIVE_ARGV + ["--w-tac-goal", "1.0"]
        T.check_effective_weights(args)
        out["cases"]["ON_ACCEPTED"] = {"ACCEPTED": True}
    except SystemExit as e:
        out["cases"]["ON_ACCEPTED"] = {"REFUSED": str(e)}

    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
