"""read_labelunsafe10k_b.py -- card PREREG_V8F_PROOF_ADD136 (LABELUNSAFE10KB): the read of card ADD135 with a share in its planted control u2.

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions; not driving performance). CPU only, banked runs.
The tool of card ADD135 (read_labelunsafe10k.py) is imported unchanged; this file sets the card, the prefix of the names and control u2, and refuses to run
unless the controls file of card ADD135 shows control u2 failed with u0, u1, u3 and u4 passed.

  controls|names  --add135-controls <file>  + the options of the tool of card ADD135
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD136"
CARD_A = "PREREG_V8F_PROOF_ADD135"
PRE = "LABELUNSAFE10KB"
RL_PREFIX = "46be1fad5df31db3"
M4_SHARE = 0.90
SLOW_MS = 1.0
DROP_MS = 0.2


def sha_lf(p):
    """sha256 of a text file with CR removed (the same on the dev box and on the pod)"""
    with io.open(p, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r", b"")).hexdigest()


def load_rl():
    m = sys.modules.get("read_labelunsafe10k")
    p = getattr(m, "__file__", None) if m is not None else None
    if not (p and os.path.isfile(p)):
        here = os.path.dirname(os.path.abspath(__file__))
        for p in (os.environ.get("LABELUNSAFE_TOOL"), os.path.join(os.path.dirname(here), "kit", "read_labelunsafe10k.py")):
            if p and os.path.isfile(p):
                break
        else:
            raise SystemExit("REFUSED: read_labelunsafe10k.py not found (set LABELUNSAFE_TOOL)")
        d = os.path.dirname(os.path.abspath(p))
        if d not in sys.path:
            sys.path.insert(0, d)
        spec = importlib.util.spec_from_file_location("read_labelunsafe10k", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["read_labelunsafe10k"] = m
        spec.loader.exec_module(m)
    if not sha_lf(p).startswith(RL_PREFIX):
        raise SystemExit("REFUSED: %s is not the tool of card ADD135 (%s)" % (p, RL_PREFIX))
    return m


RL = load_rl()
U3_A = RL.ctl_u3


def exempt_windows(A):
    """windows whose logged step speed is below SLOW_MS at one scored step or more, or falls by more than DROP_MS from one scored step to the next"""
    vl = np.asarray(A["vl"], np.float64)
    sc = np.isfinite(A["tl"])
    slow = (sc & (vl < SLOW_MS)).any(1)
    d = vl[:, 1:] - vl[:, :-1]
    drop = (sc[:, 1:] & sc[:, :-1] & (d < -DROP_MS)).any(1)
    return slow | drop


def ctl_u2(E, w, plant_s=None, ident_cl=None):
    """card ADD136, control u2: the logged track as the model track gives no unsafe window; the planted track makes one window or more unsafe, 0.90 or more of them count for M4,
    each of the others is a slow or a braking window of the logged driver, and a window unsafe by the gap alone has a model gap below 2.0 m by the second routine"""
    plant_s = RL.PLANT_S if plant_s is None else plant_s
    if not w.complete:
        return False, "the runs could not be read"
    H = E.H
    g, L = w.ref["g"], w.L
    A0 = RL.step_arrays(E, g, g if ident_cl is None else ident_cl, L)
    S0 = RL.summarize(A0)
    if S0["sel"].any() or bool(np.nansum(A0["U"][A0["ud"]]) != 0.0):
        return False, "the logged track read as the model track has an unsafe window"
    a_l, v_l = H.logged_arc(g), H.step_speeds(g)
    cl_p = H.point_at_arc(g, a_l + plant_s * v_l)
    Ap = RL.step_arrays(E, g, cl_p, L)
    Sp = RL.summarize(Ap)
    sel = Sp["sel"]
    if not sel.any():
        return False, "the planted track makes no window unsafe"
    with np.errstate(invalid="ignore"):
        m4 = Sp["min_tm"] < Sp["min_tl"] - RL.M4_DELTA
    if float(m4[sel].mean()) < M4_SHARE:
        return False, "fewer than 0.90 of the windows made unsafe by the planted track count for M4"
    Ap["vl"] = v_l
    rest = sel & ~m4
    if rest.any() and not bool(exempt_windows(Ap)[rest].all()):
        return False, "a window made unsafe by the planted track does not count for M4 and is neither slow nor braking"
    S2 = RL.second_routine(E, g, cl_p, L)
    alone = sel & Sp["has_gap"] & ~Sp["has_tau"]
    if alone.any() and not bool((S2["min_gm"][alone] < float(H.UNSAFE_GAP_M)).all()):
        return False, "a window unsafe by the gap alone has no model gap below 2.0 m"
    if not bool(np.array_equal(Sp["has_gap"][sel], S2["min_gm"][sel] < float(H.UNSAFE_GAP_M))):
        return False, "the step with a gap below 2.0 m differs between the two routines in a planted window"
    return True, ""


def ctl_u3(paths):
    """card ADD136, control u3: four test outputs, the two of the tool of card ADD135 (dev box, pod) and the two of this tool"""
    if len(paths) != 4:
        return False, "four test outputs are needed (two tools, dev box and pod)"
    for pair in (paths[:2], paths[2:]):
        ok, why = U3_A(pair)
        if not ok:
            return False, why
    for p in paths[2:]:
        with io.open(p, encoding="utf-8", errors="replace") as fh:
            if "test_read_labelunsafe10k_b" not in fh.read():
                return False, "a test output of this card's tool does not name its test file"
    return True, ""


def order_ok(path):
    """this card runs only after a run of record of card ADD135 whose control u2 failed with the other four passed"""
    try:
        with io.open(path, encoding="utf-8") as fh:
            r = json.load(fh)
    except (OSError, ValueError, TypeError):
        return False
    return r.get("card") == CARD_A and r.get("pass_u2") is False and all(r.get("pass_" + k) is True for k in ("u0", "u1", "u3", "u4"))


def bind():
    """sets the card, the names and control u2 in the imported tool"""
    RL.CARD = CARD
    RL.PRE = PRE
    RL.MECH_THRESH = PRE + "-K0-THRESHOLD-NEAR-THE-LOGGED-DRIVER"
    RL.MECH_CLOSER = PRE + "-K0-RULE-FOLLOWS-CLOSER-THAN-LOGGED"
    RL.MECH_MIXED = PRE + "-K0-MIXED"
    RL.REP_SAME = PRE + "-K0-SAME-WINDOWS-ACROSS-SEEDS"
    RL.REP_DIFF = PRE + "-K0-WINDOWS-DIFFER-BY-SEED"
    RL.ctl_u2 = ctl_u2
    RL.ctl_u3 = ctl_u3


def main(argv=None, cfg=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    path = ""
    if "--add135-controls" in argv:
        i = argv.index("--add135-controls")
        if i + 1 >= len(argv):
            print("REFUSED: --add135-controls needs a file")
            return 2
        path = argv[i + 1]
        del argv[i:i + 2]
    if not order_ok(path):
        print("REFUSED: no controls file of card ADD135 with control u2 failed and the other four passed: this card is not run")
        return 3
    bind()
    return RL.main(argv) if cfg is None else RL.main(argv, cfg)


if __name__ == "__main__":
    sys.exit(main())
