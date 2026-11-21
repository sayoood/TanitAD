r"""STOPBAL second-dose checker: a thin layer over the first dose's `theory/stopbal/box2/stopbal_check.py` (not edited). It loads that file as `stopbal_check_base`, replaces its tables
(TABLE, WEIGHTS, W_GOAL / W_LON, RUN_ORDER, the pilot tag, the pilot gate list, the frozen argv file and its pinned sha256, the pilot out dir) and re-exports everything, so the stage scripts, the job
templates and the tests call it under the old name. Every function of the base reads those names from its own module globals at call time, so the patch reaches all of them.
Where the base is found: next to this file as stopbal_check_base.py (box 2: the ship stage puts it in stopbal\check), else ../box2/stopbal_check.py (dev box).
Card of the arms: theory/stopbal/AMENDMENT_3_DOSE2.md (frozen, sha256 460fd548...): BAL-G2 `--goal-pos-weight-stop 20.81`, BAL-L2 `--lon-class-weight-stop 22.08`, seeds 0 and 1, code of record 715ef8fc."""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(HERE, "stopbal_check_base.py"), os.path.join(os.path.dirname(HERE), "box2", "stopbal_check.py")):
    if os.path.isfile(_c):
        BASE_FILE = _c
        break
else:
    raise SystemExit("REFUSED: the first dose's stopbal_check.py (stopbal_check_base.py beside this file, or ../box2/stopbal_check.py) is missing")
_spec = importlib.util.spec_from_file_location("stopbal_check_base", BASE_FILE)
base = importlib.util.module_from_spec(_spec)
sys.modules["stopbal_check_base"] = base
_spec.loader.exec_module(base)

DOSE2_ARGVS_SHA256 = "f00267cbdaffe16ddf8743553f28f4198cba8a5e67f7842dc5ea12a0bd0127f4"        # frozen/argvs_dose2.json (make_argvs_dose2.py, MEASURED 2026-11-11)
CODE_OF_RECORD = "715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf"
W_GOAL2, W_LON2 = 20.81, 22.08
_cands = (os.path.join(HERE, "argvs_dose2.json"), os.path.join(HERE, "frozen", "argvs_dose2.json"))
base.ARGVS_FILE = next((c for c in _cands if os.path.exists(c)), _cands[1])
base.ARGVS_SHA256 = DOSE2_ARGVS_SHA256
base.W_GOAL, base.W_LON = W_GOAL2, W_LON2
base.TABLE = {
    "BAL-G2_s0": {"kind": "g", "seed": 0, "init": 0, "order": 1, "blocked": None},
    "BAL-L2_s0": {"kind": "l", "seed": 0, "init": 0, "order": 2, "blocked": None},
    "BAL-G2_s1": {"kind": "g", "seed": 1, "init": 1, "order": 3, "blocked": None},
    "BAL-L2_s1": {"kind": "l", "seed": 1, "init": 1, "order": 4, "blocked": None},
}
base.WEIGHTS = {"null": (1.0, 1.0), "g": (W_GOAL2, 1.0), "l": (1.0, W_LON2)}
base.PILOT_TAG, base.PILOT_BASE = "PILOT_BAL-G2_s0", "BAL-G2_s0"
base.PARITY_TAG = ""                                                      # no NULL parity step in the second dose (the NULL arms are the first dose's banks)
base.RUN_ORDER = sorted((t for t, v in base.TABLE.items() if v["order"] and not v["blocked"]), key=lambda t: base.TABLE[t]["order"])
base.GATE_AT_PILOT = ["BAL-G2_s0", "BAL-L2_s0"]                            # the pilot job also gates the two full-length seed-0 argvs


def _pilot_out_decl(tag):
    return base.PILOT_OUT_DECL + r"\BAL-G2_s0" if base.is_pilot(tag) else base.frozen_argvs()[tag][base.frozen_argvs()[tag].index("--out") + 1]


base.out_decl = _pilot_out_decl


def _data_paths():
    """the six data files every arm reads, taken from the first run of this dose (the base reads them from its BAL-NULL_s0 tokens, which are not in this table); all four arms carry the same data flags"""
    t = base.tokens(base.RUN_ORDER[0])
    g = lambda f: t[t.index(f) + 1]                                  # noqa: E731
    return {"v9_labels_train": g("--v9-train"), "lead_train": g("--lead-train"), "speed_train": g("--speed-train"), "join_train": g("--join-train"), "p10_prior": g("--p10"), "native_tracks": g("--tracks")}


base.data_paths = _data_paths
globals().update({k: v for k, v in vars(base).items() if not k.startswith("__")})

if __name__ == "__main__":
    sys.exit(base.main(sys.argv[1:]))
