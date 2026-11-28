"""bias_common.py -- shared pieces of the BIASEDIT package (PREREG_V8F_PROOF_ADD4.md, frozen).  Imports the WSP tooling by path (read only, never edited)."""
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
BIAS = "F:/TanitAD-scratch/v8proof/bias"
WSPDIR = "F:/TanitAD-scratch/v8proof/wsp"
if WSPDIR not in sys.path:
    sys.path.insert(0, WSPDIR)
import wsp_common as C  # noqa: E402

PULL = C.PULL
STOP_I = 6                                   # GOAL22 index of STOP_POINT: tree/stack/tanitad/models/v8f_mini.py:46 (TURN_L_I, TURN_R_I, STOP_I = 1, 2, 6); stopbal_read.py:56 STOP_GI = 6
BIAS_KEY = "h_goal.bias"                     # nn.Linear(d, N_GOAL) at v8f_mini.py:732
W1, W2 = 4.5616, 20.81                       # --goal-pos-weight-stop of BAL-G / BAL-G2 (argv.txt, s0 and s1 identical; re-read by bias_build.py)
B1, B2 = math.log(W1), math.log(W2)
PARENTS = {"s0": "NULL_s0", "s1": "NULL_s1"}
TRAINED = {"s0": {"G": "G_s0", "G2": "G2_s0"}, "s1": {"G": "G_s1", "G2": "G2_s1"}}
EDITS = {"GE0": 0.0, "GE1": B1, "GE2": B2}
CKPT = BIAS + "/ckpt"
T1 = BIAS + "/t1"
STOPDEC_OUT = BIAS + "/stopdec"
EDIT_TAGS = [f"{e}_{s}" for s in ("s0", "s1") for e in EDITS]
EXTRA_TRAINED = ("NULL_s1", "G_s1", "G2_s1")          # banked arms the WSP job does not dump
# banked originals for the extra tags; ours for the edited ones
EXTRA_BANKS = {"NULL_s1": PULL + "/stopbal_BAL-NULL_s1", "G_s1": PULL + "/stopbal_BAL-G_s1", "G2_s1": PULL + "/stopbal_BAL-G2_s1"}
DUMP_TAGS = dict(C.DUMP_TAGS)
for _t, _d in EXTRA_BANKS.items():
    DUMP_TAGS[_t] = _d + "/ckpt_last.pt"
for _t in EDIT_TAGS:
    DUMP_TAGS[_t] = f"{CKPT}/{_t}/ckpt_last.pt"
MY_DUMP_TAGS = EDIT_TAGS + list(EXTRA_TRAINED)         # dumped by THIS job; the four s0 / rep arms come from wsp/t1


def parent_of(tag):
    return PARENTS[tag.split("_")[1]] if tag[:2] == "GE" else None
