"""c120_lib.py -- shared constants, pins and refusals for the CKPT120 read (PREREG_V8F_CKPT120, frozen 2026-11-16; MM rulings 1-12).  Pure python / numpy; no torch.
Code of record = commit 213250ce (branch agent/v8f-ckpt120-20261117), exported to ./tree (git archive, autocrlf off).  Comparison arms are the BANKED STRAT1 arms (their dumps / T0 passes are reused, never re-made)."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)                                   # F:/TanitAD-scratch/strat1/read
sys.path.insert(0, R)
import strat1_lib as SL  # noqa: E402  (the frozen STRAT1 constants, pinned inputs, paired(), grid_windows())

COMMIT = "213250cee3c0a695c22f7ce823f4a7aabee6e311"
TREE = os.environ.get("CK120_TREE", HERE + "/tree")
INP = HERE + "/inputs"
CKD = os.environ.get("CK120_CKPTS", HERE + "/ckpts")          # pulled arms: ckpts/<tag>/{ckpt_last.pt, BANK_MANIFEST.json, argv.json}
OUT = os.environ.get("CK120_OUT", HERE + "/out")
T1 = os.environ.get("CK120_T1", HERE + "/t1")                 # new-arm T1 dumps (causal, m 10) and the K9b logged / batch 64 tick dumps
REACH = HERE + "/reach"
NEW_ARMS = ("H1_s0", "H1_s1", "H2_s0", "H2_s1", "H1_rep", "R1_s0", "R2_s0")     # the chain's priority order (MM ruling 9)
NAV_S_ARMS = ("H2_s0", "H2_s1", "R2_s0")                      # arms with --nav-strat on (S reads the route text's next turn)
INIT_OF = {"H1_s0": 0, "H1_s1": 1, "H2_s0": 0, "H2_s1": 1, "H1_rep": 0, "R1_s0": 0, "R2_s0": 0}
SEED_OF = {"H1_s0": 0, "H1_s1": 1, "H2_s0": 0, "H2_s1": 1, "H1_rep": 2, "R1_s0": 0, "R2_s0": 0}
BASE_OF = {"H1_s0": "S1_s0", "H1_s1": "S1_s1", "H2_s0": "H1_s0", "H2_s1": "H1_s1"}      # H1 - S1 and H2 - H1 (one lever each)
FLOOR_PAIRS = (("H1_rep", "H1_s0"), ("S1_s10", "S1_s0"))     # F = the larger of the two replicate pairs (card section 0 item 5)
PINS = {                                                     # the new tables (sha256; also banked on HF rev c95d3ac0)
    "rc120_table.npz": "0f122570be5c0b9a60641ea9938102deb212752f5a67c411049e9dd116e01bf2",
    "navtext_nav_s.npz": "0f7431a18622bf11cc678a39db5fc230b21eeefb5d254928617b56525db57c81",
    "eval141_grid_truth.npz": "da9529a1f9bd4599ec44bbaff2827e12c033f314a8ff9167e94a66dbfcdccc73",
}
TREE_PINS = {                                                # blobs of the commit (git cat-file, raw bytes): the evaluators the read runs
    "stack/scripts/v8f_mini_eval.py": "008f48b79fd11ca553d118d8e29ece6aea29d1dd013653ac23b1dc7fe2dad558",
    "stack/scripts/v8f_t1_eval.py": "180044eb28246ba94a18366ed337f061ec659d6667496c0f88a4a325a0d65c93",
}
HF_REPO = "Sayood/tanitad-v8f-strat1-runs"
ARM_FILES = ("ckpt_last.pt", "argv.json")


def sha256_file(p):
    return SL.sha256_file(p)


def pinned(name):
    """path of a pinned CKPT120 input, re-hashed; a mismatch raises and nothing is read"""
    p = f"{INP}/{name}"
    got = sha256_file(p)
    if got != PINS[name]:
        raise RuntimeError(f"pinned input {name} sha256 {got[:12]} != {PINS[name][:12]}")
    return p


def tree_problem(tree=None):
    t = tree or TREE
    for rel, want in TREE_PINS.items():
        p = f"{t}/{rel}"
        if not os.path.isfile(p):
            return f"tree file missing: {p}"
        if sha256_file(p) != want:
            return f"tree file differs from the pin: {rel}"
    for rel in ("stack/scripts/v8f_mini_eval_far.py", "stack/scripts/v8f_t1_eval_far.py", "stack/scripts/ckpt120_cells.py", "stack/scripts/ckpt120_reach.py"):
        if not os.path.isfile(f"{t}/{rel}"):
            return f"tree file missing: {t}/{rel}"
    return None


def bank_refusal(arm, ckd=None, commit=COMMIT):
    """-> reason or None.  Files only.  The arm's checkpoint must exist and equal its BANK_MANIFEST.json sha256, which must name this tag and the CKPT120 code commit; the reach verdict must be PASS."""
    d = f"{ckd or CKD}/{arm}"
    ck, man_p = f"{d}/ckpt_last.pt", f"{d}/BANK_MANIFEST.json"
    if not os.path.isfile(ck):
        return f"REFUSED: checkpoint missing ({ck})"
    if not os.path.isfile(man_p):
        return f"REFUSED: BANK_MANIFEST.json missing ({man_p})"
    man = json.load(open(man_p, encoding="utf-8"))
    want = ((man.get("files") or {}).get("ckpt_last.pt") or {}).get("sha256")
    if not want:
        return "REFUSED: BANK_MANIFEST.json holds no ckpt_last.pt sha256"
    if man.get("tag") != arm:
        return f"REFUSED: the bank manifest names tag {man.get('tag')}, not {arm}"
    if man.get("commit") != commit:
        return f"REFUSED: the bank was trained at commit {man.get('commit')}, the code of record is {commit}"
    if sha256_file(ck) != want:
        return "REFUSED: checkpoint sha256 differs from its bank manifest"
    return None


def reach_verdict(arm, reach_dir=None):
    """-> (verdict, path).  verdict in PASS / HEAD-DEAD / INCONCLUSIVE / MISSING (a missing or unreadable file is never a PASS)"""
    p = f"{reach_dir or REACH}/{arm}.json"
    try:
        v = json.load(open(p, encoding="utf-8")).get("verdict")
    except Exception:                                         # noqa: BLE001
        return "MISSING", p
    return (v if v in ("PASS", "HEAD-DEAD", "INCONCLUSIVE") else "MISSING"), p


def banked_t1_dump(arm):
    return f"{SL.HERE}/t1/dump_STRAT1-{arm}_m10_causal_heads"


def new_t1_dump(arm):
    return f"{T1}/dump_CK120-{arm}_m10_causal_heads"


def banked_t0(arm, variant):
    return f"{SL.HERE}/out/t0/eval_STRAT1-{arm}_{variant}.npz"


def new_t0(arm, variant):
    return f"{OUT}/t0/eval_CK120-{arm}_{variant}.npz"


def new_t0_far(arm, variant):
    return f"{OUT}/t0/eval_CK120-{arm}_{variant}_far.npz"
