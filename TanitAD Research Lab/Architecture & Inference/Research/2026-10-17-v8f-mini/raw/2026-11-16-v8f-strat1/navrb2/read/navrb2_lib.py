"""navrb2_lib.py -- constants, window masks and refusals for the NAVRB2 read (PREREG_V8F_NAVRB2.md, FROZEN, sha256 e238ba8d; sections 5 and 6).
Pure numpy (torch only inside hist_of()).  Evidence class of every number the NAVRB2 readers print: MEASURED (ours; dumps / T0 evals, zero training); tier stamped per block.
Reuses the pinned STRAT1 reader family under F:/TanitAD-scratch/strat1/read UNEDITED (strat1_read.load_arm_cells, obd_read.cell_delta, strat1_strategic, strat1_ctx_read)."""
import hashlib
import json
import os
import sys

import numpy as np

NR = os.path.dirname(os.path.abspath(__file__))                                    # .../strat1/navrb2/read
SR = os.environ.get("NAVRB2_STRAT1_READ", "F:/TanitAD-scratch/strat1/read")        # the STRAT1 read dir (banked dumps, T0 passes, pinned inputs, frozen readers)
TREE = os.environ.get("NAVRB2_TREE", NR + "/tree")                                 # git -c core.autocrlf=false archive e71782a3 stack taniteval
CK = os.environ.get("NAVRB2_CKPTS", NR + "/ckpts")
OUT = os.environ.get("NAVRB2_OUT", NR + "/out")
T1 = os.environ.get("NAVRB2_T1", NR + "/t1")
T0 = OUT + "/t0"
COMMIT = "e71782a3b48cc975ab4c2db9f559c4a3c99624b3"                                # the commit the four R arms trained at (gate_token.json "commit", the bank manifests)
CARD_SHA256 = "e238ba8d3e41722a4d045ad728b9211c40b9acb3d8d918a1622d463535e06d27"
RB_NPZ = "F:/TanitAD-scratch/strat1/navrb2/table/navtext_nav_rb.npz"
RB_SHA256 = "f4c4ab7a8abf2498e532d648874df2549b7b83292b31766ad80e581d009bcebc"
RB_REPORT = "F:/TanitAD-scratch/strat1/navrb2/table/navtext_nav_rb_report.json"
R_ARMS = ("N1R_s0", "S1R_s0", "N1R_s1", "S1R_s1")
BANKED_OF = {"N1R_s0": "N1_s0", "S1R_s0": "S1_s0", "N1R_s1": "N1_s1", "S1R_s1": "S1_s1"}
FAMILY_OF = {"N1R": "N1", "S1R": "S1"}
SEEDS = (0, 1)
Z = "Z"                                                      # suffix of the zero-channel dump tag (Z-EVAL / Z-USE): N1R_s0Z
RB_EXPECT_WINDOWS, RB_EXPECT_EPISODES = 235, 16
RB_TURN_EXPECT_WINDOWS, RB_TURN_EXPECT_EPISODES = 43, 6
RB_TURN_EPISODES = (31, 52, 69, 102, 126, 135)
NB, SEED = 2000, 0
BANKED_DUMPS = SR + "/t1"
BANKED_T0 = SR + "/out/t0"
C0REP_RESULT = SR + "/c0rep/out/RESULT_STRAT1_C0REP.json"
STRAT1_RESULT = SR + "/out/RESULT_STRAT1.json"
N1S10_RESULT = SR + "/n1s10/out/RESULT_STRAT1_N1S10.json"
# reading choices committed before any NAVRB2 number exists (the card is silent; each is printed into the result)
READING = {
    "r1_Z-USE_separated": "Z-USE is 'separated' at a seed when, on the RB windows, real - zero is separated (CI excludes 0, either side) on at least one of the 24 cells; TAC-LAT alone is printed beside",
    "r2_guard": "the guard holds when no all-window cell is a cost at either seed under F2 and there is no ROUTE-COST; a cost is separated on the worse side and |delta| > F2",
    "r3_not_separated_both": "'not separated at both seeds' = not separated at seed 0 and not separated at seed 1",
    "r4_T0_donor_channel": "T0 donor-nav passes: the evaluator swaps only the nav array, so the channel is held at ZERO beside the donor nav (the T1 donor rule would use the donor's rows; this deviation is named in the result)",
    "r5_zeval_mask": "Z-EVAL windows = grid windows whose nav_rb frames over the item's history (cfg.hmax rows up to NOW, clipped at the clip start) are all 0.0",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rb_npz_checked():
    got = sha256_file(RB_NPZ)
    if got != RB_SHA256:
        raise RuntimeError(f"navtext_nav_rb.npz sha256 {got[:12]} != pinned {RB_SHA256[:12]}")
    return RB_NPZ


def hist_of(ckpt):
    """-> cfg.hmax of a checkpoint (the item's frame count; the Z-EVAL mask window)"""
    import torch
    return int(torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]["hmax"])


def window_state(eval_nav_rb, eval_off, ci, t, hmax):
    """-> (now [N] bool, hist_any [N] bool): the nav_rb value at the window's NOW row, and whether ANY of the item's rows (NOW-hmax+1 .. NOW, clipped at the clip's first row) is on.  Pure.
    eval_nav_rb [rows] or [rows, 1]; eval_off [clips] first row of each clip; ci, t the grid windows."""
    x = np.asarray(eval_nav_rb, np.float32).reshape(-1)
    off = np.asarray(eval_off, np.int64)
    now = np.zeros(len(ci), bool)
    anyh = np.zeros(len(ci), bool)
    for i, (c, tt) in enumerate(zip(np.asarray(ci), np.asarray(t))):
        r = int(off[c]) + int(tt)
        lo = max(int(off[c]), r - int(hmax) + 1)
        now[i] = x[r] > 0.5
        anyh[i] = bool((x[lo:r + 1] > 0.5).any())
    return now, anyh


def rb_windows(grid_path=None, join_path=None):
    """-> dict of the eval grid windows and the NAVRB2 strata: ci, t, eid, rb (state on at NOW), rb_zero_hist (no nav_rb frame on in the history: Z-EVAL needs hmax -> see window_state)"""
    sys.path.insert(0, SR)
    import strat1_lib as L
    join = dict(np.load(SR + "/inputs/join_eval141.npz", allow_pickle=False))
    ci, t = L.grid_windows(grid_path or L.pinned("eval_grid_windows.json", SR + "/inputs"), join["clip"])
    z = np.load(rb_npz_checked(), allow_pickle=False)
    if list(map(str, z["eval_clip"])) != list(map(str, join["clip"])):
        raise RuntimeError("navtext_nav_rb.npz eval clip order differs from the eval join's")
    return {"ci": ci, "t": t, "eid": ci.copy(), "eval_nav_rb": z["eval_nav_rb"], "eval_off": z["eval_off"], "clip": join["clip"]}


def strata(w, hmax):
    """-> {"rb": [N] bool, "zero_hist": [N] bool, "n": counts}: RB = state on at NOW; zero_hist = no frame on in the history window (Z-EVAL)"""
    now, anyh = window_state(w["eval_nav_rb"], w["eval_off"], w["ci"], w["t"], hmax)
    return {"rb": now, "zero_hist": ~anyh}


def count_stratum(mask, eid):
    return {"n_windows": int(np.sum(mask)), "n_episodes": int(len(np.unique(np.asarray(eid)[np.asarray(mask, bool)])))}


def refusal(arm, ck_root=None):
    """-> reason or None for an R arm: its checkpoint and BANK_MANIFEST.json must exist, name this tag, name COMMIT, and the checkpoint sha256 must equal the manifest's (files only)"""
    root = ck_root or CK
    tag = arm[:-1] if arm.endswith(Z) else arm
    if tag not in R_ARMS:
        return f"unknown arm {arm}"
    ck, mp = f"{root}/{tag}/ckpt_last.pt", f"{root}/{tag}/BANK_MANIFEST.json"
    if not os.path.isfile(ck):
        return f"REFUSED: checkpoint missing ({ck})"
    if not os.path.isfile(mp):
        return f"REFUSED: BANK_MANIFEST.json missing ({mp})"
    man = json.load(open(mp, encoding="utf-8"))
    want = ((man.get("files") or {}).get("ckpt_last.pt") or {}).get("sha256")
    if not want:
        return "REFUSED: BANK_MANIFEST.json holds no ckpt_last.pt sha256"
    if man.get("tag") != tag:
        return f"REFUSED: the bank manifest names tag {man.get('tag')}, not {tag}"
    if man.get("commit") != COMMIT:
        return f"REFUSED: the bank was trained at commit {man.get('commit')}, the code of record is {COMMIT}"
    if sha256_file(ck) != want:
        return "REFUSED: checkpoint sha256 differs from its bank manifest"
    return None


def tree_problem(tree=None):
    """the read tree must be the e71782a3 export: the two pinned evaluators and ci.py equal the STRAT1 pins, and the wrappers/model/data exist (sha256 recorded in the result)"""
    t = tree or TREE
    pins = json.load(open(SR + "/READ_PINS.json", encoding="utf-8"))["tree_sha256"]
    for rel in ("stack/scripts/v8f_t1_eval.py", "stack/scripts/v8f_mini_eval.py", "taniteval/taniteval/ci.py", "taniteval/taniteval/v8f_consistency.py", "taniteval/taniteval/four_families.py"):
        q = f"{t}/{rel}"
        if not os.path.isfile(q):
            return f"tree file missing: {q}"
        if sha256_file(q) != pins[rel]:
            return f"tree file differs from the STRAT1 pin (the pinned evaluators must be byte-equal): {rel}"
    for rel in ("stack/scripts/v8f_t1_eval_rb.py", "stack/scripts/v8f_mini_eval_rb.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py"):
        if not os.path.isfile(f"{t}/{rel}"):
            return f"tree file missing: {t}/{rel}"
    return None


def import_check(py, tree=None):
    """-> problem or None: tanitad / taniteval import from the read tree in a child with the dump's environment (';'-joined Windows paths; memory: ':'-joined imports the G: editable install)"""
    import subprocess
    t = tree or TREE
    env = dict(os.environ, PYTHONPATH=";".join(os.path.normpath(p) for p in (t + "/stack", t + "/taniteval")), CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([py, "-c", "import tanitad, taniteval; print(tanitad.__file__); print(taniteval.__file__)"], env=env, capture_output=True, encoding="utf-8")
    root = os.path.normpath(t).lower()
    paths = [x.strip() for x in r.stdout.splitlines()]
    if r.returncode or len(paths) != 2 or not all(os.path.normpath(x).lower().startswith(root) for x in paths):
        return f"tanitad / taniteval do not import from the read tree: {paths} {r.stderr[-200:]}"
    return None
