"""ds_lib.py -- shared constants and pure helpers of the DSCALE10K block-1 read (card PREREG_V8F_DSCALE10K, frozen; sha256 of the frozen text 48a74901..., MM rulings 2026-11-16 / 11-17).
No torch.  Evidence class of every number the read prints: MEASURED (ours; dumps and T0 evals, zero training)."""
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
COMMIT = "cdbff971fa6dad4d20b937fe8abda6cb9f622adb"        # code of record of every DSCALE10K arm (the card, MM ruling 4)
TREE = os.environ.get("DS_TREE", "F:/TanitAD-scratch/otband_read/tree")     # export of cdbff971 (stack + taniteval); tree_problem() re-hashes the key files against the commit's blobs
TREE_ZERO = os.environ.get("DS_TREE_ZERO", "F:/TanitAD-scratch/strat1/read/tree")   # the STRAT1 read tree (git e81bde01): the ONLY tree whose v8f_mini_eval.py has --nav-withheld; runs the `zero` pass and the K-EVALX `realx` pass
ZERO_GIT = "e81bde01"
VARIANT_TREE = {"real": "read", "donor": "read", "zero": "zero", "realx": "zero", "donorx": "zero"}     # route variant -> which tree's evaluator runs it (READ_PINS.json carries both trees' sha256)
S1R = os.environ.get("DS_STRAT1_READ", "F:/TanitAD-scratch/strat1/read")     # the STRAT1 read folder: pinned inputs, the unedited STRATEGIC helpers
INP = S1R + "/inputs"
CK = os.environ.get("DS_CKPTS", HERE + "/ckpts")             # <arm>.<stage>/{ckpt_last.pt, BANK_RECORD.json, summary.json}
T1 = os.environ.get("DS_T1", HERE + "/t1")
OUT = os.environ.get("DS_OUT", HERE + "/out")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ART = "E:/Projects/TanitAD-artifacts"
SF = ART + "/sfresh_inputs"
RAWT = ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
POD_ROOT = "/root/dscale10k"
NB, SEED = 2000, 0
EVAL_N_EPISODES, EVAL_N_WINDOWS = 141, 2399
STEPS = {"A": 19164, "B": 3000}                              # card 2.3: stage A steps (every block-1 arm), stage B steps
MIN_F_GB, MIN_COMMIT_GB = 25.0, 5.5                          # the card's own F: bar (brief item 1) and the dev-box commit rule
PINS = {                                                     # sha256 of the pinned eval inputs (the same files as strat1_lib.PINS; re-hashed on use)
    "join_eval141.npz": "f0ea722f59c1a4d8ba81ce7f9b204dd3a28950fc1c16778f63c2962786e5206a",
    "eval_grid_windows.json": "659c3b65e364ea5de87a4e17d076eba5645df95cdcf39d8abc08caab1b2d594c",
    "navtok_table.npz": "2836369e3e14a8c3c49a6b1a42116c36cda522ca42f98a6bcbc8c4e4d45c8992",
    "navtext_nav.npz": "03bd75bc3a3de17928681bedb8e46a48a891304de45d395c49cc876acf6e6160",
}
TREE_FILES = ("stack/scripts/v8f_t1_eval.py", "stack/scripts/v8f_mini_eval.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py",
              "taniteval/taniteval/ci.py", "taniteval/taniteval/v8f_consistency.py", "taniteval/taniteval/four_families.py")
# arm role -> training seed (card 2.3); R_s0 is stage A only
ARMS = {"B_s0": 0, "U_s0": 0, "B_rep": 2, "R_s0": 0}
STAGES_OF = {"B_s0": ("A", "B"), "U_s0": ("A", "B"), "B_rep": ("A", "B"), "R_s0": ("A",)}
READ_ITEMS = tuple((a, s) for a in ARMS for s in STAGES_OF[a])      # (arm, stage): 7 checkpoints
NOT_YET_ARMS = ("B_s1", "U_s1", "UE_s0", "BL_s0", "X_s0", "B_s3", "U_s3", "B_s4", "U_s4")


def item(arm, stage):
    return f"{arm}.{stage}"


def t1_tag(arm, stage):
    return f"DS-{arm}-{stage}_m10_causal_heads"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pinned(name, check=True):
    p = os.path.join(INP, name)
    if check and name in PINS:
        got = sha256_file(p)
        if got != PINS[name]:
            raise RuntimeError(f"pinned input {name} sha256 {got[:12]} != {PINS[name][:12]}")
    return p


def free_gb(path):
    return shutil.disk_usage(path).free / 2 ** 30


def commit_free_gb():
    import ctypes

    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = M()
    m.dwLength = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 2 ** 30


def disk_problem(path=None, need=MIN_F_GB, free=None):
    """-> reason or None: F: must hold at least `need` GB free (the card's bar)"""
    f = free_gb(path or HERE) if free is None else free
    return None if f >= need else f"F: has {f:.1f} GB free, the card needs at least {need:.0f} GB"


def bank_problem(arm, stage, ck=None):
    """-> reason or None.  The checkpoint is read only when its BANK_RECORD.json (written by ds_pull.py from the banker's record: sha256, source) names the sha256 the file hashes to, the record names this
    arm, stage and commit, and the stage's summary.json is done at the card's step count.  Files only; launches nothing."""
    d = f"{ck or CK}/{item(arm, stage)}"
    p, rp = f"{d}/ckpt_last.pt", f"{d}/BANK_RECORD.json"
    if not os.path.isfile(p):
        return f"REFUSED: checkpoint missing ({p})"
    if not os.path.isfile(rp):
        return f"REFUSED: BANK_RECORD.json missing ({rp})"
    r = json.load(open(rp, encoding="utf-8"))
    if (r.get("arm"), r.get("stage")) != (arm, stage):
        return f"REFUSED: the bank record names {r.get('arm')}.{r.get('stage')}, not {arm}.{stage}"
    if r.get("commit") != COMMIT:
        return f"REFUSED: the bank record names commit {r.get('commit')}, the code of record is {COMMIT}"
    if not r.get("sha256") or r.get("source") not in ("hf", "pod"):
        return "REFUSED: the bank record holds no sha256 / source"
    if r.get("steps") != STEPS[stage]:
        return f"REFUSED: the stage's summary says {r.get('steps')} steps, the card's stage {stage} is {STEPS[stage]}"
    if sha256_file(p) != r["sha256"]:
        return "REFUSED: checkpoint sha256 differs from its bank record"
    return None


def tree_of(variant):
    """the tree root that runs a route variant (explicit map, never a default)"""
    return {"read": TREE, "zero": TREE_ZERO}[VARIANT_TREE[variant]]


def tree_problem(tree=None, key="tree_sha256"):
    """-> reason or None: the key files of the read tree equal the commit's blobs (READ_PINS.json, written by make_pins.py from `git cat-file blob cdbff971:<path>`)"""
    p = HERE + "/READ_PINS.json"
    if not os.path.isfile(p):
        return "READ_PINS.json missing"
    pins = json.load(open(p, encoding="utf-8"))
    if pins.get("commit") != COMMIT:
        return "READ_PINS.json is not for the code of record"
    for rel, want in pins[key].items():
        q = f"{tree or TREE}/{rel}"
        if not os.path.isfile(q):
            return f"tree file missing: {q}"
        if sha256_file(q) != want:
            return f"tree file differs from READ_PINS.json: {rel}"
    return None


def load_ci(tree=None):
    import importlib.util
    p = (tree or TREE) + "/taniteval/taniteval/ci.py"
    spec = importlib.util.spec_from_file_location("ds_ci", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.SOURCE = p
    return m
