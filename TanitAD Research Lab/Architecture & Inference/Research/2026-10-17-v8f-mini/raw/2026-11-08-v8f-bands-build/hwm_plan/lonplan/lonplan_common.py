"""LON-PLAN build: shared paths and helpers (AMEND_HWMPLAN_3, frozen, sha256 7bbdc9fe...).
The code of record (run_hwm.py, hwm_plan.py, hwm_eval.py, hwm_data.py, plantest2_lib.py, plantest_lib.py) is IMPORTED unchanged from LONPLAN_CODE_ROOT (default: the local copy that
section 8.1 declares sha256-equal to the repo blobs of commit 4dc4876c). Nothing under that root is written. CPU only."""
import hashlib, json, os, sys
sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

CODE_ROOT = os.environ.get("LONPLAN_CODE_ROOT", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory")
HWM_DIR = os.path.join(CODE_ROOT, "hwm_plan")
for _d in ("wmact", "plantest2", "plantest", "hwm_plan"):
    _p = os.path.join(CODE_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

THREADS = 2                                  # the one thread count of every LON-PLAN run (dumps at other thread counts drift)
RECORD_FILES = ("run_hwm.py", "hwm_plan.py", "hwm_eval.py", "hwm_data.py", "goal_head.py")
RECORD_FILES_OTHER = (("plantest2", "plantest2_lib.py"), ("plantest", "plantest_lib.py"))


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def code_of_record_sha256():
    """sha256 of every code-of-record file the run reads, keyed by its file name (a missing file reads None, never silently dropped)."""
    out = {}
    for f in RECORD_FILES:
        p = os.path.join(HWM_DIR, f)
        out[f] = sha256_file(p) if os.path.exists(p) else None
    for d, f in RECORD_FILES_OTHER:
        p = os.path.join(CODE_ROOT, d, f)
        out[f] = sha256_file(p) if os.path.exists(p) else None
    return out


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=float, sort_keys=False)
