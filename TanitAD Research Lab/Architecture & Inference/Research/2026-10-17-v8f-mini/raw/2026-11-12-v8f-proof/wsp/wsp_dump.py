"""wsp_dump.py -- T1-kin dump of one WSP tag (same argv as theory/stopbal/read/run_dumps_stopbal.py: v8f_t1_eval.py run, m=10, causal, deadreckon, logged scene, periodic, --dump-ticks --dump-heads,
CPU, 2 threads) from OUR tree (commit 3e264bbf, stack + taniteval).  Strict state-dict load is v8f_t1_eval.py:511 (load_state_dict default strict=True).
Usage: python -I wsp_dump.py TAG [TAG ...] [--out DIR] [--max-windows N] [--print-argv]
  default --out = wsp/t1; a TAG whose done file exists is skipped.  --max-windows N is the self-test (evenly spaced subset; no done file is written, outputs go to --out only).
Writes <out>/{argv,log,done}_<TAG>_m10_causal_heads.* (done only after 141 episode dumps + header + ticks + checkpoint sha256 unchanged)."""
import ctypes
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp_common as C  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ART = "E:/Projects/TanitAD-artifacts"
RAW = ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
INPUTS = {"--tracks": ART + "/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache": ART + "/sfresh_inputs/cache",
          "--eval-grid": RAW + "/eval_grid_windows.json", "--eval-join": RAW + "/join_eval141.npz"}
MIN_FREE_GB = 4.0


def free_gb():
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = M()
    m.dwLength = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def make_argv(tag, out, max_windows=0):
    full = f"{tag}_m10_causal_heads"
    argv = [PY, C.TREE + "/stack/scripts/v8f_t1_eval.py", "run", "--ckpt", C.DUMP_TAGS[tag], "--tag", full]
    for k, v in INPUTS.items():
        argv += [k, v]
    argv += ["--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal",
             "--dump-ticks", out + "/ticks", "--dump-heads", "--device", "cpu"]
    if max_windows:
        argv += ["--max-windows", str(max_windows), "--no-analyze"]
    return full, argv


def run(tag, out, max_windows=0):
    full, argv = make_argv(tag, out, max_windows)
    done = f"{out}/done_{full}.txt"
    rec = {"tag": tag}
    if os.path.exists(done) and not max_windows:
        rec["status"] = "skip (done file exists)"
        return rec
    ck = C.DUMP_TAGS[tag]
    if not os.path.isfile(ck):
        rec["status"] = f"REFUSED: checkpoint missing {ck}"
        return rec
    want = None
    for mf in (os.path.dirname(ck) + "/BANK_MANIFEST.json", os.path.dirname(ck) + "/WSP_MANIFEST.json"):
        if os.path.isfile(mf):
            j = json.load(open(mf, encoding="utf-8"))
            want = (j.get("files", {}).get("ckpt_last.pt", {}) or {}).get("sha256") or j.get("ckpt_sha256")
    got = C.sha(ck)
    if want is None or want != got:
        rec["status"] = f"REFUSED: checkpoint sha256 {got[:12]} does not match its manifest ({str(want)[:12]})"
        return rec
    t0 = time.time()
    while free_gb() < MIN_FREE_GB:
        if time.time() - t0 > 3600:
            rec["status"] = "REFUSED: free memory below 4 GB for 1 h"
            return rec
        time.sleep(30)
    os.makedirs(out + "/ticks", exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS=str(C.THREADS), MKL_NUM_THREADS=str(C.THREADS), PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    open(f"{out}/argv_{full}.txt", "w", encoding="utf-8").write("\n".join(argv) + "\n")
    t1 = time.time()
    with open(f"{out}/log_{full}.txt", "w", encoding="utf-8") as lf:
        rc = subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
    rec.update(rc=rc, wall_s=round(time.time() - t1, 1), ckpt_sha256=got, threads=C.THREADS)
    if rc != 0:
        rec["status"] = "FAILED (see log)"
        return rec
    if max_windows:
        rec["status"] = "SELFTEST-ok (no done file)"
        return rec
    hdr, dd, tk = f"{out}/t1v8f_{full}.json", f"{out}/dump_{full}", f"{out}/ticks/ticks_{full}.npz"
    n = len([f for f in os.listdir(dd) if f.startswith("ep") and f.endswith(".npz")]) if os.path.isdir(dd) else 0
    if not (os.path.exists(hdr) and n == 141 and os.path.exists(tk)) or C.sha(ck) != got:
        rec["status"] = f"FAILED: outputs incomplete (header {os.path.exists(hdr)}, episode dumps {n}, ticks {os.path.exists(tk)}) or checkpoint changed"
        return rec
    json.dump({"tag": full, "limit": "causal", "ckpt_sha256": got, "n_dumps": n, "ticks": tk, "header": hdr, "rc": rc, "wall_s": rec["wall_s"], "threads": C.THREADS,
               "smoke": False, "written_by": "wsp_dump.py"}, open(done, "w", encoding="utf-8"))
    rec["status"] = "ok"
    return rec


def main():
    a = sys.argv[1:]
    out, mw, pa = C.T1, 0, False
    if "--out" in a:
        i = a.index("--out")
        out = a[i + 1]
        del a[i:i + 2]
    if "--max-windows" in a:
        i = a.index("--max-windows")
        mw = int(a[i + 1])
        del a[i:i + 2]
    if "--print-argv" in a:
        a.remove("--print-argv")
        pa = True
    assert a and all(t in C.DUMP_TAGS for t in a), f"tags must be among {list(C.DUMP_TAGS)}"
    for k, v in INPUTS.items():
        assert os.path.exists(v), (k, v)
    assert os.path.exists(C.TREE + "/stack/scripts/v8f_t1_eval.py")
    for t in a:
        if pa:
            print(" ".join(make_argv(t, out, mw)[1]))
            continue
        r = run(t, out, mw)
        print(time.strftime("%H:%M:%S"), json.dumps(r), flush=True)
        if not (r["status"] in ("ok",) or r["status"].startswith("skip") or r["status"].startswith("SELFTEST")):
            raise SystemExit(3)


if __name__ == "__main__":
    main()
