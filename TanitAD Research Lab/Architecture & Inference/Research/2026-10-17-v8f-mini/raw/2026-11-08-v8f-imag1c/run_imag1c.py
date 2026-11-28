"""IMAG1C real read (IMPL_CHOICES_IMAG1C). python run_imag1c.py [smoke]. 2 single-thread workers; every draw is saved to draws_imag1c/<role>_<s>.pkl (resume-safe).
Writes RESULT_IMAG1C.json (analysis) and RUN_STATUS_IMAG1C.txt. Memory: free physical RAM is checked before the pool starts and kept >= 2 GB."""
import ctypes, hashlib, json, os, pickle, sys, time, types
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import imag1c_lib as ML  # noqa: E402

CTX = "F:/TanitAD-scratch/imag1c_ctx.pkl"
DRAWS = os.path.join(HERE, "draws_imag1c")
NPROC = 2
_C = None


def free_gb():
    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
    s = MS(); s.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ap / 2 ** 30


def load_ctx():
    d = pickle.load(open(CTX, "rb"))
    c = types.SimpleNamespace(**d)
    c.BZ = {t: {l: np.asarray(v, np.float64) for l, v in c.BZ[t].items()} for t in c.tags}
    return ML.finish_ctx(c)


def init():
    global _C
    _C = load_ctx()


def work(job):
    role, s = job
    f = os.path.join(DRAWS, f"{role}_{s}.pkl")
    if os.path.exists(f):
        return job, 0.0
    t0 = time.time()
    r = ML.run_draw(_C, s, role)
    pickle.dump(r, open(f + ".tmp", "wb"), protocol=4)
    os.replace(f + ".tmp", f)
    return job, time.time() - t0


def main(smoke=False):
    os.makedirs(DRAWS, exist_ok=True)
    t0 = time.time()
    g = free_gb()
    print(f"free RAM {g:.2f} GB", flush=True)
    waited = 0
    while g < 3.2 and waited < 8 * 3600:                  # the box is crowded: wait for memory (>= 2 GB must stay free after a worker's ~1 GB)
        time.sleep(30); waited += 30; g = free_gb()
    if g < 3.2:
        open(os.path.join(HERE, "RUN_STATUS_IMAG1C.txt"), "w").write(f"NOT RUN: free RAM {g:.2f} GB after {waited} s"); sys.exit(2)
    globals()["NPROC"] = 2 if g >= 4.4 else 1
    print(f"free RAM {g:.2f} GB after waiting {waited} s; workers {NPROC}", flush=True)
    jobs = [("real", 0)] + [("ref", s) for s in ML.REF_SEEDS] + [("cal", s) for s in ML.CAL_SEEDS]
    if smoke:
        jobs = [("real", 0), ("ref", 1), ("cal", 1001)]
        globals()["DRAWS"] = os.path.join(HERE, "draws_imag1c_smoke"); os.makedirs(globals()["DRAWS"], exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(NPROC, initializer=init) as ex:
        for job, dt in ex.map(work, jobs):
            print(job, round(dt, 1), "s", round(time.time() - t0), "s total", flush=True)
    if smoke:
        return
    c = load_ctx()
    ld = lambda role, s: pickle.load(open(os.path.join(DRAWS, f"{role}_{s}.pkl"), "rb"))
    real, refs, cals = ld("real", 0), [ld("ref", s) for s in ML.REF_SEEDS], [ld("cal", s) for s in ML.CAL_SEEDS]
    R = ML.analyse(c, real, refs, cals)
    R["wall_s"] = round(time.time() - t0, 1)
    conv = lambda o: o.item() if isinstance(o, np.generic) else (o.tolist() if isinstance(o, np.ndarray) else str(o))
    json.dump(R, open(os.path.join(HERE, "RESULT_IMAG1C.json"), "w"), indent=1, default=conv)
    open(os.path.join(HERE, "RUN_STATUS_IMAG1C.txt"), "w").write(f"banner={R['banner']}\ncontrol2_pass={R['control2']['pass']}\ncontrol3_pass={R['control3']['pass']}\n")
    print("done", R["banner"], flush=True)


if __name__ == "__main__":
    main(len(sys.argv) > 1 and sys.argv[1] == "smoke")
