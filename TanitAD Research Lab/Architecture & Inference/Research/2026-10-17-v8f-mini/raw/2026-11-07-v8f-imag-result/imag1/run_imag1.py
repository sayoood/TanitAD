"""PREREG_V8F_IMAG1 driver. Card prefix 9,212 bytes sha256 14846eef...; IMPL_CHOICES_IMAG1.md sha256 in IMPL_CHOICES_IMAG1.sha256 (both asserted before anything is computed).
Per dataset (eval141 = deciding, train = beside): the real stage (s = 0) and 199 reference + 20 calibration permuted-label draws in a pool of single-thread worker processes,
then the analysis. Draws are appended to draws_<ds>.jsonl (resume-safe). Writes RESULT_IMAG1.json (banked after each dataset) and IMAG1_RUN.done (rc=<n>).
Usage (Git bash / cmd, cwd = this directory, OMP_NUM_THREADS=1): python -u run_imag1.py eval141 train"""
import hashlib, json, math, os, pickle, sys, time, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import numpy as np

CARD, CARD_N, CARD_SHA = "../PREREG_V8F_IMAG1.md", 9212, "14846eef59472b64a27063dcc72a4d52013d0ae839fcbd209e38db5805c46eb8"
IMPL = "IMPL_CHOICES_IMAG1.md"
NPROC = int(os.environ.get("IMAG1_NPROC", "7"))
SCR = "F:/TanitAD-scratch"
_C = None


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def check_inputs():
    assert sha_bytes(open(CARD, "rb").read()[:CARD_N]) == CARD_SHA, "the card prefix changed"
    want = open("IMPL_CHOICES_IMAG1.sha256", encoding="utf-8").read().split()[0]
    assert sha_bytes(open(IMPL, "rb").read()) == want, "IMPL_CHOICES_IMAG1.md changed after its sha256 was recorded"
    return want


def jsonable(o):
    if isinstance(o, dict):
        return {("|".join(str(x) for x in k) if isinstance(k, tuple) else str(k)): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def load_ctx(ds):
    from types import SimpleNamespace
    with open(f"{SCR}/imag1_ctx_{ds}.pkl", "rb") as fh:
        return SimpleNamespace(**pickle.load(fh))


def init_worker(ds):
    global _C
    import torch
    torch.set_num_threads(1)
    _C = load_ctx(ds)


def task(a):
    import imag1_lib as IL
    import imag1_analyse as AN
    s, role = a
    t0 = time.time()
    res = IL.run_draw(_C, s, role)
    if s == 0:
        return s, role, res, time.time() - t0
    return s, role, AN.compress(_C, res), time.time() - t0


def run_dataset(ds, out):
    import multiprocessing as mp
    import imag1_lib as IL
    import imag1_analyse as AN
    t0 = time.time()
    ctx = load_ctx(ds)
    dj = f"draws_{ds}.jsonl"
    have = {}
    if os.path.exists(dj):
        for ln in open(dj, encoding="utf-8"):
            d = json.loads(ln)
            have[(d["s"], d["role"])] = d
    realp = f"{SCR}/imag1_real_{ds}.pkl"
    real = pickle.load(open(realp, "rb")) if os.path.exists(realp) else None
    todo = ([] if real is not None else [(0, "real")]) + [(s, "ref") for s in IL.REF_SEEDS if (s, "ref") not in have] + [(s, "cal") for s in IL.CAL_SEEDS if (s, "cal") not in have]
    print(f"[{ds}] {len(todo)} tasks, pool {NPROC}", flush=True)
    times = []
    if todo:
        with mp.Pool(NPROC, initializer=init_worker, initargs=(ds,)) as pool:
            for s, role, r, dt in pool.imap_unordered(task, todo):
                times.append(dt)
                if s == 0:
                    real = r
                    pickle.dump(r, open(realp, "wb"), protocol=4)
                else:
                    have[(s, role)] = r
                    with open(dj, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(jsonable(r)) + "\n")
                print(f"[{ds}] done s={s} {role} {dt:.1f}s ({len(times)}/{len(todo)}) elapsed {time.time() - t0:.0f}s", flush=True)
    refs = [have[(s, "ref")] for s in IL.REF_SEEDS]
    cals = [have[(s, "cal")] for s in IL.CAL_SEEDS]
    # draws re-read from json carry None for NaN: restore
    fix = lambda d: json.loads(json.dumps(d), parse_constant=float) if False else d
    for dset in (refs, cals):
        for d in dset:
            for kk in ("ia1", "tsi", "or", "ia0"):
                d[kk] = {k: (float("nan") if v is None else v) for k, v in d[kk].items()}
            for kk in ("ia1_row", "or_row", "ia0_row"):
                d[kk] = {k: {a: (float("nan") if b is None else b) for a, b in v.items()} for k, v in d.get(kk, {}).items()}
    R = AN.analyse(ctx, real, refs, cals)
    R["wall_s"] = round(time.time() - t0, 1)
    R["draw_time_s"] = {"mean": float(np.mean(times)) if times else None, "n": len(times), "pool": NPROC}
    out["datasets"][ds] = jsonable(R)
    json.dump(out, open("RESULT_IMAG1.json", "w", encoding="utf-8"))
    print(f"[{ds}] analysed, wall {R['wall_s']} s", flush=True)


def main(dss):
    impl = check_inputs()
    out = {"card": CARD, "card_prefix_sha256": CARD_SHA, "impl_sha256": impl, "tier": "T0", "datasets": {}}
    if os.path.exists("RESULT_IMAG1.json"):
        try:
            out["datasets"] = json.load(open("RESULT_IMAG1.json", encoding="utf-8")).get("datasets", {})
        except Exception:
            pass
    for ds in dss:
        run_dataset(ds, out)


if __name__ == "__main__":
    rc = 0
    try:
        main(sys.argv[1:] or ["eval141", "train"])
    except BaseException:
        traceback.print_exc()
        rc = 1
    open("IMAG1_RUN.done", "w").write(f"rc={rc}\n")
    sys.exit(rc)
