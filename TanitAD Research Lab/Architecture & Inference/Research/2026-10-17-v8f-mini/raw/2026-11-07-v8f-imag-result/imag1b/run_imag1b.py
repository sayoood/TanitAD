"""PREREG_V8F_IMAG1B driver. Card prefix 4,634 bytes sha256 40500d68...; IMPL_CHOICES_IMAG1B.md sha256 in IMPL_CHOICES_IMAG1B.sha256 and every reused cache's sha256 (from the IMPL table) are asserted
before anything is computed. Per dataset (eval141 = deciding, then train = beside): the real stage (s = 0) and 199 refit reference + 20 calibration draws (labels deranged across episodes inside the fit
fold, probes refitted, the whole statistic recomputed) in a pool of at most 3 single-thread worker processes, then the analysis. Draws are appended to draws_imag1b_<ds>.jsonl (resume-safe).
RESULT_IMAG1B.json is written the moment eval141's analysis ends, BEFORE train starts; train that cannot run is recorded NOT RUN with the reason. Writes RUN_STATUS_IMAG1B.txt (the job writes
IMAG1B_RUN.done after the report). Usage (cwd = this directory; env_imag1b.cmd): python -u run_imag1b.py eval141 train"""
import hashlib, json, math, os, pickle, re, sys, time, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import numpy as np

CARD, CARD_N, CARD_SHA = "../PREREG_V8F_IMAG1B.md", 4634, "40500d68d344e56d5321d4099ea67db8da92a722a62417e1ac179fb2bc68d593"
IMPL = "IMPL_CHOICES_IMAG1B.md"
NPROC = max(1, min(3, int(os.environ.get("IMAG1B_NPROC", "3"))))          # at most 3 (Master Mind, 2026-11-07)
SCR = "F:/TanitAD-scratch"
MIN_TRAIN_GB = 3.0
INPUT_FILES = {                                                           # name in the IMPL table -> path
    "eval141": {"imag1_ctx_eval141.pkl": f"{SCR}/imag1_ctx_eval141.pkl", "imag1/labels_imag1_eval141.npz": "../imag1/labels_imag1_eval141.npz", "wmact/latents.pt": "../wmact/latents.pt",
                "imag1/latents_rand.pt": "../imag1/latents_rand.pt"},
    "train": {"imag1_ctx_train.pkl": f"{SCR}/imag1_ctx_train.pkl", "imag1/labels_imag1_train.npz": "../imag1/labels_imag1_train.npz", "wmact/latents_train.pt": "../wmact/latents_train.pt"},
}
# smoke hooks (never set in a real run; env_imag1b.cmd does not set them): a suffix for every output name and smaller reference / calibration counts
SUF = os.environ.get("IMAG1B_SUFFIX", "")
NREF, NCAL = int(os.environ.get("IMAG1B_NREF", "199")), int(os.environ.get("IMAG1B_NCAL", "20"))
_C = None


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check_card_and_impl():
    assert hashlib.sha256(open(CARD, "rb").read()[:CARD_N]).hexdigest() == CARD_SHA, "the card prefix changed"
    want = open("IMPL_CHOICES_IMAG1B.sha256", encoding="utf-8").read().split()[0]
    got = hashlib.sha256(open(IMPL, "rb").read()).hexdigest()
    assert got == want, f"{IMPL} changed after its sha256 was recorded ({got[:12]} vs {want[:12]})"
    return want


def check_inputs(ds):
    """Every reused cache of this dataset re-hashed against the sha256 written in the IMPL table."""
    txt = open(IMPL, encoding="utf-8").read()
    for name, path in INPUT_FILES[ds].items():
        m = re.search(r"`" + re.escape(name) + r"`[^\n]*?`([0-9a-f]{64})`", txt)
        assert m, f"no sha256 for {name} in {IMPL}"
        got = sha_file(path)
        assert got == m.group(1), f"{name} changed: {got[:12]} vs {m.group(1)[:12]}"
    print(f"[{ds}] reused inputs verified: {', '.join(INPUT_FILES[ds])}", flush=True)


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


def restore(o):
    """Draws re-read from json carry None for NaN: restore."""
    if isinstance(o, dict):
        return {k: restore(v) for k, v in o.items()}
    if isinstance(o, list):
        return [restore(v) for v in o]
    return float("nan") if o is None else o


def load_ctx(ds):
    from types import SimpleNamespace
    import imag1b_lib as IL
    with open(f"{SCR}/imag1_ctx_{ds}.pkl", "rb") as fh:
        c = SimpleNamespace(**pickle.load(fh))
    with open(f"{SCR}/imag1b_off_{ds}.pkl", "rb") as fh:
        off = pickle.load(fh)
    return IL.attach_imag1b(c, off)


def init_worker(ds):
    global _C
    import torch
    torch.set_num_threads(1)
    _C = load_ctx(ds)


def task(a):
    import imag1b_lib as IL
    import imag1b_analyse as AN
    s, role = a
    t0 = time.time()
    res = IL.run_draw(_C, s, role)
    if s == 0:
        return s, role, res, time.time() - t0
    return s, role, AN.compress(_C, res), time.time() - t0


def run_dataset(ds, out):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import imag1b_lib as IL
    import imag1b_analyse as AN
    t0 = time.time()
    check_inputs(ds)
    ctx = load_ctx(ds)
    dj = f"draws_imag1b_{ds}{SUF}.jsonl"
    ref_seeds, cal_seeds = IL.REF_SEEDS[:NREF], IL.CAL_SEEDS[:NCAL]
    have = {}
    if os.path.exists(dj):
        for ln in open(dj, encoding="utf-8"):
            d = json.loads(ln)
            have[(d["s"], d["role"])] = d
    realp = f"{SCR}/imag1b_real_{ds}{SUF}.pkl"
    real = pickle.load(open(realp, "rb")) if os.path.exists(realp) else None
    todo = ([] if real is not None else [(0, "real")]) + [(s, "ref") for s in ref_seeds if (s, "ref") not in have] + [(s, "cal") for s in cal_seeds if (s, "cal") not in have]
    print(f"[{ds}] {len(todo)} tasks, pool {NPROC}", flush=True)
    times = []
    if todo:
        ex = ProcessPoolExecutor(max_workers=NPROC, initializer=init_worker, initargs=(ds,))
        try:
            futs = [ex.submit(task, a) for a in todo]
            for fu in as_completed(futs):
                s, role, r, dt = fu.result()
                times.append(dt)
                if s == 0:
                    real = r
                    pickle.dump(r, open(realp, "wb"), protocol=4)
                else:
                    have[(s, role)] = r
                    with open(dj, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(jsonable(r)) + "\n")
                print(f"[{ds}] done s={s} {role} {dt:.1f}s ({len(times)}/{len(todo)}) elapsed {time.time() - t0:.0f}s", flush=True)
        except BaseException:
            for fu in futs:
                fu.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    refs = [restore(have[(s, "ref")]) for s in ref_seeds]
    cals = [restore(have[(s, "cal")]) for s in cal_seeds]
    R = AN.analyse(ctx, real, refs, cals)
    R["status"] = "RUN"
    R["wall_s"] = round(time.time() - t0, 1)
    R["draw_time_s"] = {"mean": float(np.mean(times)) if times else None, "n": len(times), "pool": NPROC}
    out["datasets"][ds] = jsonable(R)
    json.dump(out, open(f"RESULT_IMAG1B{SUF}.json", "w", encoding="utf-8"))
    print(f"[{ds}] analysed, wall {R['wall_s']} s; RESULT_IMAG1B.json written", flush=True)


def main(dss):
    from gate_mem_imag1b import free_gb
    impl = check_card_and_impl()
    out = {"card": CARD, "card_prefix_sha256": CARD_SHA, "impl_sha256": impl, "tier": "T0", "datasets": {}}
    status = []
    rc = 0
    for ds in dss:
        if ds == "eval141":
            try:
                run_dataset(ds, out)
                status.append("eval141=OK")
            except BaseException:
                traceback.print_exc()
                status.append("eval141=FAILED")
                rc = 1
                break
        else:
            try:
                fg = free_gb()
                if fg < MIN_TRAIN_GB:
                    raise MemoryError(f"free physical memory {fg:.2f} GB below the {MIN_TRAIN_GB} GB needed to start the beside pass")
                run_dataset(ds, out)
                status.append(f"{ds}=OK")
            except BaseException as e:
                traceback.print_exc()
                reason = f"{type(e).__name__}: {str(e)[:300]}"
                out["datasets"][ds] = {"status": "NOT RUN", "reason": reason}
                json.dump(out, open(f"RESULT_IMAG1B{SUF}.json", "w", encoding="utf-8"))
                status.append(f"{ds}=NOT RUN: {reason}")
    open(f"RUN_STATUS_IMAG1B{SUF}.txt", "w").write("\n".join(status) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["eval141", "train"]))
