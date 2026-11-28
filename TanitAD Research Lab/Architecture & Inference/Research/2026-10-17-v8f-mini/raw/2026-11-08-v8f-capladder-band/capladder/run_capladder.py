"""CAPLADDER statistics stage (IMPL_CHOICES sections 5-7). Main process: verifies the card prefix, IMPL_CHOICES and every reused cache by sha256, builds the Gram of every rung (GPU if no other compute job
holds it, else CPU), writes it to F:/TanitAD-scratch/capladder_K/, and hands it to 3 single-thread worker processes that do the eigendecompositions, the 199 reference + 20 calibration draws and the beside
targets and write rungs/<rung>_A.npz (and _D.npz). Resume-safe: a rung whose npz exists is skipped. Usage: python run_capladder.py   (WMACT_CODE_ROOT not needed)."""
import concurrent.futures as cf
import hashlib, json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "imag1b"))
import numpy as np
import capladder_lib as CL

FEAT = os.environ.get("CAPLADDER_FEAT", "F:/TanitAD-scratch/capladder_feat")
KDIR = os.environ.get("CAPLADDER_KDIR", "F:/TanitAD-scratch/capladder_K")
RDIR = os.environ.get("CAPLADDER_RDIR", os.path.join(HERE, "rungs"))
ONLY = [x for x in os.environ.get("CAPLADDER_ONLY", "").split(",") if x]          # debug / resume: restrict the rung table
CARD = os.path.join(HERE, "..", "PREREG_V8F_CAPLADDER.md")
CARD_SHA, CARD_BYTES = "e17a4d906751754deed04b5d237c5937926def5839a989863ff15e8b79f5bd41", 7830
CTX = "F:/TanitAD-scratch/imag1_ctx_eval141.pkl"
CTX_SHA = "46ad48c47402138dfe93fdcc6ef165aa502c8840103e32f6abfc43fc6821b7dd"
NPROC = int(os.environ.get("CAPLADDER_NPROC", "3"))
CHUNK = 16384
DIMS_G = (655360, 163840, 40960, 3072, 1024, 384, 128)


def sha(p, nbytes=None):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        if nbytes is not None:
            h.update(fh.read(nbytes))
        else:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
    return h.hexdigest()


def free_mb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 20


def wait_mem(need_mb, what):
    t0 = time.time()
    while free_mb() < need_mb:
        if time.time() - t0 > 8 * 3600:
            raise RuntimeError(f"memory gate ceiling before {what}")
        time.sleep(30)
    return free_mb()


def verify():
    assert sha(CARD, CARD_BYTES) == CARD_SHA, "the card's frozen prefix changed"
    rec = open(os.path.join(HERE, "IMPL_CHOICES_CAPLADDER.sha256")).read().split()[0]
    assert sha(os.path.join(HERE, "IMPL_CHOICES_CAPLADDER.md")) == rec, "IMPL_CHOICES_CAPLADDER.md changed after it was hashed"
    assert sha(CTX) == CTX_SHA, "imag1_ctx_eval141.pkl differs from the hash recorded by IMPL_CHOICES_IMAG1B"
    return {"card_prefix_sha256": CARD_SHA, "card_prefix_bytes": CARD_BYTES, "impl_choices_sha256": rec, "ctx_sha256": CTX_SHA}


# ------------------------------------------------------------------------------------------------ the rungs
def rung_table(have_r0):
    """name -> dict(dim, beside: bool [beside targets B, C, D and the MLP computed]); order = Gram order (big memmap rungs first)."""
    T = {}
    if have_r0:
        T["R0"] = dict(dim=640 * 1024, beside=True, src="r0")
    T["G_655360"] = dict(dim=655360, beside=False, src="gauss")
    T["R1"] = dict(dim=160 * 1024, beside=True, src="r1")
    T["G_163840"] = dict(dim=163840, beside=False, src="gauss")
    T["R2"] = dict(dim=40 * 1024, beside=True, src="r2")
    T["G_40960"] = dict(dim=40960, beside=False, src="gauss")
    T["R2m"] = dict(dim=1024, beside=True, src="r2m")
    for tag, sn in (("M2b_s0", "s0"), ("M2b_s1", "s1")):
        T[f"R3_{sn}"] = dict(dim=3072, beside=True, src=("m", tag, "stack3"))
        T[f"R4_{sn}"] = dict(dim=384, beside=True, src=("m", tag, "stack4"))
    for tag, sn in (("M2b_s0", "s0"), ("M2b_s1", "s1")):
        for k, nm in (("s", "R3s"), ("t", "R3t"), ("o", "R3o")):
            T[f"{nm}_{sn}"] = dict(dim=1024, beside=False, src=("m", tag, k))
        for l in (1, 2, 3):
            T[f"L{l}_{sn}"] = dict(dim=128, beside=False, src=("m", tag, f"l{l}"))
    for r in (100, 101):
        T[f"R3_r{r}"] = dict(dim=3072, beside=False, src=("m", f"rand{r}", "stack3"))
        T[f"R4_r{r}"] = dict(dim=384, beside=False, src=("m", f"rand{r}", "stack4"))
    for d in (3072, 1024, 384, 128):
        T[f"G_{d}"] = dict(dim=d, beside=False, src="gauss")
    return T


def chunks_for(spec, S):
    nD, nA = int(S["nD"]), int(S["nA"])
    src = spec["src"]
    if src == "gauss":
        import torch
        d = spec["dim"]
        for i, c0 in enumerate(range(0, d, CHUNK)):
            g = torch.Generator().manual_seed(1_000_003 * 0 + i)                         # seed 0, per-chunk generator
            yield torch.randn(nA, min(CHUNK, d - c0), generator=g)
        return
    if src in ("r0", "r1", "r2"):
        d = spec["dim"]
        mm = np.memmap(f"{FEAT}/{src}_fp16.bin", dtype=np.float16, mode="r", shape=(nD, d))
        for c0 in range(0, d, CHUNK):
            yield np.asarray(mm[:, c0:c0 + CHUNK], dtype=np.float32)
        del mm
        return
    if src == "r2m":
        yield np.load(f"{FEAT}/r2m_fp16.npy").astype(np.float32)
        return
    _, tag, k = src
    a = np.load(f"{FEAT}/r34_{tag}.npz")
    if k == "stack3":
        X = np.concatenate([a["s"], a["t"], a["o"]], 1)
    elif k == "stack4":
        X = np.concatenate([a["l1"], a["l2"], a["l3"]], 1)
    else:
        X = a[k]
    yield X.astype(np.float32)


# ------------------------------------------------------------------------------------------------ worker tasks
def _load_sets():
    S = np.load(os.path.join(HERE, "sets_capladder.npz"), allow_pickle=False)
    nA = int(S["nA"])
    d = dict(nA=nA, nD=int(S["nD"]), uniq=S["uniq"], fold=S["foldD"], inner=S["innerD"], pos=S["posD"], ep=S["epD"], gap=S["gap"], loggap=S["loggap"],
             clos=S["clos"], ttc=S["ttc"], idxB=S["idxB"], idxC=S["idxC"], present=S["present"])
    return d


def work_A(name, do_beside):
    """lead_gap on set A: real + 199 reference + 20 calibration columns + log(gap) real; beside targets on B and C and the PCA-256 MLP when do_beside."""
    t0 = time.time()
    os.makedirs(RDIR, exist_ok=True)
    out = os.path.join(RDIR, f"{name}_A.npz")
    if os.path.exists(out):
        return name, "A", 0.0
    s = _load_sets()
    nA, uniq = s["nA"], s["uniq"]
    n_ep = len(uniq)
    fold, inner, pos, ep = s["fold"][:nA], s["inner"][:nA], s["pos"][:nA], s["ep"][:nA]
    KA = np.load(os.path.join(KDIR, f"{name}_A.npy"))
    y, ly = s["gap"].astype(float), s["loggap"].astype(float)
    seeds = CL.REF_SEEDS + CL.CAL_SEEDS
    perm = CL.build_perm_fit(y, ep, fold, seeds)
    rows = np.arange(nA)
    Yfit = [np.concatenate([y[fold != k][:, None], perm[k], ly[fold != k][:, None]], 1) for k in range(CL.NF)]
    Yt = np.concatenate([np.repeat(y[:, None], 1 + len(seeds), 1), ly[:, None]], 1)
    res = CL.run_set(KA, rows, Yt, Yfit, fold, inner, pos, n_ep, keep_fold_splits=do_beside)
    pack = dict(sse_e=res["sse_e"], d_e=res["d_e"], pred0=res["pred"][:, 0].astype(np.float32), cstar=res["cstar"][:, :1 + 5])
    if do_beside:
        for tag, idx, cols in (("B", s["idxB"], np.stack([s["clos"][s["idxB"]], s["ttc"][s["idxB"]]], 1)), ("C", s["idxC"], y[s["idxC"]][:, None])):
            Yc = cols.astype(float)
            fo = fold[idx]
            r = CL.run_set(KA, idx, Yc, [Yc[fo != k] for k in range(CL.NF)], fo, inner[idx], pos[idx], n_ep)
            pack[f"{tag}_sse_e"], pack[f"{tag}_d_e"], pack[f"{tag}_pred"] = r["sse_e"], r["d_e"], r["pred"].astype(np.float32)
        # beside MLP (IMAG1B bank) on a 256-component PCA of the rung, fitted on the fit fold (components of the same eigendecomposition)
        import imag1b_probes as PB
        pred = np.empty(nA)
        den = np.empty(nA)
        for k, (F, S_, sp) in res["splits"].items():
            top = np.argsort(sp.s)[::-1][:256]
            sc = np.sqrt(np.maximum(sp.s[top], 1e-12))
            Xf, Xs = sp.U[:, top] * sc[None, :], sp.V[:, top] / sc[None, :]
            sd = np.sqrt((Xf ** 2).sum(0).mean() / len(F))
            bank = PB.MLPBank(["lead_gap"], {"lead_gap": "cont"}, Xf.shape[1], seed=1000 + k).fit(Xf / sd, {"lead_gap": y[F]}, inner[F] == 0)
            pred[S_] = bank.predict(Xs / sd)["lead_gap"]
            den[S_] = y[F].mean()
        sse = np.zeros(n_ep)
        dd = np.zeros(n_ep)
        np.add.at(sse, pos, (y - pred) ** 2)
        np.add.at(dd, pos, (y - den) ** 2)
        pack["mlp_sse_e"], pack["mlp_d_e"], pack["mlp_pred"] = sse[None, :], dd[None, :], pred.astype(np.float32)
    np.savez(out, **pack)
    os.remove(os.path.join(KDIR, f"{name}_A.npy"))
    return name, "A", time.time() - t0


def work_D(name):
    """lead_present on the balanced set D: regression on {0, 1}, same kernel ridge, class = prediction >= 0.5 (scored in the analysis)."""
    t0 = time.time()
    out = os.path.join(RDIR, f"{name}_D.npz")
    if os.path.exists(out):
        return name, "D", 0.0
    s = _load_sets()
    n_ep = len(s["uniq"])
    KD = np.load(os.path.join(KDIR, f"{name}_D.npy"))
    y = s["present"].astype(float)
    fold, inner, pos = s["fold"], s["inner"], s["pos"]
    res = CL.run_set(KD, np.arange(s["nD"]), y[:, None], [y[fold != k][:, None] for k in range(CL.NF)], fold, inner, pos, n_ep)
    np.savez(out, pred=res["pred"][:, 0].astype(np.float32), cstar=res["cstar"][:, 0])
    os.remove(os.path.join(KDIR, f"{name}_D.npy"))
    return name, "D", time.time() - t0


def _init_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    ver = verify()
    print("[run] verified", json.dumps(ver), flush=True)
    os.makedirs(KDIR, exist_ok=True)
    os.makedirs(RDIR, exist_ok=True)
    S = np.load(os.path.join(HERE, "sets_capladder.npz"), allow_pickle=False)
    s = _load_sets()
    nA, nD = s["nA"], s["nD"]
    st = {}
    if os.path.exists(f"{FEAT}/R0_STATUS.json"):
        st = json.load(open(f"{FEAT}/R0_STATUS.json"))
    have_r0 = st.get("status") == "OK" and (os.path.exists(f"{FEAT}/r0_fp16.bin") or os.path.exists(os.path.join(RDIR, "R0_A.npz")))  # the bin is deleted once R0's Grams are written: a resume keeps the finished rung
    print("[run] R0 status:", st.get("status"), st.get("reason", ""), "-> R0 rung", "computed" if have_r0 else "NOT RUN", flush=True)
    table = rung_table(have_r0)
    if ONLY:
        table = {k: v for k, v in table.items() if k in ONLY}
        print("[run] CAPLADDER_ONLY:", list(table), flush=True)
    import torch
    torch.set_num_threads(3)
    from r0_capladder import gpu_free_rule
    ok, det = gpu_free_rule()
    device = "cuda" if (ok and torch.cuda.is_available()) else "cpu"
    print(f"[run] Gram device {device}; gpu rule {ok}: {json.dumps({k: v for k, v in det.items() if k != 'apps'})}", flush=True)
    pool = cf.ProcessPoolExecutor(max_workers=NPROC, initializer=_init_worker)
    futs = []
    big_done = {"R0": "r0", "R1": "r1", "R2": "r2"}
    log = open(os.path.join(HERE, "DELETIONS_CAPLADDER.log"), "a", encoding="utf-8")
    for name, spec in table.items():
        need_A = not os.path.exists(os.path.join(RDIR, f"{name}_A.npz"))
        need_D = spec["beside"] and not os.path.exists(os.path.join(RDIR, f"{name}_D.npz"))
        if need_A or need_D:
            while sum(1 for f in futs if not f.done()) >= NPROC + 2:                    # at most NPROC+2 tasks outstanding (their K files wait on disk)
                time.sleep(5)
            for f in futs:
                if f.done() and f.exception() is not None:
                    raise f.exception()
            kpath = lambda k: os.path.join(KDIR, f"{name}_{k}.npy")
            sets = {}
            if need_A and not os.path.exists(kpath("A")):                                # a K file already on disk (a worker died after the Gram) is reused
                sets["A"] = (np.arange(nA), s["fold"][:nA])
            if need_D and not os.path.exists(kpath("D")):
                sets["D"] = (np.arange(nD), s["fold"])
            if sets:
                mb = wait_mem(3000, f"Gram of {name}")
                tg = time.time()
                Ks = CL.gram_folds(chunks_for(spec, S), sets, device)
                for k, K in Ks.items():
                    np.save(kpath(k) + ".tmp.npy", K)
                    os.replace(kpath(k) + ".tmp.npy", kpath(k))                           # atomic: a half-written K is never read
                print(f"[run] Gram {name} d={spec['dim']} sets={list(Ks)} {time.time() - tg:.0f}s (free RAM before {mb:.0f} MB) t={time.time() - t0:.0f}s", flush=True)
                del Ks
            if need_A:
                futs.append(pool.submit(work_A, name, spec["beside"]))
            if need_D:
                futs.append(pool.submit(work_D, name))
        if name in big_done and (spec["beside"] is True):
            p = f"{FEAT}/{big_done[name]}_fp16.bin"
            if os.path.exists(p) and all(os.path.exists(os.path.join(KDIR, f"{name}_{k}.npy")) or os.path.exists(os.path.join(RDIR, f"{name}_{k}.npz")) for k in "AD"):
                n = os.path.getsize(p)
                os.remove(p)
                log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} deleted {p} {n} bytes (Grams of {name} written; re-creatable from the cache / box 2)\n")
                log.flush()
    for f in futs:
        nm, k, secs = f.result()
        print(f"[run] task {nm}_{k} done {secs:.0f}s", flush=True)
    pool.shutdown()
    json.dump({"verified": ver, "wall_s": round(time.time() - t0, 1), "device": device, "rungs": list(table), "r0": st.get("status", "NOT RUN"), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(HERE, "RUN_STATUS_CAPLADDER.json"), "w"), indent=1)
    print(f"[run] all rungs done {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
