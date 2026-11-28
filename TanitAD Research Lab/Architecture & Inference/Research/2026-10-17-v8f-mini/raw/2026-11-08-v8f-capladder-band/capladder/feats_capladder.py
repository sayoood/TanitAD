"""CAPLADDER features on the rows of set D (IMPL_CHOICES sections 2 and 3). CPU.
R1 / R2 / R2m from the cache clips (fp8 widened to fp16, exact); R3 = read_s | read_t | read_o outputs and R4 = L1 | L2 | L3 of M2b s0 / s1 (R4 from the verified IMAG1 cache BZ,
recomputed here as a check) and of two re-initialised networks (seeds 100, 101). Needs WMACT_CODE_ROOT. Writes under F:/TanitAD-scratch/capladder_feat/."""
import hashlib, json, os, pickle, sys, time
sys.dont_write_bytecode = True
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "wmact"))
import wmact_lib as W  # noqa: E402

M, D = W.M, W.D
FEAT = os.environ.get("CAPLADDER_FEAT", "F:/TanitAD-scratch/capladder_feat")
CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips"
CTX = "F:/TanitAD-scratch/imag1_ctx_eval141.pkl"
CTX_SHA = "46ad48c47402138dfe93fdcc6ef165aa502c8840103e32f6abfc43fc6821b7dd"
TAGS = ("M2b_s0", "M2b_s1")
MAXR = int(os.environ.get("CAPLADDER_MAXROWS", "0"))      # smoke: only the first MAXR rows (in clip order); writes nothing but a report


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cache_rungs(S):
    """R1, R2 (fp16 memmaps), R2m (npy) for the D rows, read clip by clip from the cache."""
    nD = int(S["nD"])
    epD, gD = S["epD"], S["gD"]
    r1 = np.memmap(f"{FEAT}/r1_fp16.bin", dtype=np.float16, mode="w+", shape=(nD, 160 * 1024))
    r2 = np.memmap(f"{FEAT}/r2_fp16.bin", dtype=np.float16, mode="w+", shape=(nD, 40 * 1024))
    r2m = np.zeros((nD, 1024), np.float16)
    for clip in sorted(set(epD.tolist())):
        ix = np.flatnonzero(epD == clip)
        o = torch.load(f"{CACHE}/{clip}.pt", map_location="cpu", weights_only=True)
        g = torch.as_tensor(gD[ix])
        r1[ix] = o["pool8x20"][g].to(torch.float16).reshape(len(ix), -1).numpy()
        r2[ix] = o["pool4x10"][g].to(torch.float16).reshape(len(ix), -1).numpy()
        r2m[ix] = o["mean"][g].to(torch.float16).numpy()
    r1.flush(); r2.flush()
    np.save(f"{FEAT}/r2m_fp16.npy", r2m)
    return {"r1_bytes": os.path.getsize(f"{FEAT}/r1_fp16.bin"), "r2_bytes": os.path.getsize(f"{FEAT}/r2_fp16.bin")}


@torch.no_grad()
def model_rungs(S):
    nD = int(S["nD"])
    epD, gD = S["epD"], S["gD"]
    assert sha(CTX) == CTX_SHA, "imag1_ctx_eval141.pkl differs from the hash recorded by IMPL_CHOICES_IMAG1B"
    ctx = pickle.load(open(CTX, "rb"))
    BZ = {t: {l: np.asarray(ctx["BZ"][t][l]) for l in (1, 2, 3)} for t in TAGS}
    del ctx
    ds, wins, eid = W.open_windows(10)
    models = {}
    for t in TAGS:
        models[t] = W.load_model(W.CKPT[t])[0]
    cfg = W.load_model(W.CKPT["M2b_s0"])[1]
    for r in (0, 1):                                                                      # seeds 100, 101 (IMPL_CHOICES section 3)
        torch.manual_seed(100 + r)
        models[f"rand{100 + r}"] = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30))).eval()
    out = {t: {k: np.zeros((nD, 1024 if k in "sto" else 128), np.float32) for k in ("s", "t", "o", "l1", "l2", "l3")} for t in models}
    cl = np.array([ds.clip_index[c] for c in epD])
    order = np.argsort(cl, kind="stable")
    if MAXR:
        order = order[:MAXR]
    t0 = time.time()
    for b0 in range(0, len(order), 64):
        ids = order[b0:b0 + 64]
        batch = D.collate([ds.item(int(cl[i]), int(gD[i]), 0, 0) for i in ids])
        for tag, m in models.items():
            caps = {}
            hs = [getattr(m, f"read_{k}").register_forward_hook(lambda mod, inp, o, k=k: caps.__setitem__(k, o.detach().flatten(1).float().numpy())) for k in "sto"]
            try:
                res = m.forward_infer(batch)
            finally:
                for h in hs:
                    h.remove()
            z1 = m.wm.z1(res["adapt"][:, -1:])[:, 0]                                       # level-1 latent of the frame at NOW (WMAux.z1, v8f_mini.py:666-669)
            zs = m.wm.stack.initial_latents(z1)                                           # hjepa.py:625-635
            for k in "sto":
                out[tag][k][ids] = caps[k]
            for l in (1, 2, 3):
                out[tag][f"l{l}"][ids] = zs[l - 1].float().numpy()
        if (b0 // 64) % 10 == 0:
            print(f"[feats] {b0}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    rep = {}
    for tag in models:
        a = out[tag]
        done_rows = np.sort(order)
        assert all(np.isfinite(v[done_rows]).all() for v in a.values()), tag
        if tag in TAGS:
            diff = {f"l{l}": float(np.abs(a[f"l{l}"][done_rows] - BZ[tag][l][S["rowsD"]][done_rows]).max()) for l in (1, 2, 3)}
            rep[tag] = {"max_abs_diff_recomputed_vs_cache_BZ": diff}
            assert max(diff.values()) <= 1e-4, (tag, diff)
            a = dict(a)
            for l in (1, 2, 3):
                a[f"l{l}"] = BZ[tag][l][S["rowsD"]].astype(np.float32)                    # R4 is the cache's BZ (the card: IMAG1's BZ)
        if not MAXR:
            np.savez(f"{FEAT}/r34_{tag}.npz", **a)
        else:
            rep[tag + "_stats"] = {k: [float(v[done_rows].mean()), float(v[done_rows].std())] for k, v in a.items()}
    return rep


def main():
    os.makedirs(FEAT, exist_ok=True)
    torch.set_num_threads(3)
    S = np.load(os.path.join(HERE, "sets_capladder.npz"), allow_pickle=False)
    assert os.path.abspath(W.tanitad.__file__).replace("\\", "/").lower().startswith(os.path.abspath(W.CODE_ROOT).replace("\\", "/").lower())
    rep = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tanitad_file": W.tanitad.__file__, "ctx_sha256": CTX_SHA}
    if not MAXR and not os.path.exists(f"{FEAT}/r2m_fp16.npy"):
        rep["cache_rungs"] = cache_rungs(S)
    if MAXR or not all(os.path.exists(f"{FEAT}/r34_{t}.npz") for t in (*TAGS, "rand100", "rand101")):
        rep["model_rungs"] = model_rungs(S)
    if not MAXR:
        json.dump(rep, open(f"{FEAT}/feats_report.json", "w"), indent=1)
    print("[feats] done", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
