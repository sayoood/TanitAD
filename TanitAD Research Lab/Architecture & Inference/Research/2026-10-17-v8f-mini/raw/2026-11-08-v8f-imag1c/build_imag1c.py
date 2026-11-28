"""IMAG1C model side (IMPL_CHOICES_IMAG1C D1, section 3): every 16-step window of the bank (7,481), rolled out by M2b s0 / s1 (CPU, no training, no probe).
Writes F:/TanitAD-scratch/imag1c_ctx.pkl and build_imag1c.report.json. Checks (i) and (ii) of section 3 are asserted here. Usage: python build_imag1c.py (WMACT_CODE_ROOT set)."""
import hashlib, json, os, pickle, sys, time
sys.dont_write_bytecode = True
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
TH = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(TH, "wmact"))
import wmact_lib as W  # noqa: E402

CTX = "F:/TanitAD-scratch/imag1_ctx_eval141.pkl"
CTX_SHA = "46ad48c47402138dfe93fdcc6ef165aa502c8840103e32f6abfc43fc6821b7dd"
LAB = os.path.join(TH, "imag1", "labels_imag1_eval141.npz")
LAB_SHA = "bd0a265bde15b2ac9f5de680dd6d5c1e9249ae0d00a42ffebdbf0d00199c93ef"
LAT = os.path.join(TH, "wmact", "latents.pt")
LAT_SHA = "7077ec717f633e0ef0814fec0a5fb43609e714bd8f521ccf6eb5e64ad02333a4"
OUT = "F:/TanitAD-scratch/imag1c_ctx.pkl"
TAGS, LEVELS, KS = ("M2b_s0", "M2b_s1"), (1, 2, 3), (1, 2, 3)
TARGETS = ("speed", "accel", "lead_gap", "lead_closing", "ttc", "yaw_rate", "curvature", "lead_present", "lon7", "agents30")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def pack(wm, Z1, A_sc, chunk=512):
    """copied from ../imag1/build_ctx_imag1.py: per level Z0 [n, D], ORA[(l, k)], PRED[(l, k)] (HJEPAStack.encode / rollout, context 1)."""
    st = wm.stack
    z0, ora, pred = {}, {}, {}
    for s0 in range(0, Z1.shape[0], chunk):
        sl = slice(s0, s0 + chunk)
        enc = st.encode(Z1[sl], A_sc[sl])
        for l in LEVELS:
            acts = A_sc[sl][:, :3] if l == 1 else enc.act[l - 1][:, :3]
            roll = st.rollout(l, enc.z[l - 1][:, 0], acts)
            z0.setdefault(l, []).append(enc.z[l - 1][:, 0])
            for k in KS:
                ora.setdefault((l, k), []).append(enc.z[l - 1][:, k])
                pred.setdefault((l, k), []).append(roll[:, k - 1])
    cat = lambda d: {k: torch.cat(v).float().numpy() for k, v in d.items()}
    return cat(z0), cat(ora), cat(pred)


def main():
    t0 = time.time()
    assert sha(CTX) == CTX_SHA and sha(LAB) == LAB_SHA and sha(LAT) == LAT_SHA, "input hash differs"
    ctx = pickle.load(open(CTX, "rb"))
    lab = np.load(LAB)
    beid, bg = ctx["beid"].astype("U36"), np.asarray(lab["bg"]).astype(int)
    assert (lab["beid"].astype("U36") == beid).all()
    nb = len(beid)
    key2b = {(e, int(g)): i for i, (e, g) in enumerate(zip(beid, bg))}
    L = torch.load(LAT, weights_only=False)
    eid, tw, ok = np.array(L["eid"]), np.asarray(L["t"]), L["ok"].numpy()
    # first occurrence of every bank frame in the latents' windows (the bank rule: build_bank_train)
    first = {}
    for i in range(len(eid)):
        for f in range(16):
            g = int(tw[i] - 15 + f)
            if ok[i, f] and g >= 0:
                first.setdefault((str(eid[i]), g), (i, f))
    assert len(first) == nb and set(first) == set(key2b), "bank mismatch"
    A_bank = np.zeros((nb, 2), np.float32)
    Z_bank = {t: np.zeros((nb, 128), np.float32) for t in TAGS}
    for kk, (i, f) in first.items():
        b = key2b[kk]
        A_bank[b] = L["A_raw"][i, f].numpy()
        for t in TAGS:
            Z_bank[t][b] = L["Z1"][t][i, f].numpy()
    # the windows: every start g0 with 16 consecutive bank frames
    uniq = np.unique(beid)
    qe, g0s, bidx = [], [], []
    for e in uniq:
        gs = set(int(g) for g in bg[beid == e])
        for g0 in range(0, max(gs) + 1):
            if all((g0 + f) in gs for f in range(16)):
                qe.append(e); g0s.append(g0); bidx.append([key2b[(e, g0 + f)] for f in range(16)])
    qe, g0s, bidx = np.array(qe), np.array(g0s), np.array(bidx)
    nw = len(qe)
    # old windows (IMAG1's 1,976 = all-ok windows of the latents file) as (episode, g0)
    sel = np.flatnonzero(ok.all(1))
    old_keys = [(str(eid[i]), int(tw[i] - 15)) for i in sel]
    pos_new = {(e, int(g)): j for j, (e, g) in enumerate(zip(qe, g0s))}
    old_idx = np.array([pos_new[k] for k in old_keys])
    old = np.zeros(nw, bool); old[old_idx] = True
    Y = {q: np.asarray(lab[q], np.float64) for q in TARGETS}
    Yw = {q: Y[q][bidx] for q in TARGETS}
    # check (ii): horizon labels of the old windows equal the ctx
    for q in TARGETS:
        a, b = Yw[q][old_idx], ctx["Yw"][q]
        assert np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9)), f"Yw differs for {q}"
    print("check ii (labels of the old windows equal the ctx): OK", flush=True)
    out = dict(beid=beid, bg=bg, Y=Y, qe=qe, g0=g0s, bidx=bidx, Yw=Yw, old=old, old_idx=old_idx, BZ=ctx["BZ"], tags=list(TAGS), Z0={}, PRED={}, ORA={})
    rep = {"n_bank": int(nb), "n_episodes": int(len(uniq)), "n_windows": int(nw), "n_old": int(old.sum()), "windows_per_episode_min_max": [int(np.bincount(np.searchsorted(uniq, qe)).min()), int(np.bincount(np.searchsorted(uniq, qe)).max())]}
    mx = {}
    for tag in TAGS:
        m, cfg, ck = W.load_model(W.CKPT[tag])
        wm = m.wm
        Z1w = torch.from_numpy(Z_bank[tag][bidx])
        Aw = torch.from_numpy(A_bank[bidx])
        A_sc = wm.scale_actions(Aw)[:, :-1]
        z0, ora, pred = pack(wm, Z1w, A_sc)
        out["Z0"][tag], out["ORA"][tag], out["PRED"][tag] = z0, ora, pred
        d = 0.0
        for l in LEVELS:
            d = max(d, float(np.abs(z0[l][old_idx] - ctx["Z0"][tag][l]).max()))
            for k in KS:
                d = max(d, float(np.abs(ora[(l, k)][old_idx] - ctx["ORA"][tag][(l, k)]).max()), float(np.abs(pred[(l, k)][old_idx] - ctx["PRED"][tag][(l, k)]).max()))
        mx[tag] = d
        # the bank latents of the ctx against the assembled Z1 (a consistency check of the assembly)
        print("tag", tag, "max abs diff of Z0/ORA/PRED on the old windows vs ctx:", d, round(time.time() - t0, 1), "s", flush=True)
    rep["check_i_max_abs_diff_old_windows_vs_ctx"] = mx
    rep["check_i_pass"] = bool(max(mx.values()) <= 1e-4)
    rep["inputs_sha256"] = {"ctx": CTX_SHA, "labels": LAB_SHA, "latents": LAT_SHA, "hacc_s0": sha(W.CKPT["M2b_s0"]), "hacc_s1": sha(W.CKPT["M2b_s1"])}
    rep["tanitad_file"] = W.tanitad.__file__
    rep["wall_s"] = round(time.time() - t0, 1)
    assert rep["check_i_pass"], rep
    with open(OUT, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    rep["ctx_bytes"] = os.path.getsize(OUT)
    json.dump(rep, open(os.path.join(HERE, "build_imag1c.report.json"), "w"), indent=1)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
