"""g_lib.py -- shared layer of trigger probe 3 (PREREG_FROZEN.md).  Reads probes 1/2 READ-ONLY; writes only under trigger_probe3/ (small) and F:/TanitAD-scratch/trigger_probe3_scratch/ (big).
EVIDENCE CLASS of everything computed with it: MEASURED (artifacts under raw/).  ESTIMATOR: clip-cluster bootstrap B = 2000 seed 0 on the MEAN OF PER-FOLD AUCs (blind to probe-fit variance and to training variance, H-ESTIM-SEED-1).
CUDA_VISIBLE_DEVICES=-1; BLAS threads 4."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = f"{HERE}/raw"
SCR = "F:/TanitAD-scratch/trigger_probe3_scratch"
OLD = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe"
OLDRAW = f"{OLD}/raw"
OLDSCR = "F:/TanitAD-scratch/aside_from_E/probe_scratch/trigger_probe_scratch"      # probe 1's scratch (moved off E: 2026-10-29)
CACHE_CLIPS = "F:/TanitAD-scratch/speedread/cache/train/clips"
EVAL_CACHE_CLIPS = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips"
MANIFEST_TRAIN = "C:/Users/Admin/box2_ship/logs/v8f_dscale/manifest_train.jsonl"
WINDOWS_SHA = "a9a57b32ffac558929b8f57e49a979826b88d17be17e5e9d7f9347e5875783ec"
sys.path.insert(0, HERE)
import tp_lib as L  # noqa: E402  (byte-identical copy of trigger_probe/tp_lib.py)

L.C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)          # PREREG_FROZEN.md 3: the linear C grid for the 2,5xx-feature arms (tp_lib.fit_probe reads the module global at call time)

POPS = ("pstop", "pmove")
POP_INDEX = {"pstop": 0, "pmove": 1}
N_TOK = {"G8": 160, "G4": 40}
GRID_HW = {"G8": (8, 20), "G4": (4, 10)}
K_TOK = 16                  # PCA dims per token
N_SUB = 16                  # token positions sampled per window for the PCA fit
D_TOK = 1024


# ------------------------------------------------------------------------------------------------ resources
def free_mb() -> int:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def require_ram(min_mb=3000, what="heavy load", max_wait_s=900):
    """RAM free >= 3 GB before every heavy load (PREREG 7).  Polls up to ``max_wait_s``, then exits: a step never starts below the floor."""
    t0 = time.time()
    while True:
        fm = free_mb()
        if fm >= min_mb:
            return fm
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"RAM GUARD: free {fm} MB < {min_mb} MB before {what}: write HANDOFF.md and end")
        time.sleep(15)


def sha256_file(p, blk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(blk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ windows (identical to run_probe.load_pop of probe 1)
def load_windows():
    return np.load(f"{OLDRAW}/windows_train.npz", allow_pickle=False)


def load_pop(pop):
    A = load_windows()
    idx = np.load(f"{OLDRAW}/sample_{pop}.npz")["idx"]
    clip = A["clip_i"][idx].astype(np.int64)
    sha = A["episode_sha12"][clip]
    d = {"idx": idx, "y": A["go"][idx].astype(int), "y24": A["go24"][idx].astype(int), "clip": clip, "sha": sha, "outer": L.outer_fold(sha), "inner": L.inner_fold(sha),
         "E": A["E"][idx].astype(np.float64), "v0": A["v0"][idx], "t": A["t"][idx]}
    for k in ("TL_ANY", "TL_RED", "TL_GREEN", "TL_YELLOW", "NO_TL", "lead_now", "no_lead", "lead_depart"):
        d[k] = A[k][idx]
    return d


def banked(pop, fam):
    """Probe 1's banked out-of-fold scores (read-only) with y / clip for the row-order assert."""
    ov = dict(np.load(f"{OLDRAW}/oof_{pop}_V_{fam}.npz"))
    out = {"E": ov["E"], "V+E": ov["V+E"], "V": ov["V"]}
    if fam == "linear":
        out["C3:leak:E"] = ov["C3__leak__E"]
        out["C1:shuffle:E"] = ov["C1__shuffle__E"]
        out["C2:donor:E"] = ov["C2__donor__E"]
        out["y"], out["clip"] = ov["y"], ov["clip"]
    return out


# ------------------------------------------------------------------------------------------------ no-mmap row reader
class NpyRows:
    """Row-wise reader of a 2-D .npy WITHOUT memory-mapping.  A np.load(mmap_mode='r') of the 3.9 GB G8 file inflates the process working set (the extraction's memmap writes reached 6.3 GB WS and left the box
    with 0.4 GB available); plain buffered reads go through the reclaimable file cache instead."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            ver = np.lib.format.read_magic(fh)
            hdr = np.lib.format.read_array_header_1_0(fh) if ver == (1, 0) else np.lib.format.read_array_header_2_0(fh)
            self.shape, fortran, self.dtype = hdr
            assert not fortran and len(self.shape) == 2, (path, self.shape)
            self.offset = fh.tell()
        self.ncols = int(self.shape[1])

    def rows(self, a, b):
        b = min(b, self.shape[0])
        with open(self.path, "rb") as fh:
            fh.seek(self.offset + a * self.ncols * self.dtype.itemsize)
            return np.fromfile(fh, dtype=self.dtype, count=(b - a) * self.ncols).reshape(b - a, self.ncols)


# ------------------------------------------------------------------------------------------------ the two grids from ONE tensor
def g4_from_g8(x):
    """x [n, 160, C] (any float dtype; row-major 8x20 tokens) -> the exact 2x2 average [n, 40, C] float32, order (((a+b)+c)+d)*0.25 over (2r,2c),(2r,2c+1),(2r+1,2c),(2r+1,2c+1).  PREREG 2."""
    x = np.asarray(x, np.float32)
    n, T, C = x.shape
    assert T == 160, T
    g = x.reshape(n, 8, 20, C)
    a, b, c, d = g[:, 0::2, 0::2], g[:, 0::2, 1::2], g[:, 1::2, 0::2], g[:, 1::2, 1::2]
    return ((((a + b) + c) + d) * np.float32(0.25)).reshape(n, 40, C)


def g4_reference(row):
    """Independent pure-Python-loop implementation for ONE window [160, C] -> [40, C]; the same float32 operation order, so the comparison is bit-for-bit."""
    row = np.asarray(row, np.float32)
    C = row.shape[1]
    out = np.zeros((40, C), np.float32)
    for r in range(4):
        for c in range(10):
            acc = row[(2 * r) * 20 + 2 * c].copy()
            acc = acc + row[(2 * r) * 20 + 2 * c + 1]
            acc = acc + row[(2 * r + 1) * 20 + 2 * c]
            acc = acc + row[(2 * r + 1) * 20 + 2 * c + 1]
            out[r * 10 + c] = acc * np.float32(0.25)
    return out


def upsample_g4_to_g8(g4):
    """C5 content mutation: nearest-neighbour 2x2 upsample of a 4x10 grid [n, 40, C] to a fake 8x20 grid [n, 160, C] (carries no information beyond G4)."""
    n, _, C = g4.shape
    g = np.asarray(g4).reshape(n, 4, 10, C)
    return np.repeat(np.repeat(g, 2, axis=1), 2, axis=2).reshape(n, 160, C)


# ------------------------------------------------------------------------------------------------ per-token PCA (shared across positions)
def fit_token_pca(tok, rows, k=K_TOK, block=1250):
    """tok [N, m, 1024] fp16/fp32 token SUBSAMPLE (m sampled positions per window); rows = the windows of the fit.  Exact: float64 covariance of the n = len(rows)*m tokens (streamed in blocks of windows), eigh, top-k.
    Centred, NOT scaled.  Returns dict(mu [1024] f32, comp [k,1024] f32, ev [k] f64 (covariance eigenvalues), total_var f64, n int)."""
    rows = np.sort(np.asarray(rows))
    D = tok.shape[-1]
    n = len(rows) * tok.shape[1]
    mu = np.zeros(D, np.float64)
    for a in range(0, len(rows), block):
        mu += np.asarray(tok[rows[a:a + block]], np.float64).reshape(-1, D).sum(0)
    mu /= n
    mu32 = mu.astype(np.float32)
    cov = np.zeros((D, D), np.float64)
    for a in range(0, len(rows), block):
        Xc = np.asarray(tok[rows[a:a + block]], np.float32).reshape(-1, D) - mu32
        cov += (Xc.T @ Xc).astype(np.float64)
    cov /= max(n - 1, 1)
    w, V = np.linalg.eigh(cov)
    o = np.argsort(w)[::-1][:k]
    return {"mu": mu32, "comp": V[:, o].T.astype(np.float32), "ev": w[o], "total_var": float(np.trace(cov)), "n": int(n)}


class TokenStats:
    """Streaming sufficient statistics of token vectors, one block per OUTER fold (PREREG 2; replaces holding the 192k-token subsample in RAM -- free RAM on the box was 2.2 GB).
    pca(blocks) = the exact covariance eigendecomposition of the tokens of the listed blocks, identical in value to ``fit_token_pca`` on the same tokens (pytest).
    Tokens are centred on a pilot mean (the mean of the first batch added) before accumulation, so the float64 second moment does not cancel catastrophically."""

    def __init__(self, D=D_TOK, nblk=5):
        self.D, self.nblk, self.mu0 = D, nblk, None
        self.n = np.zeros(nblk, np.int64)
        self.s1 = np.zeros((nblk, D), np.float64)
        self.s2 = np.zeros((nblk, D, D), np.float64)

    def add(self, blk, X):
        X = np.asarray(X, np.float32)
        if self.mu0 is None:
            self.mu0 = X.mean(0).astype(np.float32)
        Xc = X - self.mu0
        self.n[blk] += len(X)
        self.s1[blk] += Xc.sum(0, dtype=np.float64)
        self.s2[blk] += (Xc.T @ Xc).astype(np.float64)

    def pca(self, blocks, k=K_TOK):
        blocks = list(blocks)
        n = int(self.n[blocks].sum())
        s1, s2 = self.s1[blocks].sum(0), self.s2[blocks].sum(0)
        dm = s1 / n
        cov = (s2 - n * np.outer(dm, dm)) / max(n - 1, 1)
        w, V = np.linalg.eigh(cov)
        o = np.argsort(w)[::-1][:k]
        mu = self.mu0.astype(np.float64) + dm
        return {"mu": mu.astype(np.float32), "comp": V[:, o].T.astype(np.float32), "ev": w[o], "total_var": float(np.trace(cov)), "n": n}


def project_tokens(x, pcas):
    """x [m, T, 1024] float32 -> [len(pcas), m, T*K_TOK] (token-major: token t, component j at column K_TOK*t + j) with one centred projection per PCA in ``pcas`` (stacked into one matmul)."""
    m, T, D = x.shape
    Ccat = np.concatenate([p["comp"] for p in pcas], 0)                              # [F*k, D]
    P = x.reshape(m * T, D) @ Ccat.T                                                # [m*T, F*k]
    k = pcas[0]["comp"].shape[0]
    P = P.reshape(m, T, len(pcas), k)
    out = np.empty((len(pcas), m, T * k), np.float32)
    for i, p in enumerate(pcas):
        off = (p["mu"] @ p["comp"].T).astype(np.float32)                            # [k]
        out[i] = (P[:, :, i, :] - off).reshape(m, T * k)
    return out


def assert_g8_entry(shape, n=12000):
    """C5(a): the G8 entry point accepts ONLY an [n, 163840] (8x20 x 1024) source; the 4x10 file ([n, 40960]) must raise."""
    assert tuple(shape) == (n, 163840), ("G8 entry point: expected [%d, 163840] (the 8x20 grid), got" % n, tuple(shape))


def sample_positions(N, seed, n_sub=N_SUB):
    """The fixed token subsample of the PCA fit: n_sub positions per window, for G8 (of 160) and G4 (of 40), sorted."""
    rng = np.random.default_rng(seed)
    pos8 = np.sort(np.argsort(rng.random((N, 160)), axis=1)[:, :n_sub], axis=1)
    pos4 = np.sort(np.argsort(rng.random((N, 40)), axis=1)[:, :n_sub], axis=1)
    return pos8, pos4


def fit_readout(read_chunk, N, ch, outer, pos8, pos4, k=K_TOK, D=D_TOK):
    """Pass A + fits.  read_chunk(s) -> float32 [m, 160, D] (the windows s .. s+m of the 8x20 source).  Returns (p8, p4): 6 PCAs per grid (outer fold f = all outer-train windows; 5 = all rows)."""
    st8, st4 = TokenStats(D=D, nblk=5), TokenStats(D=D, nblk=5)
    for s in range(0, N, ch):
        x = read_chunk(s)
        m = len(x)
        ar = np.arange(m)[:, None]
        t8 = x[ar, pos8[s:s + m]]
        t4 = g4_from_g8(x)[ar, pos4[s:s + m]]
        ob = outer[s:s + m]
        for b in range(5):
            sel = ob == b
            if sel.any():
                st8.add(b, t8[sel].reshape(-1, D))
                st4.add(b, t4[sel].reshape(-1, D))
    p8, p4 = [], []
    for f in range(6):
        blocks = [b for b in range(5) if b != f]
        p8.append(st8.pca(blocks, k))
        p4.append(st4.pca(blocks, k))
    return p8, p4


def transform_all(read_chunk, N, ch, p8, p4, out8, out4):
    """Pass B: out8 [6, N, 160*k], out4 [6, N, 40*k] (arrays or memmaps) filled chunk by chunk."""
    for s in range(0, N, ch):
        x = read_chunk(s)
        m = len(x)
        out8[:, s:s + m] = project_tokens(x, p8)
        out4[:, s:s + m] = project_tokens(g4_from_g8(x), p4)


# ------------------------------------------------------------------------------------------------ arms and bank
def rec(o, y, clip, outer, counts):
    return {"pooled": L.auc_ci(o, y, clip, counts=counts), "fold": L.auc_ci_fold(o, y, clip, outer, counts=counts)}


def paired(a, b, y, clip, outer, counts):
    return {"fold": L.dauc_ci_fold(a, b, y, clip, outer, counts=counts), "pooled": L.dauc_ci(a, b, y, clip, counts=counts)}


def control_arm_names(src):
    """C7: the controls are run for EVERY source name (probe 1's run_probe.py:108 gated them on src == 'V')."""
    return [f"C1:shuffle:{src}+E", f"C2:donor:{src}+E", f"C3:leak:{src}+E"]


def run_arms(tag, d, fam, arms, budget_s=None):
    """arms: ordered dict name -> (feat_fn(f) -> [N, d] float64, labels or None).  Banks raw/oofg_<tag>_<fam>.npz and raw/resg_<tag>_<fam>.json after EVERY arm; resumable."""
    y, clip, outer, inner = d["y"], d["clip"], d["outer"], d["inner"]
    counts = L.boot_counts(clip)
    resf, oofp = f"{RAW}/resg_{tag}_{fam}.json", f"{RAW}/oofg_{tag}_{fam}.npz"
    res = json.load(open(resf)) if os.path.exists(resf) else {"tag": tag, "family": fam, "n": int(len(y)), "n_go": int(y.sum()), "n_clips": int(len(np.unique(clip))), "c_grid": list(L.C_GRID), "arms": {}}
    oof = dict(np.load(oofp)) if os.path.exists(oofp) else {}
    t0 = time.time()
    for name, (fn, lab) in arms.items():
        key = name.replace(":", "__")
        if key in oof:
            continue
        require_ram(3000, f"arm {name}")
        labels = y if lab is None else lab
        t1 = time.time()
        o, hps = L.cv_oof(fn, labels, outer, inner, fam)
        oof[key] = o
        r = rec(o, labels, clip, outer, counts)
        res["arms"][name] = {**r, "hp_per_fold": hps, "seconds": time.time() - t1}
        np.savez(oofp, **oof)
        json.dump(res, open(resf, "w"), indent=1)
        print(f"{tag} {fam} {name}: fold-mean {r['fold']['auc']:.4f} [{r['fold']['lo']:.4f},{r['fold']['hi']:.4f}]  pooled {r['pooled']['auc']:.4f} [{r['pooled']['lo']:.4f},{r['pooled']['hi']:.4f}]  {time.time() - t1:.0f}s  hp {[v['hp'] for v in hps.values()]}", flush=True)
        if budget_s and time.time() - t0 > budget_s:
            print("time box: banked, re-invoke to continue", flush=True)
            return False
    return True
