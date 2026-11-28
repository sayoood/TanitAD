"""MM-E6 — SPLIT DRIFT INTO SELF-REFERENCE AND ENVIRONMENT.

Implements ``Project Steering/PREREG_DRIFT_DECOMPOSITION.md``. The
pre-registration is authoritative; nothing here redesigns it.

    z_env := P(scene_t)        the part of z_t the FROZEN scene representation explains
    z_res := z_t - z_env       the part it does NOT

``P`` is a ridge from frozen DINOv3 features of frame t onto z_t, **fit on the FIT
split of clips only** and applied to the SCORED split. The EXISTING drift
instrument (``latentmotion.py`` / E-DEC-59: RFF+ridge onto the top-8 PCA directions
of dz = z_{t+4} - z_t, clip-disjoint 10-fold, within-clip Pearson r) is then run
THREE times, changing only its INPUT COLUMN:

    r_full   input z_t      today's drift, the mixture we have been quoting
    r_env    input z_env    environment-driven predictability
    r_res    input z_res    self-referential predictability
    share_self = r_res^2 / (r_env^2 + r_res^2)

⭐ WHY THE TARGET IS THE **SAME** IN ALL THREE READS. The prereg's framing is
"Δz is foreseeable because the scene implies it" versus "Δz is foreseeable from the
latent's own coordinates" — both sentences are about the SAME Δz, and the table says
"drift computed FROM z_env", i.e. z_env is the INPUT. So all three reads predict the
identical target (top-8 PCA of the full Δz, ONE basis, computed once) and differ only
in what they may look at. Decomposing the target instead would make r_env and r_res
describe two different quantities and the ratio uninterpretable.

⛔⛔ CONTROLS THAT MUST READ THEIR KNOWN VALUE. The run is VOID if a prereg control
fails; nothing else may then be read.

  constant        a column of ones -> within-clip Pearson r EXACTLY 0.0000.
  scene_shuffled  P refit with scene features taken from OTHER clips (a derangement
                  over the FIT clips), then applied to the SCORED split's TRUE scene.
                  z_env then carries no scene information => r_env_shuf MUST COLLAPSE.
  rank_matched    a RANDOM orthonormal rank-R projection of z_t. If it MATCHES r_env,
                  then r_env is merely "drift survives any projection": VOID. The
                  MM-E4 lesson applied before the fact.

⭐ THE RANK IS NOT A FREE KNOB, AND MATCHING IT EXACTLY IS WHAT MAKES THE CONTROL
READABLE. ``P`` factors through an R-component scene basis, so rank(P) = R nominally
and exactly; the random control uses a random orthonormal R-dim subspace of the same
2048-dim latent space. Both columns enter the instrument with IDENTICAL rank and
conditioning, so a difference between them cannot be an artefact of one being
lower-rank. R defaults to 96 = ``rangeprobe_rff.D_EFF``, the internal PCA rank the
drift instrument itself uses, so the scene basis has exactly the dimensionality the
read can consume. Declared here, before the run; not chosen to make a number come out.

⭐ THREE ADDITIONS, EACH LABELLED, NONE OF WHICH CHANGES A PRE-REGISTERED READ:

 1. ``rand_res`` — the residual of the SAME random projection, giving
    ``share_self_rand``: the value share_self would take if the split carried no
    information beyond rank arithmetic. z_res is structurally near-full-rank while
    z_env is rank-R, so a raw share_self has a built-in tilt; this pair measures that
    tilt directly and is the null the real share must beat. Reading share_self without
    it would repeat the MM-E4 error in a new costume.
 2. ⛔ **A POSITIVE CONTROL ON P ITSELF** — the out-of-sample R^2 of scene -> z on the
    SCORED split, for P and for the scene-shuffled P. MEASURED in the 16-clip smoke:
    a starved P (6 FIT clips) reached in-FIT R^2 +0.012, z_env was then ~constant,
    r_env read +0.008 and share_self read 0.9998 — **a SELF-DOMINATED verdict
    manufactured entirely by an underpowered projection.** A near-zero out-of-sample
    R^2 means the instrument is not measuring the scene and the reads are vacuous;
    that is an instrument failure, never a finding about the latent.
 3. ``*_3f`` SECONDARY columns — the same construction with a scene representation
    stacked over frames (t-2, t-1, t), clamped exactly as ``v7tiny_g2.encode_clip``
    stacks its 9 input channels. ⚠️ z_t is encoded from THREE frames while the
    pre-registered scene feature is ONE, so single-frame z_env cannot represent
    short-horizon motion content and r_env is a LOWER BOUND by construction. The
    3-frame column removes that asymmetry. The PRIMARY reads and the VOID gates
    remain the pre-registered single-frame ones.

SPLITS — ``SCORED = sorted(clips)[:80]`` is EXACTLY the set the banked drift runs
used, so ``r_full`` is directly comparable to the banked positive controls (e.g.
``e4_l4_crop`` 0.4948, ``postrain30k_freeze`` 0.3905). ``FIT = sorted(clips)[80:]``
(49 clips) is disjoint from it by construction. Every hyper-parameter of P — the
scene PCA basis, the standardisation, the ridge lambda — is fit on FIT only, and
lambda is selected on a CLIP-DISJOINT inner split of FIT (random rows would let
clip-level statistics validate perfectly; documented in ``rff_fold``).

ESTIMATOR: episode-cluster bootstrap over the SCORED clips, resampling the SAME
clips jointly across columns so ``share_self`` is a PAIRED statistic. Never
``overlapping_holdout_se``; never a combination in quadrature.

TIER: T0-DIAGNOSTIC — a world-model diagnostic, NEVER a driving number.
EVIDENCE CLASS: MEASURED (ours; dev-box, CPU — the 4060 was held by D-SAFE-CAL).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

WORK = pathlib.Path(__file__).resolve().parent
OLD = pathlib.Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
                   r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
                   r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
MIRROR = pathlib.Path(r"C:\Users\Admin\tanitad-wt\stack")

# ⚠️ EAGER, ORDERED IMPORT. ``v7tiny_g2`` inserts the G: stack at sys.path[0] at
# import time, and G: dies mid-import with Errno 22. Bind ``tanitad`` from the
# off-Drive mirror FIRST so every submodule resolves there, and PRINT which tree
# won — a namespace-package shadow cannot be undone once bound.
sys.path.insert(0, str(OLD / "sp2"))
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(MIRROR))

CORPUS = pathlib.Path(os.environ.get(
    "MME6_CORPUS", str(OLD / "sp2/cache/physicalai-val130-heldout")))
DINO = pathlib.Path(os.environ.get("MME6_DINO", str(WORK / "dino_heldout_4x8")))
ARMS = os.environ.get("MME6_ARMS", "postrain30k").split(",")
OUT = pathlib.Path(os.environ.get("MME6_OUT", str(WORK / "mm_e6.json")))

F = int(os.environ.get("MME6_FRAMES", "100"))     # frames/clip, instrument default
K = int(os.environ.get("MME6_K", "4"))            # drift horizon, ticks (0.4 s)
K_FOLDS = 10
N_SCORED = int(os.environ.get("MME6_NSCORED", "80"))
N_CLIPS_ALL = int(os.environ.get("MME6_NCLIPS", "0"))    # 0 = every clip
BAND_LO, BAND_HI = 0, int(os.environ.get("MME6_BANDS", "8"))
R_SCENE = int(os.environ.get("MME6_RANK", "96"))         # = D_EFF; rank(P), declared
N_BOOT = int(os.environ.get("MME6_NBOOT", "4000"))
STACK3 = os.environ.get("MME6_STACK3", "1") == "1"
MIN_SCORED = int(os.environ.get("MME6_MIN_SCORED", "20"))   # smoke knob
MIN_FIT = int(os.environ.get("MME6_MIN_FIT", "8"))          # smoke knob
SEED = 0
# ⛔⛔ `CUDA_VISIBLE_DEVICES=""` DOES **NOT** DISABLE CUDA — MEASURED 2026-08-30 on
# this box: torch reports `is_available() True`, `device_count() 0`, and
# `.to("cuda")` STILL SUCCEEDS, landing the tensor on cuda:0. A guard written as
# `torch.device("cuda" if torch.cuda.is_available() else "cpu")` therefore selects
# the GPU while the operator believes it is on CPU — which is how this campaign put
# ~13 min of DINOv3 inference onto a 4060 that another session's run was holding.
# Only `CUDA_VISIBLE_DEVICES=-1` returns False. Same family as the df / Thor `free` /
# cgroup traps: a probe that answers a different question than the one asked.
# ⇒ The device is decided HERE, from an explicit flag, never from availability alone.
FORCE_CPU = os.environ.get("MME6_CPU", "1") == "1"


def pick_device():
    if FORCE_CPU:
        # ⛔ MM-C7: VERIFY ISOLATION BY ATTEMPTING AN ALLOCATION THAT MUST RAISE.
        # `device_count()` reads 0 under exactly the setting that still lets
        # `.to("cuda")` succeed, so a preflight asserting the count verifies
        # NOTHING. Only a failed allocation is evidence.
        try:
            torch.zeros(1).to("cuda")
            raise SystemExit(
                "[FATAL] CPU isolation requested but a CUDA allocation "
                "SUCCEEDED — set CUDA_VISIBLE_DEVICES=-1; refusing to run")
        except RuntimeError:
            pass
        return torch.device("cpu")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return torch.device("cpu")
    return torch.device("cuda")


C_FULL = "z_full (r_full)"
C_ENV = f"z_env rank{R_SCENE} (r_env)"
C_RES = "z_res (r_res)"
C_RND = f"rand{R_SCENE} (rank-matched ctrl)"
C_RNDR = "rand_res (ADDED null)"
C_SHUF = "z_env SCENE-SHUFFLED (ctrl)"
C_CONST = "constant (ctrl)"
C_ENV3 = "z_env_3f (SECONDARY)"
C_RES3 = "z_res_3f (SECONDARY)"
C_APPSHUF = "z_env APPLIED-SHUFFLED (REPAIRED ctrl)"
APPSHUF = os.environ.get("MME6_APPSHUF", "0") == "1"
SCENE_GRID = os.environ.get("MME6_SCENE_GRID", "4x8")   # provenance label only


# --------------------------------------------------------------------------- #
#  EXACT vectorisation of rangeprobe_rff.rff_fold over MANY y-vectors.
# --------------------------------------------------------------------------- #
def make_rff_multi(rff_mod):
    """Return ``rff_fold_multi``: rff_fold's math, one X-pass for all y columns.

    ⛔ WHY THIS IS SAFE AND WHY IT IS STILL VERIFIED. Every random draw inside
    ``rff_fold`` — the clip-disjoint inner split, the bandwidth subsample, the RFF
    matrix W and phases b — depends ONLY on the seed and on X's shape, never on y.
    So the 16 calls this replaces (8 PCA bands x {true, time-shuffled}) build the
    IDENTICAL features and the IDENTICAL inner split; only the ridge algebra differs
    per y, and that is vectorised column-wise with **lambda still selected per y**.
    The estimator is therefore unchanged.
    ⚠️ "Mathematically identical" is a claim, not evidence — ``verify_multi`` below
    re-runs the untouched ``rff_fold`` and compares predictions before any number is
    read. A metric with two implementations has one correct implementation at most.
    """
    D_EFF, D_RFF, LAMBDAS = rff_mod.D_EFF, rff_mod.D_RFF, rff_mod.LAMBDAS

    def rff_fold_multi(Xtr_clips, Ytr, Xte, seed=rff_mod.SEED):
        rng = np.random.default_rng(seed)
        Xtr = np.concatenate(Xtr_clips)
        nclip = len(Xtr_clips)
        bounds = np.cumsum([0] + [len(x) for x in Xtr_clips])
        n_in = max(int(round(0.25 * nclip)), 2)
        hold = set(rng.choice(nclip, size=min(n_in, nclip - 2),
                              replace=False).tolist())
        vi = np.concatenate([np.arange(bounds[c], bounds[c + 1])
                             for c in sorted(hold)])
        ti = np.setdiff1d(np.arange(len(Xtr)), vi)
        mu = Xtr.mean(0, keepdims=True)
        Xc = Xtr - mu
        k = min(D_EFF, Xc.shape[1], max(Xc.shape[0] - 1, 1))
        rs = np.random.default_rng(seed + 1)
        Om = rs.standard_normal((Xc.shape[1], k + 10))
        Yq = Xc @ Om
        for _ in range(2):
            Yq = Xc @ (Xc.T @ Yq)
        Q, _ = np.linalg.qr(Yq)
        _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
        V = Vt[:k].T
        A, B = Xc @ V, (Xte - mu) @ V
        s = A.std(0, keepdims=True) + 1e-8
        A, B = A / s, B / s

        sub = A[rng.choice(len(A), size=min(400, len(A)), replace=False)]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        med = np.sqrt(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
        gamma = max(med, 1e-6)

        W = rng.standard_normal((k, D_RFF)) / gamma
        b = rng.uniform(0, 2 * np.pi, size=D_RFF)
        PA = np.sqrt(2.0 / D_RFF) * np.cos(A @ W + b)
        PB = np.sqrt(2.0 / D_RFF) * np.cos(B @ W + b)

        ym = Ytr.mean(0, keepdims=True)                 # per y column
        yc = Ytr - ym
        G = PA[ti].T @ PA[ti]
        C = PA[ti].T @ yc[ti]
        tr = np.trace(G) / max(D_RFF, 1)
        I = np.eye(D_RFF)
        err = np.empty((len(LAMBDAS), Ytr.shape[1]))
        for li, lam in enumerate(LAMBDAS):
            w = np.linalg.solve(G + lam * tr * I, C)
            err[li] = ((PA[vi] @ w - yc[vi]) ** 2).mean(0)
        pick = err.argmin(0)                            # lambda PER y column
        G = PA.T @ PA
        C = PA.T @ yc
        tr = np.trace(G) / max(D_RFF, 1)
        out = np.empty((len(Xte), Ytr.shape[1]))
        for li in np.unique(pick):
            cols = np.where(pick == li)[0]
            w = np.linalg.solve(G + LAMBDAS[li] * tr * I, C[:, cols])
            out[:, cols] = PB @ w
        return out + ym

    return rff_fold_multi


def kfold_clip_scores_multi(X_clips, Y_list, rff_multi, within_clip_r,
                            k_folds=K_FOLDS, seed=0):
    """One score per (y, clip). Fold partition IDENTICAL to panel_kfold's."""
    n = len(X_clips)
    if n < k_folds:
        k_folds = max(2, n // 2)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k_folds)
    scores = np.full((len(Y_list), n), np.nan)
    for te in folds:
        tr = [q for q in range(n) if q not in set(te.tolist())]
        Xtr = [X_clips[q] for q in tr]
        Ytr = np.column_stack([np.concatenate([Y[q].ravel() for q in tr])
                               for Y in Y_list])
        Xte = np.vstack([X_clips[q] for q in te])
        pred = rff_multi(Xtr, Ytr, Xte)
        off = 0
        for q in te:
            m = len(X_clips[q])
            for yi, Y in enumerate(Y_list):
                scores[yi, q] = within_clip_r(pred[off:off + m, yi],
                                              Y[q].ravel())
            off += m
    if not np.isfinite(scores).all():
        raise RuntimeError("a clip went unscored — fold partition is wrong")
    return scores


def verify_multi(X_clips, Y_list, rff_multi, rff_fold, n_y=3):
    """⛔ GATE: the vectorised path must reproduce the untouched rff_fold."""
    tr = list(range(1, len(X_clips)))
    Xtr = [X_clips[q] for q in tr]
    Xte = X_clips[0]
    Ys = Y_list[:n_y]
    Ytr = np.column_stack([np.concatenate([Y[q].ravel() for q in tr]) for Y in Ys])
    got = rff_multi(Xtr, Ytr, Xte)
    worst = 0.0
    for yi, Y in enumerate(Ys):
        ref, _ = rff_fold(Xtr, [Y[q] for q in tr], Xte)
        worst = max(worst, float(np.abs(ref - got[:, yi]).max()))
    return worst


# --------------------------------------------------------------------------- #
#  P — the ridge from FROZEN scene features onto z_t. FIT SPLIT ONLY.
# --------------------------------------------------------------------------- #
def load_scene(stem, m, stack3):
    """Frozen DINOv3 features for rows 0..m-1 of one clip, streamed from disk.

    ⚠️ STREAMED BY NECESSITY. The dense scene matrix over all clips is 1.6 GB
    (4.9 GB stacked) and free host RAM was ~5 GB with another campaign live. Only
    the R-dim projection is ever retained.
    ⭐ The 3-frame stack clamps with ``max(i-j, 0)`` — EXACTLY how
    ``v7tiny_g2.encode_clip`` builds the encoder's 9 input channels, so the
    secondary scene column sees the same frames the latent did.
    """
    a = np.load(DINO / f"{stem}.npy")[:m].reshape(m, -1).astype(np.float32)
    if not stack3:
        return a
    i = np.arange(m)
    return np.concatenate([a[np.maximum(i - 2, 0)], a[np.maximum(i - 1, 0)], a], 1)


def scene_basis(ids, mlen, stack3, k, seed=SEED):
    """Streaming randomized PCA of the scene features over the FIT rows."""
    rng = np.random.default_rng(seed + 3)
    d = load_scene(ids[0], 1, stack3).shape[1]
    n = sum(mlen[i] for i in ids)
    mu = np.zeros(d, dtype=np.float64)
    for i in ids:
        mu += load_scene(i, mlen[i], stack3).sum(0, dtype=np.float64)
    mu = (mu / n).astype(np.float32)
    p = min(k + 10, n, d)
    Om = rng.standard_normal((d, p)).astype(np.float32)

    # ⚠️ CENTRING IS DONE IN THE SMALL SPACE, NOT THE BIG ONE. At the 16x40 token
    # grid d = 655,360, so `x - mu` is a 251 MB transient per clip and
    # `np.outer(mu, ...)` another 278 MB PER CLIP. Using (x @ M) - (mu @ M) and
    # deferring the rank-1 correction to a single application at the end is
    # algebraically identical and holds peak memory flat as the grid grows.
    def amul(M):
        muM = mu @ M
        return np.concatenate([(load_scene(i, mlen[i], stack3) @ M) - muM
                               for i in ids])

    def atmul(Y):
        acc = np.zeros((d, Y.shape[1]), dtype=np.float32)
        ysum = np.zeros(Y.shape[1], dtype=np.float64)
        off = 0
        for i in ids:
            x = load_scene(i, mlen[i], stack3)
            Yb = Y[off:off + len(x)]
            acc += x.T @ Yb
            ysum += Yb.sum(0, dtype=np.float64)
            off += len(x)
        acc -= np.outer(mu, ysum.astype(np.float32))
        return acc

    Y = amul(Om)
    for _ in range(2):
        Y = amul(atmul(Y))
    Q, _ = np.linalg.qr(Y)
    B = atmul(Q).T
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return mu, np.ascontiguousarray(Vt[:k].T)


def ridge_multi(A, Z, lam_scale, lam):
    G = A.T @ A
    return np.linalg.solve(G + lam * lam_scale * np.eye(G.shape[0]), A.T @ Z)


def fit_P(A_clips, Z_clips, lambdas, seed=SEED, tag=""):
    """Ridge A -> Z, lambda on a CLIP-DISJOINT inner split of the FIT clips."""
    rng = np.random.default_rng(seed + 5)
    nclip = len(A_clips)
    bounds = np.cumsum([0] + [len(x) for x in A_clips])
    n_in = max(int(round(0.25 * nclip)), 2)
    hold = set(rng.choice(nclip, size=min(n_in, nclip - 2), replace=False).tolist())
    vi = np.concatenate([np.arange(bounds[c], bounds[c + 1]) for c in sorted(hold)])
    ti = np.setdiff1d(np.arange(bounds[-1]), vi)
    A = np.concatenate(A_clips)
    Z = np.concatenate(Z_clips)
    zm = Z[ti].mean(0, keepdims=True)
    Zc = Z - zm
    scale = float(np.trace(A[ti].T @ A[ti])) / max(A.shape[1], 1)
    best, best_lam = None, lambdas[0]
    for lam in lambdas:
        Wb = ridge_multi(A[ti], Zc[ti], scale, lam)
        e = float(((A[vi] @ Wb - Zc[vi]) ** 2).mean())
        if best is None or e < best:
            best, best_lam = e, lam
    scale_all = float(np.trace(A.T @ A)) / max(A.shape[1], 1)
    zm_all = Z.mean(0, keepdims=True)
    Wf = ridge_multi(A, Z - zm_all, scale_all, best_lam)
    r2 = 1.0 - float(((A @ Wf + zm_all - Z) ** 2).sum()) / max(
        float(((Z - zm_all) ** 2).sum()), 1e-12)
    print(f"    P{tag}: lambda {best_lam:g} (clip-disjoint inner split, "
          f"{len(hold)}/{nclip} clips held) · in-FIT R2 {r2:+.4f}", flush=True)
    return Wf, zm_all, best_lam, r2


def oos_r2(A_sc, Zsc, Wp, zm):
    """⛔ THE POSITIVE CONTROL ON P: out-of-sample R^2 of scene -> z on SCORED."""
    num = sum(float(((A @ Wp + zm - Z) ** 2).sum()) for A, Z in zip(A_sc, Zsc))
    den = sum(float(((Z - zm) ** 2).sum()) for Z in Zsc)
    return 1.0 - num / max(den, 1e-12)


def participation(X):
    """Participation ratio (Tr C)^2 / Tr C^2 — the programme's own rank measure."""
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    t1 = float(np.trace(C))
    return t1 * t1 / max(float((C * C.T).sum()), 1e-30)


def boot_cluster(mats, n_boot=N_BOOT, seed=SEED):
    """Episode-cluster bootstrap over SCORED clips, PAIRED across columns."""
    rng = np.random.default_rng(seed + 7)
    keys = list(mats)
    nclip = mats[keys[0]].shape[1]
    draws = {k: np.empty(n_boot) for k in keys}
    idx = rng.integers(0, nclip, size=(n_boot, nclip))
    for b in range(n_boot):
        j = idx[b]
        for k in keys:
            draws[k][b] = mats[k][:, j].mean()
    return draws


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4)]


def share(res, env):
    a = max(float(res), 0.0) ** 2
    b = max(float(env), 0.0) ** 2
    return a / max(a + b, 1e-30)


# --------------------------------------------------------------------------- #
def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import tanitad
    print(f"  tanitad imported from {tanitad.__file__}", flush=True)
    if "tanitad-wt" not in tanitad.__file__ and "tanitad-mirror" not in tanitad.__file__:
        print("  [FATAL] tanitad resolved OUTSIDE the off-Drive mirror — refusing "
              "to run the stack from the G: mount (Errno 22 mid-import)")
        return 2
    import rangeprobe_rff as RF
    import v7tiny_g2 as G
    from rangeprobe_rff import rff_fold, within_clip_r
    rff_multi = make_rff_multi(RF)

    dev = pick_device()
    clips = sorted(CORPUS.glob("*.v2ep.pt"))
    if N_CLIPS_ALL:
        clips = clips[:N_CLIPS_ALL]
    scored, fit = clips[:N_SCORED], clips[N_SCORED:]
    if not fit:
        print("  [FATAL] no FIT clips left after the SCORED split")
        return 2
    missing = [c.stem for c in clips if not (DINO / f"{c.stem}.npy").is_file()]
    if missing:
        print(f"  [FATAL] {len(missing)} clips have no DINOv3 scene bank "
              f"(e.g. {missing[:3]}) — run mm_e6_dino_scene.py first")
        return 2
    arms = [a for a in ARMS if (OLD / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    absent = [a for a in ARMS if a not in arms]

    print("\n  MM-E6 — DRIFT DECOMPOSITION: environment vs self-reference")
    print(f"  corpus {CORPUS.name} · SCORED {len(scored)} clips / "
          f"FIT {len(fit)} clips (disjoint) · k={K} · PCA band "
          f"[{BAND_LO}:{BAND_HI}) · rank(P)={R_SCENE} · device {dev}")
    print(f"  scene bank {DINO.name} (grid {SCENE_GRID}) · repaired applied-shuffle "
          f"control {'ON' if APPSHUF else 'OFF'} · 3-frame secondary "
          f"{'ON' if STACK3 else 'OFF'} · OMP_NUM_THREADS "
          f"{os.environ.get('OMP_NUM_THREADS', 'unset')}")
    if absent:
        print(f"  ⛔ CHECKPOINT ABSENT, arm NOT run: {absent}")
    print(flush=True)

    rep = {"_evidence_class": "MEASURED (ours; dev-box CPU — the RTX 4060 was held "
                              "by the D-SAFE-CAL campaign for the whole window)",
           "eval_tier": "T0-DIAGNOSTIC", "hypothesis": "MM-E6",
           "prereg": "Project Steering/PREREG_DRIFT_DECOMPOSITION.md",
           "split": "HELD-OUT corpus; SCORED/FIT clip-disjoint",
           "corpus": str(CORPUS), "n_scored_clips": len(scored),
           "n_fit_clips": len(fit), "k": K, "pca_band": [BAND_LO, BAND_HI],
           "rank_P": R_SCENE,
           "dino_model": "facebook/dinov3-vitl16-pretrain-lvd1689m (FROZEN, "
                         "independent encoder; CPU fp32 forward cross-checked "
                         "against the banked GPU bf16 bank at cosine 0.9999)",
           "estimator": "episode-cluster bootstrap over SCORED clips, PAIRED "
                        "across columns (same resampled clips)",
           "n_boot": N_BOOT, "secondary_3frame_scene": STACK3,
           # ⚠️ PROVENANCE THAT TRAVELS WITH THE NUMBERS. Two instrument changes
           # were approved together for v2 (repaired control + 16x40 scene grid),
           # so an artifact that does not say WHICH scene bank and WHICH control it
           # used can be misread as the other run. And the probe is not
           # thread-count stable in the third decimal (MEASURED: speed +0.1572 at
           # OMP=8 vs +0.1503 at 6), so the thread count is part of the result.
           "scene_bank": str(DINO), "scene_grid": SCENE_GRID,
           "repaired_control_applied_shuffle": APPSHUF,
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
           "numpy_threads_note": "cross-arm comparisons are only valid at a FIXED "
                                 "thread count; this panel is uniform",
           "target": "top-8 PCA directions of dz = z_{t+k} - z_t; ONE basis, "
                     "identical for every column — only the INPUT changes",
           "arms_absent": absent, "arms": {}}

    for arm in arms:
        t_arm = time.time()
        w, st = G.load_arm(arm, dev)          # sets .eval() (MM-C2)
        Z, mlen = {}, {}
        with torch.no_grad():
            for c in clips:
                d = torch.load(c, map_location="cpu", weights_only=False)
                yaw = np.asarray(d["poses"], dtype=np.float64)[:, 2]
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float().numpy()
                nsc = int(np.load(DINO / f"{c.stem}.npy", mmap_mode="r").shape[0])
                # ⚠️ ROW ALIGNMENT IS THE BANKED FORMULA, UNCHANGED, so r_full is
                # comparable to the banked drift numbers on the same 80 clips.
                m = min(len(zt) - K, len(act) - K, len(spd) - K, len(yaw) - K - 1,
                        nsc)
                if m < 30:
                    continue
                Z[c.stem] = zt[:m + K].astype(np.float64)
                mlen[c.stem] = m
        del w
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        sc_ids = [c.stem for c in scored if c.stem in Z]
        fi_ids = [c.stem for c in fit if c.stem in Z]
        if len(sc_ids) < MIN_SCORED or len(fi_ids) < MIN_FIT:
            print(f"  {arm}: too few usable clips ({len(sc_ids)}/{len(fi_ids)})")
            continue
        d_z = Z[sc_ids[0]].shape[1]
        nrow = sum(mlen[i] for i in sc_ids)
        print(f"  === {arm} (step {st}) — SCORED {len(sc_ids)} clips, {nrow} rows, "
              f"d_z {d_z} ===", flush=True)

        Zfit = [Z[i][:mlen[i]] for i in fi_ids]
        Zsc = [Z[i][:mlen[i]] for i in sc_ids]
        DZ = [Z[i][K:K + mlen[i]] - Z[i][:mlen[i]] for i in sc_ids]
        zvar = sum(float(((x - np.concatenate(Zsc).mean(0)) ** 2).sum()) for x in Zsc)

        COL, diag = {}, {}
        t0 = time.time()
        for stack3 in ([False, True] if STACK3 else [False]):
            tg = "_3f" if stack3 else ""
            mu_s, V_s = scene_basis(fi_ids, mlen, stack3, R_SCENE)
            A_fit = [(load_scene(i, mlen[i], stack3) - mu_s) @ V_s for i in fi_ids]
            sd = np.concatenate(A_fit).std(0, keepdims=True) + 1e-8
            A_fit = [(x / sd).astype(np.float64) for x in A_fit]
            A_sc = [(((load_scene(i, mlen[i], stack3) - mu_s) @ V_s) / sd
                     ).astype(np.float64) for i in sc_ids]
            Wp, zm, lam, r2 = fit_P(A_fit, Zfit, RF.LAMBDAS, tag=tg)
            r2o = oos_r2(A_sc, Zsc, Wp, zm)
            print(f"    P{tg}: ⭐ OUT-OF-SAMPLE R2 (scene->z, SCORED) {r2o:+.4f}",
                  flush=True)
            env = [A @ Wp + zm for A in A_sc]
            diag[f"P{tg}"] = {"lambda": lam, "in_fit_R2": round(r2, 4),
                              "oos_R2_scored": round(r2o, 4),
                              "d_scene": int(A_sc[0].shape[1] and
                                             load_scene(sc_ids[0], 1, stack3).shape[1])}
            if not stack3:
                COL[C_FULL] = Zsc
                COL[C_ENV] = env
                COL[C_RES] = [zz - ee for zz, ee in zip(Zsc, env)]
                # --- scene_shuffled: P refit against OTHER clips' scene ---
                rr = np.random.default_rng(SEED + 11)
                n_f = len(fi_ids)
                perm = rr.permutation(n_f)
                for _ in range(200):
                    if not any(perm == np.arange(n_f)):
                        break
                    perm = rr.permutation(n_f)
                A_sh, Z_sh = [], []
                for j in range(n_f):
                    mm = min(len(A_fit[perm[j]]), len(Zfit[j]))
                    A_sh.append(A_fit[perm[j]][:mm])
                    Z_sh.append(Zfit[j][:mm])
                Wq, zmq, lamq, r2q = fit_P(A_sh, Z_sh, RF.LAMBDAS,
                                           tag=" [SCENE-SHUFFLED]")
                r2qo = oos_r2(A_sc, Zsc, Wq, zmq)
                print(f"    P [SCENE-SHUFFLED]: ⭐ OUT-OF-SAMPLE R2 {r2qo:+.4f} "
                      f"(must be ~0 or negative)", flush=True)
                diag["P_scene_shuffled"] = {"lambda": lamq,
                                            "in_fit_R2": round(r2q, 4),
                                            "oos_R2_scored": round(r2qo, 4)}
                COL[C_SHUF] = [A @ Wq + zmq for A in A_sc]
                # --- rank-matched random projection, SAME rank ---
                rq = np.random.default_rng(SEED + 13)
                Qr, _ = np.linalg.qr(rq.standard_normal((d_z, R_SCENE)))
                COL[C_RND] = [((zz - zm) @ Qr) @ Qr.T + zm for zz in Zsc]
                COL[C_RNDR] = [zz - rr2 for zz, rr2 in zip(Zsc, COL[C_RND])]
                COL[C_CONST] = [np.ones((len(x), 1)) for x in Zsc]
                if APPSHUF:
                    # ⛔⛔ THE REPAIRED SCENE CONTROL (ADDED — NOT pre-registered).
                    # The prereg's control shuffles at FIT time and then applies the
                    # shuffled map to the TRUE scene, so its output
                    #     z_env_shuf = W_shuf · scene_TRUE
                    # IS STILL A LINEAR IMAGE OF THIS CLIP'S REAL SCENE. It cannot
                    # carry "no scene information", and MEASURED across the whole
                    # range of P power — oos R^2 0.0002 to 0.5412 — it reads
                    # statistically identical to r_env every time. A control that
                    # cannot separate is not a control.
                    # ⇒ Destroy the correspondence at APPLICATION time instead:
                    #     z_env_appshuf = W_true · scene_of_ANOTHER_clip
                    # which is the same map, the same rank, the same marginal — and
                    # genuinely no information about THIS clip's scene. If r_env is
                    # scene-driven this must collapse; if it does not, r_env never
                    # depended on the scene at all.
                    rs2 = np.random.default_rng(SEED + 17)
                    ns = len(A_sc)
                    ps = rs2.permutation(ns)
                    for _ in range(200):
                        if not any(ps == np.arange(ns)):
                            break
                        ps = rs2.permutation(ns)
                    cols = []
                    for j in range(ns):
                        src, mj = A_sc[ps[j]], len(A_sc[j])
                        if len(src) < mj:
                            src = np.vstack(
                                [src, np.repeat(src[-1:], mj - len(src), 0)])
                        cols.append(src[:mj] @ Wp + zm)
                    COL[C_APPSHUF] = cols
            else:
                COL[C_ENV3] = env
                COL[C_RES3] = [zz - ee for zz, ee in zip(Zsc, env)]
            del A_fit, A_sc, V_s
        pr = {k: participation(np.concatenate(v)) for k, v in COL.items()
              if v[0].shape[1] > 1}
        print("    participation: " + " · ".join(f"{k.split(' ')[0]} {v:.2f}"
                                                 for k, v in pr.items()))
        print(f"    P prep {time.time() - t0:.0f}s", flush=True)

        # ---- the EXISTING drift read, one X-pass per column ------------------
        ALL = np.concatenate(DZ)
        mu = ALL.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(ALL - mu, full_matrices=False)
        del ALL
        Y_list = []
        for j in range(BAND_LO, BAND_HI):
            Y_list.append([(dz - mu) @ Vt[j][:, None] for dz in DZ])
        for j in range(BAND_LO, BAND_HI):
            rj = np.random.default_rng(100 + j)
            Y_list.append([y.ravel()[rj.permutation(len(y))][:, None]
                           for y in Y_list[j - BAND_LO]])
        nb = BAND_HI - BAND_LO

        worst = verify_multi(COL[C_FULL], Y_list, rff_multi, rff_fold)
        print(f"    ⛔ VECTORISATION GATE: max|rff_fold_multi - rff_fold| = "
              f"{worst:.3e} {'PASS' if worst < 1e-8 else 'FAIL'}", flush=True)
        if worst >= 1e-8:
            print("    [FATAL] the vectorised RFF path does not reproduce the "
                  "banked instrument — refusing to read any number")
            return 3

        print(f"    {'column':<34}{'r':>9}{'shuf':>9}{'r-shuf':>9}{'t':>8}"
              f"{'d':>6}", flush=True)
        print("    " + "-" * 75)
        TR, SH = {}, {}
        for cn, X in COL.items():
            s = kfold_clip_scores_multi(X, Y_list, rff_multi, within_clip_r)
            TR[cn], SH[cn] = s[:nb], s[nb:]
            dd = (TR[cn] - SH[cn]).mean(0)
            t = float(dd.mean()) / max(float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
            print(f"    {cn:<34}{TR[cn].mean():>+9.4f}{SH[cn].mean():>+9.4f}"
                  f"{dd.mean():>+9.4f}{t:>8.2f}{X[0].shape[1]:>6}", flush=True)

        # ---- reads + PAIRED episode-cluster bootstrap ------------------------
        draws = boot_cluster(TR)
        r = {cn: float(TR[cn].mean()) for cn in COL}
        sh_self = share(r[C_RES], r[C_ENV])
        sh_rand = share(r[C_RNDR], r[C_RND])
        d_self = np.array([share(draws[C_RES][b], draws[C_ENV][b])
                           for b in range(N_BOOT)])
        d_rand = np.array([share(draws[C_RNDR][b], draws[C_RND][b])
                           for b in range(N_BOOT)])
        d_diff = d_self - d_rand
        reads = {"r_full": round(r[C_FULL], 4), "r_env": round(r[C_ENV], 4),
                 "r_res": round(r[C_RES], 4),
                 "share_self": round(sh_self, 4), "share_self_ci95": ci(d_self),
                 "share_self_rank_matched_null": round(sh_rand, 4),
                 "share_self_null_ci95": ci(d_rand),
                 "share_self_minus_null": round(sh_self - sh_rand, 4),
                 "share_self_minus_null_ci95": ci(d_diff)}
        if STACK3:
            sh3 = share(r[C_RES3], r[C_ENV3])
            d3 = np.array([share(draws[C_RES3][b], draws[C_ENV3][b])
                           for b in range(N_BOOT)])
            reads.update({"SECONDARY_r_env_3f": round(r[C_ENV3], 4),
                          "SECONDARY_r_res_3f": round(r[C_RES3], 4),
                          "SECONDARY_share_self_3f": round(sh3, 4),
                          "SECONDARY_share_self_3f_ci95": ci(d3)})

        # ---- CONTROL GATES — the run is VOID if a prereg control fails -------
        r_const, r_env, r_shf, r_rnd = r[C_CONST], r[C_ENV], r[C_SHUF], r[C_RND]
        g_const = abs(r_const) < 1e-9
        # ⛔⛔ THE SCENE-SHUFFLED GATE IS A **PAIRED MARGIN**, NOT "is the shuffle
        # small". A threshold like `r_shuf < max(0.25*r_env, 0.02)` PASSES when BOTH
        # reads are noise — which is precisely the state that means P measured
        # nothing. MEASURED in the 24-clip smoke: r_env +0.0138 vs r_shuf +0.0158
        # "passed" while the shuffled read was the LARGER of the two. The prereg's
        # requirement is that shuffling COLLAPSES r_env, i.e. that the true P is
        # demonstrably ABOVE its own scrambled null:
        #     PASS iff the paired episode-cluster CI of (r_env - r_shuf) excludes 0.
        d_es = draws[C_ENV] - draws[C_SHUF]
        # ⚠️ THE FIT-TIME SHUFFLE IS THE SUPERSEDED CONTROL. It is still COMPUTED and
        # REPORTED — it is the evidence for why it was replaced — but it no longer
        # decides VOID once the repaired control is in force (approved 2026-08-30).
        g_shuf_superseded = float(np.percentile(d_es, 2.5)) > 0.0
        g_shuf = g_shuf_superseded
        if APPSHUF:
            d_ea = draws[C_ENV] - draws[C_APPSHUF]
            reads["REPAIRED_r_env_applied_shuffled"] = round(r[C_APPSHUF], 4)
            reads["REPAIRED_r_env_minus_appshuf"] = round(r[C_ENV] - r[C_APPSHUF], 4)
            reads["REPAIRED_r_env_minus_appshuf_ci95"] = ci(d_ea)
            reads["REPAIRED_control_collapses"] = bool(
                float(np.percentile(d_ea, 2.5)) > 0.0)
            # ⭐ THE SCENE GATE IS NOW THE REPAIRED CONTROL. Same requirement as the
            # prereg's — the projection must be demonstrably above a null that
            # carries no information about THIS clip's scene — but through a null
            # that can actually be zero.
            g_shuf = reads["REPAIRED_control_collapses"]
            print(f"    ⭐ REPAIRED CONTROL  r_env {r[C_ENV]:+.4f} vs "
                  f"applied-shuffled {r[C_APPSHUF]:+.4f}, paired diff "
                  f"{r[C_ENV] - r[C_APPSHUF]:+.4f} {ci(d_ea)} "
                  f"[{'COLLAPSES — r_env IS scene-driven' if float(np.percentile(d_ea, 2.5)) > 0 else 'does NOT collapse'}]")
        g_rank = abs(r_rnd - r_env) > 0.10 * max(abs(r_env), 1e-9)
        # ⛔ NOT a prereg gate, but the reads are VACUOUS without it: if P cannot
        # predict z from the scene out of sample, z_env is ~constant and
        # share_self -> 1 by construction, not by finding.
        g_P = diag["P"]["oos_R2_scored"] > 0.02
        void = not (g_const and g_shuf and g_rank)

        print(f"\n    CONTROLS  constant {r_const:+.6f} "
              f"[{'PASS' if g_const else 'FAIL'}] · scene-shuffled {r_shf:+.4f} vs "
              f"r_env {r_env:+.4f}, paired diff {r_env - r_shf:+.4f} {ci(d_es)} "
              f"[{'PASS' if g_shuf_superseded else 'FAIL'}"
              f"{' — SUPERSEDED, does not gate' if APPSHUF else ''}] · "
              f"rank-matched {r_rnd:+.4f} vs r_env {r_env:+.4f} "
              f"[{'PASS (distinct)' if g_rank else 'FAIL (matches P)'}]")
        print(f"    P POWER   out-of-sample R2 {diag['P']['oos_R2_scored']:+.4f} "
              f"[{'PASS' if g_P else '⚠️ P IS UNDERPOWERED — reads vacuous'}] · "
              f"shuffled-P {diag['P_scene_shuffled']['oos_R2_scored']:+.4f}")
        if void:
            print("    ⛔ VOID — a pre-registered control did not read its known "
                  "value. share_self is NOT reported for this arm.")
        else:
            print(f"    share_self {sh_self:.4f} {ci(d_self)}  ·  "
                  f"null (rank-matched pair) {sh_rand:.4f} {ci(d_rand)}  ·  "
                  f"diff {sh_self - sh_rand:+.4f} {ci(d_diff)}")
            if STACK3:
                print(f"    SECONDARY (3-frame scene) share_self "
                      f"{reads['SECONDARY_share_self_3f']:.4f} "
                      f"{reads['SECONDARY_share_self_3f_ci95']}")
        print(f"    arm elapsed {time.time() - t_arm:.0f}s\n", flush=True)

        rep["arms"][arm] = {
            "step": int(st), "n_scored_clips": len(sc_ids),
            "n_fit_clips": len(fi_ids), "n_rows_scored": int(nrow),
            "d_z": int(d_z), "P": diag,
            "rff_vectorisation_max_abs_dev": float(worst),
            "participation": {k: round(v, 2) for k, v in pr.items()},
            "columns": {cn: {"r": round(float(TR[cn].mean()), 4),
                             "r_time_shuffled_null": round(float(SH[cn].mean()), 4),
                             "r_minus_shuffled": round(
                                 float((TR[cn] - SH[cn]).mean()), 4),
                             "ci95_r": ci(draws[cn]),
                             "d_input": int(COL[cn][0].shape[1])}
                        for cn in COL},
            "reads": reads,
            "controls": {"constant_r": round(r_const, 8), "constant_pass": g_const,
                         "scene_shuffled_r": round(r_shf, 4),
                         "r_env_minus_r_shuffled": round(r_env - r_shf, 4),
                         "r_env_minus_r_shuffled_ci95": ci(d_es),
                         "SUPERSEDED_fit_time_shuffle_passes":
                             bool(g_shuf_superseded),
                         "scene_gate_source": ("REPAIRED applied-shuffle" if APPSHUF
                                               else "PREREG fit-time shuffle"),
                         "scene_gate_passes": bool(g_shuf),
                         "rank_matched_r": round(r_rnd, 4),
                         "rank_matched_distinct_from_P": bool(g_rank),
                         "P_oos_R2_pass": bool(g_P)},
            "VOID": bool(void)}
        OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")   # bank per arm
        del COL, TR, SH, Z, Zsc, Zfit, DZ, Y_list

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
