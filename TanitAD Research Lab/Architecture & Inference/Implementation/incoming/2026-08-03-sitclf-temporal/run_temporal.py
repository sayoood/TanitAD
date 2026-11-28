"""DOES THE SITUATION CLASSIFIER LACK TEMPORAL CONTENT? -- the discriminating experiment.

THE HYPOTHESIS UNDER TEST (PI brief 2026-08-03)
-----------------------------------------------
A sibling stream read from source that our models keep only the LAST frame's feature map
(`stack/tanitad/refs/refc.py:1112-1117`), so the cross-attended tokens are SINGLE-INSTANT; and
`long_accel` was separately found unrecoverable from the frozen v1 latents across 17 head
architectures. The proposed explanation for B4's capacity curve PEAKING AT 129 PARAMETERS is that
the head is starved of MOTION: a lane change or an intersection approach is defined by movement over
seconds, and no head of any size can see it in a still frame.

⚠️ THE HYPOTHESIS AS STATED IS NOT YET A TEST, because the deployed head does NOT read one frame.
`sitclf.causal_window` already stacks WIN=8 frames of features (offsets -7..0), so 0.7 s of history
is ALREADY in the design matrix. Anything a linear head could do with explicit differences it can
already do with the stack -- differences are a linear function of it. So "add temporal content" has
to be decomposed into mechanisms that are actually distinguishable:

  H-T1  WINDOW LENGTH.   0.7 s is too short for a manoeuvre defined over ~3 s (the label's own
                         lead_s). => at MATCHED capacity, a longer window beats WIN=8.
  H-T2  MOTION SUBSPACE. The PCA basis is fitted on the marginal FRAME distribution, so it keeps
                         directions of maximum APPEARANCE variance. Frame-to-frame change can live
                         in directions with negligible marginal variance and be truncated away at
                         rank 16. A basis fitted on the temporal DIFFERENCE keeps a different
                         subspace of the 2048-d readout -- NOT a reparameterisation, so it can
                         carry what the appearance stack cannot express at equal rank.
  H-T3  PARAMETERISATION. Even where the stack spans the differences, the optimiser may not find
                         them; explicit motion channels could help the NON-LINEAR head.

⛔ AND THE PREMISE ITSELF NEEDS A CORRECTION, MEASURED BEFORE THIS RUN. The brief reasons from
"a single RGB frame carries no relative velocity, no closing rate, no TTC". Our per-frame latent is
NOT a single RGB frame. The episode cache stores `frames_u8 [T, 9, 256, 256]` (probe 1: a real cache
tensor) and `config.py:17` reads `9 = camera (3-frame stack, D-015)`, `config.py:360` "3 RGB frames
at 100 ms spacing channel-stacked" (probe 2). So EVERY v1 latent already integrates 0.2 s of motion,
and the WIN=8 stack of them spans 0.9 s of motion-bearing evidence. The head is therefore not
motion-blind by construction; the open question is whether the motion it can see is ENOUGH and
whether the rank-16 appearance PCA preserves it. That is what H-T1 and H-T2 actually ask.

PRE-REGISTERED OUTCOMES -- both committed before the run (CLAUDE.md operating standard 5)
  CONFIRMED  if, on lane_change OR intersection, some longer-window or motion-basis arm beats the
             banked floor recipe `ridge_app16_w8` with a PAIRED episode-cluster bootstrap CI that
             EXCLUDES ZERO. => the features/window are the constraint; the programme should buy a
             temporal encoder (BACKLOG B5), not a bigger head.
  REFUTED    if no window length in {1, 2, 4, 8, 16, 32} and no motion basis separates upward from
             `ridge_app16_w8` on either powered situation. => the temporal-content hypothesis is
             refuted FOR THIS TASK and the limit is elsewhere (label definition, base rate, FOV).
  roundabout is reported but MAY NOT DECIDE ANYTHING -- it is unpowered by prior measurement
             (721 positives, its own label control fails).

THE CONTROLS -- fitted and scored BEFORE any real arm is quoted
  NEG_FEAT   every arm refitted with the image features permuted ACROSS clips, labels untouched.
             The permutation preserves WITHIN-clip temporal order, so the null still sees real
             motion -- just some other drive's. That is what makes it the right null here: a
             temporal arm cannot get credit for "motion exists", only for THIS clip's motion.
  INVARIANCE `ridge_app16_w8_diffparam` is an EXACTLY INVERTIBLE linear remap of the reference arm's
             own window into [f_t, f_t-f_{t-1}, ...]. It spans the identical hypothesis space, so
             its AP measures what a pure change of basis does through standardisation + the L2
             penalty alone. Without it, any "motion channels help" reading is unattributable.
  NEG_LABEL  the reference and the peak arm scored against labels permuted across whole CLUSTERS.
  C-POS      ⭐ THE POWER CONTROL, required by PRE_REGISTRATION.md Sec 4: a PRIVILEGED arm that MUST
             separate, or the study is UNPOWERED and no verdict may be issued. Here it is the ego
             kinematic block windowed the same way. ⛔ IT IS NOT A DEPLOYABLE -- the PI ruling bars
             ego at inference and this arm exists ONLY to prove these rows can be separated at all.
             Without it, "no temporal arm moved" is indistinguishable from "nothing can move here".
  C-POW      PRE_REGISTRATION.md Sec 4: fewer than 40 held-out positive CLUSTERS => that situation is
             UNDERPOWERED and gets NO verdict. Counted and written to disk BEFORE any score is read.

⚠️ VERDICT LANGUAGE IS CONSTRAINED BY THE PRE-REG: "a `separated: false` is UNPOWERED, not refuted".
So a null result may NOT be reported as a bare refutation. It is reported as NO EFFECT ABOVE THE
STUDY'S MDE, with the MDE stated -- the widest paired-vs-reference CI half-width actually achieved.

PROTOCOL (identical to BACKLOG B4 `run_matched_capacity.py`, so the two tables compose)
  * 2 outer folds over whole clip CLUSTERS; a row's score comes from a fit that never saw its clip.
  * inside each outer TRAIN half, 20 % of clusters select the epoch (transformer) or lambda (ridge).
  * PCA -- appearance AND motion -- fitted on that fold's FIT rows only.
  * ⭐ EVERY ARM IS SCORED ON IDENTICAL ROWS: validity is intersected with a full in-clip history at
    WIN_MAX = 32, not at each arm's own window. Comparing a WIN=32 arm on the rows it can reach
    against a WIN=8 arm on rows it cannot would confound the window with the row set, and the
    paired estimator would be invalid. This costs rows relative to B4 and the count is reported.
  * scores banked after EVERY arm, so a killed run still leaves a 0-GPU re-analysis surface.

usage:
  python run_temporal.py --substrate <npz> --out results_temporal.json [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.eval.ap_ci import (ap_episode_cluster_bootstrap,        # noqa: E402
                                ap_lift, average_precision,
                                paired_ap_episode_cluster_bootstrap)
from tanitad.eval.sitclf import (CausalSitHead, causal_window,       # noqa: E402
                                 clip_runs, cluster_folds, diff_reparam,
                                 head_param_count, predict_sit_head,
                                 ridge_param_count, ridge_scores,
                                 temporal_difference, width_for_param_budget)
from tanitad.eval.sitclf_deploy import (ScoreBundle,                 # noqa: E402
                                        four_family_report,
                                        permute_labels_by_cluster,
                                        precision_recall_at_budget)

WIN_MAX = 32                  # 3.1 s -- the longest window on the ladder; sets the common row set
EPOCHS = 8                    # sc_train.py CONFIGS, as B4
POS_WEIGHT = 20.0
SEL_FRAC = 0.20
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)     # sc_train.py:46
REFERENCE = "ridge_app16_w8"  # the banked floor recipe EXACTLY: PCA-16, WIN=8, 129 params/head
TF_BUDGET = 417_028           # the deployed transformer rung's exact parameter count
HZ = 10.0


def log(m):
    """Print, surviving a cp1252 console.

    A run that dies on the encoding of its own progress message loses the whole
    fit; the Windows console here is cp1252 and a single non-ASCII marker in a
    log line raised ``UnicodeEncodeError`` from inside ``print``.
    """
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


# --------------------------------------------------------------------------- #
# representations                                                             #
# --------------------------------------------------------------------------- #
def fit_pca(X, rows, r, seed=0, n_sample=150_000, device="cpu"):
    """PCA mean + top-r basis on the given rows ONLY (leak-free per fold)."""
    g = np.random.default_rng(seed)
    idx = rows if rows.size <= n_sample else np.sort(g.choice(rows, n_sample, replace=False))
    A = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(device)
    mu = A.mean(0, keepdim=True)
    A = A - mu
    _u, _s, v = torch.svd_lowrank(A, q=min(r + 16, A.shape[1] - 1), niter=4)
    del A
    if device != "cpu":
        torch.cuda.empty_cache()
    return mu.cpu().numpy(), v[:, :r].cpu().numpy().astype(np.float32)


def project(X, mu, W):
    """`run_matched_capacity.project` verbatim: PCA then divide by the global mean-abs.

    Each block is normalised by its OWN mean-abs, which is what lets an
    appearance+motion concat be built without reintroducing the scale-mismatch
    defect that made the banked early-concat img+ego arm lose to its own ablation.
    """
    img = (np.asarray(X, dtype=np.float32) - mu) @ W
    img /= max(float(np.abs(img).mean()), 1e-6)
    return img


def forward_stack(X, starts, ends, lead: int) -> np.ndarray:
    """⛔ FUTURE-READING. ``out[t] = [X[t+1], ..., X[t+lead]]`` inside the clip.

    Used ONLY to build the C-POS oracle probe. Rows without a full in-clip future
    are edge-held at the clip's last frame, which is conservative for a control
    (it makes the oracle slightly worse, never better).
    """
    A = np.asarray(X, dtype=np.float32)
    n, c = A.shape
    out = np.zeros((n, c * lead), dtype=np.float32)
    for a, b in zip(starts, ends):
        idx = np.arange(a, b)[:, None] + np.arange(1, lead + 1)[None, :]
        np.clip(idx, a, b - 1, out=idx)
        out[a:b] = A[idx].reshape(b - a, c * lead)
    return out


def fold_splits(cc, folds, f, seed=0):
    te = folds == f
    tr = ~te
    tr_cl = np.unique(cc[tr])
    rng = np.random.default_rng(seed + f)
    perm = rng.permutation(tr_cl)
    n_sel = max(1, int(round(SEL_FRAC * len(tr_cl))))
    sel_cl = set(int(x) for x in perm[:n_sel])
    is_sel = np.array([int(x) in sel_cl for x in cc])
    return (tr & ~is_sel), (tr & is_sel), te


def mean_ap(y, s, v):
    vals = []
    for i in range(y.shape[1]):
        m = v[:, i]
        if m.sum() < 50 or y[m, i].sum() < 5:
            continue
        vals.append(average_precision(y[m, i], s[m, i]))
    return float(np.mean(vals)) if vals else float("nan")


def train_tf_fold(X, Y, V, fit, sel, te, in_dim, win, d, device, seed=0, tag="",
                  epochs=EPOCHS):
    """One outer fold of one transformer arm; epoch selected on `sel`, scores on `te`."""
    torch.manual_seed(seed)
    m = CausalSitHead(in_dim, win, d=d, n_out=Y.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    pw = torch.full((Y.shape[1],), POS_WEIGHT, device=device)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    Xf = torch.from_numpy(np.ascontiguousarray(X[fit], dtype=np.float32))
    Yf = torch.from_numpy(np.ascontiguousarray(Y[fit], dtype=np.float32))
    Vf = torch.from_numpy(np.ascontiguousarray(V[fit], dtype=np.float32))
    n = Xf.shape[0]
    g = torch.Generator().manual_seed(seed)
    Ysel, Vsel = Y[sel].astype(np.int64), V[sel].astype(bool)
    best = (-np.inf, None, 0)
    curve = []
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, 1024):
            j = perm[b:b + 1024]
            vb = Vf[j].to(device)
            if float(vb.sum()) == 0:
                continue
            xb = Xf[j].to(device).view(len(j), win, in_dim)
            loss = (lossf(m(xb), Yf[j].to(device)) * vb).sum() / vb.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        a = mean_ap(Ysel, predict_sit_head(m, X[sel], in_dim=in_dim, device=device), Vsel)
        curve.append(round(float(a), 5))
        if np.isfinite(a) and a > best[0]:
            best = (a, predict_sit_head(m, X[te], in_dim=in_dim, device=device), ep + 1)
    log(f"    {tag}d={d} win={win} fit {int(fit.sum()):,} epoch*={best[2]}/{epochs} "
        f"sel_mAP={best[0]:.5f} curve={curve}")
    return best[1], best[2], curve


def train_ridge_fold(X, Y, V, fit, sel, te, tag=""):
    best = (-np.inf, None)
    for lam in LAMBDAS:
        a = mean_ap(Y[sel].astype(np.int64),
                    ridge_scores(X[fit], Y[fit], V[fit], X[sel], lam=lam),
                    V[sel].astype(bool))
        if np.isfinite(a) and a > best[0]:
            best = (a, lam)
    lam = best[1] if best[1] is not None else 1.0
    log(f"    {tag}ridge fit {int(fit.sum()):,} lam*={lam:g} sel_mAP={best[0]:.5f}")
    return ridge_scores(X[fit], Y[fit], V[fit], X[te], lam=lam), lam


# --------------------------------------------------------------------------- #
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--substrate", required=True)
    ap_.add_argument("--out", default="results_temporal.json")
    ap_.add_argument("--n-boot", type=int, default=2000)
    ap_.add_argument("--strata-n-boot", type=int, default=500)
    ap_.add_argument("--device", default=None)
    ap_.add_argument("--max-clips", type=int, default=0,
                     help="SMOKE ONLY: keep the first N clips. Never used for a quoted number.")
    a = ap_.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_scores = Path(a.out).with_suffix(".scores.npz")

    z = np.load(a.substrate)
    keep = slice(None)
    if a.max_clips:
        keep = np.flatnonzero(z["clip_cluster"] < a.max_clips)
        log(f"⚠️  SMOKE MODE: first {a.max_clips} clips only -- NOT a quotable run")
    F = z["F"][keep]
    Yi = z["Y"][keep].astype(np.int64)
    Y = z["Y"][keep].astype(np.float32)
    E = z["E"][keep]
    cc = z["clip_cluster"][keep]
    sits = [str(s) for s in z["situations"]]
    z = {"F": F, "Y": z["Y"][keep], "V": z["V"][keep], "E": E,
         "clip_cluster": cc, "cache_tag": z["cache_tag"][keep], "t": z["t"][keep],
         "situations": z["situations"]}
    st, en = clip_runs(cc)
    folds = cluster_folds(cc, 2, seed=0)
    log(f"substrate {F.shape[0]:,} rows x {F.shape[1]}  {len(st)} clips  device={device}")

    # ---- the COMMON row set: full in-clip history at the LONGEST window ------
    _, hist_ok = causal_window(np.zeros((len(cc), 1), np.float32), st, en, WIN_MAX)
    V = z["V"].astype(bool) & hist_ok[:, None]
    Vf = V.astype(np.float32)
    log(f"common scorable rows (full {WIN_MAX}-frame history): {int(hist_ok.sum()):,} "
        f"of {len(cc):,} ({100*hist_ok.mean():.1f}%)")
    counts = {}
    for i, s in enumerate(sits):
        counts[s] = {"scorable": int(V[:, i].sum()), "pos": int(Yi[V[:, i], i].sum()),
                     "base_rate": round(float(Yi[V[:, i], i].mean()), 6),
                     "clusters_with_a_positive":
                         int(len(np.unique(cc[V[:, i] & (Yi[:, i] > 0)])))}
        log(f"  {s}: scorable {counts[s]['scorable']:,} pos {counts[s]['pos']:,} "
            f"base {counts[s]['base_rate']:.5f} "
            f"clusters+ {counts[s]['clusters_with_a_positive']}")

    # ---- the clip-level permutation that defines NEG_FEAT (B4 verbatim) -----
    rng = np.random.default_rng(0)
    perm_clip = rng.permutation(len(st))
    shuf_rows = np.arange(len(cc))
    for i, (s0, e0) in enumerate(zip(st, en)):
        j = perm_clip[i]
        n = min(e0 - s0, en[j] - st[j])
        shuf_rows[s0:s0 + n] = st[j] + np.arange(n)
        if e0 - s0 > n:
            shuf_rows[s0 + n:e0] = st[j] + n - 1

    splits = [fold_splits(cc, folds, f, seed=0) for f in (0, 1)]
    scores: dict[str, np.ndarray] = {}
    spec: dict[str, dict] = {}
    # per-(fold, shuf) cache of the motion source, which is the expensive part
    _mot_cache: dict[tuple, np.ndarray] = {}

    def motion_source(shuf):
        """Delta_1 of the 2048-d readout -- the substrate the MOTION basis is fitted on."""
        key = ("mot", bool(shuf))
        if key not in _mot_cache:
            src = F[shuf_rows] if shuf else F
            d1, ok = temporal_difference(src, st, en, k=1)
            log(f"    motion source built (shuf={shuf}): {int(ok.sum()):,} rows with a valid delta")
            _mot_cache.clear()
            _mot_cache[key] = d1
        return _mot_cache[key]

    def build_features(rep, win, fold, shuf, reparam=False):
        """`rep` = list of (basis, rank); windowed features fitted on this fold's FIT rows only."""
        fit_rows = np.flatnonzero(splits[fold][0])
        blocks = []
        for basis, r in rep:
            if basis == "ego":
                # ⛔ PRIVILEGED POWER CONTROL ONLY -- never a deployable arm.
                # ⚠️ FIXED after the 2026-08-03 run: `shuf` must permute the ego block too.
                # The first version ignored it, so `NEG_FEAT__CPOS_*` was BYTE-IDENTICAL to the
                # real CPOS arm and its "null" reported the real ego lift (2.019/2.050/2.357 on
                # lane_change) instead of chance. It never touched any vision arm's null and never
                # touched the C-POS predicate, which reads `paired_vs_reference` -- but the
                # `paired_vs_own_null` column of the CPOS rows in that run is degenerate by
                # construction and must not be read.
                blocks.append(np.asarray(E[shuf_rows] if shuf else E, dtype=np.float32))
                continue
            if basis == "egofut":
                # ⛔⛔ THE ORACLE PROBE (PRE_REGISTRATION.md Sec 4 C-POS). It reads the ego state
                # over (t, t+r] -- i.e. the FUTURE, which is precisely the window the label's
                # detectors integrate. It MUST separate; if it does not, these rows cannot be
                # separated by anything and no verdict may be issued. It is unusable as a model
                # in every other respect and is built here, locally, rather than promoted into
                # `stack/`, so that a future-reading window can never be imported by accident.
                blocks.append(forward_stack(E[shuf_rows] if shuf else E, st, en, int(r)))
                continue
            src = (F[shuf_rows] if shuf else F) if basis == "app" else motion_source(shuf)
            mu, W = fit_pca(src, fit_rows, r, seed=0, device=device)
            blocks.append(project(src, mu, W))
        P = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, 1)
        X, _ = causal_window(P, st, en, win)
        if reparam:
            X = diff_reparam(X, win, P.shape[1])
        return X, int(P.shape[1])

    def run_arm(name, kind, rep, win, budget=None, shuf=False, epochs=EPOCHS,
                reparam=False):
        t0 = time.time()
        out = np.zeros_like(Y, dtype=np.float32)
        chosen, curves, in_dim, d = [], [], None, None
        for f in (0, 1):
            X, in_dim = build_features(rep, win, f, shuf, reparam=reparam)
            fit, sel, te = splits[f]
            if kind == "tf":
                if d is None:
                    d = width_for_param_budget(budget, in_dim, win, Y.shape[1])
                s_te, ep, curve = train_tf_fold(X, Y, Vf, fit, sel, te, in_dim, win, d,
                                                device, seed=0, tag=f"{name} ", epochs=epochs)
                chosen.append(ep)
                curves.append(curve)
            else:
                s_te, lam = train_ridge_fold(X, Y, Vf, fit, sel, te, tag=f"{name} ")
                chosen.append(lam)
            out[te] = s_te
            del X
        common = {"rep": [list(x) for x in rep], "win": win,
                  "history_s": round((win - 1) / HZ, 2), "in_dim": in_dim,
                  "flat_dim": in_dim * win, "shuffled_features": shuf,
                  "diff_reparam": reparam}
        if kind == "tf":
            spec[name] = {**common, "kind": "CausalSitHead", "d": d,
                          "params_per_head": head_param_count(in_dim, win, d, Y.shape[1]),
                          "budget": budget, "epochs_trained": epochs,
                          "epochs_selected": chosen, "sel_curve": curves}
        else:
            spec[name] = {**common, "kind": "closed-form ridge",
                          "params_per_head": ridge_param_count(in_dim * win,
                                                               per_situation=False),
                          "lambda_selected": chosen}
        scores[name] = out
        log(f"  {name}: {spec[name]['params_per_head']:,} params/head, "
            f"flat {spec[name]['flat_dim']}, history {common['history_s']}s "
            f"({time.time()-t0:.0f}s)")
        np.savez_compressed(out_scores, clip_cluster=cc, y=z["Y"], valid=V.astype(np.uint8),
                            ego=E, cache_tag=z["cache_tag"], folds=folds, t=z["t"],
                            situations=np.array(sits), **scores)

    # ------------------------------------------------------------------ ladder
    # (name, kind, representation, win, budget, epochs, reparam)
    # GROUP A -- WINDOW at MATCHED FLAT DIM 128 => 129 params/head, EXACTLY the banked floor's
    #            capacity. Anything that moves here is the window and nothing else.
    ladder = [
        ("ridge_app128_w1", "ridge", [("app", 128)], 1, None, 0, False),
        ("ridge_app64_w2", "ridge", [("app", 64)], 2, None, 0, False),
        ("ridge_app32_w4", "ridge", [("app", 32)], 4, None, 0, False),
        (REFERENCE, "ridge", [("app", 16)], 8, None, 0, False),
        ("ridge_app8_w16", "ridge", [("app", 8)], 16, None, 0, False),
        ("ridge_app4_w32", "ridge", [("app", 4)], 32, None, 0, False),
        # GROUP B -- WINDOW at FIXED RANK 16 (capacity is allowed to grow with the window)
        ("ridge_app16_w1", "ridge", [("app", 16)], 1, None, 0, False),
        ("ridge_app16_w4", "ridge", [("app", 16)], 4, None, 0, False),
        ("ridge_app16_w16", "ridge", [("app", 16)], 16, None, 0, False),
        ("ridge_app16_w32", "ridge", [("app", 16)], 32, None, 0, False),
        # GROUP C -- the MOTION SUBSPACE at the reference's own flat dim / window
        ("ridge_mot16_w8", "ridge", [("mot", 16)], 8, None, 0, False),
        ("ridge_app8mot8_w8", "ridge", [("app", 8), ("mot", 8)], 8, None, 0, False),
        ("ridge_app16mot16_w8", "ridge", [("app", 16), ("mot", 16)], 8, None, 0, False),
        ("ridge_mot16_w32", "ridge", [("mot", 16)], 32, None, 0, False),
        # THE INVARIANCE CONTROL -- an exactly invertible remap of the reference's window
        ("ridge_app16_w8_diffparam", "ridge", [("app", 16)], 8, None, 0, True),
        # ⛔ C-POS, THE POWER CONTROL. PRIVILEGED (ego), NOT A DEPLOYABLE. Must separate above the
        #    reference or this study is UNPOWERED and no verdict may be issued on these rows.
        ("CPOS_ego_w8", "ridge", [("ego", 3)], 8, None, 0, False),
        ("CPOS_ego_w32", "ridge", [("ego", 3)], 32, None, 0, False),
        # ⛔⛔ the ORACLE probe: ego over the FUTURE 3.0 s -- the label's own evidence window.
        ("CPOS_ORACLE_egofuture30", "ridge", [("egofut", 30)], 1, None, 0, False),
        # GROUP D -- the DEPLOYED transformer budget, so the finding is not a ridge artefact
        ("tf_app16_w8_d128", "tf", [("app", 16)], 8, TF_BUDGET, EPOCHS, False),
        ("tf_app16_w32", "tf", [("app", 16)], 32, TF_BUDGET, EPOCHS, False),
        ("tf_app8mot8_w8", "tf", [("app", 8), ("mot", 8)], 8, TF_BUDGET, EPOCHS, False),
        ("tf_app16_w8_diffparam", "tf", [("app", 16)], 8, TF_BUDGET, EPOCHS, True),
    ]
    order = [n for n, *_ in ladder]

    # ---- STAGE 0: the MECHANISM diagnostic behind H-T2 ----------------------
    # ⭐ This is measured BEFORE any AP, and it can refute H-T2 on its own. H-T2 claims the
    # appearance PCA truncates the motion subspace away. That is a statement about SUBSPACES and it
    # is directly checkable: how much of the temporal-difference variance survives a projection onto
    # the appearance basis the deployed arm actually uses? If almost all of it does, then rank-16
    # appearance PCA is NOT discarding motion and no motion-basis arm can be recovering anything.
    log("STAGE 0 - subspace diagnostic (does appearance PCA-16 discard the motion subspace?)")

    def _var_explained(X, rows, mu, W, chunk=20000):
        """Fraction of ``X``'s variance ABOUT ITS OWN MEAN captured by span(W).

        ⚠️ ``mu`` must be the mean of ``X`` on ``rows`` — NOT the mean of whatever
        the basis was fitted on. Centring the temporal difference by the
        *appearance* mean was the first version of this diagnostic and it is
        wrong in a way that flatters the appearance basis enormously: the
        difference has mean ~0, so subtracting a large appearance mean makes the
        total variance dominated by ``||mu_appearance||^2``, a constant offset
        that the appearance basis captures almost perfectly by construction. It
        reported the appearance basis holding 0.9520 of the delta variance; the
        correctly centred figure is what this function now returns.
        """
        tot = 0.0
        cap = 0.0
        for b in range(0, rows.size, chunk):
            A = np.asarray(X[rows[b:b + chunk]], dtype=np.float32) - mu
            tot += float(np.einsum("ij,ij->", A, A))
            P = A @ W
            cap += float(np.einsum("ij,ij->", P, P))
        return float(cap / max(tot, 1e-12))

    def _mean_of(X, rows, chunk=20000):
        acc = np.zeros(X.shape[1], np.float64)
        for b in range(0, rows.size, chunk):
            acc += np.asarray(X[rows[b:b + chunk]], dtype=np.float64).sum(0)
        return (acc / max(rows.size, 1)).astype(np.float32)[None, :]

    fit0 = np.flatnonzero(splits[0][0])
    d1_all = motion_source(False)
    _, d1_ok = temporal_difference(np.zeros((len(cc), 1), np.float32), st, en, k=1)
    fit0_d = fit0[d1_ok[fit0]]
    subspace = {}
    mu_F = _mean_of(F, fit0)
    mu_D = _mean_of(d1_all, fit0_d)
    for r in (16, 64):
        _, Wa = fit_pca(F, fit0, r, seed=0, device=device)
        _, Wm = fit_pca(d1_all, fit0_d, r, seed=0, device=device)
        cos = np.linalg.svd(Wa.T @ Wm, compute_uv=False)
        subspace[f"rank_{r}"] = {
            "_centring": "every fraction is variance ABOUT THE OWN MEAN of the matrix measured",
            "appearance_var_in_APPEARANCE_basis": round(_var_explained(F, fit0, mu_F, Wa), 5),
            "appearance_var_in_MOTION_basis": round(_var_explained(F, fit0, mu_F, Wm), 5),
            "DELTA_var_in_APPEARANCE_basis": round(_var_explained(d1_all, fit0_d, mu_D, Wa), 5),
            "DELTA_var_in_MOTION_basis": round(_var_explained(d1_all, fit0_d, mu_D, Wm), 5),
            "principal_angle_cosines": [round(float(x), 4) for x in cos],
            "mean_principal_cosine": round(float(cos.mean()), 5),
            "n_fit_rows": int(fit0.size), "n_fit_rows_with_delta": int(fit0_d.size)}
        q = subspace[f"rank_{r}"]
        log(f"  rank {r}: DELTA variance retained -- appearance basis "
            f"{q['DELTA_var_in_APPEARANCE_basis']:.4f} vs motion basis "
            f"{q['DELTA_var_in_MOTION_basis']:.4f}; appearance variance retained -- "
            f"appearance {q['appearance_var_in_APPEARANCE_basis']:.4f} vs motion "
            f"{q['appearance_var_in_MOTION_basis']:.4f}; mean principal cos "
            f"{q['mean_principal_cosine']:.4f}")
    del d1_all
    _mot_cache.clear()

    # ---- STAGE 1: THE PERMUTED-FEATURE NULL, FITTED AND SCORED FIRST --------
    log("STAGE 1 - NEG_FEAT (image features permuted ACROSS clips), fitted FIRST")
    for name, kind, rep, win, b, e, rp in ladder:
        run_arm("NEG_FEAT__" + name, kind, rep, win, budget=b, shuf=True,
                epochs=(e or EPOCHS), reparam=rp)
    null_check = {}
    for i, s in enumerate(sits):
        m = V[:, i]
        yv = Yi[m, i]
        null_check[s] = {n[len("NEG_FEAT__"):]: round(ap_lift(yv, scores[n][m, i]), 4)
                         for n in scores if n.startswith("NEG_FEAT__")}
        log(f"  NULL {s} ap-lift: " + " ".join(f"{k}={v:.3f}" for k, v in null_check[s].items()))

    # ---- STAGE 2: the real ladder ------------------------------------------
    log("STAGE 2 - the real ladder")
    for name, kind, rep, win, b, e, rp in ladder:
        run_arm(name, kind, rep, win, budget=b, shuf=False, epochs=(e or EPOCHS), reparam=rp)

    # ---- STAGE 3: NEG_LABEL -------------------------------------------------
    log("STAGE 3 - NEG_LABEL (labels permuted across whole CLUSTERS)")
    neg_label = {}
    for i, s in enumerate(sits):
        m = V[:, i]
        y_perm = permute_labels_by_cluster(Yi[:, i], cc, seed=1)[m]
        neg_label[s] = {n: round(ap_lift(y_perm, scores[n][m, i]), 5)
                        for n in (REFERENCE, "ridge_app16_w32", "ridge_mot16_w8")}
        log(f"  NEG_LABEL {s}: {neg_label[s]}")

    res = {"_what": "Is the situation classifier limited by MISSING TEMPORAL CONTENT?",
           "_hypotheses": {"H-T1": "window length", "H-T2": "motion subspace",
                           "H-T3": "explicit parameterisation"},
           "_inference_inputs": "CAMERA ONLY (frozen v1 encoder latents). No ego channel.",
           "substrate": str(a.substrate),
           "substrate_meta": json.loads(Path(a.substrate).with_suffix(".meta.json").read_text()),
           "protocol": {"win_max_common_rows": WIN_MAX, "epochs": EPOCHS,
                        "sel_frac": SEL_FRAC, "outer_folds": 2, "fold_unit": "clip cluster",
                        "pca": "appearance AND motion bases fitted on the FIT rows of each fold",
                        "estimator": "paired_ap_episode_cluster_bootstrap (taniteval draws)",
                        "n_boot": a.n_boot, "strata_n_boot": a.strata_n_boot,
                        "reference_arm": REFERENCE},
           "row_counts": counts,
           "arms": spec,
           "controls": {"NEG_FEAT_ap_lift": null_check, "NEG_LABEL_ap_lift": neg_label,
                        "H_T2_SUBSPACE_DIAGNOSTIC": subspace},
           "per_situation": {}}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ---- STAGE 4: AP with intervals, vs OWN NULL and vs the REFERENCE -------
    log("STAGE 4 - AP / AP-lift, paired vs own null and vs the reference arm")
    for i, s in enumerate(sits):
        m = V[:, i]
        yv = Yi[m, i]
        eid = cc[m]
        ones = np.ones(yv.size, bool)
        ref = scores[REFERENCE][m, i].astype(np.float64)
        row = {"n_scorable": int(m.sum()), "n_pos": int(yv.sum()),
               "base_rate": round(float(yv.mean()), 6),
               "n_clusters": int(len(np.unique(eid))),
               "n_clusters_with_a_positive": int(len(np.unique(eid[yv > 0]))),
               "arms": {}}
        for n in order:
            sc = scores[n][m, i].astype(np.float64)
            r_ = ap_episode_cluster_bootstrap(yv, sc, eid, n_boot=a.n_boot, lift=True)
            r_["ap"] = round(average_precision(yv, sc), 5)
            r_["params_per_head"] = spec[n]["params_per_head"]
            r_["flat_dim"] = spec[n]["flat_dim"]
            r_["win"] = spec[n]["win"]
            r_["history_s"] = spec[n]["history_s"]
            r_["n_nonfinite"] = int((~np.isfinite(sc)).sum())
            r_["op_5pct"] = precision_recall_at_budget(yv, sc, ones)
            r_["paired_vs_own_null"] = paired_ap_episode_cluster_bootstrap(
                yv, sc, scores["NEG_FEAT__" + n][m, i].astype(np.float64), eid,
                n_boot=a.n_boot, lift=True)
            r_["paired_vs_reference"] = (
                None if n == REFERENCE else
                paired_ap_episode_cluster_bootstrap(yv, sc, ref, eid,
                                                    n_boot=a.n_boot, lift=True))
            row["arms"][n] = r_
            pn = r_["paired_vs_own_null"]
            pr = r_["paired_vs_reference"]
            log(f"  {s} {n:>26}: AP={r_['ap']:.5f} lift={r_['point']:.3f} "
                f"P@5%={r_['op_5pct']['precision']:.4f} R@5%={r_['op_5pct']['recall']:.4f} "
                f"(fires {r_['op_5pct']['n_alarm']}/{r_['op_5pct']['n_pos']} true) "
                f"vs-null {pn['delta']:+.3f}{'*' if pn['separated'] else ' '} "
                + ("REFERENCE" if pr is None else
                   f"vs-ref {pr['delta']:+.3f} [{pr['lo']:+.3f},{pr['hi']:+.3f}]"
                   f"{' SEP' if pr['separated'] else ''}"))
        # ⛔ the PRIVILEGED power controls are excluded from every deployable ranking
        depl = [n for n in order if not n.startswith("CPOS_")]
        peak = max(depl, key=lambda n: row["arms"][n]["point"])
        row["peak_arm"] = peak
        beats = [n for n in depl
                 if n != REFERENCE
                 and row["arms"][n]["paired_vs_reference"]["separated"]
                 and row["arms"][n]["paired_vs_reference"]["delta"] > 0]
        row["arms_separating_ABOVE_reference"] = beats
        # THE STUDY'S RESOLUTION: the widest paired-vs-reference CI half-width actually achieved.
        # PRE_REGISTRATION.md Sec 7 -- "a `separated: false` is UNPOWERED, not refuted" -- so a null
        # may only be reported together with the effect size it WOULD have detected.
        halfw = [row["arms"][n]["paired_vs_reference"]["ci95"] for n in depl if n != REFERENCE]
        row["mde_ap_lift_widest_ci95"] = round(float(np.max(halfw)), 5)
        row["mde_ap_lift_median_ci95"] = round(float(np.median(halfw)), 5)
        # C-POS: did ANY arm, even a privileged one, separate above the reference on these rows?
        cpos = {n: row["arms"][n]["paired_vs_reference"] for n in order if n.startswith("CPOS_")}
        row["C_POS_separates_above_reference"] = {
            n: bool(v["separated"] and v["delta"] > 0) for n, v in cpos.items()}
        # ⭐ the pre-registered predicate keys on the ORACLE specifically. A causal ego arm is
        # allowed to lose -- that is a finding about ego, not about the study's power. Only the
        # oracle, which reads the label's own evidence window, MUST win.
        cpos_ok = bool(row["C_POS_separates_above_reference"].get(
            "CPOS_ORACLE_egofuture30", False))
        # C-POW (PRE_REGISTRATION.md Sec 4): < 40 positive CLUSTERS => no verdict for this situation.
        powered = row["n_clusters_with_a_positive"] >= 40
        if not powered:
            row["VERDICT"] = "UNDERPOWERED_C_POW"
        elif beats:
            row["VERDICT"] = "CONFIRMED"
        elif not cpos_ok:
            row["VERDICT"] = "INDETERMINATE_C_POS_FAILED"
        else:
            row["VERDICT"] = "NO_EFFECT_ABOVE_MDE"
        row["_verdict_note"] = (
            f"C-POW {row['n_clusters_with_a_positive']} positive clusters "
            f"(bar 40); C-POS separates above reference: {cpos_ok}; "
            f"study MDE on ap-lift (widest paired CI half-width) "
            f"{row['mde_ap_lift_widest_ci95']}, reference lift {row['arms'][REFERENCE]['point']}")
        res["per_situation"][s] = row
        log(f"  {s}: PEAK = {peak} | separating above {REFERENCE}: "
            f"{beats or 'NONE'} | C-POS {cpos_ok} | MDE {row['mde_ap_lift_widest_ci95']} "
            f"-> {row['VERDICT']}")
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ---- STAGE 5: the INVARIANCE control -----------------------------------
    log("STAGE 5 - parameterisation-invariance control")
    inv = {}
    for i, s in enumerate(sits):
        m = V[:, i]
        yv = Yi[m, i]
        d = paired_ap_episode_cluster_bootstrap(
            yv, scores["ridge_app16_w8_diffparam"][m, i].astype(np.float64),
            scores[REFERENCE][m, i].astype(np.float64), cc[m], n_boot=a.n_boot, lift=True)
        inv[s] = {"_what": ("an EXACTLY invertible linear remap of the reference's own window; "
                            "a non-zero delta here is standardisation + L2 penalty geometry, "
                            "NOT information, and bounds how much of any motion-channel gain "
                            "is attributable to the reparameterisation alone"),
                  "delta_ap_lift": d}
        log(f"  {s}: diffparam - reference = {d['delta']:+.4f} "
            f"[{d['lo']:+.4f},{d['hi']:+.4f}]{' SEPARATED' if d['separated'] else ' (null)'}")
    res["controls"]["PARAMETERISATION_INVARIANCE"] = inv

    # ---- STAGE 6: the FOUR FAMILIES, peak arm vs the reference --------------
    log("STAGE 6 - the four families (peak arm vs the reference arm)")
    bundle = ScoreBundle(situations=np.array(sits), arms=order, y=z["Y"],
                         valid=V.astype(np.uint8), clip_cluster=cc,
                         scores={k: v for k, v in scores.items() if not k.startswith("NEG_")},
                         ego=E, source=str(a.substrate))
    res["four_families"] = {}
    for j, s in enumerate(sits):
        peak = res["per_situation"][s]["peak_arm"]
        res["four_families"][s] = four_family_report(
            bundle, s, fused=scores[peak][:, j].astype(np.float64),
            baseline=scores[REFERENCE][:, j].astype(np.float64),
            baseline_name=REFERENCE, fused_name=f"PEAK::{peak}",
            n_boot=a.n_boot, strata_n_boot=a.strata_n_boot)
        log(f"  families {s}: peak={peak} done")
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")

    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {a.out} and {out_scores}")


if __name__ == "__main__":
    main()
