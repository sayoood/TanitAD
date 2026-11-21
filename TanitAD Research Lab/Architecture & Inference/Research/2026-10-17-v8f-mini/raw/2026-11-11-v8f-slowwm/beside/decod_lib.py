"""SLOWWM beside B1 (decodability), protocol fixed in ``../BESIDE_RULES_SLOWWM.md`` (frozen, sha256 in BESIDE_RULES_SLOWWM.sha256). Tier T0, CPU. Never deciding.

Library. Imports, never edits: ``read/lp_slowwm.py`` (``prepare_arm_read`` / ``load_checked`` / ``basis_clip_z1``: the ONE route by which a basis or adapter arm reaches z1), ``ladder_read/lr_lib.py`` (``k_of``,
``load_arm``, frames), ``wmact/wmact_lib.py`` (the episode-cluster bootstrap draws, same rng calls as ``taniteval.ci._draws``: B 2,000, seed 0), the yaw-rate tangent mask of ``taniteval.four_families``
(``MIN_DS_MPS``: a step below 0.5 m/s has no path tangent).

B1 as written: z1 = the level-1 latent the arm's own WM rolls out from the c1 context (``stack.rollout(1, z_now, logged actions)``, context 1) read at step k = ``lr_lib.k_of(1, 2.0, cum_1)`` = 10 steps
(2.0 s at 0.2 s per step); targets = measured speed and measured yaw rate at NOW + 2.0 s. NOW = window frame g = t - 15 (the programme's knot 0). Probe = ridge on train-standardised z1, alpha from
10**linspace(-3, 3, 13) by 5-fold CV grouped by clip on the TRAIN rollouts, refit on all train rollouts, scored once on eval-141. R2 and MAE carry the episode-cluster bootstrap interval (episodes of
eval-141 resampled with the probe held fixed: it answers "another draw of episodes" only).

Implementation choices where the rules are silent (none changes a rule): windows are those whose 16 frames are all real (``ok.all(1)``, ``plantest_lib.load_latents``, the set WMACT and the planner read use);
ridge = minimise ||y - Xw - b||^2 + alpha ||w||^2 on standardised X (alpha not scaled by n); the speed target is column 3 of ``native_tracks`` at row 2*frame (10 Hz rows, 5 Hz cache frames; the planner's own
``v0``); the yaw-rate target is the cached measured yaw rate (action channel 1) at frame g + k; a window is masked from the yaw-rate target unless the speeds at frames g+k and g+k-1 both exceed
``MIN_DS_MPS`` (the pair rule ``_seq_geometry`` applies to a yaw rate), or when the target is not finite."""
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.normpath(os.path.join(HERE, "..", ".."))
CODE_ROOT = "F:/TanitAD-scratch/slowwm/tree"
os.environ.setdefault("LADDER_READ_CODE_ROOT", CODE_ROOT)
os.environ.setdefault("WMACT_CODE_ROOT", os.environ["LADDER_READ_CODE_ROOT"])
sys.path.insert(0, os.path.join(THEORY, "slowwm", "read"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import lp_slowwm as S  # noqa: E402   (puts ladder_read, ladder_plantest, wmact, plantest on sys.path; asserts tanitad is under the code root)
import wmact_lib as W  # noqa: E402
from taniteval.four_families import MIN_DS_MPS  # noqa: E402

LR, LP, SH, P = S.LR, S.LP, S.SH, S.P
import tanitad  # noqa: E402
assert os.path.abspath(tanitad.__file__).replace("\\", "/").lower().startswith(CODE_ROOT.lower()), tanitad.__file__

HORIZON_S = 2.0
ALPHAS = 10.0 ** np.linspace(-3, 3, 13)
NFOLD, B_BOOT, SEED = 5, W.B_BOOT, W.SEED
MIN_RAM_GB = 8.0
OUT_DIR = os.environ.get("BESIDE_OUT", "F:/TanitAD-scratch/slowwm_beside_build/b1b2/out")
CONTROLS_JSON = os.path.join(HERE, "CONTROLS_B1B2.json")


def k_steps():
    """Level-1 steps of 2.0 s (``lr_lib.k_of`` with the level-1 cumulative stride 1)."""
    k, on_grid = LR.k_of(1, HORIZON_S, 1)
    assert on_grid and k == 10, (k, on_grid)
    return k


def wait_ram(min_gb=MIN_RAM_GB):
    if LR.free_gb() < min_gb:
        raise MemoryError(f"free RAM {LR.free_gb():.1f} GB < {min_gb} GB: nothing is killed, rerun later")


# ----------------------------------------------------------------------------------------------------------------------------- z1 and targets
@torch.no_grad()
def rolled_z1(wm, z_now, A_raw, k, chunk=512):
    """z1 rolled out from the c1 context: ``stack.rollout(1, z_now, scaled logged actions)`` at step k. z_now [N, D] (the window's knot-0 latent), A_raw [N, 16, 2] the raw logged (a, yaw rate) -> [N, D]."""
    A_sc = wm.scale_actions(torch.as_tensor(np.asarray(A_raw)).float())[:, :-1]
    out = []
    for s in range(0, z_now.shape[0], chunk):
        out.append(wm.stack.rollout(1, z_now[s:s + chunk], A_sc[s:s + chunk, :k])[:, k - 1])
    return torch.cat(out).float()


def load_tracks():
    return torch.load(LP.TRACKS, weights_only=False)["tracks"]


def targets_at(tracks, eid, t, A_raw, k):
    """Measured speed and yaw rate at NOW + k steps. Returns y_speed, y_yaw, m_speed, m_yaw (masks). The yaw-rate mask drops windows without a heading tangent (RETR-2026-09-26-YAWMASK)."""
    n = len(eid)
    sp, sp_prev = np.full(n, np.nan), np.full(n, np.nan)
    for i, (e, ti) in enumerate(zip(eid, t)):
        r = tracks.get(str(e))
        if r is None:
            continue
        r = np.asarray(r)
        f = int(ti) - 15 + k
        if 0 <= f and 2 * f < len(r):
            sp[i] = r[2 * f, 3]
        if 1 <= f and 2 * (f - 1) < len(r):
            sp_prev[i] = r[2 * (f - 1), 3]
    yaw = np.asarray(A_raw, np.float64)[:, k, 1]
    m_speed = np.isfinite(sp)
    m_yaw = np.isfinite(yaw) & np.isfinite(sp) & np.isfinite(sp_prev) & (sp > MIN_DS_MPS) & (sp_prev > MIN_DS_MPS)
    return sp, yaw, m_speed, m_yaw


def arrays_from_ctx(ctx, tracks, k=None):
    """The B1 arrays of one window set: rolled z1 [n, D], the targets and masks, the clip ids."""
    k = k or k_steps()
    Z = rolled_z1(ctx.wm, ctx.Zw[:, 0], ctx.A, k).numpy().astype(np.float64)
    ys, yy, ms, my = targets_at(tracks, ctx.eid, ctx.t, ctx.A, k)
    return dict(Z=Z, eid=np.asarray([str(e) for e in ctx.eid]), y_speed=ys, y_yaw=yy, m_speed=ms, m_yaw=my, n=len(ctx.eid), k=k)


def subset(a, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (a["n"],) else v) for k, v in a.items()} | {"n": len(idx)}


def pseudo_split(a):
    """SMOKE / control split of ONE window set into two episode-disjoint halves (even / odd rank of the sorted clip ids). Not the protocol's train set: for machinery checks only."""
    u = np.unique(a["eid"])
    even = set(u[0::2])
    m = np.array([e in even for e in a["eid"]])
    return subset(a, np.flatnonzero(m)), subset(a, np.flatnonzero(~m))


# ----------------------------------------------------------------------------------------------------------------------------- the probe
def fold_of(groups, nfold=NFOLD, seed=0):
    """Fold per row, grouped by clip: clips shuffled (seed 0) then dealt round robin."""
    u = np.unique(groups)
    rank = np.empty(len(u), int)
    rank[np.random.default_rng(seed).permutation(len(u))] = np.arange(len(u))
    return rank[np.searchsorted(u, groups)] % nfold


def _ridge_path(X, y, alphas):
    """Weights for every alpha, intercept-free after centring: X [n, d], y [n] -> (w [A, d], xm, ym). One eigendecomposition."""
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    lam, V = np.linalg.eigh(Xc.T @ Xc)
    q = V.T @ (Xc.T @ yc)
    return np.stack([V @ (q / (np.maximum(lam, 0.0) + a)) for a in alphas]), xm, ym


def fit_probe(Xtr, ytr, gtr, alphas=ALPHAS, nfold=NFOLD, seed=0):
    """Standardise with TRAIN mean / std, choose alpha by ``nfold``-fold CV grouped by clip on the train rows only (least summed squared error; ties go to the smaller alpha), refit on all train rows."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (Xtr - mu) / sd
    fold = fold_of(gtr, nfold, seed)
    sse = np.zeros(len(alphas))
    for f in range(nfold):
        tr, va = fold != f, fold == f
        w, xm, ym = _ridge_path(Z[tr], ytr[tr], alphas)
        pred = (Z[va] - xm) @ w.T + ym
        sse += ((pred - ytr[va, None]) ** 2).sum(0)
    ia = int(np.argmin(sse))
    w, xm, ym = _ridge_path(Z, ytr, [alphas[ia]])
    return dict(mu=mu, sd=sd, w=w[0], xm=xm, ym=ym, alpha=float(alphas[ia]), alpha_index=ia, cv_sse=[float(s) for s in sse])


def predict(pr, X):
    return ((X - pr["mu"]) / pr["sd"] - pr["xm"]) @ pr["w"] + pr["ym"]


def boot_for(eid):
    uniq, Wt = W.boot_weights(np.asarray(eid))
    return uniq, Wt


def score(y, p, eid, boot=None):
    """R2 (about the eval mean) and MAE with the episode-cluster bootstrap (B 2,000, seed 0, the probe fixed): per-episode sums, ratio statistics, percentile interval."""
    uniq, Wt = boot or boot_for(eid)
    e = y - p
    es = W.ep_sums({"n": np.ones(len(y)), "sy": y, "syy": y * y, "sse": e * e, "sae": np.abs(e)}, eid, uniq)

    def r2(w):
        n = w @ es["n"]
        return 1.0 - (w @ es["sse"]) / (w @ es["syy"] - (w @ es["sy"]) ** 2 / n)

    one = np.ones(len(uniq))
    rb = 1.0 - (Wt @ es["sse"]) / (Wt @ es["syy"] - (Wt @ es["sy"]) ** 2 / (Wt @ es["n"]))
    mb = (Wt @ es["sae"]) / (Wt @ es["n"])
    row = lambda pt, b: dict(point=float(pt), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))
    return {"R2": row(r2(one), rb), "MAE": row(es["sae"].sum() / es["n"].sum(), mb), "n_windows": int(len(y)), "n_episodes": int(len(uniq)),
            "y_std": float(np.std(y)), "y_mean": float(np.mean(y))}


def probe_target(Xtr, ytr, gtr, Xev, yev, eev, boot=None):
    pr = fit_probe(Xtr, ytr, gtr)
    p = predict(pr, Xev)
    return dict(alpha=pr["alpha"], cv_sse=pr["cv_sse"], eval=score(yev, p, eev, boot), _pred=p, _probe=pr)


TARGETS = (("speed_m_s", "y_speed", "m_speed"), ("yaw_rate_rad_s", "y_yaw", "m_yaw"))


def b1_rows(tr, ev):
    """The B1 rows of one arm from its train and eval arrays: per target, the masked windows, the CV alpha, eval R2 / MAE with intervals."""
    out = {}
    for name, yk, mk in TARGETS:
        mt, me = tr[mk], ev[mk]
        r = probe_target(tr["Z"][mt], tr[yk][mt], tr["eid"][mt], ev["Z"][me], ev[yk][me], ev["eid"][me])
        r.pop("_pred"), r.pop("_probe")
        r.update(n_train=int(mt.sum()), n_train_all=int(len(mt)), n_masked_train=int((~mt).sum()), n_eval_all=int(len(me)), n_masked_eval=int((~me).sum()))
        out[name] = r
    return out


# ----------------------------------------------------------------------------------------------------------------------------- B1 controls (i) and (ii) on one arm's arrays
def planted_w(D, Xtr, seed=0):
    """Fixed w (seed 0), scaled so std(z . w) = 1 on the train rows."""
    w = np.random.default_rng(seed).standard_normal(D)
    return w / np.std(Xtr @ w)


def control_permuted(tr, ev, target="y_speed", mask="m_speed", mutate=None):
    """(i) Targets of the train rows shuffled across train clips (a derangement over clips: every row takes the target of a row of ANOTHER clip, ``hjepa.donor_indices`` seed 0); eval R2 (against the TRUE eval targets) <= 0.02.
    ``mutate='score_on_train'``: the deliberate regression that scores the probe on its own train rows."""
    from tanitad.models.hjepa import donor_indices
    mt, me = tr[mask], ev[mask]
    X, y, g = tr["Z"][mt], tr[target][mt], tr["eid"][mt]
    ep = torch.as_tensor(np.unique(g, return_inverse=True)[1])
    perm = donor_indices(len(y), episode_ids=ep, seed=0).numpy()
    assert (g[perm] != g).all(), "the permutation must cross clips"
    pr = fit_probe(X, y[perm], g)
    if mutate == "score_on_train":
        r2 = score(y[perm], predict(pr, X), g)["R2"]["point"]
    else:
        r2 = score(ev[target][me], predict(pr, ev["Z"][me]), ev["eid"][me])["R2"]["point"]
    return dict(r2=float(r2), bar=0.02, ok=bool(r2 <= 0.02), alpha=pr["alpha"])


def control_planted(tr, ev, mutate=None, seed=0):
    """(ii) y = z1 . w + 0.1 eps (fixed w, seed 0; eps independent on train and eval) on the rolled z1; eval R2 within 0.02 of the planted value var(z . w) / (var(z . w) + 0.1^2) on the eval rows.
    ``mutate='misaligned'``: the deliberate regression that rolls the train targets by half the rows (every row meets another clip's target)."""
    g = np.random.default_rng(seed + 1)
    Xtr, Xev = tr["Z"], ev["Z"]
    w = planted_w(Xtr.shape[1], Xtr, seed)
    ytr, yev = Xtr @ w + 0.1 * g.standard_normal(len(Xtr)), Xev @ w + 0.1 * g.standard_normal(len(Xev))
    planted = float(np.var(Xev @ w) / (np.var(Xev @ w) + 0.1 ** 2))
    pr = fit_probe(Xtr, np.roll(ytr, len(ytr) // 2) if mutate == "misaligned" else ytr, tr["eid"])
    got = score(yev, predict(pr, Xev), ev["eid"])["R2"]["point"]
    return dict(r2=float(got), planted=planted, abs_diff=float(abs(got - planted)), tol=0.02, ok=bool(abs(got - planted) <= 0.02), alpha=pr["alpha"])


def controls_on_arrays(tr, ev):
    """(i) for both targets and (ii) on one arm's arrays, plus each red arm: the row is refused unless every control reads its known value and every red arm goes red."""
    c = {"i_permuted_speed": control_permuted(tr, ev, "y_speed", "m_speed"), "i_permuted_yaw": control_permuted(tr, ev, "y_yaw", "m_yaw"), "ii_planted": control_planted(tr, ev)}
    red = {"i_score_on_train": control_permuted(tr, ev, "y_speed", "m_speed", "score_on_train"), "ii_misaligned": control_planted(tr, ev, "misaligned")}
    c["red_arms_go_red"] = bool((not red["i_score_on_train"]["ok"]) and (not red["ii_misaligned"]["ok"]))
    c["red_arms"] = red
    c["pass"] = bool(c["i_permuted_speed"]["ok"] and c["i_permuted_yaw"]["ok"] and c["ii_planted"]["ok"] and c["red_arms_go_red"])
    return c


def controls_passed():
    """True only when ``CONTROLS_B1B2.json`` says all_pass: the runners write no arm row otherwise."""
    import json
    try:
        return bool(json.load(open(CONTROLS_JSON, encoding="utf-8")).get("all_pass") is True)
    except (OSError, ValueError):
        return False


def stamp(t0):
    return round(time.time() - t0, 1)
