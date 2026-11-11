"""SLOWWM beside protocol B3 (BESIDE_RULES_SLOWWM.md, frozen sha256 d222f9d4...13d0): the planner whose level-1 goal cost is restricted to SLOW's top-64 mu directions. Tier T1-kin planner form. CPU.

The rule, as written (section B3, "The mapping"): J-bar = the mean over 4,096 train frames (seed 0) of the Jacobian dz1/du, taking the columns of the top-64 mu directions (basis columns 0-63), a 128 x 64 matrix. S = its column space by SVD (singular values
above 1e-6 * s_max). P_S = the orthogonal projector onto S. The level-1 goal cost becomes  mean over last_k of ||P_S (z_pred - goal)||^2; everything else in the FLAT1 level-1 planner is unchanged.

Implementation choices the rule leaves open (each is recorded in the output npz and in README_B3.md; none changes a number the rule fixes):
  * z1 of a K = 3 arm is enc1 over a window of THREE consecutive projected frames, so dz1/du has a per-frame part for each of the three frames. J-bar differentiates z1 with respect to a SHIFT of the chosen u columns applied to all
    frames of the window together (the sustained shift a slow direction stands for). The variant that shifts the newest frame only is stored as ``jbar_current`` and its principal angles to S are a beside number; it is not used by the planner.
  * "4,096 train frames": frames with ok = True of the train cache (``slow_basis.py cache --split train``), drawn without replacement by ``numpy.random.default_rng(0).choice`` over the flattened ok-frame index of the clips in sorted-name order.
    A frame's window is the K frames ending at it, indices clamped at the clip start (the padding ``_z1_post`` itself applies).
  * Jacobians are taken in float64 on a deep copy of the arm's own ``wm`` (``wm.z1_from_basis`` is the only z1 route called); the planner keeps float32.
  * The wrapper is installed as an instance attribute on the stack built for the arm (``st.goal_cost``) and removed on exit; no library file is edited.
Imports only: lp_slowwm (read package), slowwm_hook, slow_basis, lp_lib, plantest_lib."""
import contextlib
import copy
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.normpath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("LADDER_READ_CODE_ROOT", "F:/TanitAD-scratch/slowwm/tree")
os.environ.setdefault("WMACT_CODE_ROOT", os.environ["LADDER_READ_CODE_ROOT"])
for _p in (os.path.join(THEORY, "slowwm"), os.path.join(THEORY, "slowwm", "read"), os.path.join(THEORY, "ladder_plantest")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import lp_slowwm as S  # noqa: E402  (puts the stack and the gates on sys.path; asserts nothing itself, lp_lib / lr_lib assert the code root)

LR, LP, SH = S.LR, S.LP, S.SH
SB = SH.SB
import tanitad  # noqa: E402

_ROOT = os.environ["LADDER_READ_CODE_ROOT"].replace("\\", "/").lower()
assert os.path.abspath(tanitad.__file__).replace("\\", "/").lower().startswith(_ROOT), tanitad.__file__

RULES_SHA = "d222f9d4f360b3239fe3054a2d7c0da80032af873ade932455c0b03c8d4013d0"
N_FRAMES, SEED, TOPN = 4096, 0, 64
SV_TOL = 1e-6
FD_STEP, FD_REL_TOL = 1e-3, 1e-3
D_U = 256
TIER = "T1-kin planner form (B3, BESIDE_RULES_SLOWWM.md)"


def sha256_file(p, blk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(blk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------------- train frames (J-bar's inputs)
def window_size(wm):
    return int(wm.enc1.window) if getattr(wm, "enc1", None) is not None else 1


def sample_train_frames(cache_dir, basis_sha, n=N_FRAMES, seed=SEED):
    """Picks [(clip, j)] of ``n`` ok frames of the train cache, plus the manifest. The manifest's basis sha must equal ``basis_sha``; every clip file read is hashed against the manifest."""
    mp = os.path.join(cache_dir, "BASIS_MANIFEST.json")
    man = json.load(open(mp, encoding="utf-8"))
    if man.get("basis_sha256") != basis_sha:
        raise SH.BasisMismatch(f"train cache {cache_dir} was built with basis {str(man.get('basis_sha256'))[:12]}, the arm's basis is {basis_sha[:12]}")
    clips = sorted(man["files_sha256"])
    oks = []
    for c in clips:
        with np.load(os.path.join(cache_dir, f"{c}.npz"), allow_pickle=False) as z:
            oks.append(np.asarray(z["ok"], bool))
    cnt = np.array([int(o.sum()) for o in oks])
    total = int(cnt.sum())
    if total < n:
        raise ValueError(f"train cache holds {total} ok frames, fewer than the {n} requested")
    flat = np.sort(np.random.default_rng(seed).choice(total, size=n, replace=False))
    cum = np.concatenate([[0], np.cumsum(cnt)])
    picks = []
    for f in flat:
        ci = int(np.searchsorted(cum, f, side="right") - 1)
        picks.append((clips[ci], int(np.nonzero(oks[ci])[0][int(f - cum[ci])])))
    return picks, man, mp


def train_windows(cache_dir, basis_sha, K, n=N_FRAMES, seed=SEED):
    """u windows [n, K, 256] float32 for the sampled frames (indices clamped at the clip start), and the provenance dict."""
    picks, man, mp = sample_train_frames(cache_dir, basis_sha, n, seed)
    out = np.zeros((len(picks), K, D_U), np.float32)
    cache, used = {}, {}
    for i, (c, j) in enumerate(picks):
        if c not in cache:
            p = os.path.join(cache_dir, f"{c}.npz")
            if sha256_file(p) != man["files_sha256"][c]:
                raise ValueError(f"{p}: sha256 differs from the train cache manifest")
            with np.load(p, allow_pickle=False) as z:
                cache[c] = np.asarray(z["zbasis"], np.float32)
            used[c] = man["files_sha256"][c]
        u = cache[c]
        out[i] = u[np.clip(np.arange(j - K + 1, j + 1), 0, len(u) - 1)]
    prov = dict(cache_dir=cache_dir, manifest_sha256=sha256_file(mp), picks_sha256=sha256_json(picks), n_frames=len(picks), seed=seed, n_clips_used=len(used), used_files_verified_against_manifest=True)
    return out, prov


# ------------------------------------------------------------------------------------------------------------- the Jacobian
def wm64_of(wm):
    """A float64 deep copy of the arm's wm: ``z1_from_basis`` is the one route differentiated."""
    w = copy.deepcopy(wm).double().eval()
    for p in w.parameters():
        p.requires_grad_(False)
    return w


def _shift_inputs(u, sh, joint):
    """u [b, K, 256], sh [b, 256] -> u + sh on every window frame (joint) or on the newest frame only."""
    if joint:
        return u + sh[:, None, :]
    m = torch.zeros(u.shape[1], 1, dtype=u.dtype)
    m[-1] = 1.0
    return u + m[None] * sh[:, None, :]


def jbar_autograd(wm64, u_win, cols, joint=True, chunk=256):
    """Mean over the windows of d z1[last] / d (u columns ``cols``): [D, len(cols)] float64, by reverse-mode autograd."""
    cols = np.asarray(cols)
    nc = len(cols)
    E = torch.zeros(nc, D_U, dtype=torch.float64)
    E[torch.arange(nc), torch.as_tensor(cols)] = 1.0
    jsum, n = None, len(u_win)
    for s in range(0, n, chunk):
        u = torch.as_tensor(u_win[s:s + chunk], dtype=torch.float64)
        x = torch.zeros(u.shape[0], nc, dtype=torch.float64, requires_grad=True)
        z = wm64.z1_from_basis(_shift_inputs(u, x @ E, joint))[:, -1]
        if jsum is None:
            jsum = np.zeros((z.shape[1], nc))
        for d in range(z.shape[1]):
            g, = torch.autograd.grad(z[:, d].sum(), x, retain_graph=d < z.shape[1] - 1)
            jsum[d] += g.sum(0).numpy()
    return jsum / n


def jbar_fd(wm64, u_win, cols, h=FD_STEP, joint=True, chunk=512):
    """The same quantity by central finite difference (step ``h``), forward passes only: independent of autograd."""
    cols = np.asarray(cols)
    jsum, n = None, len(u_win)
    with torch.no_grad():
        for s in range(0, n, chunk):
            u = torch.as_tensor(u_win[s:s + chunk], dtype=torch.float64)
            for i, c in enumerate(cols):
                e = torch.zeros(u.shape[0], D_U, dtype=torch.float64)
                e[:, int(c)] = h
                zp = wm64.z1_from_basis(_shift_inputs(u, e, joint))[:, -1]
                zm = wm64.z1_from_basis(_shift_inputs(u, -e, joint))[:, -1]
                if jsum is None:
                    jsum = np.zeros((zp.shape[1], len(cols)))
                jsum[:, i] += ((zp - zm) / (2 * h)).sum(0).numpy()
    return jsum / n


def projector_from_jbar(jbar, tol=SV_TOL):
    """S = column space of ``jbar`` by SVD (singular values above tol * s_max); P_S = U_r U_r^T."""
    U, s, _ = np.linalg.svd(np.asarray(jbar, np.float64), full_matrices=False)
    if not s[0] > 0:
        raise ValueError("J-bar is the zero matrix: no direction to project on")
    r = int((s > tol * s[0]).sum())
    Ur = U[:, :r]
    return dict(U=Ur, s=s, rank=r, P=Ur @ Ur.T)


def principal_angles_deg(U, V):
    """Angles (degrees, ascending) between the column spaces of the orthonormal bases U [D, r] and V [D, q]: min(r, q) of them."""
    sv = np.linalg.svd(U.T @ V, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, 0.0, 1.0)))


def beside_numbers(U, z_eval, topn=TOPN):
    """rank(S); the share of the centred eval-z1 variance inside S; the principal angles between S and the top-``topn`` PCA directions of eval z1 (and, for scale, the variance those PCA directions hold)."""
    z = np.asarray(z_eval, np.float64)
    zc = z - z.mean(0)
    cov = zc.T @ zc / len(zc)
    w, Q = np.linalg.eigh(cov)
    o = np.argsort(w)[::-1]
    V = Q[:, o[:topn]]
    tot = float(np.trace(cov))
    ang = principal_angles_deg(U, V)
    return dict(rank=int(U.shape[1]), variance_share_in_S=float(np.trace(U.T @ cov @ U) / tot), variance_share_top64_pca=float(w[o[:topn]].sum() / tot), principal_angles_deg=ang,
                angle_min=float(ang.min()), angle_median=float(np.median(ang)), angle_max=float(ang.max()), n_eval_frames=int(len(z)), z1_dim=int(z.shape[1]))


# ------------------------------------------------------------------------------------------------------------- the cost wrapper
def make_goal_cost(orig, P):
    """``orig`` = the stack's bound ``goal_cost``; P = [D, D] float32 tensor. Level 1: mean over last_k of ||P (z_pred - goal)||^2 (the library's own last_k rule); other levels: the library's cost."""
    def goal_cost(z_pred, goal, level, last_k=None):
        if level != 1:
            return orig(z_pred, goal, level, last_k)
        k = last_k or 2
        k = min(k, z_pred.shape[-2])
        diff = z_pred[..., -k:, :] - goal.unsqueeze(-2)
        return (diff @ P).pow(2).sum(dim=-1).mean(dim=-1)
    return goal_cost


@contextlib.contextmanager
def restricted_cost(st, P):
    """Install the restricted level-1 goal cost on the stack instance ``st`` (never on the class), remove it on exit. P: numpy or torch [D, D]."""
    if "goal_cost" in st.__dict__:
        raise RuntimeError("goal_cost is already wrapped on this stack")
    Pt = torch.as_tensor(np.asarray(P), dtype=torch.float32).contiguous()
    st.goal_cost = make_goal_cost(type(st).goal_cost.__get__(st), Pt)
    try:
        yield st
    finally:
        del st.goal_cost


# ------------------------------------------------------------------------------------------------------------- the builder
def build_arm(name, run_dir, basis_npz, ubasis_npz, train_cache, out_npz, cols=None, n=N_FRAMES, seed=SEED, frames_dir=None):
    """J-bar / S / P_S of one SLOW arm -> ``out_npz`` (atomic), with the sha256 of every input. Refuses a non-slow arm. Returns the result dict."""
    t0 = time.time()
    if S.kind_of(name) != "slow":
        raise ValueError(f"B3 runs on SLOW arms only, not {name}")
    L = S.load_checked(name, run_dir, basis_npz, ubasis_npz, frames_dir)
    B = L["B"]
    cols = SH.top_mu_columns(B, TOPN) if cols is None else np.asarray(cols)
    wm = L["model"].wm
    K = window_size(wm)
    uw, prov = train_windows(train_cache, B["sha"], K, n, seed)
    w64 = wm64_of(wm)
    jb = jbar_autograd(w64, uw, cols, joint=True)
    jc = jbar_autograd(w64, uw, cols, joint=False)
    pr = projector_from_jbar(jb)
    z_eval = np.concatenate([v.numpy() for v in SH.clip_z1_basis(wm, L["fr"]).values()])
    bn = beside_numbers(pr["U"], z_eval)
    jc_pr = projector_from_jbar(jc)
    ang_cur = principal_angles_deg(pr["U"], jc_pr["U"])
    inputs = dict(basis_npz=sha256_file(basis_npz), ubasis_npz=sha256_file(ubasis_npz), ckpt_last=sha256_file(os.path.join(run_dir, "ckpt_last.pt")), selfcheck=sha256_file(os.path.join(run_dir, "selfcheck.json")),
                  argv=sha256_file(os.path.join(run_dir, "argv.txt")), train_manifest=prov["manifest_sha256"], train_picks=prov["picks_sha256"], rules=RULES_SHA)
    res = dict(name=name, kind=L["kind"], basis_sha=B["sha"], cols=cols, window_K=K, n_frames=int(len(uw)), seed=int(seed), jbar=jb, jbar_current=jc, U=pr["U"], s=pr["s"], P=pr["P"], rank=pr["rank"],
               rank_current_variant=jc_pr["rank"], angles_joint_vs_current_deg=ang_cur, inputs_sha256=inputs, train_provenance=prov, beside=bn, sv_tol=SV_TOL, wall_s=time.time() - t0, tier=TIER)
    save_projector(out_npz, res)
    return res


def save_projector(path, res):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = {k: res[k] for k in ("name", "kind", "basis_sha", "window_K", "n_frames", "seed", "rank", "rank_current_variant", "sv_tol", "wall_s", "tier") if k in res}
    meta.update(inputs_sha256=res["inputs_sha256"], train_provenance=res["train_provenance"], beside={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res["beside"].items()},
                cols_first_last=[int(res["cols"][0]), int(res["cols"][-1])])
    tmp = path + ".tmp.npz"
    np.savez(tmp, cols=np.asarray(res["cols"]), jbar=res["jbar"], jbar_current=res["jbar_current"], U=res["U"], s=res["s"], P=res["P"], angles_joint_vs_current_deg=res["angles_joint_vs_current_deg"],
             principal_angles_deg=res["beside"]["principal_angles_deg"], meta_json=np.array(json.dumps(meta, sort_keys=True, default=str)))
    os.replace(tmp, path)


def load_projector(path):
    """The builder's npz -> dict(P float64 [D, D], U, s, rank, meta, ...)."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: np.asarray(z[k]) for k in z.files if k != "meta_json"}
        meta = json.loads(str(z["meta_json"]))
    d["meta"], d["rank"] = meta, int(meta["rank"])
    return d


def verify_projector(proj, ctx, run_dir):
    """The projector must belong to the arm being read: its basis sha and the sha256 of the run's checkpoint must equal the ones it was built from. Raises ValueError."""
    m = proj["meta"]
    bad = []
    if m["basis_sha"] != ctx.basis_sha:
        bad.append(f"projector basis {m['basis_sha'][:12]} vs the arm's {ctx.basis_sha[:12]}")
    ck = sha256_file(os.path.join(run_dir, "ckpt_last.pt"))
    if m["inputs_sha256"]["ckpt_last"] != ck:
        bad.append(f"projector built from checkpoint {m['inputs_sha256']['ckpt_last'][:12]}, the arm's is {ck[:12]}")
    if m["inputs_sha256"]["rules"] != RULES_SHA:
        bad.append("projector carries another BESIDE_RULES sha256")
    if proj["P"].shape != (ctx.D, ctx.D):
        bad.append(f"projector shape {proj['P'].shape} vs z1 width {ctx.D}")
    if bad:
        raise ValueError("; ".join(bad))
    return proj
