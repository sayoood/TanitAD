"""AMEND_HWMPLAN_5 (RA-L1): the RA pass, post hoc on a base's banked plan files. Study-local (MM ruling M2); the code of record is imported unchanged and nothing under it is written. CPU only, OMP_NUM_THREADS=2.
Tier T0 (oracle goal): a world-model diagnostic, never driving performance.

THE OPERATION (section 2.3). Per window: variable = level 1's 12 acceleration knots (scaled), start HOLD_c (S = 1 candidate), yaw channel = the base's level-1 yaw (a1_raw), cost
    (ra(F10(z0, [a_acc, yaw_b]) - z0) - target)^2,        the library's st._optimize at level-1 settings (AdamW lr 0.1, <= 90 iterations, patience 10, rel_tol 1e-4, weight decay 0, clip to B1).
The best iterate's acceleration replaces knots 1-10 of the base's plan_raw; every other knot (yaw 1-30, acceleration 11-30), a1_raw and cost1 are the base's bytes. The pass draws no random number.
Targets (--ra-l1):
    off     no pass: the output is the base, bit for bit (RA-OFF; C-OFF5)
    l2      target = ra(F10(z0, [src_acc, yaw_b]) - z0), src_acc = level 2's primitives knots 1-10 (`l2_prim_raw`; P / P23 only)      RA-L2
    l1      the same with src_acc = the base's level-1 plan knots 1-10 (`a1_raw`)                                                       RA-L1
    neg     the committed arm's target mirrored about HOLD_c's own read: 2 ra(F10(z0, [HOLD_c, yaw_b]) - z0) - target                    RA-NEG
    shuf    the committed arm with W permuted (RandomState(0)) on both sides (target and plan)                                          RA-SHUF
    oracle  target = ra(z10 - z0), the TRUE frame-10 latent; yaw at HOLD_c. A ceiling, never a plan                                      RA-ORACLE
  --mean-src {l1,l2}  (with --ra-l1 off): MEAN-src, HOLD_c's acceleration + ONE uniform shift to the source's mean over knots 1-10, clipped to B1; no world model, no RA.
Default `--ra-l1 off` disables the pass, so every argv NAMES its value (section 8.1); the wrapper refuses an argv that does not.
usage (OMP_NUM_THREADS=2):  python ra_l1.py plan --stage train|score --arm RAL1 --base A0 --chassis s0 --ra-l1 l1 --ra-src l1 --mean-src none --seeds 0 1 --out <NEW DIR> [--beside]
Refusals (RefusedRun / NoBaseSolver): unnamed flag, declared != argv, declared != built, readout / windows / checkpoint / base-file sha256 mismatch, a non-empty out folder, a wrong thread count, SCORE without BASE_SOLVER_AMEND5.md,
a base that is not the base solver (except RA-L2 on P under B3, --beside)."""
import argparse, dataclasses, json, os, shutil, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amend5_stage as ST
sys.path.insert(0, ST.LONPLAN_BUILD)
import lonplan_common as C                  # sets CUDA_VISIBLE_DEVICES, puts the code of record on sys.path
import numpy as np
import torch
import hwm_plan as H
import hwm_data as D
import plantest_lib as P
import io

MODES = ("off", "l2", "l1", "neg", "shuf", "oracle")
SRCS = ("l1", "l2", "none")
KS = 10                                     # RA reads the level-1 state after 10 actions; knots 1-10 are the pass's output
ACC = float(H.ACC_SCALE)
SRC_KEY = {"l1": "a1_raw", "l2": "l2_prim_raw"}
KEYS_DATA = ("plan_raw", "a1_raw", "cost1", "iters1", "idx", "decode_resid", "l2_prim_raw")      # the base's data keys an `off` output must carry bit for bit


class RefusedRun(Exception):
    """The wrapper refuses to plan."""


# ---------------------------------------------------------------------------------------------- readout (C-RA-SHA)
@dataclasses.dataclass
class Readout:
    mu: torch.Tensor
    sd: torch.Tensor
    W: torch.Tensor
    ym: torch.Tensor
    Wshuf: torch.Tensor
    perm: np.ndarray
    sha256: str
    lam: float
    path: str

    def read(self, Dz, shuf=False):
        """ra(D) = ((D - mu) / sd) . W + ym  (scaled acceleration, mean over the 10 transitions); Dz (..., 128) -> (...)."""
        Wv = self.Wshuf if shuf else self.W
        return (((Dz - self.mu) / self.sd) @ Wv)[..., 0] + self.ym[0]


def load_readout(path, k, sha=None):
    """C-RA-SHA: the file's sha256 must equal section 2.1's value (hashed from the bytes that are then parsed); one changed byte refuses."""
    want = ST.RA_SHA[k] if sha is None else sha
    raw = open(path, "rb").read()
    import hashlib
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise RefusedRun(f"readout {path} sha256 {got[:12]} != the frozen {want[:12]}")
    R = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    ra = R["RA"]
    mu, sd, W, ym = (torch.tensor(np.asarray(ra[x], np.float64), dtype=torch.float32) for x in ("mu", "sd", "W", "ym"))
    perm = np.random.RandomState(0).permutation(W.shape[0])
    return Readout(mu=mu, sd=sd, W=W, ym=ym, Wshuf=W[torch.as_tensor(perm)], perm=perm, sha256=got, lam=float(R["lambda"]), path=path)


# ---------------------------------------------------------------------------------------------- C-RA-FIT: rebuild the fit set from the TRAIN latents
def rebuild_ra_fit(LT, scale_actions, readout_lam, K=KS):
    """The RA fit set exactly as amend1_kstep.py builds it (frame table, K 10, finite actions) and the ridge refit at the frozen lambda. LT = the latents_train_R35_s{k}.pt dict.
    Returns dict(n_pairs, n_clips, clips, mu, sd, W, ym, Dr, Yf)."""
    tag = [t for t in LT["Z1"]][0]
    Zw, Aw, okw = LT["Z1"][tag].float(), scale_actions(LT["A_raw"].float()), LT["ok"].bool()
    tw, eidw = np.asarray(LT["t"]), list(LT["eid"])
    clips = sorted(set(eidw)); cix = {c: i for i, c in enumerate(clips)}
    tab = {}
    for w_ in range(Zw.shape[0]):
        for j in range(Zw.shape[1]):
            if bool(okw[w_, j]):
                tab.setdefault((cix[eidw[w_]], int(tw[w_]) - (Zw.shape[1] - 1) + j), (Zw[w_, j], Aw[w_, j]))
    ZT, ZK, AB = [], [], []
    for ci in range(len(clips)):
        fr = sorted(f for (c, f) in tab if c == ci)
        pos = {int(f): i for i, f in enumerate(fr)}
        Zc = torch.stack([tab[(ci, f)][0] for f in fr]); Ac = torch.stack([tab[(ci, f)][1] for f in fr])
        for i, f in enumerate(fr):
            f = int(f)
            if all((f + q) in pos for q in range(K + 1)):
                js = [pos[f + q] for q in range(K)]
                if torch.isfinite(Ac[js]).all():
                    ZT.append(Zc[i]); ZK.append(Zc[pos[f + K]]); AB.append(Ac[js].mean(0))
    Dr = (torch.stack(ZK) - torch.stack(ZT)).numpy().astype(np.float64)
    Yf = torch.stack(AB).numpy().astype(np.float64)[:, :1]
    return refit_ridge(Dr, Yf, readout_lam, clips)


def refit_ridge(Dr, Yf, lam, clips):
    mu2, sd2 = Dr.mean(0), Dr.std(0) + 1e-6
    Xs = (Dr - mu2) / sd2
    ym2 = Yf.mean(0)
    W2 = np.linalg.solve(Xs.T @ Xs + lam * len(Dr) * np.eye(Dr.shape[1]), Xs.T @ (Yf - ym2))
    return dict(n_pairs=int(len(Dr)), n_clips=len(clips), clips=clips, mu=mu2, sd=sd2, W=W2, ym=ym2, Dr=Dr, Yf=Yf)


def check_ra_fit(rebuilt, readout_np, score_eids, n_pairs_record=8306, n_clips_record=141, tol=1e-8):
    """C-RA-FIT: n_pairs / n_clips equal the record, the refit reproduces (mu, sd, W, ym) within tol, 0 TRAIN clips are SCORE episodes. readout_np: dict(mu, sd, W, ym) float64 arrays."""
    d = dict(n_pairs=rebuilt["n_pairs"], n_clips=rebuilt["n_clips"], n_pairs_record=n_pairs_record, n_clips_record=n_clips_record,
             max_abs_dW=float(np.abs(rebuilt["W"] - readout_np["W"]).max()), max_abs_dmu=float(np.abs(rebuilt["mu"] - readout_np["mu"]).max()),
             max_abs_dsd=float(np.abs(rebuilt["sd"] - readout_np["sd"]).max()), abs_dym=float(np.abs(rebuilt["ym"] - readout_np["ym"]).max()),
             train_clips_in_score_episodes=len(set(rebuilt["clips"]) & set(score_eids)), tol=tol)
    d["ok"] = bool(d["n_pairs"] == n_pairs_record and d["n_clips"] == n_clips_record and max(d["max_abs_dW"], d["max_abs_dmu"], d["max_abs_dsd"], d["abs_dym"]) <= tol and d["train_clips_in_score_episodes"] == 0)
    return d


# ---------------------------------------------------------------------------------------------- the context (model, priors, readout) and the pass
@dataclasses.dataclass
class Ctx:
    st: object
    h1: int
    lo1: torch.Tensor
    hi1: torch.Tensor
    ra: Readout
    hold_c: object                    # fn(A_raw0 (n, 2)) -> (n, >= h1, 2) scaled, clipped to B1
    ch1_scale: float
    n_iters: int = P.N_ITERS
    tag: str = ""


def load_ctx(k, readout_path=None, ckpt=None, latents=None):
    """The real context of chassis k: R35's checkpoint, its TRAIN priors (as LON-PLAN), the frozen readout."""
    c = ST.CH[k]
    ck, lat = ckpt or c["ck"], latents or c["lat"]
    m, cfg, _ = D.load_model(ck)
    wm, st = m.wm, m.wm.stack
    for p_ in wm.parameters():
        p_.requires_grad_(False)
    st.eval()
    Z, A = D.trainset_from_latents_pt(lat, c["tag"])
    D.fit_priors(st, wm, Z, A)
    geom = H.Geometry.from_stack(st, wm, goal_steps=None, stitch="decode")
    sol = H.SolverCfg(seed_mode="hold", lam=0.0, S=P.S_CAND, n_iters=P.N_ITERS, decode_lam=0.0, decode_iters=300, plan_ctx=1)
    planner = H.HWMPlanner(wm, geom, sol)
    assert geom.n_op == KS and geom.h1 == 12, (geom.n_op, geom.h1)
    ra = load_readout(readout_path or ST.readout_path(k), k)
    ctx = Ctx(st=st, h1=geom.h1, lo1=st.prior_lo_1, hi1=st.prior_hi_1, ra=ra, hold_c=planner.hold_c, ch1_scale=float(wm.ch1_scale), tag=c["tag"])
    ctx.wm = wm
    return ctx


def roll(st, z0, a, read_after=KS):
    """z0 (B, D); a (B, S, H, 2) scaled -> level-1 state after `read_after` actions (B, S, D): R35's level-1 rollout, context 1 (plan-ctx 1)."""
    z0e = z0.unsqueeze(1).expand(a.shape[0], a.shape[1], -1)
    return st.rollout(1, z0e, a)[:, :, read_after - 1]


def ra_pass(st, lo1, hi1, ra_fn, z0, acc0, yaw, target, n_iters=P.N_ITERS, read_after=KS, return_cost_fn=False):
    """The RA pass. z0 (B, D); acc0, yaw (B, h1) scaled; target (B,) scaled. -> (acc (B, h1) best iterate, cost (B,), n_iterations, hit_box (B,))."""
    base = torch.stack([acc0, yaw], -1).unsqueeze(1)                      # (B, 1, h1, 2): ONE candidate
    yawe = yaw.unsqueeze(1).unsqueeze(-1)

    def cost(a):
        a_eff = torch.cat([a[..., :1], yawe], -1)                         # only the acceleration channel is the variable
        r = ra_fn(roll(st, z0, a_eff, read_after) - z0.unsqueeze(1))
        return (r - target.unsqueeze(1)) ** 2

    if return_cost_fn:
        return cost, base
    with torch.enable_grad():
        best_a, best_cost, init_cost, hist, nit = st._optimize(cost, base, lr=float(P.LR[1]), n_iters=n_iters, patience=P.PATIENCE, rel_tol=P.REL_TOL, weight_decay=0.0, lo=lo1, hi=hi1, project=None)
    acc = best_a[:, 0, :, 0].detach()
    hit = ((acc <= lo1[0] + 0.0) | (acc >= hi1[0] - 0.0)).any(-1)
    return acc, best_cost[:, 0].detach(), int(nit), hit


def plan_arrays(ctx, base, win, mode, src="none", mean_src="none", mutate=None, built=None, batch=P.BATCH):
    """One base plan file (dict of arrays) -> the arm's plan file (dict). mode in MODES; mean_src l1|l2 with mode off gives MEAN-src. `built` (dict, filled in place) records what was actually executed
    (the C-L4-RA 'built' side). `mutate` (controls only): off_runs_pass | write12 | write_yaw | wrong_src | n_iters1.
    Reads: win['z'], win['A'] (z0 = frame 0, HOLD_c from a0, z10 for the oracle) and the base's plan arrays. Draws no random number."""
    built = built if built is not None else {}
    n = len(base["plan_raw"])
    assert win["z"].shape[0] == n and win["A"].shape[0] == n, (win["z"].shape, n)
    out = {k: np.array(v) if isinstance(v, np.ndarray) else v for k, v in base.items()}
    mode_run = "l1" if (mutate == "off_runs_pass" and mode == "off") else mode
    src_run = ("l1" if src in ("none", None) else src) if mutate == "off_runs_pass" else src
    if mutate == "wrong_src" and src_run in ("l1", "l2"):
        src_run = "l1" if src_run == "l2" else "l2"
    built.update(mode=mode, src="none", kind="off", mirror=False, perm=False, readout_sha256=ctx.ra.sha256, mean_src="none", src_key=None)
    if mode == "off" and mean_src == "none" and mode_run == "off":
        return out, built
    dt = ctx.ra.W.dtype
    z = torch.from_numpy(np.asarray(win["z"])).to(dt)
    A0 = np.asarray(win["A"], np.float64)[:, 0]
    h1 = ctx.h1
    hc_all = ctx.hold_c(A0)[:, :h1]                                          # (n, h1, 2) scaled, clipped
    a1 = np.asarray(base["a1_raw"], np.float64)
    yaw_b_all = torch.from_numpy(a1[:, :h1, 1] / ctx.ch1_scale).to(dt)
    if mean_src != "none":
        key = SRC_KEY[mean_src]
        built.update(kind="mean", mean_src=mean_src, src_key=key)
        hc_raw = hc_all[:, :KS, 0].numpy().astype(np.float64) * ACC
        flat = mean_assembly(np.asarray(base[key], np.float64)[:, :KS, 0], hc_raw, float(ctx.lo1[0]) * ACC, float(ctx.hi1[0]) * ACC)
        out["plan_raw"][:, :KS, 0] = flat
        return out, built
    built.update(kind="ra", mode=mode_run, src=(src_run if mode_run in ("l1", "l2", "neg", "shuf") else "none"), mirror=(mode_run == "neg"), perm=(mode_run == "shuf"))
    if mode_run in ("l1", "l2", "neg", "shuf"):
        key = SRC_KEY[src_run]
        built["src_key"] = key
        srcf = np.asarray(base[key], np.float64)
        if srcf.shape[1] < KS:
            raise RefusedRun(f"{key} has {srcf.shape[1]} knots, RA reads 10")
    acc_out = np.zeros((n, KS)); cost_out = np.zeros(n); hit_out = np.zeros(n, bool); it_out = np.zeros(n); tgt_out = np.zeros(n)
    full12 = None
    if mutate == "write12":
        full12 = np.zeros((n, h1))
    for b0 in range(0, n, batch):
        ii = slice(b0, min(n, b0 + batch))
        z0, hc, yaw_b = z[ii, 0], hc_all[ii], yaw_b_all[ii]
        use_shuf = mode_run == "shuf"
        ra_fn = (lambda Dz, _s=use_shuf: ctx.ra.read(Dz, shuf=_s))
        built["perm"] = bool(use_shuf and not torch.equal(ctx.ra.Wshuf, ctx.ra.W))
        with torch.no_grad():
            if mode_run == "oracle":
                target = ctx.ra.read(z[ii, KS] - z0)
                yaw_use = hc[:, :, 1]
            else:
                yaw_use = yaw_b
                srcacc = torch.from_numpy(a1[ii, :h1, 0] / ACC).to(dt)                       # knots 11-12 are never read (structural zero gradient)
                srcacc[:, :KS] = torch.from_numpy(srcf[ii, :KS, 0] / ACC).to(dt)
                target = ra_fn(roll(ctx.st, z0, torch.stack([srcacc, yaw_use], -1).unsqueeze(1))[:, 0] - z0)
                if mode_run == "neg":
                    r_h = ctx.ra.read(roll(ctx.st, z0, torch.stack([hc[:, :, 0], yaw_use], -1).unsqueeze(1))[:, 0] - z0)
                    target = 2 * r_h - target                                                    # the mirror about HOLD_c's own read
        ni = 1 if mutate == "n_iters1" else ctx.n_iters
        acc, cst, nit, hit = ra_pass(ctx.st, ctx.lo1, ctx.hi1, ra_fn, z0, hc[:, :, 0], yaw_use, target, n_iters=ni)
        acc_out[ii] = acc[:, :KS].numpy().astype(np.float64) * ACC
        cost_out[ii], hit_out[ii], it_out[ii], tgt_out[ii] = cst.numpy(), hit.numpy(), nit, target.numpy()
        if full12 is not None:
            full12[ii] = acc.numpy().astype(np.float64) * ACC
    out["plan_raw"][:, :KS, 0] = acc_out
    if mutate == "write12":
        out["plan_raw"][:, :h1, 0] = full12                                  # the deliberate defect: knots 11-12 written from the pass
    if mutate == "write_yaw":
        out["plan_raw"][:, :h1, 1] = hc_all[:, :, 1].numpy().astype(np.float64) * ctx.ch1_scale + 1e-3
    out.update(ra_acc_raw=acc_out, ra_cost=cost_out, ra_hit_box=hit_out, ra_iters=it_out, ra_target_scaled=tgt_out)
    return out, built


def mean_assembly(src_acc_raw, hc_acc_raw, lo_raw, hi_raw, shift_override=None):
    """MEAN-src: HOLD_c's acceleration (knots 1-10) + one uniform shift to the source's mean over knots 1-10, clipped to B1. Reads ONLY the source file's knots and HOLD_c (a0).
    shift_override (controls only): must equal the source-derived shift bit for bit, else RefusedRun (the guard of C-MEAN: a shift taken from logged accelerations is refused)."""
    shift = src_acc_raw.mean(1, keepdims=True) - hc_acc_raw.mean(1, keepdims=True)
    if shift_override is not None and not np.array_equal(np.asarray(shift_override), shift):
        raise RefusedRun("MEAN's shift does not come from the source file's knots (C-MEAN guard)")
    return np.clip(hc_acc_raw + shift, lo_raw, hi_raw)


# ---------------------------------------------------------------------------------------------- argv -> built (C-L4-RA) and the run
def _named(argv, flag):
    return any(t == flag or t.startswith(flag + "=") for t in argv)


def parse(argv):
    ap = argparse.ArgumentParser(prog="ra_l1 plan", allow_abbrev=False)
    ap.add_argument("--stage", required=True, choices=list(ST.STAGES)); ap.add_argument("--arm", required=True, choices=list(ST.ARMS)); ap.add_argument("--base", required=True, choices=list(ST.BASES))
    ap.add_argument("--chassis", required=True, choices=list(ST.CH)); ap.add_argument("--ra-l1", dest="ra_l1", choices=list(MODES)); ap.add_argument("--ra-src", dest="ra_src", choices=list(SRCS))
    ap.add_argument("--mean-src", dest="mean_src", choices=list(SRCS)); ap.add_argument("--seeds", nargs="+", type=int, default=[0]); ap.add_argument("--out", required=True)
    ap.add_argument("--beside", action="store_true")
    return ap.parse_args(list(argv))


def check_declared(argv, declared):
    """C-L4-RA, argv -> built (first half): every flag is NAMED (the default `off` would disable the pass silently) and the parsed arm equals the declared one. Returns the namespace; raises RefusedRun."""
    for f in ("--ra-l1", "--ra-src", "--mean-src"):
        if not _named(argv, f):
            raise RefusedRun(f"{f} is not named in the argv: its default would be a silent choice (--ra-l1 off disables the pass)")
    a = parse(argv)
    got = dict(arm=a.arm, mode=a.ra_l1, src=a.ra_src, mean_src=a.mean_src, base=a.base, stage=a.stage, chassis=a.chassis, mirror=(a.ra_l1 == "neg"), perm=(a.ra_l1 == "shuf"))
    for k in got:
        if got[k] != declared[k]:
            raise RefusedRun(f"declared {k} {declared[k]!r} != argv {got[k]!r}")
    if a.mean_src != "none" and a.ra_l1 != "off":
        raise RefusedRun("--mean-src is the no-world-model arm: it needs --ra-l1 off")
    if a.ra_l1 in ("l1", "l2") and a.ra_src != a.ra_l1:
        raise RefusedRun(f"--ra-l1 {a.ra_l1} needs --ra-src {a.ra_l1}")
    if a.ra_l1 in ("neg", "shuf") and a.ra_src not in ("l1", "l2"):
        raise RefusedRun(f"--ra-l1 {a.ra_l1} needs the committed arm's source (--ra-src l1|l2)")
    if "l2" in (a.ra_src, a.mean_src) and a.base == "A0":
        raise RefusedRun("RA-L2 / MEAN-L2 are not computable on A0 (no level-2 primitives for knots 1-10; MM ruling M3)")
    return a


def check_built(built, declared):
    """C-L4-RA, argv -> built (second half): what the code path actually executed equals the declared pass (source key, readout sha, mirror, permutation)."""
    want_key = ST.MEAN_SRC.get(declared["arm"]) if declared["kind"] == "mean" else (declared["src"] if declared["src"] in ("l1", "l2") else None)
    want_key = SRC_KEY.get(want_key) if want_key else None
    bad = []
    if built.get("src_key") != want_key:
        bad.append(f"source key {built.get('src_key')!r} != {want_key!r}")
    if declared["kind"] == "ra" and built.get("readout_sha256") != declared["readout_sha256"]:
        bad.append("readout sha256")
    if declared["kind"] == "ra" and (bool(built.get("mirror")) != declared["mirror"] or bool(built.get("perm")) != declared["perm"]):
        bad.append(f"mirror/perm built {built.get('mirror')}/{built.get('perm')} != declared {declared['mirror']}/{declared['perm']}")
    if declared["kind"] == "ra" and built.get("mode") != declared["mode"]:
        bad.append(f"mode {built.get('mode')!r} != {declared['mode']!r}")
    if declared["kind"] == "off" and built.get("kind") != "off":
        bad.append("an `off` arm ran a pass")
    if bad:
        raise RefusedRun("built != declared: " + "; ".join(bad))


def check_fresh_out(out):
    if os.path.exists(out):
        if not os.path.isdir(out):
            raise RefusedRun(f"{out} is a file")
        left = sorted(os.listdir(out))
        if left:
            raise RefusedRun(f"out folder {out} is not empty ({len(left)} entries: {left[:3]}); a pre-planted plans_*.npz is refused")


def check_stage_rules(a, declared, solver=None):
    """SCORE: refused without BASE_SOLVER_AMEND5.md (the `off` arm of C-OFF5 aside); the base must be the base solver, except RA-L2 on P beside under B3 (--beside)."""
    if a.stage == "train":
        if a.beside:
            raise RefusedRun("--beside is a SCORE notion")
        return None
    if declared["kind"] == "off" and declared["arm"] == "OFF":
        return None                                                          # RA-OFF: an identity copy, allowed before the base solver exists (C-OFF5 runs on base files only)
    bs = solver or ST.read_base_solver()                                     # raises NoBaseSolver
    if a.beside:
        if not (a.arm == "RAL2" and a.base == "P" and bs["branch"] == "B3"):
            raise RefusedRun("--beside is only RA-L2 on P's banked files under B3 (section 4)")
    elif a.base != bs["base"]:
        raise RefusedRun(f"base {a.base} is not the base solver {bs['base']} ({bs['branch']}) named in {bs['path']}")
    return bs


def verify_inputs(k):
    """C-IN (run side): checkpoint and TRAIN latents sha256-equal to the record (the readout and windows are checked where they are read)."""
    c = ST.CH[k]
    if C.sha256_file(c["ck"]) != c["cksha"]:
        raise RefusedRun("checkpoint sha256 != the record")
    if C.sha256_file(c["lat"])[:len(c["latsha"])] != c["latsha"]:
        raise RefusedRun("TRAIN latents sha256 != the record")


def run_arm(argv, declared, ctx=None, windows_src=None, windows_sha=None, enforce_threads=True, mutate=None, solver=None, base_plan_fn=None):
    """Plan one arm (all its seeds) into a fresh folder. ctx / base_plan_fn / windows_src: tests replace them; the real run builds them from the stage table."""
    a = check_declared(argv, declared)
    if mutate is not None and ctx is None:
        raise RefusedRun("mutation hooks are for the controls only")
    if enforce_threads and os.environ.get("OMP_NUM_THREADS") != str(C.THREADS):
        raise RefusedRun(f"OMP_NUM_THREADS must be {C.THREADS} (readback is bit-equal only at one thread count); it is {os.environ.get('OMP_NUM_THREADS')!r}")
    bs = check_stage_rules(a, declared, solver)
    check_fresh_out(a.out)
    k = a.chassis
    c = ST.CH[k]
    ws = windows_src or ST.windows_of(a.stage, k)
    wsha = windows_sha or (c["wsha"] if a.stage == "score" else ST.json.load(open(f"{ST.L0}/sel_{k}/lam_0/SOLVER.json", encoding="utf-8"))["windows_sha256"])
    got_w = C.sha256_file(ws)
    if got_w != wsha:
        raise RefusedRun(f"windows sha256 {got_w[:12]} != declared {wsha[:12]}")
    if ctx is None:
        verify_inputs(k)
        ctx = load_ctx(k)
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(ws, os.path.join(a.out, "windows.npz"))
    with np.load(ws, allow_pickle=False) as w:
        win = {x: w[x] for x in ("z", "A", "eid")}
    argv_sha = C.sha256_text(json.dumps(list(argv)))
    plans, base_files, builts = {}, {}, {}
    bpf = base_plan_fn or (lambda s: ST.base_plan(a.stage, a.base, k, s))
    for s in a.seeds:
        bp = bpf(s)
        base_files[os.path.basename(bp) + f"@s{s}"] = dict(path=bp, sha256=C.sha256_file(bp))
        rec = ST.json.load(open(os.path.join(os.path.dirname(bp), "SOLVER.json"), encoding="utf-8")) if os.path.exists(os.path.join(os.path.dirname(bp), "SOLVER.json")) else {}
        want = (rec.get("plans", {}).get(os.path.basename(bp)) or {}).get("sha256")
        if want and want != base_files[os.path.basename(bp) + f"@s{s}"]["sha256"]:
            raise RefusedRun(f"base plan file {bp} sha256 differs from the record in its SOLVER.json (C-IN)")
        with np.load(bp, allow_pickle=False) as z:
            base = {x: z[x] for x in z.files}
        if declared["kind"] != "off" and "a1_raw" not in base:
            raise RefusedRun(f"{bp} has no a1_raw")
        if "l2" in (declared["src"], declared["mean_src"]) and "l2_prim_raw" not in base:
            raise RefusedRun(f"{bp} has no l2_prim_raw: RA-L2 / MEAN-L2 need level 2's primitives (M4; on TRAIN the base is re-planned with --l2-image)")
        built = {}
        t0 = time.time()
        new, built = plan_arrays(ctx, base, win, a.ra_l1, a.ra_src, a.mean_src, mutate=mutate, built=built)
        check_built(built, declared)
        builts[f"s{s}"] = built
        new.update(ra_l1_planned=np.array(str(a.ra_l1) if a.mean_src == "none" else "mean"), ra_src_planned=np.array(str(a.ra_src if a.mean_src == "none" else a.mean_src)), ra_readout_sha256=np.array(ctx.ra.sha256),
                   ra_mirror=np.array(bool(built["mirror"])), ra_perm=np.array(bool(built["perm"])), ra_argv_sha256=np.array(argv_sha), ra_base=np.array(a.base),
                   ra_base_plan_sha256=np.array(base_files[os.path.basename(bp) + f"@s{s}"]["sha256"]), ra_wall_s=np.array(time.time() - t0))
        p = os.path.join(a.out, f"plans_hwm_oracle_s{s}.npz")
        np.savez(p, **new)
        plans[os.path.basename(p)] = dict(sha256=C.sha256_file(p), mode=str(new["ra_l1_planned"]))
        print("seed", s, "arm", a.arm, "wall", round(time.time() - t0, 1), "s", flush=True)
    sol = dict(kind="AMEND5_SOLVER_RA", argv=list(argv), argv_sha256=argv_sha, declared=declared, built=builts, ra_l1=a.ra_l1, ra_src=a.ra_src, mean_src=a.mean_src, readout=ctx.ra.path, readout_sha256=ctx.ra.sha256,
               base=a.base, beside=bool(a.beside), base_solver=(dict(base=bs["base"], branch=bs["branch"], sha256=bs["sha256"]) if bs else None), base_files=base_files, windows_sha256=wsha,
               ckpt_sha256=c["cksha"], solver="the library's st._optimize at level-1 settings: AdamW lr 0.1, <= 90 iterations, patience 10, rel_tol 1e-4, weight decay 0, clip to B1; S = 1 candidate from HOLD_c",
               code_sha256=C.code_of_record_sha256(), module_sha256=dict(ra_l1_py=C.sha256_file(os.path.abspath(__file__)), amend5_stage_py=C.sha256_file(os.path.join(HERE, "amend5_stage.py"))),
               omp_num_threads=os.environ.get("OMP_NUM_THREADS"), torch_threads=int(torch.get_num_threads()), plans=plans, tier="T0", evidence="MEASURED at run time",
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    C.dump_json(sol, os.path.join(a.out, "SOLVER_RA.json"))
    verify_stamps(a.out)
    open(os.path.join(a.out, "DONE"), "w").write("ok\n")
    return sol


def verify_stamps(out):
    """Read back: every plan file records the mode and readout sha256 of out/SOLVER_RA.json."""
    sol = json.load(open(os.path.join(out, "SOLVER_RA.json"), encoding="utf-8"))
    n = 0
    for f in sorted(os.listdir(out)):
        if f.startswith("plans_") and f.endswith(".npz"):
            with np.load(os.path.join(out, f), allow_pickle=False) as z:
                if "ra_l1_planned" not in z.files:
                    raise RefusedRun(f"{f} carries no ra_l1_planned stamp")
                if str(z["ra_readout_sha256"]) != sol["readout_sha256"] or str(z["ra_argv_sha256"]) != sol["argv_sha256"]:
                    raise RefusedRun(f"{f} stamp differs from SOLVER_RA.json")
            n += 1
    if not n:
        raise RefusedRun(f"{out} holds no plan file")
    return n


# ---------------------------------------------------------------------------------------------- C-READBACK of a base file
def readback_base(base, folder, seed=0, lam_override=None, l2_override=None):
    """Batch 0 of a base's banked plan file re-planned from its SOLVER.json argv at 2 threads, plan_raw and cost1 bit-equal. A0: LON-PLAN's wrapper; P / P23: AMEND_4's."""
    if base == "A0":
        import lonplan_run as W
        return W.readback(folder, "hwm_oracle", seed, lam_override=lam_override)
    sys.path.insert(0, ST.AMEND4_BUILD)
    import amend4_run as W4
    return W4.readback(folder, "hwm_oracle", seed, l2_override=l2_override)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan"); p.add_argument("--stage", required=True); p.add_argument("--arm", required=True); p.add_argument("--base", required=True); p.add_argument("--chassis", required=True)
    p.add_argument("--src-for-neg-shuf", choices=["l1", "l2"]); p.add_argument("--beside", action="store_true"); p.add_argument("--out")
    q = sp.add_parser("argv"); q.add_argument("--stage", required=True); q.add_argument("--arm", required=True); q.add_argument("--base", required=True); q.add_argument("--chassis", required=True); q.add_argument("--src-for-neg-shuf")
    a = ap.parse_args()
    b = ST.build(a.stage, a.arm, a.base, a.chassis, out=getattr(a, "out", None), src_for_nl=(a.src_for_neg_shuf if a.cmd == "plan" else a.src_for_neg_shuf))
    if a.cmd == "argv":
        print(json.dumps(b, indent=1)); return
    argv = b["argv"] + (["--beside"] if a.beside else [])
    sol = run_arm(argv, b["declared"])
    print("PLANNED", a.stage, a.arm, a.base, a.chassis, sorted(sol["plans"]), flush=True)


if __name__ == "__main__":
    main()
