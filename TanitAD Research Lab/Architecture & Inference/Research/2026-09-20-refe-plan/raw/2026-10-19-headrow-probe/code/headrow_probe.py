"""HEADROW-PROBE: are the REFE head-row levers capacity-limited or feature-limited?  CPU, fit-side OOF, f = 10 %.

REGISTERED SOURCE: PREREG_HEADROW_PROBE.md (this package; its sha256 is in PREREG_HEADROW_PROBE.sha256, written before the first fit).  NEW FILE ONLY: the landed `epr1.py` (sha256 pinned below,
verified before it is executed) is imported UNCHANGED for the bank loader + sha256 verification, PathGuard, folds, frozen rule, picks, pdms/safe, bootstrap draws and paired_delta.

ROWS  NC (col 0), DAC (1), EP (2), TTC (3) of P2's `score_head`.  ARMS  L = linear row refit from P2's row;  M = the same linear skip + a hidden-256 GELU branch whose output layer starts at 0.
Both: Adam lr 1e-3, weight decay 1e-4, 20 epochs, 256 sets per batch, BCE on the row's (soft) label in its registered convention, the SAME batches and seeds.

STAGES (each writes once, never overwrites; every fitting stage needs the gate token)
  gate    : mint gate_token.json (G-COMMIT, G-DVB, G-BANK, G-LIVE reach proof on a real batch, G-SUITE).
  cid     : C-ID identity on the real bank (z == z_stored under torch.equal, all four rows, L and M).
  main    : 4 rows x 2 arms x 2 seeds x 5 folds -> OOF logits (oof/*.npz).
  c2      : label-shuffle negative control (cold start, 5 replicates per arm) -> c2/*.npz.
  cpos    : planted nonlinear signal (the power control) -> cpos/*.npz.
  noise   : C-NOISE matched noise-row comparator (no fit).
  ares    : A-RES zero-training NC/DAC resolution read (no fit).
  nullrate: the measured false-RED rate of the C2 rule under its null (no fit).
  metrics : every statistic, the per-row calls, the controls -> probe_result.json.

NEVER READS the dev gate or navtest tables: every file access goes through the landed `_io_*` functions behind a PathGuard (forbidden tokens + allow-list).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_EPR1_SHA256 = "a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8"      # the LANDED epr1.py (tip 1afc54f6fa5e)
PARENT_COMMIT = "1afc54f6fa5ee865f2966e3c0a7b3141f24ef0c1"
PARENT_REPO_PATH = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-ep-r1/code/epr1.py"
PARENT_GIT_DIR = "C:/Users/Admin/tanitad-push/.git"


def _load_parent():
    """The landed epr1.py, executed from bytes whose sha256 equals the pin (no check-then-use gap), registered as sys.modules['epr1']."""
    p = os.path.join(HERE, "epr1.py")
    with open(p, "rb") as f:                                              # bootstrap read of the verified parent (the only direct file read at import time)
        src = f.read()
    sha = hashlib.sha256(src).hexdigest()
    if sha != PARENT_EPR1_SHA256:
        raise RuntimeError(f"epr1.py in {HERE} has sha256 {sha}, the landed pin is {PARENT_EPR1_SHA256}: refusing to import unverified code")
    cur = sys.modules.get("epr1")
    if cur is not None:
        if os.path.normcase(os.path.abspath(getattr(cur, "__file__", "") or "")) == os.path.normcase(os.path.abspath(p)):
            return cur
        raise RuntimeError("a different 'epr1' module is already imported")
    mod = types.ModuleType("epr1")
    mod.__file__ = p
    sys.modules["epr1"] = mod
    exec(compile(src, p, "exec", dont_inherit=True), mod.__dict__)
    return mod


E = _load_parent()

SCHEMA = "headrow-probe-v1"

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# REGISTERED constants (PREREG_HEADROW_PROBE.md).  Changing one changes the experiment.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
ROWS = ("nc", "dac", "ep", "ttc")                 # reporting order: NC and DAC first
ROW_COL = {"nc": 0, "dac": 1, "ep": 2, "ttc": 3}
ARMS = ("L", "M")
SEEDS = (0, 1)
N_PROPOSALS, FEAT_DIM, N_COMP = E.N_PROPOSALS, E.FEAT_DIM, E.N_COMP
HIDDEN = 256
EPOCHS = 20
BATCH_SETS = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
BCE_W = 1.0
CAP_MARGIN = 0.01                                  # CAPACITY: lower CI of (M - L) > +0.01 on both seeds
FEAT_MARGIN = 0.01                                 # FEATURE: upper CI < +0.01 on both seeds ...
P2_WITHIN = 0.01                                   # ... and |conc(L) - conc(P2 reference)| <= 0.01
C2_REPS = 5
C2_SEED = 0
C2_SHUFFLE_BASE = 20261019
NULL_TRIALS = 200
NULL_BASE = 20261019
NOISE_SIGMAS = (0.01, 0.03, 0.1, 0.3, 1.0)
NOISE_SEEDS = (0, 1)
POS_TEACHER_SEED = 20261019
MIN_RAM_BYTES = 3 * 2 ** 30
ROW_CONSISTENCY_ROWS_TOL = 1e-2
EPR1_RESULT_DEFAULT = "C:/Users/Admin/epr1/run/f10/epr1_result.json"
EPR1_ROW_PINS = {0: "f775753f0817d53607b0044e9d33f6d2c5f5b01b73a8d20567d6394ca393fa95", 1: "636f6270f52ef462fb6a1a56a990aa8c683f3c30a3394091b77d10f439852fa8"}
EPR1_MANIFEST_SHA = "6296dfc0b03905038fbf132f82009f6b6d7e8d61fa9abfb65c9164f487a5f2a1"
EPR1_FOLD_SETS = (3068, 4014, 3612, 2505, 2609)
DEFAULT_RUN_ROOT = "C:/Users/Admin/par18/probe/run"


class ProbeError(RuntimeError):
    """A precondition, identity or gate failed."""


class ReachError(RuntimeError):
    """The reach proof went RED."""


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# configuration (every lever lives here, is recorded in the gate token, and is checked against the built objects by G-DVB)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ProbeCfg:
    epochs: int = EPOCHS
    batch_sets: int = BATCH_SETS
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    hidden: int = HIDDEN
    bce_w: float = BCE_W
    seeds: tuple = SEEDS
    c2_reps: int = C2_REPS

    def sha256(self) -> str:
        return E.sha256_bytes(E.canon_json({**dataclasses.asdict(self), "betas": list(BETAS), "eps": ADAM_EPS, "optimizer": "adam_coupled_l2", "activation": "gelu_erf", "dropout": 0.0,
                                            "rows": list(ROWS), "arms": list(ARMS), "bootstrap_B": E.BOOT_B, "bootstrap_seed": E.BOOT_SEED, "schema": SCHEMA}))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# data: the landed FitData plus the full labels6, per-row labels and valid masks
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class PD:
    feat: torch.Tensor           # [N,64,256] fp16
    z6: torch.Tensor             # [N,64,6] fp32 stored logits
    lb: np.ndarray               # [N,64,6] labels6
    top8: torch.Tensor           # [N,8] int64
    log_idx: np.ndarray          # [N]
    fold: np.ndarray             # [N]
    n_logs: int
    pdms: np.ndarray             # [N,64]
    safe: np.ndarray             # [N,64] bool
    y: dict                      # col -> [N,64] float32 tensor
    valid: dict                  # col -> [N,64] bool tensor
    provenance: dict

    @property
    def n_sets(self) -> int:
        return int(self.feat.shape[0])


def valid_np(col: int, lb: np.ndarray) -> np.ndarray:
    """Valid entries of a row in its registered convention: EP is conditional (SAFE proposals only); NC, DAC, TTC use all 64."""
    if col == ROW_COL["ep"]:
        return E.safe_np(lb)
    return np.ones(lb.shape[:-1], dtype=bool)


def make_pd(feat, z6, lb, top8, log_idx, fold, n_logs, provenance=None) -> PD:
    lb = np.asarray(lb, np.float32)
    y = {c: torch.from_numpy(np.ascontiguousarray(lb[..., c])) for c in range(4)}
    valid = {c: torch.from_numpy(np.ascontiguousarray(valid_np(c, lb))) for c in range(4)}
    return PD(feat=feat, z6=z6, lb=lb, top8=top8, log_idx=np.asarray(log_idx), fold=np.asarray(fold), n_logs=int(n_logs), pdms=E.pdms_np(lb), safe=E.safe_np(lb), y=y, valid=valid,
              provenance=provenance or {})


@dataclasses.dataclass
class POptions(E.Options):
    extra_read_files: tuple = ()

    def guard(self):
        files = [self.manifest, self.ranked_logs, self.p2_ckpt, E.__file__, os.path.abspath(__file__), *self.extra_read_files]
        if self.s0_marker:
            files.append(self.s0_marker)
        return E.PathGuard(read_dirs=[self.bank_dir, self.resolved_out_dir()], read_files=files, write_dirs=[self.resolved_out_dir()])


def load_labels6(guard, opts, selected, manifest, fd) -> np.ndarray:
    """Second pass over the SAME units in the SAME order as the landed `load_fit_data`: re-hash each unit against the manifest BEFORE np.load, keep labels6 of the selected logs, and CROSS-CHECK the
    alignment with the landed FitData (log index, EP label, safe mask, pdms) so a row/label misalignment is impossible to miss."""
    sel_pos = {r["log"]: i for i, r in enumerate(selected)}
    units = [n for n in E._io_listdir(guard, opts.bank_dir) if n.endswith(".sets.npz")]
    out = np.empty((fd.n_sets, N_PROPOSALS, N_COMP), np.float32)
    off = 0
    for n in units:
        p = os.path.join(opts.bank_dir, n)
        if n not in manifest or E._io_sha256(guard, p) != manifest[n]:
            raise ProbeError(f"unit {n}: sha256 does not equal the manifest on the labels6 pass")
        z = E._io_npz(guard, p)
        lg = z["log"]
        mask = np.array([str(x) in sel_pos for x in lg], dtype=bool)
        if not mask.any():
            z.close()
            continue
        lb = z["labels6"][mask]
        lgm = lg[mask]
        z.close()
        S = len(lgm)
        if not np.array_equal(np.array([sel_pos[str(x)] for x in lgm]), fd.log_idx[off:off + S]):
            raise ProbeError(f"unit {n}: the labels6 pass is not aligned with the FitData rows (log index differs)")
        out[off:off + S] = lb
        off += S
    if off != fd.n_sets:
        raise ProbeError(f"labels6 pass read {off} sets, FitData has {fd.n_sets}")
    if not (np.array_equal(out[..., E.EP_COL], fd.y.numpy()) and np.array_equal(E.safe_np(out), fd.safe.numpy()) and np.array_equal(E.pdms_np(out), fd.pdms.numpy())):
        raise ProbeError("labels6 pass disagrees with the landed FitData (EP label / safe mask / pdms): alignment broken")
    return out


def load_probe_data(opts: POptions):
    """-> (guard, out_dir, manifest, man_sha, ranked_sha, selected, per_fold_logs, W6, b6, PD).  The landed `_setup` runs every integrity check (sha256 before np.load, frozen top-8, ranges, the P2-row
    consistency on the EP row); this adds labels6 and the row-consistency check for all four rows."""
    guard, out, manifest, man_sha, ranked_sha, selected, repl, per_fold_logs, W6, b6, fd = E._setup(opts)
    lb = load_labels6(guard, opts, selected, manifest, fd)
    step = max(1, fd.n_sets // E.ROW_CONSISTENCY_MAX_SETS)
    ii = np.arange(0, fd.n_sets, step)[:E.ROW_CONSISTENCY_MAX_SETS]
    dev = {}
    for r in ROWS:
        c = ROW_COL[r]
        chk = fd.feat[torch.from_numpy(ii)].float() @ torch.from_numpy(W6[c].copy()) + float(b6[c])
        dev[r] = float((chk - fd.z6[torch.from_numpy(ii), :, c]).abs().max())
        if not dev[r] <= ROW_CONSISTENCY_ROWS_TOL:
            raise ProbeError(f"P2 row {r}: max |feat.w+b - stored logit| = {dev[r]:.3e} > {ROW_CONSISTENCY_ROWS_TOL:g}")
    prov = dict(fd.provenance)
    prov["row_consistency_max_abs_by_row"] = dev
    pdat = make_pd(fd.feat, fd.z6, lb, fd.top8, fd.log_idx, fd.fold, fd.n_logs, prov)
    return guard, out, manifest, man_sha, ranked_sha, selected, per_fold_logs, W6, b6, pdat


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the nets, the loss, the fit
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class ProbeNet(torch.nn.Module):
    """L = the linear skip; M = the skip + W2 . GELU(W1 . x + b1) + b2.  Warm start: skip = P2's row and ref = P2's row, so the added terms are EXACTLY 0.0 at init (W2 = b2 = 0 for M).
    Cold start (C2, C-POS): skip = 0, ref = 0 and the stored logit is not used.  `probe=True` adds ALL six rows of P2's head as leaf parameters feeding the non-target columns."""

    def __init__(self, arm: str, W6, b6, col: int, seed: int, fold: int, cold: bool = False, hidden: int = HIDDEN, probe: bool = False, perturb: bool = False):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        W6 = torch.as_tensor(np.asarray(W6), dtype=torch.float32)
        b6 = torch.as_tensor(np.asarray(b6), dtype=torch.float32)
        self.arm, self.cold, self.col, self.probe, self.hidden = arm, bool(cold), int(col), bool(probe), int(hidden)
        zw, zb = torch.zeros(FEAT_DIM), torch.zeros(1)
        self.register_buffer("ref_w", zw if cold else W6[col].clone())
        self.register_buffer("ref_b", zb if cold else b6[col:col + 1].clone())
        self.skip_w = torch.nn.Parameter(zw.clone() if cold else W6[col].clone())
        self.skip_b = torch.nn.Parameter(zb.clone() if cold else b6[col:col + 1].clone())
        if arm == "M":
            g = torch.Generator().manual_seed(7919 * int(seed) + 104729 * int(col) + 1299709 * int(fold) + 11)
            bound = 1.0 / math.sqrt(FEAT_DIM)
            self.W1 = torch.nn.Parameter((torch.rand(hidden, FEAT_DIM, generator=g) * 2.0 - 1.0) * bound)
            self.b1 = torch.nn.Parameter((torch.rand(hidden, generator=g) * 2.0 - 1.0) * bound)
            self.W2 = torch.nn.Parameter(torch.zeros(1, hidden))
            self.b2 = torch.nn.Parameter(torch.zeros(1))
        if perturb:
            with torch.no_grad():
                self.skip_w += 1e-2 * torch.cos(torch.arange(FEAT_DIM, dtype=torch.float32))
                self.skip_b += 1e-2
                if arm == "M":
                    self.W2 += 1e-2 * torch.cos(torch.arange(hidden, dtype=torch.float32)).unsqueeze(0)
                    self.b2 += 1e-2
        if probe:
            self.frozen_W6 = torch.nn.Parameter(W6.clone())
            self.frozen_b6 = torch.nn.Parameter(b6.clone())

    def trainable(self) -> list:
        ps = [self.skip_w, self.skip_b]
        if self.arm == "M":
            ps += [self.W1, self.b1, self.W2, self.b2]
        return ps


def forward_z6(net: ProbeNet, feat: torch.Tensor, z6_stored: torch.Tensor, col: int | None = None) -> torch.Tensor:
    """[B,64,256] features, [B,64,6] stored logits -> [B,64,6] logits with ONLY column `col` depending on the trainable tensors."""
    col = net.col if col is None else col
    feat = feat.detach().float()                                                  # MUTATION-TARGET-DETACH: the trunk and the features are frozen
    base = torch.zeros_like(z6_stored[..., col]) if net.cold else z6_stored[..., col]
    z = base + feat @ (net.skip_w - net.ref_w) + (net.skip_b - net.ref_b)
    if net.arm == "M":
        h = torch.nn.functional.gelu(feat @ net.W1.t() + net.b1)                  # MUTATION-TARGET-HIDDEN: the hidden branch must stay on the graph
        z = z + (h @ net.W2.t() + net.b2).squeeze(-1)
    if net.probe:
        z_oth = torch.nn.functional.linear(feat, net.frozen_W6, net.frozen_b6)
        cols = [z if c == col else z_oth[..., c] for c in range(N_COMP)]
    else:
        cols = [z if c == col else z6_stored[..., c] for c in range(N_COMP)]
    return torch.stack(cols, dim=-1)


def row_loss(z6: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, col: int, w: float = BCE_W) -> torch.Tensor:
    """BCE (with logits) of the row's logit against its (soft) label, mean over the VALID entries (EP: safe proposals only)."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z6[..., col], y, reduction="none")
    m = valid.to(bce.dtype)                                                       # MUTATION-TARGET-VALID: only valid entries carry the target
    return w * (bce * m).sum() / m.sum().clamp(min=1.0)


def fit_arm(pd: PD, train_idx, cfg: ProbeCfg, arm: str, row: str, seed: int, fold: int, W6, b6, y_train=None, cold: bool = False, loss_fn=None, forward_fn=None,
            batch_hook=None, net: ProbeNet | None = None) -> ProbeNet:
    """Adam (coupled L2 weight decay), minibatches of `cfg.batch_sets` sets shuffled each epoch by a generator seeded from (seed, len(train_idx)) -> the SAME order for L, M and every row."""
    col = ROW_COL[row]
    loss_fn = loss_fn or row_loss
    forward_fn = forward_fn or forward_z6
    net = net or ProbeNet(arm, W6, b6, col, seed, fold, cold=cold, hidden=cfg.hidden)
    opt = torch.optim.Adam(net.trainable(), lr=cfg.lr, betas=BETAS, eps=ADAM_EPS, weight_decay=cfg.weight_decay)
    yt = pd.y[col] if y_train is None else y_train
    vt = pd.valid[col]
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(len(train_idx)))
    tr = torch.as_tensor(np.asarray(train_idx), dtype=torch.long)
    n = len(tr)
    for _ in range(cfg.epochs):
        order = tr[torch.randperm(n, generator=g)]
        for s in range(0, n, cfg.batch_sets):
            b = order[s:s + cfg.batch_sets]
            if batch_hook is not None:
                batch_hook(b)
            z6 = forward_fn(net, pd.feat[b], pd.z6[b], col)
            loss = loss_fn(z6, yt[b], vt[b], col, cfg.bce_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return net


def predict_col(net: ProbeNet, pd: PD, idx, col: int, chunk: int = 2048) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            ii = torch.as_tensor(np.asarray(idx[s:s + chunk]), dtype=torch.long)
            out.append(forward_z6(net, pd.feat[ii], pd.z6[ii], col)[..., col].numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, N_PROPOSALS), np.float32)


def mean_bce(net: ProbeNet, pd: PD, idx, y, valid, col: int, chunk: int = 2048) -> float:
    s = n = 0.0
    with torch.no_grad():
        for a in range(0, len(idx), chunk):
            ii = torch.as_tensor(np.asarray(idx[a:a + chunk]), dtype=torch.long)
            z = forward_z6(net, pd.feat[ii], pd.z6[ii], col)[..., col]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(z, y[ii], reduction="none").double()
            m = valid[ii].double()
            s += float((bce * m).sum())
            n += float(m.sum())
    return s / max(n, 1.0)


def wait_for_ram(min_bytes: int = MIN_RAM_BYTES, poll_s: float = 20.0, max_wait_s: float = 3 * 3600.0, log=None) -> float:
    """Block until psutil reports >= min_bytes available (the other sessions' jobs share this box).  -> seconds waited."""
    import psutil
    t0 = time.time()
    while psutil.virtual_memory().available < min_bytes:
        if time.time() - t0 > max_wait_s:
            raise ProbeError(f"available RAM stayed below {min_bytes / 2**30:.1f} GB for {max_wait_s / 3600:.1f} h")
        if log:
            log(f"RAM {psutil.virtual_memory().available / 2**30:.2f} GB < {min_bytes / 2**30:.1f} GB: waiting")
        time.sleep(poll_s)
    return time.time() - t0


def oof_logits(pd: PD, cfg: ProbeCfg, arm: str, row: str, seed: int, W6, b6, y_train=None, cold: bool = False, ram_guard: bool = False, log=None, fit_fn=None) -> dict:
    """5-fold grouped-by-log CV: train on fold != k, score fold == k.  -> {Z [N,64] float32 (the OOF logit of the replaced column), train_bce_by_fold, fit_s_by_fold}."""
    col = ROW_COL[row]
    N = pd.n_sets
    Z = np.empty((N, N_PROPOSALS), np.float32)
    tb, ts = [], []
    yt = pd.y[col] if y_train is None else y_train
    for k in range(E.N_FOLDS):
        oof = np.nonzero(pd.fold == k)[0]
        tr = np.nonzero(pd.fold != k)[0]
        if ram_guard:
            wait_for_ram(log=log)
        t0 = time.time()
        net = (fit_fn or fit_arm)(pd, tr, cfg, arm, row, seed, k, W6, b6, y_train=y_train, cold=cold)
        Z[oof] = predict_col(net, pd, oof, col)
        tb.append(mean_bce(net, pd, tr, yt, pd.valid[col], col))
        ts.append(time.time() - t0)
    return {"Z": Z, "train_bce_by_fold": tb, "fit_s_by_fold": ts}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# metrics: per-log sums -> pooled concordance / Brier / pick-level; bootstrap intervals
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def conc_per_set(Z: np.ndarray, y: torch.Tensor, valid: torch.Tensor, idx8: torch.Tensor | None = None, chunk: int = 1024, tie_half: bool = True) -> tuple:
    """Per-set concordance numerator / denominator: pairs (i, j) both valid with y_i > y_j; concordant z_i > z_j; ties count 0.5.  idx8 restricts both members to the frozen top-8."""
    N = len(Z)
    num, den = np.zeros(N), np.zeros(N)
    for s in range(0, N, chunk):
        z = torch.from_numpy(np.ascontiguousarray(Z[s:s + chunk]))
        yy, vv = y[s:s + chunk], valid[s:s + chunk]
        if idx8 is not None:
            i8 = idx8[s:s + chunk]
            z, yy, vv = torch.gather(z, 1, i8), torch.gather(yy, 1, i8), torch.gather(vv, 1, i8)
        m = (yy.unsqueeze(2) > yy.unsqueeze(1)) & vv.unsqueeze(2) & vv.unsqueeze(1)          # MUTATION-TARGET-PAIRS
        d = z.unsqueeze(2) - z.unsqueeze(1)
        n_ = ((d > 0) & m).sum((1, 2)).double()
        if tie_half:                                                                          # MUTATION-TARGET-TIE
            n_ = n_ + 0.5 * ((d == 0) & m).sum((1, 2)).double()
        num[s:s + len(z)] = n_.numpy()
        den[s:s + len(z)] = m.sum((1, 2)).double().numpy()
    return num, den


def brier_per_set(Z: np.ndarray, y: torch.Tensor, valid: torch.Tensor, chunk: int = 2048) -> tuple:
    N = len(Z)
    s_, n_ = np.zeros(N), np.zeros(N)
    for a in range(0, N, chunk):
        p = torch.sigmoid(torch.from_numpy(np.ascontiguousarray(Z[a:a + chunk])).double())
        v = valid[a:a + chunk].double()
        s_[a:a + len(p)] = (((p - y[a:a + chunk].double()) ** 2) * v).sum(1).numpy()
        n_[a:a + len(p)] = v.sum(1).numpy()
    return s_, n_


def per_log(pd: PD, v: np.ndarray) -> np.ndarray:
    return np.bincount(pd.log_idx, weights=np.asarray(v, np.float64), minlength=pd.n_logs)


def check_cluster_unit(arr_log: np.ndarray, draws: np.ndarray) -> None:
    """The unit of the bootstrap is the LOG: per-log arrays only, drawn with the landed `boot_draws(n_logs)` (shape [B, n_logs])."""
    if len(arr_log) != draws.shape[1] or int(draws.max()) >= len(arr_log):
        raise ProbeError(f"bootstrap input has {len(arr_log)} entries, the draws are over {draws.shape[1]} logs: the estimator resamples LOGS, never sets")


def ratio_ci(num_log: np.ndarray, den_log: np.ndarray, draws: np.ndarray) -> dict:
    check_cluster_unit(num_log, draws)
    if not den_log.sum() > 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "B": int(len(draws)), "n_nan": int(len(draws)), "empty": True}
    point = float(num_log.sum() / den_log.sum())
    bd = den_log[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(bd > 0, num_log[draws].sum(axis=1) / bd, np.nan)
    ok = ~np.isnan(v)
    return {"point": point, "lo": float(np.percentile(v[ok], 2.5)), "hi": float(np.percentile(v[ok], 97.5)), "B": int(len(draws)), "n_nan": int((~ok).sum())}


def diff_ci(num_a: np.ndarray, num_b: np.ndarray, den_log: np.ndarray, draws: np.ndarray) -> dict:
    return E.paired_delta(num_a, num_b, den_log, draws)


def scale100(d: dict) -> dict:
    return {k: (100.0 * v if k in ("point", "lo", "hi") else v) for k, v in d.items()}


def cell_logs(pd: PD, row: str, Z: np.ndarray, pk0: np.ndarray | None = None) -> dict:
    """Every per-log array one (row, arm, seed) cell needs, from its OOF logit table Z [N,64]."""
    col = ROW_COL[row]
    y, v = pd.y[col], pd.valid[col]
    n64, d64 = conc_per_set(Z, y, v)
    n8, d8 = conc_per_set(Z, y, v, pd.top8)
    bs, bn = brier_per_set(Z, y, v)
    z6n = pd.z6.numpy().copy()
    z6n[..., col] = Z
    pk = E.picks_np(z6n)
    pk0 = E.picks_np(pd.z6.numpy()) if pk0 is None else pk0
    r = np.arange(len(pk))
    out = {"n64": per_log(pd, n64), "d64": per_log(pd, d64), "n8": per_log(pd, n8), "d8": per_log(pd, d8), "bs": per_log(pd, bs), "bn": per_log(pd, bn),
           "pdn": per_log(pd, pd.pdms[r, pk].astype(np.float64)), "pdp": per_log(pd, pd.pdms[r, pk0].astype(np.float64)),
           "un": per_log(pd, (~pd.safe[r, pk]).astype(np.float64)), "up": per_log(pd, (~pd.safe[r, pk0]).astype(np.float64)), "ns": per_log(pd, np.ones(len(pk))),
           "changed": float((pk != pk0).sum()), "n_sets": int(len(pk))}
    return out


def p2_logs(pd: PD, row: str) -> dict:
    col = ROW_COL[row]
    return cell_logs(pd, row, pd.z6.numpy()[..., col].copy())


def cell_report(a: dict, ref: dict, draws: np.ndarray) -> dict:
    """One arm's numbers with intervals; `ref` = the P2 reference logs of the same row (in-sample for P2)."""
    return {
        "conc64": ratio_ci(a["n64"], a["d64"], draws), "conc8": ratio_ci(a["n8"], a["d8"], draws), "brier": ratio_ci(a["bs"], a["bn"], draws),
        "conc64_minus_p2": diff_ci(a["n64"], ref["n64"], a["d64"], draws), "conc8_minus_p2": diff_ci(a["n8"], ref["n8"], a["d8"], draws),
        "brier_minus_p2": diff_ci(a["bs"], ref["bs"], a["bn"], draws),
        "d_pdms_x100_vs_p2": scale100(diff_ci(a["pdn"], a["pdp"], a["ns"], draws)), "d_unsafe_vs_p2": diff_ci(a["un"], a["up"], a["ns"], draws),
        "unsafe_rate_new": float(a["un"].sum() / a["ns"].sum()), "unsafe_rate_p2": float(a["up"].sum() / a["ns"].sum()),
        "pdms_x100_new": float(100.0 * a["pdn"].sum() / a["ns"].sum()), "pdms_x100_p2": float(100.0 * a["pdp"].sum() / a["ns"].sum()),
        "changed_pick_fraction": float(a["changed"] / a["n_sets"]), "n_pairs64": float(a["d64"].sum()), "n_pairs8": float(a["d8"].sum()), "n_valid": float(a["bn"].sum()),
    }


def pair_report(m: dict, l: dict, draws: np.ndarray) -> dict:
    """M minus L, paired on the same logs and draws."""
    return {"conc64": diff_ci(m["n64"], l["n64"], m["d64"], draws), "conc8": diff_ci(m["n8"], l["n8"], m["d8"], draws), "brier": diff_ci(m["bs"], l["bs"], m["bn"], draws),
            "d_pdms_x100": scale100(diff_ci(m["pdn"], l["pdn"], m["ns"], draws)), "d_unsafe": diff_ci(m["un"], l["un"], m["ns"], draws)}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the decision rule and the control rules (pure functions of the intervals; unit-tested on fixtures)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def decide_row(per_seed: dict, c2_red: bool, cpos_green: bool) -> dict:
    """per_seed[s] = {"ml": {lo, hi}, "l_conc": float, "p2_conc": float}.  CAPACITY iff lo > +0.01 on every seed; FEATURE iff hi < +0.01 on every seed AND |L - P2| <= 0.01 on every seed.
    C2 RED voids CAPACITY; a RED C-POS voids FEATURE.  -> {call, base_call, reasons}."""
    seeds = sorted(per_seed)
    cap = all(per_seed[s]["ml"]["lo"] > CAP_MARGIN for s in seeds)
    feat_ml = all(per_seed[s]["ml"]["hi"] < FEAT_MARGIN for s in seeds)
    near = all(abs(per_seed[s]["l_conc"] - per_seed[s]["p2_conc"]) <= P2_WITHIN for s in seeds)
    feat = feat_ml and near
    base = "CAPACITY-LIMITED" if cap else ("FEATURE-LIMITED" if feat else "INCONCLUSIVE")
    call, reasons = base, []
    if base == "CAPACITY-LIMITED" and c2_red:
        call, reasons = "INCONCLUSIVE", ["C2 RED: CAPACITY void"]
    if base == "FEATURE-LIMITED" and not cpos_green:
        call, reasons = "INCONCLUSIVE", ["C-POS RED: probe underpowered, FEATURE void"]
    if base == "INCONCLUSIVE":
        if not cap and not feat_ml:
            reasons.append("M - L upper CI >= +0.01 on some seed and lower CI <= +0.01 on some seed")
        if feat_ml and not near:
            reasons.append("M - L upper CI < +0.01 on both seeds but |L - P2| > 0.01 on some seed")
    return {"call": call, "base_call": base, "reasons": reasons, "cap": cap, "feat_ml": feat_ml, "l_within_p2": near}


def c2_cell_red(replicate_conc_ci: list) -> bool:
    """RED iff on ANY replicate the lower bound of (shuffled concordance - 0.5) is > 0."""
    return any((r["lo"] - 0.5) > 0.0 for r in replicate_conc_ci)


def c2_dec_red(replicate_ml_ci: list) -> bool:
    """C2-DEC: RED iff on any replicate the lower bound of the shuffled (M - L) concordance is > the CAPACITY margin."""
    return any(r["lo"] > CAP_MARGIN for r in replicate_ml_ci)


def cleak_flag(per_seed_du: dict) -> str:
    seeds = sorted(per_seed_du)
    if all(per_seed_du[s]["lo"] > 0.0 for s in seeds):
        return "LEAK"
    if all(per_seed_du[s]["hi"] <= 0.0 for s in seeds):
        return "CLEAN"
    return "AMBIGUOUS"


def combine_outcome(calls: dict, cpos_green: bool) -> dict:
    """The outcome row: per-row calls -> committed next arms (PREREG s7)."""
    arms, notes = [], []
    for r in ROWS:
        c = calls[r]["call"]
        if c == "CAPACITY-LIMITED":
            arms.append({"row": r, "arm": "nonlinear-head registration with a powered C2 on a fresh dev read where one remains (EP-R1 dev read UNSPENT; SR-TTC consumed)" if r in ("ep", "ttc")
                         else "safety-row MLP registration, the direct test of H-REFE-RL1-NCRES"})
        elif c == "INCONCLUSIVE":
            arms.append({"row": r, "arm": "re-run this row ONCE at f = 25 % (cost named in RESULT.md)"})
    all_feat = all(calls[r]["call"] == "FEATURE-LIMITED" for r in ROWS)
    if all_feat:
        arms.append({"row": "all", "arm": "the head-row line closes; the lever is the trunk (P6, read ~25 Oct); an EP-conditional gradient into the trunk is drafted as the next A40 arm"})
    elif any(calls[r]["call"] == "FEATURE-LIMITED" for r in ROWS):
        notes.append("FEATURE rows: " + ",".join(r for r in ROWS if calls[r]["call"] == "FEATURE-LIMITED") + " close their head-row line (lever: the trunk)")
    label = "ALL-FEATURE" if all_feat else "MIXED/CAPACITY/INCONCLUSIVE"
    return {"label": label, "next_arms": arms, "notes": notes, "cpos_green": cpos_green}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# reach proof, identity
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _grads(net: ProbeNet, feat_leaf: torch.Tensor) -> dict:
    def n(t):
        return 0.0 if t is None else float(t.abs().sum())
    fw, fb = net.frozen_W6.grad, net.frozen_b6.grad
    rep = {"skip_w": n(net.skip_w.grad), "skip_b": n(net.skip_b.grad), "frozen_rows_w": n(fw), "frozen_rows_b": n(fb),
           "frozen_rows_w_per_row": [0.0 if fw is None else float(fw[r].abs().sum()) for r in range(N_COMP)], "features": n(feat_leaf.grad)}
    if net.arm == "M":
        rep.update({"W1": n(net.W1.grad), "b1": n(net.b1.grad), "W2": n(net.W2.grad), "b2": n(net.b2.grad)})
    return rep


def probe_grads(arm: str, W6, b6, row: str, batch: dict, w: float = BCE_W, perturb: bool = True, y=None, valid=None, loss_fn=None, forward_fn=None, hidden: int = HIDDEN,
                perturb_hidden: bool | None = None) -> dict:
    col = ROW_COL[row]
    loss_fn = loss_fn or row_loss
    forward_fn = forward_fn or forward_z6
    net = ProbeNet(arm, W6, b6, col, seed=0, fold=0, cold=False, hidden=hidden, probe=True, perturb=perturb)
    if arm == "M" and perturb and perturb_hidden is False:
        with torch.no_grad():
            net.W2.zero_()
            net.b2.zero_()
    feat = batch["feat"].clone().float().requires_grad_(True)
    z6 = forward_fn(net, feat, batch["z6"], col)
    loss = loss_fn(z6, batch["y"] if y is None else y, batch["valid"] if valid is None else valid, col, w)
    if loss.requires_grad:
        loss.backward()
    rep = _grads(net, feat)
    rep["_g"] = torch.cat([net.skip_w.grad.flatten() if net.skip_w.grad is not None else torch.zeros(FEAT_DIM), net.skip_b.grad.flatten() if net.skip_b.grad is not None else torch.zeros(1)])
    return rep


def reach_proof(pd: PD, W6, b6, row: str, arm: str, n: int = 512, loss_fn=None, forward_fn=None, hidden: int = HIDDEN) -> dict:
    """The 'built, tested, unreachable from its caller' proof on a REAL batch; raises ReachError listing every failure."""
    col = ROW_COL[row]
    y, v = pd.y[col], pd.valid[col]
    y8, v8 = torch.gather(y, 1, pd.top8), torch.gather(v, 1, pd.top8)
    contrast = (((y.unsqueeze(2) > y.unsqueeze(1)) & v.unsqueeze(2) & v.unsqueeze(1)).sum((1, 2)) > 0)
    if row == "ep":
        contrast = contrast & (~v).any(1)                                             # EP: the batch must also contain unsafe entries (the C6 identities need them)
    ii = contrast.nonzero().flatten()[:n]
    if len(ii) < 8:
        raise ReachError(f"{row}: fewer than 8 sets with contrast (and unsafe entries for EP): the reach proof cannot run")
    batch = {"feat": pd.feat[ii], "z6": pd.z6[ii], "y": y[ii], "valid": v[ii]}
    fails, rep = [], {"row": row, "arm": arm, "n_sets": int(len(ii))}
    kw = dict(loss_fn=loss_fn, forward_fn=forward_fn, hidden=hidden)
    full = probe_grads(arm, W6, b6, row, batch, **kw)
    zero = probe_grads(arm, W6, b6, row, batch, w=0.0, **kw)
    rep["full"] = {k: v_ for k, v_ in full.items() if not k.startswith("_")}
    rep["zero_control"] = {k: v_ for k, v_ in zero.items() if not k.startswith("_")}
    for nm in ("skip_w", "skip_b") + (("W1", "b1", "W2", "b2") if arm == "M" else ()):
        if not (full[nm] > 0.0 and math.isfinite(full[nm])):
            fails.append(f"full: gradient on {nm} is {full[nm]!r} (must be finite and > 0)")
    for nm, r_ in (("full", full), ("zero_control", zero)):
        if r_["frozen_rows_w"] != 0.0 or r_["frozen_rows_b"] != 0.0:
            fails.append(f"{nm}: a FROZEN row received a gradient (w {r_['frozen_rows_w']!r}, b {r_['frozen_rows_b']!r}, per row {r_['frozen_rows_w_per_row']}); must be EXACTLY 0.0")
        if r_["features"] != 0.0:
            fails.append(f"{nm}: the features received a gradient ({r_['features']!r}); must be EXACTLY 0.0")
    pnames = ("skip_w", "skip_b") + (("W1", "b1", "W2", "b2") if arm == "M" else ())
    if any(zero[nm] != 0.0 for nm in pnames):
        fails.append(f"zero_control: weight 0.0 still gave a probe-parameter gradient ({ {nm: zero[nm] for nm in pnames} }); the probe is not reading the loss")
    if arm == "M":
        ident = probe_grads(arm, W6, b6, row, batch, perturb_hidden=False, **kw)         # W2 = b2 = 0 exactly: dL/dh = 0 -> W1, b1 get EXACTLY 0.0, W2, b2 get > 0
        rep["zero_output_layer"] = {k: v_ for k, v_ in ident.items() if not k.startswith("_")}
        if ident["W1"] != 0.0 or ident["b1"] != 0.0:
            fails.append(f"zero-output-layer identity: W1/b1 gradient ({ident['W1']!r}, {ident['b1']!r}) is not EXACTLY 0.0 at W2 = b2 = 0")
        if not (ident["W2"] > 0.0 and ident["b2"] > 0.0):
            fails.append(f"zero-output-layer: W2/b2 gradient ({ident['W2']!r}, {ident['b2']!r}) must be > 0 (the hidden branch must be able to start learning)")
    if row == "ep":
        g_ = torch.Generator().manual_seed(11)
        y_junk = torch.rand(batch["y"].shape, generator=g_)
        inv = probe_grads(arm, W6, b6, row, batch, y=torch.where(batch["valid"], batch["y"], y_junk), **kw)
        same = bool(torch.equal(inv["_g"], full["_g"]))
        rep["c6_unsafe_label_invariance_bit_identical"] = same
        if not same:
            fails.append("C6 (i): replacing the EP labels of UNSAFE entries changed the gradient (the target is not conditional on NC = DAC = 1)")
        allun = probe_grads(arm, W6, b6, row, batch, y=y_junk, valid=torch.zeros_like(batch["valid"]), **kw)
        rep["c6_all_unsafe_grad"] = {"skip_w": allun["skip_w"], "skip_b": allun["skip_b"]}
        if allun["skip_w"] != 0.0 or allun["skip_b"] != 0.0:
            fails.append("C6 (ii): an all-unsafe batch gave a non-zero gradient")
        chg = probe_grads(arm, W6, b6, row, batch, y=torch.where(batch["valid"], 1.0 - batch["y"], batch["y"]), **kw)
        rep["c6_safe_label_change_changes_gradient"] = bool(not torch.equal(chg["_g"], full["_g"]))
        if not rep["c6_safe_label_change_changes_gradient"]:
            fails.append("C6 (iii): changing SAFE labels did not change the gradient, so (i) would mean nothing")
    # the real fit path moves the probe parameters (one epoch, one batch)
    mini = dataclasses.replace(pd, feat=batch["feat"], z6=batch["z6"], lb=pd.lb[ii.numpy()], top8=pd.top8[ii], log_idx=pd.log_idx[ii.numpy()], fold=pd.fold[ii.numpy()], pdms=pd.pdms[ii.numpy()],
                               safe=pd.safe[ii.numpy()], y={col: batch["y"]}, valid={col: batch["valid"]})
    cfg1 = ProbeCfg(epochs=1, batch_sets=len(ii))
    net = fit_arm(mini, np.arange(len(ii)), cfg1, arm, row, 0, 0, W6, b6, loss_fn=loss_fn, forward_fn=forward_fn)
    moved = float((net.skip_w.detach() - torch.as_tensor(np.asarray(W6)[col])).abs().max())
    rep["update_moved_skip_max_abs"] = moved
    if not moved > 0.0:
        fails.append("update: one optimiser step did not move the skip row (the loss is not wired to the optimiser)")
    if fails:
        raise ReachError(f"{row}/{arm} reach proof RED: " + " | ".join(fails))
    return rep


def identity_check(pd: PD, W6, b6, row: str, arm: str, net_factory=None) -> dict:
    """C-ID on the real bank: with zero training steps z == z_stored under torch.equal on ALL sets; the picks, concordance sums and Brier sums equal P2's EXACTLY; W2 = b2 = 0, skip == P2's row."""
    col = ROW_COL[row]
    net = (net_factory or (lambda: ProbeNet(arm, W6, b6, col, seed=0, fold=0, cold=False)))()
    with torch.no_grad():
        Z = np.concatenate([forward_z6(net, pd.feat[a:a + 2048], pd.z6[a:a + 2048], col)[..., col].numpy() for a in range(0, pd.n_sets, 2048)])
    stored = pd.z6.numpy()[..., col]
    eq = bool(torch.equal(torch.from_numpy(Z), torch.from_numpy(np.ascontiguousarray(stored))))
    skip_eq = bool(torch.equal(net.skip_w.detach(), torch.as_tensor(np.asarray(W6)[col]))) and bool(torch.equal(net.skip_b.detach(), torch.as_tensor(np.asarray(b6)[col:col + 1])))
    out_zero = True if arm == "L" else bool((net.W2 == 0).all() and (net.b2 == 0).all())
    a, ref = cell_logs(pd, row, Z), p2_logs(pd, row)
    sums_eq = all(np.array_equal(a[k], ref[k]) for k in ("n64", "d64", "n8", "d8", "bs", "bn", "pdn", "pdp", "un", "up")) and a["changed"] == 0.0
    ok = eq and skip_eq and out_zero and sums_eq
    return {"row": row, "arm": arm, "z_equal_stored_all_sets": eq, "skip_equals_p2_row": skip_eq, "output_layer_zero": out_zero, "sums_equal_p2_reference": sums_eq, "n_sets": pd.n_sets, "ok": ok,
            "max_abs_diff": float(np.abs(Z - stored).max())}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# controls: C2 shuffles, the null rate, C-POS planted signal, C-NOISE
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def shuffle_valid(y: torch.Tensor, valid: torch.Tensor, seed: int) -> torch.Tensor:
    """Permute the labels of each set among its VALID entries (multiset and mask unchanged).  Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    yn, vn = y.numpy().copy(), valid.numpy()
    for i in range(len(yn)):
        idx = np.nonzero(vn[i])[0]
        if len(idx) > 1:
            yn[i, idx] = yn[i, idx[rng.permutation(len(idx))]]
    return torch.from_numpy(yn)


def c2_seed(rep: int, row: str) -> int:
    return C2_SHUFFLE_BASE + 1000 * rep + 10 * ROW_COL[row]


def null_trial(pd: PD, row: str, draws: np.ndarray, trial: int, reps: int = C2_REPS, rng_fn=None) -> dict:
    """One trial of the C2 rule under its null: `reps` replicates of iid N(0,1) scores independent of the labels, the SAME per-log concordance and bootstrap, the SAME rule."""
    col = ROW_COL[row]
    cis = []
    for rep in range(reps):
        rng = np.random.default_rng(NULL_BASE + 7 * trial + rep)
        Z = rng.standard_normal((pd.n_sets, N_PROPOSALS)).astype(np.float32)
        n64, d64 = conc_per_set(Z, pd.y[col], pd.valid[col])
        cis.append(ratio_ci(per_log(pd, n64), per_log(pd, d64), draws))
    return {"trial": trial, "red": bool(c2_cell_red(cis)), "n_rep_red": int(sum((c["lo"] - 0.5) > 0 for c in cis)), "lo_minus_half": [c["lo"] - 0.5 for c in cis]}


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def planted_labels(feat: torch.Tensor, chunk: int = 1024) -> tuple:
    """C-POS: a label-independent planted nonlinear signal on the REAL features (PREREG 4.1).  -> (y [N,64] float32 soft label, info)."""
    N = feat.shape[0]
    s1 = np.zeros(FEAT_DIM)
    s2 = np.zeros(FEAT_DIM)
    cnt = 0
    for a in range(0, N, chunk):
        x = feat[a:a + chunk].reshape(-1, FEAT_DIM).double().numpy()
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
        cnt += len(x)
    mu = s1 / cnt
    sd = np.sqrt(np.maximum(s2 / cnt - mu * mu, 1e-12))
    rng = np.random.default_rng(POS_TEACHER_SEED)
    V = rng.standard_normal((32, FEAT_DIM))
    w = rng.standard_normal(32) / math.sqrt(32)
    g0 = np.empty((N, N_PROPOSALS))
    XtX = np.zeros((FEAT_DIM + 1, FEAT_DIM + 1))
    Xtg = np.zeros(FEAT_DIM + 1)
    for a in range(0, N, chunk):
        x = (feat[a:a + chunk].reshape(-1, FEAT_DIM).double().numpy() - mu) / sd
        h = torch.nn.functional.gelu(torch.from_numpy(3.0 * (x @ V.T) / math.sqrt(FEAT_DIM))).numpy()
        g = h @ w
        g0[a:a + chunk] = g.reshape(-1, N_PROPOSALS)
        X1 = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        XtX += X1.T @ X1
        Xtg += X1.T @ g
    beta = np.linalg.solve(XtX + 1e-9 * np.eye(FEAT_DIM + 1), Xtg)
    g1 = np.empty_like(g0)
    for a in range(0, N, chunk):
        x = (feat[a:a + chunk].reshape(-1, FEAT_DIM).double().numpy() - mu) / sd
        X1 = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        g1[a:a + chunk] = (g0[a:a + chunk].reshape(-1) - X1 @ beta).reshape(-1, N_PROPOSALS)
    g1 = (g1 - g1.mean()) / g1.std()
    y = 1.0 / (1.0 + np.exp(-2.0 * g1))
    return torch.from_numpy(y.astype(np.float32)), {"std_g0": float(g0.std()), "residual_linear_removed": True, "y_mean": float(y.mean()), "y_std": float(y.std())}


def noise_cell(pd: PD, row: str, sigma: float, nseed: int, i_sigma: int, draws: np.ndarray, pk0: np.ndarray) -> dict:
    col = ROW_COL[row]
    rng = np.random.default_rng(NULL_BASE + 100 * ROW_COL[row] + 10 * i_sigma + nseed)
    Z = pd.z6.numpy()[..., col] + sigma * rng.standard_normal((pd.n_sets, N_PROPOSALS)).astype(np.float32)
    z6n = pd.z6.numpy().copy()
    z6n[..., col] = Z
    pk = E.picks_np(z6n)
    r = np.arange(len(pk))
    pdn, pdp = per_log(pd, pd.pdms[r, pk].astype(np.float64)), per_log(pd, pd.pdms[r, pk0].astype(np.float64))
    un, up, ns = per_log(pd, (~pd.safe[r, pk]).astype(np.float64)), per_log(pd, (~pd.safe[r, pk0]).astype(np.float64)), per_log(pd, np.ones(len(pk)))
    return {"row": row, "sigma": sigma, "noise_seed": nseed, "changed_pick_fraction": float((pk != pk0).mean()), "d_unsafe": diff_ci(un, up, ns, draws),
            "d_pdms_x100": scale100(diff_ci(pdn, pdp, ns, draws))}


def match_noise(arm_changed: float, cells: list) -> dict:
    """The noise sigma whose seed-mean changed-pick fraction is nearest to the arm's; -> {sigma, cells}."""
    by = {}
    for c in cells:
        by.setdefault(c["sigma"], []).append(c)
    best = min(by, key=lambda s: abs(float(np.mean([c["changed_pick_fraction"] for c in by[s]])) - arm_changed))
    return {"sigma": best, "mean_changed_pick_fraction": float(np.mean([c["changed_pick_fraction"] for c in by[best]])), "cells": by[best]}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# A-RES (adopted from the draft; zero training)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def auc_pairs_per_set(s8: np.ndarray, safe8: np.ndarray) -> tuple:
    """Within-top-8 AUC numerators / denominators per set: P(s_safe > s_unsafe) over (safe, unsafe) contender pairs, ties 0.5."""
    sf = safe8[:, :, None] & (~safe8)[:, None, :]
    d = s8[:, :, None] - s8[:, None, :]
    num = (((d > 0) & sf).sum((1, 2)) + 0.5 * ((d == 0) & sf).sum((1, 2))).astype(np.float64)
    return num, sf.sum((1, 2)).astype(np.float64)


def ares_stats(pd: PD, pk_new: np.ndarray, pk_old: np.ndarray, draws: np.ndarray, flip_sign: bool = False, unsafe_mask_fn=None, top8_override=None) -> dict:
    """The statistics of PREREG 4.4 for ONE seed.  s = log(pNC * pDAC) from P2's stored logits.  Mutation hooks (flip_sign, unsafe_mask_fn, top8_override) exist for the RED arms."""
    z = pd.z6.numpy().astype(np.float64)
    s = -np.logaddexp(0.0, -z[..., 0]) - np.logaddexp(0.0, -z[..., 1])                 # log sigmoid(z0) + log sigmoid(z1), stable
    r = np.arange(len(pk_new))
    safe = pd.safe if unsafe_mask_fn is None else unsafe_mask_fn(pd.safe)
    V = safe[r, pk_old] & ~safe[r, pk_new]
    S = ~safe[r, pk_old] & safe[r, pk_new]
    D = (pk_new != pk_old) & safe[r, pk_old] & safe[r, pk_new]
    m = s[r, pk_new] - s[r, pk_old]
    if flip_sign:
        m = -m
    t8 = pd.top8.numpy() if top8_override is None else top8_override
    s8 = np.take_along_axis(s, t8, 1)
    safe8 = np.take_along_axis(safe, t8, 1)
    an, ad = auc_pairs_per_set(s8, safe8)
    best = np.argmax(s8, axis=1)
    new_in8 = (t8 == pk_new[:, None])
    is_best = new_in8[r, best]
    both = ad > 0
    out = {"n_V": int(V.sum()), "n_S": int(S.sum()), "n_D": int(D.sum())}
    out["frac_V_m_ge0"] = ratio_ci(per_log(pd, (V & (m >= 0)).astype(np.float64)), per_log(pd, V.astype(np.float64)), draws) if V.any() else None
    out["median_m_V"] = float(np.median(m[V])) if V.any() else None
    out["frac_D_m_ge0"] = ratio_ci(per_log(pd, (D & (m >= 0)).astype(np.float64)), per_log(pd, D.astype(np.float64)), draws) if D.any() else None
    out["median_m_D"] = float(np.median(m[D])) if D.any() else None
    out["auc_V"] = ratio_ci(per_log(pd, np.where(V, an, 0.0)), per_log(pd, np.where(V, ad, 0.0)), draws) if (V & both).any() else None
    out["auc_all"] = ratio_ci(per_log(pd, np.where(both, an, 0.0)), per_log(pd, np.where(both, ad, 0.0)), draws)
    out["share_V_new_is_best_pnc_pdac"] = ratio_ci(per_log(pd, (V & is_best).astype(np.float64)), per_log(pd, V.astype(np.float64)), draws) if V.any() else None
    return out


def ares_flag(per_seed: dict) -> dict:
    """R-NONE / R-RES / R-OVER / R-MIXED from the point estimates (the draft's thresholds)."""
    seeds = sorted(per_seed)
    if any(per_seed[s]["n_V"] < 100 or per_seed[s]["frac_V_m_ge0"] is None or per_seed[s]["auc_V"] is None for s in seeds):
        return {"flag": "R-NONE", "n_V": {s: per_seed[s]["n_V"] for s in seeds}}
    fr = {s: per_seed[s]["frac_V_m_ge0"]["point"] for s in seeds}
    au = {s: per_seed[s]["auc_V"]["point"] for s in seeds}
    if all(fr[s] >= 0.50 and au[s] <= 0.60 for s in seeds):
        flag = "R-RES"
    elif all(fr[s] < 0.50 and au[s] >= 0.70 for s in seeds):
        flag = "R-OVER"
    else:
        flag = "R-MIXED"
    return {"flag": flag, "frac_V_m_ge0": fr, "auc_V": au}


def ares_bruteforce(pd: PD, pk_new: np.ndarray, pk_old: np.ndarray, idx: np.ndarray) -> dict:
    """C2': an independent loop-based restatement on a subsample, to compare with the vectorised statistics."""
    z = pd.z6.numpy().astype(np.float64)
    nV = nVm = 0
    an = ad = 0.0
    for i in idx:
        s = [math.log(1 / (1 + math.exp(-z[i, j, 0]))) + math.log(1 / (1 + math.exp(-z[i, j, 1]))) for j in range(N_PROPOSALS)]
        po, pn = int(pk_old[i]), int(pk_new[i])
        if pd.safe[i, po] and not pd.safe[i, pn]:
            nV += 1
            nVm += int(s[pn] - s[po] >= 0)
        t8 = [int(j) for j in pd.top8[i].numpy()]
        for a in t8:
            for b in t8:
                if pd.safe[i, a] and not pd.safe[i, b]:
                    ad += 1
                    an += 1.0 if s[a] > s[b] else (0.5 if s[a] == s[b] else 0.0)
    return {"n_V": nV, "n_V_m_ge0": nVm, "auc_num": an, "auc_den": ad}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# I/O helpers (the only file accesses go through the landed guarded functions)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def npz_bytes(**arrs) -> bytes:
    bio = io.BytesIO()
    np.savez(bio, **arrs)
    return bio.getvalue()


def npy_bytes(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, arr)
    return bio.getvalue()


def apply_row_chunked(z6: torch.Tensor, feat: torch.Tensor, w, b, p2w, p2b, col: int, chunk: int = 1024) -> torch.Tensor:
    """The landed `apply_row` over chunks (the full float32 feature tensor would be 1 GB)."""
    return torch.cat([E.apply_row(z6[a:a + chunk], feat[a:a + chunk], w, b, p2w, p2b, col=col) for a in range(0, len(z6), chunk)])


def write_json(guard, path: str, obj) -> str:
    return E._io_write_bytes(guard, path, E.canon_json(obj), exclusive=True)


def file_sha(guard, path: str) -> str:
    return E._io_sha256(guard, path)


def code_shas(guard) -> dict:
    out = {}
    for n in ("headrow_probe.py", "test_headrow_probe.py", "epr1.py"):
        p = os.path.join(HERE, n)
        if os.path.exists(p):
            out[n] = E.sha256_bytes(E._io_read_bytes(guard, p))
    return out


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_opts(a: argparse.Namespace) -> POptions:
    out = a.out_dir or f"{DEFAULT_RUN_ROOT}/f{a.fraction}"
    extra = [a.epr1_result, os.path.join(HERE, "test_headrow_probe.py")] + [os.path.join(os.path.dirname(a.epr1_result), f"ep_row_seed{s}.npy") for s in (0, 1)]
    return POptions(fraction=a.fraction, out_dir=out, threads=a.threads, extra_read_files=tuple(extra))


def stage_gate(a: argparse.Namespace) -> dict:
    opts = make_opts(a)
    cfg = ProbeCfg()
    guard, out, manifest, man_sha, ranked_sha, selected, per_fold_logs, W6, b6, pd = load_probe_data(opts)
    E._io_makedirs(guard, out)
    tok_path = os.path.join(out, "gate_token.json")
    checks, reasons = {}, []
    # G-COMMIT
    shas = code_shas(guard)
    gc = {"epr1_sha256": shas["epr1.py"], "pinned": PARENT_EPR1_SHA256, "equal_pin": shas["epr1.py"] == PARENT_EPR1_SHA256}
    try:
        blob = subprocess.run(["git", "--git-dir", PARENT_GIT_DIR, "cat-file", "blob", f"{PARENT_COMMIT}:{PARENT_REPO_PATH}"], capture_output=True, check=True).stdout
        gc["blob_sha256_at_commit"] = E.sha256_bytes(blob)
        gc["equal_blob"] = gc["blob_sha256_at_commit"] == PARENT_EPR1_SHA256
    except Exception as ex:                                                              # noqa: BLE001
        gc["blob_error"] = repr(ex)[:200]
        gc["equal_blob"] = False
    checks["G-COMMIT"] = bool(gc["equal_pin"] and gc["equal_blob"])
    # G-DVB: every lever of the config equals what the built objects report
    dvb = {}
    for arm in ARMS:
        net = ProbeNet(arm, W6, b6, ROW_COL["ep"], 0, 0, hidden=cfg.hidden)
        opt = torch.optim.Adam(net.trainable(), lr=cfg.lr, betas=BETAS, eps=ADAM_EPS, weight_decay=cfg.weight_decay)
        d = opt.defaults
        ok = (d["lr"] == cfg.lr and tuple(d["betas"]) == BETAS and d["eps"] == ADAM_EPS and d["weight_decay"] == cfg.weight_decay and not any(isinstance(m, torch.nn.Dropout) for m in net.modules())
              and torch.equal(net.skip_w.detach(), torch.as_tensor(W6[ROW_COL["ep"]])) and (arm == "L" or (net.W1.shape == (cfg.hidden, FEAT_DIM) and net.W2.shape == (1, cfg.hidden) and bool((net.W2 == 0).all() and (net.b2 == 0).all()))))
        dvb[arm] = bool(ok)
    steps = []
    mini_n = 600
    cfg_small = ProbeCfg(epochs=cfg.epochs, batch_sets=cfg.batch_sets)
    fit_arm(pd, np.arange(mini_n), dataclasses.replace(cfg_small, epochs=2), "L", "ttc", 0, 0, W6, b6, batch_hook=lambda b: steps.append(len(b)))
    dvb["steps_equal_epochs_x_ceil_n_over_batch"] = bool(len(steps) == 2 * math.ceil(mini_n / cfg.batch_sets))
    dvb["cfg_sha256"] = cfg.sha256()
    checks["G-DVB"] = bool(all(v for k, v in dvb.items() if k != "cfg_sha256"))
    # G-BANK: the same fit logs as EP-R1 f10
    er = json.loads(E._io_read_bytes(guard, a.epr1_result).decode("utf-8"))
    gb = {"manifest_sha256": man_sha, "ranked_sha256": ranked_sha, "n_logs": pd.n_logs, "n_sets": pd.n_sets, "per_fold_sets": [int((pd.fold == k).sum()) for k in range(E.N_FOLDS)],
          "epr1_manifest_sha256": er["inputs"]["bank_manifest_sha256"], "epr1_ranked_sha256": er["inputs"]["ranked_fit_logs_sha256"], "epr1_n_logs": er["inputs"]["n_logs"], "epr1_n_sets": er["inputs"]["n_sets"],
          "epr1_per_fold_sets": [t["sets"] for t in er["inputs"]["fold_table"]], "row_consistency_max_abs_by_row": pd.provenance["row_consistency_max_abs_by_row"]}
    if a.fraction == 10:
        gb["identical_fit_logs_as_epr1"] = bool(gb["manifest_sha256"] == gb["epr1_manifest_sha256"] == EPR1_MANIFEST_SHA and gb["ranked_sha256"] == gb["epr1_ranked_sha256"]
                                                and gb["n_logs"] == gb["epr1_n_logs"] and gb["n_sets"] == gb["epr1_n_sets"] and gb["per_fold_sets"] == gb["epr1_per_fold_sets"] == list(EPR1_FOLD_SETS))
    else:
        gb["identical_fit_logs_as_epr1"] = True
        gb["note"] = f"fraction {a.fraction}: the f10 identity pins do not apply; the landed _setup verified the registered ranked logs and the bank"
    checks["G-BANK"] = bool(gb["identical_fit_logs_as_epr1"])
    # G-LIVE: the reach proof on a real batch, all rows and arms
    live = {}
    try:
        for r in ROWS:
            for arm in ARMS:
                live[f"{r}/{arm}"] = reach_proof(pd, W6, b6, r, arm)
        checks["G-LIVE"] = True
    except ReachError as ex:
        checks["G-LIVE"] = False
        reasons.append(str(ex)[:500])
    # G-SUITE: the package's synthetic tests (a RED mutation arm per check lives inside them)
    suite = {"ran": False}
    tp = os.path.join(HERE, "test_headrow_probe.py")
    if os.path.exists(tp):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        pr = subprocess.run([sys.executable, "-I", tp], capture_output=True, text=True, env=env, cwd=HERE)
        tail = (pr.stderr or "").strip().splitlines()[-3:]
        suite = {"ran": True, "returncode": pr.returncode, "tail": tail, "stderr_sha256": E.sha256_bytes((pr.stderr or "").encode("utf-8"))}
        checks["G-SUITE"] = bool(pr.returncode == 0 and any(t.startswith("OK") for t in tail))
    else:
        checks["G-SUITE"] = False
        reasons.append("test_headrow_probe.py is missing")
    ok = all(checks.values())
    tok = {"schema": SCHEMA, "ok": ok, "checks": checks, "reasons": reasons, "minted": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "bindings": {"cfg_sha256": cfg.sha256(), "code_sha256": shas, "bank_manifest_sha256": man_sha, "ranked_sha256": ranked_sha, "epr1_result_sha256": E.sha256_bytes(E._io_read_bytes(guard, a.epr1_result)),
                        "p2_rows_sha256": {r: E.sha256_bytes(E.row_bytes(W6[ROW_COL[r]], b6[ROW_COL[r]:ROW_COL[r] + 1])) for r in ROWS}, "fraction": a.fraction},
           "g_commit": gc, "g_dvb": dvb, "g_bank": gb, "g_live": live, "g_suite": suite}
    E._io_write_bytes(guard, tok_path, E.canon_json(tok), exclusive=False)         # a token is re-minted after any fix; it is never hand-edited
    _log(f"gate: {'PASS' if ok else 'FAIL'} {checks} -> {tok_path}")
    if not ok:
        raise ProbeError("gate FAILED: " + json.dumps(checks) + " " + "; ".join(reasons))
    return tok


def require_gate(guard, a: argparse.Namespace, man_sha: str, ranked_sha: str) -> dict:
    out = a.out_dir or f"{DEFAULT_RUN_ROOT}/f{a.fraction}"
    p = os.path.join(out, "gate_token.json")
    if not E._io_exists(guard, p):
        raise ProbeError(f"REFUSED: {p} does not exist (no fit without a gate pass)")
    tok = json.loads(E._io_read_bytes(guard, p).decode("utf-8"))
    cur = {"cfg_sha256": ProbeCfg().sha256(), "code_sha256": code_shas(guard), "bank_manifest_sha256": man_sha, "ranked_sha256": ranked_sha, "fraction": a.fraction}
    bad = [k for k, v in cur.items() if tok["bindings"].get(k) != v]
    if not tok.get("ok") or bad:
        raise ProbeError(f"REFUSED: gate token not OK or bindings differ: {bad}")
    return tok


def _setup_stage(a: argparse.Namespace, need_gate: bool = True):
    opts = make_opts(a)
    guard, out, manifest, man_sha, ranked_sha, selected, per_fold_logs, W6, b6, pd = load_probe_data(opts)
    tok = require_gate(guard, a, man_sha, ranked_sha) if need_gate else None
    return opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok


def stage_cid(a: argparse.Namespace) -> dict:
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = _setup_stage(a)
    res = {"schema": SCHEMA, "stage": "cid", "cells": {}}
    for r in ROWS:
        for arm in ARMS:
            res["cells"][f"{r}/{arm}"] = identity_check(pd, W6, b6, r, arm)
    res["ok"] = all(c["ok"] for c in res["cells"].values())
    write_json(guard, os.path.join(out, "cid.json"), res)
    _log(f"C-ID {'GREEN' if res['ok'] else 'RED'}")
    if not res["ok"]:
        raise ProbeError("C-ID RED: the run aborts before any fit")
    return res


def stage_main(a: argparse.Namespace) -> dict:
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = _setup_stage(a)
    cfg = ProbeCfg()
    if not E._io_exists(guard, os.path.join(out, "cid.json")):
        raise ProbeError("REFUSED: cid.json missing (C-ID must be GREEN before any fit)")
    odir = os.path.join(out, "oof")
    E._io_makedirs(guard, odir)
    for r in ROWS:
        for arm in ARMS:
            for sd in cfg.seeds:
                p = os.path.join(odir, f"oof_{r}_{arm}_s{sd}.npz")
                if E._io_exists(guard, p):
                    _log(f"skip (exists) {p}")
                    continue
                t0 = time.time()
                res = oof_logits(pd, cfg, arm, r, sd, W6, b6, ram_guard=True, log=_log)
                E._io_write_bytes(guard, p, npz_bytes(Z=res["Z"], train_bce_by_fold=np.array(res["train_bce_by_fold"]), fit_s_by_fold=np.array(res["fit_s_by_fold"])), exclusive=True)
                _log(f"main {r}/{arm}/s{sd} done in {time.time() - t0:.0f}s train_bce {np.mean(res['train_bce_by_fold']):.5f}")
    return {"ok": True}


def stage_c2(a: argparse.Namespace) -> dict:
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = _setup_stage(a)
    cfg = ProbeCfg()
    if not E._io_exists(guard, os.path.join(out, "cid.json")):
        raise ProbeError("REFUSED: cid.json missing (C-ID must be GREEN before any fit)")
    odir = os.path.join(out, "c2")
    E._io_makedirs(guard, odir)
    for r in ROWS:
        col = ROW_COL[r]
        for rep in range(1, cfg.c2_reps + 1):
            y_shuf = shuffle_valid(pd.y[col], pd.valid[col], c2_seed(rep, r))
            for arm in ARMS:
                p = os.path.join(odir, f"c2_{r}_{arm}_rep{rep}.npz")
                if E._io_exists(guard, p):
                    _log(f"skip (exists) {p}")
                    continue
                t0 = time.time()
                res = oof_logits(pd, cfg, arm, r, C2_SEED, W6, b6, y_train=y_shuf, cold=True, ram_guard=True, log=_log)
                E._io_write_bytes(guard, p, npz_bytes(Z=res["Z"], train_bce_by_fold=np.array(res["train_bce_by_fold"]), fit_s_by_fold=np.array(res["fit_s_by_fold"])), exclusive=True)
                _log(f"c2 {r}/{arm}/rep{rep} done in {time.time() - t0:.0f}s")
    return {"ok": True}


def stage_cpos(a: argparse.Namespace) -> dict:
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = _setup_stage(a)
    cfg = ProbeCfg()
    odir = os.path.join(out, "cpos")
    E._io_makedirs(guard, odir)
    y_pos, info = planted_labels(pd.feat)
    E._io_write_bytes(guard, os.path.join(odir, "planted_labels.npy"), npy_bytes(y_pos.numpy()), exclusive=True)
    col = ROW_COL["ttc"]
    pdp = dataclasses.replace(pd, y={**pd.y, col: y_pos}, valid={**pd.valid, col: torch.ones_like(pd.valid[col])})
    for arm in ARMS:
        for sd in cfg.seeds:
            p = os.path.join(odir, f"cpos_{arm}_s{sd}.npz")
            if E._io_exists(guard, p):
                continue
            t0 = time.time()
            res = oof_logits(pdp, cfg, arm, "ttc", sd, W6, b6, y_train=y_pos, cold=True, ram_guard=True, log=_log)
            E._io_write_bytes(guard, p, npz_bytes(Z=res["Z"], train_bce_by_fold=np.array(res["train_bce_by_fold"]), fit_s_by_fold=np.array(res["fit_s_by_fold"])), exclusive=True)
            _log(f"cpos {arm}/s{sd} done in {time.time() - t0:.0f}s")
    write_json(guard, os.path.join(odir, "planted_info.json"), info)
    return {"ok": True}


def stage_noise(a: argparse.Namespace) -> dict:
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = _setup_stage(a)
    draws = E.boot_draws(pd.n_logs)
    pk0 = E.picks_np(pd.z6.numpy())
    cells = []
    for r in ROWS:
        for i, sg in enumerate(NOISE_SIGMAS):
            for ns in NOISE_SEEDS:
                cells.append(noise_cell(pd, r, sg, ns, i, draws, pk0))
    write_json(guard, os.path.join(out, "noise.json"), {"schema": SCHEMA, "stage": "noise", "cells": cells, "estimator": "paired log-cluster bootstrap, B=10000, seed 20260927"})
    _log(f"noise: {len(cells)} cells")
    return {"ok": True}


def stage_ares(a: argparse.Namespace) -> dict:
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = _setup_stage(a)
    draws = E.boot_draws(pd.n_logs)
    er = json.loads(E._io_read_bytes(guard, a.epr1_result).decode("utf-8"))
    p2w, p2b = np.asarray(W6, np.float32)[E.EP_COL], np.asarray(b6, np.float32)[E.EP_COL:E.EP_COL + 1]
    pk_old = E.picks_np(pd.z6.numpy())
    res = {"schema": SCHEMA, "stage": "ares", "per_seed": {}, "controls": {}}
    pk_news, rows_used = {}, {}
    for sd in (0, 1):
        p = os.path.join(os.path.dirname(a.epr1_result), f"ep_row_seed{sd}.npy")
        raw = E._io_read_bytes(guard, p)
        sha = E.sha256_bytes(raw)
        if sha != EPR1_ROW_PINS[sd] or sha != er["final_rows"][str(sd)]["sha256"]:
            raise ProbeError(f"EP-R1 f10 row seed {sd}: sha256 {sha} != the registered pin")
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        w, b = arr[:FEAT_DIM], arr[FEAT_DIM:FEAT_DIM + 1]
        rows_used[sd] = (w, b)
        pk_news[sd] = E.picks_np(apply_row_chunked(pd.z6, pd.feat, w, b, p2w, p2b, E.EP_COL).numpy())
        st = ares_stats(pd, pk_news[sd], pk_old, draws)
        st["row_sha256"] = sha
        st["changed_pick_fraction"] = float((pk_news[sd] != pk_old).mean())
        res["per_seed"][str(sd)] = st
    res["flag"] = ares_flag(res["per_seed"])
    # controls
    pk_c0 = E.picks_np(apply_row_chunked(pd.z6, pd.feat, p2w, p2b, p2w, p2b, E.EP_COL).numpy())
    c0 = ares_stats(pd, pk_c0, pk_old, draws)
    res["controls"]["C0_identity"] = {"picks_equal_p2": bool(np.array_equal(pk_c0, pk_old)), "n_V": c0["n_V"], "n_S": c0["n_S"],
                                      "ok": bool(np.array_equal(pk_c0, pk_old) and c0["n_V"] == 0 and c0["n_S"] == 0)}
    rng = np.random.default_rng(NULL_BASE)
    perm = np.argsort(rng.random((pd.n_sets, 8)), axis=1)
    safe8_sh = np.take_along_axis(np.take_along_axis(pd.safe, pd.top8.numpy(), 1), perm, 1)
    z = pd.z6.numpy().astype(np.float64)
    s = -np.logaddexp(0.0, -z[..., 0]) - np.logaddexp(0.0, -z[..., 1])
    an, ad = auc_pairs_per_set(np.take_along_axis(s, pd.top8.numpy(), 1), safe8_sh)
    ci = ratio_ci(per_log(pd, an), per_log(pd, ad), draws)
    res["controls"]["C1prime_shuffled_safe_flags"] = {"auc": ci, "covers_half": bool(ci["lo"] <= 0.5 <= ci["hi"])}
    sub = np.sort(np.random.default_rng(NULL_BASE + 1).choice(pd.n_sets, 500, replace=False))
    bf = ares_bruteforce(pd, pk_news[0], pk_old, sub)
    sub_pd = dataclasses.replace(pd, z6=pd.z6[sub], top8=pd.top8[sub], safe=pd.safe[sub], log_idx=pd.log_idx[sub], fold=pd.fold[sub], lb=pd.lb[sub], pdms=pd.pdms[sub], feat=pd.feat[:1], y={}, valid={})
    vec = ares_stats(sub_pd, pk_news[0][sub], pk_old[sub], draws)
    vec_m_ge0 = int(round(vec["frac_V_m_ge0"]["point"] * vec["n_V"])) if vec["frac_V_m_ge0"] else 0
    res["controls"]["C2prime_independent_restatement"] = {"brute": bf, "vectorised": {"n_V": vec["n_V"], "n_V_m_ge0": vec_m_ge0}, "ok": bool(bf["n_V"] == vec["n_V"] and bf["n_V_m_ge0"] == vec_m_ge0)}
    flipped = ares_stats(pd, pk_news[0], pk_old, draws, flip_sign=True)
    f0, f1 = res["per_seed"]["0"]["frac_V_m_ge0"], flipped["frac_V_m_ge0"]
    res["controls"]["C3_mutation_sign_of_m"] = {"fraction": f0["point"] if f0 else None, "flipped": f1["point"] if f1 else None,
                                                 "red_as_required": bool(f0 and f1 and f0["point"] != f1["point"])}
    write_json(guard, os.path.join(out, "ares.json"), res)
    _log(f"A-RES flag {res['flag']['flag']}")
    return res


def stage_nullrate(a: argparse.Namespace) -> dict:
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = _setup_stage(a, need_gate=False)
    draws = E.boot_draws(pd.n_logs)
    res = {"schema": SCHEMA, "stage": "nullrate", "trials": NULL_TRIALS, "reps": C2_REPS, "rows": {}}
    for r in ROWS:
        t0 = time.time()
        trs = [null_trial(pd, r, draws, t) for t in range(NULL_TRIALS)]
        k_trial = int(sum(t["red"] for t in trs))
        k_rep = int(sum(t["n_rep_red"] for t in trs))
        res["rows"][r] = {"trials_with_red": k_trial, "rate_per_cell": k_trial / NULL_TRIALS, "wilson95_cell": wilson(k_trial, NULL_TRIALS), "replicates_red": k_rep,
                          "rate_per_replicate": k_rep / (NULL_TRIALS * C2_REPS), "wilson95_replicate": wilson(k_rep, NULL_TRIALS * C2_REPS), "seconds": time.time() - t0}
        _log(f"nullrate {r}: cell {k_trial}/{NULL_TRIALS} replicate {k_rep}/{NULL_TRIALS * C2_REPS}")
    p_cell = float(np.mean([res["rows"][r]["rate_per_cell"] for r in ROWS]))
    res["mean_rate_per_cell"] = p_cell
    res["family_rate_8_cells"] = 1.0 - (1.0 - p_cell) ** 8
    write_json(guard, os.path.join(out, "nullrate.json"), res)
    return res


def _load_oof(guard, path: str) -> dict:
    z = np.load(io.BytesIO(E._io_read_bytes(guard, path)), allow_pickle=False)
    return {k: z[k] for k in z.files}


def stage_metrics(a: argparse.Namespace) -> dict:
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = _setup_stage(a)
    cfg = ProbeCfg()
    draws = E.boot_draws(pd.n_logs)
    pk0 = E.picks_np(pd.z6.numpy())
    res = {"schema": SCHEMA, "stage": "metrics", "fraction": a.fraction, "n_logs": pd.n_logs, "n_sets": pd.n_sets, "gate_token_sha256": E.sha256_bytes(E._io_read_bytes(guard, os.path.join(out, "gate_token.json"))),
           "cfg_sha256": cfg.sha256(), "estimator": "paired log-cluster bootstrap, B=10000, seed 20260927, percentile 2.5/97.5 (answers: another draw of LOGS only)", "rows": {}}
    nullr = json.loads(E._io_read_bytes(guard, os.path.join(out, "nullrate.json")).decode("utf-8")) if E._io_exists(guard, os.path.join(out, "nullrate.json")) else None
    noise = json.loads(E._io_read_bytes(guard, os.path.join(out, "noise.json")).decode("utf-8")) if E._io_exists(guard, os.path.join(out, "noise.json")) else None
    # C-POS
    cpos = {"seeds": {}}
    if E._io_exists(guard, os.path.join(out, "cpos", "planted_labels.npy")):
        y_pos = torch.from_numpy(np.load(io.BytesIO(E._io_read_bytes(guard, os.path.join(out, "cpos", "planted_labels.npy")))))
        v1 = torch.ones_like(pd.valid[3])
        for sd in cfg.seeds:
            lg = {}
            for arm in ARMS:
                z = _load_oof(guard, os.path.join(out, "cpos", f"cpos_{arm}_s{sd}.npz"))["Z"]
                n64, d64 = conc_per_set(z, y_pos, v1)
                lg[arm] = (per_log(pd, n64), per_log(pd, d64))
            cpos["seeds"][str(sd)] = {"L": ratio_ci(*lg["L"], draws), "M": ratio_ci(*lg["M"], draws), "M_minus_L": diff_ci(lg["M"][0], lg["L"][0], lg["M"][1], draws)}
        cpos["green"] = bool(all(cpos["seeds"][str(sd)]["M_minus_L"]["lo"] > CAP_MARGIN for sd in cfg.seeds))
    else:
        cpos["green"] = None
    res["c_pos"] = cpos
    cpos_green = bool(cpos["green"])
    calls = {}
    for r in ROWS:
        col = ROW_COL[r]
        ref_logs = p2_logs(pd, r)
        ref = cell_report(ref_logs, ref_logs, draws)
        row = {"p2_reference": {k: ref[k] for k in ("conc64", "conc8", "brier", "unsafe_rate_p2", "pdms_x100_p2")}, "seeds": {}}
        per_seed_dec = {}
        for sd in cfg.seeds:
            cells, logs = {}, {}
            for arm in ARMS:
                o = _load_oof(guard, os.path.join(out, "oof", f"oof_{r}_{arm}_s{sd}.npz"))
                logs[arm] = cell_logs(pd, r, o["Z"], pk0)
                cells[arm] = cell_report(logs[arm], ref_logs, draws)
                cells[arm]["train_bce_mean_over_folds"] = float(np.mean(o["train_bce_by_fold"]))
                cells[arm]["fit_seconds_total"] = float(np.sum(o["fit_s_by_fold"]))
            ml = pair_report(logs["M"], logs["L"], draws)
            row["seeds"][str(sd)] = {"L": cells["L"], "M": cells["M"], "M_minus_L": ml}
            per_seed_dec[sd] = {"ml": ml["conc64"], "l_conc": cells["L"]["conc64"]["point"], "p2_conc": ref["conc64"]["point"]}
        # C2
        c2 = {"reps": {}, "cells": {}}
        red_any = False
        for arm in ARMS:
            cis = []
            for rep in range(1, cfg.c2_reps + 1):
                pth = os.path.join(out, "c2", f"c2_{r}_{arm}_rep{rep}.npz")
                if not E._io_exists(guard, pth):
                    cis = None
                    break
                z = _load_oof(guard, pth)["Z"]
                n64, d64 = conc_per_set(z, pd.y[col], pd.valid[col])
                cis.append(ratio_ci(per_log(pd, n64), per_log(pd, d64), draws))
            if cis is None:
                c2["cells"][arm] = {"missing": True}
                continue
            c2["cells"][arm] = {"replicates": cis, "red": bool(c2_cell_red(cis)), "min_lo_minus_half": min(c["lo"] - 0.5 for c in cis), "max_lo_minus_half": max(c["lo"] - 0.5 for c in cis)}
            red_any = red_any or c2["cells"][arm]["red"]
        mls = []
        if all("replicates" in c2["cells"].get(arm, {}) for arm in ARMS):
            for rep in range(1, cfg.c2_reps + 1):
                ln = {}
                for arm in ARMS:
                    z = _load_oof(guard, os.path.join(out, "c2", f"c2_{r}_{arm}_rep{rep}.npz"))["Z"]
                    n64, d64 = conc_per_set(z, pd.y[col], pd.valid[col])
                    ln[arm] = (per_log(pd, n64), per_log(pd, d64))
                mls.append(diff_ci(ln["M"][0], ln["L"][0], ln["M"][1], draws))
            c2["dec_replicates_M_minus_L"] = mls
            c2["dec_red"] = bool(c2_dec_red(mls))
            red_any = red_any or c2["dec_red"]
            c2["complete"] = True
        else:
            c2["complete"] = False
        c2["row_red"] = bool(red_any)
        row["c2"] = c2
        dec = decide_row(per_seed_dec, c2_red=red_any if c2["complete"] else True, cpos_green=cpos_green)
        if not c2["complete"]:
            dec["reasons"].append("C2 incomplete: treated as RED for CAPACITY")
        row["decision"] = dec
        calls[r] = dec
        if r in ("nc", "dac"):
            du = {sd: row["seeds"][str(sd)]["M"]["d_unsafe_vs_p2"] for sd in cfg.seeds}
            row["c_leak"] = {"flag": cleak_flag(du), "M_d_unsafe_vs_p2": {str(k): v for k, v in du.items()}, "L_d_unsafe_vs_p2": {str(sd): row["seeds"][str(sd)]["L"]["d_unsafe_vs_p2"] for sd in cfg.seeds}}
            if noise:
                cells_r = [c for c in noise["cells"] if c["row"] == r]
                row["c_leak"]["matched_noise"] = {str(sd): match_noise(row["seeds"][str(sd)]["M"]["changed_pick_fraction"], cells_r) for sd in cfg.seeds}
        res["rows"][r] = row
    res["outcome"] = combine_outcome(calls, cpos_green)
    res["null_rate"] = nullr
    res["noise_file"] = bool(noise)
    write_json(guard, os.path.join(out, "probe_result.json"), res)
    _log("metrics written: " + " ".join(f"{r}={calls[r]['call']}" for r in ROWS))
    return res


STAGES = {"gate": stage_gate, "cid": stage_cid, "main": stage_main, "c2": stage_c2, "cpos": stage_cpos, "noise": stage_noise, "ares": stage_ares, "nullrate": stage_nullrate, "metrics": stage_metrics}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="HEADROW-PROBE. Never reads the dev gate or navtest tables.")
    ap.add_argument("stage", choices=tuple(STAGES))
    ap.add_argument("--fraction", type=int, default=10, choices=(10, 25))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epr1-result", default=EPR1_RESULT_DEFAULT)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    STAGES[a.stage](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
