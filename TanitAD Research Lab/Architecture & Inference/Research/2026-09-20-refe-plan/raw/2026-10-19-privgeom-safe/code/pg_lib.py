"""PRIVGEOM-SAFE library: the L / L+G / G-only head fits (out-of-fold), the within-token AUC with the paired log-cluster bootstrap, the REGISTERED decision functions, the controls.

REGISTERED SOURCE: PREREG_PRIVGEOM_SAFE.md (sha256 PREREG_SHA256) + Amendment 1 (the code pins).  CPU only.  The estimator helpers are the census's (`fastneg_lib`, pinned); the pick rule
is the pick-rule audit's (`pickrule_lib`, pinned).  Every constant cites the prereg section it comes from.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "ref")))
import fastneg_lib as FL                                                                 # noqa: E402

SCHEMA = "privgeom-safe-v1"
PREREG_SHA256 = "c6d6e1fb228f7e4806d5b0316699f8fef05855684d0c535bd246248217bd7084"
STRATA_SHA256 = "67f43e55dd7c693e31cd991efc3d51fedc278331ba4f8f882e6adbdae8203dc6"
# ---- recipe (s4) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
LR_TOK, LR_G, WD, BETAS, EPS = 1e-3, 1e-2, 1e-4, (0.9, 0.999), 1e-8
EPOCHS, BATCH_SETS = 20, 256
SEEDS = (0, 1)
ROWS = ("nc", "dac")
ROW_COL = {"nc": 0, "dac": 1}
# ---- decision (s6) -------------------------------------------------------------------------------------------------------------------------------------------------------------------
INFO_MARGIN, NOJUMP_MARGIN = 0.02, 0.01
NULL_REPS, NULL_SALT, NULL_SEED_BASE, NULL_VOID_FIRES = 5, 7919, 20261019, 2
PLANT_SIGMA, PLANT_SEED = 1.0, 20261019 + 31
REACH_MIN_AUC = 0.99
AGREE_MIN = 0.99
LOSS_FRAC_DISCLOSE = 0.05
BOOT_B, BOOT_SEED = 10000, 20260927


class PgError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the registered decision functions (pure; `branch_table_failures` is the table every RED mutation must break)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def info_call(ci: dict) -> bool:
    """s6: the lower CI of Delta is > +0.02."""
    return bool(ci["lo"] > INFO_MARGIN)


def no_jump_call(ci: dict) -> bool:
    """s6: the upper CI of Delta is < +0.01."""
    return bool(ci["hi"] < NOJUMP_MARGIN)


def seed_call(ci: dict) -> str:
    if info_call(ci):
        return "INFO-UPSTREAM"
    if no_jump_call(ci):
        return "NO-JUMP"
    return "INCONCLUSIVE"


def row_call(calls: list) -> str:
    """BOTH seeds must agree: INFO-UPSTREAM iff all INFO-UPSTREAM, NO-JUMP iff all NO-JUMP, else INCONCLUSIVE."""
    if len(calls) and all(c == "INFO-UPSTREAM" for c in calls):
        return "INFO-UPSTREAM"
    if len(calls) and all(c == "NO-JUMP" for c in calls):
        return "NO-JUMP"
    return "INCONCLUSIVE"


def null_fires(cis_by_seed: list) -> bool:
    """s7 C-NULL: a replicate fires for a row iff the INFO-UPSTREAM rule holds on BOTH seeds."""
    return bool(len(cis_by_seed) and all(info_call(c) for c in cis_by_seed))


def null_void(fires: list) -> bool:
    """s7: VOID iff the null fires on >= 2 of the replicates."""
    return bool(sum(1 for f in fires if f) >= NULL_VOID_FIRES)


def power_ok(cis_by_seed: list) -> bool:
    """s7 C-POWER: the registered INFO-UPSTREAM rule must FIRE on the weak plant on both seeds."""
    return bool(len(cis_by_seed) and all(info_call(c) for c in cis_by_seed))


def final_call(call: str, void_null: bool, void_inputs: bool, power: bool) -> str:
    """s6 conversions: they can only REMOVE a call."""
    if void_inputs:
        return "VOID-INPUTS"
    if void_null:
        return "VOID"
    if call == "NO-JUMP" and not power:
        return "UNDERPOWERED"
    return call


def branch_table_failures(mod=None) -> list:
    """Every entry must hold; a RED mutation of any decision function must make at least one entry fail."""
    m = mod if mod is not None else sys.modules[__name__]
    bad = []
    ci = lambda lo, hi: {"lo": lo, "hi": hi}                                                # noqa: E731

    def chk(name, got, want):
        if got != want:
            bad.append((name, got, want))
    chk("info lo just above", m.info_call(ci(0.0201, 0.05)), True)
    chk("info lo at bound", m.info_call(ci(0.02, 0.05)), False)
    chk("info lo below", m.info_call(ci(0.0199, 0.05)), False)
    chk("nojump hi just below", m.no_jump_call(ci(-0.05, 0.0099)), True)
    chk("nojump hi at bound", m.no_jump_call(ci(-0.05, 0.01)), False)
    chk("nojump hi above", m.no_jump_call(ci(-0.05, 0.0101)), False)
    chk("seed info", m.seed_call(ci(0.03, 0.06)), "INFO-UPSTREAM")
    chk("seed nojump", m.seed_call(ci(-0.03, 0.005)), "NO-JUMP")
    chk("seed straddle", m.seed_call(ci(-0.01, 0.05)), "INCONCLUSIVE")
    chk("seed straddle 2", m.seed_call(ci(0.0, 0.015)), "INCONCLUSIVE")
    chk("row both info", m.row_call(["INFO-UPSTREAM", "INFO-UPSTREAM"]), "INFO-UPSTREAM")
    chk("row one info", m.row_call(["INFO-UPSTREAM", "INCONCLUSIVE"]), "INCONCLUSIVE")
    chk("row info vs nojump", m.row_call(["INFO-UPSTREAM", "NO-JUMP"]), "INCONCLUSIVE")
    chk("row both nojump", m.row_call(["NO-JUMP", "NO-JUMP"]), "NO-JUMP")
    chk("row one nojump", m.row_call(["NO-JUMP", "INCONCLUSIVE"]), "INCONCLUSIVE")
    chk("row empty", m.row_call([]), "INCONCLUSIVE")
    chk("null fires both", m.null_fires([ci(0.03, 0.06), ci(0.025, 0.07)]), True)
    chk("null fires one seed", m.null_fires([ci(0.03, 0.06), ci(0.0, 0.07)]), False)
    chk("null void 2 of 5", m.null_void([True, True, False, False, False]), True)
    chk("null void 1 of 5", m.null_void([True, False, False, False, False]), False)
    chk("null void 0 of 5", m.null_void([False] * 5), False)
    chk("power both", m.power_ok([ci(0.03, 0.06), ci(0.03, 0.06)]), True)
    chk("power one", m.power_ok([ci(0.03, 0.06), ci(0.01, 0.06)]), False)
    chk("final nojump unpowered", m.final_call("NO-JUMP", False, False, False), "UNDERPOWERED")
    chk("final info unpowered stays", m.final_call("INFO-UPSTREAM", False, False, False), "INFO-UPSTREAM")
    chk("final nojump powered", m.final_call("NO-JUMP", False, False, True), "NO-JUMP")
    chk("final void null", m.final_call("INFO-UPSTREAM", True, False, True), "VOID")
    chk("final void inputs", m.final_call("NO-JUMP", False, True, True), "VOID-INPUTS")
    chk("final void inputs over null", m.final_call("INFO-UPSTREAM", True, True, True), "VOID-INPUTS")
    return bad


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# G handling
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def standardise(g: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """g [N, 64, k] -> float32 standardised with the TRAINING sets' per-column mean / std over their proposals (std < 1e-8 -> the column is set to 0)."""
    k = g.shape[-1]
    tr = np.asarray(g[train_idx], np.float64).reshape(-1, k)
    mu, sd = tr.mean(axis=0), tr.std(axis=0)
    ok = sd >= 1e-8
    out = (np.asarray(g, np.float64) - mu) / np.where(ok, sd, 1.0)
    out[..., ~ok] = 0.0
    return out.astype(np.float32)


def permute_within_set(g: np.ndarray, seed: int) -> np.ndarray:
    """s7 C-NULL: one permutation of the 64 proposals PER SET moves the whole k-vector."""
    rng = np.random.default_rng(int(seed))
    N = g.shape[0]
    perm = np.argsort(rng.random((N, g.shape[1])), axis=1)
    return np.take_along_axis(g, perm[:, :, None], axis=1)


def weak_plant(label: np.ndarray, col: int) -> np.ndarray:
    """s7 C-POWER: label + 1.0 * N(0, 1), one column [N, 64, 1] float32."""
    eps = np.random.default_rng(PLANT_SEED + int(col)).standard_normal(label.shape)
    return (np.asarray(label, np.float64) + PLANT_SIGMA * eps).astype(np.float32)[..., None]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the head fits (s4)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Head:
    """One row's head.  arm 'L': z = z_P2 + feat.(w - w_P2) + (b - b_P2).  'LG': L + g.u (u = 0 at init, its own Adam group).  'G': z = g.u + c (no token, no P2 logit)."""

    def __init__(self, arm: str, p2_w, p2_b, k: int, lr_tok: float = LR_TOK, lr_g: float = LR_G, wd: float = WD):
        import torch
        self.torch, self.arm, self.k = torch, arm, int(k)
        self.use_tok = arm in ("L", "LG")
        self.p2_w = torch.as_tensor(np.asarray(p2_w, np.float32).copy())
        self.p2_b = torch.as_tensor(np.asarray([p2_b], np.float32).copy())
        groups = []
        if self.use_tok:
            self.w = torch.nn.Parameter(self.p2_w.clone())
            self.b = torch.nn.Parameter(self.p2_b.clone())
            groups.append({"params": [self.w, self.b], "lr": lr_tok})
        if arm in ("LG", "G"):
            if self.k < 1:
                raise PgError(f"arm {arm} needs at least one G column")
            self.u = torch.nn.Parameter(torch.zeros(self.k, dtype=torch.float32))
            ps = [self.u]
            if arm == "G":
                self.c = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
                ps.append(self.c)
            groups.append({"params": ps, "lr": lr_g})
        self.opt = torch.optim.Adam(groups, betas=BETAS, eps=EPS, weight_decay=wd)

    def logit(self, feat32, z_st, g32):
        if self.arm == "G":
            return g32 @ self.u + self.c
        z = z_st + (feat32 @ (self.w - self.p2_w) + (self.b - self.p2_b))
        if self.arm == "LG":
            z = z + g32 @ self.u
        return z

    def step(self, feat16, z_st, y, g):
        F = self.torch.nn.functional
        feat32 = feat16.detach().float() if feat16 is not None else None
        z = self.logit(feat32, z_st, g)
        loss = F.binary_cross_entropy_with_logits(z, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.detach())

    def params(self) -> dict:
        out = {}
        if self.use_tok:
            out["w"], out["b"] = self.w.detach().numpy().copy(), self.b.detach().numpy().copy()
        if self.arm in ("LG", "G"):
            out["u"] = self.u.detach().numpy().copy()
        if self.arm == "G":
            out["c"] = self.c.detach().numpy().copy()
        return out


def fit_head(arm: str, feat, z_st: np.ndarray, y: np.ndarray, g, train_idx: np.ndarray, p2_w, p2_b, seed: int, epochs: int = EPOCHS, batch_sets: int = BATCH_SETS, ram_guard=None):
    """feat [N,64,256] fp16 (array or memmap); z_st, y [N,64] float32; g [N,64,k] float32 STANDARDISED or None.  Batch order as the census (`fastneg_lib.fit_rows`)."""
    import torch
    k = 0 if g is None else g.shape[-1]
    head = Head(arm, p2_w, p2_b, k)
    gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(len(train_idx)))
    tr = torch.as_tensor(np.asarray(train_idx), dtype=torch.long)
    n = len(tr)
    if ram_guard is not None:
        ram_guard()
    for _ in range(epochs):
        order = tr[torch.randperm(n, generator=gen)]
        for s in range(0, n, batch_sets):
            bi = order[s:s + batch_sets].numpy()
            f = torch.from_numpy(np.ascontiguousarray(feat[bi])) if head.use_tok else None
            gg = torch.from_numpy(np.ascontiguousarray(g[bi])) if g is not None else None
            head.step(f, torch.from_numpy(z_st[bi]), torch.from_numpy(y[bi]), gg)
    return head.params()


def score_head(arm: str, prm: dict, feat, z_st: np.ndarray, g, idx: np.ndarray, p2_w, p2_b, chunk: int = 1024) -> np.ndarray:
    import torch
    out = np.empty((len(idx), z_st.shape[1]), np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            ii = np.asarray(idx[s:s + chunk])
            if arm == "G":
                z = torch.from_numpy(np.ascontiguousarray(g[ii])) @ torch.as_tensor(prm["u"]) + torch.as_tensor(prm["c"])
            else:
                f = torch.from_numpy(np.ascontiguousarray(feat[ii])).float()
                zs = torch.from_numpy(z_st[ii])
                z = zs + (f @ (torch.as_tensor(prm["w"]) - torch.as_tensor(np.asarray(p2_w, np.float32))) + (torch.as_tensor(prm["b"]) - torch.as_tensor(np.asarray([p2_b], np.float32))))
                if arm == "LG":
                    z = z + torch.from_numpy(np.ascontiguousarray(g[ii])) @ torch.as_tensor(prm["u"])
            out[s:s + len(ii)] = z.numpy()
    return out


def oof_logits(arm: str, feat, z_st: np.ndarray, y: np.ndarray, g_raw, fold: np.ndarray, seed: int, p2_w, p2_b, ram_guard=None, trace=None) -> np.ndarray:
    """5-fold grouped-by-log out-of-fold logits [N, 64] (folds = the census fold of each set's log)."""
    N = len(fold)
    out = np.zeros((N, z_st.shape[1]), np.float32)
    for k in sorted(set(int(x) for x in fold)):
        tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
        gs = None if g_raw is None else standardise(g_raw, tr)
        prm = fit_head(arm, feat, z_st, y, gs, tr, p2_w, p2_b, seed, ram_guard=ram_guard)
        out[te] = score_head(arm, prm, feat, z_st, gs, te, p2_w, p2_b)
        if trace is not None:
            trace.append({"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)), **{kk: (vv.tolist() if kk in ("u", "c") else None) for kk, vv in prm.items() if kk in ("u", "c")}})
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Analyzer:
    """Within-token AUC in F (= Q4) and Q1-3 with the paired log-cluster bootstrap over the subset's logs."""

    def __init__(self, labels: np.ndarray, code: np.ndarray, log_idx: np.ndarray, n_logs: int, B: int = BOOT_B, seed: int = BOOT_SEED):
        self.labels, self.code, self.log_idx, self.n_logs = labels, code, log_idx, n_logs
        self.draws = FL.boot_draws(n_logs, B, seed)
        self.cls = {r: (labels[..., ROW_COL[r]] == 0.0, labels[..., ROW_COL[r]] == 1.0) for r in ROWS}      # A1.3: unsafe = 0, safe = 1 (0.5 excluded)

    def per_log_stats(self, z: np.ndarray, row: str) -> dict:
        u, s = self.cls[row]
        st = FL.pair_stats(z, u, s, self.code, names=("Q4", "Q123", "ALLQ"))
        return {k: (FL.per_log(v[0], self.log_idx, self.n_logs), FL.per_log(v[1], self.log_idx, self.n_logs)) for k, v in st.items()}

    def auc(self, stats: dict, stratum: str) -> dict:
        n, d = stats[stratum]
        return FL.ratio_ci(n, d, self.draws)

    def diff(self, stats_a: dict, stats_b: dict, stratum: str) -> dict:
        na, da = stats_a[stratum]
        nb, db = stats_b[stratum]
        return FL.diff_ci(na, da, nb, db, self.draws)
