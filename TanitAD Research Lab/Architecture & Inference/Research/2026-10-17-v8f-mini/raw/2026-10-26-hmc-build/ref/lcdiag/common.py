"""Shared loaders for the lcdiag probes (DESCRIPTIVE; decides nothing). The row functions are NOT re-defined: ``comps`` (t1k_read.py) and ``lat_rows``
(t1klc_read.py) are exec'd from their banked source by AST, exactly as t1klc_read.py does, and every per-pair decomposition below is checked against them."""
import ast
import math
import os
import sys

import numpy as np

SCR = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/t1kin"
ROOT = "C:/Users/Admin/t1k"
sys.path.insert(0, os.path.join(ROOT, "taniteval"))
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB  # noqa: E402

DT, VMIN = 0.2, 5.2
ns = {"np": np, "torch": torch, "FF": FF, "PB": PB, "math": math, "DT": DT, "VMIN": VMIN}
for src, names in ((f"{SCR}/t1k_read.py", ("comps", "paired")), (f"{SCR}/t1klc_read.py", ("lat_rows",))):
    for node in ast.parse(open(src, encoding="utf-8").read()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), src, "exec"), ns)
comps, paired, lat_rows = ns["comps"], ns["paired"], ns["lat_rows"]

DUMPS = {
    "causal": (f"{SCR}/consolidated_c", "dumps_HACC_s{s}_m10_causal.npz"),
    "logged": (f"{SCR}/consolidated", "dumps_HACC_s{s}_m10.npz"),
    "plainB_logged": (f"{SCR}/consolidated", "dumps_B_s{s}_m10.npz"),            # plain M2b, NO post-map (logged bucket)
    "strict": (f"{SCR}/consolidated_s", "dumps_HACC_s{s}_m10_strict.npz"),
    "hlatB": (f"{SCR}/consolidated_l", "dumps_B_s{s}_m10.npz"),                  # HACC-LAT arm B (NOT plain M2b)
    "hlatA": (f"{SCR}/consolidated_l", "dumps_A_s{s}_m10.npz"),
}


def load(cfg, s):
    d, pat = DUMPS[cfg]
    z = np.load(os.path.join(d, pat.format(s=s)))
    return {k: z[k] for k in z.files}


def pair_err(X, G, vmin=VMIN):
    """Per-pair L2 ingredients with lat_rows' OWN definitions: e [n,29] = |kappa_X - kappa_G|, fast mask [n,29], GT pair speed vG [n,29],
    and the per-window share ``sh`` = m e / c (sum over pairs = the window's L2 value; NaN-free, 0 where c = 0)."""
    Xt, Gt = torch.as_tensor(np.asarray(X, np.float64)), torch.as_tensor(np.asarray(G, np.float64))
    P, Q = FF._seq_geometry(Xt, DT), FF._seq_geometry(Gt, DT)
    bp = P["pair_valid"] & Q["pair_valid"]
    vG = 0.5 * (Q["speed"][:, 1:] + Q["speed"][:, :-1])
    fast = (bp & (vG >= vmin)).numpy()
    e = (P["curvature"] - Q["curvature"]).abs().numpy()
    c = fast.sum(1)
    sh = np.where(fast, e, 0.0) / np.maximum(c, 1)[:, None]
    return {"e": e, "fast": fast, "c": c, "sh": sh, "vG": vG.numpy(), "kX": P["curvature"].numpy(), "kG": Q["curvature"].numpy(),
            "hX": P["heading"].numpy(), "vX": (0.5 * (P["speed"][:, 1:] + P["speed"][:, :-1])).numpy()}


def check_against_rows(X, G, pe):
    """Control: the per-pair shares must sum to lat_rows' L2 window values (max |d| <= 1e-12, same NaN pattern)."""
    L2 = lat_rows(X, G)["L2_curvature_mae_fast_1pm"]
    mine = np.where(pe["c"] > 0, pe["sh"].sum(1), np.nan)
    same_nan = bool((np.isnan(L2) == np.isnan(mine)).all())
    d = float(np.nanmax(np.abs(L2 - mine)))
    return {"max_abs": d, "same_nan": same_nan, "PASS": same_nan and d <= 1e-12}


def boot(a, b, eid):
    r = paired(np.asarray(a, np.float64), np.asarray(b, np.float64), eid)
    return r
