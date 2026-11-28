"""FASTNEG-CENSUS library: speeds, strata, within-set AUC with a log-cluster bootstrap, the registered decision rule, the L-arm refit.

REGISTERED SOURCE: PREREG_FASTNEG_CENSUS.md (sha256 b6cece236d5aa77dd412b62341ee1691177fd3dabad30b3c297881ad6d9e5634) + PREREG_FASTNEG_CENSUS_AMENDMENT1.md
(sha256 f53ae097d153681bd2de07ae552a5d48b0b2d47f996408d1a75588c618a40491).  Every constant below cites its source.  Nothing here reads a label file by itself except the functions that
are handed label arrays; the ONLY file reader that the label-free strata step uses is `read_members`, which REFUSES any member whose name contains "label".
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
import zipfile

import numpy as np

SCHEMA = "fastneg-census-v1"
PREREG_SHA256 = "b6cece236d5aa77dd412b62341ee1691177fd3dabad30b3c297881ad6d9e5634"
AMENDMENT1_SHA256 = "f53ae097d153681bd2de07ae552a5d48b0b2d47f996408d1a75588c618a40491"
EPR1_SHA256 = "a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8"          # the landed epr1.py at the tip (also pinned by HEADROW-PROBE)

# ---- registered constants (source in the comment) -------------------------------------------------------------------------------------------------------------------------------
HORIZON_S, N_STEPS, DT_S = 4.0, 20, 0.2        # A1.2: 20 steps @ 5 Hz = 4 s (refe/model.py:94; SRC_DT in srttc_s0_dump.py)
N_PROP, FEAT_DIM = 64, 256
STAND_V = 0.5                                  # parent s3: standstill iff v_b < 0.5 m/s (strict)
BOOT_B, BOOT_SEED = 10000, 20260927            # parent s4
DEFICIT_MARGIN = 0.02                          # parent s5
SLOPE_MARGIN = 0.01                            # parent s5
SEEDS = (0, 1)                                 # parent s2
SIZES = (25, 50, 100)                          # parent s4(c)
EPOCHS, BATCH_SETS = 20, 256                   # parent s4(c) / HEADROW-PROBE s3.2
LR, WD, BETAS, EPS = 1e-3, 1e-4, (0.9, 0.999), 1e-8
SUB_SEED_BASE, SUB_SEED_MULT = 20261019, 7     # A1.4
NULL_REPS, NULL_SALT = 5, 7919                 # A1.5
NULL_VOID_FIRES = 2                            # parent s4(c0): VOID iff the null fires on >= 2 of 5
MIN_AVAIL_GB = 3.0                             # brief: psutil available >= 3 GB before each fit
N_FOLDS = 5
ROWS = ("nc", "dac")
ROW_COL = {"nc": 0, "dac": 1}
STRATA_NAMES = ("S", "Q1", "Q2", "Q3", "Q4")
BANK_MEMBERS = ("log", "token", "step", "rank", "ckpt_step", "logits")    # the ONLY bank members the label-free strata step reads
STORE_MEMBERS = ("token", "step", "rank", "ckpt_step", "traj3")           # the ONLY store members the label-free strata step reads


class LabelAccessError(RuntimeError):
    """A label-free step tried to open a member whose name contains 'label'."""


class CensusError(RuntimeError):
    """An integrity check, a pin or a precondition failed."""


# ---- hashing / json ------------------------------------------------------------------------------------------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def canon_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=True).encode("utf-8")


def arrays_digest(arrs: dict) -> str:
    """Deterministic digest of named arrays (name, dtype, shape, raw bytes): independent of any zip timestamp."""
    h = hashlib.sha256()
    for k in sorted(arrs):
        a = np.ascontiguousarray(arrs[k])
        h.update(k.encode()), h.update(str(a.dtype).encode()), h.update(str(a.shape).encode()), h.update(a.tobytes())
    return h.hexdigest()


# ---- the label-free reader -----------------------------------------------------------------------------------------------------------------------------------------------------
OPENED: list = []          # (file basename, member) for every member this module has read through `read_members` (the spy record that strata.json embeds)


def read_members(path: str, names) -> dict:
    """Read ONLY the named members of an npz through zipfile.  Refuses any name containing 'label' BEFORE touching the file (a label-free step cannot use this to read labels)."""
    for n in names:
        if "label" in str(n).lower():
            raise LabelAccessError(f"REFUSED: member {n!r} of {path!r} is a label member; this step is label-free")
    out = {}
    with zipfile.ZipFile(path, "r") as zf:
        have = set(zf.namelist())
        for n in names:
            m = n + ".npy"
            if m not in have:
                raise CensusError(f"{path}: no member {n!r}")
            with zf.open(m, "r") as f:
                out[n] = np.lib.format.read_array(f, allow_pickle=False)
            OPENED.append((os.path.basename(path), n))
    return out


# ---- landed epr1 (pinned) ------------------------------------------------------------------------------------------------------------------------------------------------------
def find_epr1(explicit: str | None = None) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [explicit] if explicit else []
    cands += [os.path.join(here, "..", "..", "2026-10-18-ep-r1", "code", "epr1.py"),
              os.path.join(here, "epr1.py"),
              "C:/Users/Admin/par18/fastneg/src/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-ep-r1/code/epr1.py"]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.normpath(c)
    raise CensusError("the landed epr1.py was not found; probed: " + "; ".join(str(c) for c in cands))


def load_epr1(explicit: str | None = None):
    """Import the LANDED epr1.py after checking its sha256 against the pin (a tampered copy is refused)."""
    if "epr1" in sys.modules and getattr(sys.modules["epr1"], "_FASTNEG_PINNED", False):
        return sys.modules["epr1"]
    p = find_epr1(explicit)
    sha = sha256_file(p)
    if sha != EPR1_SHA256:
        raise CensusError(f"epr1.py at {p} has sha256 {sha} != the pinned {EPR1_SHA256}")
    spec = importlib.util.spec_from_file_location("epr1", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["epr1"] = mod
    spec.loader.exec_module(mod)
    mod._FASTNEG_PINNED = True
    return mod


# ---- speeds and strata ---------------------------------------------------------------------------------------------------------------------------------------------------------
def segment_lengths(traj3: np.ndarray) -> np.ndarray:
    """[..., 20, 3] ego-frame (x, y, yaw) -> [..., 20] Euclidean xy step lengths; the first segment starts at the ego origin (0, 0) at t0 (A1.2)."""
    xy = np.asarray(traj3, np.float64)[..., :2]
    prev = np.concatenate([np.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], axis=-2)
    return np.sqrt(((xy - prev) ** 2).sum(-1))


def mean_speed(traj3: np.ndarray) -> np.ndarray:
    """Mean speed over the 4 s horizon = path length / 4.0 s (A1.2)."""
    return segment_lengths(traj3).sum(-1) / HORIZON_S


def build_strata(v: np.ndarray, v_b: np.ndarray) -> tuple:
    """v [N,64] proposal mean speeds, v_b [N] base-pick mean speeds -> (r [N,64] float64, nan on S sets), cuts (c25, c50, c75), code [N,64] uint8: 0 = S, 1..4 = Q1..Q4.
    A1.2: S = every proposal of a set with v_b < 0.5; quartiles over every proposal of every non-S set (numpy.quantile, linear); ties go to the LOWER stratum (r <= c)."""
    v = np.asarray(v, np.float64)
    v_b = np.asarray(v_b, np.float64)
    s_set = v_b < STAND_V
    r = np.full(v.shape, np.nan)
    ns = ~s_set
    r[ns] = v[ns] / v_b[ns][:, None]
    cuts = np.quantile(r[ns].ravel(), [0.25, 0.5, 0.75]) if ns.any() else np.array([np.nan] * 3)
    code = np.zeros(v.shape, np.uint8)
    rn = r[ns]
    code[ns] = (1 + (rn > cuts[0]).astype(np.int64) + (rn > cuts[1]).astype(np.int64) + (rn > cuts[2]).astype(np.int64)).astype(np.uint8)
    return r, cuts, code


# ---- AUC (pairs inside one set) + the log-cluster bootstrap ---------------------------------------------------------------------------------------------------------------------
def group_masks(ci: np.ndarray, cj: np.ndarray) -> dict:
    """Pair-membership masks.  ci = stratum code of the UNSAFE member, cj = of the SAFE member (broadcast [n,64,1] / [n,1,64]).
    WITHIN-stratum groups (A1.3): both members in the stratum.  S, Q1..Q4, Q123 = Q1 u Q2 u Q3.  Cross groups (descriptive): the unsafe member in the stratum, the safe one anywhere in the set (x*)."""
    g = {"S": (ci == 0) & (cj == 0)}
    for k in (1, 2, 3, 4):
        g[f"Q{k}"] = (ci == k) & (cj == k)
    g["Q123"] = (ci >= 1) & (ci <= 3) & (cj >= 1) & (cj <= 3)
    g["ALLQ"] = (ci >= 1) & (cj >= 1)
    for k in (1, 2, 3, 4):
        g[f"xQ{k}"] = (ci == k) & (cj >= 1)
    g["xQ123"] = (ci >= 1) & (ci <= 3) & (cj >= 1)
    g["xS"] = (ci == 0) & (cj == 0)
    return g


def pair_stats(z: np.ndarray, unsafe: np.ndarray, safe: np.ndarray, code: np.ndarray, chunk: int = 512, names=None) -> dict:
    """Per-set concordance numerators and pair counts, for every group.  z [N,64] scores (higher = safer); unsafe, safe [N,64] bool; code [N,64] uint8.
    A pair (u unsafe, v safe) is concordant iff z_v > z_u, a tie counts 0.5.  -> {group: (num [N] float64, den [N] float64)}."""
    N = len(z)
    acc: dict = {}
    for s in range(0, N, chunk):
        sl = slice(s, s + chunk)
        zz = np.asarray(z[sl], np.float32)
        un, sf, cd = unsafe[sl], safe[sl], code[sl]
        zi, zj = zz[:, :, None], zz[:, None, :]
        w = (zj > zi).astype(np.float32) + 0.5 * (zj == zi).astype(np.float32)
        pv = un[:, :, None] & sf[:, None, :]
        for name, m in group_masks(cd[:, :, None], cd[:, None, :]).items():
            if names is not None and name not in names:
                continue
            mm = pv & m
            num = (w * mm).sum((1, 2), dtype=np.float64)
            den = mm.sum((1, 2), dtype=np.float64)
            a = acc.setdefault(name, ([], []))
            a[0].append(num)
            a[1].append(den)
    return {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in acc.items()}


def per_log(vals: np.ndarray, log_idx: np.ndarray, n_logs: int) -> np.ndarray:
    return np.bincount(log_idx, weights=vals, minlength=n_logs).astype(np.float64)


def boot_draws(n_logs: int, B: int = BOOT_B, seed: int = BOOT_SEED) -> np.ndarray:
    """Identical to the landed `epr1.boot_draws` (a test compares them)."""
    return np.random.default_rng(seed).integers(0, n_logs, size=(B, n_logs))


def _ci(point: float, v: np.ndarray) -> dict:
    ok = ~np.isnan(v)
    if not ok.any():
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_nan": int(len(v))}
    return {"point": float(point), "lo": float(np.percentile(v[ok], 2.5)), "hi": float(np.percentile(v[ok], 97.5)), "n_nan": int((~ok).sum())}


def ratio_ci(num: np.ndarray, den: np.ndarray, draws: np.ndarray) -> dict:
    """sum(num)/sum(den) over logs with the percentile interval of the log-cluster bootstrap (the unit is the LOG)."""
    d = den.sum()
    point = float(num.sum() / d) if d > 0 else float("nan")
    bd = den[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(bd > 0, num[draws].sum(axis=1) / bd, np.nan)
    return {**_ci(point, v), "pairs": float(d)}


def diff_ci(num_a, den_a, num_b, den_b, draws: np.ndarray) -> dict:
    """c_a - c_b with c = ratio of sums over the SAME resampled logs (paired).  The two denominators may differ (F against pooled Q1-Q3)."""
    da, db = den_a.sum(), den_b.sum()
    point = float(num_a.sum() / da - num_b.sum() / db) if da > 0 and db > 0 else float("nan")
    ba, bb = den_a[draws].sum(axis=1), den_b[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where((ba > 0) & (bb > 0), num_a[draws].sum(axis=1) / ba - num_b[draws].sum(axis=1) / bb, np.nan)
    return {**_ci(point, v), "pairs_a": float(da), "pairs_b": float(db)}


# ---- the registered decision rule (parent s5 / s6, amended s A1.5 / A1.6) -----------------------------------------------------------------------------------------------------
def deficit_call(ci: dict) -> str:
    """ci = the interval of AUC_F - AUC_Q123 (one row, one seed, L arm at 100 %).  DEFICIT iff hi < -0.02; NO_DEFICIT iff lo > -0.02; else INCONCLUSIVE."""
    if ci["hi"] < -DEFICIT_MARGIN:
        return "DEFICIT"
    if ci["lo"] > -DEFICIT_MARGIN:
        return "NO_DEFICIT"
    return "INCONCLUSIVE"


def slope_call(ci: dict) -> str:
    """ci = the interval of AUC_F(100 %) - AUC_F(25 %) (one row, one seed).  COVERAGE iff lo > +0.01; FEATURE iff hi < +0.01; else INCONCLUSIVE."""
    if ci["lo"] > SLOPE_MARGIN:
        return "COVERAGE"
    if ci["hi"] < SLOPE_MARGIN:
        return "FEATURE"
    return "INCONCLUSIVE"


def row_class(deficits: list, slopes: list) -> str:
    """A row is classified only if BOTH seeds agree.  DEFICIT on all seeds -> COVERAGE / FEATURE / INCONCLUSIVE from the slope calls; NO_DEFICIT on all seeds -> NO_DEFICIT; else INCONCLUSIVE."""
    if all(d == "DEFICIT" for d in deficits):
        if all(s == "COVERAGE" for s in slopes):
            return "COVERAGE"
        if all(s == "FEATURE" for s in slopes):
            return "FEATURE"
        return "INCONCLUSIVE"
    if all(d == "NO_DEFICIT" for d in deficits):
        return "NO_DEFICIT"
    return "INCONCLUSIVE"


NEXT_ARM = {
    "COVERAGE": "FASTNEG-BANK (SH1): a fast / lead-ward `fast` family in build_scorer_targets.py (EST 26.4 CPU-h), then HN-screen arm C under an amendment of PREREG_P6HN_SCREEN (3 seeds + an A replicate; endpoints E1-E3 of the Lab report s4.5)",
    "FEATURE": "PRIVGEOM-PROBE (D2, CPU EST 13.5 h): privileged PDM geometry scalars added to the token as a diagnostic only; which scalars carry the jump names the scorer-branch lever (SH3 / SH5); no jump -> the trunk (P6 read; SH7 / SH8 are A40 and PI-gated)",
    "NO_DEFICIT": "P-B is not a fast-band resolution deficit; the leak is in how the pick rule COMPOSES the rows (planner.py:558-561). Next: a 0-GPU pick-rule audit on S0: the unsafe-pick rate of an EP-led composite as a function of the NC / DAC veto threshold, rows unchanged",
    "INCONCLUSIVE": "re-run ONCE at f = 25 % (the next nested size), cost named before the run",
    "VOID": "the powered null fired on >= 2 of 5 replicates for a row: the census is VOID; the slope test is not trustworthy on this rig",
}


def census_outcome(nc: str, dac: str, void: bool) -> dict:
    """Precedence (A1.6): VOID > COVERAGE on NC or DAC > FEATURE on both > NO_DEFICIT on both > INCONCLUSIVE (everything else)."""
    if void:
        out = "VOID"
    elif nc == "COVERAGE" or dac == "COVERAGE":
        out = "COVERAGE"
    elif nc == "FEATURE" and dac == "FEATURE":
        out = "FEATURE"
    elif nc == "NO_DEFICIT" and dac == "NO_DEFICIT":
        out = "NO_DEFICIT"
    else:
        out = "INCONCLUSIVE"
    return {"outcome": out, "next_arm": NEXT_ARM[out], "nc": nc, "dac": dac, "void": bool(void)}


def null_fires(slope_cis_by_seed: list) -> bool:
    """A null replicate FIRES for a row iff the COVERAGE slope test passes (lo > +0.01) on ALL seeds (A1.5)."""
    return all(ci["lo"] > SLOPE_MARGIN for ci in slope_cis_by_seed)


def null_void(fires_by_row: dict) -> bool:
    """VOID iff, for any row, the null fired on >= 2 of its replicates (parent s4(c0))."""
    return any(sum(1 for f in fires if f) >= NULL_VOID_FIRES for fires in fires_by_row.values())


def branch_table_failures(mod) -> list:
    """The synthetic branch table every decision function must satisfy.  `mod` is this module or a MUTATED copy of it; returns the list of failing cases (empty = all branches right)."""
    bad = []

    def chk(name, got, want):
        if got != want:
            bad.append(f"{name}: got {got!r}, want {want!r}")
    ci = lambda lo, hi: {"lo": lo, "hi": hi}                                                          # noqa: E731
    chk("deficit hi<-m", mod.deficit_call(ci(-0.09, -0.03)), "DEFICIT")
    chk("deficit straddle -m", mod.deficit_call(ci(-0.05, 0.00)), "INCONCLUSIVE")
    chk("deficit lo>-m (zero)", mod.deficit_call(ci(-0.01, 0.02)), "NO_DEFICIT")
    chk("deficit lo>-m (negative lo)", mod.deficit_call(ci(-0.015, 0.03)), "NO_DEFICIT")
    chk("deficit hi=-m is not <", mod.deficit_call(ci(-0.09, -0.02)), "INCONCLUSIVE")
    chk("deficit lo=-m is not >", mod.deficit_call(ci(-0.02, 0.05)), "INCONCLUSIVE")
    chk("deficit hi negative but above -m", mod.deficit_call(ci(-0.05, -0.01)), "INCONCLUSIVE")
    chk("slope lo>m", mod.slope_call(ci(0.02, 0.05)), "COVERAGE")
    chk("slope hi<m", mod.slope_call(ci(-0.01, 0.005)), "FEATURE")
    chk("slope straddle", mod.slope_call(ci(0.0, 0.03)), "INCONCLUSIVE")
    chk("slope lo=m is not >", mod.slope_call(ci(0.01, 0.05)), "INCONCLUSIVE")
    chk("slope hi=m is not <", mod.slope_call(ci(-0.05, 0.01)), "INCONCLUSIVE")
    chk("slope hi<m but lo>0 (tiny positive slope)", mod.slope_call(ci(0.001, 0.008)), "FEATURE")
    chk("row COVERAGE", mod.row_class(["DEFICIT", "DEFICIT"], ["COVERAGE", "COVERAGE"]), "COVERAGE")
    chk("row FEATURE", mod.row_class(["DEFICIT", "DEFICIT"], ["FEATURE", "FEATURE"]), "FEATURE")
    chk("row seeds disagree on slope", mod.row_class(["DEFICIT", "DEFICIT"], ["COVERAGE", "FEATURE"]), "INCONCLUSIVE")
    chk("row slope inconclusive on one seed", mod.row_class(["DEFICIT", "DEFICIT"], ["COVERAGE", "INCONCLUSIVE"]), "INCONCLUSIVE")
    chk("row NO_DEFICIT", mod.row_class(["NO_DEFICIT", "NO_DEFICIT"], ["COVERAGE", "COVERAGE"]), "NO_DEFICIT")
    chk("row deficit on one seed only", mod.row_class(["DEFICIT", "NO_DEFICIT"], ["COVERAGE", "COVERAGE"]), "INCONCLUSIVE")
    chk("row deficit inconclusive on one seed", mod.row_class(["DEFICIT", "INCONCLUSIVE"], ["COVERAGE", "COVERAGE"]), "INCONCLUSIVE")
    chk("outcome COVERAGE on NC", mod.census_outcome("COVERAGE", "FEATURE", False)["outcome"], "COVERAGE")
    chk("outcome COVERAGE on DAC", mod.census_outcome("NO_DEFICIT", "COVERAGE", False)["outcome"], "COVERAGE")
    chk("outcome FEATURE both", mod.census_outcome("FEATURE", "FEATURE", False)["outcome"], "FEATURE")
    chk("outcome FEATURE + NO_DEFICIT", mod.census_outcome("FEATURE", "NO_DEFICIT", False)["outcome"], "INCONCLUSIVE")
    chk("outcome NO_DEFICIT both", mod.census_outcome("NO_DEFICIT", "NO_DEFICIT", False)["outcome"], "NO_DEFICIT")
    chk("outcome NO_DEFICIT + INCONCLUSIVE", mod.census_outcome("NO_DEFICIT", "INCONCLUSIVE", False)["outcome"], "INCONCLUSIVE")
    chk("outcome VOID overrides COVERAGE", mod.census_outcome("COVERAGE", "COVERAGE", True)["outcome"], "VOID")
    chk("outcome next arm FEATURE", mod.census_outcome("FEATURE", "FEATURE", False)["next_arm"].split()[0], "PRIVGEOM-PROBE")
    chk("null fires both seeds", mod.null_fires([ci(0.02, 0.05), ci(0.03, 0.06)]), True)
    chk("null fires one seed only", mod.null_fires([ci(0.02, 0.05), ci(0.0, 0.06)]), False)
    chk("null void 2 of 5", mod.null_void({"nc": [True, True, False, False, False], "dac": [False] * 5}), True)
    chk("null void 1 of 5", mod.null_void({"nc": [True, False, False, False, False], "dac": [False] * 5}), False)
    chk("null void on the other row", mod.null_void({"nc": [False] * 5, "dac": [False, True, False, True, False]}), True)
    return bad


# ---- the L-arm refit (HEADROW-PROBE s3.2 recipe; A1.4) -----------------------------------------------------------------------------------------------------------------------
def make_incl(code: np.ndarray, u: np.ndarray, frac: float, stratum: int = 4) -> np.ndarray:
    """float32 [N,64] loss-inclusion weights: entries of `stratum` (default Q4 = F) are included iff u < frac; every other entry is always included."""
    incl = np.ones(code.shape, np.float32)
    if frac < 1.0:
        incl[(code == stratum) & ~(u < frac)] = 0.0
    return incl


def subsample_priority(shape, seed: int) -> np.ndarray:
    """The per-entry priority u ~ U(0,1) drawn once per head-fit seed (A1.4); frac 0.25 is nested in 0.5 is nested in 1.0."""
    return np.random.default_rng(SUB_SEED_BASE + SUB_SEED_MULT * int(seed)).random(shape, dtype=np.float32)


def wait_for_ram(min_gb: float = MIN_AVAIL_GB, poll_s: float = 20.0, log=print, max_wait_s: float | None = None) -> float:
    """Block until psutil's available RAM >= min_gb (re-checked every poll_s).  Returns the available GB at release."""
    import psutil
    t0 = time.time()
    while True:
        av = psutil.virtual_memory().available / 2 ** 30
        if av >= min_gb:
            return av
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            raise CensusError(f"available RAM {av:.2f} GB < {min_gb} GB after {max_wait_s} s")
        log(f"[ram] available {av:.2f} GB < {min_gb} GB; waiting {poll_s:.0f} s")
        time.sleep(poll_s)


class LRows:
    """The L arm for the rows in `cols` (NC = column 0, DAC = column 1), fitted in lockstep.  Every row has its own parameters (w [256], b [1]) and its own Adam, so each is a separate fit up to
    the shared batches; z = z_stored + feat . (w - p2_w) + (b - p2_b) (cold: z = feat . w + b); features are cast to fp32 and DETACHED by construction (they are plain tensors, never leaves)."""

    def __init__(self, p2_w: np.ndarray, p2_b: np.ndarray, cols=(0, 1), cold: bool = False, lr: float = LR, wd: float = WD, betas=BETAS, eps: float = EPS):
        import torch
        self.torch = torch
        self.cols, self.cold = tuple(cols), bool(cold)
        self.p2_w = [torch.as_tensor(np.asarray(p2_w[c], np.float32).copy()) for c in self.cols]
        self.p2_b = [torch.as_tensor(np.asarray(p2_b[c:c + 1], np.float32).copy()) for c in self.cols]
        zero = lambda t: torch.zeros_like(t)                                                          # noqa: E731
        self.w = [torch.nn.Parameter(zero(p) if cold else p.clone()) for p in self.p2_w]
        self.b = [torch.nn.Parameter(zero(p) if cold else p.clone()) for p in self.p2_b]
        self.opts = [torch.optim.Adam([w, b], lr=lr, betas=betas, eps=eps, weight_decay=wd) for w, b in zip(self.w, self.b)]

    def logit(self, r: int, feat32, z_stored_r):
        if self.cold:
            return feat32 @ self.w[r] + self.b[r]
        return z_stored_r + (feat32 @ (self.w[r] - self.p2_w[r]) + (self.b[r] - self.p2_b[r]))

    def _row_loss(self, r: int, feat32, z_stored_r, y_r, incl):
        F = self.torch.nn.functional
        z = self.logit(r, feat32, z_stored_r)
        if incl is None:
            return F.binary_cross_entropy_with_logits(z, y_r)
        l = F.binary_cross_entropy_with_logits(z, y_r, reduction="none")
        return (l * incl).sum() / incl.sum().clamp(min=1.0)

    def grads(self, feat16, z_stored, y, incl=None) -> list:
        """The gradients of every row's loss w.r.t. its own (w, b), WITHOUT stepping (the reach proof uses the very same `_row_loss` the training uses)."""
        feat32 = feat16.detach().float()
        out = []
        for r in range(len(self.cols)):
            loss = self._row_loss(r, feat32, z_stored[..., r], y[..., r], incl)
            self.opts[r].zero_grad(set_to_none=True)
            loss.backward()
            out.append((self.w[r].grad.detach().clone(), self.b[r].grad.detach().clone(), float(loss.detach())))
        return out

    def step_batch(self, feat16, z_stored, y, incl=None) -> list:
        """One minibatch.  feat16 [B,64,256] fp16 tensor, z_stored [B,64,len(cols)] fp32, y [B,64,len(cols)] fp32, incl [B,64] float32 or None.  Returns the per-row loss values."""
        feat32 = feat16.detach().float()                                           # MUTATION-TARGET-DETACH: the features are frozen
        losses = []
        for r in range(len(self.cols)):
            loss = self._row_loss(r, feat32, z_stored[..., r], y[..., r], incl)
            self.opts[r].zero_grad(set_to_none=True)
            loss.backward()
            self.opts[r].step()
            losses.append(float(loss.detach()))
        return losses

    def params(self) -> tuple:
        return (np.stack([w.detach().numpy().copy() for w in self.w]), np.concatenate([b.detach().numpy().copy() for b in self.b]))


def fit_rows(feat_mm, z_st: np.ndarray, y: np.ndarray, train_idx: np.ndarray, p2_w: np.ndarray, p2_b: np.ndarray, seed: int, incl: np.ndarray | None = None, gen_salt: int = 0,
             cold: bool = False, epochs: int = EPOCHS, batch_sets: int = BATCH_SETS, cols=(0, 1), ram_guard=None, trace: list | None = None) -> tuple:
    """Fit the L arm on the sets `train_idx` (ascending).  feat_mm: [N,64,256] fp16 array or memmap; z_st/y: [N,64,len(cols)] fp32 (stored logits / labels of the fitted rows); incl: [N,64] float32 or None.
    Batch order: `torch.randperm` from `Generator().manual_seed(seed * 1_000_003 + len(train_idx) + gen_salt)`, reshuffled every epoch (HEADROW-PROBE s3.2; gen_salt = 0 for every registered fit).
    -> (w [len(cols),256] float32, b [len(cols)] float32)."""
    import torch
    head = LRows(p2_w, p2_b, cols=cols, cold=cold)
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(len(train_idx)) + int(gen_salt))
    tr = torch.as_tensor(np.asarray(train_idx), dtype=torch.long)
    n = len(tr)
    if ram_guard is not None:
        ram_guard()
    for ep in range(epochs):
        order = tr[torch.randperm(n, generator=g)]
        for s in range(0, n, batch_sets):
            bi = order[s:s + batch_sets].numpy()
            f = torch.from_numpy(np.ascontiguousarray(feat_mm[bi]))
            zs, yy = torch.from_numpy(z_st[bi]), torch.from_numpy(y[bi])
            mm = None if incl is None else torch.from_numpy(incl[bi])
            ls = head.step_batch(f, zs, yy, mm)
            if trace is not None:
                trace.append(ls)
    return head.params()


def score_rows(feat_mm, z_st: np.ndarray, idx: np.ndarray, w: np.ndarray, b: np.ndarray, p2_w: np.ndarray, p2_b: np.ndarray, cols=(0, 1), cold: bool = False, chunk: int = 1024) -> np.ndarray:
    """The refit's logits on the sets `idx` -> [len(idx), 64, len(cols)] float32."""
    import torch
    out = np.empty((len(idx), N_PROP, len(cols)), np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            ii = np.asarray(idx[s:s + chunk])
            f = torch.from_numpy(np.ascontiguousarray(feat_mm[ii])).float()
            zs = torch.from_numpy(z_st[ii])
            for r, c in enumerate(cols):
                wr, br = torch.as_tensor(w[r]), torch.as_tensor(b[r:r + 1])
                if cold:
                    z = f @ wr + br
                else:
                    z = zs[..., r] + (f @ (wr - torch.as_tensor(np.asarray(p2_w[c], np.float32))) + (br - torch.as_tensor(np.asarray(p2_b[c:c + 1], np.float32))))
                out[s:s + len(ii), :, r] = z.numpy()
    return out
