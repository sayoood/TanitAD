"""SH5-DAC-PROX library (numpy only, CPU): the registered constants, the key-time points, the clearance targets, the 2-parameter stacker, the within-token AUC with the paired
log-cluster bootstrap, the pick-level read, the reliability-at-picks diagnostic, the REGISTERED decision functions and the branch table every mutation must break.

REGISTERED SOURCE: PREREG_SH5_DAC_PROX.md (its sha256 is in PREREG_SH5_DAC_PROX.sha256 and SEALED_AT.txt, written before any label array of this arm is opened), as amended by
PREREG_SH5_DAC_PROX_AMENDMENT1.md (code only), PREREG_SH5_DAC_PROX_AMENDMENT2.md (the CTX-REACHABLE rule: lower CI > 0 AND point >= +0.02; `seed_call`, `REACH_LO_MIN`, `REACH_POINT_MIN`) and
PREREG_SH5_DAC_PROX_AMENDMENT3.md (the C-NULL-LABEL fire rule IS the Amendment-2 rule: `null_label_fires` reads `seed_call`).
The estimator helpers are the census's (`fastneg_lib`, pinned) and the pick-rule audit's (`pickrule_lib`, pinned), loaded by sha256 from the first directory that holds all three
pinned files (env SH5_REF_DIR, then ../ref, then the PRIVGEOM package's ref/).  Every constant cites the prereg section it comes from.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "sh5-dac-prox-v1"

# ---- pinned landed modules (prereg s12) -------------------------------------------------------------------------------------------------------------------------------------------
REF_PINS = {
    "fastneg_lib.py": "0c0b399b37a47ee3e17f9d5484bc7f9521ff7dbe6bea37e8c7a38042668667ca",
    "pickrule_lib.py": "555c0a3468acd94adeeeb10f47527644a2b9d49722f554fed2da1fb29668eaeb",
    "epr1.py": "a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8",
}
REF_CANDIDATES = [os.environ.get("SH5_REF_DIR") or "", os.path.normpath(os.path.join(HERE, "..", "ref")), "C:/Users/Admin/par18/privgeom/ref"]
STRATA_SHA256 = "67f43e55dd7c693e31cd991efc3d51fedc278331ba4f8f882e6adbdae8203dc6"      # the census strata (`code` array); label-free by the census's own proof

# ---- registered design constants (prereg s3 - s7) ----------------------------------------------------------------------------------------------------------------------------------
KEY_TIMES_S = tuple(0.5 * k for k in range(9))          # t = 0, 0.5, ..., 4.0 s = the PRIVGEOM key times (pg_geom.KEY_STEPS on the 0.1 s grid)
SRC_DT = 0.2                                            # traj3 is 20 poses at 0.2 s, 0.2 ... 4.0 s, ego frame (x, y, yaw)
N_PTS, N_REG, CLEAR_CAP = 9, 9, 10.0                  # N_REG = the 9 key-time |clearances|; |dac_clear_min| is EXCLUDED (sign-clean targets, MM ruling 2026-10-19, prereg s3)
SEEDS = (0, 1)
FOLDS = 5
EPOCHS_TOTAL = 12                                       # token passes; 3 cycles x 4 inner passes when the ctx is sharded, 1 x 12 when it is resident (s5)
CYCLES_SHARDED, INNER_SHARDED = 3, 4
TOKENS_PER_STEP, P_SUB = 8, 32
LR, WD, BETAS, EPS, WARMUP = 3e-4, 1e-4, (0.9, 0.999), 1e-8, 50
W_REG, HUBER_DELTA = 0.1, 1.0
D_MODEL, HEADS, LAYERS, FFN, FOURIER_K = 256, 4, 2, 512, 6
INFO_MARGIN, NOTREACH_MARGIN = 0.02, 0.01               # s7: NOT-REACHABLE iff hi < +0.01 on both seeds (unchanged by Amendments 2 and 3); INFO_MARGIN = the SEALED CTX-REACHABLE bar (lo > +0.02): HISTORICAL since Amendment 3 (no decision function reads it; the sealed-rule simulations, the sealed-MDE arithmetic and the RED mutation that restores the sealed rule do)
REACH_LO_MIN, REACH_POINT_MIN = 0.0, 0.02              # AMENDMENT 2 (MM ruling 2026-10-19, before any SH5 number): CTX-REACHABLE per seed iff Delta_F lower CI > 0 (significance) AND point >= +0.02 (materiality), on BOTH seeds
BOOT_B, BOOT_SEED = 10000, 20260927
NULL_REPS, NULL_SALT, NULL_SEED_BASE = 5, 7919, 20261019           # C-NULL (clearance targets permuted within token)
NULL_LABEL_REPS = 3                                                # C-NULL-LABEL (machinery null; an ADDITION, s8)
PLANT_SIGMA_M = 1.15                                               # C-POWER weak plant: signed clearance + N(0, sigma m); sigma is CALIBRATED in calibrate_plant.py so an IDEAL reader gains +0.03 (s8)
POWER_EFFECT = 0.03                                                # the effect C-POWER is sized to (MM ruling 2026-10-19: PRIVGEOM's sign-clean read is +0.028 / +0.035, not +0.16)
POWER_POINT_LO, POWER_POINT_HI = 0.015, 0.045                      # C-POWER: the weak plant's Delta_F point must sit in this band AND its lower CI must exclude 0, on both seeds
REACH_MIN_AUC, REACH_MIN_R2 = 0.95, 0.50                           # C-REACH (the R^2 bar is set from the CPU synthetic: 0.595 at 40 epochs of a 3 k-proposal task)
FID_TOL, FID_MIN_FRAC = 1e-3, 0.99                                 # C-FID: |logit(ctx build) - census z6| <= 1e-3 on >= 99 % of sets
MIN_AVAIL_GB_FIT, MIN_AVAIL_GB_BUILD = 4.0, 8.0
DAC_COL = 1                                                        # labels6 / z6 column of the DAC row

STAND_FOLD_NOTE = "folds = the census fold of each log (rank mod 5); a fold's tokens are never in that fold's training set"


class Sh5Error(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def locate_ref() -> str:
    for d in REF_CANDIDATES:
        if d and all(os.path.exists(os.path.join(d, n)) for n in REF_PINS):
            if all(sha256_file(os.path.join(d, n)) == s for n, s in REF_PINS.items()):
                return os.path.abspath(d)
    raise Sh5Error(f"no directory with the three pinned landed modules {sorted(REF_PINS)} at their pinned sha256: tried {[c for c in REF_CANDIDATES if c]}")


_REF: dict = {}


def refs():
    """-> (fastneg_lib, pickrule_lib) imported from the pinned directory (cached)."""
    if not _REF:
        d = locate_ref()
        if d not in sys.path:
            sys.path.insert(0, d)
        _REF["FL"] = importlib.import_module("fastneg_lib")
        _REF["PK"] = importlib.import_module("pickrule_lib")
        _REF["dir"] = d
    return _REF["FL"], _REF["PK"]


# ---- the branch's inputs and targets (prereg s3) ----------------------------------------------------------------------------------------------------------------------------------------
def repair_last_heading(traj3: np.ndarray) -> np.ndarray:
    """SPEC_NAVTEST Amendment 7 (planner.repair_last_heading): the last pose takes the previous pose's heading.  A COPY."""
    out = np.array(traj3, dtype=np.float64, copy=True)
    out[..., -1, 2] = out[..., -2, 2]
    return out


def keypoints(traj3: np.ndarray, repair: bool = True) -> np.ndarray:
    """traj3 [..., 20, 3] (x, y, yaw at t = 0.2 ... 4.0 s, ego frame) -> [..., 9, 3] at the key times 0, 0.5, ..., 4.0 s.
    t = 0 is the ego origin (0, 0, 0).  Between source poses x and y interpolate linearly and the heading through its unit vector (no wrap artefacts).  float32."""
    P = repair_last_heading(traj3) if repair else np.asarray(traj3, np.float64)
    lead = list(P.shape[:-2])
    full = np.concatenate([np.zeros(lead + [1, 3]), P], axis=-2)               # 21 poses, t = 0.0, 0.2, ..., 4.0
    kt = np.array(KEY_TIMES_S)
    i0 = np.minimum(np.floor(kt / SRC_DT + 1e-9).astype(int), 19)
    fr = ((kt - SRC_DT * i0) / SRC_DT)[..., None]                              # [9, 1]
    a, b = full[..., i0, :], full[..., i0 + 1, :]
    xy = (1 - fr) * a[..., :2] + fr * b[..., :2]
    c = (1 - fr[..., 0]) * np.cos(a[..., 2]) + fr[..., 0] * np.cos(b[..., 2])
    s = (1 - fr[..., 0]) * np.sin(a[..., 2]) + fr[..., 0] * np.sin(b[..., 2])
    return np.concatenate([xy, np.arctan2(s, c)[..., None]], axis=-1).astype(np.float32)


def clearance_targets(g: np.ndarray) -> np.ndarray:
    """g [..., >=11] PRIVGEOM scalars (col 0 dac_clear_min, col 1 dac_frac_out, cols 2..10 the 9 key-time signed clearances) -> [..., 9] = the 9 key-time |clearance|, metres, capped at 10.
    SIGN-CLEAN: `dac_clear_min` (col 0) and `dac_frac_out` (col 1) are NEVER targets.  Why: c_min is the minimum over ALL steps, so c_t >= c_min at every key step; if c_min >= 0 (DAC-safe) then
    |c_t| >= |c_min| everywhere, hence any key step with |c_t| < |c_min| PROVES DAC = 0 (PRIVGEOM-SAFE RESULT 5.2: precision 1.000, recall 88.7 %).  The 9 |c_t| alone carry no such pair; that is
    tested by `sh5_admit.py` with planted positive and negative controls (prereg s3.3), not by inspection."""
    g = np.asarray(g)
    return np.minimum(np.abs(g[..., 2:11]), CLEAR_CAP).astype(np.float32)


def signed_key_clearance(g: np.ndarray) -> np.ndarray:
    """[..., 9] the SIGNED key-time clearances (used ONLY to build the planted controls; never an input of the real arms)."""
    return np.asarray(g)[..., 2:11].astype(np.float32)


# ---- the 2-parameter stacker (prereg s5) ------------------------------------------------------------------------------------------------------------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_stacker(zl: np.ndarray, zs: np.ndarray, y: np.ndarray, iters: int = 50, ridge: float = 1e-6) -> tuple:
    """z = a * zl + b * zs (no intercept: the base rate is inside P2's logit), BCE on the (soft) label, Newton from (a, b) = (1, 0), ridge pulls toward (1, 0).
    DEGENERATE RULE (C-ID): if std(zs) < 1e-9 the answer is EXACTLY (1.0, 0.0), so a zero branch gives L+S = L bit for bit.  -> (a, b, ok)."""
    zl = np.asarray(zl, np.float64).ravel()
    zs = np.asarray(zs, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    if zs.size == 0 or float(zs.std()) < 1e-9:
        return 1.0, 0.0, True
    th = np.array([1.0, 0.0])
    X = np.stack([zl, zs], axis=1)
    n = len(y)
    for _ in range(iters):
        p = _sig(X @ th)
        g = X.T @ (p - y) / n + ridge * (th - np.array([1.0, 0.0]))
        w = p * (1 - p)
        H = (X * w[:, None]).T @ X / n + ridge * np.eye(2)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return 1.0, 0.0, False
        th = th - step
        if not np.all(np.isfinite(th)):
            return 1.0, 0.0, False
        if float(np.abs(step).max()) < 1e-10:
            break
    return float(th[0]), float(th[1]), True


def apply_stacker(zl, zs, a: float, b: float) -> np.ndarray:
    if a == 1.0 and b == 0.0:
        return np.asarray(zl, np.float32).copy()                     # bit-identical to L (C-ID)
    return (a * np.asarray(zl, np.float64) + b * np.asarray(zs, np.float64)).astype(np.float32)


def oof_stack(zl: np.ndarray, zs_oof: np.ndarray, y: np.ndarray, fold: np.ndarray) -> tuple:
    """Leave-one-fold-out stacker on the OOF matrix: for each fold k the 2 weights are fitted on the OOF predictions of the OTHER folds and applied to fold k.
    (The models behind the other folds' predictions DID train on fold k: a second-order path through 2 numbers, disclosed in s5 and bounded by C-NULL-LABEL.)
    zl, zs_oof, y [N,64]; fold [N].  -> (z_comb [N,64] float32, params [(a,b,ok)] per fold)."""
    out = np.empty(zl.shape, np.float32)
    params = []
    for k in range(FOLDS):
        te = fold == k
        tr = ~te
        if not te.any():
            params.append((1.0, 0.0, True))
            continue
        a, b, ok = fit_stacker(zl[tr], zs_oof[tr], y[tr])
        params.append((a, b, ok))
        out[te] = apply_stacker(zl[te], zs_oof[te], a, b)
    return out, params


def plant_noise(token: str, shape: tuple, sigma: float) -> np.ndarray:
    """the C-POWER plant's noise for ONE token (all its sets): N(0, sigma), seeded by the token's own name so that a token's noise is the same in every analysis set and in the design calibration."""
    seed = (int(hashlib.sha256(str(token).encode()).hexdigest()[:8], 16) + NULL_SEED_BASE) % (2 ** 32)
    return np.random.default_rng(seed).normal(0.0, sigma, shape)


# ---- permutations for the controls (prereg s8) ---------------------------------------------------------------------------------------------------------------------------------------
def permute_within_token(tgt: np.ndarray, token_idx: np.ndarray, seed: int) -> np.ndarray:
    """tgt [N,64,K]; every proposal of every set of a token receives the K-vector of another proposal of the SAME token (one permutation over the token's pooled proposals).
    The marginal per token is unchanged and the proposal <-> clearance match is destroyed."""
    out = np.array(tgt, copy=True)
    rng = np.random.default_rng(seed)
    for t in np.unique(token_idx):
        rows = np.flatnonzero(token_idx == t)
        flat = out[rows].reshape(-1, tgt.shape[-1])
        out[rows] = flat[rng.permutation(len(flat))].reshape(len(rows), tgt.shape[1], tgt.shape[-1])
    return out


def permute_within_set(a: np.ndarray, seed: int) -> np.ndarray:
    """a [N,64,...]: an independent permutation of the 64 proposals inside every set."""
    out = np.array(a, copy=True)
    rng = np.random.default_rng(seed)
    for i in range(len(out)):
        out[i] = out[i][rng.permutation(out.shape[1])]
    return out


# ---- the analyzer (prereg s6) --------------------------------------------------------------------------------------------------------------------------------------------------------
class Analyzer:
    """Within-token (inside ONE set) DAC AUC in F (= Q4) and Q1-3 with the paired log-cluster bootstrap over the analysis set's logs.  unsafe = label 0, safe = label 1 (0.5 excluded)."""

    def __init__(self, lab_dac: np.ndarray, code: np.ndarray, log_idx: np.ndarray, n_logs: int, B: int = BOOT_B, seed: int = BOOT_SEED):
        self.FL, self.PK = refs()
        self.code, self.log_idx, self.n_logs = code, log_idx, n_logs
        self.draws = self.FL.boot_draws(n_logs, B, seed)
        self.unsafe, self.safe = lab_dac == 0.0, lab_dac == 1.0

    def stats(self, z: np.ndarray) -> dict:
        st = self.FL.pair_stats(z, self.unsafe, self.safe, self.code, names=("Q4", "Q123"))
        return {k: (self.FL.per_log(v[0], self.log_idx, self.n_logs), self.FL.per_log(v[1], self.log_idx, self.n_logs)) for k, v in st.items()}

    def auc(self, st: dict, stratum: str = "Q4") -> dict:
        n, d = st[stratum]
        return self.FL.ratio_ci(n, d, self.draws)

    def diff(self, st_a: dict, st_b: dict, stratum: str = "Q4") -> dict:
        return self.FL.diff_ci(st_a[stratum][0], st_a[stratum][1], st_b[stratum][0], st_b[stratum][1], self.draws)

    def diff_of_diffs(self, a1: dict, a0: dict, b1: dict, b0: dict, stratum: str = "Q4") -> dict:
        """(AUC(a1) - AUC(a0)) - (AUC(b1) - AUC(b0)) on the SAME resampled logs (the attribution (L+S) - (S-noprox) is a difference of two such deltas against the same L)."""
        FL = self.FL
        d = self.draws
        pts, vals = [], []
        for (s1, s0) in ((a1, a0), (b1, b0)):
            n1, d1 = s1[stratum]
            n0, d0 = s0[stratum]
            pts.append(n1.sum() / d1.sum() - n0.sum() / d0.sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                vals.append(n1[d].sum(1) / d1[d].sum(1) - n0[d].sum(1) / d0[d].sum(1))
        v = vals[0] - vals[1]
        return FL._ci(float(pts[0] - pts[1]), v)


# ---- the registered decision functions (prereg s7); pure; `branch_table_failures` is the table every mutation must break ---------------------------------------------------------
def seed_call(ci: dict) -> str:
    """One seed's call on Delta_F = AUC_F(L+S) - AUC_F(L): REACH iff lo > 0 AND point >= +0.02 (Amendment 2; the sealed rule was lo > +0.02); NOT iff hi < +0.01 (unchanged); else INC.
    A NaN lower bound or point is never REACH, a NaN upper bound never NOT.  Used for the raw call AND, unchanged, for the ctx-shuffled arm's GAIN-NOT-FROM-CTX test."""
    if ci["lo"] > REACH_LO_MIN and ci["point"] >= REACH_POINT_MIN:
        return "REACH"
    if ci["hi"] < NOTREACH_MARGIN:
        return "NOT"
    return "INC"


def reach_call(calls: list) -> str:
    """CTX-REACHABLE iff BOTH seeds REACH; NOT-REACHABLE iff BOTH seeds NOT; otherwise INCONCLUSIVE (two seeds that disagree are INCONCLUSIVE)."""
    if len(calls) == len(SEEDS) and all(c == "REACH" for c in calls):
        return "CTX-REACHABLE"
    if len(calls) == len(SEEDS) and all(c == "NOT" for c in calls):
        return "NOT-REACHABLE"
    return "INCONCLUSIVE"


def pick_gain_seed(dpdms_ci: dict, ddac_ci: dict) -> bool:
    """P2-rule pick level, one seed: dPDMS x100 lower CI > 0 AND the DAC-unsafe-rate change upper CI <= 0."""
    return bool(dpdms_ci["lo"] > 0.0 and ddac_ci["hi"] <= 0.0)


def pick_gain_call(per_seed: list) -> bool:
    return len(per_seed) == len(SEEDS) and all(per_seed)


def null_label_fires(cis_by_seed: list) -> bool:
    """A C-NULL-LABEL replicate FIRES iff it satisfies the CTX-REACHABLE rule on the seed(s) it was run for: `seed_call(ci) == "REACH"`, i.e. Delta_F lower CI > 0 AND point >= +0.02 (AMENDMENT 3, MM ruling
    2026-10-19 s5; the sealed rule was lower CI > +0.02, and Amendment 2 had kept it for VOID while moving the call).  ONE definition: this function calls `seed_call` and holds no second copy of the thresholds,
    so a machinery null is read by the very test the claim uses.  The VOID COUNT rule (`null_label_void`: any replicate fires) is unchanged."""
    return bool(len(cis_by_seed) > 0 and all(seed_call(c) == "REACH" for c in cis_by_seed))


def null_label_void(fires: list) -> bool:
    """VOID iff ANY machinery-null replicate fires (a fire under a noise-trained branch is a leak, not a finding; the simulated false-fire rate per replicate is in s8)."""
    return bool(any(fires))


def power_ok(cis_by_seed: list) -> bool:
    """C-POWER (sized at +0.03, MM ruling 2026-10-19): the weak plant, whose IDEAL reader gains +0.03, must read Delta_F with its lower CI > 0 and its point inside [+0.015, +0.045] on BOTH seeds.
    (Amendment 2 replaced the +0.02 lower-CI bar of the CTX-REACHABLE rule by `lo > 0 and point >= +0.02`, which this plant's +0.03 effect can clear; C-POWER itself is unchanged by Amendments 2 and 3.)"""
    return bool(len(cis_by_seed) == len(SEEDS) and all(c["lo"] > 0.0 and POWER_POINT_LO <= c["point"] <= POWER_POINT_HI for c in cis_by_seed))


def next_arm(outcome: str, pick_gain: bool) -> str:
    if outcome == "CTX-REACHABLE":
        return "INTEGRATE-SH5" if pick_gain else "STOP-AT-DIAGNOSTIC"
    if outcome == "NOT-REACHABLE":
        return "P6-PROX-AUX"
    return "NAMED-BY-REPORT"


def final_outcome(raw: str, gates_ok: bool, creach_ok: bool, void_label: bool, power: bool, shuf_gain: bool = False) -> str:
    """The conversions can only REMOVE a call, never create one (s7).  shuf_gain = the ctx-SHUFFLED arm (the branch sees ANOTHER token's ctx) passes the CTX-REACHABLE rule on both seeds:
    the gain is then the branch re-learning the proposal geometry, not a read of this token's ctx."""
    if not gates_ok:
        return "NO-CALL(GATE)"
    if not creach_ok:
        return "NO-CALL(C-REACH)"
    if void_label:
        return "VOID(C-NULL-LABEL)"
    if raw == "NOT-REACHABLE" and not power:
        return "UNDERPOWERED"
    if raw == "CTX-REACHABLE" and shuf_gain:
        return "GAIN-NOT-FROM-CTX"
    return raw


def branch_table_failures(mod=None) -> list:
    """Returns the list of failed expectations of the decision table (empty = pass).  `mod` is the module under test (a mutated copy in the RED tests)."""
    m = mod if mod is not None else sys.modules[__name__]
    ci = lambda lo, hi: {"point": (lo + hi) / 2, "lo": lo, "hi": hi}      # noqa: E731
    cp = lambda point, lo, hi: {"point": point, "lo": lo, "hi": hi}       # noqa: E731  (explicit point: the Amendment-2 CTX-REACHABLE rule reads the point as well as the interval)
    bad = []

    def chk(name, got, want):
        if got != want:
            bad.append(f"{name}: got {got!r} want {want!r}")
    chk("seed REACH", m.seed_call(ci(0.021, 0.08)), "REACH")
    # ---- Amendment 2: CTX-REACHABLE per seed iff lower CI > 0 AND point >= +0.02.  Each conjunct has its boundary and a PLANTED case that dropping the OTHER conjunct would wrongly pass.
    chk("A2 both conjuncts hold", m.seed_call(cp(0.030, 0.010, 0.050)), "REACH")
    chk("A2 the sealed edge lo == +0.02 is now REACH (the sealed rule said INC)", m.seed_call(ci(0.02, 0.08)), "REACH")
    chk("A2 lower CI == 0 is not REACH (strict)", m.seed_call(cp(0.030, 0.0, 0.060)), "INC")
    chk("A2 lower CI barely above 0 with point >= +0.02 is REACH", m.seed_call(cp(0.025, 1e-9, 0.050)), "REACH")
    chk("A2 point == +0.02 is REACH (>=)", m.seed_call(cp(0.020, 0.001, 0.040)), "REACH")
    chk("A2 point just below +0.02 is not REACH", m.seed_call(cp(0.0199, 0.001, 0.040)), "INC")
    chk("A2 planted +0.019, significant: not REACH (a seed that drops the point conjunct passes it)", m.seed_call(cp(0.019, 0.006, 0.032)), "INC")
    chk("A2 planted +0.005, significant, upper CI 0.012: not REACH, INC", m.seed_call(cp(0.005, 0.001, 0.012)), "INC")
    chk("A2 planted +0.005, significant, upper CI 0.009: not REACH, NOT", m.seed_call(cp(0.005, 0.001, 0.009)), "NOT")
    chk("A2 planted +0.025, NOT significant (lower CI < 0): not REACH (a seed that drops the lower-CI conjunct passes it)", m.seed_call(cp(0.025, -0.003, 0.053)), "INC")
    chk("A2 planted +0.040, lower CI exactly 0: not REACH", m.seed_call(cp(0.040, 0.0, 0.080)), "INC")
    chk("A2 a negative lower CI with a huge point is not REACH", m.seed_call(cp(0.100, -0.001, 0.200)), "INC")
    chk("A2 nan point is not REACH", m.seed_call({"point": float("nan"), "lo": 0.01, "hi": 0.08}), "INC")
    chk("seed NOT", m.seed_call(ci(-0.05, 0.009)), "NOT")
    chk("A2 NOT-REACHABLE is unchanged: hi < 0.01, any point", m.seed_call(cp(0.001, -0.004, 0.0099)), "NOT")
    chk("seed edge hi == 0.01 is not NOT", m.seed_call(ci(-0.05, 0.01)), "INC")
    chk("seed straddle", m.seed_call(ci(-0.01, 0.03)), "INC")
    chk("seed nan lo", m.seed_call({"point": float("nan"), "lo": float("nan"), "hi": float("nan")}), "INC")
    chk("both REACH", m.reach_call(["REACH", "REACH"]), "CTX-REACHABLE")
    chk("one REACH one INC", m.reach_call(["REACH", "INC"]), "INCONCLUSIVE")
    chk("REACH vs NOT disagree", m.reach_call(["REACH", "NOT"]), "INCONCLUSIVE")
    chk("both NOT", m.reach_call(["NOT", "NOT"]), "NOT-REACHABLE")
    chk("one NOT", m.reach_call(["NOT", "INC"]), "INCONCLUSIVE")
    chk("pick gain seed", m.pick_gain_seed(ci(0.5, 2.0), ci(-0.03, 0.0)), True)
    chk("pick gain: dpdms lo == 0 fails", m.pick_gain_seed(ci(0.0, 2.0), ci(-0.03, 0.0)), False)
    chk("pick gain: dac hi > 0 fails", m.pick_gain_seed(ci(0.5, 2.0), ci(-0.03, 0.001)), False)
    chk("pick gain needs both seeds", m.pick_gain_call([True, False]), False)
    chk("pick gain both", m.pick_gain_call([True, True]), True)
    chk("label null fires", m.null_label_fires([ci(0.03, 0.1)]), True)
    # ---- Amendment 3: a C-NULL-LABEL replicate FIRES iff it is CTX-REACHABLE by the Amendment-2 rule (lower CI > 0 AND point >= +0.02), read on its own seed; the VOID count rule is unchanged
    chk("A3 label null fires at the sealed edge lo == 0.02 (the sealed rule said no)", m.null_label_fires([ci(0.02, 0.1)]), True)
    chk("A3 label null fires: point 0.05, lo 0.01 (the sealed rule did not fire)", m.null_label_fires([cp(0.05, 0.01, 0.09)]), True)
    chk("A3 label null: lo > 0.02 still fires whatever the point", m.null_label_fires([cp(0.025, 0.021, 0.03)]), True)
    chk("A3 planted null replicate +0.025, lower CI > 0: FIRES (the sealed rule missed it)", m.null_label_fires([cp(0.025, 0.004, 0.046)]), True)
    chk("A3 planted null replicate +0.015, significant: does not fire (point conjunct)", m.null_label_fires([cp(0.015, 0.004, 0.026)]), False)
    chk("A3 planted null replicate +0.05, lower CI < 0: does not fire (lower-CI conjunct)", m.null_label_fires([cp(0.05, -0.002, 0.102)]), False)
    chk("A3 label null: lower CI == 0 does not fire (strict)", m.null_label_fires([cp(0.04, 0.0, 0.08)]), False)
    chk("A3 label null: lower CI barely above 0 with point >= +0.02 fires", m.null_label_fires([cp(0.025, 1e-9, 0.05)]), True)
    chk("A3 label null: point == +0.02 fires (>=)", m.null_label_fires([cp(0.020, 0.001, 0.04)]), True)
    chk("A3 label null: point just below +0.02 does not fire", m.null_label_fires([cp(0.0199, 0.001, 0.04)]), False)
    chk("A3 label null: nan point does not fire", m.null_label_fires([{"point": float("nan"), "lo": 0.01, "hi": 0.08}]), False)
    chk("A3 label null: nan lower bound does not fire", m.null_label_fires([{"point": 0.03, "lo": float("nan"), "hi": 0.08}]), False)
    chk("A3 label null: no seed read does not fire", m.null_label_fires([]), False)
    chk("A3 label null over several seeds fires only if EVERY seed is CTX-REACHABLE", m.null_label_fires([cp(0.03, 0.01, 0.05), cp(0.03, -0.01, 0.07)]), False)
    chk("A3 label null over several seeds, all CTX-REACHABLE", m.null_label_fires([cp(0.03, 0.01, 0.05), cp(0.025, 0.002, 0.05)]), True)
    grid = [cp(pt, lo, 0.2) for pt in (-0.01, 0.0, 0.0199, 0.02, 0.0201, 0.05) for lo in (-0.01, 0.0, 1e-9, 0.01, 0.02, 0.03)]
    chk("A3 label null IS seed_call == REACH on a grid (one definition, no second copy of the rule)", [m.null_label_fires([g]) for g in grid], [m.seed_call(g) == "REACH" for g in grid])
    chk("label null void on any fire", m.null_label_void([False, True, False]), True)
    chk("label null clean", m.null_label_void([False, False, False]), False)
    chk("A3 the VOID COUNT rule is unchanged: one fire of three voids, none does not", [m.null_label_void([False, False, True]), m.null_label_void([True, True, True]), m.null_label_void([])], [True, True, False])
    chk("A3 VOID on a machinery that manufactures +0.03 (the amended call would read it CTX-REACHABLE): VOID", m.final_outcome("CTX-REACHABLE", True, True, m.null_label_void([m.null_label_fires([cp(0.03, 0.008, 0.052)])]), True), "VOID(C-NULL-LABEL)")
    chk("power needs both seeds", m.power_ok([ci(0.01, 0.05)]), False)
    chk("power both", m.power_ok([ci(0.01, 0.05), ci(0.01, 0.05)]), True)
    chk("power: lower CI must exclude 0", m.power_ok([ci(0.0, 0.06), ci(0.01, 0.05)]), False)
    chk("power: point above the band fails", m.power_ok([ci(0.01, 0.12), ci(0.02, 0.07)]), False)
    chk("power: point below the band fails", m.power_ok([ci(0.001, 0.02), ci(0.01, 0.05)]), False)
    chk("next arm integrate", m.next_arm("CTX-REACHABLE", True), "INTEGRATE-SH5")
    chk("next arm stop", m.next_arm("CTX-REACHABLE", False), "STOP-AT-DIAGNOSTIC")
    chk("next arm P6", m.next_arm("NOT-REACHABLE", False), "P6-PROX-AUX")
    chk("next arm inconclusive", m.next_arm("INCONCLUSIVE", True), "NAMED-BY-REPORT")
    chk("final: conversions cannot create a call", m.final_outcome("INCONCLUSIVE", True, True, False, True), "INCONCLUSIVE")
    chk("final: NOT without power -> UNDERPOWERED", m.final_outcome("NOT-REACHABLE", True, True, False, False), "UNDERPOWERED")
    chk("final: NOT with power stands", m.final_outcome("NOT-REACHABLE", True, True, False, True), "NOT-REACHABLE")
    chk("final: REACH stands without power", m.final_outcome("CTX-REACHABLE", True, True, False, False), "CTX-REACHABLE")
    chk("final: gate fail", m.final_outcome("CTX-REACHABLE", False, True, False, True), "NO-CALL(GATE)")
    chk("final: C-REACH fail", m.final_outcome("CTX-REACHABLE", True, False, False, True), "NO-CALL(C-REACH)")
    chk("final: label-null void", m.final_outcome("CTX-REACHABLE", True, True, True, True), "VOID(C-NULL-LABEL)")
    chk("final: shuffled-ctx arm also gains", m.final_outcome("CTX-REACHABLE", True, True, False, True, True), "GAIN-NOT-FROM-CTX")
    chk("final: shuffled gain does not touch NOT", m.final_outcome("NOT-REACHABLE", True, True, False, True, True), "NOT-REACHABLE")
    chk("final: shuffled gain does not touch INC", m.final_outcome("INCONCLUSIVE", True, True, False, True, True), "INCONCLUSIVE")
    # GAIN-NOT-FROM-CTX reads the ctx-shuffled arm with the NEW rule (the composition `analyze` uses: shuf_gain = every seed's seed_call(delta_F_shuf) == "REACH")
    shuf = [m.seed_call(cp(0.025, 0.002, 0.050)) for _ in range(2)]                    # REACH under Amendment 2 (the sealed rule said INC)
    chk("A2 S-shuf seed call uses the new rule", shuf, ["REACH", "REACH"])
    chk("A2 S-shuf passing the NEW rule on both seeds -> GAIN-NOT-FROM-CTX", m.final_outcome("CTX-REACHABLE", True, True, False, True, all(c == "REACH" for c in shuf)), "GAIN-NOT-FROM-CTX")
    shuf1 = [m.seed_call(cp(0.025, 0.002, 0.050)), m.seed_call(cp(0.025, -0.002, 0.050))]
    chk("A2 S-shuf passing on ONE seed only does not convert", m.final_outcome("CTX-REACHABLE", True, True, False, True, all(c == "REACH" for c in shuf1)), "CTX-REACHABLE")
    return bad


# ---- the pick-level read and the reliability-at-picks diagnostic (prereg s6, DESCRIPTIVE) ----------------------------------------------------------------------------------------------
def pick_level(z6: np.ndarray, lab: np.ndarray, z_dac_new: np.ndarray, ep_col, stats) -> dict:
    """Picks by P2's own rule (`pickrule_lib.rule_pick`, planner.py:558-561, float64) for: P2 (stored z6); the same rule with the DAC column replaced (`rule[S]`); and the EP-led composite
    (column 2 = EP-R1's OOF EP) with and without the replacement.  Label-lookup PDMS x 100 and the NC / DAC / EP-R1-unsafe pick rates, with paired log-cluster differences to P2.
    `stats` = pickrule_lib.Stats bound to this analysis set's logs.  Tier: label-lookup, SEEN scenes, P2 in-sample."""
    FL, PK = refs()
    pdms = PK.pdms_from_labels(lab)
    ones = np.ones(len(lab))
    zs = {"P2": z6}
    z_r = z6.copy()
    z_r[..., DAC_COL] = z_dac_new
    zs["rule[S]"] = z_r
    if ep_col is not None:
        z_ep = z6.copy()
        z_ep[..., 2] = ep_col
        zs["EP"] = z_ep
        z_eps = z_ep.copy()
        z_eps[..., DAC_COL] = z_dac_new
        zs["EP[S]"] = z_eps
    fl = {k: PK.pick_flags(lab, pdms, PK.rule_pick(z)) for k, z in zs.items()}
    out = {}
    for k, f in fl.items():
        out[k] = {"nc0_rate": stats.rate(f["nc0"]), "dac0_rate": stats.rate(f["dac0"]), "unsafe_epr1_rate": stats.rate(f["unsafe_epr1"]), "pdms_x100": PK.scale_ci(stats.mean(f["pd"]), 100.0)}
    base = fl["P2"]
    for k, f in fl.items():
        if k == "P2":
            continue
        out[k]["vs_P2"] = {"d_dac0": stats.rate_diff(f["dac0"], base["dac0"]), "d_nc0": stats.rate_diff(f["nc0"], base["nc0"]), "d_unsafe_epr1": stats.rate_diff(f["unsafe_epr1"], base["unsafe_epr1"]),
                           "d_pdms_x100": PK.scale_ci(stats.ratio_diff(f["pd"], ones, base["pd"], ones), 100.0)}
    if "EP[S]" in fl:
        for a_, b_ in (("EP[S]", "EP"),):
            fa, fb = fl[a_], fl[b_]
            out[f"{a_} minus {b_}"] = {"d_dac0": stats.rate_diff(fa["dac0"], fb["dac0"]), "d_pdms_x100": PK.scale_ci(stats.ratio_diff(fa["pd"], ones, fb["pd"], ones), 100.0)}
    return out


def reliability_at_picks(z6_rule: np.ndarray, z_dac_row: np.ndarray, lab_dac: np.ndarray, stats, code: np.ndarray = None) -> dict:
    """R53 diagnostic: at the picks of the rule `z6_rule`, observed DAC-unsafe mass / predicted mass of the row `z_dac_row` (predicted = 1 - sigmoid(z)).  ratio 1 = calibrated at the picks;
    < 1 over-predicts the risk, > 1 under-predicts it.  Whole set and the F band (pick's stratum = Q4).  Log-cluster CIs.  A threshold share is NOT a calibration claim (R53)."""
    FL, PK = refs()
    pk = PK.rule_pick(z6_rule)
    z = PK.at(z_dac_row, pk)
    pred = 1.0 - PK.sigmoid64(z)
    obs = (PK.at(lab_dac, pk) == 0.0).astype(np.float64)
    out = {"all": stats.ratio(obs, pred), "n_unsafe_picks": int(obs.sum()), "predicted_mass": float(pred.sum()), "observed_mass": float(obs.sum())}
    if code is not None:
        inF = (PK.at(code, pk) == 4).astype(np.float64)
        out["F"] = stats.ratio(obs * inF, pred * inF)
        out["n_picks_in_F"] = int(inF.sum())
    return out
