"""PRIVGEOM-CLEAN library: the SIGN-CLEAN block, its preparation (within-set centring), the planted / null blocks, and the REGISTERED decision functions with their branch table.

The fit / score / estimator machinery is PRIVGEOM-SAFE's, imported UNCHANGED (`pg_lib`, sha256-pinned by the prereg): this module adds no optimiser, no recipe constant and no estimator.  Everything here is
a pure function so that `branch_table_failures` and its RED mutations pin it.  REGISTERED SOURCE: PREREG_PRIVGEOM_CLEAN.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "ref")))
import pg_lib as P                                                                       # noqa: E402

FL = P.FL
SCHEMA = "privgeom-clean-v1"
ROW, ROW_COL = "dac", 1
PRIMARY_ARM = "LGc"                               # DQ1 (prereg s3, s14): "LGc" = the within-set-centred block (default), "LGu" = the uncentred block (PRIVGEOM-SAFE `drop0` exactly).  The companion is the other one.
COMPANION_ARM = "LGu" if PRIMARY_ARM == "LGc" else "LGc"
# ---- the block (prereg s3): the 9 key-step |signed clearance| columns = g[..., 2:11]; |dac_clear_min| (col 0) and dac_frac_out (col 1) are EXCLUDED ---------------------------------------------
BLOCK_COLS = tuple(range(2, 11))
BLOCK_NAMES = tuple(f"abs_clear_t{s:02d}" for s in (0, 5, 10, 15, 20, 25, 30, 35, 40))
# ---- decision (prereg s6) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
CONFIRM_POINT, NULL_UPPER = 0.02, 0.02            # CONFIRM iff lower CI > 0 AND point >= +0.02 (SH5 Amendment 2's CTX-REACHABLE rule); NULL iff upper CI < +0.02 (the interval lies wholly below the same materiality bar)
NULL_REPS, NULL_SALT, NULL_SEED_BASE = 5, 7919, 20261019
PLANT_SEED = 20261019 + 31
PLANT_SIGMA = None                                  # set from plant_calibration.json by the prereg (a constant, committed before any read)
REACH_MIN_AUC, AGREE_MIN, LOSS_FRAC_DISCLOSE = 0.99, 0.99, 0.05
SEEDS = P.SEEDS
BOOT_B, BOOT_SEED = P.BOOT_B, P.BOOT_SEED


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the registered decision functions
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def confirm_call(ci: dict) -> bool:
    """s6: the lower CI of Delta_F is > 0 AND its point is >= +0.02."""
    return bool(ci["lo"] > 0.0 and ci["point"] >= CONFIRM_POINT)


def null_call(ci: dict) -> bool:
    """s6: the upper CI of Delta_F is < +0.02 (wholly below the CONFIRM point bar, so CONFIRM and NULL are mutually exclusive: point <= upper CI)."""
    return bool(ci["hi"] < NULL_UPPER)


def seed_call(ci: dict) -> str:
    """CONFIRM is tested FIRST (one definition, used for the raw call, the machinery null and the plant)."""
    if confirm_call(ci):
        return "CONFIRM"
    if null_call(ci):
        return "NULL"
    return "INCONCLUSIVE"


def row_call(calls: list) -> str:
    """BOTH seeds must agree."""
    if len(calls) and all(c == "CONFIRM" for c in calls):
        return "CONFIRM"
    if len(calls) and all(c == "NULL" for c in calls):
        return "NULL"
    return "INCONCLUSIVE"


def small_reading(cis: list) -> bool:
    """DESCRIPTIVE sub-reading (of an INCONCLUSIVE or a NULL): a real but sub-bar held-out effect (both seeds: lower CI > 0 and point < +0.02)."""
    return bool(len(cis) and all(c["lo"] > 0.0 and c["point"] < CONFIRM_POINT for c in cis))


def null_fires(ci: dict) -> bool:
    """s7 C-NULL: a replicate FIRES iff it satisfies THE CALL's own rule (`seed_call == CONFIRM`) on the seed it is read on (SH5 Amendment 3: a machinery null read by the test the claim uses)."""
    return seed_call(ci) == "CONFIRM"


def null_void(fires: list) -> bool:
    """s7: VOID iff ANY replicate fires."""
    return bool(any(fires))


def power_ok(cis_by_seed: list) -> bool:
    """s7 C-POWER: the CONFIRM rule must FIRE on the planted effect on BOTH seeds."""
    return bool(len(cis_by_seed) and all(seed_call(c) == "CONFIRM" for c in cis_by_seed))


def final_call(call: str, admit_ok: bool, void_inputs: bool, void_null: bool, power: bool) -> str:
    """s6 conversions (they can only REMOVE a call; CONFIRM is never converted by the power control)."""
    if not admit_ok:
        return "NO-CALL(ADMIT)"
    if void_inputs:
        return "VOID-INPUTS"
    if void_null:
        return "VOID"
    if call == "NULL" and not power:
        return "UNDERPOWERED"
    return call


def branch_table_failures(mod=None) -> list:
    """Every entry must hold; a RED mutation of any decision function must make at least one entry fail."""
    m = mod if mod is not None else sys.modules[__name__]
    bad = []
    ci = lambda lo, hi, pt=None: {"lo": lo, "hi": hi, "point": (lo + hi) / 2 if pt is None else pt}   # noqa: E731

    def chk(name, got, want):
        if got != want:
            bad.append((name, got, want))
    chk("confirm: lo>0 and point at bound", m.confirm_call(ci(0.001, 0.04, 0.02)), True)
    chk("confirm: point just below", m.confirm_call(ci(0.001, 0.04, 0.0199)), False)
    chk("confirm: lo at 0", m.confirm_call(ci(0.0, 0.05, 0.03)), False)
    chk("confirm: lo below 0", m.confirm_call(ci(-0.001, 0.05, 0.03)), False)
    chk("confirm: lo just above 0", m.confirm_call(ci(1e-6, 0.05, 0.03)), True)
    chk("confirm: clearly", m.confirm_call(ci(0.012, 0.045, 0.028)), True)
    chk("null: hi just below", m.null_call(ci(-0.05, 0.0199, 0.0)), True)
    chk("null: hi at bound", m.null_call(ci(-0.05, 0.02, 0.0)), False)
    chk("null: hi above", m.null_call(ci(-0.05, 0.0201, 0.0)), False)
    chk("null: hi at the old SH5 NOT-REACHABLE bound is not enough to stay INCONCLUSIVE", m.null_call(ci(-0.01, 0.015, 0.0)), True)
    chk("seed confirm", m.seed_call(ci(0.012, 0.045, 0.028)), "CONFIRM")
    chk("seed null", m.seed_call(ci(-0.03, 0.005, -0.01)), "NULL")
    chk("seed straddle", m.seed_call(ci(-0.01, 0.05, 0.02)), "INCONCLUSIVE")
    chk("seed small real effect, upper CI above the bar", m.seed_call(ci(0.003, 0.025, 0.014)), "INCONCLUSIVE")
    chk("seed small real effect, upper CI below the bar", m.seed_call(ci(0.003, 0.0195, 0.011)), "NULL")
    chk("seed point>=0.02 but lo<=0", m.seed_call(ci(-0.002, 0.045, 0.021)), "INCONCLUSIVE")
    chk("row both confirm", m.row_call(["CONFIRM", "CONFIRM"]), "CONFIRM")
    chk("row one confirm", m.row_call(["CONFIRM", "INCONCLUSIVE"]), "INCONCLUSIVE")
    chk("row confirm vs null", m.row_call(["CONFIRM", "NULL"]), "INCONCLUSIVE")
    chk("row both null", m.row_call(["NULL", "NULL"]), "NULL")
    chk("row one null", m.row_call(["NULL", "INCONCLUSIVE"]), "INCONCLUSIVE")
    chk("row empty", m.row_call([]), "INCONCLUSIVE")
    chk("small both seeds", m.small_reading([ci(0.003, 0.025, 0.014), ci(0.002, 0.026, 0.0145)]), True)
    chk("small: one seed has point>=0.02", m.small_reading([ci(0.003, 0.025, 0.014), ci(0.002, 0.04, 0.021)]), False)
    chk("small: lower CI <= 0", m.small_reading([ci(0.0, 0.025, 0.012), ci(0.002, 0.026, 0.0145)]), False)
    chk("small: empty", m.small_reading([]), False)
    chk("null fires = the call's rule (confirm)", m.null_fires(ci(0.004, 0.04, 0.022)), True)
    chk("null fires: point >= 0.02 but lower CI <= 0 does not fire", m.null_fires(ci(-0.001, 0.05, 0.021)), False)
    chk("null fires: lo > 0 but point < 0.02 does not fire", m.null_fires(ci(0.001, 0.03, 0.015)), False)
    chk("null void: one of five", m.null_void([False, False, True, False, False]), True)
    chk("null void: none", m.null_void([False] * 5), False)
    chk("power both seeds", m.power_ok([ci(0.01, 0.05, 0.03), ci(0.012, 0.05, 0.031)]), True)
    chk("power one seed", m.power_ok([ci(0.01, 0.05, 0.03), ci(-0.001, 0.05, 0.03)]), False)
    chk("power empty", m.power_ok([]), False)
    chk("final: null unpowered", m.final_call("NULL", True, False, False, False), "UNDERPOWERED")
    chk("final: confirm stays when unpowered", m.final_call("CONFIRM", True, False, False, False), "CONFIRM")
    chk("final: null powered", m.final_call("NULL", True, False, False, True), "NULL")
    chk("final: void null", m.final_call("CONFIRM", True, False, True, True), "VOID")
    chk("final: void inputs", m.final_call("NULL", True, True, False, True), "VOID-INPUTS")
    chk("final: void inputs over null", m.final_call("CONFIRM", True, True, True, True), "VOID-INPUTS")
    chk("final: admit refused over everything", m.final_call("CONFIRM", False, True, True, True), "NO-CALL(ADMIT)")
    chk("final: inconclusive stays", m.final_call("INCONCLUSIVE", True, False, False, False), "INCONCLUSIVE")
    return bad


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def clean_block(g: np.ndarray) -> np.ndarray:
    """g [N, 64, 15] -> the 9 key-step |clearances| [N, 64, 9] float64.  Columns 0 (|c_min|) and 1 (frac_out) and every NC column are NEVER read."""
    return np.abs(np.asarray(g, np.float64)[..., list(BLOCK_COLS)])


def block_hash(g: np.ndarray, ok: np.ndarray) -> str:
    """sha256 of the sign-clean block of the OK sets (float64 |g[..., 2:11]|, C-contiguous): the one definition used by the G-LIVE registration and by the gate's C-ID-EXTRACT."""
    import hashlib
    okb = clean_block(g)[np.asarray(ok, bool)]
    return hashlib.sha256(np.ascontiguousarray(okb).tobytes()).hexdigest()


def centre_within_set(x: np.ndarray) -> np.ndarray:
    """subtract each set's mean over its 64 proposals from every column (removes the BETWEEN-set information and nothing else)."""
    x = np.asarray(x, np.float64)
    return x - x.mean(axis=1, keepdims=True)


def prepared(x: np.ndarray, centred: bool) -> np.ndarray:
    """the block as the fit sees it BEFORE the fold-wise standardisation (`pg_lib.standardise`)."""
    return np.ascontiguousarray((centre_within_set(x) if centred else np.asarray(x, np.float64)).astype(np.float32))


def weak_plant(label: np.ndarray, sigma: float) -> np.ndarray:
    """s7 C-POWER: label + sigma * N(0, 1), one column [N, 64, 1] float64 (seeded, DAC column)."""
    eps = np.random.default_rng(PLANT_SEED + ROW_COL).standard_normal(label.shape)
    return (np.asarray(label, np.float64) + float(sigma) * eps)[..., None]


def label_plant(label: np.ndarray) -> np.ndarray:
    return np.asarray(label, np.float64)[..., None]


def null_block(block: np.ndarray, rep: int) -> np.ndarray:
    """s7 C-NULL: the block's 64 proposals permuted WITHIN each set (one permutation per set moves the whole 9-vector); centring is permutation-invariant, so this IS the within-set-centred null."""
    return P.permute_within_set(np.asarray(block, np.float32), NULL_SEED_BASE + NULL_SALT * int(rep))
