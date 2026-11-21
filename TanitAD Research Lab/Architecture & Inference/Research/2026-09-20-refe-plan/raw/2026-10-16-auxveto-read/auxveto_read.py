#!/usr/bin/env python3
"""auxveto_read.py -- the read of `eval/PREREG_AUXVETO.md`, implemented EXACTLY as §3-§9 specify (nothing reinterpreted; open details are fixed in QUESTIONS.md).

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I auxveto_read.py --gate      # pre-read gates only (§9 checks 1, 2, 3; no arm Delta is computed)
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I auxveto_read.py --run       # the full read -> AUXVETO_READ.json (and RESULT.md via --write-result)

THE RULE (§4), one rule, no fitted parameter: a proposal is VETOED when any footprint-corner sample over the 41-step horizon has a signed distance < -m (m = 0.25 m)
outside the arm's drivable area; the pick is the highest B score among the NON-vetoed proposals; every proposal vetoed => B's pick.  The signed distance of EVERY arm is
computed from the binary MASK (predicted: logit > 0) by the label store's own half-cell-corrected Euclidean distance transform, on the raster's own grid.
THE ESTIMATOR (§5): stratified navtest-population Delta = 529/12146 * Delta_Z + 11617/12146 * Delta_N, paired log-cluster bootstrap over the 123 C3 logs, 10,000 resamples, seed 20260927.
THE CI'S QUESTION: "another draw of logs".  BLIND to the head's training variance (one seed) and to the tracker.

Evidence class of every output: MEASURED (this script, on the artifacts named in AUXVETO_READ.json).  No number here is tuned: the margin, the rule, the arms and the estimator are the prereg's.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PK = HERE.parent.parent
for _p in (HERE, PK / "eval", PK / "refe", PK / "raw/2026-10-16-auxveto-feasibility"):
    sys.path.insert(0, str(_p))
import geodac as G  # noqa: E402  -- the validated instrument (footprint, pose densify, bilinear sampling), reused, not reimplemented

# ------------------------------------------------------------------------------------------------------------------ the prereg's constants
PREREG_PATH = PK / "eval/PREREG_AUXVETO.md"
PREREG_PREFIX = "ef9f17fa26a9d818"                       # the file sha256 of PREREG_AUXVETO.md at its landing commit 4f7e105
TOKENS_SHA_PREFIX = "030725670eac74e8"                   # §2 tokens.json sha256 `03072566...e2254d`
TABLE_SHA_PREFIX = "cf30519051bea03b"                    # §2 table.npz sha256 `cf30519051bea03b...`
STORE_SHA_PREFIX = "985a3bc77f3ab7a9"                    # §3 label-store sha256 `985a3bc7...`
N_Z, N_N = 529, 11617                                    # §5: the census and the non-zeroed population
N_POP = N_Z + N_N                                        # 12,146 = all of navtest
W_Z, W_N = N_Z / N_POP, N_N / N_POP                      # §5: the weights, FIXED in every resample
MARGIN = 0.25                                            # §4: an INSTRUMENT property, not tuned
MARGIN_CURVE = (0.0, 0.5, 1.0, 2.0)                      # §4: reported, never gating
B_BOOT, SEED = 10000, 20260927                           # §5
IOU_TARGET, IOU_TOL = 0.675218, 1e-6                     # §3 identity check
IOU_LO, IOU_HI = 4, 24                                   # x in [0, 40) m on the 2 m grid whose x0 = -8
BIG_M = 1000.0                                           # a mask with no background / no drivable cell (label store's degenerate branches)
GATE_FEATURES = ("zero", "random")

C3 = PK / "raw/2026-10-11-c3-selection-headroom"
STORE = Path("D:/Projects/TanitAD-artifacts/aux_drivable/label_store_navtest")
ART = Path("D:/Projects/TanitAD-artifacts")
UNIT_DIRS = {"V-PRED": ART / "scene_tap_probe_a2/units", "V-PRED-2k": ART / "scene_tap_probe/units", "V-PRED-400": ART / "scene_tap_probe_a3r/units"}
TRACKED_DIR = Path("C:/Users/Admin/tmpauxv/tracked")
PRED_GEOMETRY = PK / "raw/2026-10-16-auxveto-feasibility/PRED_GEOMETRY.json"
PRED_GEOM = G.PRED_GEOM
GT_GEOM = G.GT_GEOM


# ------------------------------------------------------------------------------------------------------------------ prereg guard
def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assert_prereg() -> str:
    """Refuse to run unless the prereg file is the one in force (sha256 prefix ef9f17fa26a9d818)."""
    h = sha256_file(PREREG_PATH)
    if not h.startswith(PREREG_PREFIX):
        raise SystemExit(f"REFUSED: {PREREG_PATH} sha256 {h[:16]} != the prereg in force {PREREG_PREFIX}")
    return h


# ------------------------------------------------------------------------------------------------------------------ the world (one scorer's proposals + the table)
@dataclass
class World:
    token: list           # [N] 16-hex token strings (TABLE order); every lookup is by this string, never by row position
    log: list             # [N]
    is_z: np.ndarray      # [N] bool: the Z census (u400 pick has NC x DAC = 0)
    proposals: np.ndarray  # [N, M, 8, 3] ego-frame rear-axle poses at 0.5 ... 4.0 s
    sel: np.ndarray       # [N, M] the B selection score (rl_val_eval.select(...)[1])
    sub: np.ndarray       # [N, M, 6] NAVSIM v1.1 sub-scores (NC, DAC, EP, TTC, C, DDC)
    pdms: np.ndarray      # [N, M]
    b_pick: np.ndarray = field(init=False)   # [N] B's pick = argmax of `sel` (lowest index on ties), fixed at construction

    def __post_init__(self):
        self.b_pick = b_picks(self.sel)

    @property
    def n(self):
        return len(self.token)


def b_picks(sel: np.ndarray) -> np.ndarray:
    """B's pick = argmax of the selection score, lowest index on ties (`rl_val_eval.select`)."""
    return np.asarray([int(np.argmax(s)) for s in sel])


def load_world():
    """-> (World, banked pick [N])."""
    import rl_val_eval as R
    z = np.load(C3 / "table.npz", allow_pickle=False)
    tj = json.load(open(C3 / "tokens.json"))
    token = [str(t) for t in z["token"]]
    sel = np.stack([R.select(z["logits"][i], "navsim_v1_lane", space="logit")[1] for i in range(len(token))])
    st = np.array([tj["stratum"][t] for t in token])
    if not set(st) <= set(GATE_FEATURES):
        raise SystemExit(f"unknown strata {set(st)}")
    return World(token=token, log=[tj["token_log"][t] for t in token], is_z=(st == "zero"), proposals=z["proposals"].astype(np.float64), sel=sel,
                 sub=z["sub"].astype(np.float64), pdms=z["pdms"].astype(np.float64)), z["pick"].astype(np.int64)


# ------------------------------------------------------------------------------------------------------------------ the raster -> signed distance
def sdf_from_mask(mask, cell_m: float) -> np.ndarray:
    """The label store's own `signed_distance` (refe/aux_label_store.py), on the raster's OWN grid: metres, + inside / - outside, half-cell corrected
    (the zero crossing sits on the cell edge and the slope is metric under bilinear sampling).  No padding, no clip.  A mask with no background reads
    +BIG_M everywhere, a mask with no drivable cell -BIG_M (the store's degenerate branches; `scipy` would measure the distance to the array border)."""
    from scipy.ndimage import distance_transform_edt as edt
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.full(m.shape, -BIG_M)
    if m.all():
        return np.full(m.shape, BIG_M)
    d_in, d_out = edt(m), edt(~m)
    return np.where(m, d_in - 0.5, -(d_out - 0.5)) * cell_m


def pred_mask(logit, sdf) -> np.ndarray:
    """§3 'Which raster output is read': the MASK, logit > 0, for every predicted arm.  The head's SDF channel is NOT read."""
    return np.asarray(logit) > 0


def gt_mask_to_2m(mask05: np.ndarray) -> np.ndarray:
    """The GT mask [160, 128] (0.5 m cells) on the head's 2 m grid [40, 32]: the 4 x 4 mean >= 0.5 (aux_s1_readout.py:321, pred_geometry_check.py)."""
    return mask05.astype(np.float64).reshape(40, 4, 32, 4).mean(axis=(1, 3)) >= 0.5


def iou_pooled(pred: np.ndarray, truth: np.ndarray, lo: int = IOU_LO, hi: int = IOU_HI) -> float:
    """Pooled IoU of boolean masks [N, 40, 32] over the x cells [lo, hi) (x in [0, 40) m)."""
    p, t = pred[:, lo:hi], truth[:, lo:hi]
    return float((p & t).sum() / (p | t).sum())


def assemble_units(unit_files, table_tokens):
    """The held-out cross-fold predictions -> {token string: (logit [40,32], sdf [40,32])}.  `held_idx` indexes the TABLE's row order, so the token string is
    taken from the TABLE's token list (never from tokens.json's order, which differs).  Every token must be held out exactly once."""
    n = len(table_tokens)
    logit = np.full((n, 40, 32), np.nan, np.float32)
    sdf = np.full((n, 40, 32), np.nan, np.float32)
    seen = np.zeros(n, int)
    for f in unit_files:
        z = np.load(f, allow_pickle=False)
        h = z["held_idx"]
        logit[h], sdf[h] = z["logit"], z["sdf"]
        seen[h] += 1
    if not (seen == 1).all():
        raise SystemExit(f"held-out coverage is not exactly-once: min {seen.min()} max {seen.max()}")
    return {table_tokens[i]: (logit[i], sdf[i]) for i in range(n)}


# ------------------------------------------------------------------------------------------------------------------ the veto statistic and the pick
def min_margins(field: np.ndarray, geom: G.RasterGeom, poses: np.ndarray) -> np.ndarray:
    """[M, 8, 3] -> [M]: the minimum, over the 41 steps x 4 footprint corners, of the bilinearly sampled signed field (`geodac.geodac`, unknown = drivable)."""
    return G.geodac(field, geom, poses, horizon="navsim41", mode="bilinear", unknown="drivable").min_margin_m


def min_margins_corners(field: np.ndarray, geom: G.RasterGeom, corners: np.ndarray) -> np.ndarray:
    """[M, 41, 4, 2] footprint corners (e.g. NAVSIM's tracked states) -> [M]."""
    return G.geodac_corners(field, geom, corners, mode="bilinear", unknown="drivable").min_margin_m


def vetoed(margin: np.ndarray, m: float) -> np.ndarray:
    """§4: VETOED iff some footprint sample lies more than m metres outside, i.e. its signed distance is strictly below -m."""
    return np.asarray(margin) < -m


def masked_pick(sel: np.ndarray, veto: np.ndarray, b_pick: int) -> int:
    """§4 (2): the highest B score among the NON-vetoed proposals (lowest index on ties); (3): every proposal vetoed => B's pick."""
    veto = np.asarray(veto, bool)
    if veto.all():
        return int(b_pick)
    return int(np.argmax(np.where(veto, -np.inf, sel)))


def all_picks(w: World, veto: np.ndarray) -> np.ndarray:
    bp = w.b_pick
    return np.asarray([masked_pick(w.sel[i], veto[i], bp[i]) for i in range(w.n)])


def margins_from_masks(w: World, masks: dict, cell_m: float, geom: G.RasterGeom, *, transform=None) -> np.ndarray:
    """-> [N, M] min margins.  masks: {token: bool mask}; the signed distance of each is derived from its mask (the rule's §3)."""
    out = np.empty(w.sel.shape)
    for i, t in enumerate(w.token):
        mk = masks[t]
        if transform is not None:
            mk = transform(mk)
        out[i] = min_margins(sdf_from_mask(mk, cell_m), geom, w.proposals[i])
    return out


def odac_veto(w: World) -> np.ndarray:
    """O-DAC (the oracle): NAVSIM's own DAC sub-score vetoes every proposal with DAC = 0."""
    return w.sub[:, :, 1] == 0


# ------------------------------------------------------------------------------------------------------------------ outcomes of a set of picks
def pick_outcomes(w: World, picks: np.ndarray) -> dict:
    """Table lookups at the picks, x100 (§5), plus the recovery indicators (Q5)."""
    ar = np.arange(w.n)
    s, p = w.sub[ar, picks], w.pdms[ar, picks]
    return {"pdms": 100 * p, "nc": 100 * s[:, 0], "dac": 100 * s[:, 1], "ep": 100 * s[:, 2], "ttc": 100 * s[:, 3], "comfort": 100 * s[:, 4], "ddc": 100 * s[:, 5],
            "rec_ncdac": (s[:, 0] * s[:, 1] > 0).astype(float), "rec_dac": (s[:, 1] > 0).astype(float), "rec_pdms": (p > 0).astype(float)}


DELTA_KEYS = ("pdms", "nc", "dac", "ep", "ttc", "comfort", "ddc")


def deltas(w: World, picks: np.ndarray) -> dict:
    a, b = pick_outcomes(w, picks), pick_outcomes(w, w.b_pick)
    return {k: a[k] - b[k] for k in DELTA_KEYS}


# ------------------------------------------------------------------------------------------------------------------ the estimator (§5)
def log_index(logs):
    names = sorted(set(logs))
    pos = {n: i for i, n in enumerate(names)}
    return np.array([pos[l] for l in logs]), names


def boot_counts(n_logs: int, B: int = None, seed: int = None) -> np.ndarray:
    """[B, n_logs] multiplicity of every log in every resample.  `rng.integers(0, L, L)` per resample, logs sorted by name (= proxy_eval.boot's consumption)."""
    B = B_BOOT if B is None else B
    rng = np.random.default_rng(SEED if seed is None else seed)
    counts = np.zeros((B, n_logs))
    for b in range(B):
        counts[b] = np.bincount(rng.integers(0, n_logs, n_logs), minlength=n_logs)
    return counts


BOOT_STATS = {"strat_boot_calls": 0, "strat_boot_wall_s": 0.0, "boot_counts_wall_s": None}


def strat_boot(X, is_z, log_ix, counts) -> dict:
    """X [S, N] per-token statistics -> for each: the stratified point (W_Z * mean_Z + W_N * mean_N), its paired log-cluster bootstrap 95 % CI, and the
    Z-only and N-only means with their CIs.  In every resample Delta_Z and Delta_N are ratio-of-sums over the resampled logs' tokens; the weights stay FIXED."""
    t0 = time.time()
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    is_z = np.asarray(is_z, bool)
    L = counts.shape[1]
    zf, nf = is_z.astype(float), (~is_z).astype(float)
    nz, nn = np.bincount(log_ix, weights=zf, minlength=L), np.bincount(log_ix, weights=nf, minlength=L)
    sz = np.stack([np.bincount(log_ix, weights=x * zf, minlength=L) for x in X])
    sn = np.stack([np.bincount(log_ix, weights=x * nf, minlength=L) for x in X])
    dz, dn = counts @ nz, counts @ nn
    if (dz == 0).any() or (dn == 0).any():
        raise SystemExit("a bootstrap resample holds no Z (or no N) token")
    bz, bn = (counts @ sz.T) / dz[:, None], (counts @ sn.T) / dn[:, None]
    pz, pn = X[:, is_z].mean(axis=1), X[:, ~is_z].mean(axis=1)
    bs = W_Z * bz + W_N * bn
    q = lambda a: np.percentile(a, [2.5, 97.5], axis=0)  # noqa: E731
    (lo, hi), (zlo, zhi), (nlo, nhi) = q(bs), q(bz), q(bn)
    BOOT_STATS["strat_boot_calls"] += 1
    BOOT_STATS["strat_boot_wall_s"] += time.time() - t0
    return {"point": W_Z * pz + W_N * pn, "lo": lo, "hi": hi, "z": pz, "z_lo": zlo, "z_hi": zhi, "n": pn, "n_lo": nlo, "n_hi": nhi}


# ------------------------------------------------------------------------------------------------------------------ §9 validity checks (each returns (ok, detail))
def check_b_reproduces(b_rule: np.ndarray, b_banked: np.ndarray):
    """§9.1: B reproduces `pick` on 1,058 / 1,058."""
    n_ok = int((np.asarray(b_rule) == np.asarray(b_banked)).sum())
    return n_ok == len(b_banked), {"n_equal": n_ok, "of": int(len(b_banked))}


def check_iou_identity(iou: float):
    """§9.2: the mask the script reads reproduces A2's held-out IoU [0, 40) 0.675218 (+-1e-6)."""
    return abs(iou - IOU_TARGET) <= IOU_TOL, {"iou": iou, "target": IOU_TARGET, "tol": IOU_TOL, "abs_diff": abs(iou - IOU_TARGET)}


def check_zero_control(n_vetoed_total: int, delta_pdms_point: float, picks_equal_b: bool):
    """§9.3: an all-drivable raster vetoes nothing, and Delta = 0.0 exactly."""
    return bool(n_vetoed_total == 0 and delta_pdms_point == 0.0 and picks_equal_b), {"n_vetoed_proposals": int(n_vetoed_total), "delta_pdms_x100": float(delta_pdms_point), "picks_equal_B": bool(picks_equal_b)}


def check_odac_bounds(rec_odac: float, rec_arms: dict):
    """§9.4: O-DAC's Z-stratum recovery >= that of every raster arm."""
    worst = max(rec_arms.values())
    return bool(rec_odac >= worst), {"odac_recovery": rec_odac, "arms": rec_arms, "max_arm": worst}


def check_yflip_below(delta_gt: float, delta_flip: float):
    """§9.5: a y-flipped V-GT raster must fall below V-GT's point estimate."""
    return bool(delta_flip < delta_gt), {"delta_vgt": delta_gt, "delta_vgt_yflip": delta_flip}


# ------------------------------------------------------------------------------------------------------------------ the decision list (§6), applied mechanically
def decide(vpred: dict, vgt: dict) -> dict:
    """vpred / vgt: {'point','lo','hi'} of the primary stratified Delta PDMS x100 (unrounded).  Read top-down."""
    lo, hi, pt = vpred["lo"], vpred["hi"], vpred["point"]
    gt_sep_above = vgt["lo"] > 0
    if lo > 0 and pt >= 0.30:
        return {"row": "AV-GO", "gt_branch": None}
    if lo > 0 and pt < 0.30:
        return {"row": "AV-PROMISING", "gt_branch": None}
    if hi < 0:
        return {"row": "AV-HARM", "gt_branch": "GT ceiling positive (V-GT lower bound > 0): the lever is raster quality (S1c)" if gt_sep_above else "GT ceiling not positive: AV-NULL's next arm"}
    if gt_sep_above:      # V-PRED not separated (lo <= 0 <= hi) AND V-GT lower bound > 0
        return {"row": "AV-CEILING-ONLY", "gt_branch": None}
    return {"row": "AV-NULL", "gt_branch": None}


def prereg_next_arm(row: str) -> str:
    """The row's committed next arm, quoted verbatim from the prereg file's §6 table."""
    for line in PREREG_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"| **{row}**"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return cells[3]
    raise SystemExit(f"row {row} not found in the prereg")


def prereg_condition(row: str) -> str:
    for line in PREREG_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"| **{row}**"):
            return [c.strip() for c in line.strip().strip("|").split("|")][1]
    raise SystemExit(f"row {row} not found in the prereg")


# ------------------------------------------------------------------------------------------------------------------ real-data loading
def load_gt_masks(w: World):
    import aux_label_store as A
    return {t: (A.read_label(STORE, lg, t) > 0) for t, lg in zip(w.token, w.log)}


def load_pred_masks(w: World, dirname: Path):
    files = [dirname / f"V_s0_f{k}.npz" for k in range(5)]
    for f in files:
        if not f.exists():
            raise SystemExit(f"missing head units: {f}  (the artifacts drive may have flipped D: -> E:; `subst D: E:\\`)")
    u = assemble_units(files, w.token)
    return {t: pred_mask(*u[t]) for t in w.token}, files


def stack_masks(w: World, masks: dict):
    return np.stack([masks[t] for t in w.token])


def load_tracked(w: World):
    files = sorted(glob.glob(str(TRACKED_DIR / "tracked_*of*.npz")))
    if not files:
        raise SystemExit(f"no tracker shards at {TRACKED_DIR}")
    out, shas = {}, {}
    for f in files:
        z = np.load(f, allow_pickle=False)
        for tk, c in zip(z["token"], z["corners_sim_ego"]):
            out[str(tk)] = c
        shas[Path(f).name] = sha256_file(f)
    missing = [t for t in w.token if t not in out]
    if missing:
        raise SystemExit(f"tracker shards miss {len(missing)} tokens")
    return out, shas


# ------------------------------------------------------------------------------------------------------------------ the full read
def cj(o):
    if isinstance(o, dict):
        return {str(k): cj(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [cj(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return cj(o.tolist())
    return o


def row(res: dict, k: int) -> dict:
    return {"point": float(res["point"][k]), "lo": float(res["lo"][k]), "hi": float(res["hi"][k]), "z": float(res["z"][k]), "z_lo": float(res["z_lo"][k]), "z_hi": float(res["z_hi"][k]),
            "n": float(res["n"][k]), "n_lo": float(res["n_lo"][k]), "n_hi": float(res["n_hi"][k])}


STAT_ROWS = ("d_pdms", "d_nc", "d_dac", "d_ep", "d_ttc", "d_comfort", "d_ddc", "lvl_pdms", "rec_ncdac", "rec_dac", "rec_pdms", "b_vetoed", "all_vetoed", "changed", "frac_vetoed")


def arm_block(w: World, veto: np.ndarray, picks: np.ndarray, log_ix, counts) -> dict:
    """Everything §8 asks for one arm: paired deltas of every sub-score, the Z recovery, the false-veto rate, the all-vetoed share, the changed-pick share."""
    d, o = deltas(w, picks), pick_outcomes(w, picks)
    ar = np.arange(w.n)
    b_vet = veto[ar, w.b_pick].astype(float)
    X = np.stack([d["pdms"], d["nc"], d["dac"], d["ep"], d["ttc"], d["comfort"], d["ddc"], o["pdms"], o["rec_ncdac"], o["rec_dac"], o["rec_pdms"], b_vet,
                  veto.all(axis=1).astype(float), (picks != w.b_pick).astype(float), veto.mean(axis=1)])
    res = strat_boot(X, w.is_z, log_ix, counts)
    blk = {name: row(res, k) for k, name in enumerate(STAT_ROWS)}
    z = w.is_z
    blk["z_recovery"] = {"ncdac_count": int(o["rec_ncdac"][z].sum()), "of": int(z.sum()), "dac_only_count": int(o["rec_dac"][z].sum()), "pdms_pos_count": int(o["rec_pdms"][z].sum())}
    blk["false_veto_N"] = {"b_pick_vetoed_count": int(b_vet[~z].sum()), "of": int((~z).sum()), "b_pick_vetoed_in_Z_count": int(b_vet[z].sum()), "of_Z": int(z.sum())}
    blk["n_vetoed_proposals"] = {"Z": int(veto[z].sum()), "N": int(veto[~z].sum()), "of_per_stratum": [int(veto[z].size), int(veto[~z].size)]}
    blk["all_vetoed_tokens"] = {"Z": int(veto[z].all(axis=1).sum()), "N": int(veto[~z].all(axis=1).sum())}
    blk["changed_picks"] = {"Z": int((picks != w.b_pick)[z].sum()), "N": int((picks != w.b_pick)[~z].sum())}
    return blk


def tactical_levels(w: World, log_ix, counts) -> dict:
    """§8 TACTICAL baselines per stratum: random (expected over the 64), B, best-of-64 (oracle max)."""
    ar = np.arange(w.n)
    X = np.stack([100 * w.pdms.mean(axis=1), 100 * w.pdms[ar, w.b_pick], 100 * w.pdms.max(axis=1)])
    res = strat_boot(X, w.is_z, log_ix, counts)
    return {n: row(res, k) for k, n in enumerate(("random", "B", "best_of_64"))}


def run_gate(w: World, b_banked, gt_masks):
    """Gates 1-3 of §9, which compute no arm Delta.  Returns (record, pred masks of V-PRED)."""
    rec = {}
    ok1, d1 = check_b_reproduces(w.b_pick, b_banked)
    rec["1_B_reproduces_pick"] = {"ok": ok1, **d1}
    ncdac = w.sub[np.arange(w.n), w.b_pick, 0] * w.sub[np.arange(w.n), w.b_pick, 1]
    pop_ok = bool((ncdac[w.is_z] == 0).all() and (ncdac[~w.is_z] > 0).all() and w.is_z.sum() == N_Z and (~w.is_z).sum() == N_Z)
    rec["0_population_identity"] = {"ok": pop_ok, "Z": int(w.is_z.sum()), "N_sample": int((~w.is_z).sum()), "Z_ncdac_all_zero": bool((ncdac[w.is_z] == 0).all()), "N_ncdac_all_pos": bool((ncdac[~w.is_z] > 0).all())}
    gt2 = np.stack([gt_mask_to_2m(gt_masks[t]) for t in w.token])
    ious = {}
    pm, files = load_pred_masks(w, UNIT_DIRS["V-PRED"])
    iou = iou_pooled(stack_masks(w, pm), gt2)
    ok2, d2 = check_iou_identity(iou)
    rec["2_vpred_mask_iou_identity"] = {"ok": ok2, **d2}
    banked = json.load(open(PRED_GEOMETRY))
    for arm, key in (("V-PRED-2k", "PROBE_2000_updates_V_s0"), ("V-PRED-400", "A3R_400_updates")):
        m2, _ = load_pred_masks(w, UNIT_DIRS[arm])
        v = iou_pooled(stack_masks(w, m2), gt2)
        ious[arm] = {"iou": v, "banked_PRED_GEOMETRY": banked[key]["iou_x0_40m_identity"], "abs_diff": abs(v - banked[key]["iou_x0_40m_identity"])}
    ious["V-PRED"] = {"iou": iou, "banked_PRED_GEOMETRY": banked["A2_10000_updates"]["iou_x0_40m_identity"]}
    rec["2b_head_quality_iou"] = ious
    # 3: zero control through the full pipeline, on both geometries
    zc = {}
    for nm, shape, cell, geom in (("pred_grid_40x32", (40, 32), 2.0, PRED_GEOM), ("gt_grid_160x128", (160, 128), 0.5, GT_GEOM)):
        masks = {t: np.ones(shape, bool) for t in w.token}
        mg = margins_from_masks(w, masks, cell, geom)
        v = vetoed(mg, MARGIN)
        picks = all_picks(w, v)
        d = deltas(w, picks)
        res = strat_boot(np.stack([d["pdms"]]), w.is_z, *_log_counts(w))
        ok3, d3 = check_zero_control(int(v.sum()), float(res["point"][0]), bool((picks == w.b_pick).all()))
        zc[nm] = {"ok": ok3, **d3}
    rec["3_zero_control"] = {"ok": all(v["ok"] for v in zc.values()), **zc}
    return rec, pm


_LC_CACHE = {}


def _log_counts(w: World):
    key = tuple(w.log)
    if key not in _LC_CACHE:
        ix, names = log_index(w.log)
        t0 = time.time()
        _LC_CACHE[key] = (ix, boot_counts(len(names)))
        BOOT_STATS["boot_counts_wall_s"] = time.time() - t0
    return _LC_CACHE[key]


_CITY = {}


def city_block(w: World, picks: np.ndarray, base_picks: np.ndarray):
    """Q13: descriptive per-city split.  City = the label-store shard header `map`."""
    import aux_label_store as A
    if not _CITY:
        heads, tot = {}, {}
        for f in sorted(glob.glob(str(STORE / "shards" / "*.aux1"))):
            h = A.read_shard(f, with_blobs=False)[0]
            tot[h["map"]] = tot.get(h["map"], 0) + len(h["entries"])
            heads[h["log"]] = h["map"]
        _CITY["heads"], _CITY["tot"] = heads, tot
    heads, tot = _CITY["heads"], _CITY["tot"]
    cities = np.array([heads[l] for l in w.log])
    d = deltas(w, picks)["pdms"]
    out = {}
    for c in sorted(set(cities)):
        m = cities == c
        z, n = m & w.is_z, m & ~w.is_z
        zc, T = int(z.sum()), int(tot[c])
        dz, dn = (float(d[z].mean()) if z.any() else None), (float(d[n].mean()) if n.any() else None)
        strat = (zc / T) * dz + ((T - zc) / T) * dn if (dz is not None and dn is not None) else None
        out[c] = {"n_Z": zc, "n_N": int(n.sum()), "navtest_tokens_in_city": T, "delta_Z": dz, "delta_N": dn, "delta_city_stratified": strat,
                  "z_recovered": int((pick_outcomes(w, picks)["rec_ncdac"][z]).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", default=str(HERE / "AUXVETO_READ.json"))
    a = ap.parse_args()
    t_all = time.time()
    prereg_sha = assert_prereg()
    for p, pre, nm in ((C3 / "tokens.json", TOKENS_SHA_PREFIX, "tokens.json"), (C3 / "table.npz", TABLE_SHA_PREFIX, "table.npz")):
        if not sha256_file(p).startswith(pre):
            raise SystemExit(f"REFUSED: {nm} sha256 {sha256_file(p)[:16]} != {pre}")
    store_man = json.load(open(STORE / "manifest.json"))
    if not store_man["store_sha256"].startswith(STORE_SHA_PREFIX):
        raise SystemExit(f"REFUSED: label store sha256 {store_man['store_sha256'][:16]} != {STORE_SHA_PREFIX}")
    w, b_banked = load_world()
    gt_masks = load_gt_masks(w)
    gate, vpred_masks = run_gate(w, b_banked, gt_masks)
    gate["all_pass"] = all(v["ok"] for k, v in gate.items() if isinstance(v, dict) and "ok" in v)
    if a.gate or not gate["all_pass"]:
        json.dump(cj({"stage": "gate", "prereg_sha256": prereg_sha, "gate": gate}), open(HERE / "GATE.json", "w"), indent=1)
        print(json.dumps(cj(gate), indent=1))
        if not gate["all_pass"]:
            print("INVALID: a pre-read gate failed; no arm was computed")
            sys.exit(2)
        return
    if not a.run:
        return

    # ---------------------------------------------------------------- the arms
    log_ix, counts = _log_counts(w)
    arms, margins = {}, {}
    gt_stack = {t: m for t, m in gt_masks.items()}
    margins["V-GT"] = margins_from_masks(w, gt_stack, 0.5, GT_GEOM)
    margins["V-GT-YFLIP"] = margins_from_masks(w, gt_stack, 0.5, GT_GEOM, transform=lambda m: m[:, ::-1])
    margins["V-PRED"] = margins_from_masks(w, vpred_masks, 2.0, PRED_GEOM)
    for arm in ("V-PRED-2k", "V-PRED-400"):
        mk, _ = load_pred_masks(w, UNIT_DIRS[arm])
        margins[arm] = margins_from_masks(w, mk, 2.0, PRED_GEOM)
    tracked, tracked_shas = load_tracked(w)
    mt = np.empty(w.sel.shape)
    for i, t in enumerate(w.token):
        mt[i] = min_margins_corners(sdf_from_mask(gt_masks[t], 0.5), GT_GEOM, tracked[t].astype(np.float64))
    margins["V-GT-TRACKED"] = mt
    vetoes = {k: vetoed(v, MARGIN) for k, v in margins.items()}
    vetoes["O-DAC"] = odac_veto(w)
    picks = {k: all_picks(w, v) for k, v in vetoes.items()}
    blocks = {k: arm_block(w, vetoes[k], picks[k], log_ix, counts) for k in picks}
    np.savez_compressed(HERE / "AUXVETO_PICKS.npz", token=np.array(w.token), is_z=w.is_z, b_pick=w.b_pick, veto_O_DAC=vetoes["O-DAC"],
                        **{f"pick__{k}": v for k, v in picks.items()}, **{f"margin__{k}": v.astype(np.float32) for k, v in margins.items()})

    # ---------------------------------------------------------------- §9 checks 4 and 5 (the data-dependent ones)
    rec_z = {k: float(blocks[k]["z_recovery"]["ncdac_count"]) for k in blocks}
    raster_arms = {k: rec_z[k] for k in ("V-PRED", "V-GT", "V-PRED-2k", "V-PRED-400", "V-GT-TRACKED")}
    ok4, d4 = check_odac_bounds(rec_z["O-DAC"], raster_arms)
    ok5, d5 = check_yflip_below(blocks["V-GT"]["d_pdms"]["point"], blocks["V-GT-YFLIP"]["d_pdms"]["point"])
    gate["4_odac_bounds_rasters"] = {"ok": ok4, **d4, "definition": "recovery = Z tokens whose pick has NC x DAC > 0 (QUESTIONS.md Q5)",
                                     "dac_only_variant_POSTHOC": {k: blocks[k]["z_recovery"]["dac_only_count"] for k in blocks}}
    gate["5_yflip_mutation_below_vgt"] = {"ok": ok5, **d5}
    gate["all_pass"] = all(v["ok"] for k, v in gate.items() if isinstance(v, dict) and "ok" in v)

    # ---------------------------------------------------------------- margin curve and head-quality curve
    curve = {}
    for arm in ("V-PRED", "V-GT"):
        curve[arm] = {}
        for m in (0.0, 0.25, 0.5, 1.0, 2.0):
            v = vetoed(margins[arm], m)
            pk = all_picks(w, v)
            bl = arm_block(w, v, pk, log_ix, counts)
            curve[arm][f"{m:g}"] = {"d_pdms": bl["d_pdms"], "z_recovery": bl["z_recovery"], "false_veto_N": bl["false_veto_N"], "all_vetoed_tokens": bl["all_vetoed_tokens"],
                                    "changed_picks": bl["changed_picks"], "n_vetoed_proposals": bl["n_vetoed_proposals"], "d_dac": bl["d_dac"], "d_nc": bl["d_nc"], "d_ep": bl["d_ep"]}
    odac_rec = blocks["O-DAC"]["z_recovery"]["ncdac_count"]
    tact = tactical_levels(w, log_ix, counts)
    cities = {k: city_block(w, picks[k], w.b_pick) for k in ("V-PRED", "V-GT", "V-PRED-2k", "V-PRED-400")}

    dec = decide(blocks["V-PRED"]["d_pdms"], blocks["V-GT"]["d_pdms"])
    alt_gt_reading = {"V-GT_point_positive": bool(blocks["V-GT"]["d_pdms"]["point"] > 0)}
    out = {"tool": "auxveto_read/1", "prereg": {"path": "eval/PREREG_AUXVETO.md", "sha256": prereg_sha, "landing_commit": "4f7e105"},
           "inputs": {"tokens_sha256": sha256_file(C3 / "tokens.json"), "table_sha256": sha256_file(C3 / "table.npz"), "label_store_sha256": store_man["store_sha256"],
                      "head_units": {arm: {"dir": str(d), "sha256": {f"V_s0_f{k}.npz": sha256_file(d / f"V_s0_f{k}.npz") for k in range(5)}} for arm, d in UNIT_DIRS.items()},
                      "tracker_shards_sha256": tracked_shas, "geodac_sha256": sha256_file(PK / "raw/2026-10-16-auxveto-feasibility/geodac.py")},
           "estimator": {"kind": "paired log-cluster bootstrap, stratified (529/12146 Z census + 11617/12146 N sample)", "n_logs": int(counts.shape[1]), "resamples": B_BOOT, "seed": SEED,
                         "weights": {"Z": W_Z, "N": W_N}, "ci": "95 % percentile",
                         "question_answered": "would another draw of LOGS say this?  BLIND to the head's TRAINING variance (one seed, H-ESTIM-SEED-1) and to the tracker; raster inference variance is zero (deterministic)",
                         "wall_s": BOOT_STATS, "note_wall": "boot_counts_wall_s = drawing the 10,000 x 123 resample-multiplicity matrix; strat_boot_wall_s = every bootstrap evaluation in this run (sum over strat_boot_calls calls)"},
           "tier": "NAVSIM v1.1 non-reactive table lookup (the programme's NAVSIM tier); base = u400 (SFT-4 arm B)",
           "validity": gate, "decision": dec, "decision_alt_reading_note": alt_gt_reading,
           "arms": blocks, "margin_curve": curve, "head_quality_curve": {arm: {"iou": gate["2b_head_quality_iou"][arm]["iou"], "d_pdms": blocks[arm]["d_pdms"], "z_recovery": blocks[arm]["z_recovery"]} for arm in ("V-PRED-400", "V-PRED-2k", "V-PRED")},
           "tactical_levels": tact, "odac_z_recovery": odac_rec, "per_city_descriptive": cities,
           "margin_distribution_descriptive": {arm: {"share_proposals_with_margin_below_-0.25": float((margins[arm] < -0.25).mean())} for arm in margins},
           "wall_s": time.time() - t_all}
    json.dump(cj(out), open(a.out, "w"), indent=1)
    print(json.dumps(cj({"validity_all_pass": gate["all_pass"], "decision": dec, "V-PRED": blocks["V-PRED"]["d_pdms"], "V-GT": blocks["V-GT"]["d_pdms"], "wall_s": out["wall_s"], "bootstrap": BOOT_STATS}), indent=1))


if __name__ == "__main__":
    main()
