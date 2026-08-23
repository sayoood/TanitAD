"""Clean v2-line validation split — feasibility, balancing, and a frozen manifest.

Backlog item **A3** (`Project Steering/BACKLOG.md`), C64 option B: build a validation split for
the v2 line from the 9,987 clips the `physicalai-v2bal` selection never took, so the split is
disjoint from that arm's training corpus *by construction*.

This module deliberately reports THREE things that the prior feasibility pass did not separate:

1. **COUNT feasibility** (`feasibility`) — can the remainder supply the per-cell counts a
   distribution-matched split needs?  ⚠️ This is what the 2026-07-29 pass measured, and it
   measured it on **one marginal axis**. Headroom falls with every axis added; the function
   therefore takes the axis set explicitly and reports the **binding cell**, never a total.
2. **DISTRIBUTIONAL balance** (`standardised_diffs`) — do the selected clips actually *look
   like* the training corpus on the four metric families?  Count feasibility does not imply
   this: the remainder is the residue of a manoeuvre-balanced selection and is skewed **inside**
   every cell, not only across cells.
3. **POOL DEPTH** (`census_fraction`) — what fraction of each cell's available clips the split
   consumes. A cell drawn at 90 %+ is a near-census: it leaves no room for a second disjoint
   split, and it is the leftovers of a quota selector rather than a sample of that manoeuvre.

Selection is by **greedy forward balancing** on standardised mean differences over the
four-family axis set, not by cell quotas — because the measurement above shows cell quotas do
not deliver balance.

⚠️ **Clip-granularity is the ceiling, and it is not our choice.** PhysicalAI-AV ships no
session/drive identifier (probed: `clip_index.parquet` 3 cols, `metadata/data_collection.parquet`
5 cols, `metadata/feature_presence.parquet` 36 feature flags) and its egomotion `timestamp` is
**clip-local microseconds**, not an absolute clock — so neither an id join nor an L2D-style
time-overlap test can establish drive-level disjointness. `assert_disjoint` says what it can
prove and the manifest records the granularity.

CLI:
    python clean_val_select.py --pool <v2_pool_scored.parquet> \
        --selection <r0_selection_v2.parquet> --n 600 --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from pool_columns import to_rates, validate_pool

# --------------------------------------------------------------------------- #
# The axis set — one block per metric family (`CLAUDE.md`, binding 2026-08-02).
# A split balanced only on ADE-relevant axes reproduces the defect that rule exists to stop.
# Every entry is a column `pool_columns` validates, or a rate derived from a COUNT column.
# --------------------------------------------------------------------------- #
FAMILY_AXES: dict[str, tuple[str, ...]] = {
    "longitudinal": ("mean_v", "stop_frac", "hw", "city"),
    "lateral": ("cum_head", "net_head"),
    "tactical": ("tl_rate", "tr_rate", "ac_rate", "bs_rate", "has_turn", "has_brake"),
    "strategic": ("junction",),
}
ALL_AXES: tuple[str, ...] = tuple(a for axes in FAMILY_AXES.values() for a in axes)

#: Cell axes for the COUNT-feasibility view (coarse, interpretable, one per family).
DEFAULT_CELL_AXES = ("junction", "has_turn", "speed")

#: A cell drawn above this fraction of its available clips is flagged as a near-census.
CENSUS_WARN = 0.60
#: Conventional balance bar from the matching literature: |d| < 0.1 balanced, < 0.2 tolerable.
BALANCE_BAR = 0.10


# --------------------------------------------------------------------------- #
# Balance
# --------------------------------------------------------------------------- #

def standardised_diffs(train: pd.DataFrame, val: pd.DataFrame,
                       axes: tuple[str, ...] = ALL_AXES) -> dict[str, float]:
    """Cohen's d per axis, standardised on the TRAIN sd (the reference distribution).

    Positive d = the split sits above train on that axis.
    """
    out: dict[str, float] = {}
    for a in axes:
        x, y = train[a].to_numpy(float), val[a].to_numpy(float)
        sd = x.std(ddof=1)
        out[a] = float((y.mean() - x.mean()) / sd) if sd > 0 else 0.0
    return out


def balance_summary(d: dict[str, float]) -> dict:
    v = np.abs(np.fromiter(d.values(), float))
    per_family = {f: float(np.abs([d[a] for a in axes if a in d]).max())
                  for f, axes in FAMILY_AXES.items() if any(a in d for a in axes)}
    return {"max_abs_d": float(v.max()), "median_abs_d": float(np.median(v)),
            "n_axes_over_bar": int((v > BALANCE_BAR).sum()), "bar": BALANCE_BAR,
            "worst_axis": max(d, key=lambda k: abs(d[k])), "per_family_max_abs_d": per_family}


# --------------------------------------------------------------------------- #
# Cells + COUNT feasibility
# --------------------------------------------------------------------------- #

def speed_edges(train: pd.DataFrame) -> np.ndarray:
    """Tercile edges of `mean_v` taken from TRAIN — the distribution being matched."""
    return train["mean_v"].quantile([1 / 3, 2 / 3]).to_numpy()


def cell_labels(df: pd.DataFrame, edges: np.ndarray,
                axes: tuple[str, ...] = DEFAULT_CELL_AXES) -> pd.Series:
    parts = []
    for a in axes:
        if a == "speed":
            parts.append("s" + pd.Series(np.digitize(df["mean_v"], edges), index=df.index).astype(str))
        else:
            parts.append(df[a].astype(int).astype(str))
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out.rename("cell")


@dataclass
class Feasibility:
    n_val: int
    cell_axes: tuple[str, ...]
    n_cells: int
    binding_cell: str
    binding_headroom: float
    n_cells_below_3x: int
    n_cells_infeasible: int
    n_max_at_3x: int
    n_max_at_2x: int
    per_cell: list[dict]

    def as_dict(self) -> dict:
        return asdict(self)


def feasibility(train: pd.DataFrame, remainder: pd.DataFrame, n_val: int,
                cell_axes: tuple[str, ...] = DEFAULT_CELL_AXES) -> Feasibility:
    """Per-cell count feasibility of a distribution-matched split of size `n_val`.

    ⚠️ Headroom is a property of the AXIS SET, not of the corpus. Reporting it without the
    axis set (as `6.77x` was) overstates feasibility by however many axes were left out.
    """
    edges = speed_edges(train)
    tc = cell_labels(train, edges, cell_axes).value_counts()
    rc = cell_labels(remainder, edges, cell_axes).value_counts()
    p = tc / len(train)
    need = (p * n_val).round().astype(int)
    avail = rc.reindex(need.index).fillna(0).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        headroom = np.where(need > 0, avail / need.replace(0, np.nan), np.inf)
    headroom = pd.Series(headroom, index=need.index)
    depth = (avail / p).replace([np.inf, -np.inf], np.nan)   # N supportable at 1x per cell
    rows = [{"cell": c, "train_n": int(tc[c]), "train_p": float(p[c]), "need": int(need[c]),
             "available": int(avail[c]), "headroom": float(headroom[c])} for c in need.index]
    rows.sort(key=lambda r: r["headroom"])
    return Feasibility(
        n_val=n_val, cell_axes=tuple(cell_axes), n_cells=len(need),
        binding_cell=str(headroom.idxmin()), binding_headroom=float(headroom.min()),
        n_cells_below_3x=int((headroom < 3).sum()), n_cells_infeasible=int((headroom < 1).sum()),
        n_max_at_3x=int(np.nanmin(depth) / 3), n_max_at_2x=int(np.nanmin(depth) / 2),
        per_cell=rows)


def census_fraction(val: pd.DataFrame, remainder: pd.DataFrame, edges: np.ndarray,
                    cell_axes: tuple[str, ...] = DEFAULT_CELL_AXES) -> list[dict]:
    """How much of each cell's available pool the split consumed (the pool-depth cost)."""
    vc = cell_labels(val, edges, cell_axes).value_counts()
    rc = cell_labels(remainder, edges, cell_axes).value_counts()
    rows = [{"cell": str(c), "taken": int(vc[c]), "available": int(rc.get(c, 0)),
             "census_fraction": float(vc[c] / rc[c]) if rc.get(c, 0) else float("nan")}
            for c in vc.index]
    rows.sort(key=lambda r: -(r["census_fraction"] if r["census_fraction"] == r["census_fraction"] else 0))
    return rows


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def balanced_select(train: pd.DataFrame, remainder: pd.DataFrame, n_val: int,
                    axes: tuple[str, ...] = ALL_AXES, seed: int = 0,
                    cell_quota: bool = False,
                    cell_axes: tuple[str, ...] = DEFAULT_CELL_AXES) -> pd.DataFrame:
    """Greedy forward selection minimising the summed squared standardised difference.

    At each step the clip whose addition moves the running mean closest to the train mean on
    ALL axes is taken. O(n_val * |remainder| * |axes|) and fully deterministic given `seed`
    (the seed only breaks the first tie, where every candidate is equally good).

    `cell_quota=True` additionally caps each joint cell at its train proportion — the HYBRID
    arm. It buys the joint-cell match that pure mean-balancing gives up, and pays for it in
    pool depth (the binding cell is drawn to near-census). Measure both; neither dominates.
    """
    if n_val > len(remainder):
        raise ValueError(f"cannot draw {n_val} from a remainder of {len(remainder)}")
    X = remainder[list(axes)].to_numpy(float)
    mu = train[list(axes)].to_numpy(float).mean(axis=0)
    sd = train[list(axes)].to_numpy(float).std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    Xn, mun = X / sd, mu / sd                                  # work in sd units

    cell_of = quota = used = None
    if cell_quota:
        edges = speed_edges(train)
        tc = cell_labels(train, edges, cell_axes).value_counts()
        cells = cell_labels(remainder, edges, cell_axes)
        codes, uniq = pd.factorize(cells)
        cell_of = codes
        quota = np.array([int(round(tc.get(c, 0) / len(train) * n_val)) for c in uniq])
        used = np.zeros(len(uniq), int)

    rng = np.random.default_rng(seed)
    taken = np.zeros(len(X), bool)
    first = int(rng.integers(len(X)))
    taken[first] = True
    S = Xn[first].copy()
    if cell_quota:
        used[cell_of[first]] += 1

    for k in range(1, n_val):
        cand = (S + Xn) / (k + 1) - mun                        # [N, A] post-add difference
        obj = (cand ** 2).sum(axis=1)
        obj[taken] = np.inf
        if cell_quota:
            full = used >= quota
            if full.all():                                     # rounding left the quota short
                full[:] = False
            obj[full[cell_of]] = np.inf
        j = int(obj.argmin())
        taken[j] = True
        S += Xn[j]
        if cell_quota:
            used[cell_of[j]] += 1
    return remainder.loc[taken]


def stratified_select(train: pd.DataFrame, remainder: pd.DataFrame, n_val: int,
                      cell_axes: tuple[str, ...] = DEFAULT_CELL_AXES,
                      seed: int = 0) -> pd.DataFrame:
    """Cell-quota selection — the classic 'match the strata' recipe, kept as the CONTROL arm.

    Its balance is measured against `balanced_select`; it is not the recommended path.
    """
    edges = speed_edges(train)
    tc = cell_labels(train, edges, cell_axes).value_counts()
    need = (tc / len(train) * n_val).round().astype(int)
    rem = remainder.assign(_cell=cell_labels(remainder, edges, cell_axes))
    rng = np.random.default_rng(seed)
    picks = []
    for c, k in need.items():
        pool_c = rem.index[rem["_cell"] == c].to_numpy()
        if k <= 0 or len(pool_c) == 0:
            continue
        take = min(int(k), len(pool_c))
        picks.append(rng.choice(pool_c, take, replace=False))
    idx = np.concatenate(picks) if picks else np.array([], dtype=remainder.index.dtype)
    return remainder.loc[idx]


# --------------------------------------------------------------------------- #
# Disjointness + manifest
# --------------------------------------------------------------------------- #

def assert_disjoint(val_ids, train_ids) -> dict:
    """C64's own rule: print the intersection as an artifact, EVEN WHEN IT IS EMPTY."""
    v, t = set(map(str, val_ids)), set(map(str, train_ids))
    inter = sorted(v & t)
    return {"granularity": "clip_id",
            "n_val": len(v), "n_train": len(t), "n_intersection": len(inter),
            "intersection": inter[:50], "disjoint": not inter,
            "drive_level_provable": False,
            "drive_level_reason": (
                "PhysicalAI-AV ships no session/drive id (clip_index 3 cols; "
                "data_collection {country,month,hour_of_day,platform_class,radar_config}; "
                "feature_presence = 36 presence flags) and egomotion `timestamp` is clip-local "
                "microseconds, not an absolute clock — so neither an id join nor an "
                "L2D-style time-overlap test can be run. Clip-level is the provable ceiling.")}


def manifest_sha256(clip_ids) -> str:
    """Hash of the sorted clip list — the trainer preflight refuses on mismatch."""
    h = hashlib.sha256()
    for c in sorted(map(str, clip_ids)):
        h.update(c.encode()); h.update(b"\n")
    return h.hexdigest()


def build_manifest(train: pd.DataFrame, remainder: pd.DataFrame, val: pd.DataFrame,
                   *, n_val: int, seed: int, cell_axes=DEFAULT_CELL_AXES) -> dict:
    edges = speed_edges(train)
    d = standardised_diffs(train, val)
    return {
        "name": f"physicalai-v2clean-val-{len(val)}",
        "n_val": len(val), "requested_n": n_val, "seed": seed,
        "source": {"pool_train_n": len(train), "remainder_n": len(remainder)},
        "clip_ids": sorted(map(str, val["clip_id"])),
        "sha256": manifest_sha256(val["clip_id"]),
        "disjointness": assert_disjoint(val["clip_id"], train["clip_id"]),
        "balance": {"per_axis_d": d, **balance_summary(d),
                    "family_axes": {k: list(v) for k, v in FAMILY_AXES.items()}},
        "pool_depth": {"cell_axes": list(cell_axes), "speed_edges": edges.tolist(),
                       "per_cell": census_fraction(val, remainder, edges, cell_axes),
                       "census_warn": CENSUS_WARN},
    }


# --------------------------------------------------------------------------- #

def load(pool_path: str, selection_path: str,
         exclude_paths: "list[str] | None" = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pool + selection, validate semantics, derive rates, split train/remainder.

    De-duplicates `clip_id` (the pool carries one clip twice, under two chunk numbers —
    the reason the corpus has been quoted as both 18,987 and 18,988).

    ⭐ `exclude_paths` — parquet files whose `clip_id`s are dropped from the REMAINDER before
    any selection. **Pass the v1 parity selection here.** Disjointness from v2corpus's training
    set is not enough: 62 of a first 600-clip draw sat in v1's TRAIN split, so scoring v1 on
    that val would have scored it partly on its own training data — C64 in mirror image.
    """
    pool = pd.read_parquet(pool_path)
    validate_pool(pool).raise_if_bad()
    pool = to_rates(pool).drop_duplicates("clip_id").reset_index(drop=True)
    sel = pd.read_parquet(selection_path)
    sid = set(sel["clip_id"].astype(str))
    in_sel = pool["clip_id"].astype(str).isin(sid)
    train = pool[in_sel].reset_index(drop=True)
    remainder = pool[~in_sel].reset_index(drop=True)
    for p in exclude_paths or []:
        ex = set(pd.read_parquet(p)["clip_id"].astype(str))
        remainder = remainder[~remainder["clip_id"].astype(str).isin(ex)].reset_index(drop=True)
    return train, remainder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="parquet(s) whose clip_ids are removed from the remainder "
                         "(pass the v1 parity selection: r0/phase0_selection.parquet)")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    train, remainder = load(a.pool, a.selection, a.exclude)
    print(f"[load] train={len(train)} remainder={len(remainder)}")

    feas = feasibility(train, remainder, a.n)
    print(f"[feasibility] axes={feas.cell_axes} binding={feas.binding_cell} "
          f"headroom={feas.binding_headroom:.2f} n_max@3x={feas.n_max_at_3x}")

    val = balanced_select(train, remainder, a.n, seed=a.seed, cell_quota=True)
    man = build_manifest(train, remainder, val, n_val=a.n, seed=a.seed)
    man["feasibility"] = feas.as_dict()
    print(f"[balance] max|d|={man['balance']['max_abs_d']:.4f} "
          f"over-bar={man['balance']['n_axes_over_bar']}/{len(ALL_AXES)}")

    with open(f"{a.out}/v2_clean_val_manifest.json", "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    print(f"[write] {a.out}/v2_clean_val_manifest.json sha256={man['sha256'][:16]}")


if __name__ == "__main__":
    main()
