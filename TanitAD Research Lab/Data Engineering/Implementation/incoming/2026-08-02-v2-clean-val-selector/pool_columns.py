"""Column semantics for `v2_pool_scored.parquet` — the registry a selector must pass through.

WHY THIS FILE EXISTS
--------------------
`Project Steering/BACKLOG.md` item **A3** is blocked on exactly one sentence:

    "`stopped`/`city`/`hw` are not clean 0/1 and `lk` is NOT a rate — establish column
     semantics before any selector."

The failure it records (commit `fe400f0`) was arithmetic on a misread column: `lk` was treated
as a rate and produced `needed_in_val = 48330` for a 600-clip split. That is not a bug in the
selector, it is a missing **contract** between the scorer and every consumer of its output.

This module is that contract, as code rather than prose. Every semantic below is derived from
the EMITTING code (`score_v2_pool.py:64-88`, which itself reuses `stack/scripts/refb_labels.py`
verbatim), and every one is backed by an identity that must hold on the real bytes — so a future
consumer cannot re-misread a column silently: it fails `validate_pool()` first.

THE SEMANTIC CLASSES (the distinction A3 was missing)
----------------------------------------------------
- ``BINARY``   — {0, 1} presence flag. Usable directly as a stratum.
- ``FRACTION`` — already a per-window rate in [0, 1]. Usable directly as a stratum MEAN.
  (`stopped`/`city`/`hw` are here: not "unclean 0/1" but honest fractions-of-window.)
- ``COUNT``    — a per-window FRAME COUNT out of `nlab`. ⛔ NEVER a stratum by itself;
  must be divided by `nlab` first. (`lk`/`tl`/`tr`/`ac`/`bs` and their `*2` twins are here.)
- ``SCALAR``   — a physical quantity with a unit (m/s, degrees, metres, frames).

Public API: ``COLUMNS``, ``rate_columns()``, ``to_rates(df)``, ``validate_pool(df)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# `score_v2_pool.py` builds labels with refb_labels.LABEL_HORIZON. Duplicated as a
# CONSTANT-WITH-A-CHECK (test_label_horizon_matches_refb_labels) rather than imported,
# so this module stays standalone-runnable per the intake contract.
LABEL_HORIZON = 20  # 2 s @ 10 Hz

Kind = Literal["binary", "fraction", "count", "scalar"]

# The 5 maneuver classes, in the bincount order refb_labels emits.
MANEUVER_CLASSES = ("lk", "tl", "tr", "ac", "bs")
MANEUVER_CLASSES_V2 = ("lk2", "tl2", "tr2", "ac2", "bs2")


@dataclass(frozen=True)
class ColSpec:
    """One column's meaning, its domain, and how to turn it into a stratum axis."""

    name: str
    kind: Kind
    unit: str
    meaning: str
    source: str                      # line in the emitting code
    lo: float | None = None
    hi: float | None = None
    denom: str | None = None         # for COUNT: the column to divide by
    family: str | None = None        # the metric family this axis speaks to

    @property
    def stratifiable_directly(self) -> bool:
        """A COUNT must be normalised by `denom` before it is comparable across clips."""
        return self.kind != "count"


def _c(*a, **k) -> ColSpec:
    return ColSpec(*a, **k)


COLUMNS: dict[str, ColSpec] = {
    s.name: s
    for s in [
        # ---- window bookkeeping -------------------------------------------------
        _c("T", "scalar", "frames", "poses in the camera-aligned first-20 s window",
           "score_v2_pool.py:65", lo=1),
        _c("nlab", "scalar", "frames", "labelled timesteps = T - LABEL_HORIZON; "
           "the DENOMINATOR of every maneuver count", "score_v2_pool.py:87", lo=0),
        _c("win_s", "scalar", "s", "egomotion span the window covers (~20.1 s)",
           "score_v2_pool.py:119", lo=0),
        # ---- LONGITUDINAL -------------------------------------------------------
        _c("mean_v", "scalar", "m/s", "mean speed over the window",
           "score_v2_pool.py:79", lo=0, family="longitudinal"),
        _c("stop_frac", "fraction", "1", "fraction of frames with v < 0.5 m/s (STRICT stop)",
           "score_v2_pool.py:79", lo=0, hi=1, family="longitudinal"),
        _c("stopped", "fraction", "1", "fraction of frames with v < 1.0 m/s (LOOSE stop) — "
           "a fraction, NOT a 0/1 flag; superset of stop_frac",
           "score_v2_pool.py:83", lo=0, hi=1, family="longitudinal"),
        _c("city", "fraction", "1", "fraction of frames with 1 <= v <= 12 m/s",
           "score_v2_pool.py:84", lo=0, hi=1, family="longitudinal"),
        _c("hw", "fraction", "1", "fraction of frames with v > 12 m/s",
           "score_v2_pool.py:85", lo=0, hi=1, family="longitudinal"),
        _c("dist_m", "scalar", "m", "distance travelled = sum(v) * 0.1",
           "score_v2_pool.py:80", lo=0, family="longitudinal"),
        # ---- LATERAL ------------------------------------------------------------
        _c("net_head", "scalar", "deg", "|wrapped yaw(T-1) - yaw(0)| — NET heading change",
           "score_v2_pool.py:81", lo=0, hi=180, family="lateral"),
        _c("cum_head", "scalar", "deg", "sum of |per-step wrapped dyaw| — CUMULATIVE turning; "
           "unbounded, and >= net_head by the triangle inequality",
           "score_v2_pool.py:82", lo=0, family="lateral"),
        # ---- TACTICAL (counts — the class A3 misread) ----------------------------
        *[_c(n, "count", "frames", f"frames labelled {n} by refb_labels.maneuver_labels "
             "(v1) — a COUNT out of nlab, not a rate", "score_v2_pool.py:86",
             lo=0, denom="nlab", family="tactical") for n in MANEUVER_CLASSES],
        *[_c(n, "count", "frames", f"frames labelled {n[:-1]} by maneuver_labels_v2 "
             "(curvature-gated) — a COUNT out of nlab", "score_v2_pool.py:87",
             lo=0, denom="nlab", family="tactical") for n in MANEUVER_CLASSES_V2],
        _c("has_turn", "binary", "1", "tl + tr > 0 anywhere in the window",
           "score_v2_pool.py:88", family="tactical"),
        _c("has_brake", "binary", "1", "bs > 0 anywhere in the window",
           "score_v2_pool.py:88", family="tactical"),
        _c("has_stop", "binary", "1", "any frame with v < 0.5 m/s",
           "score_v2_pool.py:88", family="tactical"),
        # ---- STRATEGIC ----------------------------------------------------------
        _c("junction", "binary", "1", "route_from_future_v21 returned 'tight_transient' at any "
           "5-frame stride — the v2.1 junction PROXY (not a map label)",
           "score_v2_pool.py:76-78", family="strategic"),
    ]
}

#: Rate name for each COUNT column (`lk` -> `lk_rate`).
RATE_SUFFIX = "_rate"


def rate_columns() -> dict[str, str]:
    """{count column -> derived rate column name}."""
    return {n: n + RATE_SUFFIX for n, s in COLUMNS.items() if s.kind == "count"}


def to_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with a `<count>_rate` column for every COUNT column.

    This is the ONLY sanctioned way to use a maneuver column as a stratum axis.
    Rows with `nlab == 0` yield NaN rather than inf (they are unlabelled windows).
    """
    out = df.copy()
    for col, rate in rate_columns().items():
        if col not in out.columns:
            continue
        denom = out[COLUMNS[col].denom].astype("float64")
        out[rate] = np.where(denom > 0, out[col].astype("float64") / denom, np.nan)
    return out


def stratifiable(df: pd.DataFrame) -> list[str]:
    """Columns present in `df` that may be used as a stratum axis AS THEY ARE."""
    return [c for c in df.columns
            if c in COLUMNS and COLUMNS[c].stratifiable_directly and COLUMNS[c].kind != "scalar"
            or c.endswith(RATE_SUFFIX)]


# --------------------------------------------------------------------------- #
# Validation — the identities that make a misread impossible to ship silently
# --------------------------------------------------------------------------- #

@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    n_bad: int = 0


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    n_rows: int = 0

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def raise_if_bad(self) -> "ValidationReport":
        if not self.ok:
            bad = "; ".join(f"{c.name}: {c.detail}" for c in self.checks if not c.ok)
            raise ValueError(f"pool column semantics violated ({bad})")
        return self

    def as_dict(self) -> dict:
        return {"n_rows": self.n_rows, "ok": self.ok,
                "checks": [{"name": c.name, "ok": c.ok, "n_bad": c.n_bad,
                            "detail": c.detail} for c in self.checks]}


def validate_pool(df: pd.DataFrame, *, atol: float = 1e-6) -> ValidationReport:
    """Check every semantic claim in `COLUMNS` against the actual bytes.

    The identities are the point: they are what turns "I read the scorer" into evidence.
    """
    rep = ValidationReport(n_rows=len(df))

    def add(name: str, mask_bad: "pd.Series | np.ndarray", detail_ok: str, detail_bad: str):
        n_bad = int(np.asarray(mask_bad).sum())
        rep.checks.append(Check(name, n_bad == 0,
                                detail_ok if n_bad == 0 else f"{detail_bad} (n={n_bad})", n_bad))

    # 1. domains
    for name, spec in COLUMNS.items():
        if name not in df.columns:
            continue
        col = df[name]
        if spec.kind == "binary":
            add(f"domain:{name}", ~col.isin([0, 1]), f"{name} ∈ {{0,1}}", f"{name} outside {{0,1}}")
        else:
            bad = pd.Series(False, index=df.index)
            if spec.lo is not None:
                bad |= col < spec.lo - atol
            if spec.hi is not None:
                bad |= col > spec.hi + atol
            add(f"domain:{name}", bad,
                f"{name} ∈ [{spec.lo}, {spec.hi}]", f"{name} out of [{spec.lo}, {spec.hi}]")

    # 2. maneuver counts sum to nlab (v1 AND v2) — the identity that proves they are COUNTS
    for tag, classes in (("v1", MANEUVER_CLASSES), ("v2", MANEUVER_CLASSES_V2)):
        if all(c in df.columns for c in classes) and "nlab" in df.columns:
            s = sum(df[c] for c in classes)
            add(f"identity:sum({tag})==nlab", s != df["nlab"],
                f"{'+'.join(classes)} == nlab (⇒ COUNTS, not rates)",
                f"{'+'.join(classes)} != nlab")

    # 3. nlab == T - LABEL_HORIZON
    if {"nlab", "T"} <= set(df.columns):
        add("identity:nlab==T-20", df["nlab"] != df["T"] - LABEL_HORIZON,
            f"nlab == T - {LABEL_HORIZON}", f"nlab != T - {LABEL_HORIZON}")

    # 4. the speed bands partition the window: stopped + city + hw == 1
    if {"stopped", "city", "hw"} <= set(df.columns):
        s = df["stopped"] + df["city"] + df["hw"]
        add("identity:stopped+city+hw==1", (s - 1.0).abs() > 1e-5,
            "speed bands partition the window (⇒ FRACTIONS, not flags)",
            "speed bands do not sum to 1")

    # 5. strict stop ⊆ loose stop
    if {"stop_frac", "stopped"} <= set(df.columns):
        add("identity:stop_frac<=stopped", df["stop_frac"] > df["stopped"] + atol,
            "stop_frac (v<0.5) <= stopped (v<1.0)", "stop_frac > stopped")

    # 6. cumulative turning >= net turning
    if {"cum_head", "net_head"} <= set(df.columns):
        add("identity:cum_head>=net_head", df["cum_head"] < df["net_head"] - 1e-3,
            "cum_head >= net_head (triangle inequality)", "cum_head < net_head")

    # 7. presence flags agree with the counts they summarise
    if {"has_turn", "tl", "tr"} <= set(df.columns):
        add("identity:has_turn==(tl+tr>0)", df["has_turn"] != ((df["tl"] + df["tr"]) > 0).astype(int),
            "has_turn == (tl+tr > 0)", "has_turn disagrees with tl+tr")
    if {"has_brake", "bs"} <= set(df.columns):
        add("identity:has_brake==(bs>0)", df["has_brake"] != (df["bs"] > 0).astype(int),
            "has_brake == (bs > 0)", "has_brake disagrees with bs")
    if {"has_stop", "stop_frac"} <= set(df.columns):
        add("identity:has_stop==(stop_frac>0)",
            df["has_stop"] != (df["stop_frac"] > 0).astype(int),
            "has_stop == (stop_frac > 0)", "has_stop disagrees with stop_frac")

    return rep
