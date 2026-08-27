#!/usr/bin/env python3
"""RELEASE GATE — the fixed criteria every TanitAD model release must pass.

    python tools/release_gate.py --model <registry-key> --artifacts <dir> \
        [--baseline <key>] [--json out.json]

ONE COMMAND -> ONE VERDICT. The verdict is a CONJUNCTION over named criteria,
never a pooled score: a single number would hide exactly the trade-off the four
families exist to expose (PI, 2026-08-02, binding).

WHY THIS EXISTS
---------------
`criteria_check.py` answers *"is this artifact COMPLETE?"*. That is necessary and
not sufficient. A release can be perfectly complete and still be a disaster:

    MEASURED 2026-08-18, T1, ff_stageA_{cl,ha}.json, 6844 windows / 40 episodes:
        stage-a-repaired  closed loop   ade_dense_m = 9.3697
        hold-action control             ade_dense_m = 0.4246
    The control beat the model by 22x. Both artifacts are four-families complete.

A gate that never compares against the control cannot see that. So this tool adds
the criteria completeness cannot express — control floors, regression, tier,
estimator VALUE (not merely its name), leak declarations, parity, and grid
identity — and it CONSUMES `criteria_check` for the completeness half rather than
reimplementing it.

THE FOUR STATES, AND WHY THERE ARE FOUR
---------------------------------------
    PASS         the criterion was evaluated and satisfied
    FAIL         the criterion was evaluated and violated              -> exit 1
    CANNOT-RULE  the criterion could NOT be evaluated at these settings -> exit 2
    N/A          the criterion does not apply here, WITH ITS REASON

*** CANNOT-RULE is the state this programme learned the hard way. ***
The O6 rank gate could never rule -- it compared a spectrum of n=24 against a
ceiling of 1024, so no checkpoint could ever have moved it -- and it returned a
perpetual INCONCLUSIVE that read like caution instead of like a broken
instrument. GATE_PROTOCOL.md §0.7 states the general form: *"a gate secondary
whose value is fixed by a LABEL or HARNESS defect carries zero information about
the model"*. So every criterion here answers a PREFLIGHT question BEFORE it is
evaluated -- *could this criterion have come out differently if the model were
better?* -- and a criterion that cannot rule says so LOUDLY and does not pass.

THE FAIL / CANNOT-RULE BOUNDARY (stated once, applied everywhere)
-----------------------------------------------------------------
    A criterion FAILS when the release COULD have supplied the evidence and
    did not.  A DECLARATION (tier, estimator, parity stamp, inference inputs,
    goal source) is free to emit; its absence is a release defect.

    A criterion CANNOT-RULE when the evidence does not exist to be supplied --
    no control arm was banked, the baseline is not in scope, the episode count
    is below the estimator's floor. The gate then has nothing to rule ON, and
    saying "PASS" there would be a fabrication.

Exit codes: 0 all PASS/N-A · 1 any FAIL · 2 no FAIL but >=1 CANNOT-RULE ·
3 the gate itself could not run (registry unreadable, bad path).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import criteria_check as cc  # noqa: E402  -- CONSUMED, never reimplemented

SPEC_DOC = "products/P7-TanitEval/RELEASE_GATE.md"

PASS, FAIL, CANNOT_RULE, NA = "PASS", "FAIL", "CANNOT-RULE", "N/A"
_ORDER = {FAIL: 0, CANNOT_RULE: 1, NA: 2, PASS: 3}

# Parity is sacred (CLAUDE.md): anything that re-selects episodes breaks
# cross-arm comparability and must be refused.
CANON_CORPUS = "physicalai-train-e438721ae894"
CANON_SKIP_HASH = "f09e44db"

# The episode-cluster bootstrap resamples EPISODES, not windows. With a handful
# of clusters the interval is not decision-grade however many windows sit inside
# them -- 6844 windows over 3 episodes is 3 independent draws.
EPISODE_CLUSTER_FLOOR = 8

_CONTROL_TOKENS = ("ha", "hold", "hold_action", "holdaction", "holdv0", "hold_v0",
                   "cv", "constvel", "const_vel", "constant_velocity", "v0")

_GOOD_ESTIMATORS = ("episode_cluster_bootstrap", "paired_episode_cluster_bootstrap")

# WHICH CRITERIA BLOCK A RELEASE IS THE REGISTRY'S CALL, NOT THIS FILE'S.
# `CRITERIA_REGISTRY.json:release_gate.blocking` is the single source; this map only
# says which RG-row answers each registry entry, so the gate and the criteria census
# read ONE list. A blocking rule that lived only here would drift from the census the
# first time either was edited -- the exact decay the registry exists to stop.
_REGISTRY_TO_RG = {
    "all four families PRESENT or REFUSED-with-reason-and-n": ("RG-02",),
    "hyg.tier_stamp": ("RG-03",),
    "hyg.estimator_named": ("RG-05",),
    "hyg.n_reported": ("RG-14",),
    "hyg.no_forbidden_estimator": ("RG-06",),
    "hyg.parity": ("RG-12",),
    "ctrl.floor_comparison": ("RG-10",),
    "leak_guards.vision_only_inference": ("RG-07",),
    "leak_guards.goal_situation_disjoint": ("RG-08",),
}

# Structural criteria the registry's list presupposes rather than names. Each blocks,
# and each carries the sentence that earns it.
_ALWAYS_BLOCKING = {
    "RG-01": "a gate that swept nothing must never read as a pass",
    "RG-04": "registry release_gate.blocking_note: '⛔ A T1 artifact is REQUIRED: T0 is "
             "a world-model diagnostic and a release may never be gated on it alone.'",
    "RG-09": "the registry's route_head_echo is optional as COMPLETENESS (not every "
             "artifact has a route score); a near-1.0 route score WITHOUT its echo "
             "test is a leak, and flagship v1 scored exactly 1.0000 by echo",
    "RG-13": "a delta across different windows is not a delta",
}


# ------------------------------------------------------------ metric specs ---
@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str
    paths: tuple
    lower_better: bool
    interval_key: str | None = None


# Per-family headline metrics. ⛔ NEVER pooled into one score -- the families are
# reported side by side precisely so a longitudinal collapse cannot be averaged
# away by a good lateral number.
FAMILY_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("speed_mae_mps", "LONGITUDINAL",
               ("four_families.longitudinal.speed_mae_mps", "headline.speed_mae_mps"),
               True, "LON_speed_mae_mps"),
    MetricSpec("along_mae_m", "LONGITUDINAL",
               ("four_families.longitudinal.along_mae_m", "headline.long_abs_2s_m"),
               True, "LON_along_mae_m"),
    MetricSpec("cross_mae_m", "LATERAL",
               ("four_families.lateral.cross_mae_m", "headline.lat_abs_2s_m"),
               True, "LAT_cross_mae_m"),
    MetricSpec("heading_mae_deg", "LATERAL",
               ("four_families.lateral.heading_mae_deg", "headline.heading_mae_2s_deg"),
               True, "LAT_heading_mae_deg"),
    MetricSpec("yaw_rate_mae_degps", "LATERAL",
               ("four_families.lateral.yaw_rate_mae_degps", "headline.yaw_rate_mae_dps"),
               True, None),
    MetricSpec("curvature_mae_1pm", "LATERAL",
               ("four_families.lateral.curvature_mae_1pm", "headline.curv_mae_1_per_m"),
               True, None),
    MetricSpec("lat_decision_kappa", "TACTICAL",
               ("four_families.tactical.lateral_decision.kappa",), False,
               "TAC_lat_decision_correct"),
    MetricSpec("lon_decision_kappa", "TACTICAL",
               ("four_families.tactical.longitudinal_decision.kappa",), False,
               "TAC_lon_decision_correct"),
    MetricSpec("goal_point_error_m", "TACTICAL",
               ("four_families.tactical.goal_setting.goal_point_error_m",), True,
               "TAC_goal_point_error_m"),
    MetricSpec("strategic_decision", "STRATEGIC",
               ("strategic.decision_accuracy", "strategic.accuracy",
                "four_families.strategic.decision_accuracy"), False, None),
    MetricSpec("strategic_route", "STRATEGIC",
               ("strategic.route_quality", "strategic.goal_setting",
                "headline.route_acc", "four_families.strategic.route_quality"), False, None),
)

# ADE/FDE ride ALONGSIDE the families. ⛔ Never present a horizon sweep of ADE as
# "the result" -- it is one row of four families (PI, binding).
COMPANION_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("ade_dense_m", "COMPANION",
               ("intervals.metrics.ade_dense_m", "headline.ade_0_2s", "ade_0_2s"),
               True, "ade_dense_m"),
    MetricSpec("fde_last_m", "COMPANION",
               ("intervals.metrics.fde_last_m",), True, "fde_last_m"),
)

FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")


def _num(val):
    """Unwrap the two idioms a scalar arrives in: bare, or {'mean': x}."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    if isinstance(val, dict):
        for k in ("mean", "value", "point", "full_set"):
            if isinstance(val.get(k), (int, float)) and not isinstance(val.get(k), bool):
                return float(val[k])
    return None


def read_metric(doc: dict, spec: MetricSpec):
    """Return (value, interval_or_None). None value = not emitted in this artifact."""
    val = None
    for p in spec.paths:
        found, v = cc._dig(doc, p)
        if found:
            val = _num(v)
            if val is not None:
                break
    iv = None
    if spec.interval_key:
        found, v = cc._dig(doc, f"intervals.metrics.{spec.interval_key}")
        if found and isinstance(v, dict):
            iv = v
            if val is None:
                val = _num(v)
    return val, iv


# -------------------------------------------------------------- artifacts ---
PRIMARY, CONTROL, OTHER_TIER, NOT_THIS_MODEL = \
    "PRIMARY", "CONTROL", "OTHER-TIER", "NOT-THIS-MODEL"
OUT_OF_SCOPE, UNKNOWN, UNREADABLE = "OUT-OF-SCOPE", "UNKNOWN-SCOPE", "UNREADABLE"


@dataclass
class Artifact:
    path: str
    doc: dict | None
    scope: str = ""
    scope_why: str = ""
    tier: str | None = None
    tier_raw: str | None = None
    arm: str = ""
    arm_key: str = ""
    role: str = ""
    is_control: bool = False
    n_windows = None
    n_episodes = None
    grid: str = ""
    check: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def _arm_of(doc: dict, path: str) -> tuple[str, str]:
    for k in ("arm", "model", "run", "arm_key", "release"):
        v = doc.get(k)
        if isinstance(v, str) and v:
            return v, str(doc.get("arm_key") or "")
    return os.path.splitext(os.path.basename(path))[0], str(doc.get("arm_key") or "")


def _is_control(doc: dict, arm: str, arm_key: str, fname: str) -> bool:
    if doc.get("is_control") is True or doc.get("control") is True:
        return True
    toks = set()
    for blob in (arm_key, arm, fname):
        for piece in str(blob).replace(":", " ").replace("-", " ").replace("_", " ") \
                .replace(".", " ").split():
            toks.add(piece.lower())
    return bool(toks & set(_CONTROL_TOKENS))


def _grid_of(doc: dict) -> str:
    gf = doc.get("grid_fingerprint")
    if isinstance(gf, dict):
        bits = [str(gf.get(k)) for k in
                ("eid_sha1", "gt_sha1", "n_windows", "horizon_steps", "dt_s", "n_episodes")]
        return "|".join(bits)
    return "|".join(str(doc.get(k)) for k in
                    ("n_windows", "horizon_steps", "dt_s", "n_episodes"))


def _matches_model(art: Artifact, key: str) -> bool:
    hay = f"{art.arm} {art.arm_key} {art.name}".lower().replace("-", "").replace("_", "")
    return key.lower().replace("-", "").replace("_", "") in hay


def load_artifacts(target: str, registry: dict) -> list[Artifact]:
    if os.path.isdir(target):
        paths = sorted(glob.glob(os.path.join(target, "**", "*.json"), recursive=True))
    else:
        paths = sorted(glob.glob(target))
    out = []
    for p in paths:
        doc = cc._read_json(Path(p))
        if not isinstance(doc, dict):
            out.append(Artifact(path=p, doc=None, scope=UNREADABLE,
                                scope_why="could not be read or parsed after retries",
                                role=UNREADABLE))
            continue
        a = Artifact(path=p, doc=doc)
        scope, why = cc.scope_of(doc, registry)
        a.scope_why = why
        if scope == cc.IN_SCOPE:
            a.scope = cc.IN_SCOPE
        elif scope == cc.OUT_OF_SCOPE:
            a.scope, a.role = OUT_OF_SCOPE, OUT_OF_SCOPE
        else:
            a.scope, a.role = UNKNOWN, UNKNOWN
        a.arm, a.arm_key = _arm_of(doc, p)
        a.is_control = _is_control(doc, a.arm, a.arm_key, a.name)
        a.tier, a.tier_raw = cc.resolve_tier(doc, registry)
        a.n_windows = doc.get("n_windows")
        a.n_episodes = doc.get("n_episodes")
        a.grid = _grid_of(doc)
        out.append(a)
    return out


def assign_roles(arts: list[Artifact], model: str, primary_arm_key: str,
                 registry: dict) -> None:
    for a in arts:
        if a.scope != cc.IN_SCOPE:
            continue
        if not _matches_model(a, model):
            a.role = NOT_THIS_MODEL
            continue
        # ⚠️ The primary arm is chosen by ARM, NOT by tier. Choosing it by tier would
        # let a T0-only release lose its primary entirely, and RG-04 -- the criterion
        # whose whole job is to say "this release is T0 and cannot be cleared" --
        # would never get to speak; the operator would see a scope error instead of
        # the real defect. A criterion must be able to report its own failure.
        if a.is_control:
            a.role = CONTROL
        elif not primary_arm_key or a.arm_key == primary_arm_key \
                or f"_{primary_arm_key}." in a.name:
            a.role = PRIMARY
        else:
            a.role = OTHER_TIER
        if a.role in (PRIMARY, CONTROL):
            a.check = cc.check_artifact(a.doc, registry)


# ----------------------------------------------------------------- verdict ---
@dataclass
class Row:
    id: str
    label: str
    state: str
    detail: str
    can_rule: bool = True
    why_not: str = ""
    family: str = "-"
    blocking: bool = True
    blocking_why: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "state": self.state,
                "detail": self.detail, "can_rule": self.can_rule,
                "why_cannot_rule": self.why_not, "family": self.family,
                "blocking": self.blocking, "blocking_source": self.blocking_why}


def blocking_for(cid: str, registry: dict, cfg: "Cfg") -> tuple[bool, str]:
    """Is this criterion release-BLOCKING or ADVISORY? The registry decides.

    ⚠️ The registry records `pi_decision_open`: whether a FAIL blocks or advises is
    the PI's call. Until it is made, the gate follows the recommendation ON RECORD
    (blocking on the listed set, advisory on regression) and PRINTS which it used,
    so nobody has to guess which policy produced a verdict.
    """
    root = cid.split(".")[0]
    spec = registry.get("release_gate", {})
    for entry in spec.get("blocking", []):
        if root in _REGISTRY_TO_RG.get(entry, ()):
            return True, f"registry release_gate.blocking: {entry!r}"
    if root in _ALWAYS_BLOCKING:
        return True, _ALWAYS_BLOCKING[root]
    if root == "RG-11":
        if cfg.regression_blocking:
            return True, "--regression-blocking was passed, overriding the registry"
        return False, ("registry release_gate.advisory: 'regression vs the previous "
                       "release (per family)' — advisory until two clean releases "
                       "exist to compare, or a regression gate with no trustworthy "
                       "baseline manufactures false failures")
    return True, "not named by the registry — blocking by default, the safe direction"


def _cannot(cid, label, why, family="-") -> Row:
    return Row(cid, label, CANNOT_RULE,
               f"⛔ THIS CRITERION CANNOT RULE HERE: {why}", False, why, family)


@dataclass
class Cfg:
    model: str
    baseline: str | None
    primary_arm_key: str
    control_margin: float
    regression_tol: float
    accept_regression: dict
    parity_key: str
    parity_skip_hash: str
    echo_threshold: float
    episode_floor: int
    regression_blocking: bool = False


# ------------------------------------------------------------- criteria -----
def rg01_scope(arts, cfg) -> Row:
    """A gate that swept nothing must never read as a pass."""
    prim = [a for a in arts if a.role == PRIMARY]
    mine = [a for a in arts if a.role in (PRIMARY, CONTROL, OTHER_TIER)]
    if not mine:
        in_scope = [a.name for a in arts if a.scope == cc.IN_SCOPE]
        return Row("RG-01", "scope: the sweep found artifacts for this model", FAIL,
                   f"NO artifact matches --model {cfg.model!r}. "
                   f"{len(in_scope)} in-scope driving artifacts were read and none "
                   f"belong to it: {in_scope[:8]}. A gate reports the scope it swept, "
                   f"not the model you meant.")
    if not prim:
        return Row("RG-01", "scope: the sweep found artifacts for this model", FAIL,
                   f"{len(mine)} artifact(s) match {cfg.model!r} but NONE is a primary "
                   f"driving arm (roles: {sorted({a.role for a in mine})}). "
                   f"Name it with --primary-arm-key.")
    if len(prim) > 1:
        return Row("RG-01", "scope: the sweep found artifacts for this model", FAIL,
                   f"AMBIGUOUS primary arm — {[a.name for a in prim]}. A release has one "
                   f"headline arm; disambiguate with --primary-arm-key.")
    return Row("RG-01", "scope: the sweep found artifacts for this model", PASS,
               f"primary={prim[0].name} (arm={prim[0].arm!r}, tier={prim[0].tier}), "
               f"{sum(a.role == CONTROL for a in arts)} control, "
               f"{sum(a.role == OTHER_TIER for a in arts)} other-tier")


def rg02_completeness(arts, cfg) -> Row:
    """Four families PRESENT or REFUSED-WITH-A-REASON. Delegated to criteria_check."""
    prim = [a for a in arts if a.role == PRIMARY]
    if not prim:
        return _cannot("RG-02", "completeness: four families present or refused",
                       "no primary arm was resolved (see RG-01)")
    bad = []
    for a in prim:
        for v in a.check.get("violations", []):
            fam = next((f for f, rows in a.check["families"].items()
                        if any(r["id"] == v["id"] for r in rows)), "HYGIENE/LEAK")
            bad.append(f"{a.name}:{fam}/{v['id']} ({v['detail'][:70]})")
    work = sum(a.check.get("n_work_items", 0) for a in prim)
    if bad:
        return Row("RG-02", "completeness: four families present or refused", FAIL,
                   f"{len(bad)} required criterion/criteria SILENTLY ABSENT — neither "
                   f"emitted nor refused with a reason: {bad}")
    return Row("RG-02", "completeness: four families present or refused", PASS,
               f"every required criterion is PRESENT or REFUSED-with-a-reason; "
               f"{work} open work item(s) (refused/partial) carried forward")


def rg03_tier_stamp(arts, cfg) -> Row:
    mine = [a for a in arts if a.role in (PRIMARY, CONTROL, OTHER_TIER)]
    if not mine:
        return _cannot("RG-03", "tier: every artifact carries a recognised tier stamp",
                       "no artifact matched this model (see RG-01)")
    un = [f"{a.name} (raw={a.tier_raw!r})" for a in mine if not a.tier]
    if un:
        return Row("RG-03", "tier: every artifact carries a recognised tier stamp", FAIL,
                   f"UNSTAMPED or unrecognised tier on {len(un)} artifact(s): {un}. "
                   f"A number with no tier is not quotable.")
    return Row("RG-03", "tier: every artifact carries a recognised tier stamp", PASS,
               ", ".join(f"{a.name}={a.tier}" for a in mine))


def rg04_t1_present(arts, cfg, registry) -> Row:
    """⛔ T0 is a WM diagnostic. A release may NEVER be gated on T0 alone."""
    mine = [a for a in arts if a.role in (PRIMARY, CONTROL, OTHER_TIER)]
    if not mine:
        return _cannot("RG-04", "tier: T1 present; the release is not gated on T0 alone",
                       "no artifact matched this model (see RG-01)")
    vals = registry.get("tiers", {}).get("values", {})
    t1 = [a for a in mine if a.tier == "T1"]
    drv = [a for a in mine if vals.get(a.tier or "", {}).get("is_driving_performance")]
    tiers = sorted({a.tier or "UNSTAMPED" for a in mine})
    if not t1:
        return Row("RG-04", "tier: T1 present; the release is not gated on T0 alone", FAIL,
                   f"NO T1 (action-closed) artifact. Tiers found: {tiers}. "
                   f"T0 is a teacher-forced world-model diagnostic and is NEVER driving "
                   f"performance — a release cannot be cleared on it.")
    prim_t1 = [a for a in arts if a.role == PRIMARY and a.tier == "T1"]
    if not prim_t1:
        return Row("RG-04", "tier: T1 present; the release is not gated on T0 alone", FAIL,
                   f"T1 artifacts exist ({[a.name for a in t1]}) but the PRIMARY arm is "
                   f"not one of them — the headline would be a non-T1 number.")
    return Row("RG-04", "tier: T1 present; the release is not gated on T0 alone", PASS,
               f"primary is T1; {len(drv)} driving-performance artifact(s), tiers {tiers}")


def rg05_estimator(arts, cfg) -> Row:
    """Named is not enough — the VALUE must be the episode-cluster bootstrap."""
    prim = [a for a in arts if a.role == PRIMARY]
    if not prim:
        return _cannot("RG-05", "estimator: paired episode-cluster bootstrap",
                       "no primary arm was resolved (see RG-01)")
    bad, ok = [], []
    for a in prim:
        blob = json.dumps(a.doc.get("intervals", {}), ensure_ascii=False).lower()
        named = str(cc._dig(a.doc, "intervals.estimator")[1] or
                    a.doc.get("_estimator") or a.doc.get("estimator") or "").lower()
        hit = [g for g in _GOOD_ESTIMATORS if g in named or g in blob]
        if not named and not hit:
            bad.append(f"{a.name}: estimator NOT NAMED — an interval without its "
                       f"estimator is inadmissible")
        elif not hit:
            bad.append(f"{a.name}: estimator is {named[:70]!r}, not the "
                       f"episode-cluster bootstrap")
        else:
            ok.append(f"{a.name}={hit[0]}")
    if bad:
        return Row("RG-05", "estimator: paired episode-cluster bootstrap", FAIL,
                   "; ".join(bad))
    return Row("RG-05", "estimator: paired episode-cluster bootstrap", PASS, "; ".join(ok))


def rg06_forbidden_estimator(arts, cfg) -> Row:
    """⛔ overlapping_holdout_se biases the POINT ESTIMATE, not merely the interval."""
    mine = [a for a in arts if a.role in (PRIMARY, CONTROL)]
    if not mine:
        return _cannot("RG-06", "estimator: overlapping_holdout_se refused",
                       "no primary or control arm was resolved (see RG-01)")
    bad = []
    for a in mine:
        row = next((r for r in a.check.get("hygiene", [])
                    if r["id"] == "hyg.no_forbidden_estimator"), None)
        if row and row["state"] == cc.ABSENT:
            bad.append(f"{a.name}: {row['detail']}")
    if bad:
        return Row("RG-06", "estimator: overlapping_holdout_se refused", FAIL,
                   "; ".join(bad) + " — it is 1.107-3.100x too narrow AND shifts the "
                   "point estimate -6.67%..+11.69%, up to a SIGN FLIP on paired deltas")
    return Row("RG-06", "estimator: overlapping_holdout_se refused", PASS,
               "not used as a decision-grade interval in any release artifact")


def _leak_row(arts, cid, label, guard_id, decl_paths, forbidden_tokens,
              must_not_be_false, extra) -> Row:
    """Scan ONLY the declaration fields, never the whole protocol blob.

    Scanning the blob would flag an artifact whose note reads "no privileged
    inputs" — a disavowal read as a use, the same mistake `_check_forbidden_
    estimator` had to be taught out of.
    """
    prim = [a for a in arts if a.role == PRIMARY]
    if not prim:
        return _cannot(cid, label, "no primary arm was resolved (see RG-01)")
    bad, good = [], []
    for a in prim:
        row = next((r for r in a.check.get("leak_guards", []) if r["id"] == guard_id), None)
        if row is None or row["state"] == cc.ABSENT:
            bad.append(f"{a.name}: UNDECLARED. {extra} The gate cannot check the leak, "
                       f"and an undeclared release is not admissible — this is a free "
                       f"DECLARATION, not a measurement.")
            continue
        if must_not_be_false:
            found, v = cc._dig(a.doc, must_not_be_false)
            if found and v is False:
                bad.append(f"{a.name}: {must_not_be_false} is explicitly FALSE — {extra}")
                continue
        decl = " ".join(json.dumps(cc._dig(a.doc, p)[1], ensure_ascii=False)
                        for p in decl_paths if cc._dig(a.doc, p)[0]).lower()
        hits = [t for t in forbidden_tokens if t in decl]
        if hits:
            bad.append(f"{a.name}: the declaration names {hits} AT INFERENCE — {extra}")
        else:
            good.append(f"{a.name}: {decl[:110]}")
    if bad:
        return Row(cid, label, FAIL, "; ".join(bad))
    return Row(cid, label, PASS, "; ".join(good))


def rg07_vision_only(arts, cfg) -> Row:
    """⛔ BINDING (PI 2026-08-03): labels may use ego; INFERENCE IS VISION-ONLY."""
    return _leak_row(
        arts, "RG-07", "leak guard: vision-only at inference", "vision_only_inference",
        ("protocol.inference_inputs",),
        ("ego_state", "ego state", "ego-state", "privileged", "gt_speed", "gt speed",
         "future pose", "ground truth ego", "ego dynamics", "nav_cmd", "route_graded"),
        "protocol.vision_only",
        "the situation labels are derived from EGO DYNAMICS "
        "(stack/tanitad/data/situations.py), so ego at inference reads the label's "
        "own source — a LEAK, not a capability.")


def rg08_goal_situation_disjoint(arts, cfg) -> Row:
    """⛔ BINDING (PI 2026-08-03): a goal input must not carry the classifier's output."""
    return _leak_row(
        arts, "RG-08", "leak guard: goal / situation information-disjoint",
        "goal_situation_disjoint", ("protocol.goal_source",),
        ("situation classifier", "situation_classifier", "sitclf", "situation posterior",
         "situation argmax", "situation embedding", "situation class", "sit_clf"),
        None,
        "if the goal carries the classifier's output the two paths are one, "
        "attribution dies, and the planner echoes the classifier the way the route "
        "head echoed its nav input.")


def rg09_route_echo(arts, cfg) -> Row:
    """A route head that is a bijection of its own input scores 1.0000 BY ECHO.

    MEASURED 2026-08-03, flagship v1: 369/369 and 81/81, scored 1.0000. An echo of
    its own input, read as skill. So: any near-perfect route/goal score must ship
    its echo test. If no route/goal score exists at all there is nothing to echo,
    and the honest state is N/A with the reason — never a silent PASS.
    """
    cid, label = "RG-09", "leak guard: route-head echo test on a near-1.0 route score"
    prim = [a for a in arts if a.role == PRIMARY]
    if not prim:
        return _cannot(cid, label, "no primary arm was resolved (see RG-01)")
    scored, bad = [], []
    for a in prim:
        for spec in FAMILY_METRICS:
            if spec.family != "STRATEGIC":
                continue
            v, _ = read_metric(a.doc, spec)
            if v is None:
                continue
            scored.append(f"{a.name}:{spec.name}={v:.4f}")
            if v >= cfg.echo_threshold:
                echo = cc._dig(a.doc, "strategic.echo_test")[1]
                if not echo:
                    bad.append(f"{a.name}:{spec.name}={v:.4f} >= {cfg.echo_threshold} "
                               f"with NO strategic.echo_test — indistinguishable from "
                               f"the flagship-v1 bijection that scored 1.0000 by echo")
    if bad:
        return Row(cid, label, FAIL, "; ".join(bad), family="STRATEGIC")
    if not scored:
        return Row(cid, label, NA,
                   "no route/goal score is emitted by this release, so there is nothing "
                   "to echo-test. REASON: the strategic family is declined on this corpus "
                   "(no map, no lane graph, no route signal in PhysicalAI-AV). This N/A "
                   "re-arms the moment a route score appears.", family="STRATEGIC")
    return Row(cid, label, PASS,
               f"route/goal scores below the echo threshold {cfg.echo_threshold}: "
               f"{scored}", family="STRATEGIC")


def rg10_control_floor(arts, cfg) -> list[Row]:
    """⭐ THE CRITERION THAT SEES THE 22x CONTROL WIN.

    MEASURED 2026-08-18, T1: stage-a-repaired closed loop ade_dense_m 9.3697 against
    a hold-action control of 0.4246. Both artifacts are four-families complete. Only
    a comparison against the control can see it. Reported PER FAMILY, never pooled.
    """
    cid, label = "RG-10", "floor: the release beats its control, per family"
    prim = [a for a in arts if a.role == PRIMARY]
    ctrls = [a for a in arts if a.role == CONTROL]
    if not prim:
        return [_cannot(cid, label, "no primary arm was resolved (see RG-01)")]
    if not ctrls:
        return [_cannot(cid, label,
                        "NO control arm (hold-v0 / hold-action / constant-velocity) was "
                        "banked in the swept scope. A headline with no floor cannot be "
                        "read: the programme has MEASURED a control beating a model by "
                        "22x on a complete artifact. Bank the control on the SAME windows "
                        "and re-run.")]
    p = prim[0]
    same = [c for c in ctrls if c.grid == p.grid]
    if not same:
        return [_cannot(cid, label,
                        f"control(s) {[c.name for c in ctrls]} exist but NONE shares the "
                        f"primary's grid fingerprint ({p.grid!r}) — a comparison across "
                        f"different windows is not a comparison")]
    c = same[0]
    rows, losses = [], []
    for fam in FAMILIES:
        specs = [s for s in FAMILY_METRICS if s.family == fam]
        beaten, cmped, missing = [], [], []
        for s in specs:
            mv, _ = read_metric(p.doc, s)
            cv, _ = read_metric(c.doc, s)
            if mv is None or cv is None:
                missing.append(s.name)
                continue
            margin = abs(cv) * cfg.control_margin
            loses = (mv - cv > margin) if s.lower_better else (cv - mv > margin)
            wins = (cv - mv > margin) if s.lower_better else (mv - cv > margin)
            who = "model wins" if wins else ("CONTROL wins" if loses else "tie")
            cmped.append(f"{s.name} model={mv:.4f} control={cv:.4f} {who}")
            if loses:
                beaten.append(f"{s.name}: model={mv:.4f} vs control={cv:.4f} "
                              f"({(mv / cv if cv else float('inf')):.2f}x)")
        if not cmped:
            rows.append(Row(f"{cid}.{fam[:3]}", f"{label} [{fam}]", NA,
                            f"neither arm emits a {fam} metric the gate can compare "
                            f"(missing: {missing}); nothing to floor-check",
                            family=fam))
        elif beaten:
            losses.extend(beaten)
            rows.append(Row(f"{cid}.{fam[:3]}", f"{label} [{fam}]", FAIL,
                            f"⛔ THE CONTROL ({c.arm}) BEATS THE RELEASE: {beaten}",
                            family=fam))
        else:
            rows.append(Row(f"{cid}.{fam[:3]}", f"{label} [{fam}]", PASS,
                            "; ".join(cmped), family=fam))
    # ADE/FDE ride alongside, explicitly labelled. ⛔ They are not a fifth family and
    # a horizon sweep of ADE is never "the result" — but the 22x control win showed
    # up HERE first, so the companion is floor-checked too.
    comp, comp_beaten = [], []
    for s in COMPANION_METRICS:
        mv, _ = read_metric(p.doc, s)
        cv, _ = read_metric(c.doc, s)
        if mv is None or cv is None:
            continue
        comp.append(f"{s.name} model={mv:.4f} control={cv:.4f}")
        margin = abs(cv) * cfg.control_margin
        if (mv - cv > margin) if s.lower_better else (cv - mv > margin):
            comp_beaten.append(f"{s.name}: model={mv:.4f} vs control={cv:.4f} "
                               f"({(mv / cv if cv else float('inf')):.2f}x)")
    if not comp:
        state, detail = NA, "no companion metric is comparable between the two arms"
    elif comp_beaten:
        state = FAIL
        detail = f"⛔ THE CONTROL ({c.arm}) BEATS THE RELEASE: {comp_beaten}"
    else:
        state, detail = PASS, f"control={c.name}; " + "; ".join(comp)
    rows.append(Row(f"{cid}.CMP", f"{label} [ADE/FDE companion — NOT a family]",
                    state, detail, family="COMPANION"))
    return rows


def rg11_regression(arts, base_arts, cfg) -> list[Row]:
    cid, label = "RG-11", "regression: no family regresses vs the previous release"
    prim = [a for a in arts if a.role == PRIMARY]
    if not prim:
        return [_cannot(cid, label, "no primary arm was resolved (see RG-01)")]
    if not cfg.baseline:
        return [Row(cid, label, NA,
                    "no --baseline was named. REASON: a first release has nothing to "
                    "regress against. Every later release MUST pass one.")]
    bp = [a for a in base_arts if a.role == PRIMARY]
    if not bp:
        return [_cannot(cid, label,
                        f"--baseline {cfg.baseline!r} was named but no primary driving "
                        f"artifact for it is in the swept scope. A named-but-absent "
                        f"baseline is not a pass.")]
    p, b = prim[0], bp[0]
    if p.grid != b.grid:
        return [_cannot(cid, label,
                        f"the release and the baseline are on DIFFERENT grids "
                        f"({p.grid!r} vs {b.grid!r}) — a paired comparison across "
                        f"different windows is invalid, not merely weaker")]
    rows = []
    for fam in FAMILIES:
        specs = [s for s in FAMILY_METRICS if s.family == fam]
        worse, cmped = [], []
        for s in specs:
            nv, _ = read_metric(p.doc, s)
            ov, _ = read_metric(b.doc, s)
            if nv is None or ov is None:
                continue
            tol = abs(ov) * cfg.regression_tol
            regressed = (nv - ov > tol) if s.lower_better else (ov - nv > tol)
            cmped.append(f"{s.name} {ov:.4f}->{nv:.4f}")
            if regressed:
                worse.append(f"{s.name}: {ov:.4f} -> {nv:.4f}")
        if not cmped:
            rows.append(Row(f"{cid}.{fam[:3]}", f"{label} [{fam}]", NA,
                            f"no comparable {fam} metric in both releases", family=fam))
        elif worse:
            trade = cfg.accept_regression.get(fam)
            if trade:
                rows.append(Row(f"{cid}.{fam[:3]}", f"{label} [{fam}]", PASS,
                                f"REGRESSED {worse} — TRADE DECLARED: {trade}", family=fam))
            else:
                rows.append(Row(f"{cid}.{fam[:3]}", f"{label} [{fam}]", FAIL,
                                f"regressed vs {cfg.baseline}: {worse}. Either fix it or "
                                f"state the trade with "
                                f"--accept-regression {fam}='<reason>'", family=fam))
        else:
            rows.append(Row(f"{cid}.{fam[:3]}", f"{label} [{fam}]", PASS,
                            "; ".join(cmped), family=fam))
    return rows


def rg12_parity(arts, cfg, registry) -> Row:
    """Parity is sacred: anything that re-selects episodes breaks comparability.

    The registry owns `hyg.parity` and its key list; this criterion adopts those
    keys and adds the bite completeness cannot have — the recorded value must NAME
    the canonical corpus and skip-hash, not merely exist.
    """
    cid, label = "RG-12", "parity: canonical corpus + skip-hash stamped"
    prim = [a for a in arts if a.role in (PRIMARY, CONTROL)]
    if not prim:
        return _cannot(cid, label, "no primary or control arm was resolved (see RG-01)")
    reg_keys = tuple(next((c.get("keys", []) for c in registry
                           .get("artifact_hygiene", {}).get("criteria", [])
                           if c.get("id") == "hyg.parity"), []))
    paths = reg_keys + ("parity.corpus", "parity.key", "parity.skip_hash",
                        "corpus.key", "corpus.skip_hash", "protocol.skip_hash",
                        "grid_fingerprint.corpus", "skip_hash")
    bad = []
    for a in prim:
        blob = " ".join(str(cc._dig(a.doc, p)[1]) for p in paths
                        if cc._dig(a.doc, p)[0])
        if not blob.strip():
            bad.append(f"{a.name}: NO parity stamp at any of {list(paths[:8])} — the "
                       f"corpus a number was computed on is a free DECLARATION, and its "
                       f"absence is a release defect, not an unrulable criterion")
        elif cfg.parity_key not in blob:
            bad.append(f"{a.name}: parity stamp does not name {cfg.parity_key!r} "
                       f"(found {blob[:90]!r}) — a re-selected episode set breaks "
                       f"cross-arm comparability and must be refused")
        elif cfg.parity_skip_hash not in blob:
            bad.append(f"{a.name}: corpus named but skip-hash {cfg.parity_skip_hash!r} "
                       f"absent (found {blob[:90]!r})")
    if bad:
        return Row(cid, label, FAIL, "; ".join(bad))
    return Row(cid, label, PASS,
               f"all release artifacts stamp {cfg.parity_key} / {cfg.parity_skip_hash}")


def rg13_grid_identity(arts, base_arts, cfg) -> Row:
    cid, label = "RG-13", "grid: every compared arm sits on one identical window set"
    comp = [a for a in arts if a.role in (PRIMARY, CONTROL)] + \
           [a for a in base_arts if a.role == PRIMARY]
    if len(comp) < 2:
        return Row(cid, label, NA,
                   "fewer than two arms are being compared, so there is no grid to "
                   "reconcile (a single-arm release has nothing to mismatch)")
    grids = {}
    for a in comp:
        grids.setdefault(a.grid, []).append(a.name)
    if len(grids) > 1:
        return Row(cid, label, FAIL,
                   f"{len(grids)} DIFFERENT grid fingerprints among the compared arms: "
                   f"{ {k[:44]: v for k, v in grids.items()} }. Deltas across different "
                   f"windows are not deltas.")
    return Row(cid, label, PASS,
               f"{len(comp)} arms share one grid: {list(grids)[0][:70]}")


def rg14_interval_can_rule(arts, cfg) -> Row:
    """⭐ THE PREFLIGHT THE O6 RANK GATE NEVER RAN.

    The episode-cluster bootstrap resamples EPISODES. Below a floor of clusters
    the interval is not decision-grade however many windows sit inside them, and
    a gate leaning on it would return a perpetual INCONCLUSIVE that looks like
    caution. It says so instead, and names what it disarms.
    """
    cid, label = "RG-14", "estimator can rule: episode clusters >= floor"
    prim = [a for a in arts if a.role == PRIMARY]
    if not prim:
        return _cannot(cid, label, "no primary arm was resolved (see RG-01)")
    a = prim[0]
    if a.n_episodes is None:
        return Row(cid, label, FAIL,
                   f"{a.name}: n_episodes NOT REPORTED. n is a required declaration "
                   f"(hyg.n_reported); without it the estimator's resolving power is "
                   f"unknowable and no interval here is decision-grade.")
    if int(a.n_episodes) < cfg.episode_floor:
        return _cannot(cid, label,
                       f"{a.name} has n_episodes={a.n_episodes}, below the "
                       f"episode-cluster floor of {cfg.episode_floor}. The bootstrap "
                       f"resamples EPISODES, so {a.n_windows} windows do not help — "
                       f"they are {a.n_episodes} independent draws. ⚠️ THIS ALSO "
                       f"DISARMS the interval half of RG-10 and RG-11, which fall back "
                       f"to point estimates; their point comparisons still rule.")
    return Row(cid, label, PASS,
               f"n_episodes={a.n_episodes} >= floor {cfg.episode_floor}, "
               f"n_windows={a.n_windows} — the episode-cluster bootstrap can resolve")


# ----------------------------------------------------------------- driver ---
def run_gate(artifacts_dir: str, cfg: Cfg, registry: dict) -> dict:
    arts = load_artifacts(artifacts_dir, registry)
    assign_roles(arts, cfg.model, cfg.primary_arm_key, registry)
    base_arts: list[Artifact] = []
    if cfg.baseline:
        base_arts = load_artifacts(artifacts_dir, registry)
        assign_roles(base_arts, cfg.baseline, cfg.primary_arm_key, registry)

    rows: list[Row] = [rg01_scope(arts, cfg), rg02_completeness(arts, cfg),
                       rg03_tier_stamp(arts, cfg), rg04_t1_present(arts, cfg, registry),
                       rg05_estimator(arts, cfg), rg06_forbidden_estimator(arts, cfg),
                       rg07_vision_only(arts, cfg), rg08_goal_situation_disjoint(arts, cfg),
                       rg09_route_echo(arts, cfg)]
    rows += rg10_control_floor(arts, cfg)
    rows += rg11_regression(arts, base_arts, cfg)
    rows += [rg12_parity(arts, cfg, registry), rg13_grid_identity(arts, base_arts, cfg),
             rg14_interval_can_rule(arts, cfg)]

    for r in rows:
        r.blocking, r.blocking_why = blocking_for(r.id, registry, cfg)

    n_fail = sum(r.state == FAIL for r in rows)
    n_cannot = sum(r.state == CANNOT_RULE for r in rows)
    blk_fail = sum(r.state == FAIL and r.blocking for r in rows)
    blk_cannot = sum(r.state == CANNOT_RULE and r.blocking for r in rows)
    verdict = "RELEASE-BLOCKED" if blk_fail else \
              ("RELEASE-INCONCLUSIVE" if blk_cannot else "RELEASE-CLEARED")
    return {"spec": SPEC_DOC, "registry_version": registry.get("version"),
            "model": cfg.model, "baseline": cfg.baseline,
            "artifacts_dir": artifacts_dir,
            "blocking_policy": registry.get("release_gate", {}).get(
                "pi_decision_open", "no release_gate block in the registry"),
            "scope": [{"file": a.path, "scope": a.scope, "role": a.role,
                       "tier": a.tier, "arm": a.arm, "arm_key": a.arm_key,
                       "n_windows": a.n_windows, "n_episodes": a.n_episodes,
                       "grid": a.grid, "why": a.scope_why} for a in arts],
            "criteria": [r.as_dict() for r in rows],
            "counts": {"PASS": sum(r.state == PASS for r in rows), "FAIL": n_fail,
                       "CANNOT_RULE": n_cannot, "NA": sum(r.state == NA for r in rows),
                       "BLOCKING_FAIL": blk_fail, "BLOCKING_CANNOT_RULE": blk_cannot,
                       "ADVISORY_FAIL": n_fail - blk_fail},
            "verdict": verdict,
            "verdict_is": "a CONJUNCTION over named criteria — ⛔ NOT a pooled score",
            "_rows": rows, "_arts": arts}


_MARK = {PASS: "  PASS  ", FAIL: "  FAIL  ", CANNOT_RULE: "CAN'T-RULE", NA: "  N/A   "}


def render(res: dict) -> str:
    out = [f"TANITAD RELEASE GATE — model {res['model']!r}"
           f"{' vs baseline ' + repr(res['baseline']) if res['baseline'] else ''}",
           f"spec: {res['spec']}   registry v{res['registry_version']}", "=" * 78, ""]

    # ⛔ SCOPE FIRST, ALWAYS. A gate that silently swept one directory reports that
    # directory's verdict, not the model's — the `df`/Thor-`free`/cgroup trap in a
    # fourth costume. Every file read is listed, including the ones excluded.
    out.append(f"SCOPE SWEPT: {res['artifacts_dir']} (recursive)")
    out.append(f"{len(res['scope'])} file(s) read. A finding here is a finding ABOUT "
               f"THIS SCOPE.")
    for s in res["scope"]:
        out.append(f"   [{s['role'] or s['scope']:<14}] {os.path.basename(s['file'])}"
                   f"  tier={s['tier'] or '-'} arm={s['arm']!r} "
                   f"n_win={s['n_windows']} n_ep={s['n_episodes']}")
        if s["role"] in (UNKNOWN, UNREADABLE):
            out.append(f"        -> {s['why']} (NOT counted as compliant)")
    out.append("")

    pre = [r for r in res["_rows"] if not r.can_rule]
    out.append("CAN-RULE PREFLIGHT")
    if pre:
        out.append("   ⛔ these criteria CANNOT RULE at these settings. They are NOT "
                   "passes and NOT a perpetual INCONCLUSIVE — each names what it needs:")
        for r in pre:
            out.append(f"   [{r.id}] {r.label}")
            out.append(f"        -> {r.why_not}")
    else:
        out.append("   every criterion that applies could rule at these settings.")
    # A family whose EVERY criterion is N/A is invisible in a table of passes. The
    # hierarchy is the programme's thesis: if the strategic level is never measured
    # we cannot claim it works, and a quiet "N/A" must not read as "fine".
    for fam in FAMILIES:
        fr = [r for r in res["_rows"] if r.family == fam]
        if fr and all(r.state == NA for r in fr):
            out.append(f"   ⭐ {fam} IS WHOLLY UNMEASURED IN THIS RELEASE — every "
                       f"{fam} criterion returned N/A. The release makes NO checkable "
                       f"{fam} claim. Reason(s): {fr[0].detail[:150]}")
    out.append("")

    out.append("CRITERIA   (blocking vs advisory comes from "
               "CRITERIA_REGISTRY.json:release_gate, not from this tool)")
    for r in res["_rows"]:
        tag = "" if r.blocking else "  [ADVISORY]"
        out.append(f"   [{_MARK[r.state]}] {r.id:<9} {r.label}{tag}")
        if r.state != PASS or len(r.detail) < 400:
            out.append(f"              {r.detail}")
        if r.state in (FAIL, CANNOT_RULE) and not r.blocking:
            out.append(f"              (advisory because: {r.blocking_why})")
    out.append("")

    out.append("PER-FAMILY PANEL  (⛔ never pooled — a composite hides the trade-off)")
    for fam in FAMILIES + ("COMPANION",):
        fr = [r for r in res["_rows"] if r.family == fam]
        if not fr:
            out.append(f"   {fam:<12} no family-scoped criterion evaluated")
            continue
        worst = sorted(fr, key=lambda r: _ORDER[r.state])[0].state
        out.append(f"   {fam:<12} {worst:<11} "
                   f"({', '.join(f'{r.id}={r.state}' for r in fr)})")
    out.append("")

    c = res["counts"]
    out.append(f"PASS {c['PASS']} · FAIL {c['FAIL']} · CANNOT-RULE {c['CANNOT_RULE']} "
               f"· N/A {c['NA']}")
    out.append(f"   of the FAILs, {c['BLOCKING_FAIL']} are BLOCKING and "
               f"{c['ADVISORY_FAIL']} advisory; "
               f"{c['BLOCKING_CANNOT_RULE']} blocking criteria CANNOT RULE")
    out.append(f"   policy: {res['blocking_policy']}")
    out.append(f"VERDICT: {res['verdict']}   [{res['verdict_is']}]")
    return "\n".join(out)


def _kv_list(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--accept-regression needs FAMILY=reason, got {p!r}")
        k, v = p.split("=", 1)
        if not v.strip():
            raise SystemExit(f"--accept-regression {k}= needs a REASON; a blank trade "
                             f"is a shrug, not a decision")
        out[k.strip().upper()] = v.strip()
    return out


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="registry key / arm of the release")
    ap.add_argument("--artifacts", required=True, help="directory of eval artifacts")
    ap.add_argument("--baseline", default=None, help="previous release key")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--registry", default=str(cc.REGISTRY))
    ap.add_argument("--primary-arm-key", default="cl",
                    help="which arm is the headline (default: cl, action-closed)")
    ap.add_argument("--control-margin", type=float, default=0.0,
                    help="relative slack before a control win counts as a loss")
    ap.add_argument("--regression-tol", type=float, default=0.0,
                    help="relative slack before a family counts as regressed")
    ap.add_argument("--accept-regression", action="append", default=[],
                    metavar="FAMILY=reason",
                    help="declare a regression trade EXPLICITLY and on the record")
    ap.add_argument("--parity-key", default=CANON_CORPUS)
    ap.add_argument("--parity-skip-hash", default=CANON_SKIP_HASH)
    ap.add_argument("--echo-threshold", type=float, default=0.99)
    ap.add_argument("--episode-floor", type=int, default=EPISODE_CLUSTER_FLOOR)
    ap.add_argument("--regression-blocking", action="store_true",
                    help="promote RG-11 from advisory to blocking. The registry keeps "
                         "it advisory until two clean releases exist to compare — a "
                         "regression gate with no trustworthy baseline manufactures "
                         "false failures.")
    a = ap.parse_args(argv)

    registry = cc._read_json(Path(a.registry))
    if registry is None:
        print(f"FATAL: cannot read criteria registry at {a.registry}", file=sys.stderr)
        return 3
    if not os.path.exists(a.artifacts) and not glob.glob(a.artifacts):
        print(f"FATAL: --artifacts path does not exist: {a.artifacts}", file=sys.stderr)
        return 3

    cfg = Cfg(model=a.model, baseline=a.baseline, primary_arm_key=a.primary_arm_key,
              control_margin=a.control_margin, regression_tol=a.regression_tol,
              accept_regression=_kv_list(a.accept_regression),
              parity_key=a.parity_key, parity_skip_hash=a.parity_skip_hash,
              echo_threshold=a.echo_threshold, episode_floor=a.episode_floor,
              regression_blocking=a.regression_blocking)
    res = run_gate(a.artifacts, cfg, registry)
    print(render(res))

    if a.json_out:
        pub = {k: v for k, v in res.items() if not k.startswith("_")}
        Path(a.json_out).write_text(
            json.dumps(pub, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {a.json_out}")

    if res["counts"]["BLOCKING_FAIL"]:
        return 1
    if res["counts"]["BLOCKING_CANNOT_RULE"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
