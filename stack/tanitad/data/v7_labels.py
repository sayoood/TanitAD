"""The ONE B1 label consumer — v7 strategic/tactical ground truth.

Read by every B1 consumer (v7f, refc_v3, refa_v1, refd) so the D-LABEL-GT
conditions are implemented ONCE. Spec: ``Project Steering/SPEC_V7_LABELS_CONSUMER.md``.

⭐ THIS MODULE VALIDATES ITS ASSUMPTIONS AGAINST THE DATA AT LOAD TIME AND FAILS
LOUDLY ON DIVERGENCE. It trusts neither the spec nor ``B1_TRAINING_PREP.md``: a
spec is a document, documents go stale, and this corpus has already had its blob
re-pinned five times with **six copies in circulation under three roots** whose
md5s differ. ⇒ :func:`load_v7_labels` asserts record count, ``schema_version`` and
``vocab``, and returns the file's **md5 in the manifest**, so *"which blob did this
run read"* is answerable from the run's own output rather than from a path.

⚠️ THE THREE BINDING CONDITIONS, each with the failure that earned it:

1. ⛔ **``nav_command`` IS AN ORACLE ON ALL 4,719 RECORDS** — every one carries
   ``provenance: "ego-future"``, i.e. it is computed from the ego's own future
   path. ⚠️ **But the ``oracle: true`` BOOLEAN is present on only 4,190 of them
   (529 missing, 11.2 %)**; those records carry a ``reason`` instead and are all
   ``NAV_FOLLOW_ROAD``. The spec's "oracle:true, 100 %" is true of the
   PROVENANCE and false of the FLAG — a guard keyed on the boolean would let 11 %
   of the corpus through as non-oracle. See :func:`is_oracle_nav`. Feeding it at inference reproduces the flagship-v1 route echo exactly:
   that head was a bijection of the nav we fed it (369/369, 81/81) and **scored
   1.0000 by echoing its own input**. ⇒ it is unreachable without
   ``allow_oracle_nav=True``, and that flag STAMPS the manifest so no eval can
   quote such an arm without the stamp being visible. ⭐ The admissible goal
   signal is :attr:`V7Label.tac_anchor` — a PREDICTED geometric goal point
   (``goal_x_m``/``goal_y_m``/``t_reach_s``), which is what the literature shows
   actually works (+4.7 PDMS vs +0.2 for a categorical command).

2. ⭐ **``a_tac`` IS ALREADY FACTORED into lat and lon** and must never be
   flattened. The 5-way softmax that MIXED lat and lon is the programme's single
   largest known defect — it explains 0/881 accelerate, the speed-fan, and the
   absence of longitudinal signal in selection. Flattening here would rebuild it
   in the labels. :func:`flatten_tactical_actions` exists ONLY so a test can
   assert it is refused.

3. **Head width stays FULL vocab; the MASK is applied to the LOSS.** A masked
   class must be exactly an absent class — :func:`assert_mask_matches_presence`
   fails loudly if a future blob makes a masked class non-empty (training signal
   thrown away) or leaves an unmasked class absent (a dead logit that degrades
   calibration).

⚠️ MEASURED DIVERGENCES FROM THE SPEC (2026-08-30, this blob) — reported, never
invented, and never silently dropped. The spec states ``disputed``,
``time_basis`` and ``t_nominal_s`` "returned ZERO hits", and asks for
``alpamayo.agree`` at the top of ``alpamayo``. **All four DO exist, one level
deeper than the spec's search reached:**

* ``disputed`` / ``time_basis`` / ``t_nominal_s`` live **per goal** inside
  ``g_tac.goals.<TOKEN>`` — thousands of occurrences (e.g. ``YIELD.disputed`` on
  626 records, ``CORRIDOR_OFFSET.disputed`` on 885).
* ``agree`` lives in ``alpamayo.lateral.agree`` and ``alpamayo.longitudinal.agree``
  (4,719 each), not at ``alpamayo.agree``.

Had this module followed the spec it would have **dropped a disputed-flag that
exists on thousands of goals** — the D-LABEL-GT conditions require it be
respected. This is why the loader measures rather than trusts.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from tanitad.models.vocab_v7 import (NOT_YET_EXTRACTABLE, STRATEGIC_ACTION_TOKENS_V7,
                                     STRATEGIC_GOAL_TOKENS_V7,
                                     TACTICAL_LAT_ACTIONS_V7,
                                     TACTICAL_LON_ACTIONS_V7)

__all__ = ["HEADS", "LabelManifest", "V7Label", "assert_mask_matches_presence",
           "class_weights", "effective_mask", "flatten_tactical_actions",
           "head_mask", "is_oracle_nav", "load_v7_labels", "oracle_nav"]

#: The four supervised heads. ⛔ ``tac_lat`` and ``tac_lon`` are SEPARATE by
#: design (condition 2) — never merge them into one 25-way or 64-way head.
HEADS: dict[str, tuple[str, ...]] = {
    "tac_lat": tuple(TACTICAL_LAT_ACTIONS_V7),
    "tac_lon": tuple(TACTICAL_LON_ACTIONS_V7),
    "str_action": tuple(STRATEGIC_ACTION_TOKENS_V7),
    "str_goal": tuple(STRATEGIC_GOAL_TOKENS_V7),
}

EXPECTED_SCHEMA = "s2-geom-v7"
EXPECTED_VOCAB = "v7"


class OracleNavRefused(RuntimeError):
    """Raised when oracle nav is reached without an explicit opt-in."""


@dataclass(frozen=True)
class V7Label:
    """One clip's labels. ⛔ ``_oracle`` is private — go through :func:`oracle_nav`."""

    clip_id: str
    tac_lat: str
    tac_lon: str
    str_action: str
    str_goal: str
    #: ⭐ the ADMISSIBLE goal signal: a predicted geometric goal point.
    tac_anchor: dict[str, Any] | None
    bands: dict[str, Any]
    t0_s: float
    horizon: dict[str, Any]
    #: audit-only, NEVER a training input (spec §6)
    audit: dict[str, Any] = field(default_factory=dict)
    _oracle: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class LabelManifest:
    """What this run actually read. Goes into the run config verbatim."""

    path: str
    md5: str
    n_records: int
    schema_version: str
    vocab: str
    #: ⛔ the stamp. An arm trained with oracle nav carries True here and no eval
    #: can quote it without the flag being visible.
    allow_oracle_nav: bool
    divergences: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "md5": self.md5, "n_records": self.n_records,
                "schema_version": self.schema_version, "vocab": self.vocab,
                "allow_oracle_nav": self.allow_oracle_nav,
                "divergences": list(self.divergences),
                "_read": "md5 is the identity — six copies of this blob exist "
                         "under three roots and their md5s differ."}


def _goal_audit(g_tac: dict[str, Any] | None) -> dict[str, Any]:
    """Per-goal ``disputed`` / ``time_basis`` / ``t_nominal_s``.

    ⚠️ These are the fields the spec reported as non-existent. They are nested
    inside ``g_tac.goals.<TOKEN>`` and are surfaced here rather than dropped.
    """
    goals = (g_tac or {}).get("goals") or {}
    out: dict[str, Any] = {}
    for token, rec in goals.items():
        if not isinstance(rec, dict):
            continue
        keep = {k: rec[k] for k in ("disputed", "time_basis", "t_nominal_s",
                                    "held", "grounded", "grounding") if k in rec}
        if keep:
            out[token] = keep
    return out


def load_v7_labels(path: str | Path, *, allow_oracle_nav: bool = False,
                   require_records: int | None = None,
                   require_schema: str = EXPECTED_SCHEMA,
                   require_vocab: str = EXPECTED_VOCAB,
                   ) -> tuple[list[V7Label], LabelManifest]:
    """Load the B1 label blob, validating every assumption against the data.

    ``require_records`` refuses a blob of unexpected size. Pass it whenever the
    expected count is known: with six copies in circulation, a silently smaller
    file is a corpus change wearing a filename.
    """
    p = Path(path)
    raw = p.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]

    schemas = Counter(r.get("schema_version") for r in recs)
    vocabs = Counter(r.get("vocab") for r in recs)
    if len(schemas) != 1 or next(iter(schemas)) != require_schema:
        raise ValueError(
            f"[v7_labels] ⛔ schema_version mismatch in {p}: found {dict(schemas)}, "
            f"required {require_schema!r}. A blob with a different schema is a "
            f"different experiment — refusing rather than coercing.")
    if len(vocabs) != 1 or next(iter(vocabs)) != require_vocab:
        raise ValueError(f"[v7_labels] ⛔ vocab mismatch in {p}: {dict(vocabs)} "
                         f"!= {require_vocab!r}")
    if require_records is not None and len(recs) != require_records:
        raise ValueError(
            f"[v7_labels] ⛔ {p} has {len(recs)} records, expected "
            f"{require_records}. md5={md5}. Six copies of this blob exist under "
            f"three roots with differing md5s — check which one you pointed at.")

    divergences: list[str] = []
    nav_prov = Counter((r.get("nav_command") or {}).get("provenance")
                       for r in recs)
    if set(nav_prov) - {"ego-future"}:
        divergences.append(
            f"nav_command.provenance is not uniformly 'ego-future': "
            f"{dict(nav_prov)} — a real nav-system source would change the "
            f"oracle argument and must be reviewed, not silently accepted.")

    labels: list[V7Label] = []
    for r in recs:
        a_tac = r.get("a_tac") or {}
        g_tac = r.get("g_tac") or {}
        alp = r.get("alpamayo") or {}
        labels.append(V7Label(
            clip_id=r["clip_id"],
            tac_lat=a_tac.get("lat"), tac_lon=a_tac.get("lon"),
            str_action=(r.get("a_str") or {}).get("token"),
            str_goal=(r.get("g_str") or {}).get("token"),
            tac_anchor=g_tac.get("anchor"),
            bands=r.get("bands") or {},
            t0_s=float(r.get("t0_s", float("nan"))),
            horizon=r.get("horizon") or {},
            audit={
                "turn_suppression": r.get("turn_suppression"),
                "unassigned_manoeuvres": (r.get("bands") or {}).get(
                    "unassigned_manoeuvres") or [],
                "goal_flags": _goal_audit(g_tac),
                "g_tac_violations": g_tac.get("violations"),
                "a_tac_args": {"lat": a_tac.get("lat_args"),
                               "lon": a_tac.get("lon_args")},
                "a_tac_truncated": a_tac.get("truncated"),
                "serves_goals": a_tac.get("serves_goals"),
                # ⚠️ nested one level below where the spec looked
                "alpamayo_agree": {
                    "lateral": (alp.get("lateral") or {}).get("agree"),
                    "longitudinal": (alp.get("longitudinal") or {}).get("agree")},
            },
            _oracle={"nav_command": r.get("nav_command")},
        ))

    return labels, LabelManifest(
        path=str(p), md5=md5, n_records=len(recs),
        schema_version=next(iter(schemas)), vocab=next(iter(vocabs)),
        allow_oracle_nav=bool(allow_oracle_nav),
        divergences=tuple(divergences))


def oracle_nav(label: V7Label, manifest: LabelManifest) -> dict[str, Any]:
    """The oracle ``nav_command`` — refused unless the manifest carries the stamp.

    ⛔ This is the ONLY route to the field, and it checks the MANIFEST rather than
    a bare argument so the permission and the run's recorded config cannot
    disagree. An arm that used oracle nav is identifiable from its own artifacts.
    """
    if not manifest.allow_oracle_nav:
        raise OracleNavRefused(
            "[v7_labels] ⛔ nav_command is an ORACLE — provenance 'ego-future', "
            "computed from the ego's own future path, on all records. Feeding it "
            "at inference reproduces the flagship-v1 route echo (a bijection of "
            "its own input that scored 1.0000).\n"
            "     ⇒ pass allow_oracle_nav=True to load_v7_labels if this is a "
            "deliberate oracle-ceiling arm; the flag stamps the manifest so no "
            "eval can quote the arm without it being visible.\n"
            "     ⭐ For a usable goal signal use V7Label.tac_anchor — a "
            "PREDICTED geometric goal point, which is admissible and is the "
            "lever the literature shows actually works.")
    return label._oracle["nav_command"]


def is_oracle_nav(label: "V7Label") -> bool:
    """Is this record's nav an oracle? Keyed on PROVENANCE, not the flag.

    ⛔ MEASURED 2026-08-30 — THE `oracle` BOOLEAN IS ABSENT ON 529 OF 4,719
    RECORDS (11.2 %). Those records carry a ``reason`` instead (*"curve, not a
    turn — nav does not command a turn"*), are all ``NAV_FOLLOW_ROAD``, and are
    all still ``provenance: "ego-future"`` — i.e. **still computed from the ego's
    own future**. A guard written as ``if rec["oracle"]`` would raise KeyError,
    and one written as ``rec.get("oracle") is True`` would silently let **11 % of
    the corpus through as non-oracle**.

    ⇒ ``provenance`` is the load-bearing field; the boolean is a convenience that
    is not always written. The spec's "oracle:true, 100 %" is true of the
    provenance and NOT of the flag.
    """
    nav = label._oracle.get("nav_command") or {}
    return nav.get("provenance") == "ego-future" or bool(nav.get("oracle"))


def head_mask(head: str) -> tuple[bool, ...]:
    """Per-class LOSS mask: True = trainable, False = masked (not yet extractable).

    ⛔ The HEAD keeps full vocab width; only the loss is masked.
    """
    if head not in HEADS:
        raise KeyError(f"unknown head {head!r}; known: {sorted(HEADS)}")
    return tuple(t not in NOT_YET_EXTRACTABLE for t in HEADS[head])


def effective_mask(labels: Sequence["V7Label"], head: str
                   ) -> tuple[tuple[bool, ...], dict[str, str]]:
    """The mask a trainer should actually use: vocab-masked UNION empirically-absent.

    ⛔ WHY THIS IS NOT JUST ``head_mask``. MEASURED 2026-08-30 on the released
    blob: ``vocab_v7.NOT_YET_EXTRACTABLE`` contains
    ``LANE_CHANGE_L_FOLLOW_ROUTE`` / ``LANE_CHANGE_R_FOLLOW_ROUTE`` (**strategic
    goal** tokens) but NOT ``LANE_CHANGE_L`` / ``LANE_CHANGE_R`` (**tactical lat
    action** tokens). The two pairs differ only by a suffix. Both tactical tokens
    have **zero occurrences** in the corpus, so relying on the vocab mask alone
    leaves **two dead logits** on the ``tac_lat`` head — present in the softmax,
    never trained, degrading calibration.

    ⚠️ This is a NAME-COLLISION trap: the spec's §3 arithmetic ("the 8
    NOT_YET_EXTRACTABLE tokens are exactly the classes with zero occurrences")
    holds only if the two families are not confused, and it does not hold here.

    ⭐ WHY THE TWO TACTICAL TOKENS ARE ABSENT — established from source, not
    inferred, so no consumer re-derives it. There is **no LANE_CHANGE branch
    anywhere in the extractor**: ``ego_manoeuvre.py:97`` declares
    ``lateral_class`` over ``JUNCTION_TURN_L/R | ROAD_BEND_L/R | NUDGE_L/R |
    STRAIGHT``, and ``s2_geom_emit_v7.py`` (:446/:449/:452) emits only NUDGE,
    TURN or LANE_KEEP. It is an **extractor limit, not a corpus accident**, and
    it is the PI's 2026-08-16 lane-change ruling applied consistently: with no
    lane reference, lateral offset alone is not evidence of a lane change, so
    NUDGE is the HONEST label rather than a lost one.

    ⛔ **CONSUMER-VISIBLE CONSEQUENCE — ``NUDGE_L/R`` IS A SUPERSET.** The nudge
    test is ``abs(lat) >= NUDGE_LAT_M`` with ``NUDGE_LAT_M = 1.0`` m and **no
    upper bound** (``ego_manoeuvre.py:82``, :318), and a lane is 3.2-3.7 m — so a
    real lane change satisfies it *by construction*. ⚠️ **And the same line also
    requires ``abs(peak_yaw) >= 5.0`` degrees**, so a GENTLE lane change fails the
    nudge test and lands in ``LANE_KEEP`` instead. ⇒ a lane change is absorbed
    into **NUDGE or LANE_KEEP depending on yaw**, not reliably into either.
    Do not read ``NUDGE_L`` as "a small lateral correction", and do not read
    ``LANE_KEEP`` as "stayed in lane". The lane-detector reference is the
    instrument that would separate them.

    ⚠️ USE ON THE FULL CORPUS LOAD, never on a small split — a class absent from
    200 windows is not a class absent from the corpus, and masking it would throw
    away real signal. :func:`assert_mask_matches_presence` remains the alarm that
    the VOCAB needs fixing; this keeps the trainer safe meanwhile.

    Returns ``(mask, provenance)`` where provenance maps each masked token to
    ``"vocab"``, ``"absent"`` or ``"both"`` — so a run can report WHY a logit was
    switched off.
    """
    attr = {"tac_lat": "tac_lat", "tac_lon": "tac_lon",
            "str_action": "str_action", "str_goal": "str_goal"}[head]
    seen = Counter(getattr(x, attr) for x in labels)
    mask, prov = [], {}
    for tok in HEADS[head]:
        by_vocab = tok in NOT_YET_EXTRACTABLE
        by_absence = seen.get(tok, 0) == 0
        if by_vocab or by_absence:
            prov[tok] = ("both" if by_vocab and by_absence
                         else ("vocab" if by_vocab else "absent"))
        mask.append(not (by_vocab or by_absence))
    return tuple(mask), prov


def assert_mask_matches_presence(labels: Sequence[V7Label],
                                 heads: Iterable[str] | None = None) -> dict:
    """⛔ A masked class must be exactly an absent class, per head.

    Fails loudly in BOTH directions, because both are silent damage:
      * masked but PRESENT  -> training signal thrown away;
      * unmasked but ABSENT -> a dead logit that degrades calibration.
    """
    attr = {"tac_lat": "tac_lat", "tac_lon": "tac_lon",
            "str_action": "str_action", "str_goal": "str_goal"}
    report = {}
    problems = []
    for head in (heads or HEADS):
        seen = Counter(getattr(x, attr[head]) for x in labels)
        present = {t for t in HEADS[head] if seen.get(t, 0) > 0}
        masked = {t for t in HEADS[head] if t in NOT_YET_EXTRACTABLE}
        absent = set(HEADS[head]) - present
        report[head] = {"width": len(HEADS[head]), "present": sorted(present),
                        "masked": sorted(masked), "counts": dict(seen)}
        if masked - absent:
            problems.append(f"{head}: MASKED BUT PRESENT {sorted(masked - absent)} "
                            f"— this blob carries training signal the mask throws "
                            f"away; unmask it or explain why")
        if absent - masked:
            problems.append(f"{head}: ABSENT BUT UNMASKED {sorted(absent - masked)} "
                            f"— dead logits that degrade calibration; mask them")
    if problems:
        raise AssertionError("[v7_labels] ⛔ mask/presence mismatch:\n  "
                             + "\n  ".join(problems))
    return report


def class_weights(labels: Sequence[V7Label], head: str,
                  *, scheme: str = "inverse") -> dict[str, float]:
    """Skew weights COMPUTED FROM THE LOADED SPLIT, never hardcoded.

    ⚠️ ``HOLD_MAIN_ROAD`` is 52.3 % of ``str_action`` in this blob. A hardcoded
    weight silently mis-weights the next blob — the derived-constant trap. Masked
    classes get weight 0.0: they contribute no loss, so any other value is a lie
    about what the objective does.
    """
    attr = {"tac_lat": "tac_lat", "tac_lon": "tac_lon",
            "str_action": "str_action", "str_goal": "str_goal"}[head]
    seen = Counter(getattr(x, attr) for x in labels)
    n_present = sum(1 for t in HEADS[head] if seen.get(t, 0) > 0)
    total = sum(seen.get(t, 0) for t in HEADS[head])
    out: dict[str, float] = {}
    for t in HEADS[head]:
        c = seen.get(t, 0)
        if t in NOT_YET_EXTRACTABLE or c == 0:
            out[t] = 0.0
        elif scheme == "inverse":
            out[t] = total / (n_present * c)
        elif scheme == "uniform":
            out[t] = 1.0
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
    return out


def flatten_tactical_actions(labels: Sequence[V7Label]) -> list[str]:
    """⛔ REFUSED BY DESIGN. Exists only so a test can assert the refusal.

    Collapsing lat x lon into one class rebuilds the 5-way softmax that mixed the
    two axes — the programme's single largest known defect, the one mechanism
    that explains 0/881 accelerate, the speed-fan, and the absence of any
    longitudinal signal in selection. The labels arrive factored; keep them so.
    """
    raise NotImplementedError(
        "[v7_labels] ⛔ flattening a_tac.lat x a_tac.lon into one class is "
        "REFUSED. That is the 5-way-softmax defect (0/881 accelerate, the "
        "speed-fan) rebuilt in the labels. Use two heads: HEADS['tac_lat'] and "
        "HEADS['tac_lon'].")
