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

# ⚠️ the emitter raises the SAME exception type the model-side guard uses, so
# a missing nav is one error class end to end rather than two that must be
# caught separately at the data boundary and in the forward.
from tanitad.models.nav_conditioning import NavTokenMissing
from tanitad.models.vocab_v7 import (NAV_COMMAND_TOKENS as NAV_TOKENS,
                                     NOT_YET_EXTRACTABLE, STRATEGIC_ACTION_TOKENS_V7,
                                     STRATEGIC_GOAL_TOKENS_V7,
                                     TACTICAL_GOAL_EXCLUSIVE,
                                     TACTICAL_GOAL_NEEDS_PERCEPTION,
                                     TACTICAL_GOAL_TOKENS_V7,
                                     TACTICAL_LAT_ACTIONS_V7,
                                     TACTICAL_LON_ACTIONS_V7)

__all__ = ["HEADS", "IGNORE_W", "LabelManifest", "TAC_GOAL_HEAD",
           "TAC_GOAL_TOKENS", "TacGoalEmitter", "V7Label",
           "assert_mask_matches_presence",
           "class_weights", "effective_mask", "flatten_tactical_actions",
           "goal_pos_weight", "goal_supervision_census", "head_mask",
           "is_oracle_nav", "load_v7_labels", "NavEmitter",
           "NavTokenMissing",
           "oracle_nav", "tactical_goal_targets"]

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
    #: ⭐⭐ THE 22-TOKEN TACTICAL GOAL SET — minted on every clip since
    #: v7.0 and, until 2026-09-06, read into ``audit`` only and therefore
    #: trained by nothing. It is MULTI-LABEL (2–7 tokens/record, mean
    #: 2.751 MEASURED on the v7.2 train blob), so it is a SET, never a
    #: class. ⚠️ It is a TARGET. Feeding it back as a model INPUT would
    #: be the situation-classifier back door the PI ruled out on
    #: 2026-08-03 — state what a goal input is computed from, and this
    #: is computed from the label.
    tac_goals: frozenset[str] = field(default_factory=frozenset)
    #: per-token ``provenance``/``time_basis``/``state`` as EMITTED. The
    #: negative policy is derived from this rather than from a constant.
    tac_goal_meta: dict[str, Any] = field(default_factory=dict)
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
            tac_goals=frozenset(g_tac.get("goals") or {}),
            tac_goal_meta={k: v for k, v in
                           (g_tac.get("goals") or {}).items()
                           if isinstance(v, dict)},
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

    # ⭐⭐ THE NEGATIVE POLICY IS DERIVED FROM THE BLOB, NOT DECLARED.
    # A token whose annotations are ALL geometry-emitted is exhaustively
    # labelled, so its absence is a true negative. A token with ANY
    # ``vlm-cot`` annotation is a per-clip TEXT claim, and its absence
    # means only that the caption stayed silent. ⚠️ MEASURED on the v7.2
    # train blob, the frozen ``TACTICAL_GOAL_NEEDS_PERCEPTION``
    # DECLARATION and the DATA disagree on four tokens — ``YIELD`` (609),
    # ``EVADE_IN_CORRIDOR`` (238), ``LANE_CHANGE_L`` (23), ``LANE_CHANGE_R``
    # (15) are CoT-sourced in the blob while absent from the declared set —
    # so trusting the declaration would supervise 4,534 unknowable
    # lane-change negatives as true.
    global _MEASURED_GEOMETRY_TOKENS
    _cot_backed = {t for lb in labels for t, m in lb.tac_goal_meta.items()
                   if m.get("provenance") == "vlm-cot"}
    _MEASURED_GEOMETRY_TOKENS = frozenset(
        t for t in TACTICAL_GOAL_TOKENS_V7
        if t not in _cot_backed and t not in TACTICAL_GOAL_NEEDS_PERCEPTION)

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


#: torch ``cross_entropy``'s ignore_index — the no-label marker for windows a
#: clip's single record does not describe.
IGNORE_ID = -100


def window_in_band(label: "V7Label", t_now_s: float) -> bool:
    """Is the window whose NOW is ``t_now_s`` described by this clip's record?

    ⭐ ONE IMPLEMENTATION OF THE WINDOW RULE. It was derived for the refav1
    loader on 2026-09-01 and is lifted here rather than copied, because a
    second copy is a second rule and the programme has already paid for that
    (the vocabulary-duplication precedent).

    THE RULE, and why: a clip carries ONE record anchored at ``t0_s`` whose
    tactical band describes raw seconds ``[t0+lo, t0+hi]``. A window at
    ``t_now`` has its own forward tactical band ``[t_now+lo, t_now+hi]``. The
    two overlap by at least half their width exactly when
    ``|t_now - t0| <= (hi - lo) / 2`` — so that is the admission test, and the
    half-width is computed FROM THE RECORD'S OWN BANDS, never hardcoded (a
    derived constant that silently changes when its input changes is the
    `HORIZON` trap). At the shipped bands this is +-2.0 s, which reproduces
    the v6 trainer's ``S2WindowSupervision._in_band`` default
    ``valid_window_s (-2, 2)`` as a DERIVATION instead of a coincidence.
    """
    lo, hi = label.bands["tactical_s"]
    return abs(float(t_now_s) - float(label.t0_s)) <= (float(hi) - float(lo)) / 2.0


def tactical_class_ids(label: "V7Label", t_now_s: float
                       ) -> tuple[int, int]:
    """``(lat_id, lon_id)`` in the v7 vocabulary, or ``(-100, -100)`` when this
    window is outside the record's band. ⛔ Never clamps to a 'neutral' class —
    an unlabelled window must not train a wrong one."""
    if not window_in_band(label, t_now_s):
        return IGNORE_ID, IGNORE_ID
    lat = HEADS["tac_lat"].index(label.tac_lat)
    lon = HEADS["tac_lon"].index(label.tac_lon)
    return lat, lon


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




# ---------------------------------------------------------------------------
# TACTICAL GOAL SET — the 22-token vocabulary that was minted and never trained
# ---------------------------------------------------------------------------
#: ⭐ THE GAP THIS CLOSES (audit 2026-09-06, ``AUDIT_RESULT.json``). The v7
#: emitter mints a 22-token tactical goal set on EVERY one of 4,572 training
#: clips — including a ground-truth traffic-light COLOUR — and until now no head
#: was sized on it, no loss referenced it and no gradient could reach it. It is
#: a STRUCTURAL absence, not a zero weight, and it is the mechanism behind two
#: PI observations on the refcv4b video: *the model never brakes for a red
#: light*, and *lane changes never activate*.
#:
#: ⛔ THIS IS A SEPARATE SURFACE FROM :data:`HEADS`, ON PURPOSE. ``HEADS`` is the
#: four SOFTMAX action/strategy heads; a clip carries EXACTLY ONE class of each.
#: The goal set is **MULTI-LABEL** — MEASURED on the v7.2 train blob, 2 to 7
#: tokens per record, mean 2.751 — so it is 22 independent sigmoids under BCE,
#: never a softmax. Merging it into ``HEADS`` would make a set look like a
#: choice, which is the 5-way-softmax defect wearing a new costume.
TAC_GOAL_HEAD = "tac_goal"
TAC_GOAL_TOKENS: tuple[str, ...] = tuple(TACTICAL_GOAL_TOKENS_V7)

#: Per-element target marker: this (record, token) cell carries NO evidence and
#: must not train either way. Mirrors :data:`IGNORE_ID`'s role for the softmax
#: heads — an unlabelled cell must never train a wrong one.
IGNORE_W = 0.0


#: ``token -> the tokens whose PRESENCE makes it FALSE``, built from the
#: frozen exclusion table. ⭐ This is the ENTAILED-NEGATIVE source and it
#: is what makes the traffic-light and lane-change tokens trainable at all:
#: they are 100 % CoT-backed, so the provenance policy alone leaves them
#: with ZERO supervised negatives and a BCE head that can only learn
#: "always 1". ⚠️ The entailment is the emitter's OWN invariant — every
#: record is validated against this table by ``V7.validate_goal_set`` — so
#: reading a negative off it assumes nothing about caption completeness.
_EXCLUDED_BY: dict[str, frozenset[str]] = {
    t: frozenset(b for a, b in
                 [(x, y) for x, y in TACTICAL_GOAL_EXCLUSIVE]
                 + [(y, x) for x, y in TACTICAL_GOAL_EXCLUSIVE]
                 if a == t)
    for t in TACTICAL_GOAL_TOKENS_V7}


def entailed_false(present) -> frozenset[str]:
    """Tokens the present set makes FALSE by the frozen exclusion table."""
    out: set[str] = set()
    for t in present:
        out |= _EXCLUDED_BY.get(t, frozenset())
    return frozenset(out) - set(present)


def _goal_provenance(label: "V7Label", token: str) -> str | None:
    """The MEASURED provenance of one goal annotation, or None if absent."""
    rec = (label.tac_goal_meta or {}).get(token)
    return (rec or {}).get("provenance") if isinstance(rec, dict) else None


def tactical_goal_targets(label: "V7Label", t_now_s: float, *,
                          negatives: str = "measured",
                          ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(y, w)`` for one window: 22 targets and 22 per-class loss weights.

    ``y[i]`` is 1.0 when token ``i`` is in this clip's goal set, else 0.0.
    ``w[i]`` is 1.0 when that cell carries EVIDENCE and :data:`IGNORE_W` when it
    does not. Outside the record's band every ``w`` is :data:`IGNORE_W` — the
    same rule :func:`tactical_class_ids` applies, derived from the record's own
    bands rather than hardcoded.

    ⛔⛔ WHY ABSENCE IS NOT A NEGATIVE FOR EVERY TOKEN — MEASURED, and it decides
    whether this head learns a fact or a labelling artefact. The goal set has two
    provenances in the released blob and they differ in what an ABSENT token
    means:

    ``geometry`` (9,106 annotations over 9 tokens)
        The ego-geometry emitter is EXHAUSTIVE — it inspects every clip and
        writes the token when the geometry holds. Absence therefore IS a
        negative, and is supervised as one.

    ``vlm-cot`` (3,472 annotations over 15 tokens — every traffic-light and
    every lane-change token)
        The token exists only because a per-clip TEXT happened to mention it.
        Absence means *"the caption did not say so"*, which is not *"it did not
        happen"*. MEASURED on the same blob: **692 clips carry a VISIBLE
        traffic-light box and no traffic-light token**, and of the 4,572 clips
        only **998 were asked the traffic-light grounding question at all** —
        the other 3,574 were asked about a pedestrian, a vehicle, or nothing, so
        their ``scene.traffic_light_visible == False`` is NOT-PROBED, not
        absent. Supervising those 3,574 cells as negatives would teach the head
        that ~78 % of the corpus has no traffic light, on no evidence.

    ⇒ ``negatives="measured"`` (the default) reads the provenance PER TOKEN FROM
    THE LOADED SPLIT, exactly as :func:`effective_mask` reads presence, and
    supervises a negative only for tokens this blob actually emits from
    geometry. ⚠️ It deliberately does NOT use
    :data:`~tanitad.models.vocab_v7.TACTICAL_GOAL_NEEDS_PERCEPTION`: MEASURED,
    the declared set and the blob DISAGREE on four tokens — ``YIELD`` (609),
    ``EVADE_IN_CORRIDOR`` (238), ``LANE_CHANGE_L`` (23) and ``LANE_CHANGE_R``
    (15) are all CoT-sourced in the data while absent from the frozen
    perception set. Trusting the declaration would supervise 4,534 unknowable
    lane-change negatives as true.

    ``negatives="all"`` supervises every absent cell as a negative. It is the
    cheap arm and it is offered only so the two can be COMPARED; it is not the
    default, and an arm using it must say so.

    ``negatives="geometry"`` uses the frozen ``TACTICAL_GOAL_NEEDS_PERCEPTION``
    declaration instead of the blob. Kept so the declaration/data divergence
    above is measurable rather than argued.
    """
    n = len(TAC_GOAL_TOKENS)
    present = label.tac_goals or frozenset()
    if not window_in_band(label, t_now_s):
        return (0.0,) * n, (IGNORE_W,) * n
    entailed = entailed_false(present)
    y, w = [], []
    for tok in TAC_GOAL_TOKENS:
        hit = tok in present
        y.append(1.0 if hit else 0.0)
        if hit:
            w.append(1.0)                       # a POSITIVE is always evidence
        elif tok in entailed:
            # ⭐ ENTAILED FALSE by a token that IS present. Admissible for
            # every provenance, because it is a consequence of the label
            # rather than a claim about what the caption omitted.
            w.append(1.0)
        elif negatives == "all":
            w.append(1.0)
        elif negatives == "geometry":
            w.append(IGNORE_W if tok in TACTICAL_GOAL_NEEDS_PERCEPTION else 1.0)
        elif negatives == "measured":
            w.append(1.0 if tok in _MEASURED_GEOMETRY_TOKENS else IGNORE_W)
        else:
            raise ValueError(
                f"[v7_labels] unknown negatives policy {negatives!r}; "
                f"expected 'measured' | 'geometry' | 'all'")
    return tuple(y), tuple(w)


#: Filled by :func:`load_v7_labels` from the blob it actually read. ⛔ Module
#: state is normally a smell; here it is the point — the negative policy must be
#: a property of THE LOADED SPLIT, never of a constant that outlives it. It
#: starts EMPTY so a consumer that never loaded a blob supervises NOTHING rather
#: than silently supervising everything.
_MEASURED_GEOMETRY_TOKENS: frozenset[str] = frozenset()


def goal_supervision_census(labels: Sequence["V7Label"]) -> dict[str, Any]:
    """What the goal head will actually be trained on. Goes into ``config.json``.

    ⛔ Reports POSITIVES, NEGATIVES and IGNORED per token. A head reported only
    by its positive count hides the class-imbalance trap: ``LANE_CHANGE_R`` at
    15 of 4,572 is 0.33 %, and any unweighted objective predicts it never while
    scoring 99.67 % 'accuracy'.
    """
    out: dict[str, Any] = {}
    for i, tok in enumerate(TAC_GOAL_TOKENS):
        pos = neg = ign = 0
        for lb in labels:
            _, w = tactical_goal_targets(lb, lb.t0_s)
            if tok in (lb.tac_goals or frozenset()):
                pos += 1
            elif w[i] > 0.0:
                neg += 1
            else:
                ign += 1
        out[tok] = {"pos": pos, "neg": neg, "ignored": ign,
                    "prevalence": pos / max(len(labels), 1),
                    "provenance": sorted({p for lb in labels
                                          if (p := _goal_provenance(lb, tok))}),
                    "supervised_negative": tok in _MEASURED_GEOMETRY_TOKENS,
                    "entailed_false_by": sorted(_EXCLUDED_BY.get(tok, ()))}
    return out


def goal_pos_weight(labels: Sequence["V7Label"], *,
                    cap: float = 50.0) -> tuple[float, ...]:
    """Per-class ``pos_weight`` for ``BCEWithLogitsLoss``, FROM THE SPLIT.

    ``pos_weight[i] = n_neg / n_pos`` over the cells this policy actually
    supervises, capped at ``cap``.

    ⛔ NEVER HARDCODE THESE. ``HOLD_MAIN_ROAD`` is 52.3 % of ``str_action`` in
    this blob and a hardcoded weight silently mis-weights the next one — the
    derived-constant trap that moved ``HORIZON`` 7 -> 8 and turned a
    reproduction into a different experiment.

    ⚠️ THE CAP IS NOT COSMETIC. Uncapped, ``LANE_CHANGE_R`` (15 pos) would take
    ``pos_weight`` ~304 and a single positive would dominate the batch gradient.
    ⛔ A class with ZERO supervised positives gets 0.0 and MUST also be masked —
    a pos_weight on an empty class is a weight on nothing.
    """
    n = len(TAC_GOAL_TOKENS)
    pos = [0] * n
    neg = [0] * n
    for lb in labels:
        y, w = tactical_goal_targets(lb, lb.t0_s)
        for i in range(n):
            if w[i] <= 0.0:
                continue
            if y[i] > 0.5:
                pos[i] += 1
            else:
                neg[i] += 1
    return tuple(0.0 if pos[i] == 0 else min(cap, neg[i] / pos[i])
                 for i in range(n))


class TacGoalEmitter:
    """``ep_idx`` -> per-window ``(y, w)`` targets. The join, mirroring
    :class:`NavEmitter` deliberately.

    ⛔ It RAISES on an unmapped episode for the same reason ``NavEmitter`` does:
    a default would attach one clip's goal set to another clip's windows and
    train on a plausible wrong signal, which is worse than a crash.

    ⭐ It is a TARGET emitter, so it needs no oracle stamp — ``g_tac.goals`` is a
    label, not an input. ⚠️ That is exactly why it must never be read back into
    the model's input path: the PI's 2026-08-03 ruling is that a goal INPUT may
    not carry the situation classifier's output, and by the same argument a goal
    TARGET may not re-enter as a feature.
    """

    def __init__(self, labels: Sequence["V7Label"],
                 clip_id_by_ep_idx: dict[int, str], *,
                 negatives: str = "measured"):
        if negatives not in ("measured", "geometry", "all"):
            raise ValueError(f"[tac_goal] unknown negatives {negatives!r}")
        self.negatives = negatives
        self.clip_id_by_ep_idx = dict(clip_id_by_ep_idx)
        self._by_clip = {x.clip_id: x for x in labels}
        self.n_tokens = len(TAC_GOAL_TOKENS)

    def __call__(self, ep_idx, t_now_s=None, dt: float = 0.1):
        """``(y [B, 22] float32, w [B, 22] float32)``.

        ``t_now_s`` is the window's NOW in RAW CLIP SECONDS. Pass it, or every
        window is scored at the record's own anchor and the band test becomes a
        tautology.
        """
        import torch
        idx = [int(i) for i in (ep_idx.tolist() if hasattr(ep_idx, "tolist")
                                else ep_idx)]
        if t_now_s is None:
            times = [None] * len(idx)
        else:
            times = [float(x) for x in (t_now_s.tolist()
                     if hasattr(t_now_s, "tolist") else t_now_s)]
        ys, ws = [], []
        for e, t in zip(idx, times):
            clip = self.clip_id_by_ep_idx.get(e)
            if clip is None:
                raise NavTokenMissing(
                    f"[tac_goal] ⛔ ep_idx {e} has no clip_id mapping. Refusing "
                    f"rather than defaulting: a default would attach another "
                    f"clip's goal set to this window.")
            rec = self._by_clip.get(clip)
            if rec is None:
                raise NavTokenMissing(
                    f"[tac_goal] ⛔ clip {clip!r} (ep_idx {e}) has no label "
                    f"record.")
            y, w = tactical_goal_targets(
                rec, rec.t0_s if t is None else t, negatives=self.negatives)
            ys.append(list(y))
            ws.append(list(w))
        return (torch.tensor(ys, dtype=torch.float32),
                torch.tensor(ws, dtype=torch.float32))

    def provenance(self) -> dict[str, Any]:
        """Goes into ``config.json`` beside the label manifest."""
        return {"tac_goal_negatives": self.negatives,
                "tac_goal_tokens": list(TAC_GOAL_TOKENS),
                "tac_goal_n_tokens": self.n_tokens,
                "n_clips_mapped": len(self.clip_id_by_ep_idx),
                "n_label_records": len(self._by_clip),
                "supervised_negative_tokens":
                    sorted(_MEASURED_GEOMETRY_TOKENS)}


# ---------------------------------------------------------------------------
# NAV EMISSION — the join from per-clip labels to per-window model inputs
# ---------------------------------------------------------------------------
class NavEmitter:
    """Turn ``ep_idx`` into the ``(nav_token, nav_args)`` the model consumes.

    PI DIRECTIVE 2026-08-30: the nav command conditions all three layers. The
    labels are **per clip**; the model needs them **per window**. This is that
    join, and it lives here rather than in the dataset because ``ep_idx`` indexes
    the cache's sorted ``<clip_id>.v2ep.pt`` order and a consumer is expected to
    resolve the clip id "from the cache listing rather than from anything stored"
    in the batch (``train_flagship4b.py:149-151``).

    ⭐ IT CANNOT EMIT WITHOUT THE ORACLE STAMP. Every arg goes through
    :func:`oracle_nav`, which checks the MANIFEST — so a nav-conditioned arm
    physically cannot train unless its manifest carries ``allow_oracle_nav=True``,
    and that flag is recorded in ``config.json``. The gate is not a convention
    someone must remember; it is the only code path to the value.

    ⚠️ THE ARG SEMANTICS ARE BEHIND ONE FUNCTION AND THE CHOICE IS OPEN.
    ``distance_m``/``time_s`` are measured **to the nav point from t0**, so a
    window at offset t is closer to it than t0 was. Two readings:

      ``t0_constant``  every window in a clip carries the t0 values. Simple, and
                       wrong in an interesting way: the window 15 s later still
                       says "turn right in 106.5 m" when it is 8 m away.
      ``decremented``  subtract the ego's travelled distance / elapsed time. What
                       a real nav system does — ⚠️ but it makes the args a
                       FUNCTION OF EGO STATE, so it must be checked against the
                       goal/situation information-disjointness rule before it
                       ships (PI 2026-08-03: state what the goal is computed from).

    ⛔ The DataFlyWheel owns that specification. Until it lands, the default is
    ``t0_constant`` and the choice is RECORDED in :meth:`provenance` — swapping
    it is a one-line change to :meth:`_args_for_window`, by design.
    """

    def __init__(self, labels: Sequence[V7Label], manifest: LabelManifest,
                 clip_id_by_ep_idx: dict[int, str], *,
                 semantics: str = "t0_constant"):
        if semantics not in ("t0_constant", "decremented"):
            raise ValueError(f"[nav] unknown semantics {semantics!r}")
        self.manifest = manifest
        self.semantics = semantics
        self.clip_id_by_ep_idx = dict(clip_id_by_ep_idx)
        self._by_clip = {x.clip_id: x for x in labels}
        self._tok2id = {t: i for i, t in enumerate(NAV_TOKENS)}

    def _args_for_window(self, nav: dict[str, Any], t_offset_s: float
                         ) -> tuple[float, float]:
        """⭐ THE ONE FUNCTION THE SEMANTICS LIVE IN (see the class docstring)."""
        args = nav.get("args") or {}
        d, t = float(args.get("distance_m", 0.0)), float(args.get("time_s", 0.0))
        if self.semantics == "t0_constant":
            return d, t
        # ``decremented`` — deliberately NOT the default; see the disjointness note
        return d, max(t - float(t_offset_s), 0.0)

    def __call__(self, ep_idx, t_last=None, dt: float = 0.1):
        """``ep_idx`` [B] -> ``(token_id [B] int64, args [B, 2] float32)``.

        ⛔ RAISES on an unmapped episode. A default here would silently feed one
        clip's route to another's windows — worse than a crash, because the model
        would train on a plausible wrong signal.
        """
        import torch
        idx = [int(i) for i in (ep_idx.tolist() if hasattr(ep_idx, "tolist")
                                else ep_idx)]
        offs = ([float(x) * dt for x in (t_last.tolist()
                 if hasattr(t_last, "tolist") else t_last)]
                if t_last is not None else [0.0] * len(idx))
        ids, args = [], []
        for e, off in zip(idx, offs):
            clip = self.clip_id_by_ep_idx.get(e)
            if clip is None:
                raise NavTokenMissing(
                    f"[nav] ⛔ ep_idx {e} has no clip_id mapping. Refusing rather "
                    f"than defaulting: a default would feed ANOTHER clip's route "
                    f"to this window and train on a plausible wrong signal.")
            rec = self._by_clip.get(clip)
            if rec is None:
                raise NavTokenMissing(
                    f"[nav] ⛔ clip {clip!r} (ep_idx {e}) has no label record.")
            nav = oracle_nav(rec, self.manifest)      # ⭐ the gate, not a lookup
            tok = nav.get("token")
            if tok not in self._tok2id:
                raise NavTokenMissing(f"[nav] ⛔ unknown nav token {tok!r}")
            ids.append(self._tok2id[tok])
            args.append(list(self._args_for_window(nav, off)))
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(args, dtype=torch.float32))

    def provenance(self) -> dict[str, Any]:
        """Goes into ``config.json`` beside the label manifest."""
        return {"nav_arg_semantics": self.semantics,
                "n_clips_mapped": len(self.clip_id_by_ep_idx),
                "n_label_records": len(self._by_clip),
                **self.manifest.to_dict()}
