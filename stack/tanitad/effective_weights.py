"""⛔⛔ THE EFFECTIVE WEIGHT IS NOT THE ARGPARSE DEFAULT — AND NOTHING TOLD THE OPERATOR.

THE DEFECT (MEASURED 2026-09-06, two independent enumerations agreeing on
byte-identical line numbers)
---------------------------------------------------------------------------
``V6LossWeights.for_stage(stage)`` in ``scripts/train_v6_staged.py`` REWRITES
the argparse defaults per stage.  ``S-T`` zeroes **ten** terms — ``o1_ctrl``,
``o1_fact``, ``o1_scene``, ``o2_nearfield``, ``o3_masked``, ``o5_rollout``,
``o6_sigreg``, ``s1_latent``, ``w_s2_goal``, ``w_s1_multi``.  So::

    python3 scripts/train_v6_staged.py --stage S-T --w-o5 1.0 ...

**trains nothing on O5**, stamps ``w_o5: 1.0`` into the args block of
``config.json``, and prints not one word about it.  The operator asked for a
term; a layer they cannot see discarded it; the run record advertises the term
they asked for.

⭐ THIS IS THE MECHANISM BEHIND THE ALREADY-MEASURED v7f DEFECT: **42 of 138
optimizer tensors received no gradient — 5,305,667 params = 52.2 % of a declared
trainable budget** — because the objectives were weighted 0.0 and the loss
function *guards* those terms (``if w > 0.0:``), so the modules never entered the
autograd graph at all.  ⭐ ``p.grad is None`` is the sound discriminator; the
weight's own value never was.

⚠️ THE DISTINCTION THAT MAKES THIS A GUARD AND NOT A NUISANCE
--------------------------------------------------------------
**A default that is 0.0 is FINE.**  The zero defaults are deliberate — the
trainer's own comment says they exist so that *"adding the seams to the code
cannot change a run that does not ask for them"*.  Likewise a stage zeroing a
term the operator never mentioned is the **staged ladder working as designed**.

⛔ What must be refused is exactly one thing: **a value the operator ASKED FOR
being silently discarded.**  That distinction cannot be drawn from the value —
an operator may legitimately pass the default — so it is drawn from **argv**, by
:func:`explicit_dests`, which re-parses the *same* command line against a parser
whose defaults are unique sentinels.  Anything not a sentinel was typed.

⚠️ AND CHECK THE MASK BEFORE CALLING ANYTHING DEAD.  MEASURED this week: the
route head's apparent zero gradient was a **validity mask, not a dead head** —
forcing ``route_valid=True`` moved the loss 0.0 -> 0.687 and produced gradient.
A zero loss can come from (a) an argparse weight, (b) a stage override, (c) a
structural precondition, or (d) an all-invalid batch.  :class:`WeightRow`
separates all four — ``OFF_BY_*`` / ``OFF_BY_STAGE`` / ``NO_GRAPH`` /
``TRAINS_IF_MASK`` — because a table that conflates them generates false alarms,
which is worse than the silence it replaces.

⭐ THE PATTERN THIS GENERALISES.  ``refc_v3_train._check_goal_point_args``
already refuses ``--goal-point-inject`` together with ``--goal-point-w 0``,
"a head that is built, stamped, and supervised by nothing".  That is correct and
it is per-flag.  This module is the same judgement made **general** and made
**auditable**: every term, every launch, a table in ``config.json``.

⭐ FLAG DESIGN, per the ``--refuse-unreached`` / ``--allow-unreached``
precedent: a bare refusal that fires on honest launches gets deleted, so the
refusal ships WITH its acknowledgement flag.  The acknowledgement does not make
the finding disappear — it is recorded in the stamp as ``acknowledged: true``,
so a run that overrode the guard says so in its own artifacts.

⛔ THIS MODULE CHANGES NO WEIGHT.  It reads, classifies, prints and refuses.
Flipping a default would silently change the recipe every banked arm was trained
under, and the arms are the comparison basis.

Evidence class: MEASURED (ours; source of ``train_v6_staged.py`` /
``refc_v3_train.py`` at 625516 / 233816 bytes, 2026-09-06).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# the verdict lattice
# ---------------------------------------------------------------------------

#: The term is weighted, its structure exists, and it will build a graph.
TRAINS = "TRAINS"
#: Weighted and built, but the loss rides a per-batch VALIDITY MASK.  A zero
#: loss here is NOT evidence of a dead head — check the mask first.  This is the
#: route-head lesson: forcing ``route_valid=True`` moved the loss 0.0 -> 0.687.
TRAINS_IF_MASK = "TRAINS_IF_MASK"
#: Default 0.0 and the operator said nothing.  BY DESIGN — the seam is off so
#: that adding it to the code cannot change a run that does not ask for it.
OFF_BY_DEFAULT = "OFF_BY_DEFAULT"
#: The operator explicitly asked for 0.0.  Declared, recorded, fine.
OFF_BY_OPERATOR = "OFF_BY_OPERATOR"
#: A non-zero default that a later layer zeroes, with the operator SILENT.
#: This is the staged ladder working as designed.  Visible, never refused.
OFF_BY_LAYER = "OFF_BY_LAYER"
#: ⛔ REFUSAL.  The operator explicitly passed a non-zero weight and a later
#: layer zeroed it.  The launch line advertises a term that trains nothing.
DISCARDED = "DISCARDED"
#: ⛔ REFUSAL.  Effective weight > 0 but a STRUCTURAL precondition is absent —
#: the head is stamped and untrainable, or the loss term is skipped outright.
NO_GRAPH = "NO_GRAPH"

#: The two verdicts that stop a launch.
REFUSING = (DISCARDED, NO_GRAPH)

#: How the "did the operator type this?" question was answered.  ``argv`` is
#: the sound answer; ``unavailable`` means a caller did not supply the parser
#: and the command line, and it is REPORTED rather than guessed — inferring
#: explicitness from the value alone is the error this module exists to avoid.
SRC_ARGV = "argv"
SRC_UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# "did the operator actually type this flag?"
# ---------------------------------------------------------------------------

class _Sentinel:
    """Unique, non-string, non-float: argparse leaves it alone."""

    __slots__ = ("dest",)

    def __init__(self, dest: str) -> None:
        self.dest = dest

    def __repr__(self) -> str:                       # pragma: no cover
        return f"<unset {self.dest}>"


def explicit_dests(parser: argparse.ArgumentParser,
                   argv: Sequence[str] | None) -> set[str] | None:
    """The set of dests the operator EXPLICITLY supplied on ``argv``.

    ⛔ Not a heuristic and not a value comparison.  The *same* parser is
    re-parsed with every default replaced by a unique sentinel object; any dest
    that is not its sentinel afterwards was typed.  This gets abbreviations,
    ``--flag=value``, ``nargs``, ``store_true`` and ``append`` right for free,
    because argparse itself does the parsing.

    Returns ``None`` when the answer cannot be established (no parser, no argv,
    or a parser that refuses to re-parse).  ⚠️ ``None`` means UNKNOWN and is
    reported as such — it must never be silently read as "nothing was explicit",
    which is how a guard becomes cover.

    The parser is restored to its original defaults before returning, so this is
    safe to call on the live parser object.
    """
    if parser is None or argv is None:
        return None
    actions = [ac for ac in parser._actions if ac.option_strings]
    saved = [(ac, ac.default) for ac in actions]
    sentinels: dict[str, _Sentinel] = {}
    try:
        for ac in actions:
            if ac.dest in (argparse.SUPPRESS, "help"):
                continue
            s = _Sentinel(ac.dest)
            sentinels[ac.dest] = s
            ac.default = s
        # ⚠️ A parser that has already accepted this argv will accept it again.
        # If it does not (a caller passed a doctored namespace, or a mutually
        # exclusive group changed), we report UNKNOWN rather than guessing.
        ns = parser.parse_args(list(argv))
    except SystemExit:
        return None
    except Exception:                                # pragma: no cover
        return None
    finally:
        for ac, default in saved:
            ac.default = default
    out: set[str] = set()
    for dest, s in sentinels.items():
        if getattr(ns, dest, s) is not s:
            out.add(dest)
    return out


# ---------------------------------------------------------------------------
# the terms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TermSpec:
    """One loss term, as the OPERATOR sees it and as the RUN will see it.

    ``declared``  the argparse default -- what the flag's help text promises.
    ``requested`` what the operator's command line resolves to before any
                  later layer runs (== ``declared`` when they were silent).
    ``effective`` what the loss function will actually multiply by, AFTER every
                  later layer (stage override, mode gate, resolver).
    ``layer``     names the layer that changed ``requested`` -> ``effective``.
                  ⛔ Required whenever they differ: a table that says a weight
                  was discarded without naming what discarded it is a rumour.
    ``needs``     a STRUCTURAL precondition, as ``(met, "what is missing")``.
                  ``met=False`` with ``effective > 0`` is the stamped-but-
                  untrainable case (the ``--goal-point-inject --goal-point-w 0``
                  and ``--agents off --w-agent 1.0`` family).
    ``mask``      names a per-batch VALIDITY MASK the term rides, or ``None``.
                  Carrying this is what stops the table manufacturing false
                  alarms: a masked term with zero loss is not a dead head.
    """
    term: str
    flag: str
    dest: str
    declared: float
    requested: float
    effective: float
    layer: str | None = None
    needs: tuple[bool, str] | None = None
    mask: str | None = None


@dataclass(frozen=True)
class WeightRow:
    """One row of the effective-weight table: default -> layer -> effective."""
    term: str
    flag: str
    declared: float
    requested: float
    effective: float
    explicit: bool | None
    layer: str | None
    mask: str | None
    missing: str | None
    status: str
    builds_graph: bool

    def as_json(self) -> dict:
        return asdict(self)


def classify(spec: TermSpec, explicit: set[str] | None) -> WeightRow:
    """One :class:`TermSpec` -> one verdict.

    The order of the tests is the whole design:

    1. a STRUCTURAL absence beats everything -- a weighted term whose module was
       never built trains nothing no matter who asked for what (``NO_GRAPH``);
    2. then, and only then, "was this discarded behind the operator's back?"
       -- which needs ``explicit``, never the value;
    3. a term the operator zeroed themselves, or that was never on, is fine;
    4. what is left is training, and the only remaining question is whether a
       validity mask can still starve it.
    """
    was = None if explicit is None else (spec.dest in explicit)
    missing = None
    if spec.needs is not None and not spec.needs[0]:
        missing = spec.needs[1]
    zeroed = (spec.effective == 0.0 and spec.requested != 0.0)

    if spec.effective > 0.0 and missing:
        status, graph = NO_GRAPH, False
    elif zeroed and was:
        status, graph = DISCARDED, False
    elif zeroed:
        # the ladder doing its job on a default the operator never mentioned
        status, graph = OFF_BY_LAYER, False
    elif spec.effective == 0.0:
        status = OFF_BY_OPERATOR if was else OFF_BY_DEFAULT
        graph = False
    elif spec.mask:
        status, graph = TRAINS_IF_MASK, True
    else:
        status, graph = TRAINS, True
    return WeightRow(term=spec.term, flag=spec.flag, declared=spec.declared,
                     requested=spec.requested, effective=spec.effective,
                     explicit=was, layer=spec.layer, mask=spec.mask,
                     missing=missing, status=status, builds_graph=graph)


def classify_all(specs: Iterable[TermSpec],
                 explicit: set[str] | None) -> list[WeightRow]:
    return [classify(s, explicit) for s in specs]


# ---------------------------------------------------------------------------
# the refusal
# ---------------------------------------------------------------------------

def refusals(rows: Sequence[WeightRow], *, where: str) -> list[str]:
    """The problems a launch must not proceed with.  ASCII only (cp1252)."""
    out: list[str] = []
    for r in rows:
        if r.status == DISCARDED:
            out.append(
                f"{r.flag} {r.requested:g} in {where}: {r.layer} forces the "
                f"effective weight to 0.0, so the term is REMOVED FROM THE "
                f"LOSS and its modules never enter the autograd graph -- while "
                f"the launch line and config.json advertise "
                f"{r.flag} {r.requested:g}. You asked for this term; a layer "
                f"you cannot see discarded it. Drop the flag (the run is "
                f"unchanged), move to a stage that keeps it, or pass "
                f"--allow-discarded-weights to record the override.")
        elif r.status == NO_GRAPH:
            out.append(
                f"{r.flag} {r.effective:g} in {where}: {r.missing}. The weight "
                f"is stamped and the loss term is skipped -- a head that is "
                f"built, stamped, and supervised by nothing. Build the "
                f"precondition or set {r.flag} 0.")
    return out


def unknown_explicitness_warning(rows: Sequence[WeightRow]) -> str | None:
    """⚠️ Say so when the argv answer was unavailable, rather than passing.

    A guard that cannot tell explicit from default has not measured anything,
    and reporting nothing would be exactly the silence this module replaces.
    """
    if rows and rows[0].explicit is None:
        return ("effective-weight audit ran WITHOUT the command line, so "
                "'did the operator ask for this?' is UNKNOWN and no weight "
                "could be refused for being discarded. The table below is "
                "still correct about default/effective/graph.")
    return None


# ---------------------------------------------------------------------------
# the table -- ASCII ONLY. Non-ASCII in print() is fatal on cp1252.
# ---------------------------------------------------------------------------

_HDR = ("term", "flag", "default", "asked", "effective", "src", "graph",
        "status", "layer/mask/missing")


def _why(r: WeightRow) -> str:
    bits = []
    # ⚠️ Name the layer WHENEVER one acted, not only when it changed the
    # number. ``lambda_plan``'s argparse default is ``None`` and
    # ``STAGE_LAMBDA_PLAN[stage]`` supplies the value; a table that hid that
    # because default == effective would be telling the operator the flag's
    # own default was in force, which is the misreading this module exists to
    # stop -- one layer down.
    if r.layer:
        bits.append(f"layer={r.layer}")
    if r.mask:
        bits.append(f"mask={r.mask}")
    if r.missing:
        bits.append(f"missing={r.missing}")
    return "; ".join(bits)


def render_table(rows: Sequence[WeightRow], *, where: str,
                 tag: str = "v6") -> str:
    """The launch-time table.  ⛔ A run record that does not carry its effective
    weights cannot be audited afterwards, and three arm-substitutions have
    already been found in this programme."""
    body = []
    for r in rows:
        src = ("op" if r.explicit else "dflt") if r.explicit is not None \
            else "?"
        body.append((r.term, r.flag, f"{r.declared:g}", f"{r.requested:g}",
                     f"{r.effective:g}", src, "yes" if r.builds_graph else "no",
                     r.status, _why(r)))
    cols = [max(len(str(x[i])) for x in [_HDR] + body)
            for i in range(len(_HDR))]
    def line(vals):
        return "  ".join(str(v).ljust(cols[i]) for i, v in enumerate(vals))
    out = [f"[{tag}] EFFECTIVE WEIGHTS ({where}) -- default -> layer -> "
           f"effective -> builds a graph?",
           f"[{tag}] " + line(_HDR),
           f"[{tag}] " + "  ".join("-" * c for c in cols)]
    for b in body:
        out.append(f"[{tag}] " + line(b))
    n_graph = sum(1 for r in rows if r.builds_graph)
    out.append(f"[{tag}] {n_graph}/{len(rows)} terms build a graph. A term "
               f"with graph=no contributes EXACTLY zero gradient; "
               f"'p.grad is None' is the sound check, never the weight value.")
    return "\n".join(out)


def stamp(rows: Sequence[WeightRow], *, where: str, explicit_source: str,
          acknowledged: bool = False) -> dict:
    """The ``config.json`` block.  A GATE ROW CARRIES ITS ARM."""
    return {
        "_what": "argparse default -> later layers -> effective weight, and "
                 "whether the term therefore builds an autograd graph at all",
        "_why": "V6LossWeights.for_stage(stage) rewrites the argparse "
                "defaults per stage; the effective weight is not the default. "
                "MEASURED: 42/138 optimizer tensors took no gradient "
                "(5,305,667 params = 52.2 % of a declared trainable budget) "
                "because zero-weighted terms are guarded out of the loss.",
        "_discriminator": "p.grad is None -- never the weight's value",
        "where": where,
        "explicit_source": explicit_source,
        "acknowledged_discarded": bool(acknowledged),
        "n_terms": len(rows),
        "n_builds_graph": sum(1 for r in rows if r.builds_graph),
        "terms": [r.as_json() for r in rows],
        "_evidence_class": "MEASURED (ours; this run's own configuration)",
    }
