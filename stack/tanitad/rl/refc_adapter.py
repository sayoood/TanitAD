"""``tanitad.rl.refc_adapter`` — the RL binding for the WHOLE refc-v3 model family.

``refcv3_adapter`` binds refcv3 and is kept verbatim (two tests import it). This module
generalises that binding to **refcv4b and refcv5**, which are *not* new model classes —
they are config variants of the same :class:`~tanitad.refs.refc_v3.RefCV3Model`
(`refc_v3.py:628`, `RefCV3Config:319`; `refcv5_preflight.py:283` imports the same class).

⛔⛔ THE BUG THIS MODULE EXISTS TO REMOVE — A SILENTLY DIFFERENT POLICY.
``refcv3_adapter.make_refcv3_sample_fn`` calls::

    model(frames, nav_cmd, v0, steps=..., lan=...)

which is **five of the eight** parameters ``RefCV3Model.forward`` accepts (`refc_v3.py:986-991`).
refcv4b is trained with ``--ego-state-inject --ego-dropout 0.5`` (`MODEL_REGISTRY.md` §4.6),
so its policy is conditioned on ``ego_state [B, 5]``. The forward **fails loud when
``ego_state`` is SUPPLIED to a build that would drop it** (`refc_v3.py:1000-1001`) — but it
is silent in the other direction: **omitting** ``ego_state`` on a build trained with it
returns a perfectly well-formed fan from a *differently conditioned* policy, and nothing
raises. RL would then post-train a policy that is not the deployed one, and the result
table would read as a lever effect.

⇒ :func:`assert_conditioning` makes the omission LOUD. It reads the requirement from the
**model's own config**, never from a caller-supplied name, and refuses a batch that drops a
channel the checkpoint was trained with. That is the positive assertion CLAUDE.md's mount
rules demand, applied to conditioning: *absence of a key is not evidence the model does not
want it.*

⚠️ ``ego_dropout 0.5`` at TRAIN time does not make ``ego_state`` optional at RL time. Dropout
makes the policy robust to the channel being missing; it does not make "always missing" the
same distribution the checkpoint was fitted under, and the RL stage must sample the policy it
intends to ship.

Everything downstream of the ``sample_fn`` seam is model-agnostic and unchanged:
``sample_offsets`` and ``gt_context`` are re-exported from ``refcv3_adapter`` verbatim.

Tier: T0 training-side. Evidence class of the claims in this docstring: MEASURED
(`MODEL_REGISTRY.md` §4.6 argv) / PUBLISHED-CODE (`refc_v3.py` line cites).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import PostTrainConfig
# Re-exported verbatim: these are model-agnostic and must not fork.
from .refcv3_adapter import gt_context, sample_offsets

__all__ = ["FORWARD_KEYS", "ConditioningError", "conditioning_requirements",
           "assert_conditioning", "forward_kwargs", "make_refc_sample_fn",
           "sample_offsets", "gt_context", "ChannelRequirement",
           "RequirementDeclarationError", "CHANNEL_REQUIREMENTS",
           "requirement_report"]

#: Every conditioning channel an RL rollout MAY and MUST be given, in signature
#: order. ⭐ THAT IS NOT "every channel the forward accepts": the live forward takes
#: TWELVE (`refc_v3.py:1361-1373`) and five of them are declared must-not-be-plumbed
#: by the seams that own them. ``frames`` and ``steps`` are positional/explicit.
#:
#:     required = signature - tanitad.channel_admissibility.excluded_channels()
#:
#: ⚠️ ``agent_gt`` WAS MISSING FROM THIS TUPLE, and the stale citation it carried
#: (`:986-991` — exactly one line short of the signature's last parameter) is the
#: fingerprint: the tuple was written against a forward that ended at ``withheld_speed``,
#: and the refcv5 WP-6 agent seam appended ``agent_gt`` afterwards. Because
#: :func:`forward_kwargs` builds its kwargs by iterating THIS tuple, a channel absent
#: from it is **never passed at all** — no error, no warning — so an ``--agents oracle``
#: build driven through this adapter tripped the model's own guard (*"no agent_gt
#: reached the forward"*) even when the batch carried it, and the message accused the
#: caller of an omission the ADAPTER had made. Same defect class the module docstring
#: above describes for ``ego_state``, one channel later.
#:
#: ⛔⛔ AND THE OPPOSITE DEFECT IS WORSE, WHICH IS WHY THE TUPLE IS SHORT ON PURPOSE.
#: The first version of the drift test told the reader to fix the next gap by adding
#: ``gp_point``/``gp_valid`` here. That advice would have MANUFACTURED A LABEL LEAK:
#: the only supplier a batch has for a goal point is the ego's own future pose. The
#: goal-point stream overruled it, and the same question — *where would an RL rollout
#: GET this value?* — has since excluded three more:
#:
#:     gp_point / gp_valid      the goal point IS the label            PERMANENT
#:     nav_args                 ships a ``time_s`` slot that is the
#:                              ego's future speed profile inverted    as BUILT
#:     v_max_ms / v_max_valid   the PI ruled the channel legitimate,
#:                              but this corpus's value is the ego's
#:                              own realised future speed              this CORPUS
#:
#: ⭐ Each of those is declared IN THE SEAM THAT OWNS IT, with its reason, its
#: evidence and its UNBLOCK CONDITION — ``tanitad.refs.goal_point``,
#: ``tanitad.models.nav_conditioning``, ``tanitad.refs.max_speed_input``. Read the
#: reasons there; two of the three exclusions are temporary and say what lifts them.
#:
#: ⛔ KEEP THIS DERIVED FROM THE SIGNATURE MINUS THE DECLARATIONS, NEVER
#: HAND-MAINTAINED. ``test_rl_forward_keys_cover_signature.py`` compares it against
#: ``inspect.signature(RefCV3Model.forward)`` and the seams' union at test time, and
#: ``tanitad.rl.channel_guard`` repeats the comparison at LAUNCH time — in both
#: directions, because a hand-written list rots the next time the forward grows a
#: channel, and because the guard that only checked one direction spent a day
#: refusing correct launches while recommending the leak.
FORWARD_KEYS = ("nav_cmd", "v0", "lan", "nav_known", "ego_state", "withheld_speed",
                "agent_gt")

#: ⛔⛔ WHY THIS IS A DECLARATION AND NO LONGER A ``dict[str, str | None]``.
#:
#: The map this replaced read::
#:
#:     req[key] = bool(getattr(cfg, flag, False)) if flag else False
#:
#: — so **every entry whose flag was ``None`` evaluated False and was never checked.**
#: Six of the seven channels were ``None``. The map was, in effect, a list of the
#: channels it did NOT check, wearing a guard's clothes: ``assert_conditioning`` is
#: wired (`:make_refc_sample_fn`) and has been all along, so this was never a wiring
#: gap — it was a COVERAGE gap, and the two look identical from the call site.
#:
#: ⭐ Two of the six carried a real reason (``nav_cmd``, ``agent_gt``). Four carried
#: nothing, or carried *"no config flag exists — plumbed, not asserted"*, which
#: records a MECHANISM and not a JUDGEMENT. And for three of those four the premise
#: was simply FALSE — a declaring field does exist; it is **NESTED**, and the flat
#: ``getattr(cfg, flag, False)`` above could never have reached it:
#:
#:     v0           cfg.core.anchors.v0_conditioned   (`refc.py:383`)   2 levels deep
#:     lan          cfg.core.graft_lan                (`refc.py:693`)   1 level deep
#:     nav_known    cfg.core.nav_known_channel        (`refc.py:836`)   1 level deep
#:
#: The old note looked for the NAMES ``nav_known_inject`` / ``withheld_speed_inject``,
#: did not find them, and concluded no field existed. It was hunting a naming
#: convention, not a fact. ⇒ the predicate is now a **dotted path** read by
#: :func:`_read_predicate`, which **RAISES** on a path that does not resolve instead
#: of returning False. A predicate that cannot be read is a check that can never
#: fire, and that failure must be loud at the first batch rather than silent forever.
#:
#: ⭐ SYMMETRY WITH ``tanitad.channel_admissibility``. That module made an EXCLUSION
#: unconstructible without ``reason``/``unblock``/``evidence``/``owner``. An
#: **unasserted required channel** is the same object one level down — a decision
#: somebody made that the next reader cannot see — so it earns the same discipline.
#: ⛔ "No config flag exists" is not a reason; it is a fact about the code. The
#: ``agent_gt`` entry below is the proof of the difference: it says what guards the
#: channel INSTEAD, and where.
_MIN_REASON = 40
_MIN_UNBLOCK = 20
_MIN_EVIDENCE = 8
_PLACEHOLDERS = frozenset({"tbd", "todo", "n/a", "na", "none", "-", "?", "unknown",
                           "see above", "fixme", "xxx", "no config flag exists"})
_WHY = {
    "owner": "who made this call",
    "reason": "what SILENTLY changes when the channel is absent (asserted), or why "
              "no assertion is possible or desirable (unasserted)",
    "unblock": "what would make the channel assertable",
    "evidence": "the evidence class and its citation",
}


class RequirementDeclarationError(ValueError):
    """A channel's requirement declaration is malformed."""


@dataclass(frozen=True)
class ChannelRequirement:
    """How :func:`assert_conditioning` decides whether one channel is REQUIRED.

    ``channel``    the forward kwarg name, exactly as ``RefCV3Model.forward`` spells it.
    ``owner``      who made the call.
    ``reason``     ASSERTED: what silently changes when the channel is absent — the
                   thing the refusal is protecting. UNASSERTED: why nothing here can
                   or should assert it. ⛔ Not "no flag exists".
    ``evidence``   evidence class + citation, per the operating standard.
    ``predicates`` dotted paths on ``model.cfg``; the channel is required when **any**
                   reads truthy. Empty ⇒ NOT ASSERTED, and then ``unblock`` is
                   mandatory.
    ``unblock``    what would make it assertable. Required exactly when ``predicates``
                   is empty, because "nothing could assert this" and "nobody wrote a
                   predicate" must not look the same — which is the whole defect the
                   ``None`` map had.
    """

    channel: str
    owner: str
    reason: str
    evidence: str
    predicates: tuple[str, ...] = ()
    unblock: str = ""

    def __post_init__(self) -> None:
        if not (isinstance(self.channel, str) and self.channel.isidentifier()):
            raise RequirementDeclarationError(
                f"channel must be the forward kwarg's identifier, got "
                f"{self.channel!r}; a non-identifier can never match a signature "
                f"parameter, so the requirement would cover nothing while looking "
                f"like it covers something.")
        checks = [("owner", self.owner, 3), ("reason", self.reason, _MIN_REASON),
                  ("evidence", self.evidence, _MIN_EVIDENCE)]
        if not self.predicates:
            checks.append(("unblock", self.unblock, _MIN_UNBLOCK))
        for name, value, floor in checks:
            if not isinstance(value, str):
                raise RequirementDeclarationError(
                    f"{self.channel}: {name} must be a string, got {type(value)!r}")
            text = value.strip()
            if not text:
                raise RequirementDeclarationError(
                    f"{self.channel}: {name} is EMPTY. ⛔ An UNASSERTED required "
                    f"channel that cannot say {_WHY[name]} is indistinguishable from "
                    f"one somebody forgot — which is exactly how `v0`, `lan`, "
                    f"`nav_known` and `withheld_speed` sat unchecked behind a `None`.")
            if text.lower().strip(".") in _PLACEHOLDERS:
                raise RequirementDeclarationError(
                    f"{self.channel}: {name} is the placeholder {text!r}. Write "
                    f"{_WHY[name]}. ⛔ 'no config flag exists' is a fact about the "
                    f"code, not a reason — and for three channels it was also FALSE.")
            if len(text) < floor:
                raise RequirementDeclarationError(
                    f"{self.channel}: {name} is {len(text)} chars ({text!r}); at "
                    f"least {floor} are needed to state {_WHY[name]}. A one-word "
                    f"reason cannot be reviewed, and an unreviewable non-assertion "
                    f"is a suppression.")
        for p in self.predicates:
            if not (isinstance(p, str) and p
                    and all(part.isidentifier() for part in p.split("."))):
                raise RequirementDeclarationError(
                    f"{self.channel}: predicate {p!r} is not a dotted attribute "
                    f"path. It would never resolve, and a predicate that never "
                    f"resolves is the never-firing check this class exists to end.")

    @property
    def asserted(self) -> bool:
        return bool(self.predicates)

    def to_dict(self) -> dict:
        """The shape a run record / preflight artifact banks."""
        return {"channel": self.channel, "owner": self.owner,
                "reason": self.reason, "evidence": self.evidence,
                "predicates": list(self.predicates), "unblock": self.unblock,
                "asserted": self.asserted}


#: ⭐ THE RULING PER CHANNEL. Every line cite below was re-read from source on
#: 2026-09-07; the ones this file previously carried (`refc_v3.py:986-991`, `:1000`,
#: `:1005`, `:1006`, `:1012`, `:380`, `:356`) were ALL STALE — the live forward begins
#: at `refc_v3.py:1361`. Coordinates rot faster than reasons; both are checked by
#: ``test_rl_channel_value_flow.py``, which RESOLVES every predicate against a real
#: config rather than trusting this comment.
CHANNEL_REQUIREMENTS: tuple[ChannelRequirement, ...] = (
    ChannelRequirement(
        channel="ego_state",
        owner="refcv4b ego seam (E11'/X15)",
        predicates=("ego_state_inject",),
        reason="the v4 [B, 5] ego block. Omitting it on a build trained with it "
               "returns a well-formed fan from a differently conditioned policy and "
               "nothing raises — the defect this whole module exists to remove. The "
               "forward refuses the OTHER direction loudly (`refc_v3.py:1413`) but is "
               "silent on omission, so this side has to be ours.",
        evidence="PUBLISHED-CODE `refc_v3.py:454` (the flag), `:1413` (the converse "
                 "refusal); MEASURED MODEL_REGISTRY.md §4.6 argv "
                 "(`--ego-state-inject --ego-dropout 0.5`)."),

    # ⭐⭐ THE ONE THAT MATTERS MOST, AND THE WORST OF THE FOUR.
    ChannelRequirement(
        channel="v0",
        owner="refcv4b anchor seam (v0-conditioned vocabulary)",
        predicates=("core.anchors.v0_conditioned", "core.sel_reach_clamp"),
        reason="⛔ a missing v0 does not merely re-condition the policy, it SILENTLY "
               "REPLACES THE ACTION SPACE. `refc.py:3097` computes "
               "`v_ms = v0 if (sel_reach_clamp and v0 is not None) else None`; "
               "`refc.py:2128` hands that to `roll_bank`; and `refc.py:1722` reads "
               "`if v_ms is None ...: v = full(ref_speed)` — so EVERY anchor is rolled "
               "at the 10 m/s reference instead of the window's measured speed, with "
               "nothing raising. MEASURED gap between those two vocabularies "
               "(`refc.py:363-370`): 0.3773 m oracle-in-vocabulary fixed vs 0.2610 m "
               "v0-conditioned. Second site: `refc.py:2360` skips the S2 reachability "
               "band entirely when v_ms is None. Third: `refc.py:2970-2977` sets v = 0 "
               "and keep = 0, so the rollout runs keep = 0 on 100 % of rows where "
               "training saw keep = 1 on (1 - ego_dropout) of them — and this module's "
               "own docstring already refuses that argument for ego_state. ⭐ v0 at t0 "
               "is the PI's one ruled-admissible ego input (2026-09-02) and the whole "
               "longitudinal / max-speed story rests on it.",
        evidence="PUBLISHED-CODE `refc.py:383` (the flag, TWO levels nested), `:3097`, "
                 "`:2128`, `:1722-1725`, `:2360`, `:2970-2977`; `refc_v3.py:583`/`:751` "
                 "pin `sel_reach_clamp = True` as a v3 PRECONDITION, so the second "
                 "predicate covers the fixed-bank builds the first does not."),

    ChannelRequirement(
        channel="lan",
        owner="LAN route-corridor seam (graft_lan)",
        predicates=("core.graft_lan",),
        reason="⛔ `refc.py:3072` reads `if self.cfg.graft_lan and lan is not None:` — "
               "with lan None the route encoder (`lan_enc`, `lan_direction`) is SKIPPED "
               "and the decoder runs with no corridor; `refc.py:2887` states it outright "
               "(*None -> the seam is skipped entirely*). ⭐ The 'the route is missing on "
               "~75 % of windows anyway' defence does NOT apply: when graft_lan is on the "
               "trainer swaps in `lan_dataset_class` so EVERY window carries a lan tensor, "
               "and 'no route here' is expressed by the per-anchor `valid` FLAG "
               "(`LanConfig.feats = 4` = cos, sin, lat_norm, valid). lan=None is not that "
               "in-distribution state — it is the batch forgetting the key, which bypasses "
               "the encoder rather than feeding it an honest zero.",
        evidence="PUBLISHED-CODE `refc.py:693` (the flag, nested), `:3072`, `:2887`, "
                 "`:497-512` (LanConfig, the valid flag); `refc_v3_train.py:3746-3754` "
                 "and `refc_train.py:713-718` (the field is always minted)."),

    # ---- NOT ASSERTED. Each says why, and what would change that. ----------
    ChannelRequirement(
        channel="nav_cmd",
        owner="nav seam / the C6 confound arm",
        reason="⛔ asserting it would be WRONG, not merely unnecessary. `nav_inject` "
               "exists and defaults True (`refc_v3.py:404`), but REF-C's published arm "
               "decodes with nav_cmd=None ON PURPOSE (CLAUDE.md, the C6 confound), so a "
               "requirement keyed on that flag would refuse the programme's own standard "
               "eval arm. The nav channel's honesty is the `os_navzero` arm's job.",
        unblock="⛔ nothing should. This is a decision, not a gap: a per-arm 'this run "
                "intends to supply nav' declaration could carry it, but the guard would "
                "then be asserting the caller's intent, not the checkpoint's training.",
        evidence="PUBLISHED-CODE `refc_v3.py:404` (nav_inject), `refc.py:2967-2969` "
                 "(nav_cmd=None -> the `follow` fallback); CLAUDE.md C6 note."),

    ChannelRequirement(
        channel="nav_known",
        owner="E1 nav companion-bit seam",
        reason="⭐ GUARDED ELSEWHERE, and a flat assertion here would be actively wrong. "
               "A declaring field DOES exist — `cfg.core.nav_known_channel` — contrary to "
               "this map's previous claim; but the model already refuses BOTH silent "
               "directions: `refc.py:3006-3009` when nav_known is supplied to a build with "
               "the gate off, and `refc.py:3017-3022` when the gate is on AND a nav_cmd "
               "was supplied but the bit is missing (*Defaulting it to 1.0 would assert a "
               "judgement the labeller never made*). ⛔ The third branch is why we must "
               "NOT assert: `refc.py:3012-3016` — gate on, nav_cmd None — legitimately "
               "defaults the bit to 0.0, because *the `follow` fallback IS the sentinel*. "
               "Since REF-C's published arm decodes with nav_cmd=None, requiring "
               "nav_known whenever the gate is on would refuse that arm. Same trap as "
               "nav_cmd above, one channel over. There is no SILENT divergence left to "
               "catch, only a plumbing duty FORWARD_KEYS already discharges.",
        unblock="a batch-conditional predicate rather than a config one — require "
                "nav_known only when the batch also carries nav_cmd. ⚠️ That is exactly "
                "what `refc.py:3017-3022` already does, one layer down and with a better "
                "message, so duplicating it here would buy nothing but a second place to "
                "keep in sync.",
        evidence="PUBLISHED-CODE `refc.py:836` (the field, nested), `:3006-3009`, "
                 "`:3010-3022` (both refusals + the legitimate default)."),

    ChannelRequirement(
        channel="withheld_speed",
        owner="H-EGO-LIT-4 withheld-bank seam",
        reason="⭐ NOT AN EXTERNAL INPUT AT ALL — `None` is the CORRECT rollout value, so "
               "asserting it would refuse the normal path. It is the model's OWN predicted "
               "2 s speed routed back in (`refc.py:1697`: *the model's OWN predicted "
               "speed*), and when the caller passes None the hierarchy hook FILLS IT IN "
               "ITSELF: `refc.py:2944-2946`, `if withheld_speed is None: withheld_speed = "
               "hk.get('bank_speed_pred')`. It is read only under "
               "`anchor_withheld_bank == 'pred'` and only on withheld rows "
               "(`refc.py:1674-1679`), and its one external use is the eval-time SHUFFLE "
               "control, which passes a PERMUTED copy. On a non-hier build it falls back "
               "to the fixed roll, which `refc.py:1663-1666` documents as intended (*a "
               "caller that has no goal head*). ⇒ the previous note's *plumbed, not "
               "asserted* was right by accident and for no stated reason.",
        unblock="⛔ nothing should — this is a decision. If a future arm ever supplies a "
                "MEASURED speed here rather than the model's own prediction, that arm "
                "would need its own declaring field, and this record must be revisited "
                "then rather than silently inherited.",
        evidence="PUBLISHED-CODE `refc.py:1697`, `:2944-2946` (the self-supply), "
                 "`:1663-1679` (`_withheld_ref_speed`, the documented fallbacks), "
                 "`refc_v3.py:1489`/`:1498` (plumbed through both arms)."),

    ChannelRequirement(
        channel="agent_gt",
        owner="refcv5 WP-6 agent seam",
        reason="⭐ GUARDED ELSEWHERE — the model refuses BOTH directions loudly and "
               "unconditionally, so there is no SILENT divergence for this guard to "
               "catch: `refc_v3.py:1398-1403` when agent_gt is supplied to a build with "
               "no agent seam, and `:1404-1409` when an `--agents oracle` build receives "
               "none. ⚠️ It could not be asserted here even if it needed to be: the "
               "declaring field is NESTED (`cfg.core.agents.enable` / `.oracle`), and "
               "under the flat getattr this map used to run, a predicate naming it would "
               "have read False forever. That trap is now closed for everyone by "
               "`_read_predicate`, which raises on an unresolvable path.",
        unblock="nothing needs to — but if the model's two refusals were ever relaxed, "
                "the predicates ('core.agents.enable', 'core.agents.oracle') now resolve "
                "correctly and this record can simply be given them.",
        evidence="PUBLISHED-CODE `refc_v3.py:1397` (`_ag = getattr(self.cfg.core, "
                 "'agents', None)`), `:1398-1403`, `:1404-1409`."),
)

#: Back-compat alias, DERIVED so it can never desync from the records above.
#: ⚠️ Its values are now DOTTED PATHS, not flat attribute names — a bare
#: ``getattr(cfg, flag, False)`` on one of them is precisely the bug this replaced.
#: Read :data:`CHANNEL_REQUIREMENTS`; this exists only for two legacy call sites.
_REQUIRING_FLAG: dict[str, str | None] = {
    r.channel: (r.predicates[0] if r.predicates else None)
    for r in CHANNEL_REQUIREMENTS
}

_MISSING = object()


class ConditioningError(RuntimeError):
    """A batch omits a conditioning channel the checkpoint was trained with."""


def _read_predicate(cfg, dotted: str) -> bool:
    """Read a DOTTED config path, RAISING if it does not resolve.

    ⛔⛔ THE ONE LINE THAT MATTERS. The map this replaced read
    ``getattr(cfg, flag, False)`` — flat, and defaulting to False. A nested field
    (``core.anchors.v0_conditioned``) or a renamed one therefore read **False
    forever**: a check that is present in the source, passes every batch, and can
    never fire. That is a false green wearing a guard's clothes, and it is not a
    hypothetical — it is what happened to ``v0``, ``lan`` and ``nav_known``.

    ⇒ an unresolvable predicate is a DECLARATION BUG and must be loud at the first
    batch. Refusing here is the only reading that cannot be mistaken for "this build
    does not need the channel".
    """
    obj, parts = cfg, dotted.split(".")
    for i, part in enumerate(parts):
        nxt = getattr(obj, part, _MISSING)
        if nxt is _MISSING:
            walked = ".".join(parts[:i]) or "cfg"
            raise ConditioningError(
                f"conditioning predicate {dotted!r} does not resolve on this build: "
                f"{walked} has no attribute {part!r}. ⛔ REFUSING rather than reading "
                f"False. A predicate that cannot be read is a check that CAN NEVER "
                f"FIRE — it looks like a guard and passes every batch. That is how "
                f"v0, lan and nav_known sat unasserted behind a flat getattr on a "
                f"NESTED field. Fix the path, or drop the requirement — but do not "
                f"let it read False.")
        obj = nxt
    return bool(obj)


def conditioning_requirements(model) -> dict[str, bool]:
    """Which forward channels this build was TRAINED with, read from its own config.

    ⛔ Read from ``model.cfg``, never from a run name or a caller argument — `M51`
    (2026-09-05): *a name is not provenance*. A field that DECLARES the fact is the
    only admissible source.

    ⛔ Raises :class:`ConditioningError` when a declared predicate does not resolve.
    See :func:`_read_predicate` for why that is not optional.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise ConditioningError("model has no .cfg; cannot establish what it was "
                                "trained with, and guessing is how a differently "
                                "conditioned policy gets post-trained silently")
    return {r.channel: any(_read_predicate(cfg, p) for p in r.predicates)
            for r in CHANNEL_REQUIREMENTS}


def requirement_report() -> list[dict]:
    """The declarations, for a run record / preflight artifact.

    A conditioning contract that is checked but not written down is not evidence —
    and an UNASSERTED channel's reason is the half a reader cannot reconstruct.
    """
    return [r.to_dict() for r in CHANNEL_REQUIREMENTS]


def assert_conditioning(model, batch) -> dict[str, bool]:
    """REFUSE a batch that drops a channel the model was trained with.

    Returns the requirement map so the caller can record it in the run config —
    a conditioning contract that is checked but not written down is not evidence.
    """
    req = conditioning_requirements(model)
    by_channel = {r.channel: r for r in CHANNEL_REQUIREMENTS}
    missing = [k for k, needed in req.items()
               if needed and batch.get(k) is None]
    if missing:
        why = "\n".join(
            f"  - {k}: {by_channel[k].reason}" for k in missing if k in by_channel)
        raise ConditioningError(
            f"checkpoint was trained WITH {missing} (its own config says so) but the "
            f"batch supplies None. Sampling would return a well-formed fan from a "
            f"DIFFERENTLY CONDITIONED policy and nothing would raise. Supply the "
            f"channel, or retrain the intent. Batch keys present: "
            f"{sorted(k for k in batch if batch.get(k) is not None)}\n"
            f"What each missing channel silently changes:\n{why}")
    return req


def forward_kwargs(batch, cfg: PostTrainConfig) -> dict:
    """The kwargs for one ``RefCV3Model.forward``, from the batch."""
    kw = {k: batch.get(k) for k in FORWARD_KEYS}
    kw["steps"] = int(getattr(cfg, "decoder_steps", 0))
    return kw


def make_refc_sample_fn(model, cfg: PostTrainConfig, *, build_ctx=None,
                        reference=None, generator: torch.Generator | None = None,
                        strict_conditioning: bool = True):
    """``sample_fn(batch, cfg) -> (traj, logp, ctx[, extras])`` for ANY refc-v3 build.

    Identical in contract to ``refcv3_adapter.make_refcv3_sample_fn`` — the fan is still
    ``base = anchor_traj - offset`` and exploration is still the surrogate Gaussian over
    the emitted offset — with two differences:

    1. **every** forward channel is plumbed (``nav_known``, ``ego_state``,
       ``withheld_speed``, ``agent_gt`` in addition to ``nav_cmd``/``v0``/``lan``);
    2. the conditioning contract is ASSERTED on the first batch, so refcv4b's
       ``ego_state`` cannot be dropped silently.

    ``strict_conditioning=False`` exists only for a deliberate ablation arm that means to
    sample the un-conditioned policy; it must be typed, and it is recorded.
    """
    checked = {"done": False}

    def sample_fn(batch, cfg_in: PostTrainConfig):
        if strict_conditioning and not checked["done"]:
            assert_conditioning(model, batch)
            checked["done"] = True
        frames = batch["frames"]
        kw = forward_kwargs(batch, cfg_in)
        out = model(frames, **kw)
        anchor_traj = out["anchor_traj"]                   # [B, N, S, 2]
        offset = out["offset"]                             # [B, N, S, 2]
        base = anchor_traj - offset                        # the anchors alone

        off_g, logp = sample_offsets(offset, cfg_in, generator=generator)
        traj = base.unsqueeze(2) + off_g                   # [B, N, G, S, 2]

        ctx = dict(build_ctx(batch, out)) if build_ctx else {}
        ctx.setdefault("dt", cfg_in.dt)

        if reference is None:
            return traj, logp, ctx
        # The trust-region pair: the LIVE deterministic fan against the FROZEN
        # reference's fan on the SAME inputs -- including the same conditioning.
        with torch.no_grad():
            ref_out = reference(frames, **kw)
        return traj, logp, ctx, {"anchor_pair": (anchor_traj,
                                                 ref_out["anchor_traj"])}

    return sample_fn
