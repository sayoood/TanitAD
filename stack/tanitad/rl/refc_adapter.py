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

import torch
from torch import Tensor

from .config import PostTrainConfig
# Re-exported verbatim: these are model-agnostic and must not fork.
from .refcv3_adapter import gt_context, sample_offsets

__all__ = ["FORWARD_KEYS", "ConditioningError", "conditioning_requirements",
           "assert_conditioning", "forward_kwargs", "make_refc_sample_fn",
           "sample_offsets", "gt_context"]

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

#: Maps a forward kwarg to the ``RefCV3Config`` flag that DECLARES the build was trained
#: with it. A channel whose flag is True was trained WITH that input, so an RL rollout
#: that omits it samples a different policy.
#:
#: ⚠️ ONLY ``ego_state`` IS CONFIG-DECLARED, AND THE MAP SAYS SO RATHER THAN INVENTING
#: THE OTHERS. Verified against source: ``ego_state_inject`` exists (`refc_v3.py:380`,
#: enforced at `:1000`), but there is **no** ``nav_known_inject`` and no
#: ``withheld_speed_inject`` — those two are plumbed straight through to the core
#: (`:1025-1031`) with nothing governing them. Writing a flag name that does not exist
#: would make ``getattr(cfg, flag, False)`` read False forever: a check that can never
#: fire, which is the false-green class wearing a guard's clothes.
#:
#: ⛔ ``nav_cmd`` is DELIBERATELY NOT ASSERTED even though ``nav_inject`` exists and
#: defaults True (`:356`). REF-C is evaluated with ``nav_cmd=None`` on purpose
#: (CLAUDE.md, the C6 confound note), so asserting it would refuse the programme's own
#: standard eval arm. The nav channel's honesty is the ``os_navzero`` arm's job, not
#: this guard's.
_REQUIRING_FLAG: dict[str, str | None] = {
    "ego_state": "ego_state_inject",
    "nav_known": None,        # no config flag exists — plumbed, not asserted
    "withheld_speed": None,   # no config flag exists — plumbed, not asserted
    "lan": None,
    "nav_cmd": None,          # see the note above: asserting it would be wrong
    "v0": None,
    # ⚠️ NOT asserted HERE, and not because no field declares it. The declaring
    # field is NESTED — ``cfg.core.agents.enable``/``.oracle`` (`refc_v3.py:1005`) —
    # while this map is read with a FLAT ``getattr(cfg, flag, False)``, so writing
    # ``"agents"`` here would read False forever: the never-firing check this map's
    # own note warns about. It needs no local guard either, because the model refuses
    # BOTH directions loudly and unconditionally: `refc_v3.py:1006` when ``agent_gt``
    # is supplied to a build with no agent seam, and `:1012` when an ``--agents
    # oracle`` build receives none. There is therefore no SILENT divergence for
    # ``assert_conditioning`` to catch — only a plumbing duty, which ``FORWARD_KEYS``
    # above now discharges.
    "agent_gt": None,
}


class ConditioningError(RuntimeError):
    """A batch omits a conditioning channel the checkpoint was trained with."""


def conditioning_requirements(model) -> dict[str, bool]:
    """Which forward channels this build was TRAINED with, read from its own config.

    ⛔ Read from ``model.cfg``, never from a run name or a caller argument — `M51`
    (2026-09-05): *a name is not provenance*. A field that DECLARES the fact is the
    only admissible source.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise ConditioningError("model has no .cfg; cannot establish what it was "
                                "trained with, and guessing is how a differently "
                                "conditioned policy gets post-trained silently")
    req: dict[str, bool] = {}
    for key, flag in _REQUIRING_FLAG.items():
        req[key] = bool(getattr(cfg, flag, False)) if flag else False
    return req


def assert_conditioning(model, batch) -> dict[str, bool]:
    """REFUSE a batch that drops a channel the model was trained with.

    Returns the requirement map so the caller can record it in the run config —
    a conditioning contract that is checked but not written down is not evidence.
    """
    req = conditioning_requirements(model)
    missing = [k for k, needed in req.items()
               if needed and batch.get(k) is None]
    if missing:
        raise ConditioningError(
            f"checkpoint was trained WITH {missing} (its own config says so) but the "
            f"batch supplies None. Sampling would return a well-formed fan from a "
            f"DIFFERENTLY CONDITIONED policy and nothing would raise. Supply the "
            f"channel, or retrain the intent. Batch keys present: "
            f"{sorted(k for k in batch if batch.get(k) is not None)}")
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
