"""The DECLARED ALLOWANCE for loading a pre-2026-09-04 REF-C cold start.

⛔ WHY THIS MODULE EXISTS — PI DECISION QUEUE item 8, option (c).

``rl_pilot_refc21.py`` dies at load with::

    RuntimeError: Missing key(s) in state_dict: "decoder.anchor_controls".

MEASURED (this box, ``refc.refc_config()``): the model builds **488** state-dict
keys and 104,191,577 parameters; the July cold start
(``refc-diffusion-base-v21-30k``, 2026-07-20) carries **487**. The one key it
does not carry is ``decoder.anchor_controls``, registered UNCONDITIONALLY at
``refc.py:1496`` by the same 2026-09-04 change (``187c513``) that introduced
``AnchorConfig.v0_conditioned`` (``refc.py:388``). The checkpoint simply
predates the buffer.

⛔⛔ WHY ``strict=False`` IS THE WRONG FIX, AND IT IS THE TEMPTING ONE.
``strict=False`` does not default *this* key — it defaults **whatever happens to
be missing**, for ever, silently. And ``anchor_controls`` initialises to
``torch.zeros`` (``refc.py:1496``; MEASURED: shape [128, 2], 0 non-zero entries
at construction), so a ``v0_conditioned=True`` build that loaded this way would
roll EVERY anchor from an (accel, curvature) pair of ``(0, 0)``. Every candidate
in the vocabulary becomes "do nothing", the anchored Gaussian is centred on it,
and the arm TRAINS and CONVERGES. ``refc.py:2277`` already calls exactly this
*"a plausible-looking WRONG experiment"*.

⭐ It is worse than the 2026-09-04 units failure it rhymes with. There, reading
``controls`` column 1 as curvature instead of lateral acceleration produced a
**396 g** table — arithmetically perfect, and absurd enough to be caught. A
silently-zeroed ``anchor_controls`` produces **no absurd number at all**: it
produces a slightly worse planner and a full set of plausible R1/R2/R3.

## The rule this module enforces

Permit ``decoder.anchor_controls`` to be absent **only when the decoder that is
about to receive the weights will not read it** — and refuse otherwise.

⭐ WHY THAT IS THE RIGHT KEY TO HANG IT ON, derived from the code rather than
chosen for convenience. ``anchor_controls`` is read at exactly two sites, and
both are behind ``anchor_v0_cond``:

* ``refc.py:1727`` (``anchor_control_seq``) — reached only from the WP-4
  sampler, which ``refc.py:2270-2278`` REFUSES outright unless
  ``anchor_v0_cond`` is True;
* ``refc.py:1836`` / ``roll_bank`` (``refc.py:1715``) — with ``anchor_v0_cond``
  False, ``roll_bank`` returns the stored ``anchors`` expanded unchanged and
  never touches the controls at all.

⇒ with the flag False the defaulted zeros are **unreachable**, not merely
harmless-looking; with it True they are the whole action space. The allowance
therefore keys on the exact flag that decides whether the tensor matters, which
makes it a narrow checkable rule rather than a blanket relaxation.

## Three properties that make this not-a-relaxation

1. **The load stays ``strict=True``.** This module does not soften the load; it
   COMPLETES the checkpoint. The permitted tensor is inserted explicitly from
   the model's own initialised buffer and ``load_state_dict`` is then called
   with ``strict=True``, so any OTHER missing key, and ANY unexpected key, still
   raises exactly as before. ⛔ ``strict=False`` appears nowhere in this file.
2. **The permitted set is a LITERAL** (:data:`ALLOWED_ABSENT_KEYS`), not "the
   keys that turned out to be missing". A second missing key is a refusal.
3. **What was defaulted is STAMPED** into the run's ``config.json`` as
   ``anchor_controls_source``, in the same spirit as ``anchor_meta.py``'s
   ``control_units_source = "cli-override-legacy-file"``: the RECORD says the
   value came from a defaulting path and not from the checkpoint, so a reader
   who opens the run in isolation is not left to infer it.

## ⚠️ Where the flag is read from, and why not from the CLI

``v0_conditioned`` is read off the **constructed module** — the owning
submodule's ``anchor_v0_cond`` attribute, reached by walking the missing key's
own path (so ``decoder.anchor_controls`` resolves ``model.decoder`` and
``core.decoder.anchor_controls`` resolves ``model.core.decoder``, and the same
code serves ``RefCModel`` and ``RefCV3Model`` without a family branch).

⛔ It is NOT read from an operator flag, and there is deliberately no
``--allow-missing-anchor-controls``. A switch like that would let an operator
ASSERT the precondition they are supposed to be PROVING.
``test_refc_cold_start_allowance.py`` pins this by asserting that
``rl_pilot_refc21.main``'s parser exposes no option that can set it, and that
``refc.refc_config().anchors.v0_conditioned`` is literally ``False``.

⚠️ **The checkpoint's own config is consulted, and it is a VETO, not the
source.** If the checkpoint carries ``anchors.v0_conditioned`` and it says
``True``, the allowance is REFUSED — a checkpoint trained v0-conditioned cannot
legitimately lack the buffer, and proceeding would also mean running the weights
in a different action space than they were trained in. If the checkpoint does
NOT carry the leaf, that is recorded as ``ckpt_v0_conditioned: null`` with
``ckpt_v0_source: "ABSENT"`` and the allowance is marked
``UNVERIFIED_BY_CKPT_CONFIG`` — it is never silently reported as agreement.

⛔ **AND THE ABSENT CASE IS NOT AN INFERENCE ABOUT THE OPERATOR'S INTENT.** It
cannot be resolved by reading harder: ``AnchorConfig.v0_conditioned`` did not
EXIST as a field on 2026-07-20, so no config written by that run could carry it.
A rule that required the checkpoint's config to confirm the flag would refuse
this checkpoint permanently and unconditionally — i.e. it would be the
"do nothing" default wearing a guard's costume. What actually protects the run
in that case is condition (R) below, which is about THIS process and is
verifiable here: the decoder about to receive the weights will not read the
tensor. Callers that want the stricter rule anyway can pass
``require_ckpt_confirmation=True``; it only ever refuses more.

## The two conditions, named separately because they answer different questions

* **(R) the RUN condition** — *will the zeros be read?* Answered by
  ``anchor_v0_cond`` on the constructed decoder. Required. This is the safety
  condition.
* **(C) the CHECKPOINT condition** — *was this checkpoint trained
  v0-conditioned?* Answered by the checkpoint's config where it carries the
  leaf. A veto when it says True; recorded as UNVERIFIED when absent. This is
  the provenance condition, and ``rl_pilot_refc21.assert_config_contract``
  (with ``_v0_corroboration``) is its primary owner — this module re-checks it
  so the allowance cannot be granted against an explicit contradiction even if
  a caller skips that contract.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "ALLOWED_ABSENT_KEYS",
    "ANCHOR_CONTROLS_KEY_SUFFIX",
    "STAMP_KEY",
    "SOURCE_CHECKPOINT",
    "SOURCE_DEFAULTED",
    "ColdStartRefused",
    "load_cold_start",
    "plan_cold_start_load",
]

#: ⛔ THE COMPLETE, LITERAL SET OF STATE-DICT KEYS THIS MODULE MAY DEFAULT.
#: A missing key outside this tuple is a REFUSAL, not a defaulting. Written as a
#: literal on purpose: "whatever is missing" is `strict=False` with extra steps.
ALLOWED_ABSENT_KEYS: tuple[str, ...] = ("decoder.anchor_controls",)

#: The same key as it appears on a nested family (``RefCV3Model.core.decoder``).
#: Matching on the SUFFIX is what lets one rule serve both model families; the
#: owning module is still resolved from the full path, never guessed.
ANCHOR_CONTROLS_KEY_SUFFIX = "decoder.anchor_controls"

#: The field written into the run's ``config.json``.
STAMP_KEY = "anchor_controls_source"

#: ``anchor_controls`` came from the checkpoint, as normal. Nothing was allowed.
SOURCE_CHECKPOINT = "checkpoint"

#: ``anchor_controls`` was ABSENT from the checkpoint and was defaulted to the
#: model's own zero-initialised buffer under the declared allowance. The decoder
#: is not v0-conditioned, so the tensor is never read.
SOURCE_DEFAULTED = "defaulted-zeros-v0-unconditioned"


class ColdStartRefused(RuntimeError):
    """The checkpoint may not be loaded, and the message says why.

    Raised INSTEAD of loading. ⛔ A caller that catches this and retries with
    ``strict=False`` has reintroduced exactly the defect this module removes.
    """


def _owner_module(model: nn.Module, key: str) -> nn.Module:
    """The submodule that OWNS ``key`` — walked from the key's own path.

    ⭐ Structural, not a lookup table: ``decoder.anchor_controls`` resolves
    ``model.decoder``; ``core.decoder.anchor_controls`` resolves
    ``model.core.decoder``. A family that nests differently is served without a
    branch, and a key whose owner does not exist raises here rather than
    silently falling back to the root module (which would read the WRONG
    object's flag and is the failure this function exists to prevent).
    """
    parent = key.rsplit(".", 1)[0] if "." in key else ""
    if not parent:
        return model
    try:
        return model.get_submodule(parent)
    except AttributeError as exc:
        raise ColdStartRefused(
            f"⛔ COLD START REFUSED — `{key}` names a submodule `{parent}` this "
            f"model does not have, so the flag the allowance keys on cannot be "
            f"read from its true owner. Refusing rather than falling back to "
            f"the root module, whose flag would be a different object's.") from exc


def _v0_from_ckpt_cfg(ckpt_cfg_leaves) -> tuple[bool | None, str]:
    """``(value, source)`` for ``anchors.v0_conditioned`` in the CHECKPOINT'S
    own config, or ``(None, "ABSENT")``.

    ⚠️ Both nestings are accepted because the two model families genuinely
    differ (``RefCV3Config.core`` IS a ``RefCConfig``), and a config that
    carries BOTH and disagrees with itself is refused rather than resolved by
    precedence — a tie-break here would be a guess about which half is real.
    """
    if not isinstance(ckpt_cfg_leaves, dict) or not ckpt_cfg_leaves:
        return None, "ABSENT"
    seen = {k: bool(ckpt_cfg_leaves[k])
            for k in ("anchors.v0_conditioned", "core.anchors.v0_conditioned")
            if k in ckpt_cfg_leaves}
    if not seen:
        return None, "ABSENT"
    vals = set(seen.values())
    if len(vals) > 1:
        raise ColdStartRefused(
            "⛔ COLD START REFUSED — the checkpoint's own config carries "
            "`anchors.v0_conditioned` TWICE and the two copies DISAGREE: "
            + ", ".join(f"{k} = {v!r}" for k, v in sorted(seen.items()))
            + ". Neither is authoritative and picking one would be a guess "
              "about which half of the record is real. Fix the checkpoint's "
              "config, do not relax this."
        )
    return vals.pop(), "+".join(sorted(seen))


def plan_cold_start_load(model: nn.Module, state_dict, *,
                         ckpt_cfg_leaves=None,
                         require_ckpt_confirmation: bool = False) -> dict:
    """Decide — BEFORE any weight lands — what may be defaulted, or refuse.

    Returns the stamp dict. Raises :class:`ColdStartRefused` and loads nothing
    when the allowance does not apply.

    ⭐ It refuses BEFORE the load rather than after, for the same reason
    ``rl_pilot_refc21.assert_config_contract`` does: a refusal after
    ``load_state_dict`` leaves a mismatched model in memory and reads to the
    operator as a loading failure rather than a contract failure.
    """
    if not isinstance(state_dict, dict):
        raise ColdStartRefused(
            f"⛔ COLD START REFUSED — expected a state dict, got "
            f"{type(state_dict).__name__}.")

    built = model.state_dict()
    missing = sorted(set(built) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(built))

    stamp: dict = {
        "_what": "the declared allowance for a pre-2026-09-04 REF-C cold "
                 "start (PI DECISION QUEUE item 8, option (c)). Records what "
                 "this load was permitted to default, and on what grounds.",
        "_evidence_class": "MEASURED (ours; this run's own load)",
        "keys_built": len(built),
        "keys_in_checkpoint": len(state_dict),
        "missing_from_checkpoint": missing,
        "unexpected_in_checkpoint": unexpected,
        "allowed_absent_keys": list(ALLOWED_ABSENT_KEYS),
        "strict": True,
        STAMP_KEY: SOURCE_CHECKPOINT,
        "defaulted_keys": [],
    }

    # ---- an UNEXPECTED key is never permitted -------------------------- #
    # ⛔ This is not the same question as a missing one. A key the model does
    # not have means the checkpoint was written by a DIFFERENT architecture,
    # and there is no "harmless because unread" argument available for it.
    if unexpected:
        raise ColdStartRefused(
            f"⛔ COLD START REFUSED — the checkpoint carries "
            f"{len(unexpected)} key(s) this model does not have: "
            f"{unexpected[:8]}. That is an ARCHITECTURE mismatch, not a "
            f"missing default, and no allowance covers it.")

    if not missing:
        return stamp

    # ---- only the declared key may be absent ---------------------------- #
    not_allowed = [k for k in missing
                   if k not in ALLOWED_ABSENT_KEYS
                   and not k.endswith("." + ANCHOR_CONTROLS_KEY_SUFFIX)]
    if not_allowed:
        raise ColdStartRefused(
            f"⛔ COLD START REFUSED — {len(missing)} key(s) are missing from "
            f"the checkpoint and {len(not_allowed)} of them are OUTSIDE the "
            f"declared allowance {list(ALLOWED_ABSENT_KEYS)}: "
            f"{not_allowed[:8]}.\n"
            f"⛔ The allowance is a LITERAL list of one key that is provably "
            f"unread when the decoder is not v0-conditioned. It is not "
            f"'default whatever is missing' — that is `strict=False`, which "
            f"`refc.py:2277` already calls a plausible-looking WRONG "
            f"experiment.")

    # ---- (R) THE RUN CONDITION: will this decoder READ the zeros? ------- #
    # ⛔ EVERY missing key is checked against ITS OWN owner, not just the first.
    # A model carrying two anchor decoders would otherwise have one of them
    # authorise a default for the other — the allowance would be granted by a
    # module that is not the one about to read the zeros. `missing` is normally
    # a single key; the loop is here so that "normally" is not load-bearing.
    owners = {k: _owner_module(model, k) for k in missing}
    for k, owner in owners.items():
        if not hasattr(owner, "anchor_v0_cond"):
            raise ColdStartRefused(
                f"⛔ COLD START REFUSED — cannot establish whether "
                f"`{k}` is read: its owning module "
                f"({type(owner).__name__}) has no `anchor_v0_cond` attribute, "
                f"so the flag the allowance keys on does not exist here. "
                f"Refusing rather than assuming a value for it.")
    conditioned = {k: o for k, o in owners.items() if bool(o.anchor_v0_cond)}
    key = missing[0]
    owner = owners[key]
    run_v0 = any(bool(o.anchor_v0_cond) for o in owners.values())

    if run_v0:
        key = next(iter(conditioned))
        owner = conditioned[key]
        raise ColdStartRefused(
            f"⛔ COLD START REFUSED — `{key}` is absent from the checkpoint "
            f"AND this decoder is v0-CONDITIONED "
            f"(`{type(owner).__name__}.anchor_v0_cond = True`).\n"
            f"⛔ Defaulting it would hand the model an all-ZERO (accel, "
            f"curvature) vocabulary: `roll_bank` (refc.py:1715) would roll "
            f"EVERY anchor from (0, 0), so every candidate in the action "
            f"space becomes 'do nothing' and the anchored Gaussian is centred "
            f"on it. refc.py:2277 calls exactly this 'a plausible-looking "
            f"WRONG experiment' — it would train, converge and mean nothing.\n"
            f"⛔ Fix the INPUTS, not this guard: load a checkpoint that "
            f"carries `{key}`, or install a control vocabulary with "
            f"`load_anchors(anchors, controls=...)`, or build the model "
            f"un-conditioned (`anchors.v0_conditioned = False`) if that is "
            f"genuinely the arm you mean to run.")

    # ---- (C) THE CHECKPOINT CONDITION: does the record contradict it? --- #
    ckpt_v0, ckpt_src = _v0_from_ckpt_cfg(ckpt_cfg_leaves)
    if ckpt_v0 is True:
        raise ColdStartRefused(
            f"⛔ COLD START REFUSED — `{key}` is absent from the checkpoint, "
            f"but the CHECKPOINT'S OWN CONFIG ({ckpt_src}) says it was "
            f"trained with `v0_conditioned = True`.\n"
            f"⛔ Those two facts cannot both be true of a healthy artifact: "
            f"the buffer is registered UNCONDITIONALLY (refc.py:1496), so a "
            f"v0-conditioned run could not have written a state dict without "
            f"it. Either the config is wrong or the weights are, and "
            f"defaulting the tensor would bury the question under a run that "
            f"looks fine.\n"
            f"⛔ It would ALSO mean post-training weights in a different "
            f"action space than they were trained in — see "
            f"`rl_pilot_refc21.CFG_CONSEQUENCE['anchors.v0_conditioned']`.")

    if ckpt_v0 is None and require_ckpt_confirmation:
        raise ColdStartRefused(
            f"⛔ COLD START REFUSED — `require_ckpt_confirmation=True` and the "
            f"checkpoint carries no `anchors.v0_conditioned` leaf, so the "
            f"allowance cannot be confirmed from the checkpoint's own record. "
            f"⚠️ Note that a 2026-07-20 checkpoint CANNOT carry this leaf: the "
            f"field did not exist until 2026-09-04 (refc.py:388). Passing this "
            f"flag for such a checkpoint is a permanent refusal by "
            f"construction, not a check it could pass.")

    stamp.update({
        STAMP_KEY: SOURCE_DEFAULTED,
        "defaulted_keys": list(missing),
        "run_v0_conditioned": run_v0,
        "run_v0_source": f"{type(owner).__name__}.anchor_v0_cond "
                         f"(the CONSTRUCTED module, not a CLI flag)",
        "ckpt_v0_conditioned": ckpt_v0,
        "ckpt_v0_source": ckpt_src,
        "ckpt_confirmation": ("CONFIRMED" if ckpt_v0 is False
                              else "UNVERIFIED_BY_CKPT_CONFIG"),
        "_why_permitted":
            f"`{key}` is read ONLY when the decoder is v0-conditioned "
            f"(refc.py:1727 via the WP-4 sampler, which refc.py:2270 refuses "
            f"un-conditioned; and refc.py:1836/roll_bank, which returns the "
            f"stored `anchors` unchanged when the flag is False). This "
            f"decoder's flag is False, so the defaulted tensor is UNREACHABLE "
            f"on every code path in this process.",
        "_caveat":
            "the tensor below was NOT read from the checkpoint. It is the "
            "model's own zero-initialised buffer (refc.py:1496). Any future "
            "change that makes `anchor_controls` readable with "
            "`anchor_v0_cond = False` INVALIDATES this allowance and must "
            "update `tanitad.refs.cold_start`, not work around it."
            if ckpt_v0 is not False else
            "the checkpoint's own config confirms `v0_conditioned = False`; "
            "the defaulted tensor is the model's zero-initialised buffer and "
            "is unreachable on every code path in this process.",
    })
    return stamp


def load_cold_start(model: nn.Module, state_dict, *,
                    ckpt_cfg_leaves=None,
                    require_ckpt_confirmation: bool = False) -> dict:
    """Load ``state_dict`` into ``model`` under the declared allowance.

    ⭐ THE LOAD IS ``strict=True``. Nothing here relaxes it — a permitted key is
    INSERTED from the model's own buffer and the strict load then sees a
    complete state dict. A second missing key, or any unexpected key, still
    raises, which is the entire difference between this and ``strict=False``.

    Returns the stamp; raises :class:`ColdStartRefused` without loading if the
    allowance does not apply.
    """
    stamp = plan_cold_start_load(
        model, state_dict, ckpt_cfg_leaves=ckpt_cfg_leaves,
        require_ckpt_confirmation=require_ckpt_confirmation)

    to_load = state_dict
    if stamp["defaulted_keys"]:
        built = model.state_dict()
        to_load = dict(state_dict)
        for k in stamp["defaulted_keys"]:
            ref = built[k]
            to_load[k] = ref.detach().clone()
            stamp.setdefault("defaulted_detail", {})[k] = {
                "shape": list(ref.shape),
                "dtype": str(ref.dtype),
                "nonzero_entries": int(torch.count_nonzero(ref)),
                "source": "model's own initialised buffer (refc.py:1496 "
                          "registers it as torch.zeros)",
            }

    # ⛔ strict=True, unconditionally. Do not add a flag here.
    model.load_state_dict(to_load, strict=True)

    # ⭐ ASSERT ON THE ARTIFACT, NOT ON THE RETURN. `load_state_dict` returning
    # without raising is a claim the call makes about itself; the admissible
    # evidence is the state dict that now exists on the model.
    after = model.state_dict()
    if len(after) != stamp["keys_built"]:
        raise ColdStartRefused(
            f"⛔ POST-LOAD ASSERTION FAILED — the model holds {len(after)} "
            f"state-dict keys after the load, expected {stamp['keys_built']}.")
    for k in stamp["defaulted_keys"]:
        if k not in after:
            raise ColdStartRefused(
                f"⛔ POST-LOAD ASSERTION FAILED — `{k}` was defaulted but is "
                f"not in the loaded model's state dict.")
    stamp["keys_after_load"] = len(after)
    return stamp
