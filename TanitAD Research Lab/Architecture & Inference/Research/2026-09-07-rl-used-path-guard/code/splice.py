"""Splice the conditioning contract into refcv3_adapter.py.

Authored on LOCAL disk, written to LOCAL disk. The caller copies the result to G:
in ONE op and verifies by `git hash-object`.
"""
import io
import sys

SRC = sys.argv[1]      # G: refcv3_adapter.py (read only)
BLOCK = sys.argv[2]    # the new block
OUT = sys.argv[3]      # local output

t = io.open(SRC, encoding="utf-8").read()
block = io.open(BLOCK, encoding="utf-8").read()
n0 = len(t)

# ---- 1. the docstring gets a pointer to the contract -----------------------
OLD_DOC_TAIL = '''⚠️ ``offset`` can be exactly 0 for an anchor the decoder does not move. Under
the multiplicative form the sample density then collapses (zero scale), so the
scale is floored at ``min_scale`` — otherwise ``logp`` is -inf and the loss is
NaN, which surfaces as a dead run rather than as the degenerate anchor it is.
"""'''
NEW_DOC_TAIL = '''⚠️ ``offset`` can be exactly 0 for an anchor the decoder does not move. Under
the multiplicative form the sample density then collapses (zero scale), so the
scale is floored at ``min_scale`` — otherwise ``logp`` is -inf and the loss is
NaN, which surfaces as a dead run rather than as the degenerate anchor it is.

⛔⛔ WHICH MODEL THIS BINDS — READ THIS BEFORE THE SENTENCE ABOVE.
This module is DUCK-TYPED: it imports no model at all, and its two callers hand
it DIFFERENT CLASSES.

    stack/scripts/rl_pilot_refc21.py:169   refc.RefCModel        (the only
                                           production RL caller)
    stack/tests/test_rl_refcv3_integration.py:31   refc_v3.RefCV3Model

⚠️ The first line of this docstring said "a real ``RefCV3Model``" and was wrong
for the caller that matters. ``refc.RefCModel`` (`refc.py:2515`, forward
`:2870-2881`) and ``refc_v3.RefCV3Model`` (`refc_v3.py:763`, forward
`:1361-1372`) are separate classes in separate modules with no inheritance
between them; `refc.py` does not import, subclass or alias the other. Their
signatures differ by seven channels and their configs nest differently
(``RefCV3Config.core`` IS a ``RefCConfig``, `refc_v3.py:347`).

⇒ **the conditioning contract below is resolved from the model it is HANDED**,
never from a tuple written here. See :func:`forward_conditioning_channels`.
"""'''
assert t.count(OLD_DOC_TAIL) == 1, "docstring tail anchor not unique"
t = t.replace(OLD_DOC_TAIL, NEW_DOC_TAIL, 1)

# ---- 2. the block goes in ahead of the factory -----------------------------
ANCHOR = "def make_refcv3_sample_fn(model, cfg: PostTrainConfig, *,"
assert t.count(ANCHOR) == 1, "factory anchor not unique"
t = t.replace(ANCHOR, block.rstrip("\n") + "\n\n\n" + ANCHOR, 1)

# ---- 3. the factory asserts the contract on EVERY batch --------------------
OLD_FACTORY = '''def make_refcv3_sample_fn(model, cfg: PostTrainConfig, *,
                          build_ctx=None, reference=None,
                          generator: torch.Generator | None = None):
    """Return a ``sample_fn(batch, cfg) -> (traj, logp, ctx)`` for a RefCV3Model.

    ``batch`` must be a mapping carrying at least ``frames``; optional
    ``nav_cmd`` / ``v0`` / ``lan`` are forwarded. ``build_ctx(batch, out)``
    supplies the REWARD CONTEXT — scene facts only.

    ⛔ The context is never taken from the model's own outputs. The reward may
    not read a model-produced ranking, and the cheapest way to guarantee that is
    to never hand it one: ``build_ctx`` receives the batch, and anything it adds
    is audited by ``assert_selector_disjoint`` on every step.
    """
    def sample_fn(batch, cfg_in: PostTrainConfig):
        frames = batch["frames"]
        out = model(frames, batch.get("nav_cmd"), batch.get("v0"),
                    steps=int(getattr(cfg_in, "decoder_steps", 0)),
                    lan=batch.get("lan"))
        anchor_traj = out["anchor_traj"]                   # [B, N, S, 2]'''

NEW_FACTORY = '''def _forward_kwargs(batch, cfg_in: PostTrainConfig) -> dict:
    """The kwargs for ONE forward, built by iterating :data:`PLUMBED_CHANNELS`.

    ⭐ Derived, so the plumbed set the contract ENFORCES and the plumbing the
    forward RECEIVES are one object and cannot drift. (They were previously two:
    a positional call here and nothing enforcing it anywhere.)
    """
    kw = {k: batch.get(k) for k in PLUMBED_CHANNELS}
    kw["steps"] = int(getattr(cfg_in, "decoder_steps", 0))
    return kw


def make_refcv3_sample_fn(model, cfg: PostTrainConfig, *,
                          build_ctx=None, reference=None,
                          generator: torch.Generator | None = None,
                          strict_conditioning: bool = True):
    """Return a ``sample_fn(batch, cfg) -> (traj, logp, ctx)`` for the handed model.

    ``batch`` must be a mapping carrying at least ``frames``; the channels in
    :data:`PLUMBED_CHANNELS` (``nav_cmd`` / ``v0`` / ``lan``) are forwarded.
    ``build_ctx(batch, out)`` supplies the REWARD CONTEXT — scene facts only.

    ⛔ The context is never taken from the model's own outputs. The reward may
    not read a model-produced ranking, and the cheapest way to guarantee that is
    to never hand it one: ``build_ctx`` receives the batch, and anything it adds
    is audited by ``assert_selector_disjoint`` on every step.

    ⛔⛔ THE CONDITIONING CONTRACT IS ASSERTED ON **EVERY** BATCH (2026-09-07).
    Before this, the only production RL script in the repo reached a real planner
    through this function with **no conditioning check of any kind** — not this
    adapter's (there was none), not ``channel_guard``'s launch preflight, and not
    ``rl_control_space_preflight`` (MEASURED: 0 occurrences of every guard token
    in ``rl_pilot_refc21.py``). A batch that dropped ``v0`` would have post-trained
    a policy on a DIFFERENT ACTION SPACE — every anchor rolled at the 10 m/s
    reference (`refc.py:3097` → `:2128` → `:1722`) — and returned a well-formed
    fan with nothing raising.

    ⛔ Per-batch, not once-only. The cost is settled: a sibling MEASURED the
    equivalent check at **0.181-0.184 µs**, or **1/17,495 of the smallest forward
    in the repo**. A latch is not a cheaper guard; it is a guard that stops
    guarding after batch 1, and the defect it would miss is the same size.

    ``strict_conditioning=False`` exists only for a deliberate ablation arm that
    means to sample the un-conditioned policy. It must be typed, and it is
    RECORDED on the returned callable (``.strict_conditioning`` /
    ``.conditioning_contract``) so a run record can bank the arm rather than take
    its word for it.
    """
    # ⭐ ONE map for the whole rollout, asserted on every batch. `None` when the
    # ablation escape hatch is typed: nothing is resolved and nothing can raise,
    # which is exactly what an un-conditioned arm asked for.
    contract = RefCConditioningContract(model) if strict_conditioning else None

    def sample_fn(batch, cfg_in: PostTrainConfig):
        if contract is not None:
            contract.check(batch)
        frames = batch["frames"]
        kw = _forward_kwargs(batch, cfg_in)
        out = model(frames, **kw)
        anchor_traj = out["anchor_traj"]                   # [B, N, S, 2]'''

assert t.count(OLD_FACTORY) == 1, "factory body anchor not unique"
t = t.replace(OLD_FACTORY, NEW_FACTORY, 1)

# ---- 4. the reference forward must use the SAME conditioning ---------------
OLD_REF = '''        with torch.no_grad():
            ref_out = reference(frames, batch.get("nav_cmd"), batch.get("v0"),
                                steps=int(getattr(cfg_in, "decoder_steps", 0)),
                                lan=batch.get("lan"))
        return traj, logp, ctx, {"anchor_pair": (anchor_traj,
                                                 ref_out["anchor_traj"])}

    return sample_fn'''
NEW_REF = '''        with torch.no_grad():
            ref_out = reference(frames, **kw)
        return traj, logp, ctx, {"anchor_pair": (anchor_traj,
                                                 ref_out["anchor_traj"])}

    # ⭐ "it must be typed, and it is RECORDED" — true of the OBJECT, not just of
    # the docstring. A preflight or a run record reads these rather than trusting
    # an argv string that says what the operator meant to type.
    sample_fn.strict_conditioning = bool(strict_conditioning)
    sample_fn.conditioning_contract = contract
    return sample_fn'''
assert t.count(OLD_REF) == 1, "reference-call anchor not unique"
t = t.replace(OLD_REF, NEW_REF, 1)

io.open(OUT, "w", encoding="utf-8", newline="\n").write(t)
print(f"spliced: {n0} -> {len(t)} chars, {t.count(chr(10))} lines")
import ast
ast.parse(t)
print("AST: result parses OK")
