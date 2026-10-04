"""``tanitad.rl.ddv2_refc_chain`` — DiffusionDriveV2's RL chain bound to refcv5-v2's decoder.

⭐ WHAT THIS MODULE IS. The model-specific half of the DDv2 port. :mod:`tanitad.rl.ddv2_rl`
holds the released arithmetic and needs one callable, ``x0_fn(x_t, t) -> x0_hat``, in the
diffusion state's normalised units. This module builds that callable from a refcv5-v2
``AnchoredDiffusionDecoder`` (the WP-4 control-space sampler, ``refc.py:2042-2140``) without
touching the decoder's code:

* :func:`capture_sampler_inputs` records ``(kv, cond, bank, v_ms)`` AT the decoder's own
  ``_sample`` call during an ordinary forward, so the conditioning the chain sees is exactly what
  the deployed model computes (frames, nav, ego block, ctx, target latent — all upstream);
* :func:`make_x0_fn` runs ONE sampler pass, ``_decode_ctrl`` on the rolled path of the state,
  and returns ``x_in + du`` — the residual of ``refc.py:2118``;
* :func:`native_sample` drives the DEPLOYED ladder ``[10, 0]`` through that same callable with
  the decoder's own ``DDIMSchedule`` — the PARITY INSTRUMENT: it must reproduce
  ``decoder._sample`` bit-for-bit, which is what proves the callable is the model and not a
  look-alike.

⛔ THE TWO RELEASE CHOICES WHOSE MEANING DOES NOT TRANSFER, EXPOSED AS DECLARED KNOBS.
Both are the release's arithmetic; neither is in the paper; both were chosen for a state in
metres (``x/50, y/20``) and this state is control (``a_lon/4, a_lat/3`` m/s^2):

* ``input_clamp`` — the release clamps the decoder INPUT to [-1, 1] and adds the offset to the
  clamped state (``rl.py:826-827``, ``:477``). Here that is ``|a_lon| <= 4, |a_lat| <= 3`` m/s^2.
* ``DDV2Constants.clip_sample`` — the x0_hat clamp inside the step (SPEC A18).

The deployed refcv5-v2 sampler applies NEITHER. MEASURED 2026-09-15 on one real window: its
x0_hat reached |a_lon| 7.99 and |a_lat| 21.0 m/s^2, i.e. far outside the box — so a
release-exact chain can differ from the deployed sampler for a reason unrelated to RL. The
chain driver records how often each clamp binds; an arm states which setting it ran.

⛔ GROUPS. ``sampler_groups > 1`` is REFUSED by ``refc.py:2281-2291`` because three
N-indexed call sites (``loss_cls``, the anchor priors, ``sel_idx``) would be mis-indexed. The
chain never calls ``forward`` with groups: it calls ``_decode_ctrl`` directly on ``G * N``
queries. That is exact because every ``CrossAttnLayer`` attends the map (and agents) from each
query INDEPENDENTLY — there is no self-attention across anchors (``refc.py:1357-1372``) — so a
query's output does not depend on which other queries share the call. Pinned in the tests.

⭐ refcv7 NEW-1 (2026-09-27): ON A RESIDUAL BUILD THE SAMPLER'S STATE IS Delta, NOT THE PLAN.
``_sample`` then takes ``prior=(a0, kappa0, v)`` and rolls every state as ``P + Delta``
(``kinematic_prior.roll_plan``). The capture forwards that prior VERBATIM and RECORDS it
(:attr:`SamplerInputs.prior`), and every roll here composes on it the way ``_sample`` does:
:func:`state_to_path` REFUSES a residual build without it, an off build with it, and a speed
that is not the prior's. On an off build nothing here changes: ``prior`` is ``None`` and every
call is the pre-NEW-1 call.

Tier: T0 machinery. Evidence class: MEASURED (``tests/test_ddv2_refc_chain.py``).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch import Tensor

from tanitad.models import kinematic_prior as _kp
from tanitad.rl import ddv2_rl as D

__all__ = ["SamplerInputs", "capture_sampler_inputs", "anchor_state", "make_x0_fn",
           "state_to_path", "native_sample", "ChainSettings", "clamp_rates"]


@dataclass
class SamplerInputs:
    """What ``_sample`` was called with. ``v`` is the speed ``_sample`` itself derives
    (``refc.py:2085-2086``): ``v_ms`` when given, else the anchor reference speed.

    ``prior`` (refcv7 NEW-1) is ``_sample``'s ``prior=`` VERBATIM: ``(a0 [B], kappa0 [B],
    v [B])`` on a residual build, ``None`` on every other build. When it is set ``v`` IS
    ``prior[2]`` -- ``_sample``'s own rule: the fan is rolled from the prior's speed, the one
    ``roll_bank`` rolled the bank from, which on a WITHHELD row is not ``v_ms``."""
    kv: Tensor          # [B, P, d]
    cond: Tensor        # [B, d]
    bank: Tensor        # [B, N, S, 2]
    v_ms: Tensor | None
    v: Tensor           # [B] float32
    steps: int
    out: tuple | None = None
    #: RNG state at entry to ``_sample`` (CPU, and CUDA when on GPU): its first draw is the
    #: anchored-Gaussian ``eps`` (``refc.py:2105``), so a parity check can reproduce it exactly
    rng_cpu: Tensor | None = None
    rng_cuda: Tensor | None = None
    #: refcv7 NEW-1: the prior ``_sample`` composed on (see the class docstring)
    prior: tuple | None = None

    def replay_eps(self, decoder) -> Tensor:
        """Re-draw the ``eps`` ``_sample`` drew, from the recorded RNG state (state restored)."""
        x0_n = anchor_state(decoder, self.bank.shape[0], self.bank.dtype)
        saved_cpu = torch.get_rng_state()
        saved_cuda = torch.cuda.get_rng_state(x0_n.device) if x0_n.is_cuda else None
        try:
            torch.set_rng_state(self.rng_cpu)
            if x0_n.is_cuda and self.rng_cuda is not None:
                torch.cuda.set_rng_state(self.rng_cuda, x0_n.device)
            return torch.randn_like(x0_n)
        finally:
            torch.set_rng_state(saved_cpu)
            if saved_cuda is not None:
                torch.cuda.set_rng_state(saved_cuda, x0_n.device)


@contextlib.contextmanager
def capture_sampler_inputs(decoder):
    """Record ``decoder._sample``'s inputs (and outputs) for the forwards inside the block.

    ⛔ Refuses a decoder that has no sampler, or a call with agents: refcv5-v2 was trained with
    ``--agents off`` (``cross_agent: false``), and a chain that silently dropped agent tokens on
    an agent build would be a different model.
    """
    if getattr(decoder, "control_head", None) is None or getattr(decoder, "sched", None) is None:
        raise D.Ddv2ConfigError("decoder has no WP-4 sampler (control_head/sched is None); "
                                "the DDv2 chain binds to `sampler='ddim'` builds only")
    if str(getattr(decoder.cfg, "sampler_space", "control")) != "control":
        raise D.Ddv2ConfigError("the binding is for sampler_space='control'")
    records: list[SamplerInputs] = []
    orig = decoder._sample

    def hook(kv, cond, bank, v_ms, steps, agents=None, agent_pad=None, agent_pos=None,
             bev=None, prior=None):
        if agents is not None:
            raise D.Ddv2ConfigError("agent tokens reached the sampler; the binding is for an "
                                    "agent-free build (refcv5-v2 ran --agents off)")
        # ⛔ refcv6 coupling (1), 2026-09-23. `_sample` gained a ninth parameter when the
        # BEV-at-the-candidate's-waypoints coupling was wired to its consumer (it had been
        # built and never called -- 0 forward-hook fires). A FIXED-ARITY hook is how a
        # signature change becomes a TypeError three tests deep, so `bev` is accepted here,
        # forwarded verbatim, and REFUSED for the same reason `agents` is: refcv5-v2 was
        # trained with no BEV coupling, and a chain that silently dropped it on a
        # coupling-built model would be a different model.
        if bev is not None:
            raise D.Ddv2ConfigError("a BEV map reached the sampler (refcv6 coupling (1)); the "
                                    "binding is for the refcv5-v2 build, which has no "
                                    "`bev_coupling`")
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state(bank.device) if bank.is_cuda else None
        # ⛔ refcv7 NEW-1, 2026-09-27. `_sample` gained a TENTH parameter, `prior`, and the
        # Thor full-suite gate caught this fixed-arity hook on it (8 TypeErrors). It is
        # accepted, forwarded VERBATIM and RECORDED -- never refused like `agents`/`bev`: it
        # is not a coupling the refcv5-v2 build lacks, it is the reference frame of the
        # sampler's state on a residual build, and a chain that dropped it would roll Delta
        # as if it were the plan. Forwarded only when given, so an off build's call is the
        # pre-NEW-1 call byte for byte (the decoder passes it only then, too).
        out = (orig(kv, cond, bank, v_ms, steps, agents, agent_pad, agent_pos, bev)
               if prior is None else
               orig(kv, cond, bank, v_ms, steps, agents, agent_pad, agent_pos, bev,
                    prior=prior))
        b = bank.shape[0]
        v = (bank.new_full((b,), decoder.anchor_ref_speed) if v_ms is None
             else v_ms.reshape(-1).to(torch.float32))
        if prior is not None:
            v = prior[2]              # `_sample`'s own rule on a residual build
        records.append(SamplerInputs(kv=kv, cond=cond, bank=bank, v_ms=v_ms, v=v,
                                     steps=int(steps), out=out,
                                     rng_cpu=rng_cpu, rng_cuda=rng_cuda, prior=prior))
        return out

    decoder._sample = hook
    try:
        yield records
    finally:
        # restore the BOUND method exactly (deleting the instance attribute re-exposes it)
        del decoder._sample
        assert decoder._sample.__func__ is orig.__func__


def anchor_state(decoder, batch: int, dtype=torch.float32) -> Tensor:
    """``[B, N, S, 2]`` normalised anchor control sequence — ``_sample``'s ``x0_n``
    (``refc.py:2079-2080``)."""
    norm = torch.tensor(tuple(decoder.cfg.control_norm), dtype=dtype,
                        device=decoder.anchor_controls.device)
    return decoder.anchor_control_seq(batch, dtype) / norm


def state_to_path(decoder, x_n: Tensor, v: Tensor, prior: tuple | None = None) -> Tensor:
    """Normalised control state ``[B, M, S, 2]`` -> metres ``[B, M, S, 2]`` through the decoder's
    own integrator (``refc.py:2025-2040``).

    refcv7 NEW-1: on a RESIDUAL build the chain's state is the sampler's INTERNAL residual
    Delta, and its path is ``decoder._roll_state(Delta, v, False, prior)`` -- the roll
    ``_sample`` itself makes (``kinematic_prior.roll_plan(Delta, P)``) -- never
    ``decoder._state_to_path``, which takes EXPORTED absolute controls. ``prior``
    (:attr:`SamplerInputs.prior`) is therefore REQUIRED there, REFUSED on an off build, and
    ``v`` must be the prior's own roll speed. Each mismatch raises rather than rolling Delta as
    if it were the plan. Off build, ``prior=None``: the pre-NEW-1 call, unchanged.
    """
    norm = x_n.new_tensor(tuple(decoder.cfg.control_norm))
    mode = str(getattr(decoder, "residual_prior", _kp.RESIDUAL_PRIOR_OFF))
    if prior is None:
        if mode != _kp.RESIDUAL_PRIOR_OFF:
            raise D.Ddv2ConfigError(
                f"refcv7 NEW-1: this decoder was built with residual_prior={mode!r}, so the "
                "chain's state is the RESIDUAL Delta; rolling it without the prior would hand "
                "back Delta as the plan. Pass the captured `SamplerInputs.prior`.")
        return decoder._state_to_path(x_n * norm, v, False)
    if mode == _kp.RESIDUAL_PRIOR_OFF:
        raise D.Ddv2ConfigError(
            "refcv7 NEW-1: a prior was handed to a decoder built with residual_prior='off'; "
            "composing it would roll a plan this model never emits.")
    if not torch.equal(v.reshape(-1).to(torch.float32),
                       prior[2].reshape(-1).to(torch.float32)):
        raise D.Ddv2ConfigError(
            "refcv7 NEW-1: the speed handed to the chain is not the prior's roll speed. "
            "`_sample` rolls a residual build's fan from `prior[2]`, which on a WITHHELD row "
            "is not the raw v0. Pass `SamplerInputs.v`.")
    return decoder._roll_state(x_n * norm, v, False, prior)


def make_x0_fn(decoder, inputs: SamplerInputs, *, input_clamp: bool):
    """-> ``x0_fn(x_n [B, M, S, 2], t) -> x0_hat_n``, one sampler pass of THIS decoder.

    ``input_clamp=False`` is the deployed pass exactly (``refc.py:2111-2118``);
    ``input_clamp=True`` is the release's (``rl.py:826-827``, ``:477``).
    ``M`` may be any multiple of ``N`` (groups): see the module docstring for why that is exact.
    On a residual build (refcv7 NEW-1) the state is Delta and is rolled on ``inputs.prior``
    (:func:`state_to_path`), exactly as ``_sample`` rolls it.
    """
    kv, cond, v, prior = inputs.kv, inputs.cond, inputs.v, inputs.prior

    def x0_fn(x_n: Tensor, t: int) -> Tensor:
        if x_n.dim() != 4 or x_n.shape[0] != kv.shape[0]:
            raise D.Ddv2ConfigError(f"state {tuple(x_n.shape)} vs captured batch {kv.shape[0]}")
        x_in = x_n.clamp(-1.0, 1.0) if input_clamp else x_n
        x_path = state_to_path(decoder, x_in, v, prior)
        tt = torch.full((x_n.shape[0],), float(t), device=x_n.device, dtype=torch.float32)
        _, du = decoder._decode_ctrl(kv, cond, x_path, tt, None, None, None)
        return x_in + du

    return x0_fn


def native_sample(decoder, inputs: SamplerInputs, eps: Tensor, *, steps: int | None = None
                  ) -> tuple[Tensor, Tensor]:
    """The DEPLOYED sampler loop re-driven through :func:`make_x0_fn` -> ``(fan, u0_hat)``.

    ``eps`` must be the draw ``_sample`` made (``refc.py:2105``). A parity instrument only.
    """
    cfg = decoder.cfg
    b = inputs.bank.shape[0]
    x0_n = anchor_state(decoder, b, inputs.bank.dtype)
    dev = x0_n.device
    t0 = int(cfg.sampler_infer_t)
    k = int(steps) if steps else int(cfg.sampler_steps)
    ladder = decoder.sched.infer_timesteps(t0, max(k, 1))
    decoder.sched.to(dev)
    x_n = decoder.sched.add_noise(x0_n, eps, torch.tensor(t0, device=dev))
    fn = make_x0_fn(decoder, inputs, input_clamp=False)
    x0_hat_n = x_n
    for i, t in enumerate(ladder):
        t_prev = ladder[i + 1] if i + 1 < len(ladder) else 0
        x0_hat_n = fn(x_n, t)
        x_n = decoder.sched.step(x0_hat_n, x_n, torch.tensor(t, device=dev),
                                 torch.tensor(t_prev, device=dev))
    norm = x0_hat_n.new_tensor(tuple(cfg.control_norm))
    u0_hat = x0_hat_n * norm
    if inputs.prior is None:
        return decoder._state_to_path(u0_hat, inputs.v, False), u0_hat
    # refcv7 NEW-1: `_sample`'s own tail on a residual build -- the fan is Delta rolled on P
    # (`_roll_state`) and `u0_hat` leaves ABSOLUTE (`_export_controls`), so parity is checked
    # against what the decoder EMITS, never against its internal residual.
    return (state_to_path(decoder, x0_hat_n, inputs.v, inputs.prior),
            decoder._export_controls(u0_hat, inputs.prior))


@dataclass(frozen=True)
class ChainSettings:
    """The chain an arm runs, recorded verbatim. Defaults = the release."""
    groups: int = D.DDV2.group_size
    input_clamp: bool = True
    consts: D.DDV2Constants = D.DDV2

    def to_dict(self) -> dict:
        return {"groups": self.groups, "input_clamp": self.input_clamp,
                "labels": D.rollout_labels(self.consts.rollout_steps, self.consts.label_span),
                "consts": self.consts.to_dict()}


def clamp_rates(chain: Tensor, x0: Tensor) -> dict[str, float]:
    """How often each release clamp BINDS in a rollout, per control channel.

    ``chain [B, M, S, 2, T+1]``: inputs are ``chain[..., :-1]``; ``x0 [B, M, S, 2, T]``.
    """
    inp = chain[..., :-1]
    out = {}
    for c, name in ((0, "a_lon"), (1, "a_lat")):
        out[f"input_clamp_frac_{name}"] = float((inp[..., c, :].abs() > 1).float().mean())
        out[f"x0_clamp_frac_{name}"] = float((x0[..., c, :].abs() > 1).float().mean())
        out[f"x0_clamp_frac_final_{name}"] = float((x0[..., c, -1].abs() > 1).float().mean())
    return out
