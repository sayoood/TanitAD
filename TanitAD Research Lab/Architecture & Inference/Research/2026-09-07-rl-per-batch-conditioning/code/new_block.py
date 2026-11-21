# -*- coding: utf-8 -*-
"""The replacement block for tanitad/rl/refc_adapter.py (from `assert_conditioning`
to the end of `make_refc_sample_fn`). Applied by apply_patch.py."""

NEW = '''def _missing_channels(required, batch) -> list[str]:
    """The REQUIRED channels this batch drops.

    One ``dict.get`` per REQUIRED channel — not per declared channel. On today's
    fleet (`--anchor-v0-conditioned`, `--ego-state-inject`) that is 2-3 lookups.
    MEASURED cost of the whole per-batch check: 0.184 us, against a 436.7 ms
    deployed-scale forward and a 4.030 ms smoke forward — see :class:`ConditioningContract`.
    """
    return [k for k in required if batch.get(k) is None]


def _conditioning_refusal(missing, batch, *, batch_index: int | None = None) -> str:
    """The refusal text. ONE source, so the per-batch path and the single-shot
    :func:`assert_conditioning` cannot drift into saying different things."""
    by_channel = {r.channel: r for r in CHANNEL_REQUIREMENTS}
    why = "\\n".join(
        f"  - {k}: {by_channel[k].reason}" for k in missing if k in by_channel)
    where = ("" if batch_index is None else
             f" ⛔ REFUSED ON BATCH #{batch_index} of this rollout"
             f"{' — a LATER batch, which a once-only check would have missed' if batch_index > 1 else ''}.")
    return (
        f"checkpoint was trained WITH {missing} (its own config says so) but the "
        f"batch supplies None.{where} Sampling would return a well-formed fan from a "
        f"DIFFERENTLY CONDITIONED policy and nothing would raise. Supply the "
        f"channel, or retrain the intent. Batch keys present: "
        f"{sorted(k for k in batch if batch.get(k) is not None)}\\n"
        f"What each missing channel silently changes:\\n{why}")


class ConditioningContract:
    """The requirement map computed ONCE; the contract asserted on EVERY batch.

    ⛔⛔ THE DEFECT THIS REMOVES. ``make_refc_sample_fn`` used to carry
    ``checked = {"done": False}`` and assert on the FIRST batch only. A rollout whose
    *later* batches drop a channel was uncaught — and the cost of one missed batch is
    not "a softer policy", it is **a different action space**:

    * `refc.py:3097` — ``v_ms = v0 if (cfg.sel_reach_clamp and v0 is not None) else None``
    * `refc.py:2128` — ``bank = self.roll_bank(v_ms, ego_keep, ...)``
    * `refc.py:1722` — ``if v_ms is None ...: v = full(ref_speed)``

    ⇒ every anchor rolled at the 10 m/s reference instead of the window's measured
    speed. MEASURED vocabulary gap (`refc.py:363-370`): **0.3773 m** oracle-in-vocabulary
    for the fixed-path set vs **0.2610 m** for the v0-conditioned family. Two further
    silent sites on the same omission: `refc.py:2360` skips the S2 reachability band
    entirely, and `refc.py:2970-2977` sets ``keep = 0`` on 100 % of rows where training
    saw ``keep = 1`` on ``1 - ego_dropout`` of them.

    ⭐ WHY A CACHE RATHER THAN A MOVED CHECK. :func:`conditioning_requirements` walks a
    dotted config path per predicate and is the only part with any cost; the assertion
    itself is a handful of ``dict.get`` calls. So the map is resolved once and the
    batch is checked against it every time.

    ⛔ AND THE COST WAS MEASURED, NOT ASSERTED (CPU, this box, min of 7x2000 reps):

    ======================================  ==========  ==================  ===================
    path                                    per call    vs smoke forward    vs deployed forward
    ======================================  ==========  ==================  ===================
    ``conditioning_requirements`` ONCE        2.535 us         0.0629 %            0.00058 %
    ``ConditioningContract.check`` EVERY      0.184 us         0.0046 %            0.00004 %
    ``assert_conditioning`` UNCACHED          3.071 us         0.0762 %            0.00070 %
    ``RefCV3Model.forward`` smoke B=1         4.030 ms            —                    —
    ``RefCV3Model.forward`` default B=1     436.692 ms            —                    —
    ======================================  ==========  ==================  ===================

    ⚠️ REPORTED HONESTLY: the once-only optimisation **was never paying for itself**.
    One check is **1/21,937 of the *smoke* forward** — the smallest forward in the repo,
    used here as a conservative floor — and 1/2,377,009 of the deployed-scale one. Even
    the fully UNCACHED call would have been 0.076 % of that smoke forward. The cache is
    kept because it is strictly cheaper (16.7x on the per-batch path) and because resolving
    the map once makes a DECLARATION bug raise at a single deterministic point rather
    than on every batch; it is not what makes the guard affordable.

    ⚠️ SCOPE, STATED RATHER THAN IMPLIED. The map is keyed on the identity of
    ``model.cfg``: a *swapped* config re-resolves (pinned by a test), an *in-place
    mutation* of the same config object does not. That is deliberate — a config mutated
    mid-rollout changes the policy under the optimiser and is a different defect class
    than a batch that forgot a key, which is what this guard is for.
    """

    __slots__ = ("_model", "_cfg", "_req", "_required", "batches", "resolutions")

    def __init__(self, model):
        self._model = model
        self._cfg = _MISSING          # never a real cfg, so the first check resolves
        self._req: dict[str, bool] = {}
        self._required: tuple[str, ...] = ()
        #: how many batches this contract has CHECKED (not merely seen)
        self.batches = 0
        #: how many times the requirement map was resolved. The point of the cache is
        #: that this stays 1 while ``batches`` grows — pinned by a test, because a
        #: cache nobody measures is indistinguishable from no cache at all.
        self.resolutions = 0

    @property
    def requirements(self) -> dict[str, bool]:
        """The resolved map, or ``{}`` before the first batch."""
        return dict(self._req)

    @property
    def required(self) -> tuple[str, ...]:
        """The channels this build's own config says it was TRAINED with."""
        return self._required

    def check(self, batch) -> dict[str, bool]:
        """Assert the contract for ONE batch. Called on every batch of a rollout."""
        cfg = getattr(self._model, "cfg", None)
        if cfg is not self._cfg:
            # raises ConditioningError on a missing .cfg or an unresolvable predicate,
            # and does NOT record the cfg, so a declaration bug stays loud
            self._req = conditioning_requirements(self._model)
            self._cfg = cfg
            self._required = tuple(k for k, needed in self._req.items() if needed)
            self.resolutions += 1
        self.batches += 1
        missing = _missing_channels(self._required, batch)
        if missing:
            raise ConditioningError(
                _conditioning_refusal(missing, batch, batch_index=self.batches))
        return self._req


def assert_conditioning(model, batch) -> dict[str, bool]:
    """REFUSE a batch that drops a channel the model was trained with.

    The single-shot form: it resolves the map every call. Inside a rollout use
    :class:`ConditioningContract`, which resolves once and asserts every batch.

    Returns the requirement map so the caller can record it in the run config —
    a conditioning contract that is checked but not written down is not evidence.
    """
    req = conditioning_requirements(model)
    missing = _missing_channels(
        tuple(k for k, needed in req.items() if needed), batch)
    if missing:
        raise ConditioningError(_conditioning_refusal(missing, batch))
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
    2. the conditioning contract is asserted on **EVERY** batch, so refcv4b's
       ``ego_state`` — or refcv5's ``v0`` — cannot be dropped silently, **including by
       a batch that is not the first one**.

    ⛔ (2) USED TO READ "on the first batch". It was a cost/benefit call that had
    INVERTED: see :class:`ConditioningContract` for what a missed batch actually costs
    (the action space is replaced, not softened) and for the measured per-batch cost
    that says the optimisation was never buying anything.

    ``strict_conditioning=False`` exists only for a deliberate ablation arm that means to
    sample the un-conditioned policy; it must be typed, and it is recorded — literally,
    on the returned callable as ``.strict_conditioning`` and ``.conditioning_contract``
    (``None`` when off), so a run record can bank the arm rather than take its word.
    """
    # ⭐ ONE map for the whole rollout, asserted on every batch. `None` when the
    # ablation escape hatch is typed: nothing is resolved and nothing can raise, which
    # is exactly what an un-conditioned arm asked for.
    contract = ConditioningContract(model) if strict_conditioning else None

    def sample_fn(batch, cfg_in: PostTrainConfig):
        if contract is not None:
            contract.check(batch)
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

    # ⭐ "it must be typed, and it is RECORDED" — now true of the object, not just of
    # the docstring. A preflight/run record reads these instead of trusting an argv.
    sample_fn.strict_conditioning = bool(strict_conditioning)
    sample_fn.conditioning_contract = contract
    return sample_fn
'''
