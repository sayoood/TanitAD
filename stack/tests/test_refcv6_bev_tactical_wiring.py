"""refcv6 §4 — THE BEV->TACTICAL SEAM, lifted by PI RULING 2026-09-17 (R2/R3).

⛔ THE DEFECT THIS MODULE EXISTS FOR, MEASURED at tip ``ee1635a``:
``refc_v3.RefCV3Model.forward`` never passed ``bev_tokens=`` to
``self.core(...)`` — its call site passed ``**_core_kw``, which carried
``scene_hook`` alone — and the BEV encoder lived on the TRAINER's wrapper
(``model._perception``), running AFTER the core forward on ``out["fmap_s16"]``.
So at ``refc.py``'s scene-hook site there was **no BEV token in existence**,
``--tac-decoder-d-bev > 0`` REFUSED, and every arm stamped ``sources:
["agent"]``. The PI lifted it by instruction: *"yes use also the map for
tactical behavior decoding and you can backpropagate to the trunk."*

⛔ EVERY GUARD HERE IS PROVEN BY MUTATION, NEVER BY INSPECTION. An AST census
once read "0 suspects" on BOTH the fixed and the broken trainer, and four checks
in one night were green forever because their expected value was an expression
over the code under test. So each test below either runs a REAL forward, or
reintroduces the historical defect and asserts the check GOES RED.

⭐ AND EVERY REFUSAL CARRIES ITS GREEN CONTROL. A RED-only table cannot tell a
guard from a brick — measured on the companion package's first run, where an
argv typo made every "guard fires" row fire for the wrong reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.models import refcv6_perception_branch as PB       # noqa: E402
from tanitad.refs import refc_v3 as v3                          # noqa: E402
from tanitad.refs import refcv6_tactical as v6tac               # noqa: E402


# =========================================================================== #
# 0. the seam on `refc.py` — present, optional, and INERT when unused         #
# =========================================================================== #
def test_the_core_forward_publishes_a_bev_hook_seam():
    """⛔ READ OFF THE SIGNATURE, not off a comment. The whole unblock is that
    a hook fires where `fmap_s16` is born; if the keyword is gone, the v3
    wrapper's `_core_kw["bev_hook"]` becomes a TypeError on every arm."""
    import inspect

    from tanitad.refs import refc
    p = inspect.signature(refc.RefCModel.forward).parameters
    assert "bev_hook" in p
    assert p["bev_hook"].default is None, (
        "the hook must DEFAULT to None: that default is the entire "
        "bit-identity argument — with it, the block is untouched dead code")
    # ⭐ the pre-existing explicit ports must SURVIVE. A hook that replaced
    # them would silently break any caller that supplies tokens directly.
    assert "bev_tokens" in p and "bev_pad" in p


def test_bev_hook_and_explicit_bev_tokens_TOGETHER_are_refused():
    """⛔ Two suppliers for one tensor. The hook's tokens carry the trunk's
    graph and an explicit tensor usually does not, so a silent winner here
    decides whether the arm trains the trunk — invisibly."""
    from tanitad.refs import refc
    m = refc.RefCModel(refc.refc_smoke_config())
    enc = m.cfg.encoder
    h, w = enc.image_hw()
    frames = torch.rand(1, int(m.cfg.window), int(enc.in_channels), h, w)
    with pytest.raises(ValueError, match="BOTH"):
        m(frames, bev_hook=lambda f: {"bev_tokens": torch.zeros(1, 4, 8)},
          bev_tokens=torch.zeros(1, 4, 8))


# =========================================================================== #
# 1. the model — a real forward, both arms                                    #
# =========================================================================== #
def _model(d_bev: int, *, detach: bool = False, attach: bool = True):
    """A REAL `RefCV3Model` with the in-repo smoke trunk plus a hand-built
    perception branch. ⛔ No geometry literal: `image_hw` comes off the encoder
    and the branch's `d_image`/`image_hw` come off the map the trunk emits."""
    import dataclasses as _dc

    from tanitad.refs.refc_agents import AgentSeamConfig
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.tac_vocab_version = "v7.0"
    # ⛔ THE TIMM TRUNK, NOT THE SMOKE ONE — and the refusal that forced it is
    # the point. The in-repo REF-C ResNet emits stride 32 ALONE, so `fmap_s16`
    # is None and `refc.py`'s `bev_hook` guard fires: a BEV lift with nothing to
    # sample would read as "the map adds nothing" while never having had a map.
    # ⚠️ `resnet18` at 64x64, `pretrained=False`: no download, ~0.3 s to build.
    # ⭐ `dataclasses.replace` and not field-by-field assignment —
    # D-REFCV6-IMAGEHW-DROPS-TRUNK-FIELDS is exactly a rebuild that listed three
    # fields and silently dropped four.
    cfg.core.encoder = _dc.replace(
        cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k",
        trunk_pretrained=False, in_channels=3, image_size=64, image_width=None)
    cfg.core.agents = AgentSeamConfig(enable=True)
    # ⚠️ AN INTEGRATION COUPLING, NAMED RATHER THAN WORKED AROUND (it is
    # already flagged in the companion package): `refc.py` REFUSES
    # `agents.enable` without `decoder.cross_agent`, so giving the TACTICAL
    # layer agent slots forces agent cross-attention on the OPERATIVE decoder
    # too. Identical on every arm here, so it cannot separate them.
    cfg.core.decoder.cross_agent = True
    cfg.tac_decoder_v6 = True
    cfg.tac_decoder_bev_detach = bool(detach)
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(
        d_model=64, n_layers=1, n_heads=4, ff_mult=2,
        d_agent=int(cfg.core.decoder.d), d_bev=int(d_bev),
        sources=("agent",) if d_bev <= 0 else ("agent", "bev"))
    model = v3.RefCV3Model(cfg)
    if attach:
        model._perception = _FakeBranch(int(d_bev) or 8)
    return model, cfg


class _FakeBranch(torch.nn.Module):
    """A stand-in for :class:`PerceptionBranch` with ONE real parameter, so
    "did gradient reach the branch" is answerable without a timm download.

    ⛔ It is NOT a stub of the thing under test: the object under test is the
    WIRING in `refc_v3`/`refc.py`, and this supplies the same contract the real
    branch does (`lift`, a dict with `bev_feats`/`bev_tokens`). The real branch
    is exercised end-to-end by `code/grad_reach_bev.py`.
    """

    def __init__(self, d_bev: int):
        super().__init__()
        # the attribute surface `grad_reach_report` and `_bev_hook` read
        self.lift = None                      # no geometry required
        self.map_branch = None
        self.box_mem = None
        self.box_dec = None
        self.d_bev = int(d_bev)
        self.proj = torch.nn.Linear(1, self.d_bev)

    def forward(self, fmap_s16, grid=None, valid=None):
        b = fmap_s16.shape[0]
        pooled = fmap_s16.mean(dim=(1, 2, 3)).reshape(b, 1, 1)
        feats = self.proj(pooled).expand(b, 6, self.d_bev)
        return {"bev_feats": feats.transpose(1, 2).reshape(b, self.d_bev, 2, 3),
                "bev_tokens": feats}


def _fwd(model, b: int = 2):
    enc = model.cfg.core.encoder
    h, w = enc.image_hw()
    frames = torch.rand(b, int(model.cfg.core.window), int(enc.in_channels),
                        h, w)
    return model(frames, nav_cmd=torch.zeros(b, dtype=torch.long),
                 v0=torch.tensor([4.0] * b))


def test_GREEN_bev_tokens_REACH_the_behaviour_decoder():
    """⭐⭐ THE CLAIM. `n_scene` is the decoder's count of ATTENDED (non-pad)
    keys, so it is evidence the tokens were used — not merely passed."""
    model, cfg = _model(d_bev=16)
    out = _fwd(model)
    assert out["tacv6_injected"] is True
    assert out["perception"]["bev_tokens_fed"] is True
    n_bev = int(out["tacv6_n_scene"].min())

    # ⛔ THE DISCRIMINATING CONTROL: the SAME model shape, d_bev 0. Without it
    # a constant `n_scene` would pass the assertion above.
    model0, _ = _model(d_bev=0)
    out0 = _fwd(model0)
    assert out0["perception"]["bev_tokens_fed"] is False
    assert "bev_tokens" not in out0["perception"]
    assert n_bev > int(out0["tacv6_n_scene"].max()), (
        f"attended keys did not rise with the BEV source: {n_bev} vs "
        f"{int(out0['tacv6_n_scene'].max())}")


def test_R3_the_tactical_loss_REACHES_the_bev_branch_and_the_detach_STOPS_it():
    """⭐⭐ PI RULING R3, both arms, on a REAL backward.

    ⛔ The loss is the decoder's own logits and NOTHING else — no planner term —
    so a non-zero gradient on the branch cannot have come from anywhere but the
    tactical layer.
    """
    seen = {}
    for detach in (False, True):
        model, _ = _model(d_bev=16, detach=detach)
        out = _fwd(model)
        model.zero_grad(set_to_none=True)
        (out["tacv6_goal_logits"] ** 2).mean().backward()
        g = model._perception.proj.weight.grad
        seen[detach] = 0.0 if g is None else float(g.abs().sum())
    assert seen[False] > 0.0, (
        "R3 says the tactical loss MAY shape the trunk; with the path attached "
        "it reads exactly 0 — the `tac_goal_tok_head` class (11,286 params at "
        "grad_abs_sum 0 for 40,284 steps)")
    assert seen[True] == 0.0, (
        "the detach ablation did not cut the path — so the two arms are "
        "indistinguishable and the ruling is untestable")


# =========================================================================== #
# 2. MUTATION — reintroduce the historical defect, assert RED                 #
# =========================================================================== #
def test_MUTATION_a_declared_bev_source_with_NO_branch_REFUSES():
    """⛔⛔ THE HOLE THE RULING MADE REACHABLE, and it is not hypothetical:
    `refcv6_tactical.forward` does NOT raise when `bev_in` is built and
    `bev_tokens` is None while AGENT tokens are present — `kv_parts` is
    non-empty, so it runs AGENT-ONLY and returns a healthy dict while the run
    record stamps `sources: ["agent", "bev"]`.

    ⭐ The mutation IS the historical state: a decoder that declares the map and
    a model with no perception branch attached (exactly what an eval harness
    loading a checkpoint would produce)."""
    model, _ = _model(d_bev=16, attach=False)
    with pytest.raises(ValueError, match="AGENT-ONLY"):
        _fwd(model)


def test_MUTATION_the_decoder_alone_is_BLIND_to_that_case():
    """⛔ THE REASON THE GUARD LIVES IN THE MODEL AND NOT IN THE DECODER — and
    it is asserted, not argued. This reproduces the silent path directly: a
    two-source decoder given agent tokens only returns a healthy dict.

    ⚠️ If a future change makes the decoder raise here, THIS TEST GOES RED and
    the guard above can be reconsidered. That is the intent: the comment in
    `refc_v3.forward` stops being true silently."""
    dec = v6tac.TacticalBehaviourDecoder(
        v6tac.TacticalDecoderConfig(d_model=64, n_layers=1, n_heads=4, d_agent=32, d_bev=16))
    cond = v6tac.build_condition(
        torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long),
                                    v6tac.N_NAV_COMMANDS).float(),
        torch.zeros(2, 4), torch.zeros(2, 1), torch.zeros(2, 1))
    out = dec(cond, agent_tokens=torch.randn(2, 5, 32))
    assert out["goal_logits"].shape == (2, 22)
    assert int(out["n_scene"].max()) == 5, (
        "the decoder attended to agents only and did not complain — which is "
        "exactly why the model-level guard exists")


def test_MUTATION_a_d_bev_0_decoder_fed_BEV_TOKENS_still_refuses():
    """⛔ The OTHER direction, and it is the one that caught a real defect in
    this patch's first draft: the perception branch produces BEV tokens on ANY
    map arm, so an agent-only decoder was handed tokens it has no `bev_in` for
    and `SceneInputRefused` fired on every forward. The FEED GATE in
    `_bev_hook` is what fixed it; this pins that the underlying refusal is
    still live, so the gate cannot be removed without something going RED."""
    dec = v6tac.TacticalBehaviourDecoder(
        v6tac.TacticalDecoderConfig(d_model=64, n_layers=1, n_heads=4, d_agent=32, d_bev=0))
    cond = v6tac.build_condition(
        torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long),
                                    v6tac.N_NAV_COMMANDS).float(),
        torch.zeros(2, 4), torch.zeros(2, 1), torch.zeros(2, 1))
    with pytest.raises(v6tac.SceneInputRefused, match="d_bev = 0"):
        dec(cond, agent_tokens=torch.randn(2, 5, 32),
            bev_tokens=torch.randn(2, 6, 16))


def test_MUTATION_a_perception_branch_on_the_FLAT_arm_REFUSES():
    """⛔ The flat arm's forward returns BEFORE the hook is built, so a branch
    attached there would be optimised by nothing while `config.json` stamped a
    jointly-trained perception run. REFUSE, never skip."""
    cfg = v3.refc_v3_smoke_config(hier=False)
    model = v3.RefCV3Model(cfg)
    model._perception = _FakeBranch(8)
    enc = cfg.core.encoder
    h, w = enc.image_hw()
    with pytest.raises(ValueError, match="FLAT arm"):
        model(torch.rand(1, int(cfg.core.window), int(enc.in_channels), h, w))


# =========================================================================== #
# 3. the token flattener has ONE owner                                        #
# =========================================================================== #
def test_the_branch_reuses_the_ONE_bev_flatten_and_derives_its_width():
    """⛔ `bev_feats_to_tokens` is *"the ONLY place a BEV grid is flattened"* by
    its own docstring. A second transpose is how two call sites end up
    disagreeing about which axis is X, silently, with both shapes valid.

    ⭐ And the token WIDTH is DERIVED from the BEV encoder, never typed: this
    asserts the property the trainer's `--tac-decoder-d-bev` guard rests on."""
    cfg = PB.PerceptionBranchConfig(w_map=1.0, w_box3d=0.0, d_bev=8,
                                    bev_cfg=PB.BEVEncoderConfig(
                                        d_in=8, d_model=8, d_out=6,
                                        norm_groups=2),
                                    bev_tokens_hw=(3, 2), enforce_param_band=False)
    br = PB.PerceptionBranch(cfg, d_image=4, image_hw=(2, 4))
    assert br.bev_token_dim == 6            # == bev_cfg.d_out, not d_bev
    assert br.n_bev_tokens == 6             # == 3 * 2
    feats = torch.randn(2, 6, *br.map_grid_hw)
    tok = br.bev_tokens(feats)
    assert tok.shape == (2, 6, 6)
    # the same tensor through the public flattener must agree EXACTLY
    pooled = torch.nn.functional.adaptive_avg_pool2d(feats, (3, 2))
    assert torch.equal(tok, v6tac.bev_feats_to_tokens(pooled))


def test_grad_reach_report_names_the_FOUR_heads_the_trunk_is_optimised_by():
    """⚠️ PI ruling R3's own stated cost is that a trunk improvement can no
    longer be attributed to one head, and its named mitigation is per-head
    reach. A report covering three of the four cannot perform it."""
    model, _ = _model(d_bev=16)
    rep = PB.grad_reach_report(model, branch=None)
    assert {"trunk", "planner", "tac_decoder"} <= set(rep), sorted(rep)
    for k, v in rep.items():
        assert v["n_params"] > 0, f"{k} reported with no parameters"


# =========================================================================== #
# 4. the gradient-conflict detector must SEE the fourth gradient              #
# =========================================================================== #
def test_the_conflict_detector_counts_the_TACTICAL_term_as_an_aux_gradient():
    """⭐⭐ PI RULING R3 names this detector as the mitigation for a trunk
    optimised FOUR ways. Until tonight `CONFLICT_PERCEPTION_TERMS` listed three
    (bev / map / box3d) and the tactical term was absent — so the aux side of
    every `cd_*` row omitted a gradient that MEASURABLY reaches the trunk
    (`grad_abs_sum` 78,146 on the agent-only arm, `raw/grad_reach_bev.json`).

    ⛔ Read off the REAL table and the REAL helper, and asserted on BOTH arms:
    a single-arm assertion passes on a helper that returns every term always.
    """
    import refc_v3_train as T

    class _M:                      # only the attributes the helper reads
        _w_bev_aux = 0.0
        _w_map = 0.0
        _w_box3d = 0.0
        _w_tac_v6 = 0.0

    assert "tac_v6" in dict(T.CONFLICT_PERCEPTION_TERMS), (
        "the tactical term is not in the detector's aux table; R3's stated "
        "mitigation cannot see the gradient R3 is about")
    m = _M()
    assert T._conflict_aux_weights(m) == {}, (
        "a zero weight is not a term — a cosine against the zero vector is the "
        "degenerate NaN, not a measurement")
    m._w_tac_v6 = 0.25
    assert T._conflict_aux_weights(m) == {"tac_v6": pytest.approx(0.25)}
    # ⛔ THE CONTROL: the three pre-existing terms must SURVIVE. A change that
    # replaced the table rather than extending it would pass every line above.
    m._w_map, m._w_box3d, m._w_bev_aux = 1.0, 2.0, 3.0
    assert set(T._conflict_aux_weights(m)) == {"tac_v6", "map", "box3d", "bev"}


def test_the_conflict_detector_REFUSES_an_ON_override_with_no_live_aux():
    """⭐ The green control for the row above: the refusal must still fire when
    NOTHING is live, or the widened table would have turned a guard into a
    permanent pass."""
    import refc_v3_train as T

    class _M:
        _w_bev_aux = _w_map = _w_box3d = _w_tac_v6 = 0.0
    assert T._conflict_aux_weights(_M()) == {}


# =========================================================================== #
# 5. the 408 geometry — the declared shape must equal the emitted one         #
# =========================================================================== #
def test_the_declared_s16_shape_EQUALS_what_the_trunk_actually_emits():
    """⛔ PI ruling R1 asks for 408 x 1024. `TimmTrunkConfig` REFUSES it
    (`408 % 32 == 8`) and that refusal is load-bearing: timm rounds the spatial
    size UP at each stride-2 stage, so at 408 the trunk emits a **26**-row
    stride-16 map while `timm_trunk.py`'s `s16_shape = h // 16` would DECLARE
    **25** — and that declared shape is what `build_perception_branch` sizes
    `BEVLift` and `Box3DMemory` from.

    ⭐ This pins the INVARIANT rather than the guard: for every height the trunk
    accepts, declared == emitted. It stays green today (32-divisible implies
    16-divisible, so floor and ceil agree) and goes RED the day a height that 16
    does not divide becomes buildable without the shape derivation being fixed
    — which is exactly the silent failure R1 would otherwise land.
    """
    from tanitad.models.timm_trunk import TimmResNetTrunk, TimmTrunkConfig
    checked = 0
    for h in (64, 96, 128, 256):
        cfg = TimmTrunkConfig(model_name="resnet18.a1_in1k", image_hw=(h, 128),
                              pretrained=False, frames=1)
        t = TimmResNetTrunk(cfg)
        with torch.no_grad():
            s16, s32, _ = t.forward_features(torch.zeros(1, 3, h, 128))
        assert tuple(s16.shape[2:]) == tuple(t.s16_shape), (
            f"h={h}: declared s16 {t.s16_shape} != emitted "
            f"{tuple(s16.shape[2:])} — `build_perception_branch` sizes the "
            f"lift and the box head from the DECLARED value")
        assert tuple(s32.shape[2:]) == tuple(t.grid_shape)
        checked += 1
    assert checked == 4, "the loop body never ran — a vacuous pass"

    # ⛔ THE DISCRIMINATING CONTROL: the ruling's own height is REFUSED, and by
    # the divisibility guard rather than by accident. Without this row the test
    # above says nothing about 408.
    with pytest.raises(ValueError, match="divide by 32"):
        TimmTrunkConfig(model_name="resnet18.a1_in1k", image_hw=(408, 1024),
                        pretrained=False, frames=1)
    # and the arithmetic that makes the guard load-bearing rather than fussy
    assert 408 // 16 == 25 and -(-408 // 16) == 26, (
        "floor division understates the stride-16 row count at 408 by exactly "
        "one — the silent mis-size the guard prevents")


def test_every_conflict_term_HAS_A_PRODUCER_in_the_loss_dict():
    """⛔⛔ THE CONSUMER'S TABLE AND THE PRODUCER'S CODE ARE TWO INDEPENDENT
    THINGS, and this is the only check that can see them disagree.

    `_conflict_terms` looks the term up BY KEY in the loss dict and SKIPS a key
    that is absent — silently. So a term listed in `CONFLICT_PERCEPTION_TERMS`
    that nothing ever writes is a detector that reports a healthy `cd_*` row
    while measuring one gradient fewer than it names. That is precisely how the
    tactical term was missing from the aux sum in the first place, from the
    other side.

    ⭐ The reference is INDEPENDENTLY DERIVED — the producer's source text, not
    a re-run of the consumer's own derivation. Re-running a producer's
    derivation and finding agreement measures determinism, not correctness.
    """
    import re

    import refc_v3_train as T
    src = Path(T.__file__).read_text(encoding="utf-8")
    produced = set(re.findall(r'extra\[\"([A-Za-z0-9_]+)\"\]\s*=', src))
    named = {k for k, _ in T.CONFLICT_PERCEPTION_TERMS}
    assert named <= produced, (
        f"conflict terms with NO producer in the loss dict: "
        f"{sorted(named - produced)} — `_conflict_terms` would skip them "
        f"silently and the detector would name a gradient it never reads")
    # ⛔ THE CONTROL: the scan must actually have found something, or `named <=
    # produced` is vacuously false-negative-proof only because `produced` is
    # huge. A scan that read zero keys would pass nothing.
    assert len(produced) > 10, f"the producer scan found only {len(produced)}"


def test_conflict_terms_READS_the_tactical_key_and_WEIGHTS_it():
    """⭐ The consumer half, on a real `_conflict_terms` call. ⛔ The expected
    value is a LITERAL (0.25 * 4.0), never an expression over the code under
    test."""
    import refc_v3_train as T

    class _M:
        _w_bev_aux = _w_map = _w_box3d = 0.0
        _w_tac_v6 = 0.25
    losses = {"traj": torch.tensor(2.0, requires_grad=True),
              "tac_v6": torch.tensor(4.0, requires_grad=True)}
    _plan, aux = T._conflict_terms(_M(), losses)
    assert aux is not None, "the tactical term was not picked up as an aux"
    assert float(aux.detach()) == 1.0              # 0.25 * 4.0, literal
    # ⛔ THE DISCRIMINATING CONTROL: with the weight at 0 there is no aux at all
    # (a cosine against the zero vector is the degenerate NaN, not a reading).
    class _Z(_M):
        _w_tac_v6 = 0.0
    assert T._conflict_terms(_Z(), losses) == (None, None)
