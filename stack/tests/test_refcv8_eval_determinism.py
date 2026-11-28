"""refcv8: the EVAL forward is a function of the inference seed (2026-10-12, the launch gate's G-EVAL control).

MEASURED (pre-gate, Thor CPU, d014ff2, the provisional A2 argv): the trainer's model, forwarded twice on one fixed batch
under ``torch.random.fork_rng`` + ``manual_seed(1234)`` (``launch_gate._forward_out``), gave different bytes on 18
outputs -- every one downstream of the refcv8 extra candidates (``r8_cand_src``, ``u0_hat``, ``sel_score`` ...); the
same at ``OMP_NUM_THREADS=1``. Mechanism (bisected on the dev-box rig): two EVAL draws took their values from the
decoder's persistent ``R8Generator`` -- the allocated candidates' source-anchor choice
(``refcv8_conditioning.select_source_anchors``) and their sampler noise (``refc._sample``'s ``eps_x``). That generator
advances on every call and no seed / fork restores it. Each site alone leaves outputs differing; fixing both reads 0.

The fix (``R8Generator.rand_eval`` / ``randn_eval``): at eval the VALUES come from a fresh generator seeded from the
device's global RNG state (read, never consumed) + the r8 seed + a site salt; the persistent training stream is still
advanced by the identical draw. Pinned here:
  * GREEN: the eval forward repeats bit for bit under a forked seed (model level AND the gate's own ``_forward_out``),
    emission on and off; the refcv7 build is the control that must read 0 before and after;
  * the sampler is still a sampler: a different inference seed draws different extras;
  * the base candidates and the global stream are untouched (vs the d014ff2 semantics);
  * TRAINING is bit-identical to d014ff2 -- outputs, loss, every grad and the generator state, including a train step
    taken AFTER eval forwards (the in-run eval cadence must not move the training trajectory);
  * RED ARM: restoring the d014ff2 semantics (eval values from the persistent stream) must go non-reproducible again.
"""
from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
from tanitad.refs import refcv8_conditioning as C  # noqa: E402

#: the provisional refcv8 A2 recipe as the trainer's `_pin_refcv8` pins it (pre-gate 2026-10-12, argv sha256 d1fa48b2)
#: -- the knobs that build the extra-candidate path. `seed` is the R8Config default (20261004).
A2 = dict(n_alloc=32, alloc_top_k=4, alloc_min_each=2, alloc_emit=True, emit_start=1000, lat_prior_dropout=0.3,
          cond_dropout=0.15, w_cons=0.05, w_sat=0.1, w_alloc_l1=1.0, w_listwise=1.0, w_subscore=0.5,
          speed_input="label", v9_cons=True, w_v9_cons=0.05)
VMAX = (13.8889, 27.7778, 33.3333)


def _build(refcv8: bool = True, emit_live: bool = False):
    T = R.trainer()
    cfg, m = R.build(T, refcv8, seed=0, vmax=True, **(A2 if refcv8 else {}))
    if refcv8:
        m.core.decoder.r8_emit_live = emit_live        # False = refcv8 step 0 under emit_start 1000 (the launch state)
    return T, cfg, m


def _fwd(cfg, m, bt, seed: int = 1234, train: bool = False):
    """The gate's discipline at model level: a FORKED global RNG, re-seeded, then the forward."""
    m.train(train)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        ph = bt["pose_hist"]
        b = int(ph.shape[0])
        with (torch.enable_grad() if train else torch.no_grad()):
            m.core.set_ego_window(ph, int(ph.shape[1]))
            return m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"], steps=cfg.core.decoder.diffusion_steps,
                     v_max_ms=torch.tensor(VMAX[:b]), v_max_valid=torch.ones(b))


def _flat(o, p="out") -> dict:
    r = {}
    if torch.is_tensor(o):
        r[p] = o.detach().clone()
    elif isinstance(o, dict):
        for k in o:
            r.update(_flat(o[k], f"{p}.{k}"))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            r.update(_flat(v, f"{p}[{i}]"))
    return r


def _differ(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b)
                  if k not in a or k not in b or a[k].shape != b[k].shape or not torch.equal(a[k], b[k]))


@contextlib.contextmanager
def _d014ff2_semantics():
    """The pre-fix behaviour: an eval draw takes its VALUES from the persistent stream (what d014ff2's
    `select_source_anchors` and `refc._sample` did in both modes)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(C.R8Generator, "rand_eval", lambda self, shape, device, site: self.rand(shape, device))
    mp.setattr(C.R8Generator, "randn_eval",
               lambda self, shape, device, site, dtype=torch.float32: self.randn(shape, device, dtype))
    try:
        yield
    finally:
        mp.undo()


# =========================================================================== #
# GREEN: the eval forward repeats under a forked seed                          #
# =========================================================================== #
@pytest.mark.parametrize("emit_live", [False, True])
def test_GREEN_the_r8_eval_forward_repeats_bit_for_bit_under_a_forked_seed(emit_live):
    _T, cfg, m = _build(True, emit_live)
    bt = R.batch(cfg, b=3)
    a, b = _flat(_fwd(cfg, m, bt)), _flat(_fwd(cfg, m, bt))
    assert "out.r8_cand_src" in a and int(a["out.r8_cand_src"].shape[1]) == 20 + 32    # the extras ARE built
    assert _differ(a, b) == []


def test_GREEN_the_gates_own_control_reads_zero_on_the_r8_build():
    """`launch_gate._forward_out` itself: compute_losses_v3 under fork_rng + manual_seed(1234), stopped at the model
    output -- the exact call the G-EVAL determinism control makes twice."""
    G = importlib.import_module("launch_gate")
    T, cfg, m = _build(True, emit_live=False)
    m.eval()
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    ph = R.batch(cfg, b=2)["pose_hist"].clone()
    v0 = batch["pose_last"][:, 3].reshape(-1).to(torch.float32)
    ph[:, :, 3] = v0[:, None]
    ph[:, :, 0] = v0[:, None] * 0.1 * torch.arange(ph.shape[1], dtype=torch.float32)[None]
    batch.update(pose_hist=ph, nav_cmd=torch.tensor([1, 2]), nav_valid=torch.tensor([True, True]),
                 v_max_ms=torch.tensor(VMAX[:2]), v_max_valid=torch.ones(2))
    m._w_goal_point = m._w_tac_goal = 0.0
    m._tac_goal_pos_weight = m._tac_goal_class_mask = None

    class _A:
        mode = "diffusion"
    d1 = G.output_digests(G._forward_out(T, m, batch, _A, 1234))
    d2 = G.output_digests(G._forward_out(T, m, batch, _A, 1234))
    assert "out.r8_cand_src" in d1
    assert sorted(k for k in d1 if d2.get(k) != d1[k]) == []


def test_CONTROL_the_refcv7_build_repeats_bit_for_bit():
    """refcv7 passed this gate: no refcv8 seams, no dedicated generator -- must read 0 (it did on d014ff2 too)."""
    _T, cfg, m = _build(False)
    bt = R.batch(cfg, b=3)
    a, b = _flat(_fwd(cfg, m, bt)), _flat(_fwd(cfg, m, bt))
    assert "out.r8_cand_src" not in a and _differ(a, b) == []


# =========================================================================== #
# the sampler is still a sampler; the base candidates and the global stream    #
# =========================================================================== #
def test_a_different_inference_seed_still_draws_different_extras():
    _T, cfg, m = _build(True, emit_live=False)
    bt = R.batch(cfg, b=3)
    a, b = _flat(_fwd(cfg, m, bt, seed=1234)), _flat(_fwd(cfg, m, bt, seed=777))
    d = _differ(a, b)
    assert "out.r8_cand_src" in d and "out.u0_hat" in d


def test_the_base_candidates_and_the_global_stream_are_untouched_vs_d014ff2():
    """New eval vs the d014ff2 semantics on identical models: the BASE columns of every per-candidate output are equal
    (refcv7's noise stream is not touched), and the global RNG leaves the forward in the same state."""
    _T, cfg, m_new = _build(True, emit_live=False)
    _T, _cfg, m_old = _build(True, emit_live=False)
    bt = R.batch(cfg, b=3)
    torch.manual_seed(5)
    new = _flat(_fwd(cfg, m_new, bt))
    with _d014ff2_semantics():
        old = _flat(_fwd(cfg, m_old, bt))
    nb = int(m_new.core.decoder.anchors.shape[0])
    ncand = int(new["out.r8_cand_src"].shape[1])
    assert ncand == nb + 32
    per_cand = [k for k, v in new.items() if v.dim() >= 2 and v.shape[1] == ncand]
    assert {"out.u0_hat", "out.sel_score", "out.anchor_traj"} <= set(per_cand)
    assert [k for k in per_cand if not torch.equal(new[k][:, :nb], old[k][:, :nb])] == []
    assert torch.equal(new["out.traj"], old["out.traj"])          # emission off: the emitted plan is the base's
    # the global stream: an eval forward draws exactly what d014ff2's drew from it
    st = {}
    for tag, sem in (("new", contextlib.nullcontext()), ("old", _d014ff2_semantics())):
        _T, c2, m2 = _build(True, emit_live=False)
        with sem:
            torch.manual_seed(9)
            m2.eval()
            ph = bt["pose_hist"]
            with torch.no_grad():
                m2.core.set_ego_window(ph, int(ph.shape[1]))
                m2(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"], steps=c2.core.decoder.diffusion_steps,
                   v_max_ms=torch.tensor(VMAX), v_max_valid=torch.ones(3))
            st[tag] = torch.get_rng_state().clone()
    assert torch.equal(st["new"], st["old"])


def test_eval_generator_reads_the_global_state_and_never_consumes_it():
    g = C.R8Generator(20261004)
    torch.manual_seed(1234)
    s0 = torch.get_rng_state().clone()
    x = torch.rand(8, generator=g.eval_generator("cpu", C.EVAL_SITE_SELECT_SOURCE))
    assert torch.equal(torch.get_rng_state(), s0)                                     # read, not drawn
    assert torch.equal(x, torch.rand(8, generator=g.eval_generator("cpu", C.EVAL_SITE_SELECT_SOURCE)))
    assert not torch.equal(x, torch.rand(8, generator=g.eval_generator("cpu", C.EVAL_SITE_EXTRA_EPS)))   # site salt
    assert not torch.equal(x, torch.rand(8, generator=C.R8Generator(7).eval_generator("cpu", 1)))      # r8 seed
    torch.rand(1)                                                                     # the global stream moves ...
    assert not torch.equal(x, torch.rand(8, generator=g.eval_generator("cpu", C.EVAL_SITE_SELECT_SOURCE)))



@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device here (the 2026-10-12 dev-box run was CPU-only)")
def test_eval_generator_on_cuda_reads_the_device_state_and_never_consumes_it():
    """The in-run eval runs on the GPU: the CUDA branch reads `torch.cuda.get_rng_state(dev)` -- the generator the BASE
    candidates' `torch.randn(..., device=dev)` draws from -- and must leave it exactly as it found it."""
    torch.manual_seed(1234)
    s0 = torch.cuda.get_rng_state()
    x = C.R8Generator(20261004).rand_eval((64,), "cuda", C.EVAL_SITE_EXTRA_EPS)
    assert torch.equal(torch.cuda.get_rng_state(), s0)
    torch.manual_seed(1234)
    assert torch.equal(x, C.R8Generator(20261004).rand_eval((64,), "cuda", C.EVAL_SITE_EXTRA_EPS))
    torch.randn(1, device="cuda")                                       # the device stream moves ...
    assert not torch.equal(x, C.R8Generator(20261004).rand_eval((64,), "cuda", C.EVAL_SITE_EXTRA_EPS))

# =========================================================================== #
# TRAINING is bit-identical to d014ff2 (outputs, loss, grads, generator)       #
# =========================================================================== #
def _live(o, p="out") -> dict:
    r = {}
    if torch.is_tensor(o) and o.requires_grad and o.is_floating_point():
        r[p] = o
    elif isinstance(o, dict):
        for k in o:
            r.update(_live(o[k], f"{p}.{k}"))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            r.update(_live(v, f"{p}[{i}]"))
    return r


def _train_eval_train(emit_live: bool) -> dict:
    """TRAIN fwd+bwd -> two EVAL forwards -> TRAIN fwd+bwd on one model; every output, the loss, every grad and the
    dedicated generator's state after each phase."""
    _T, cfg, m = _build(True, emit_live)
    bt = R.batch(cfg, b=3)
    res = {}

    def step(tag):
        m.zero_grad(set_to_none=True)
        out = _fwd(cfg, m, bt, seed=2, train=True)
        loss = sum(v.float().mean() for v in _live(out).values())
        loss.backward()
        res[tag] = {"out": _flat(out), "loss": loss.detach().clone(),
                    "grads": {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None},
                    "gen": m.core.decoder.r8_gen.get("cpu").get_state().clone()}
    step("train1")
    _fwd(cfg, m, bt)
    _fwd(cfg, m, bt, seed=777)
    res["gen_after_eval"] = m.core.decoder.r8_gen.get("cpu").get_state().clone()
    step("train2")
    return res


@pytest.mark.parametrize("emit_live", [False, True])
def test_training_is_bit_identical_to_d014ff2_including_a_step_after_eval_forwards(emit_live):
    new = _train_eval_train(emit_live)
    with _d014ff2_semantics():
        old = _train_eval_train(emit_live)
    for tag in ("train1", "train2"):
        assert len(new[tag]["grads"]) > 100 and len(new[tag]["out"]) > 50
        assert _differ(new[tag]["out"], old[tag]["out"]) == []
        assert _differ(new[tag]["grads"], old[tag]["grads"]) == []
        assert torch.equal(new[tag]["loss"], old[tag]["loss"]) and torch.equal(new[tag]["gen"], old[tag]["gen"])
    assert torch.equal(new["gen_after_eval"], old["gen_after_eval"])   # eval advances the TRAINING stream as before


def test_training_never_takes_the_eval_route_and_eval_takes_it_at_both_sites(monkeypatch):
    calls = []
    orig = C.R8Generator.eval_generator
    monkeypatch.setattr(C.R8Generator, "eval_generator",
                        lambda self, device, site: calls.append(int(site)) or orig(self, device, site))
    _T, cfg, m = _build(True, emit_live=False)
    bt = R.batch(cfg, b=3)
    _fwd(cfg, m, bt, seed=2, train=True)
    assert calls == []
    _fwd(cfg, m, bt)
    assert sorted(calls) == sorted([C.EVAL_SITE_SELECT_SOURCE, C.EVAL_SITE_EXTRA_EPS])


# =========================================================================== #
# RED ARM: the d014ff2 semantics are not reproducible                          #
# =========================================================================== #
def test_RED_ARM_the_d014ff2_semantics_do_not_repeat_under_a_forked_seed():
    _T, cfg, m = _build(True, emit_live=False)
    bt = R.batch(cfg, b=3)
    with _d014ff2_semantics():
        a, b = _flat(_fwd(cfg, m, bt)), _flat(_fwd(cfg, m, bt))
    d = _differ(a, b)
    # the pre-gate's list (Thor, 8 decoder layers) less the 4 layers this rig does not have
    assert {"out.r8_cand_src", "out.u0_hat", "out.sel_score", "out.sel_score_v3", "out.refined_logits",
            "out.r8_sub_logits", "out.anchor_traj", "out.offset", "out.goal_dist"} <= set(d)
    assert "out.traj" not in d                                     # emission off: the emitted plan never moved


# =========================================================================== #
# the emission switch is BORN at refcv8 step 0 (trainer-vs-loader r8_emit_keep) #
# =========================================================================== #
def test_a_fresh_build_is_born_at_step0_of_its_emission_schedule():
    """MEASURED (pre-gate d6/d7, Thor): G-EVAL's trainer model -- built, never stepped -- differed from the loader's on
    `out.r8_emit_keep`: the switch defaulted to True while the loader schedules the checkpoint's step 0 (False under
    emit_start 1000). Literals: emit_start 1000 -> not live at step 0; emit_start 0 -> live (the tiny arms)."""
    T = R.trainer()
    _c, m_late = R.build(T, True, seed=0, vmax=True, **A2)                             # emit_start 1000
    _c, m_zero = R.build(T, True, seed=0, vmax=True, **{**A2, "emit_start": 0})
    assert m_late.core.decoder.r8_emit_live is False
    assert m_zero.core.decoder.r8_emit_live is True


def test_GREEN_a_never_scheduled_build_and_the_loaders_step0_schedule_give_the_same_forward():
    """The G-EVAL comparison in miniature: the trainer side never calls the schedule; the loader side applies it at
    `checkpoint_r8_step(keys, 0)` exactly as `refcv7_loader.py:517-519` does for the gate's {"step": 0} checkpoint."""
    from tanitad.train import refcv8_train as RT
    T = R.trainer()
    cfg, m_t = R.build(T, True, seed=0, vmax=True, **A2)
    _c, m_l = R.build(T, True, seed=0, vmax=True, **A2)
    m_l.load_state_dict(m_t.state_dict())
    assert RT.apply_emit_schedule(m_l, RT.checkpoint_r8_step(m_t.state_dict().keys(), 0)) is False
    bt = R.batch(cfg, b=3)
    a, b = _flat(_fwd(cfg, m_t, bt)), _flat(_fwd(cfg, m_l, bt))
    assert "out.r8_emit_keep" in a and _differ(a, b) == []
