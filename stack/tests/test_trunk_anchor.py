"""D-V7-TRUNK-ANCHOR — ``--w-trunk-anchor`` is WIRED, and its monitor with it.

`PREREG_V7F.md` rung R3 asks whether a trainable DINO trunk beats a frozen one
*without* the Observer Effect. Its ``anchored`` arm could not run:
``--w-trunk-anchor`` was DECLARED but NOT WIRED and refused at any non-zero
value (D-V7-DINO-SEED). This file pins the wiring.

⛔ **THE LOAD-BEARING TEST IS THE FIRST ONE.** ``--w-trunk-anchor 0.0`` (the
default) must be BIT-IDENTICAL to the pre-change trainer — no module, no forward
hook, no extra log key, no optimizer-group change. Everything else here is a
refusal or a control; that one protects every arm already trained.

The rest, in the order the brief names them:
  * a non-zero weight with **no seed** REFUSES, **naming the flag** — a frozen
    copy of a randomly-initialised trunk is a random-feature regulariser
    wearing an anchor's name;
  * the anchor's gradient reaches the **trunk** and **not the teacher**;
  * the teacher stays frozen (``requires_grad=False``, outside every optimizer
    group) **across a real step**;
  * the monitor reads a **known value** on an unchanged encoder and **MOVES**
    when the encoder is perturbed — the deliberate-regression control, without
    which a monitor that never trips proves nothing (§6.2b: *"if the monitor
    does not trip `full`, the monitor is VOID"*).

⭐ Two controls here read values that are known EXACTLY, which is what makes the
rest of the panel trustworthy (the 2026-08-22 rule): the anchor against its own
untouched teacher is **exactly 0.0**, and the monitor's constant-only control is
**exactly 0.0**.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
_ROOT = _STACK.parent
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

import train_v6_staged as T  # noqa: E402
from train_v6_staged import (  # noqa: E402
    ENCODER_SEED_FORMAT, ENCODER_SEED_MARKER, EncoderTokenTap,
    OBS_MONITOR_FLOOR, ObserverEffectMonitor, TRUNK_ANCHOR_CAVEAT, TrunkAnchor,
    V6LossWeights, anchor_and_monitor_step, apply_encoder_seed,
    assert_trunk_anchor_preflight, build_observer_monitor, build_parser,
    build_stack_from_args, build_trunk_anchor, build_trunk_optimizer,
    observer_targets, synthetic_train_batch, trunk_anchor_loss, v6_loss_step,
)

BASE = ["--out", "UNUSED", "--stage", "S-W", "--frame-h", "32",
        "--frame-w", "64", "--enc-dim", "24", "--enc-depth", "1",
        "--enc-heads", "2", "--patch", "16", "--readout-grid", "2",
        "--readout-grid-w", "2", "--readout-dim", "12", "--pred-dim", "24",
        "--pred-depth", "1", "--pred-heads", "2", "--window", "3",
        "--d-tac", "12", "--d-str", "8"]

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _args(*extra):
    return build_parser().parse_args(BASE + list(extra))


def _stack(*extra):
    return build_stack_from_args(_args(*extra))


def _batch(st, *, batch=2, k=3, seed=7):
    b = synthetic_train_batch(st, batch=batch, k=k, seed=seed)
    b["gt_wp"] = torch.randn(batch, 3, 2,
                             generator=torch.Generator().manual_seed(seed + 1))
    return b


def _write_seed(tmp_path, stack, *, model_id=MODEL_ID, drop=("pos",)):
    """A VALID encoder seed carrying ``stack.encoder``'s own tensors.

    ⚠️ Deliberately built here rather than through
    ``dinov3_seed_checkpoint.py``: this file tests the ANCHOR, and a test that
    also had to reproduce the DINOv3 conversion would fail for the converter's
    reasons. ``drop`` reproduces the one irreducible loss the converter
    declares — DINOv3 is RoPE-only, so ``pos`` is left at its own init."""
    sd = {k: v.clone() for k, v in stack.encoder.state_dict().items()
          if k not in drop}
    ec = stack.cfg.encoder
    ih, iw = ec.image_hw()
    prov = {
        "format": ENCODER_SEED_FORMAT, "dinov3_model_id": model_id,
        "dinov3_variant": "vitb16", "source_sha256": "0" * 64,
        "left_at_init_keys": sorted(drop),
        "layer_scale_folded": True, "n_mapped": len(sd),
        "n_skipped_allowlist": 0, "n_left_at_init": len(drop),
        "declared_losses": ["`pos` is LEFT AT ITS OWN INIT (DINOv3 is "
                            "RoPE-only)"],
        "target_geometry": {
            "class": type(stack.encoder).__name__,
            "d_model": int(ec.d_model), "depth": int(ec.depth),
            "n_heads": int(ec.n_heads), "patch_size": int(ec.patch_size),
            "in_channels": int(ec.in_channels), "image_size": int(ih),
            "image_width": int(iw), "n_tokens": int(stack.encoder.n_tokens)},
    }
    p = Path(tmp_path) / "seed.pt"
    torch.save({ENCODER_SEED_MARKER: 1, "encoder": sd, "_provenance": prov}, p)
    return p


def _seeded(tmp_path, *extra):
    """``(args, stack, seed_report)`` for an arm that really loaded a seed."""
    st = _stack()
    p = _write_seed(tmp_path, st)
    a = _args("--init-encoder-from", str(p), *extra)
    rep = apply_encoder_seed(a, st)
    assert rep["init_encoder_from"] == str(p)
    return a, st, rep


# ===========================================================================
# 1. ⛔ THE LOAD-BEARING ONE — the default is bit-identical
# ===========================================================================

def _repo_root():
    """The git checkout to read history from.

    ⚠️ THE SUITE OFTEN RUNS FROM A NON-GIT MIRROR (the G: mount cannot
    execute the stack), and a git call there fails in a way indistinguishable
    from "no such revision" -- i.e. the load-bearing identity test would SKIP
    forever and nobody would notice. So: ``TANITAD_REPO`` if set, else the
    first ancestor carrying a ``.git``, else ``None`` and an honest skip."""
    env = os.environ.get("TANITAD_REPO")
    if env and (Path(env) / ".git").exists():
        return Path(env)
    for d in [_ROOT, *_ROOT.parents]:
        if (d / ".git").exists():
            return d
    return None


_SBS_CACHE: dict = {}


def _side_by_side(rel: str, marker: str, modname: str):
    """Newest revision of ``rel`` that does NOT contain ``marker``.

    Resolved by CONTENT, not by ``HEAD`` (C75): HEAD moves under us — a
    sibling's whole-index commit sweeps in-progress files into it — and a
    HEAD-relative identity test then compares a module with itself and passes
    forever. ``None`` ⇒ the caller SKIPS; a self-comparison dressed as a real
    test is worse than an honest skip."""
    if modname in _SBS_CACHE:
        return _SBS_CACHE[modname]
    _SBS_CACHE[modname] = None
    root = _repo_root()
    if root is None:
        return None
    try:
        # ⚠️ BOUNDED. An unbounded `git log -- <path>` walks the whole
        # history and takes MINUTES on the G: mount, which turns a correctness
        # test into a suite-wide stall. The revision we need is the newest one
        # lacking the marker, so 40 revisions of THIS file is ample.
        log = subprocess.run(["git", "log", "--format=%H", "-n", "40",
                              "--", rel],
                             cwd=root, capture_output=True, timeout=45)
        if log.returncode != 0:
            return None
        for sha in log.stdout.decode().split():
            r = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=root,
                               capture_output=True, timeout=45)
            if r.returncode != 0 or not r.stdout:
                continue
            if marker.encode() in r.stdout:
                continue
            src, ref = r.stdout, sha
            break
        else:
            return None
    except Exception:
        return None
    tmp = Path(tempfile.mkdtemp()) / f"{modname}.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location(modname, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    mod._ref = ref
    _SBS_CACHE[modname] = mod
    return mod


def test_the_default_constructs_NOTHING_and_registers_no_hook():
    """⛔ THE FIRST HALF OF BYTE-IDENTITY: no object, no hook, no group change.

    A forward hook is invisible in a loss value and in a state_dict, so it
    cannot be caught by a numeric comparison — it has to be asserted directly.
    """
    a = _args()
    assert float(a.w_trunk_anchor) == 0.0
    assert int(a.obs_monitor_every) == 0
    st = _stack()
    assert build_trunk_anchor(a, st, {"init_encoder_from": None}) is None
    assert build_observer_monitor(a, st) is None
    assert assert_trunk_anchor_preflight(a) == 0.0
    st.eval()
    L = v6_loss_step(st, _batch(st), stage="S-W", weights=V6LossWeights(),
                     o1_k=3, o5_k=3)
    assert not st.encoder._forward_hooks, \
        "a forward hook survived a default-path loss step"
    leaked = [k for k in L["log"]
              if k.startswith(("trunk_anchor", "obs_", "observer"))]
    assert not leaked, f"the default log grew keys: {leaked}"
    trainable = [p for p in st.parameters() if p.requires_grad]
    _, rep = build_trunk_optimizer(a, st, trainable)
    assert rep["n_groups"] == 1 and rep["trunk_lr_split"] is False


def test_v6_loss_step_ITSELF_never_mentions_the_anchor_or_the_monitor():
    """⛔ THE GIT-FREE HALF OF BYTE-IDENTITY, and it ALWAYS runs.

    The identity test above needs a git reference, and on this repo's mount
    git can take MINUTES under load, so it is allowed to SKIP. That must not
    leave the guarantee unprotected. What makes the default path
    bit-identical is STRUCTURAL: the anchor and the monitor compose with the
    objective at the O7-sibling site in ``train()``, AFTER ``v6_loss_step``
    has returned, and ``v6_loss_step`` itself was not edited. This asserts
    exactly that -- if a later change moves the term INSIDE the loss
    function, the byte-identity argument weakens and this test says so
    before anyone has to re-derive it."""
    src = inspect.getsource(v6_loss_step)
    for name in ("trunk_anchor", "EncoderTokenTap", "ObserverEffectMonitor",
                 "obs_monitor", "anchor_and_monitor_step",
                 "TRUNK_ANCHOR_CAVEAT"):
        assert name not in src, (
            f"`{name}` appeared inside v6_loss_step -- the default path is "
            f"no longer structurally untouched")
    # ... and the composition really does happen at the O7-sibling site
    tsrc = inspect.getsource(T.train)
    assert "anchor_and_monitor_step(anchor, obs_mon, tap, L, batch, step)" in tsrc
    assert "tap.arm()" in tsrc and "tap.disarm()" in tsrc


@pytest.mark.parametrize("stage", ["S-W", "S-T"])
def test_the_default_loss_is_bit_identical_to_the_PRE_ANCHOR_trainer(stage):
    """⛔ THE ONE THAT PROTECTS EVERY ARM ALREADY TRAINED.

    The pre-change trainer against the current one, MODEL held fixed (both run
    against the current ``tanitad.models.v6``) so any difference is mine."""
    old = _side_by_side("stack/scripts/train_v6_staged.py",
                        "TRUNK_ANCHOR_CAVEAT", "train_v6_pre_trunk_anchor")
    if old is None:
        pytest.skip("git could not produce a pre-anchor trainer revision")
    st = _stack()
    st.eval()                       # the incumbent path is not reproducible
    b = _batch(st)
    kw = dict(stage=stage, o1_k=3, o5_k=3)
    torch.manual_seed(3)
    lo = old.v6_loss_step(st, b, weights=old.V6LossWeights(),
                          generator=torch.Generator().manual_seed(11), **kw)
    torch.manual_seed(3)
    ln = v6_loss_step(st, b, weights=V6LossWeights(),
                      generator=torch.Generator().manual_seed(11), **kw)
    assert torch.equal(lo["loss"].detach(), ln["loss"].detach()), \
        f"{stage}: the DEFAULT loss MOVED against {old._ref}"
    assert set(lo["log"]) == set(ln["log"]), \
        f"{stage}: the DEFAULT log keys moved against {old._ref}"
    for k in lo["log"]:
        if isinstance(lo["log"][k], float):
            assert lo["log"][k] == ln["log"][k], f"{stage}: log[{k}] moved"


# ===========================================================================
# 2. the refusals — each names the flag that fixes it
# ===========================================================================

def test_a_nonzero_weight_with_NO_SEED_refuses_and_names_the_flag():
    a = _args("--w-trunk-anchor", "1.0", "--obs-monitor-every", "10")
    with pytest.raises(SystemExit) as e:
        assert_trunk_anchor_preflight(a)
    msg = str(e.value)
    assert "--init-encoder-from" in msg and "--w-trunk-anchor" in msg
    assert "random" in msg.lower(), \
        "the refusal must say WHY a seedless anchor is meaningless"


def test_a_nonzero_weight_WITHOUT_THE_MONITOR_refuses(tmp_path):
    """§6.2b: the monitor is what makes a trainable trunk defensible.

    An anchor running with nothing watching the corruption it exists to
    prevent is a weight, not a defence — and R3 scores the ``anchored`` arm on
    the monitor, not on the term."""
    st = _stack()
    p = _write_seed(tmp_path, st)
    a = _args("--w-trunk-anchor", "1.0", "--init-encoder-from", str(p))
    with pytest.raises(SystemExit) as e:
        assert_trunk_anchor_preflight(a)
    assert "--obs-monitor-every" in str(e.value)


def test_a_negative_weight_refuses():
    a = _args("--w-trunk-anchor", "-1.0")
    with pytest.raises(SystemExit) as e:
        assert_trunk_anchor_preflight(a)
    assert "negative" in str(e.value)


def test_the_teacher_model_id_is_CROSS_CHECKED_against_the_seed_stamp(tmp_path):
    """``--trunk-anchor-model`` stops being inert: a disagreement REFUSES.

    Reason (a) of the three that made O7's teacher unusable was *anchoring to a
    different network*. A separately-named teacher could reintroduce it
    silently; a checked one cannot."""
    st = _stack()
    p = _write_seed(tmp_path, st, model_id="facebook/dinov3-vitl16-pretrain")
    a = _args("--w-trunk-anchor", "1.0", "--obs-monitor-every", "10",
              "--init-encoder-from", str(p))          # default model id: vitb16
    rep = apply_encoder_seed(a, st)
    with pytest.raises(SystemExit) as e:
        build_trunk_anchor(a, st, rep)
    msg = str(e.value)
    assert "vitl16" in msg and "vitb16" in msg
    # ... and it PASSES once the two agree
    a2 = _args("--w-trunk-anchor", "1.0", "--obs-monitor-every", "10",
               "--init-encoder-from", str(p),
               "--trunk-anchor-model", "facebook/dinov3-vitl16-pretrain")
    assert build_trunk_anchor(a2, st, rep) is not None


def test_an_anchor_on_a_FROZEN_trunk_refuses(tmp_path):
    a, st, rep = _seeded(tmp_path, "--w-trunk-anchor", "1.0",
                         "--obs-monitor-every", "10")
    for p in st.encoder.parameters():
        p.requires_grad_(False)
    with pytest.raises(SystemExit) as e:
        build_trunk_anchor(a, st, rep)
    assert "frozen" in str(e.value).lower()


# ===========================================================================
# 3. the tap — the exact live tokens, or a refusal
# ===========================================================================

def test_the_tap_returns_the_EXACT_live_trunk_tokens():
    st = _stack()
    st.eval()
    frames = torch.randn(2, 3, st.cfg.encoder.in_channels,
                         *st.cfg.encoder.image_hw())
    tap = EncoderTokenTap(st.encoder).arm()
    z = st.encode_window(frames)
    tap.disarm()
    enc_in, tok = tap.one()
    assert enc_in.shape[0] == 2 * 3
    direct = st.encoder(frames.reshape(6, *frames.shape[2:]))
    assert torch.equal(tok, direct), "the tap did not capture the live tokens"
    assert z.shape[:2] == (2, 3)


def test_the_tap_REFUSES_on_a_capture_count_other_than_one():
    """A second ``stack.encoder`` call would make the anchor pull on whichever
    tensor landed last. It refuses instead of guessing."""
    st = _stack()
    tap = EncoderTokenTap(st.encoder).arm()
    with pytest.raises(RuntimeError, match="captured 0"):
        tap.one()
    x = torch.randn(1, st.cfg.encoder.in_channels, *st.cfg.encoder.image_hw())
    st.encoder(x)
    st.encoder(x)
    with pytest.raises(RuntimeError, match="captured 2"):
        tap.one()
    tap.disarm()
    assert not st.encoder._forward_hooks


# ===========================================================================
# 4. the term — L2, and a control that reads a value known EXACTLY
# ===========================================================================

def test_the_anchor_against_its_OWN_untouched_teacher_is_EXACTLY_zero(tmp_path):
    """⭐ THE CONTROL THAT MUST READ A KNOWN VALUE.

    The teacher is a ``deepcopy`` of the live trunk taken before any step, so
    at step 0 the two token fields are the SAME tensor field and the MSE is
    exactly 0. Any other reading means the teacher is not the seed."""
    a, st, rep = _seeded(tmp_path, "--w-trunk-anchor", "1.0",
                         "--obs-monitor-every", "10")
    anch = build_trunk_anchor(a, st, rep)
    st.eval()
    frames = torch.randn(2, 3, st.cfg.encoder.in_channels,
                         *st.cfg.encoder.image_hw())
    tap = EncoderTokenTap(st.encoder).arm()
    st.encode_window(frames)
    tap.disarm()
    enc_in, tok = tap.one()
    term, log = anch.loss(enc_in, tok, b=2, w=3)
    # ⚠️ 0 UP TO FLOATING-POINT BATCH-SHAPE NOISE, NOT BIT-EXACTLY 0 --
    # MEASURED and stated rather than papered over. The live trunk runs on
    # B*W images and the teacher on B (newest-frame only), and a different
    # batch extent selects a different GEMM blocking, so the same weights on
    # the same pixels differ in the last bits. MEASURED here: MSE 1.5e-14 and
    # rel_drift ~1e-8, i.e. 6+ orders below anything the anchor must resolve.
    assert float(term.detach()) < 1e-9,         f"anchor against its OWN teacher read {float(term.detach()):.3e} -- "         f"that is not floating-point noise, so the teacher is not the seed"
    assert log["trunk_anchor_rel_drift"] < 1e-5
    assert log["trunk_anchor_form"] == "l2_mse_per_patch_token"


def test_the_form_is_L2_and_not_COSINE():
    """A pure RESCALE of the tokens is invisible to cosine and must not be
    invisible here — §10 D7 asks for an MSE anchor, and the norms are half of
    what the Observer Effect destroys."""
    z = torch.randn(4, 7, 16)
    assert float(trunk_anchor_loss(z, z)) == 0.0
    assert float(trunk_anchor_loss(2.0 * z, z)) > 0.0
    cos = torch.nn.functional.cosine_similarity(
        (2.0 * z).reshape(-1, 16), z.reshape(-1, 16), dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5), \
        "the fixture must be a pure rescale for the contrast to mean anything"


def test_the_anchor_gradient_reaches_the_TRUNK_and_NOT_the_teacher(tmp_path):
    a, st, rep = _seeded(tmp_path, "--w-trunk-anchor", "1.0",
                         "--obs-monitor-every", "10")
    anch = build_trunk_anchor(a, st, rep)
    # perturb the LIVE trunk so the term is non-zero and has a gradient
    with torch.no_grad():
        for p in st.encoder.parameters():
            p.add_(0.05 * torch.randn_like(p))
    frames = torch.randn(2, 3, st.cfg.encoder.in_channels,
                         *st.cfg.encoder.image_hw())
    tap = EncoderTokenTap(st.encoder).arm()
    st.encode_window(frames)
    tap.disarm()
    enc_in, tok = tap.one()
    term, _ = anch.loss(enc_in, tok, b=2, w=3)
    assert float(term.detach()) > 0.0
    st.zero_grad(set_to_none=True)
    term.backward()
    live_g = [p.grad for _, p in st.encoder.named_parameters()
              if p.grad is not None]
    assert live_g, "the anchor's gradient never reached the trunk"
    assert any(float(g.abs().sum()) > 0 for g in live_g), \
        "the trunk's anchor gradient is identically zero"
    assert all(p.grad is None for p in anch.teacher.parameters()), \
        "the TEACHER received a gradient — it is not frozen"
    # and nothing outside the encoder was touched by THIS term alone
    other = [n for n, p in st.named_parameters()
             if p.grad is not None and st.group_of(n) != "encoder"]
    assert not other, f"the anchor reached non-trunk parameters: {other[:5]}"


def test_the_teacher_stays_frozen_and_OUTSIDE_every_optimizer_group(tmp_path):
    """``requires_grad=False``, absent from every param group, and BYTE-EQUAL
    after a real ``opt.step()``.

    Two guards, not one: the teacher is a free-standing object (so
    ``stack.parameters()`` cannot sweep it) AND every tensor is frozen."""
    a, st, rep = _seeded(tmp_path, "--w-trunk-anchor", "1.0",
                         "--obs-monitor-every", "10", "--lr", "0.1")
    anch = build_trunk_anchor(a, st, rep)
    assert all(not p.requires_grad for p in anch.teacher.parameters())
    before = [p.detach().clone() for p in anch.teacher.parameters()]
    trainable = [p for p in st.parameters() if p.requires_grad]
    t_ids = {id(p) for p in anch.teacher.parameters()}
    assert not (t_ids & {id(p) for p in trainable}), \
        "a teacher tensor is inside `trainable`"
    opt, _ = build_trunk_optimizer(a, st, trainable)
    for grp in opt.param_groups:
        assert not (t_ids & {id(p) for p in grp["params"]}), \
            "a teacher tensor is inside an optimizer group"
    st.train()
    b = _batch(st)
    tap = EncoderTokenTap(st.encoder).arm()
    L = v6_loss_step(st, b, stage="S-W", weights=V6LossWeights(),
                     o1_k=3, o5_k=3)
    tap.disarm()
    anchor_and_monitor_step(anch, None, tap, L, b, 1)
    assert "trunk_anchor" in L["log"]
    opt.zero_grad(set_to_none=True)
    L["loss"].backward()
    opt.step()
    for i, p in enumerate(anch.teacher.parameters()):
        assert torch.equal(p, before[i]), \
            f"teacher tensor {i} MOVED across an optimizer step"
    assert all(p.grad is None for p in anch.teacher.parameters())


# ===========================================================================
# 5. the monitor — a known value, a floor, and the deliberate regression
# ===========================================================================

def _feed(mon, enc, frames, targ, *, chunk=8):
    for i in range(0, frames.shape[0], chunk):
        f = frames[i:i + chunk]
        with torch.no_grad():
            mon.observe(enc(f), f, targ[i:i + chunk])


def test_the_monitor_reads_a_KNOWN_VALUE_and_MOVES_when_the_encoder_is_perturbed():
    """⭐ THE DELIBERATE-REGRESSION CONTROL.

    §6.2b: *"if the monitor does not trip `full`, the monitor is VOID"*. A
    monitor that never moves cannot certify an ``anchored`` arm, so the moving
    half is tested as hard as the reading half.

      * unchanged encoder + a target that IS linear in its features ⇒ rho ~ 1,
        and the read is DETERMINISTIC (the same buffer reads the same number);
      * a completely different trunk on the same targets ⇒ rho collapses.
    """
    st = _stack()
    st.eval()
    enc = st.encoder
    n = 96
    g = torch.Generator().manual_seed(5)
    frames = torch.randn(n, st.cfg.encoder.in_channels,
                         *st.cfg.encoder.image_hw(), generator=g)
    mon = ObserverEffectMonitor(d_model=int(st.cfg.encoder.d_model),
                                dims=8, window=256, every=1, seed=0)
    with torch.no_grad():
        feat = enc(frames).float().mean(dim=1) @ mon.R
    w = torch.randn(mon.dims, 3, generator=g)
    targ = feat @ w                       # exactly linearly decodable
    _feed(mon, enc, frames, targ)
    r1 = mon.read(1)
    assert r1["observer_effect"] in ("OK", "TRIPPED")
    assert r1["obs_n"] == n and r1["obs_d"] == 8
    assert r1["obs_rho_dynamic"] > 0.9, r1["obs_rho_dynamic"]
    # deterministic: the same buffer reads the same number
    assert mon.read(1)["obs_rho_dynamic"] == r1["obs_rho_dynamic"]

    # ---- the regression arm: a DIFFERENT trunk, same targets ---------------
    torch.manual_seed(999)
    st2 = _stack()
    st2.eval()
    mon2 = ObserverEffectMonitor(d_model=int(st.cfg.encoder.d_model),
                                 dims=8, window=256, every=1, seed=0)
    assert torch.equal(mon2.R, mon.R), "the probe basis must be the SAME"
    _feed(mon2, st2.encoder, frames, targ)
    r2 = mon2.read(1)
    assert r2["obs_rho_dynamic"] < 0.5 * r1["obs_rho_dynamic"], \
        (f"the monitor did NOT move on a different trunk: "
         f"{r1['obs_rho_dynamic']} -> {r2['obs_rho_dynamic']} — a monitor that "
         f"cannot trip is VOID (PREREG_V7F §6.2b)")


def test_the_monitor_controls_read_their_known_values():
    """constant-only EXACTLY 0.0 · time-shuffled at the floor · n and d printed.

    Three of the four 2026-08-22 probe failures were caught ONLY because a
    control read the same value as the thing being measured."""
    st = _stack()
    st.eval()
    g = torch.Generator().manual_seed(11)
    n = 96
    frames = torch.randn(n, st.cfg.encoder.in_channels,
                         *st.cfg.encoder.image_hw(), generator=g)
    mon = ObserverEffectMonitor(d_model=int(st.cfg.encoder.d_model),
                                dims=8, window=256, every=1, seed=0)
    with torch.no_grad():
        feat = st.encoder(frames).float().mean(dim=1) @ mon.R
    targ = feat @ torch.randn(mon.dims, 3, generator=g)
    _feed(mon, st.encoder, frames, targ)
    r = mon.read(1)
    assert r["obs_rho_constant"] == 0.0, "the constant control must be EXACT"
    assert abs(r["obs_rho_shuffled"]) < 0.5, (
        f"the time-shuffled control read {r['obs_rho_shuffled']} - a "
        f"probe that scores high on shuffled features is measuring itself")
    assert r["obs_rho_dynamic"] > r["obs_rho_shuffled"]
    assert "obs_rho_pixel" in r and "obs_beats_pixel_floor" in r
    assert r["obs_n"] == n and r["obs_d"] == mon.dims
    assert r["obs_ridge_lambda"] == 1.0, "lambda must be FIXED, never selected"


def test_an_UNDERPOWERED_buffer_says_so_instead_of_emitting_a_number():
    """n << d correctly chooses maximal shrinkage and EVERY arm then reads the
    floor — the 2026-08-22 failure #4. It is refused, not reported."""
    mon = ObserverEffectMonitor(d_model=24, dims=16, window=256, every=1)
    r = mon.read(1)
    assert r["observer_effect"] == "UNDERPOWERED"
    assert r["obs_n"] == 0 and r["obs_n_needed"] == 64
    assert "obs_rho_dynamic" not in r


def test_the_CAVEAT_rides_in_the_code_output_not_only_in_a_doc(capsys):
    """⚠️ BINDING (D-V7-DINO-SEED): the step-0 control does NOT measure the
    published DINOv3 — ``pos`` is left at its own init (DINOv3 is RoPE-only,
    ``ViTEncoder`` is learned-APE-only) and CLS/register/mask are dropped. No
    reading may be quoted beside rho 0.91 without that sentence, so the
    sentence travels WITH the reading."""
    for frag in ("RoPE-only", "learned-APE-only", "0.91", "register"):
        assert frag in TRUNK_ANCHOR_CAVEAT
    assert "constant-only" in OBS_MONITOR_FLOOR and "EXACTLY 0.0" in OBS_MONITOR_FLOOR
    st = _stack()
    a = _args("--obs-monitor-every", "5")
    mon = build_observer_monitor(a, st)
    assert mon is not None
    printed = capsys.readouterr().out
    assert TRUNK_ANCHOR_CAVEAT in printed, \
        "the caveat must be printed by the code, not only written in a doc"
    assert OBS_MONITOR_FLOOR in printed
    rec = mon.read(1)
    assert rec["obs_caveat"] == TRUNK_ANCHOR_CAVEAT
    assert rec["obs_floor"] == OBS_MONITOR_FLOOR


def test_the_monitor_targets_are_the_DYNAMIC_ones_taken_from_the_batch():
    st = _stack()
    b = _batch(st)
    t = observer_targets(b)
    assert t.shape == (2, 3)
    assert torch.equal(t[:, 0], b["v0"].float())
    assert torch.equal(t[:, 1], b["actions2"][:, -1, 0].float())
    assert ObserverEffectMonitor.TARGETS == ("speed", "steer", "accel")


def test_the_monitor_runs_ALONE_without_an_anchor(tmp_path):
    """R3's ``full`` and ``frozen`` arms carry the monitor and NO anchor —
    that is exactly how the monitor gets to trip its own regression arm."""
    st = _stack()
    a = _args("--obs-monitor-every", "1")
    assert assert_trunk_anchor_preflight(a) == 0.0
    assert build_trunk_anchor(a, st, {"init_encoder_from": None}) is None
    mon = build_observer_monitor(a, st)
    tap = EncoderTokenTap(st.encoder).arm()
    b = _batch(st)
    L = v6_loss_step(st, b, stage="S-W", weights=V6LossWeights(), o1_k=3,
                     o5_k=3)
    tap.disarm()
    before = float(L["loss"].detach())
    anchor_and_monitor_step(None, mon, tap, L, b, 1)
    assert float(L["loss"].detach()) == before, \
        "the monitor must not touch the loss"
    assert "trunk_anchor" not in L["log"]
    assert L["log"]["observer_effect"] == "UNDERPOWERED"
