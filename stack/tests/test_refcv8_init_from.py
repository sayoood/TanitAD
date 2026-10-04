"""refcv8 WP-B -- ``--init-from``: the WARM START as the trainer performs it (``refcv8_train.warm_start_from``).

Pinned on the real CPU rig (``_refcv8_rig``: DDIM control-space sampler, residual prior, behaviour decoder):
1. a refcv7 checkpoint loaded into a FRESH refcv8 build (every seam on, allocation on) reproduces refcv7's forward bit
   for bit (plan, pick, base fan, scores) -- the DESIGN sec. 3.7 claim, through the trainer's own load path;
2. the record: md5, source step, keys loaded, new refcv8 keys (> 0), fresh optimiser;
3. every way the load could silently be a different experiment is REFUSED: a source key the build lacks, a missing key
   that is not a refcv8 seam, a shape mismatch, a missing file;
4. the train() wiring: resume wins over --init-from, and the warm start happens BEFORE G-DVB / config.json.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
from tanitad.train import refcv8_train as RT  # noqa: E402

KEYS = ("traj", "sel_idx", "sel_score_v3", "anchor_logits", "traj_base", "sel_idx_base")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    T = R.trainer()
    cfg7, m7 = R.build(T, False)
    bt = R.batch(cfg7)
    o7 = R.forward(cfg7, m7, bt)
    p = tmp_path_factory.mktemp("initfrom") / "refcv7_like.pt"
    torch.save({"model": m7.state_dict(), "step": 50400}, str(p))
    return T, cfg7, m7, bt, o7, p


def test_the_warm_start_reproduces_the_source_forward_bit_for_bit(src):
    T, cfg7, m7, bt, o7, p = src
    cfg8, m8 = R.build(T, True, seed=7)        # every seam on, a DIFFERENT init: nothing is shared by luck
    rec = RT.warm_start_from(m8, str(p))
    assert rec["source_step"] == 50400 and rec["n_new_refcv8_keys"] > 0 and rec["optimizer"].startswith("fresh")
    assert rec["n_loaded"] == len(m7.state_dict()) and len(rec["md5"]) == 32
    o8 = R.forward(cfg8, m8, bt)
    for k in KEYS:
        if k in o7:
            a, b = o7[k], o8[k]
            if k in ("anchor_logits",):
                b = b[:, :a.shape[1]]
            assert torch.equal(a, b), k


def test_DELIBERATE_REGRESSION_a_build_with_a_moved_seam_gate_is_not_the_identity(src):
    """The identity above must be able to fail: open one zero-init gate after the load -> the plan moves."""
    T, cfg7, m7, bt, o7, p = src
    cfg8, m8 = R.build(T, True, seed=7)
    RT.warm_start_from(m8, str(p))
    with torch.no_grad():
        m8.core.decoder.r8_mod.proj[-1].weight.normal_(0.0, 0.1)
    o8 = R.forward(cfg8, m8, bt)
    assert not torch.equal(o7["sel_score_v3"], o8["sel_score_v3"][:, :o7["sel_score_v3"].shape[1]])


def test_every_silent_mismatch_is_refused(src, tmp_path):
    T, cfg7, m7, bt, o7, p = src
    sd = m7.state_dict()
    cfg8, m8 = R.build(T, True, n_alloc=4)
    extra = dict(sd, **{"core.decoder.not_in_the_build": torch.zeros(1)})
    torch.save({"model": extra}, str(tmp_path / "extra.pt"))
    with pytest.raises(SystemExit, match="does not have"):
        RT.warm_start_from(m8, str(tmp_path / "extra.pt"))
    k_plain = next(k for k in sd if not RT.is_refcv8_key(k) and k.startswith("core.decoder."))
    torch.save({"model": {k: v for k, v in sd.items() if k != k_plain}}, str(tmp_path / "missing.pt"))
    with pytest.raises(SystemExit, match="NOT refcv8 seams"):
        RT.warm_start_from(m8, str(tmp_path / "missing.pt"))
    k_shape = next(k for k, v in sd.items() if torch.is_floating_point(v) and v.ndim == 2)
    torch.save({"model": dict(sd, **{k_shape: torch.zeros(3, 3)})}, str(tmp_path / "shape.pt"))
    with pytest.raises(SystemExit, match="init-from"):
        RT.warm_start_from(m8, str(tmp_path / "shape.pt"))
    with pytest.raises(SystemExit, match="no such file"):
        RT.warm_start_from(m8, str(tmp_path / "nope.pt"))


def test_the_seam_key_rule_on_literals():
    assert RT.is_refcv8_key("core.decoder.r8_mod.proj.0.weight")
    assert RT.is_refcv8_key("tac_decoder_v6.r8_cons_lat.weight")
    assert not RT.is_refcv8_key("core.decoder.layers.0.attn.in_proj_weight")
    assert not RT.is_refcv8_key("core.encoder.r8x.weight")                      # a prefix, not a component


def test_train_applies_it_only_without_a_resume_and_before_gdvb():
    T = R.trainer()
    src_txt = inspect.getsource(T.train).replace("\r\n", "\n")
    i = src_txt.index("_init_stamp = dict(r8train.warm_start_from(model, args.init_from), applied=True)")
    g = src_txt.index("_dvb.refuse_on_mismatch(model, args, build_parser(), where=\"train\")")
    resume = src_txt.index("if ck.exists():\n            _init_stamp")
    assert resume < i < g
    assert '"init_from": _init_stamp,' in src_txt
