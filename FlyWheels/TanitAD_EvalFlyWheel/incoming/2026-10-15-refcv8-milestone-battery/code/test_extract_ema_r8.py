"""CPU tests for extract_ema_r8.py.  NEW file.

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' python -m pytest -q -p no:cacheprovider test_extract_ema_r8.py

Known-value control: a tiny model is TRAINED with a real `WeightEMA` (tied parameter, frozen layer, float + int buffers); the pair `extract_core` writes from the saved `ckpt` must equal
`WeightEMA.model_state_dict` of the objects that trained, key by key, and a rebuilt model that went through the loader's `requires_grad_(False)` must still work.  The real-pair
assertion (`check_real_pair`) has synthetic deliberate-regression arms; running it on the real 30k pair is a separate recorded step (raw/snap40k/check_real_30k_pair.json).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")
pytest.importorskip("tanitad.train.weight_ema")
import extract_ema_r8 as X  # noqa: E402


def test_selftest_all_checks_hold_with_the_local_writer():
    r = X.selftest(None)
    assert r["ok"], [k for k, v in r["checks"].items() if not v]
    assert len(r["checks"]) >= 14


def test_selftest_with_the_trainers_own_save_ema_model_if_importable():
    try:
        save, src, _ = X._trainer_save_ema()
    except Exception as exc:                                                    # noqa: BLE001
        pytest.skip(f"trainer not importable here: {type(exc).__name__}")
    assert src.endswith("::_save_ema_model")
    r = X.selftest(save)
    assert r["ok"] and r["save_ema_source"] == "the trainer's own _save_ema_model"


def test_mutation_forgetting_the_requires_grad_restore_is_caught(monkeypatch):
    monkeypatch.setattr(X, "restore_trainable_from_shadow", lambda m, k: {})
    with pytest.raises(ValueError, match="EMA covers"):                          # EMAStateError is a ValueError
        X.selftest(None)


def test_restore_trainable_handles_a_tied_parameter_per_object():
    m = X._tiny()
    for p in m.parameters():
        p.requires_grad_(False)
    # only the SECOND alias of the tied parameter is named in the shadow: per-object logic must still make it trainable
    rec = X.restore_trainable_from_shadow(m, {"out_tied.weight", "out_tied.bias", "lin1.weight"})
    assert m.out.weight.requires_grad and m.out.bias.requires_grad and m.lin1.weight.requires_grad
    assert not m.lin1.bias.requires_grad and not m.frozen.weight.requires_grad
    assert rec["n_trainable_restored"] == 3


def _write_pair(tmp_path, mutate=None, ema_mutate=None):
    m = X._tiny()
    raw = {"model": m.state_dict(), "step": 7}
    em = {k: (v + 1 if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in m.state_dict().items()}
    em["out_tied.weight"], em["out_tied.bias"] = em["out.weight"], em["out.bias"]       # the EMA file ALIASES the shadow for a tied parameter, as model_state_dict does
    ema = {"model": em, "step": 7, "ema_decay": 0.9998, "num_updates": 7, "kind": "ema_model_only"}
    if mutate:
        mutate(raw)
    if ema_mutate:
        ema_mutate(ema)
    torch.save(raw, tmp_path / "raw.pt")
    torch.save(ema, tmp_path / "ema.pt")
    cfg = tmp_path / "config.json"
    cfg.write_text('{"ema": {"decay": 0.9998, "n_tensors": 9}}', encoding="utf-8")
    return tmp_path / "raw.pt", tmp_path / "ema.pt", cfg


def test_check_real_pair_synthetic_pass_and_each_regression_goes_red(tmp_path):
    r = X.check_real_pair(*_write_pair(tmp_path))
    assert r["ok"], r["checks"]

    def drop_key(e):
        del e["model"]["frozen.bias"]
    assert not X.check_real_pair(*_write_pair(tmp_path, ema_mutate=drop_key))["checks"]["key_set_equal"]

    def wrong_dtype(e):
        e["model"]["lin1.weight"] = e["model"]["lin1.weight"].double()
    assert not X.check_real_pair(*_write_pair(tmp_path, ema_mutate=wrong_dtype))["checks"]["shape_dtype_equal_on_every_tensor"]

    def wrong_decay(e):
        e["ema_decay"] = 0.999
    assert not X.check_real_pair(*_write_pair(tmp_path, ema_mutate=wrong_decay))["checks"]["decay_is_the_runs"]

    def wrong_meta(e):
        e["optimizer"] = {}
    assert not X.check_real_pair(*_write_pair(tmp_path, ema_mutate=wrong_meta))["checks"]["ema_meta_keys_are_the_trainers"]

    def int_buffer_moved(e):
        e["model"]["bn.num_batches_tracked"] = e["model"]["bn.num_batches_tracked"] + 5
    assert not X.check_real_pair(*_write_pair(tmp_path, ema_mutate=int_buffer_moved))["checks"]["only_float_tensors_differ"]
