"""weight_interp.py (E2-WISE): partial interpolation toward the init. CPU, tiny synthetic checkpoints in ``tmp_path``, no scipy.

* alpha = 1 reproduces the run bit for bit; alpha = 0 gives the init on S and the run off S; alpha = 0.5 is the exact midpoint.
* Prefixes match whole dotted components. A shape mismatch / absent / non-floating tensor in S is copied and listed.
* Every refusal fires. A bf16 tensor comes back as bf16. Prefixes covering only init-absent tensors leave the output equal to the run.
* The CLI writes a loadable file, a provenance sidecar whose ``perception_untouched`` is complete, and refuses to overwrite.
* MUTATION: the interpolation line replaced by a copy of the run tensor (and by the init tensor) is rebuilt in memory and the
  alpha = 0.5 check must go RED on it.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))
import weight_interp as W                                      # noqa: E402

S = "core.decoder,scorer,core.lat_log_prior"


def _t(vals, dtype=torch.float32):
    return torch.tensor(vals, dtype=dtype)


def _run_model():
    """Values are small integers so every midpoint is exactly representable (even in bf16)."""
    return {
        "core.decoder.w": _t([[2.0, 4.0], [6.0, 8.0]]),
        "core.decoder.b": _t([10.0, 12.0]),
        "core.decoderX.w": _t([100.0, 200.0]),                  # NOT under the prefix core.decoder
        "scorer.goal_point.w": _t([4.0, 8.0, 12.0]),
        "core.lat_log_prior": _t([2.0, 6.0]),                   # the prefix IS the parameter
        "core.encoder.w": _t([[1.0, 3.0], [5.0, 7.0]]),         # outside S
        "core.agent_head.w": _t([9.0, 11.0]),                   # outside S
        "scorer.new_only.w": _t([5.0, 5.0]),                    # in S, absent from the init
        "scorer.resized.w": _t([1.0, 2.0, 3.0]),                # in S, other shape in the init
        "scorer.counter": torch.tensor(7, dtype=torch.int64),   # in S, non-floating
        "core.decoder.bf": _t([4.0, 8.0], torch.bfloat16),
        "core.encoder.bf": _t([4.0, 8.0], torch.bfloat16),
        "core.note": "a non-tensor entry",
    }


def _init_model():
    return {
        "core.decoder.w": _t([[0.0, 0.0], [2.0, 0.0]]),
        "core.decoder.b": _t([2.0, 4.0]),
        "core.decoderX.w": _t([0.0, 0.0]),
        "scorer.goal_point.w": _t([0.0, 0.0, 0.0]),
        "core.lat_log_prior": _t([0.0, 2.0]),
        "core.encoder.w": _t([[9.0, 9.0], [9.0, 9.0]]),
        "core.agent_head.w": _t([0.0, 0.0]),
        "scorer.resized.w": _t([0.0, 0.0]),
        "scorer.counter": torch.tensor(0, dtype=torch.int64),
        "core.decoder.bf": _t([0.0, 0.0], torch.bfloat16),
        "core.encoder.bf": _t([0.0, 0.0], torch.bfloat16),
        "core.note": "other",
    }


def _ck(model):
    return {"model": model, "step": 6000, "ema_decay": 0.999, "num_updates": 6000, "kind": "ema_model_only"}


def _pfx(s=S):
    return s.split(",")


@pytest.fixture()
def cks():
    return _ck(_run_model()), _ck(_init_model())


def _same(a, b):
    return W._bit_equal(a, b)


def test_alpha_one_reproduces_the_run_bit_for_bit(cks):
    run, init = cks
    out, info = W.interp_state(run, init, 1.0, _pfx())
    assert list(out) == list(run["model"])
    for k, v in run["model"].items():
        assert _same(out[k], v) if torch.is_tensor(v) else out[k] == v, k
    assert info["n_interpolated"] == 5


def test_alpha_one_is_exact_for_awkward_random_floats():
    """The literal init + a*(run-init) is NOT exact at a = 1 for arbitrary fp32; the tool must be."""
    g = torch.Generator().manual_seed(3)
    run = _ck({"core.decoder.w": torch.randn(4096, generator=g) * 3.7})
    init = _ck({"core.decoder.w": torch.randn(4096, generator=g) * 0.013})
    literal = init["model"]["core.decoder.w"] + 1.0 * (run["model"]["core.decoder.w"] - init["model"]["core.decoder.w"])
    assert not _same(literal, run["model"]["core.decoder.w"])            # the trap is real on this data
    out, _ = W.interp_state(run, init, 1.0, ["core.decoder"])
    assert _same(out["core.decoder.w"], run["model"]["core.decoder.w"])
    out0, _ = W.interp_state(run, init, 0.0, ["core.decoder"])
    assert _same(out0["core.decoder.w"], init["model"]["core.decoder.w"])


def test_alpha_zero_is_init_on_S_and_run_off_S(cks):
    run, init = cks
    out, info = W.interp_state(run, init, 0.0, _pfx())
    for k in info["interpolated"]:
        assert _same(out[k], init["model"][k]), k
    for k in info["untouched"] + list(info["copied_in_S"]):
        assert _same(out[k], run["model"][k]), k
    assert _same(out["core.encoder.w"], run["model"]["core.encoder.w"]) and _same(out["core.decoderX.w"], run["model"]["core.decoderX.w"])


def _midpoint_gap(fn, cks_):
    run, init = cks_
    out = fn(run, init)
    gap = 0.0
    for k in ("core.decoder.w", "core.decoder.b", "scorer.goal_point.w", "core.lat_log_prior"):
        exp = (run["model"][k].double() + init["model"][k].double()) / 2
        gap = max(gap, float((out[k].double() - exp).abs().max()))
    return gap


def _real(run, init):
    return W.interp_state(run, init, 0.5, _pfx())[0]


def test_alpha_half_is_the_exact_midpoint(cks):
    assert _midpoint_gap(_real, cks) == 0.0
    out = _real(*cks)
    assert out["core.decoder.w"].tolist() == [[1.0, 2.0], [4.0, 4.0]]
    assert out["core.lat_log_prior"].tolist() == [1.0, 4.0]


def test_dotted_component_prefix_matching_is_exact(cks):
    assert W.match_prefix("core.decoder", "core.decoder") and W.match_prefix("core.decoder.w", "core.decoder")
    assert not W.match_prefix("core.decoderX.w", "core.decoder") and not W.match_prefix("core.dec", "core.decoder")
    assert not W.match_prefix("core.decoder", "core.decoder.w")
    run, init = cks
    out, info = W.interp_state(run, init, 0.5, ["core.decoder"])
    assert "core.decoderX.w" in info["untouched"] and "core.decoderX.w" not in info["interpolated"]
    assert _same(out["core.decoderX.w"], run["model"]["core.decoderX.w"])
    assert sorted(info["interpolated"]) == ["core.decoder.b", "core.decoder.bf", "core.decoder.w"]


def test_shape_mismatch_absent_and_nonfloating_are_copied_and_listed(cks):
    run, init = cks
    out, info = W.interp_state(run, init, 0.5, _pfx())
    assert set(info["copied_in_S"]) == {"scorer.new_only.w", "scorer.resized.w", "scorer.counter"}
    assert "absent" in info["copied_in_S"]["scorer.new_only.w"] and "shape" in info["copied_in_S"]["scorer.resized.w"]
    assert "non-floating" in info["copied_in_S"]["scorer.counter"]
    for k in info["copied_in_S"]:
        assert _same(out[k], run["model"][k]) and out[k].data_ptr() != run["model"][k].data_ptr()
    assert info["n_copied_in_S"] == 3 and info["n_interpolated"] == 5 and info["n_untouched"] == 4 and info["n_nontensor"] == 1


def test_bf16_is_cast_back_to_bf16(cks):
    out, _ = W.interp_state(*cks, 0.5, _pfx("core.decoder"))
    assert out["core.decoder.bf"].dtype == torch.bfloat16 and out["core.decoder.bf"].tolist() == [2.0, 4.0]
    assert out["core.decoder.w"].dtype == torch.float32
    assert out["core.encoder.bf"].dtype == torch.bfloat16


def test_prefixes_covering_only_init_absent_tensors_leave_the_run_unchanged(cks):
    run, init = cks
    run["model"] = {"core.decoder.w": run["model"]["core.decoder.w"], "scorer.new_only.w": run["model"]["scorer.new_only.w"]}
    out, info = W.interp_state(run, init, 0.5, ["scorer"])
    assert info["n_interpolated"] == 0 and info["n_copied_in_S"] == 1
    assert all(_same(out[k], run["model"][k]) for k in run["model"])


def test_non_model_entries_are_copied_unchanged(cks):
    run, init = cks
    out, _ = W.interp_checkpoint(run, init, 0.5, _pfx())
    assert list(out) == list(run) and out["kind"] == "ema_model_only" and out["step"] == 6000 and out["num_updates"] == 6000
    assert out["ema_decay"] == 0.999 and out["model"]["core.note"] == "a non-tensor entry"


def test_refusals(cks):
    run, init = cks
    for bad in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(W.InterpError, match="alpha"):
            W.interp_state(run, init, bad, _pfx())
    with pytest.raises(W.InterpError, match="matches no tensor"):
        W.interp_state(run, init, 0.5, ["core.decoder", "core.nope"])
    with pytest.raises(W.InterpError, match="matches no tensor"):
        W.interp_state(run, init, 0.5, ["core.dec"])                    # a substring, not a component
    with pytest.raises(W.InterpError, match="non-empty"):
        W.interp_state(run, init, 0.5, [])
    with pytest.raises(W.InterpError, match="share no tensor"):
        W.interp_state(run, _ck({"other.w": _t([1.0])}), 0.5, _pfx())
    with pytest.raises(W.InterpError, match="'model' dict"):
        W.interp_state(run, {"state": {}}, 0.5, _pfx())


def test_non_finite_result_refuses(cks):
    run, init = cks
    init["model"]["core.decoder.b"] = _t([float("inf"), 4.0])
    with pytest.raises(W.InterpError, match="non-finite"):
        W.interp_state(run, init, 0.5, _pfx())


def _save(tmp_path, cks_):
    pr, pi = tmp_path / "run.pt", tmp_path / "init.pt"
    torch.save(cks_[0], pr)
    torch.save(cks_[1], pi)
    return pr, pi


def test_cli_writes_loadable_file_provenance_and_refuses_overwrite(cks, tmp_path):
    pr, pi = _save(tmp_path, cks)
    po = tmp_path / "out.pt"
    argv = ["--run", str(pr), "--init", str(pi), "--alpha", "0.5", "--prefixes", S, "--out", str(po)]
    assert W.main(argv) == 0
    ck = torch.load(po, map_location="cpu", weights_only=False)
    assert list(ck) == list(cks[0]) and ck["kind"] == "ema_model_only"
    assert ck["model"]["core.decoder.w"].tolist() == [[1.0, 2.0], [4.0, 4.0]]
    prov = json.loads(Path(str(po) + ".provenance.json").read_text(encoding="utf-8"))
    assert prov["alpha"] == 0.5 and prov["prefixes"] == _pfx() and len(prov["sha256_out"]) == 64 and len(prov["md5_out"]) == 32
    assert len(prov["sha256_run"]) == 64 and len(prov["sha256_init"]) == 64 and prov["sha256_run"] != prov["sha256_init"]
    assert prov["n_interpolated"] == 5 and prov["n_copied_in_S"] == 3 and prov["n_untouched"] == 4
    assert set(prov["copied_in_S"]) == {"scorer.new_only.w", "scorer.resized.w", "scorer.counter"}
    pu = prov["perception_untouched"]
    assert set(pu) == {"core.decoderX.w", "core.encoder.w", "core.agent_head.w", "core.encoder.bf"} | set()
    assert prov["perception_untouched_all_bit_equal"] is True
    assert all(_same(ck["model"][k], cks[0]["model"][k]) for k in pu)
    assert W._hash(po, "sha256") == prov["sha256_out"]
    assert W.main(argv) == 2                                            # exists
    assert W.main(["--run", str(pr), "--init", str(pi), "--alpha", "0.5", "--prefixes", S, "--out", str(pr)]) == 2   # out == input
    assert W.main(["--run", str(pr), "--init", str(pi), "--alpha", "1.5", "--prefixes", S, "--out", str(tmp_path / "x.pt")]) == 2
    assert W.main(["--run", str(pr), "--init", str(pi), "--alpha", "0.5", "--prefixes", "core.nope", "--out", str(tmp_path / "y.pt")]) == 2
    assert not (tmp_path / "x.pt").exists() and not (tmp_path / "y.pt").exists() and not list(tmp_path.glob("*.tmp"))


def test_cli_alpha_one_file_has_identical_tensor_bytes(cks, tmp_path):
    pr, pi = _save(tmp_path, cks)
    po = tmp_path / "a1.pt"
    assert W.main(["--run", str(pr), "--init", str(pi), "--alpha", "1", "--prefixes", S, "--out", str(po)]) == 0
    a, b = (torch.load(p, map_location="cpu", weights_only=False)["model"] for p in (pr, po))
    assert all(_same(b[k], a[k]) for k in a if torch.is_tensor(a[k]))
    assert W._state_digest(a) == W._state_digest(b)


def _mutant(old: str, new: str):
    src = Path(W.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, f"the mutation target {old!r} must occur exactly once"
    mod = types.ModuleType("weight_interp_mutant")
    exec(compile(src.replace(old, new), "weight_interp_mutant", "exec"), mod.__dict__)
    return lambda run, init: mod.interp_state(run, init, 0.5, _pfx())[0]


def test_the_midpoint_gap_is_zero_on_the_real_source_and_positive_on_every_mutant(cks):
    assert _midpoint_gap(_real, cks) == 0.0
    line = "torch.lerp(i32, r32, a)               # THE INTERPOLATION: init + a * (run - init), fp32"
    for new in ("r32", "i32", "torch.lerp(i32, r32, 0.25)", "i32 + a * (r32 - i32) * 2"):
        assert _midpoint_gap(_mutant(line, new), cks) > 0.1, new
