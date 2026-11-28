"""The D1 decay branch's build-only checks keep the branch source real (MM 4a3d05, 2026-11-28).

MEASURED 2026-11-28 03:07Z on Thor: the first D1 explore gate (commit c0957d36, profile refcv9-full-d1) FAILED G-HYG / G-DVB /
G-EVAL in 2 s each, because `job_model`'s sentinel argv pointed `--wsd-branch-from` at `__gate_unread__/wsd-branch-from` and the
trainer's `lr_schedule.branch_preflight` refuses a missing branch checkpoint before anything is built. The fix declares the
branch source a build input for refcv9-full-d1 ONLY, at module level (no PROFILES dict changes, so no profile fingerprint moves).
"""
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_gate as LG  # noqa: E402

SRC = "/home/nvidia/r8ns/r9e3/runs/r9e3-K2DC02/ckpt_full_30000.pt"
D1_ARGV = ["--steps", "36000", "--wsd-branch-from", SRC, "--wsd-branch-decay", "6000",
           "--anchors", "/home/nvidia/data/anchors.npz", "--init-from", "/home/nvidia/data/ema.pt",
           "--out", "/home/nvidia/r8ns/r9e3/runs/r9e3-K2DC02-d1"]


def _ctx(profile: str, argv=D1_ARGV, pmap=()):
    return SimpleNamespace(argv=list(argv), path_map=[list(x) for x in pmap], prof=LG.PROFILES[profile])


def _val(argv, flag):
    return argv[argv.index(flag) + 1]


def test_the_d1_profile_declares_the_branch_source_as_a_build_input_and_only_it():
    d1 = LG.model_build_input_flags(LG.PROFILES["refcv9-full-d1"])
    assert "--wsd-branch-from" in d1
    assert d1[: len(LG.PROFILES["refcv9-full-d1"]["model_input_flags"])] == tuple(LG.PROFILES["refcv9-full-d1"]["model_input_flags"])
    for name in ("refcv9-full", "refcv9-e2", "refcv9-e1rig"):
        assert LG.model_build_input_flags(LG.PROFILES[name]) == tuple(LG.PROFILES[name]["model_input_flags"]), name
    assert LG.BUILD_INPUT_EXTRA == {"refcv9-full-d1": ("--wsd-branch-from",)}


def test_the_fix_is_module_level_no_profile_dict_carries_the_branch_flag():
    for name, prof in LG.PROFILES.items():
        assert "--wsd-branch-from" not in tuple(prof.get("model_input_flags", ())), name


def test_d1_sentinel_argv_keeps_the_branch_source_real_and_still_sentinels_every_other_unkept_path(tmp_path):
    ctx = _ctx("refcv9-full-d1")
    argv, rec = LG._sentinel_argv(ctx, LG.model_build_input_flags(ctx.prof), tmp_path)
    assert _val(argv, "--wsd-branch-from") == SRC
    assert rec["kept"]["--wsd-branch-from"] == SRC
    assert "--wsd-branch-from" not in rec["sentinel"]
    # the control: an unkept data path is still a non-existent sentinel (the principle the fix must not weaken)
    assert "--init-from" not in LG.PROFILES["refcv9-full-d1"]["model_input_flags"]
    assert "__gate_unread__" in _val(argv, "--init-from") and not Path(_val(argv, "--init-from")).exists()


def test_the_branch_source_follows_the_path_map(tmp_path):
    ctx = _ctx("refcv9-full-d1", pmap=[("/home/nvidia/r8ns/r9e3/runs/r9e3-K2DC02", "F:/refcv9_e3/gate/d1_src")])
    argv, _ = LG._sentinel_argv(ctx, LG.model_build_input_flags(ctx.prof), tmp_path)
    assert _val(argv, "--wsd-branch-from") == "F:/refcv9_e3/gate/d1_src/ckpt_full_30000.pt"


def test_REGRESSION_ARM_without_the_extra_the_branch_source_is_the_sentinel_again(tmp_path, monkeypatch):
    """The measured defect, re-installed: the same call must then hand the trainer the non-existent sentinel."""
    monkeypatch.setattr(LG, "BUILD_INPUT_EXTRA", {})
    ctx = _ctx("refcv9-full-d1")
    argv, rec = LG._sentinel_argv(ctx, LG.model_build_input_flags(ctx.prof), tmp_path)
    assert "__gate_unread__" in _val(argv, "--wsd-branch-from")
    assert "--wsd-branch-from" in rec["sentinel"]


def test_job_model_uses_the_helper_at_every_build_input_site():
    src = inspect.getsource(LG.job_model)
    # the sentinel argv + the branch-source mapping + G-EVAL + G-DVB inputs_read
    assert src.count("model_build_input_flags(ctx.prof)") == 4
    assert "branch_source_sentinel_patch(ctx, pat, T, model_build_input_flags(ctx.prof), scratch)" in src
    assert 'ctx.prof["model_input_flags"]' not in src


# --------------------------------------------------------------------------------------------- #
# the second D1 defect, MEASURED 2026-11-28 on Thor once the branch source was kept real: the      #
# trainer's branch preflight compares the SENTINEL argv with the source run's REAL argv            #
# --------------------------------------------------------------------------------------------- #
import argparse  # noqa: E402
import json  # noqa: E402

from tanitad.train import lr_schedule as LRS  # noqa: E402


def _parser():
    p = argparse.ArgumentParser()
    for f in ("--steps", "--wsd-branch-from", "--wsd-branch-decay", "--anchors", "--init-from", "--out", "--lr", "--save-at"):
        p.add_argument(f, default=None)
    return p


def _source_dir(tmp_path, lr="1e-4"):
    d = tmp_path / "src_run"
    d.mkdir()
    (d / "ckpt_full_30000.pt").write_bytes(b"\0" * 16)
    src_argv = ["--steps", "73280", "--save-at", "30000", "--anchors", "/home/nvidia/data/anchors.npz",
                "--init-from", "/home/nvidia/data/ema.pt", "--out", "/home/nvidia/r8ns/r9e3/runs/r9e3-K2DC02", "--lr", lr]
    (d / "config.json").write_text(json.dumps({"argv": src_argv}), encoding="utf-8")
    return d


def _branch_argv(src_dir):
    return ["--steps", "36000", "--wsd-branch-from", str(src_dir / "ckpt_full_30000.pt"), "--wsd-branch-decay", "6000",
            "--anchors", "/home/nvidia/data/anchors.npz", "--init-from", "/home/nvidia/data/ema.pt",
            "--out", "/home/nvidia/r8ns/r9e3/runs/r9e3-K2DC02-d1", "--lr", "1e-4"]


def _preflight_in_job(tmp_path, src_dir, patched=True):
    """What job_model does: the sentinel argv, the (optional) source mapping, then the trainer's own preflight."""
    ctx = _ctx("refcv9-full-d1", argv=_branch_argv(src_dir))
    keep = LG.model_build_input_flags(ctx.prof)
    argv, _ = LG._sentinel_argv(ctx, keep, tmp_path / "scratch")
    pat = LG._Patcher()
    T = SimpleNamespace(_lrs=LRS)
    try:
        rec = LG.branch_source_sentinel_patch(ctx, pat, T, keep, tmp_path / "scratch") if patched else None
        p = _parser()
        return LRS.branch_preflight(p.parse_args(argv), p), rec
    finally:
        pat.restore()


def test_REGRESSION_ARM_unpatched_the_preflight_refuses_on_every_sentinelled_path(tmp_path):
    src = _source_dir(tmp_path)
    with pytest.raises(SystemExit, match="--init-from"):
        _preflight_in_job(tmp_path, src, patched=False)


def test_patched_the_preflight_passes_and_says_what_it_mapped(tmp_path):
    src = _source_dir(tmp_path)
    out, rec = _preflight_in_job(tmp_path, src)
    assert out["ckpt"] == src / "ckpt_full_30000.pt"
    assert rec["patched"] is True and "--init-from" in rec["sentinel"]
    assert LRS.read_source_namespace.__module__ == LRS.__name__          # restored after the job


def test_CONTROL_a_differing_NON_path_key_is_still_refused_with_the_patch(tmp_path):
    src = _source_dir(tmp_path, lr="2e-4")
    with pytest.raises(SystemExit, match="--lr"):
        _preflight_in_job(tmp_path, src)


def test_the_real_read_still_refuses_a_source_without_config(tmp_path):
    src = _source_dir(tmp_path)
    (src / "config.json").unlink()
    with pytest.raises(SystemExit, match="cannot read the source run"):
        _preflight_in_job(tmp_path, src)


def test_no_branch_flag_patches_nothing(tmp_path):
    ctx = _ctx("refcv9-full", argv=["--steps", "10", "--out", str(tmp_path / "o")])
    pat = LG._Patcher()
    assert LG.branch_source_sentinel_patch(ctx, pat, SimpleNamespace(_lrs=LRS), (), tmp_path) is None
    assert pat._undo == []


def test_the_inputs_read_of_a_kept_branch_source_is_its_fingerprint(tmp_path):
    ck = tmp_path / "ckpt_full_30000.pt"
    ck.write_bytes(b"x" * 64)
    ctx = _ctx("refcv9-full-d1", argv=["--wsd-branch-from", str(ck), "--out", str(tmp_path / "o")])
    got = LG._inputs_read(ctx, LG.model_build_input_flags(ctx.prof))
    assert set(got) == {"--wsd-branch-from"}
    assert LG.fp_key(got["--wsd-branch-from"]) == LG.fp_key(LG.fingerprint(ck))


@pytest.mark.parametrize("name", ["refcv9-full", "refcv9-e2"])
def test_other_profiles_still_sentinel_a_passed_branch_flag(tmp_path, name):
    ctx = _ctx(name)
    argv, rec = LG._sentinel_argv(ctx, LG.model_build_input_flags(ctx.prof), tmp_path)
    assert "--wsd-branch-from" in rec["sentinel"]
