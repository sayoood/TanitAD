"""PREREG_V8F_ROLLHIST items N2-N4 (the bank generator's executor flags, the bank-header levers, the declaration flags) and the card's reachability table (section 4.2): R4, R5, and R1-R3 on CPU.

CPU, fp32, one thread (R2 of the card: every bit-identity test). Each test builds its bank with the DATA generator on the trainer's own synthetic world (the ``pack`` fixture of
``test_v8f_dagger_train.py``); a generator that breaks fails the test, not a fixture. WHAT A CPU FIXTURE CANNOT SHOW (left to the pod-side pilot, HANDOFF_B1.md): that the real checkpoint's
rollout reaches the same bounds on real data, and the real-bank coverage / K-B2 on the pod's checkpoint.

  R4  the bank holds readout C* under hb0.2     bank-level: the stored last-knot acceleration a0 = the executed one (gain 1); per call: heading step <= 0.2 x distance + 1e-12 and the a3 control;
                                                the control: ``--readout pinned --kappa-max none`` = today's generator, array for array, max abs difference 0.0
  R5  the argv declares what the bank is        declared != the loaded header -> REFUSED (both flags, both directions); non-default at --dagger-p 0 -> REFUSED; a header without the field reads pinned / none
  R1-R3 the swap of an RH bank reaches the loss  dagger_probe PASS on the RH bank with its p = 0 controls EXACTLY 0.0; S's own loss and O's loss reach read_s/head_s and read_o/head_o, p = 0 reads 0.0
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", REPO / "stack" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_dagger as DG  # noqa: E402
import v8f_dagger_bank as B  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
import v8f_t1_eval as T1  # noqa: E402
import v8f_t1_eval_hb as HB  # noqa: E402
import v8f_t1_readout as RO  # noqa: E402
from test_v8f_dagger_train import GATE_ARGV, _bank_sha, clone_bank, dvb, gate_parts, pack, probe_parts  # noqa: E402,F401  (pack is a fixture)

torch.set_num_threads(1)
PINNED_FN = T1.hermite_frames                                          # the pinned tool's own function, captured before any run installs anything
RH_FLAGS = ["--readout", "causal_g", "--acc-gain", "1", "--path-exec", "--kappa-max", "0.2"]
PINNED = {"readout": "pinned", "acc_gain": None, "path_exec": False, "kappa_max": "none"}
RH = {"readout": "causal_g", "acc_gain": 1.0, "path_exec": True, "kappa_max": "0.2"}
DECL_RH = ["--dagger-bank-readout", "causal_g1x", "--dagger-bank-kappa", "0.2"]
CONSTANTS_SHA256 = "8f94b3b5e35f4457be51d695842457a340d1228ab48c5862f4db1e5063f3ffaf"      # DG.constants_sha256() of the tip 7b26a8ac (N3 does not change constants())


def gen_argv(pk, out, *extra, replay="0.1"):
    w = pk.world
    return ["gen", "--ckpt", str(pk.ck), "--m2b-argv", str(pk.root / "argv.json"), "--cache-dir", str(w), "--tracks", str(w / "native_tracks.pt"), "--join-train", str(w / "join_train.npz"),
            "--eval-grid", str(w / "eval_grid_windows.json"), "--out-dir", str(out), "--batch", "8", "--replay-frac", replay, *extra]


def _arrays(out):
    return DG.load_shard(Path(out) / "shard_000of001.npz")


def _same_arrays(a, b):
    assert sorted(a) == sorted(b)
    return all((np.asarray(a[k]).shape == np.asarray(b[k]).shape) and bool(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True)) for k in a)


def _max_abs_diff(a, b):
    worst = 0.0
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.size and x.dtype.kind == "f":
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            worst = max(worst, float(np.nanmax(d)) if np.isfinite(d).any() else 0.0)
    return worst


@pytest.fixture(scope="module")
def banks(pack, tmp_path_factory):
    """pinned (defaults), pinned (named), pinned through a NULL executor (today's code path, nothing installed), RH; plus the per-call record of the RH run's installed frames function."""
    root = tmp_path_factory.mktemp("rollhist")
    k = SimpleNamespace(pk=pack)
    k.default, k.named, k.null, k.rh = root / "default", root / "named", root / "null", root / "rh"
    assert B.main(gen_argv(pack, k.default)) == 0
    assert B.main(gen_argv(pack, k.named, "--readout", "pinned", "--kappa-max", "none")) == 0
    real = B.executor_installed
    B.executor_installed = lambda tool, gen: contextlib.nullcontext(tool)                     # today's generator: nothing installed, the pinned tool as it is
    try:
        assert B.main(gen_argv(pack, k.null)) == 0
    finally:
        B.executor_installed = real
    calls = []

    @contextlib.contextmanager
    def recording(tool, gen):
        with real(tool, gen):
            inner = tool.hermite_frames

            def shim(k_prev, knots, m, psi0):
                out = inner(k_prev, knots, m, psi0)
                calls.append((np.array(k_prev, np.float64), np.array(knots, np.float64), int(m), float(psi0), tuple(np.array(x, np.float64) for x in out)))
                return out
            tool.hermite_frames = shim
            try:
                yield tool
            finally:
                tool.hermite_frames = inner
    B.executor_installed = recording
    try:
        assert B.main(gen_argv(pack, k.rh, *RH_FLAGS)) == 0
    finally:
        B.executor_installed = real
    k.calls = calls
    return k


# ------------------------------------------------------------------------------------------------ N2 / R4
def test_r4_control_pinned_none_is_todays_generator_array_for_array_max_abs_difference_zero(banks):
    a_def, _ = _arrays(banks.default)
    a_named, _ = _arrays(banks.named)
    a_null, _ = _arrays(banks.null)
    assert _same_arrays(a_def, a_null) and _same_arrays(a_named, a_null)
    assert _max_abs_diff(a_def, a_null) == 0.0 and _max_abs_diff(a_named, a_null) == 0.0
    assert DG.content_sha256(a_def) == DG.content_sha256(a_null) == DG.content_sha256(a_named)
    assert DG.sha256_file(banks.default / "bank.json") == DG.sha256_file(banks.named / "bank.json")


def test_r4_rh_bank_is_not_the_pinned_bank_the_non_vacuity_control(banks):
    a_rh, _ = _arrays(banks.rh)
    a_p, _ = _arrays(banks.default)
    assert not _same_arrays(a_rh, a_p)
    assert _max_abs_diff(a_rh, a_p) > 1e-6


def _a3_worst(arrays, gain):
    """the stored one-frame acceleration at the last executed knot (kin3[:, 1]) minus ``gain`` x the executed one, from the stored knot positions of rows j-2 .. j"""
    worst, n = 0.0, 0
    for i in range(len(arrays["ci"])):
        if int(arrays["reason"][i]) == DG.NONFINITE:
            continue
        P = np.asarray(arrays["pose"][i], np.float64)[-3:, :2]
        if not np.isfinite(P).all():
            continue
        s1, s2 = np.hypot(*(P[1] - P[0])) / 0.2, np.hypot(*(P[2] - P[1])) / 0.2
        worst, n = max(worst, abs(float(arrays["kin3"][i][1]) - gain * (s2 - s1) / 0.2)), n + 1
    return worst, n


def test_r4_the_rh_bank_holds_the_executed_acceleration_and_the_pinned_bank_does_not(banks):
    a_rh, _ = _arrays(banks.rh)
    worst, n = _a3_worst(a_rh, 1.0)
    assert n > 0 and worst <= 1e-6, (worst, n)
    a_p, _ = _arrays(banks.default)
    worst_p, n_p = _a3_worst(a_p, 1.0)
    assert n_p > 0 and worst_p > 1e-3, f"the control: the pinned bank carries the next-plan-knot acceleration (read {worst_p}); the check would pass vacuously otherwise"


def test_r4_every_call_of_the_installed_frames_obeys_the_bound_and_the_acceleration_control(banks):
    assert len(banks.calls) > 50
    worst_a3, bound_acted = 0.0, 0
    for k_prev, knots, m, psi0, (xy, psi, v) in banks.calls:
        prev, pxy = psi0, knots[0]
        for f in range(len(psi)):
            step = abs(HB.wrap_pi(float(psi[f]) - prev))
            lim = 0.2 * float(np.hypot(*(xy[f] - pxy)))
            assert step <= lim + 1e-12, (f, step, lim)
            prev, pxy = float(psi[f]), xy[f]
        raw = RO.backward_frames(PINNED_FN, k_prev, knots, m, psi0)
        bound_acted += int(not np.array_equal(psi, raw[1]))
        assert np.array_equal(xy, raw[0])                                                       # --path-exec: positions from the executed knots only
        K = np.vstack([k_prev[None], knots[:m + 1]])
        s = np.linalg.norm(np.diff(K, axis=0), axis=-1) / 0.2
        worst_a3 = max(worst_a3, abs((v[-1] - v[-2]) / 0.1 - (s[m] - s[m - 1]) / 0.2))
    assert worst_a3 <= 1e-6


def test_the_executor_is_installed_for_the_run_and_restored_after(pack, tmp_path):
    seen = []
    real = B.executor_installed

    @contextlib.contextmanager
    def spy(tool, gen):
        with real(tool, gen):
            seen.append((tool.hermite_frames is not PINNED_FN, getattr(tool.hermite_frames, "__wrapped__", None) is not None))
            yield tool
    B.executor_installed = spy
    try:
        assert B.main(gen_argv(pack, tmp_path / "b", *RH_FLAGS, replay="0")) == 0
        assert B.main(gen_argv(pack, tmp_path / "c", replay="0")) == 0
    finally:
        B.executor_installed = real
    assert T1.hermite_frames is PINNED_FN
    assert seen == [(True, True), (False, False)], "RH installs a wrapped function; the defaults install NOTHING (the same function object)"


def test_defaults_install_nothing_by_identity():
    gen = B.generator_record(SimpleNamespace())
    assert gen == PINNED
    with B.executor_installed(T1, gen):
        assert T1.hermite_frames is PINNED_FN
    assert RO.readout_frames(PINNED_FN, "pinned") is PINNED_FN and HB.bounded_frames(PINNED_FN, None) is PINNED_FN


@pytest.mark.parametrize("extra", [["--acc-gain", "1"], ["--path-exec"], ["--readout", "causal_g"]])
def test_the_generator_refuses_a_declared_and_inert_or_missing_parameter(pack, tmp_path, extra):
    with pytest.raises(SystemExit) as e:
        B.main(gen_argv(pack, tmp_path / "x", *extra, replay="0"))
    assert "REFUSED" in str(e.value)
    assert not (tmp_path / "x" / "bank.json").exists()


def test_tool_files_name_the_two_new_tools_and_provenance_carries_the_generator(banks):
    for f in ("stack/scripts/v8f_t1_readout.py", "stack/scripts/v8f_t1_eval_hb.py"):
        assert f in B.TOOL_FILES
    for out, want in ((banks.default, None), (banks.rh, RH)):                   # N2: a default bank carries NO generator record anywhere; a non-default one carries it everywhere
        sp = json.loads((out / "shard_000of001.provenance.json").read_text(encoding="utf-8"))
        bj, pj = json.loads((out / "bank.json").read_text(encoding="utf-8")), json.loads((out / "PROVENANCE.json").read_text(encoding="utf-8"))
        _, h = _arrays(out)
        if want is None:
            assert "generator" not in sp and "generator" not in sp["plan"] and "generator" not in bj["common"] and "generator" not in pj and "generator" not in h
        else:
            assert sp["generator"] == want == sp["plan"]["generator"] == bj["common"]["generator"] == pj["generator"] == h["generator"]
        assert "stack/scripts/v8f_t1_readout.py" in sp["tool_sha256"] and "stack/scripts/v8f_t1_eval_hb.py" in sp["tool_sha256"]


def _vol(o):
    """a provenance with the fields that differ between ANY two runs (clock, argv paths, threads, the tool files' own sha) removed"""
    o = copy.deepcopy(o)
    for k in ("created_unix", "argv", "tool_sha256", "versions", "threads", "seconds", "causal_bucket_stats"):          # wall clock; a process-global counter that accumulates across runs in one process
        o.pop(k, None)
    return o


def test_n2_a_default_bank_is_byte_identical_to_todays_generator(banks):
    """The control that matters: the defaults write today's files, byte for byte -- the shard npz, bank.json (whose sha IS the bank's sha256), PROVENANCE.json -- against a run through a NULL executor
    (nothing installed, no record: the unpatched code path)."""
    for name in ("shard_000of001.npz", "bank.json"):
        assert (banks.default / name).read_bytes() == (banks.null / name).read_bytes(), name
    assert sorted(p.name for p in banks.default.iterdir()) == sorted(p.name for p in banks.null.iterdir())
    for name in ("PROVENANCE.json", "shard_000of001.provenance.json"):
        a, b = (json.loads((d / name).read_text(encoding="utf-8")) for d in (banks.default, banks.null))
        assert _vol(a) == _vol(b), name
    assert banks.default.joinpath("bank.json").read_bytes() == banks.named.joinpath("bank.json").read_bytes(), "an explicit --readout pinned --kappa-max none is the same bank"


@pytest.mark.skipif(not os.environ.get("ROLLHIST_ORIG_STACK"), reason="needs ROLLHIST_ORIG_STACK = the UNPATCHED tip export's stack/ directory")
def test_n2_a_default_bank_is_byte_identical_to_the_unpatched_generators_files(banks, tmp_path):
    out = tmp_path / "unpatched"
    r = subprocess.run([sys.executable, str(Path(os.environ["ROLLHIST_ORIG_STACK"]) / "scripts" / "v8f_dagger_bank.py"), *gen_argv(banks.pk, out)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONPATH": os.environ["ROLLHIST_ORIG_STACK"], "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    for name in ("shard_000of001.npz", "bank.json"):
        assert (banks.default / name).read_bytes() == (out / name).read_bytes(), name
    assert sorted(p.name for p in banks.default.iterdir()) == sorted(p.name for p in out.iterdir())
    for name in ("PROVENANCE.json", "shard_000of001.provenance.json"):
        a, b = (json.loads((d / name).read_text(encoding="utf-8")) for d in (banks.default, out))
        assert _vol(a) == _vol(b), name


# ------------------------------------------------------------------------------------------------ N3
def _strip_generator(src, dst):
    def sh(arrays, header):
        header = copy.deepcopy(header)
        header.pop("generator", None)
        return arrays, header

    def cm(common):
        common = copy.deepcopy(common)
        common.pop("generator", None)
        return common
    return clone_bank(src, dst, shard_fn=sh, common_fn=cm)


def test_n3_built_levers_read_the_loaded_header_and_default_to_pinned_none(banks, tmp_path):
    J = banks.pk.J
    rh = DG.DaggerSwap(DG.DaggerBank.load(banks.rh, clip_names=J["clip"]), p=0.1)
    pn = DG.DaggerSwap(DG.DaggerBank.load(banks.default, clip_names=J["clip"]), p=0.1)
    old = DG.DaggerSwap(DG.DaggerBank.load(_strip_generator(banks.default, tmp_path / "old"), clip_names=J["clip"]), p=0.1)
    for dag, want in ((rh, ("causal_g1x", "0.2")), (pn, ("pinned", "none")), (old, ("pinned", "none")), (DG.DaggerSwap(), ("pinned", "none"))):
        lv = dag.built_levers()
        assert (lv["dagger_bank_readout"], lv["dagger_bank_kappa"]) == want
    assert not any(k in DG.constants() for k in ("generator", "readout", "kappa"))


def test_n3_constants_are_unchanged():
    assert DG.constants_sha256() == CONSTANTS_SHA256


# ------------------------------------------------------------------------------------------------ N4 / R5
def _bad(pk, bank, extra, p="0.1"):
    parts = gate_parts(pk, ["--init-ckpt", str(pk.ck), "--dagger-p", p, "--dagger-bank", str(bank), "--dagger-bank-sha256", _bank_sha(bank), *extra])
    return dvb(parts)


def test_n4_flag_checks_name_both_dests_and_the_defaults_are_the_legacy_bank():
    assert TR.FLAG_CHECKS["dagger_bank_readout"] == "dagger:dagger_bank_readout" and TR.FLAG_CHECKS["dagger_bank_kappa"] == "dagger:dagger_bank_kappa"
    a = TR.build_parser().parse_args(["--out", "x", *GATE_ARGV])
    assert (a.dagger_bank_readout, a.dagger_bank_kappa) == ("pinned", "none")


def test_r5_declared_equals_the_header_passes_and_each_mismatch_is_refused(banks):
    pk = banks.pk
    assert _bad(pk, banks.rh, DECL_RH) == [], "the control: RH declared, RH bank"
    assert _bad(pk, banks.default, []) == [], "the control: pinned declared (the default), pinned bank"
    for extra, key in (([], "dagger_bank_readout"), (["--dagger-bank-kappa", "0.2"], "dagger_bank_readout"), (["--dagger-bank-readout", "causal_g1x"], "dagger_bank_kappa")):
        bad = _bad(pk, banks.rh, extra)
        assert any(key in m and "loaded bank's header" in m for m in bad), (extra, bad)
    for extra, key in ((DECL_RH, "dagger_bank_readout"), (["--dagger-bank-kappa", "0.2"], "dagger_bank_kappa"), (["--dagger-bank-readout", "causal_g1x"], "dagger_bank_readout")):
        bad = _bad(pk, banks.default, extra)
        assert any(key in m and "loaded bank's header" in m for m in bad), (extra, bad)


def test_r5_a_header_without_the_field_reads_pinned_none(banks, tmp_path):
    old = _strip_generator(banks.default, tmp_path / "old")
    assert _bad(banks.pk, old, []) == []
    assert any("dagger_bank_readout" in m for m in _bad(banks.pk, old, ["--dagger-bank-readout", "causal_g1x"]))


def _set_generator(src, dst, gen):
    def sh(arrays, header):
        header = copy.deepcopy(header)
        header["generator"] = gen
        return arrays, header

    def cm(common):
        common = copy.deepcopy(common)
        common["generator"] = gen
        return common
    return clone_bank(src, dst, shard_fn=sh, common_fn=cm)


def test_the_readout_token_is_one_spelling():
    assert DG.readout_token("pinned") == "pinned"
    assert DG.readout_token("causal_g", 1.0, True) == "causal_g1x" and DG.readout_token("causal_g", 1, False) == "causal_g1"
    assert DG.readout_token("causal_g", 0.5, False) == "causal_g0.5" and DG.readout_token("causal_g", 0.5, True) == "causal_g0.5x"
    for bad in (("pinned", 1.0, False), ("pinned", None, True), ("causal_g", None, False), ("backward", 1.0, False)):
        with pytest.raises(ValueError):
            DG.readout_token(*bad)
    for tok in ("pinned", "causal_g1x", "causal_g1", "causal_g0.5", "causal_g0.5x"):
        assert DG.parse_readout_token(tok) == tok
    for tok in ("causal_g", "causal_g1.0x", "causal_g01x", "causal_gx", "causal_g1xx", "backward", ""):
        with pytest.raises(ValueError):
            DG.parse_readout_token(tok)


def test_r5_the_declaration_names_the_whole_readout_gain_and_path_exec_included(banks, tmp_path):
    pk = banks.pk
    nopath = _set_generator(banks.rh, tmp_path / "g1_nopath", {"readout": "causal_g", "acc_gain": 1.0, "path_exec": False, "kappa_max": "0.2"})
    half = _set_generator(banks.rh, tmp_path / "g05_path", {"readout": "causal_g", "acc_gain": 0.5, "path_exec": True, "kappa_max": "0.2"})
    for bank, ok, wrong in ((nopath, "causal_g1", "causal_g1x"), (half, "causal_g0.5x", "causal_g1x")):
        assert DG.DaggerSwap(DG.DaggerBank.load(bank, clip_names=pk.J["clip"]), p=0.1).built_levers()["dagger_bank_readout"] == ok
        assert _bad(pk, bank, ["--dagger-bank-readout", ok, "--dagger-bank-kappa", "0.2"]) == [], "the control: the matching token is accepted"
        bad = _bad(pk, bank, ["--dagger-bank-readout", wrong, "--dagger-bank-kappa", "0.2"])
        assert any("dagger_bank_readout" in m and ok in m for m in bad), (wrong, bad)
    for tok in ("causal_g", "causal_g1.0x"):                                                         # the bare base name and a non-canonical spelling are no tokens
        with pytest.raises(SystemExit):
            TR.build_parser().parse_args(["--out", "x", *GATE_ARGV, "--dagger-bank-readout", tok])


@pytest.mark.parametrize("extra", [["--dagger-bank-readout", "causal_g1x"], ["--dagger-bank-kappa", "0.2"], DECL_RH])
def test_r5_a_declaration_at_dagger_p_zero_is_refused(banks, extra):
    bad = dvb(gate_parts(banks.pk, ["--dagger-p", "0", *extra]))
    assert any("does nothing" in m for m in bad), bad
    assert dvb(gate_parts(banks.pk, ["--dagger-p", "0"])) == [], "the control: p = 0 with the defaults"


def test_r5_choices_are_closed():
    for flag, val in (("--dagger-bank-readout", "backward"), ("--dagger-bank-kappa", "0.3")):
        with pytest.raises(SystemExit):
            TR.build_parser().parse_args(["--out", "x", *GATE_ARGV, flag, val])


# ------------------------------------------------------------------------------------------------ R1-R3 on CPU, tiny model, the RH bank
def _rh_pack(banks):
    return SimpleNamespace(bank=banks.rh, world=banks.pk.world, J=banks.pk.J)


def test_r1_dagger_probe_passes_on_the_rh_bank_and_the_p0_controls_read_exactly_zero(banks):
    model, batch, cfg, dag = probe_parts(_rh_pack(banks), p=0.5, n=24)
    assert dag.built_levers()["dagger_bank_readout"] == "causal_g1x"
    res = TR.dagger_probe(model, batch, cfg, dag)
    assert res["pass"] is True
    assert res["control_p0_loss_diff"] == 0.0 and res["control_p0_grad_diff"] == 0.0 and res["control_p0_same_object"] is True
    assert res["control_prior_grad_diff"] == 0.0 and res["control_wm_grad_diff"] == 0.0
    assert res["loss_diff"] != 0.0 and res["grad_diff_all"] > 0.0


def _grad_sum_diff(model, b_a, b_b, term, prefixes):
    def grads(b):
        for p in model.parameters():
            p.grad = None
        torch.manual_seed(7)
        T = model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]
        T[term].backward()
        return {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in model.named_parameters() if n.startswith(prefixes)}
    ga, gb = grads(b_a), grads(b_b)
    return float(sum((ga[n] - gb[n]).abs().sum() for n in ga)), float(sum(g.abs().sum() for g in ga.values()))


def test_r2_r3_s_and_o_train_on_the_swapped_rolled_history_and_the_p0_controls_read_exactly_zero(banks):
    model, batch, cfg, dag = probe_parts(_rh_pack(banks), p=1.0, n=24)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name in ("head_s",):
            for p in getattr(model, name).parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    pert = dag.apply(batch, dag.PROBE_STEP)
    zero = dag.apply(batch, dag.PROBE_STEP, p=0.0)
    assert zero is batch
    d_s, n_s = _grad_sum_diff(model, pert, batch, "ckpt", ("read_s", "head_s"))
    d_o, n_o = _grad_sum_diff(model, pert, batch, "ade", ("read_o", "head_o"))
    assert n_s > 0 and n_o > 0 and d_s > 0.0 and d_o > 0.0, (d_s, n_s, d_o, n_o)
    z_s, _ = _grad_sum_diff(model, zero, batch, "ckpt", ("read_s", "head_s"))
    z_o, _ = _grad_sum_diff(model, zero, batch, "ade", ("read_o", "head_o"))
    assert z_s == 0.0 and z_o == 0.0
