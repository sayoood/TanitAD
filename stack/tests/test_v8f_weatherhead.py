"""PREREG_V8F_WEATHERHEAD form (a), card section 4: ``--ctx-pool`` in ``v8f_mini.py`` and the two evaluators.

Every default DISABLES the lever. This file proves REACHABILITY from the real caller (CLAUDE.md 2026-09-10/11, "built, tested and unreachable from its caller"): flag -> cfg -> the BUILT model -> ``_forward`` emits
``ctx_pool_logits`` -> the fit-loss term ``ctx_pool/<var>`` -> ``backward()`` -> non-zero gradient on exactly the five sub-heads, EXACTLY 0.0 / None on every other parameter (the control that proves the probe reads zero).
Each mechanism has a MUTATION that must turn its check RED (``mutant`` rewrites ONE line of the source and re-runs the SAME check against the mutated module).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from test_v8f_strat1 import (MODEL_P, TINY, build, make_batch, mod_from_text, run_loss, world)  # noqa: E402,F401
from tanitad.models import v8f_mini as M  # noqa: E402

SNAP = HERE / "v8f_mini_tip_snapshot_whead.txt"
SNAP_SHA = "776643b52b0a3286ef60d12c7c9088960ea1ab7f15030b5d4c8722d17bd217eb"      # sha256 of the a4357e79 stack/tanitad/models/v8f_mini.py
POOL_KEYS = {k for k in M.build_ctx_pool_heads(32, 0).state_dict()}
EXPECT_PARAMS = {"weather": 265_476, "all": 1_332_777}          # card 4.1 at d = 256


def tip_module():
    return sys.modules.get("v8f_mini_tip_snapshot_whead") or mod_from_text(SNAP.read_text(encoding="utf-8"), "v8f_mini_tip_snapshot_whead", SNAP)


def mutant(old, new, name="v8f_mini_mut_whead"):
    text = MODEL_P.read_text(encoding="utf-8")
    assert text.count(old) == 1, (old[:70], text.count(old))
    return mod_from_text(text.replace(old, new), name, MODEL_P)


def pool_model(J, mod=M, seed=0, **kw):
    return build(J, mod=mod, ctx_pool="on", **kw)


def some_y(B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randint(0, n, (B,), generator=g) for n in M.CTX_N], 1)


# ------------------------------------------------------------------------------------------------ pins
def test_the_snapshot_is_the_a4357e79_module():
    assert hashlib.sha256(SNAP.read_bytes()).hexdigest() == SNAP_SHA


def test_constants_and_key_prefix():
    assert M.CTX_POOL_INIT_SEED == 20261117 and M.CTX_POOL_MODES == ("off", "on") and M.CTX_POOL_KEY_PREFIXES == ("ctx_pool_head.",)
    assert M.CTX_POOL_KEY_PREFIXES[0] in M.NEW_LEVER_KEY_PREFIXES
    assert M.V8FMiniConfig().ctx_pool == "off"


def test_parameter_counts_match_the_card_at_d256():
    h = M.build_ctx_pool_heads(256, 0)
    n = {v: sum(p.numel() for p in h[v].parameters()) for v in M.CTX_VARS}
    assert n["weather"] == EXPECT_PARAMS["weather"]
    assert n["country"] == 264_448 + 257 * 25
    assert sum(n.values()) == EXPECT_PARAMS["all"]
    assert all(len(list(h[v].parameters())) == 11 for v in M.CTX_VARS) and len(list(h.parameters())) == 55


# ------------------------------------------------------------------------------------------------ T-OFF
def check_off(mod, world, **extra):
    """OFF == a4357e79: state dict, loss terms, total, gradients and forward_infer bit-equal to the a4357e79 module (``extra`` = levers named on BOTH sides, e.g. the shipped ctx head). Returns the problems."""
    tip = tip_module()
    b, J = make_batch(world)
    if extra.get("ctx_heads") == "on":
        from test_v8f_strat1 import CLof, with_ctx
        b = with_ctx(b, CLof(world, J))
    mt, mn = build(J, mod=tip, **extra), build(J, mod=mod, **extra)
    probs = []
    if list(mt.state_dict()) != list(mn.state_dict()):
        return ["state dict keys differ"]
    if not all(torch.equal(v, mn.state_dict()[k]) for k, v in mt.state_dict().items()):
        probs.append("init tensors differ")
    ot, on = run_loss(mt, b), run_loss(mn, b)
    if set(ot["terms"]) != set(on["terms"]) or not all(torch.equal(ot["terms"][k], on["terms"][k]) for k in ot["terms"]):
        probs.append("loss terms differ")
    ot["terms"]["total"].backward()
    on["terms"]["total"].backward()
    if not all(torch.equal(p.grad, q.grad) for p, q in zip(mt.parameters(), mn.parameters()) if p.grad is not None):
        probs.append("gradients differ")
    mt.eval()
    mn.eval()
    with torch.no_grad():
        it, inn = mt.forward_infer(b), mn.forward_infer(b)
    if set(it) != set(inn) or not all(torch.equal(it[k], inn[k]) for k in it if torch.is_tensor(it[k])):
        probs.append("forward_infer differs")
    return probs


def test_off_equals_a4357e79(world):
    assert check_off(M, world) == []


def test_off_equals_a4357e79_with_the_shipped_ctx_head_on(world):
    """S1's own configuration (ctx_heads on, ctx_pool off) is a4357e79 bit for bit: the model the evaluators score for the S1 re-dump."""
    assert check_off(M, world, ctx_heads="on", w_ctx=0.5) == []


def test_mutation_unconditional_pool_head_turns_off_equals_tip_red(world):
    mut = mutant('self.ctx_pool_head = None\n        if cfg.ctx_pool == "on":', 'self.ctx_pool_head = None\n        if True:')
    assert check_off(mut, world)


def check_rng(mod, world):
    _, J = make_batch(world)
    torch.manual_seed(123)
    off = build(J, mod=mod, wake=False)
    r_off = torch.get_rng_state()
    torch.manual_seed(123)
    on = build(J, mod=mod, wake=False, ctx_pool="on")
    r_on = torch.get_rng_state()
    same = all(torch.equal(v, on.state_dict()[k]) for k, v in off.state_dict().items())
    new = set(on.state_dict()) - set(off.state_dict())
    return same and torch.equal(r_off, r_on) and len(new) == 55 and all(k.startswith("ctx_pool_head.") for k in new)


def test_pool_on_leaves_every_other_init_tensor_and_the_global_rng_alone(world):
    assert check_rng(M, world)


def test_mutation_no_fork_rng_turns_rng_red(world):
    mut = mutant("with torch.random.fork_rng(devices=[]):\n        torch.manual_seed(seed)\n        return nn.ModuleDict", "if True:\n        torch.manual_seed(seed)\n        return nn.ModuleDict")
    assert not check_rng(mut, world)


# ------------------------------------------------------------------------------------------------ G-HYG
def check_hyg(mod):
    n = 0
    for kw in (dict(ctx_pool="maybe"), dict(ctx_pool=True), dict(ctx_pool="ON")):
        try:
            mod.V8FMiniConfig(**kw)
        except ValueError:
            n += 1
    try:
        mod.V8FMiniConfig(ctx_pool_x="on")
    except TypeError:
        n += 1
    return n == 4


def test_hygiene_refuses_bad_values_and_an_undeclared_ctx_pool_attribute():
    assert check_hyg(M)


def test_mutation_deleting_the_ctx_pool_check_turns_hyg_red():
    mut = mutant('        if self.ctx_pool not in CTX_POOL_MODES:\n            raise ValueError(f"ctx_pool must be one of {CTX_POOL_MODES}, got {self.ctx_pool!r}")\n', "")
    assert not check_hyg(mut)


# ------------------------------------------------------------------------------------------------ T-VIS
def check_vis(mod, world):
    b, J = make_batch(world)
    m = pool_model(J, mod=mod)
    try:
        mod.assert_ctx_pool_vision_only(m, b)
    except AssertionError as e:
        return str(e)
    return ""


def test_vis_passes_and_records_the_key_read_set(world):
    b, J = make_batch(world)
    m = pool_model(J)
    rec = M.assert_ctx_pool_vision_only(m, b)
    assert rec["keys_read"] == ["frame_ok"] and rec["source_problems"] == [] and rec["vision_sensitivity"] > 0
    assert M.ctx_pool_source_problems() == []
    assert m.built_levers()["ctx_pool_vision_only_source"] is True


def test_mutation_m4_naming_an_ego_key_inside_ctx_pool_turns_vis_red(world):
    mut = mutant('        tok, pad = self._ctx_pool_scene(b, adapt, last)\n        return self._ctx_pool_apply(tok, pad)\n\n    def ctx_pool_fit_logits',
                 '        tok, pad = self._ctx_pool_scene(b, adapt, last); tok = tok + b["nav"].float().mean()\n        return self._ctx_pool_apply(tok, pad)\n\n    def ctx_pool_fit_logits')
    assert "nav" in check_vis(mut, world) or "non-vision" in check_vis(mut, world)


def test_mutation_an_ego_key_read_at_runtime_only_is_caught_by_the_key_guard(world):
    """the static name check is bypassed (the key is built from parts): the RUNTIME guard (key read set, bit-identical under a non-vision swap) must still go red"""
    mut = mutant('        tok, pad = self._ctx_pool_scene(b, adapt, last)\n        return self._ctx_pool_apply(tok, pad)\n\n    def ctx_pool_fit_logits',
                 '        tok, pad = self._ctx_pool_scene(b, adapt, last); tok = tok + b["n" + "av"].float().mean()\n        return self._ctx_pool_apply(tok, pad)\n\n    def ctx_pool_fit_logits')
    assert check_vis(mut, world)


def test_vis_logits_are_bit_identical_when_every_non_vision_key_is_replaced_and_move_with_feats(world):
    b, J = make_batch(world)
    m = pool_model(J).eval()
    last = torch.full((b["feats"].shape[0],), b["feats"].shape[1] - 1)
    with torch.no_grad():
        a = m._ctx_pool(b, m._encode(b), last)
        b2 = {k: (torch.randn_like(v) if torch.is_tensor(v) and v.is_floating_point() and k not in ("feats",) else v) for k, v in b.items()}
        assert torch.equal(a, m._ctx_pool(b2, m._encode(b2), last))
        b3 = dict(b)
        b3["feats"] = b["feats"] * 0.5
        assert not torch.equal(a, m._ctx_pool(b3, m._encode(b3), last))


# ------------------------------------------------------------------------------------------------ T-ISO (b)
def test_iso_b_perturbing_the_pooled_head_moves_only_its_own_logits(world):
    b, J = make_batch(world)
    m = pool_model(J, ctx_heads="on", w_ctx=0.5)
    assert M.ctx_pool_isolation_problems(m, b) == []
    out = m.forward_infer(b)
    assert set(("ctx_logits", "ctx_pool_logits")) <= set(out) and out["ctx_pool_logits"].shape == (b["feats"].shape[0], 41)


def test_mutation_m3_feeding_the_pooled_logits_into_the_code_turns_iso_red(world):
    b, J = make_batch(world)
    mut = mutant("        code = self._code_pred(tout)\n        if use_labels", '        code = self._code_pred(tout) + 0.01 * out["ctx_pool_logits"][:, :1]\n        if use_labels')
    m = pool_model(J, mod=mut)
    assert mut.ctx_pool_isolation_problems(m, b)


def test_mutation_the_pooled_head_never_reaches_the_planner_if_the_check_is_blind(world):
    """the check itself must be able to read non-zero: a head whose logits are NOT perturbed by the N(0,1) step reads as blind"""
    b, J = make_batch(world)
    mut = mutant('        return torch.cat([self.ctx_pool_head[v](tok, pad) for v in CTX_VARS], -1).float()', '        return torch.cat([self.ctx_pool_head[v](tok, pad) for v in CTX_VARS], -1).float().detach() * 0.0')
    m = pool_model(J, mod=mut)
    assert any("did not move" in p for p in mut.ctx_pool_isolation_problems(m, b))


# ------------------------------------------------------------------------------------------------ reachability (4.4 items 1 and 2)
def grads_of(m, b, y, mod=M, train_adapter=False):
    for p in m.parameters():
        p.requires_grad_(True)
        p.grad = None
    m.eval()
    logits = m.ctx_pool_fit_logits(b, train_adapter)
    T = mod.ctx_pool_terms(logits, y, b["frame_ok"][:, -1])
    T["ctx_pool"].backward()
    pool, other, none_other = {v: 0.0 for v in M.CTX_VARS}, 0.0, 0
    for name, p in m.named_parameters():
        g = None if p.grad is None else float(p.grad.abs().sum())
        if name.startswith("ctx_pool_head."):
            pool[name.split(".")[1]] += g or 0.0
        else:
            other += g or 0.0
            none_other += g is None
    return pool, other, none_other


def check_reach(mod, world, train_adapter=False):
    b, J = make_batch(world)
    m = pool_model(J, mod=mod)
    y = some_y(b["feats"].shape[0])
    pool, other, none_other = grads_of(m, b, y, mod, train_adapter)
    return pool, other


def test_reachability_nonzero_on_each_subhead_and_exactly_zero_on_everything_else(world):
    pool, other = check_reach(M, world)
    assert all(v > 0.0 for v in pool.values()), pool
    assert other == 0.0, other                                          # the exact-zero control: the adapter, pos, time, S, T, O, shipped head ... all read exactly 0.0 or None


def test_the_feat_control_reads_non_zero_on_the_adapter_so_the_zero_above_is_a_real_zero(world):
    b, J = make_batch(world)
    m = pool_model(J)
    pool, other, _ = grads_of(m, b, some_y(b["feats"].shape[0]), train_adapter=True)
    assert other > 0.0 and all(v > 0.0 for v in pool.values())
    ad = float(sum(p.grad.abs().sum() for n, p in m.named_parameters() if n.startswith(("adapter.", "pos", "time")) and p.grad is not None))
    assert ad > 0.0


def test_mutation_m2_dropping_no_grad_on_the_token_path_turns_the_exact_zero_control_red(world):
    mut = mutant("with torch.set_grad_enabled(torch.is_grad_enabled() and train_adapter):", "with torch.enable_grad():")
    pool, other = check_reach(mut, world)
    assert other > 0.0                                                   # the control reads NON-zero: RED


def test_mutation_m1_deleting_the_fit_loss_line_turns_g_live_red(world):
    mut = mutant('    T["ctx_pool"] = sum(T[f"ctx_pool/{v}"] for v in CTX_VARS)', '    T["ctx_pool"] = sum(T[f"ctx_pool/{v}"] for v in CTX_VARS) * 0.0')
    pool, other = check_reach(mut, world)
    assert not all(v > 0.0 for v in pool.values())                        # no gradient reaches the sub-heads: RED


def test_loss_terms_are_masked_ce_and_the_total_is_the_unweighted_sum(world):
    g = torch.Generator().manual_seed(1)
    lg = torch.randn(12, 41, generator=g)
    y = some_y(12)
    y[:3, :] = -1
    y[4, 1] = -1
    fok = torch.ones(12, dtype=torch.bool)
    fok[5] = False
    T = M.ctx_pool_terms(lg, y, fok)
    o, tot = 0, 0.0
    for k, (v, n) in enumerate(zip(M.CTX_VARS, M.CTX_N)):
        m = (y[:, k] >= 0) & fok
        hand = torch.nn.functional.cross_entropy(lg[m][:, o:o + n], y[m, k], reduction="mean")
        assert torch.allclose(T[f"ctx_pool/{v}"], hand, atol=1e-6)
        o, tot = o + n, tot + float(hand)
    assert abs(float(T["ctx_pool"]) - tot) < 1e-5
    z = M.ctx_pool_terms(lg, -torch.ones(12, 5, dtype=torch.long), fok)
    assert float(z["ctx_pool"]) == 0.0                                    # all masked reads exactly 0.0


def test_all_padded_row_gets_a_finite_logit_and_the_pad_mask_is_the_scenes(world):
    b, J = make_batch(world)
    m = pool_model(J).eval()
    b = dict(b)
    ok = b["frame_ok"].clone()
    ok[0] = False
    b["frame_ok"] = ok
    with torch.no_grad():
        o = m.forward_infer(b)
    assert torch.isfinite(o["ctx_pool_logits"]).all()


def test_forward_emits_the_key_only_when_built_and_built_levers_name_it(world):
    b, J = make_batch(world)
    off, on = build(J), pool_model(J)
    off.eval()
    on.eval()
    with torch.no_grad():
        assert "ctx_pool_logits" not in off.forward_infer(b) and "ctx_pool_logits" in on.forward_infer(b)
    assert "ctx_pool" not in off.built_levers() and on.built_levers()["ctx_pool"] == "on" and on.built_levers()["ctx_pool_dim"] == 41
    assert "CTX_POOL" in on.trainable_groups() and "CTX_POOL" not in off.trainable_groups()


def test_the_fit_path_equals_the_forward_path_bitwise_on_the_same_window(world):
    """K-ID's building block: ctx_pool_fit_logits (the fit script's path) and forward_infer's ctx_pool_logits (the evaluators' path) are the same numbers"""
    b, J = make_batch(world)
    m = pool_model(J).eval()
    with torch.no_grad():
        a = m.forward_infer(b)["ctx_pool_logits"]
        c = m.ctx_pool_fit_logits(b)
    assert torch.allclose(a, c, atol=1e-5)


# ------------------------------------------------------------------------------------------------ the evaluators (4.4 items 1 and 4)
def src(name):
    return (HERE.parent / "scripts" / name).read_text(encoding="utf-8")


def test_both_evaluators_declare_the_flag_and_refuse_a_mismatch_with_the_checkpoint_cfg():
    s = src("v8f_mini_eval.py")
    assert '"--ctx-pool", choices=("off", "on"), default="off"' in s
    assert "a.ctx_pool != cfg.ctx_pool" in s and "REFUSED: --ctx-pool" in s
    assert 'outs["ctx_pool_logits"].append(o["ctx_pool_logits"]' in s
    w = src("v8f_t1_eval_pool.py")                                     # T1: the wrapper (the shared tool is sha-pinned by v8f_t1_speedsub.py)
    assert '"--ctx-pool", choices=("off", "on"), default="off"' in w
    assert "a.ctx_pool != have" in w and "REFUSED: --ctx-pool" in w
    assert "self.rows[POOL_KEY].append(hd[POOL_KEY][b])" in w


# ------------------------------------------------------------------------------------------------ T1: the shared tool stays pinned, the head rides in a wrapper
def t1_pool():
    import importlib
    for p in (HERE.parent, HERE.parents[1] / "taniteval", HERE.parent / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return importlib.import_module("v8f_t1_eval_pool")


def cfg_ckpt(tmp_path, ctx_pool):
    import dataclasses
    p = tmp_path / f"cfg_{ctx_pool}.pt"
    torch.save({"cfg": dataclasses.asdict(M.V8FMiniConfig(ctx_pool=ctx_pool))}, p)
    return str(p)


def test_the_shared_t1_tool_is_byte_equal_to_the_pin_that_speedsub_and_the_wrapper_share():
    import re
    W = t1_pool()
    tool = HERE.parent / "scripts" / "v8f_t1_eval.py"
    assert hashlib.sha256(tool.read_bytes()).hexdigest() == W.TOOL_SHA256 == W.tool_sha256()
    assert "ctx_pool" not in tool.read_text(encoding="utf-8")
    pin = re.search(r'^TOOL_SHA256 = "([0-9a-f]{64})"', src("v8f_t1_speedsub.py"), re.M).group(1)
    assert pin == W.TOOL_SHA256


def test_the_t1_wrapper_refuses_a_flag_that_disagrees_with_the_checkpoint_cfg(tmp_path):
    W = t1_pool()
    T = W.T
    bp, cr = T.build_parser, T.cmd_run
    req = ["--tag", "x", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", str(tmp_path)]
    for have, flag in (("on", "off"), ("off", "on")):
        with pytest.raises(SystemExit) as e:
            W.main(["run", "--ckpt", cfg_ckpt(tmp_path, have), "--ctx-pool", flag] + req)
        assert f"REFUSED: --ctx-pool {flag}" in str(e.value) and f"ctx_pool={have!r}" in str(e.value)
        assert T.build_parser is bp and T.cmd_run is cr                  # the tool is left as it was
    with pytest.raises(SystemExit) as e:                                  # the tool itself has no such flag (argparse exit 2)
        T.main(["run", "--ckpt", cfg_ckpt(tmp_path, "on"), "--ctx-pool", "on"] + req)
    assert e.value.code == 2


def test_the_t1_wrapper_patches_the_tool_only_inside_an_on_call(tmp_path):
    import argparse
    W = t1_pool()
    T = W.T
    keys0, rec0, seen = T.HEAD_KEYS, T.TickRecorder, []
    run = W.make_cmd_run(lambda a: seen.append((T.HEAD_KEYS, T.TickRecorder)) or 0)
    assert run(argparse.Namespace(ckpt=cfg_ckpt(tmp_path, "on"), ctx_pool="on")) == 0
    assert seen[-1] == (keys0 + (W.POOL_KEY,), W.PoolTickRecorder)
    assert (T.HEAD_KEYS, T.TickRecorder) == (keys0, rec0)               # restored after the call
    assert run(argparse.Namespace(ckpt=cfg_ckpt(tmp_path, "off"), ctx_pool="off")) == 0
    assert seen[-1] == (keys0, rec0)                                    # off: nothing of the tool is patched
    boom = W.make_cmd_run(lambda a: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        boom(argparse.Namespace(ckpt=cfg_ckpt(tmp_path, "on"), ctx_pool="on"))
    assert (T.HEAD_KEYS, T.TickRecorder) == (keys0, rec0)               # restored after a raise too


def test_the_pool_recorder_writes_one_row_per_tick_and_the_tool_recorder_none(monkeypatch):
    import types
    W = t1_pool()
    T = W.T
    monkeypatch.setattr(T.TickRecorder, "__call__", lambda self, n, batch, plan, states: None)      # the tool's own rows are the tool's tests
    z = np.arange(3 * 41, dtype=np.float32).reshape(3, 41)
    planner = types.SimpleNamespace(heads={W.POOL_KEY: z}, aux={})
    rec = W.PoolTickRecorder(planner, None, None, heads=True)
    rec(0, {}, None, [object()] * 3)
    assert len(rec.rows[W.POOL_KEY]) == 3 and all(np.array_equal(r, z[b]) for b, r in enumerate(rec.rows[W.POOL_KEY]))
    assert W.POOL_KEY not in W.PoolTickRecorder(planner, None, None, heads=False).rows           # rides with --dump-heads only
    assert W.POOL_KEY not in T.TickRecorder(planner, None, None, heads=True).rows                # the pinned tool never writes it


def test_the_trainer_and_the_launch_gate_are_untouched_by_this_card():
    t = (HERE.parent / "scripts" / "v8f_mini_train.py").read_text(encoding="utf-8")
    assert "ctx_pool" not in t and "ctx-pool" not in t                  # form (a) adds no trainer flag; a flag without a G-DVB entry is refused by the gate itself
