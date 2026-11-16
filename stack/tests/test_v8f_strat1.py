"""PREREG_V8F_STRAT1 (BUILD_SPEC section 9): ``--nav-source`` / ``--nav-text-*`` / ``--ctx-*`` / ``--w-ctx`` in ``v8f_mini.py``, ``v8f_mini_data.py``, ``v8f_mini_train.py``, ``v8f_mini_eval.py``, ``v8f_t1_eval.py``.

Every default DISABLES the lever. THIS file proves reachability from the real caller (CLAUDE.md "built, tested and unreachable from its caller"): argv -> make_config -> cfg -> the BUILT model / optimiser groups / label arrays
-> declared_vs_built (bad_ctx) -> ctx_head_probe + nav_source_probe (selfcheck.json) -> ctx_stage BEFORE the ego stages -> the batch the LOSS receives. Each mechanism has a MUTATION that must turn its check RED
(``mutant`` below rewrites ONE line of the source and re-runs the same check against the mutated module / trainer).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
SNAP = Path(__file__).with_name("v8f_mini_tip_snapshot_strat1.txt")
SNAP_SHA = "388f453a752d9c0cf20f04838d6c63d03ee7a9c0a7511401433d205bd91c9dd3"        # sha256 of the cdbff971 stack/tanitad/models/v8f_mini.py
MODEL_P, TRAIN_P, DATA_P = (REPO / "stack" / "tanitad" / "models" / "v8f_mini.py", REPO / "stack" / "scripts" / "v8f_mini_train.py", REPO / "stack" / "tanitad" / "data" / "v8f_mini_data.py")
RUN = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
REAL_TABLE = Path("F:/TanitAD-scratch/strat1/labels/strat_labels_table.npz")
CTX_KEYS = {"ctx_head.0.weight", "ctx_head.0.bias", "ctx_head.2.weight", "ctx_head.2.bias"}


# ------------------------------------------------------------------------------------------------ fixtures
def write_ctx_table(path, J, seed=0, n_clips=None):
    """A strat-labels-shaped table ROW-ALIGNED to the join J: random ids; flag 2 on a few steps, one invalid cell per variable, twilight present."""
    rng = np.random.default_rng(seed)
    N = int(J["off"][-1] + J["n"][-1])
    z = {"train_clip": np.asarray(J["clip"]).astype(str), "train_off": np.asarray(J["off"]), "train_n": np.asarray(J["n"]), "train_flag": np.zeros(N, np.uint8)}
    z["train_flag"][0:2] = 2
    for v, (nm, n) in enumerate(zip(M.CTX_VARS, M.CTX_N)):
        ids = rng.integers(0, n, N).astype(np.int8)
        ok = np.ones(N, bool)
        ok[20 + v] = False
        ids[~ok] = -1
        z[f"train_{nm}_id"], z[f"train_{nm}_valid"] = ids, ok
        z[f"vocab_{nm}"] = np.array([f"c{i}" for i in range(n)])
    z["train_illumination_id"][30:40] = M.CTX_TWILIGHT_ID
    z["train_illumination_valid"][30:40] = True
    np.savez(path, **z)
    return path


def write_nav_npz(path, J, seed=0):
    """A text-nav file row-aligned to J: a different (valid) nav than the join's."""
    rng = np.random.default_rng(seed)
    N = int(J["off"][-1] + J["n"][-1])
    nav = np.zeros((N, 5), np.float32)
    side = rng.integers(0, 3, N)
    nav[np.arange(N), side] = 1.0
    nav[:, 3] = rng.uniform(0, 3, N).astype(np.float32)
    nav[:, 4] = 1.0
    np.savez(path, train_clip=np.asarray(J["clip"]).astype(str), train_off=np.asarray(J["off"]), train_n=np.asarray(J["n"]), train_nav=nav, train_nav_d=(nav[:, 3] * 100).astype(np.float32))
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("strat1world")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    return {"root": root, "ctx": write_ctx_table(root / "ctx.npz", J), "nav": write_nav_npz(root / "nav.npz", J), **w}


def make_batch(world, n=16, seed=0, hmax=10):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    return D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx]), J


def CLof(world, J):
    return D.ctx_label_arrays(J, world["ctx"], sha(world["ctx"]), "train")


def with_ctx(b, CL):
    return {**b, "ctx_y": TR.ctx_stage(b, CL)}


def build(J, mod=M, seed=0, wake=True, **kw):
    cfg = mod.V8FMiniConfig(**{**TINY, **kw})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    m = mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd)
    if wake:
        with torch.no_grad():
            w = m.head_o[2].weight
            w.copy_(torch.randn(w.shape, generator=torch.Generator().manual_seed(11)) * 0.05)
    return m


def run_loss(m, b, seed=3):
    torch.manual_seed(seed)
    return m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(seed))


def mod_from_text(text, name, path):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(text, str(path), "exec"), mod.__dict__)
    return mod


def tip_module():
    return sys.modules.get("v8f_mini_tip_snapshot_strat1") or mod_from_text(SNAP.read_text(encoding="utf-8"), "v8f_mini_tip_snapshot_strat1", SNAP)


def mutant(old, new, name="v8f_mini_mut_strat1"):
    """The model module with ONE edit (the mutation arm of a check): it must make the same check fail."""
    text = MODEL_P.read_text(encoding="utf-8")
    assert text.count(old) == 1, (old[:60], text.count(old))
    return mod_from_text(text.replace(old, new), name, MODEL_P)


def args(world, extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", *extra])


# ------------------------------------------------------------------------------------------------ pins
def test_the_snapshot_is_the_tip_module():
    assert hashlib.sha256(SNAP.read_bytes()).hexdigest() == SNAP_SHA


def test_pins_constants_and_the_real_table():
    assert M.CTX_VARS == ("country", "weather", "surface", "illumination", "period_of_day") and M.CTX_N == (25, 4, 3, 4, 5) and M.CTX_DIM == 41
    assert M.CTX_VAR_W == (1.0, 1.0, 0.5, 1.0, 1.0) and M.CTX_TWILIGHT_ID == 3 and M.NAV_SOURCES == ("v9", "text")
    assert M.CTX_KEY_PREFIXES[0] in M.NEW_LEVER_KEY_PREFIXES
    if REAL_TABLE.exists():
        with np.load(REAL_TABLE, allow_pickle=False) as z:
            assert tuple(len(z[f"vocab_{v}"]) for v in M.CTX_VARS) == M.CTX_N
            assert [str(x) for x in z["vocab_illumination"]][M.CTX_TWILIGHT_ID] == "twilight"


def test_every_new_flag_defaults_off_and_has_a_flag_check():
    a = args(None)
    assert (a.nav_source, a.nav_text_npz, a.nav_text_sha256, a.ctx_heads, a.w_ctx, a.ctx_labels, a.ctx_labels_sha256, a.ctx_lr_mult, a.ctx_detach) == ("v9", "", "", "off", 0.0, "", "", 1.0, "off")
    for k in ("nav_source", "nav_text_npz", "nav_text_sha256", "ctx_heads", "w_ctx", "ctx_labels", "ctx_labels_sha256", "ctx_lr_mult", "ctx_detach"):
        assert k in TR.FLAG_CHECKS


def test_the_ctx_head_and_the_platform_sidecar_stay_out_of_the_trainer():
    src = TRAIN_P.read_text(encoding="utf-8") + DATA_P.read_text(encoding="utf-8") + MODEL_P.read_text(encoding="utf-8")
    assert "strat_platform_by_clip" not in src and "platform_id" not in src and "visibility_id" not in src         # LABEL_SPEC sections 2 and 5


# ------------------------------------------------------------------------------------------------ T-OFF / T-RNG / T-HYG
def check_off(mod, world):
    """OFF == TIP: state dict, loss terms, total, forward_infer and gradients bit-equal to the cdbff971 module; no new key. Returns the problems."""
    tip = tip_module()
    b, J = make_batch(world)
    mt, mn = build(J, mod=tip), build(J, mod=mod)
    probs = []
    if list(mt.state_dict()) != list(mn.state_dict()):
        probs.append("state dict keys differ")
        return probs
    if not all(torch.equal(v, mn.state_dict()[k]) for k, v in mt.state_dict().items()):
        probs.append("init tensors differ")
    ot, on = run_loss(mt, b), run_loss(mn, b)
    if set(ot["terms"]) != set(on["terms"]) or not all(torch.equal(ot["terms"][k], on["terms"][k]) for k in ot["terms"]):
        probs.append("loss terms differ")
    ot["terms"]["total"].backward(); on["terms"]["total"].backward()
    if not all(torch.equal(p.grad, q.grad) for p, q in zip(mt.parameters(), mn.parameters()) if p.grad is not None):
        probs.append("gradients differ")
    mt.eval(); mn.eval()
    with torch.no_grad():
        it, inn = mt.forward_infer(b), mn.forward_infer(b)
    if set(it) != set(inn) or not all(torch.equal(it[k], inn[k]) for k in it if torch.is_tensor(it[k])):
        probs.append("forward_infer differs")
    return probs


def test_off_equals_tip(world):
    assert check_off(M, world) == []


def test_mutation_unconditional_head_turns_off_equals_tip_red(world):
    mut = mutant('self.ctx_head = None\n        if cfg.ctx_heads == "on":', 'self.ctx_head = None\n        if True:')
    assert check_off(mut, world)


def check_rng(mod, world):
    _, J = make_batch(world)
    torch.manual_seed(123); off = build(J, mod=mod, wake=False); r_off = torch.get_rng_state()
    torch.manual_seed(123); on = build(J, mod=mod, wake=False, ctx_heads="on", w_ctx=0.5); r_on = torch.get_rng_state()
    same = all(torch.equal(v, on.state_dict()[k]) for k, v in off.state_dict().items())
    return same and torch.equal(r_off, r_on) and set(on.state_dict()) - set(off.state_dict()) == CTX_KEYS


def test_ctx_on_leaves_every_other_init_tensor_and_the_global_rng_alone(world):
    assert check_rng(M, world)


def test_mutation_no_fork_rng_turns_rng_red(world):
    mut = mutant("with torch.random.fork_rng(devices=[]):\n                torch.manual_seed(CTX_INIT_SEED)", "if True:\n                torch.manual_seed(CTX_INIT_SEED)")
    assert not check_rng(mut, world)


def check_hyg(mod):
    bad = [dict(ctx_heads="maybe"), dict(ctx_detach="x"), dict(w_ctx=-1.0), dict(w_ctx=float("nan")), dict(nav_source="osm")]
    n = 0
    for kw in bad:
        try:
            mod.V8FMiniConfig(**kw)
        except ValueError:
            n += 1
    try:
        mod.V8FMiniConfig(ctx_headz="on")
    except TypeError:
        n += 1
    return n == len(bad) + 1


def test_hygiene_refuses_bad_values_and_unknown_keys():
    assert check_hyg(M)


def test_mutation_deleting_the_nav_source_check_turns_hyg_red():
    mut = mutant('        if self.nav_source not in NAV_SOURCES:\n            raise ValueError(f"nav_source must be one of {NAV_SOURCES}, got {self.nav_source!r}")\n', "")
    assert not check_hyg(mut)


# ------------------------------------------------------------------------------------------------ T-CTX-LOSS / T-GRAD
def hand_ctx(logits, y):
    num = den = 0.0
    o = 0
    for v, (n, w) in enumerate(zip(M.CTX_N, M.CTX_VAR_W)):
        t = y[:, v]
        m = t >= 0
        ce = torch.nn.functional.cross_entropy(logits[:, o:o + n][m], t[m].long(), reduction="sum") / max(int(m.sum()), 1) if m.any() else torch.zeros(())
        num, den, o = num + w * ce, den + w, o + n
    return num / den


def check_ctx_loss(mod, world):
    b, J = make_batch(world)
    CL = CLof(world, J)
    bb = with_ctx(b, CL)
    m = build(J, mod=mod, ctx_heads="on", w_ctx=0.5)
    o1 = run_loss(m, bb)
    m.cfg.w_ctx = 0.0
    o0 = run_loss(m, bb)
    moved = float(o1["terms"]["total"] - o0["terms"]["total"])
    want = 0.5 * float(o1["terms"]["ctx"])
    hand = float(hand_ctx(o1["ctx_logits"].detach(), bb["ctx_y"]))
    return abs(moved - want) <= 1e-4 * abs(want) and abs(hand - float(o1["terms"]["ctx"])) < 1e-5 and want > 0


def test_ctx_term_reaches_the_total_with_its_weight_and_matches_a_hand_computation(world):
    assert check_ctx_loss(M, world)


def test_mutation_deleting_the_total_line_turns_the_loss_check_red(world):
    mut = mutant('        total = total + cfg.w_ctx * T["ctx"] if cfg.ctx_heads == "on" else total', "        pass")
    assert not check_ctx_loss(mut, world)


def test_masked_and_twilight_targets_contribute_exactly_zero(world):
    b, J = make_batch(world)
    m = build(J, ctx_heads="on", w_ctx=0.5)
    y = torch.full((16, 5), -1, dtype=torch.int8)
    out = run_loss(m, {**b, "ctx_y": y})
    assert float(out["terms"]["ctx"]) == 0.0
    y2 = y.clone(); y2[:, 3] = M.CTX_TWILIGHT_ID                      # twilight is masked in the LABEL arrays; here the loss sees it as a class: it must be masked by ctx_label_arrays, not here
    CL = CLof(world, J)
    assert (CL["y"][:, 3] != M.CTX_TWILIGHT_ID).all() and CL["n_twilight_masked"] >= 10


def check_grad(mod, world):
    b, J = make_batch(world)
    CL = CLof(world, J)
    bb = with_ctx(b, CL)
    res = {}
    for det in ("off", "on"):
        m = build(J, mod=mod, ctx_heads="on", w_ctx=0.5, ctx_detach=det)
        T = run_loss(m, bb)["terms"]
        ad = [p for mm in (m.adapter, m.pos, m.time) for p in (mm.parameters() if isinstance(mm, torch.nn.Module) else [mm])]
        g = torch.autograd.grad(T["ctx"], ad, allow_unused=True)
        res[det] = sum(float(x.abs().sum()) for x in g if x is not None)
    m = build(J, mod=mod, ctx_heads="on", w_ctx=0.0)
    T = run_loss(m, bb)["terms"]
    h0 = torch.autograd.grad(T["total"], list(m.ctx_head.parameters()), allow_unused=True)
    m = build(J, mod=mod, ctx_heads="on", w_ctx=0.5)
    T = run_loss(m, bb)["terms"]
    h1 = torch.autograd.grad(T["total"], list(m.ctx_head.parameters()), allow_unused=True)
    z0 = sum(float(x.abs().sum()) for x in h0 if x is not None)
    z1 = sum(float(x.abs().sum()) for x in h1 if x is not None)
    return res["off"] > 0.0 and res["on"] == 0.0 and z0 == 0.0 and z1 > 0.0


def test_head_gradient_adapter_gradient_and_the_exact_zero_controls(world):
    assert check_grad(M, world)


def test_mutation_dropping_the_detach_turns_grad_red(world):
    mut = mutant("            tok = tok.detach()\n        return self.ctx_head", "            pass\n        return self.ctx_head")
    assert not check_grad(mut, world)


# ------------------------------------------------------------------------------------------------ T-VIS / T-ISO
def check_vis(mod, world):
    b, J = make_batch(world)
    m = build(J, mod=mod, ctx_heads="on", w_ctx=0.5)
    try:
        mod.assert_ctx_head_vision_only(m, b)
    except AssertionError:
        return False
    return mod.ctx_head_source_problems() == []


def test_the_ctx_head_is_vision_only(world):
    assert check_vis(M, world)


def test_mutation_p10f_in_the_head_turns_vis_red(world):
    mut = mutant("return self.ctx_head(tok.mean(1)).float()", 'return self.ctx_head(tok.mean(1) + b["p10f"][:, -1, :1]).float()')
    assert not check_vis(mut, world)


def check_iso(mod, world):
    b, J = make_batch(world)
    m = build(J, mod=mod, ctx_heads="on", w_ctx=0.5)
    return mod.ctx_isolation_problems(m, b) == []


def test_the_ctx_logits_feed_no_planner_or_goal_output(world):
    assert check_iso(M, world)


def test_mutation_feeding_the_logits_to_s_turns_iso_red(world):
    mut = mutant("        ck_s = self.run_s(b, adapt, last_s)\n        out[\"ckpt\"] = ck_s", "        ck_s = self.run_s(b, adapt, last_s) + (out[\"ctx_logits\"][:, :1, None] * 0.01 if \"ctx_logits\" in out else 0.0)\n        out[\"ckpt\"] = ck_s")
    assert not check_iso(mut, world)


# ------------------------------------------------------------------------------------------------ T-LBL / T-STAGE / T-NAV
def test_ctx_labels_mask_flag_invalid_twilight_and_refuse_misalignment_and_a_wrong_sha(world, tmp_path):
    _, J = make_batch(world)
    CL = CLof(world, J)
    assert (CL["y"][0:2] == -1).all()                                  # flag 2
    for v in range(5):
        assert CL["y"][20 + v, v] == -1                                # invalid cell
    assert (CL["y"][30:40, 3] == -1).all() and CL["n_twilight_masked"] >= 10
    assert CL["y"].dtype == np.int8 and CL["y"].shape == (int(J["off"][-1] + J["n"][-1]), 5)
    with pytest.raises(ValueError, match="sha256"):
        D.ctx_label_arrays(J, world["ctx"], "0" * 64, "train")
    with pytest.raises(ValueError, match="sha256"):
        D.ctx_label_arrays(J, world["ctx"], "", "train")
    z = dict(np.load(world["ctx"], allow_pickle=False))
    z["train_off"] = z["train_off"] + 1
    bad = tmp_path / "shift.npz"
    np.savez(bad, **z)
    with pytest.raises(ValueError, match="train_off"):
        D.ctx_label_arrays(J, bad, sha(bad), "train")


def test_ctx_stage_reads_the_logged_frame_and_runs_before_every_ego_stage(world):
    b, J = make_batch(world)
    CL = CLof(world, J)
    y = TR.ctx_stage(b, CL)
    for i in range(len(y)):
        assert (y[i].numpy() == CL["y"][int(J["off"][int(b["ci"][i])]) + int(b["t"][i])]).all()
    src = TRAIN_P.read_text(encoding="utf-8")
    i_main = src.index("def main(")
    loop = src[i_main:]
    i_stage = loop.index("ctx_lab = ctx_stage(b, CL)")
    assert i_stage < loop.index("b = comp.apply(b, step)") and i_stage < loop.index("b = ego.apply(b, step)") and i_stage < loop.index("b = dag.apply(b, step)") and i_stage < loop.index("b = rcv.apply(b, step)")
    assert loop.index('b["ctx_y"] = ctx_lab') > loop.index("b = dag.apply(b, step)")
    mut = src[:i_main] + loop.replace("ctx_lab = ctx_stage(b, CL)", "pass", 1).replace('b["ctx_y"] = ctx_lab', 'b["ctx_y"] = ctx_stage(b, CL)', 1)           # the mutation: staged AFTER dag.apply
    ml = mut[i_main:]
    assert not ml.index('b["ctx_y"] = ctx_stage(b, CL)') < ml.index("b = dag.apply(b, step)")


def test_nav_text_arrays_refuses_misalignment_a_wrong_sha_and_differs_from_v9(world, tmp_path):
    _, J = make_batch(world)
    nav, nav_d, rec = D.nav_text_arrays(J, world["nav"], sha(world["nav"]), "train")
    assert rec["n_text_ne_v9"] > 0 and nav.shape == (len(J["nav"]), 5) and rec["sha256"] == sha(world["nav"])
    with pytest.raises(ValueError, match="sha256"):
        D.nav_text_arrays(J, world["nav"], "0" * 64, "train")
    z = dict(np.load(world["nav"], allow_pickle=False))
    z["train_n"] = z["train_n"] + 1
    bad = tmp_path / "bad.npz"
    np.savez(bad, **z)
    with pytest.raises(ValueError, match="train_n"):
        D.nav_text_arrays(J, bad, sha(bad), "train")
    z = dict(np.load(world["nav"], allow_pickle=False))
    z["train_nav"][0, 3] = 7.0
    bad2 = tmp_path / "bad2.npz"
    np.savez(bad2, **z)
    with pytest.raises(ValueError, match="channel 3"):
        D.nav_text_arrays(J, bad2, sha(bad2), "train")


# ------------------------------------------------------------------------------------------------ T-INIT / T-OPT / T-DVB
def test_init_ckpt_into_a_ctx_model_has_exactly_the_ctx_keys_fresh_and_off_is_strict(world, tmp_path, monkeypatch):
    b, J = make_batch(world)
    off = build(J)
    ck = tmp_path / "m2b.pt"
    torch.save({"model": off.state_dict(), "cfg": off.cfg.__dict__, "step": 3, "arm": "M2b", "seed": 0}, ck)
    on = build(J, ctx_heads="on", w_ctx=0.5)
    rec = TR.load_init_ckpt(str(ck), on, on.cfg, apply=True)
    assert set(rec["fresh_keys"]) == CTX_KEYS and rec["n_mismatch_after_load"] == 0
    TR.load_init_ckpt(str(ck), build(J), build(J).cfg, apply=True)                       # off: strict, no fresh keys
    monkeypatch.setattr(M, "NEW_LEVER_KEY_PREFIXES", tuple(p for p in M.NEW_LEVER_KEY_PREFIXES if p not in M.CTX_KEY_PREFIXES))          # the mutation: the prefix is dropped
    with pytest.raises(Exception):
        TR.load_init_ckpt(str(ck), build(J, ctx_heads="on", w_ctx=0.5), on.cfg, apply=True)


def test_dvb_refuses_each_misdeclared_flag_and_passes_the_declared_ones(world):
    b, J = make_batch(world)
    CL = CLof(world, J)

    def gate(extra, **over):
        a = args(world, extra)
        cfg = TR.make_config(a)
        cfg = M.V8FMiniConfig(**{**cfg.__dict__, **{k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}, "d": 32, "ff": 64, **over})
        mu, sd = M.cons_stats(J)
        model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
        ids = {id(p) for p in model.ctx_head.parameters()} if model.ctx_head is not None else set()
        groups = [{"params": [p for p in model.parameters() if id(p) not in ids], "weight_decay": a.wd}]
        if ids:
            groups.append({"params": list(model.ctx_head.parameters()), "weight_decay": a.wd, "lr": a.lr * a.ctx_lr_mult, "name": "ctx", "lr_mult": a.ctx_lr_mult})
        opt = torch.optim.AdamW(groups, lr=a.lr)
        info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True, "kin_mode": a.kin_mode, "p10_units": None,
                "nav_source": {"source": a.nav_source, "join_nav_digest": "x"}}
        if a.nav_source == "text":
            info["nav_source"].update({"sha256": sha(world["nav"]), "nav_digest": "d", "n_text_ne_v9": 5})
        if a.ctx_heads == "on":
            info["ctx_labels"] = {"path": os.path.abspath(a.ctx_labels), "sha256": sha(world["ctx"]), "valid_share": {v: 0.99 for v in M.CTX_VARS}}
        return TR.bad_ctx(a, model, info, opt)

    on = ["--ctx-heads", "on", "--w-ctx", "0.5", "--ctx-labels", str(world["ctx"]), "--ctx-labels-sha256", sha(world["ctx"]), "--ctx-lr-mult", "10"]
    txt = ["--nav-source", "text", "--nav-text-npz", str(world["nav"]), "--nav-text-sha256", sha(world["nav"])]
    assert gate([]) == [] and gate(on, ctx_heads="on", w_ctx=0.5) == [] and gate(txt, nav_source="text") == []
    assert any("w_ctx is 0" in m for m in gate(["--ctx-heads", "on", "--ctx-labels", str(world["ctx"]), "--ctx-labels-sha256", sha(world["ctx"])], ctx_heads="on"))
    assert gate(["--w-ctx", "0.5"])                                                                   # a weight with no head
    assert gate(["--ctx-lr-mult", "10"]) and gate(["--ctx-detach", "on"])                              # levers with no head
    assert gate(["--ctx-heads", "on", "--w-ctx", "0.5", "--ctx-labels", str(world["ctx"]), "--ctx-labels-sha256", "0" * 64], ctx_heads="on", w_ctx=0.5)   # sha mismatch
    assert gate(["--ctx-labels", str(world["ctx"])])                                                   # labels named with the head off
    assert gate(["--nav-text-npz", str(world["nav"])])                                                 # a text file with source v9
    assert gate(["--nav-source", "text", "--nav-text-npz", str(world["nav"]), "--nav-text-sha256", "0" * 64], nav_source="text")
    assert gate(txt, nav_source="v9")                                                                  # declared text, built v9
    assert gate(on, ctx_heads="off")                                                                   # declared on, built off


def test_remove_a_flag_check_entry_is_refused_by_declared_vs_built(world, monkeypatch):
    b, J = make_batch(world)
    a = args(world)
    cfg = TR.make_config(a)
    cfg = M.V8FMiniConfig(**{**cfg.__dict__, **{k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}, "d": 32, "ff": 64})
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), *M.cons_stats(J))
    opt = torch.optim.AdamW([p for p in model.parameters()], lr=a.lr)
    info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True, "kin_mode": a.kin_mode, "p10_units": None, "nav_source": {"source": "v9", "join_nav_digest": "x"}}
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != "ctx_lr_mult"})
    assert any("ctx_lr_mult" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))


# ------------------------------------------------------------------------------------------------ end to end on the synthetic world
def _sc(out):
    return json.loads((Path(out) / "selfcheck.json").read_text(encoding="utf-8"))


def _argv(world, tag, extra):
    return ["--out", str(Path(world["root"]) / tag), *RUN, *extra]


def C0(world):
    return ["--nav-source", "v9", "--nav-text-npz", "", "--nav-text-sha256", "", "--ctx-heads", "off", "--w-ctx", "0", "--ctx-labels", "", "--ctx-labels-sha256", "", "--ctx-lr-mult", "1", "--ctx-detach", "off"]


def S1(world, detach="off"):
    return ["--nav-source", "text", "--nav-text-npz", str(world["nav"]), "--nav-text-sha256", sha(world["nav"]), "--ctx-heads", "on", "--w-ctx", "0.5", "--ctx-labels", str(world["ctx"]),
            "--ctx-labels-sha256", sha(world["ctx"]), "--ctx-lr-mult", "10", "--ctx-detach", detach]


def N1(world):
    return C0(world)[:0] + ["--nav-source", "text", "--nav-text-npz", str(world["nav"]), "--nav-text-sha256", sha(world["nav"])] + C0(world)[6:]


def test_end_to_end_c0_n1_s1_selfchecks_and_steps(world):
    for tag, extra in (("c0", C0(world)), ("n1", N1(world)), ("s1", S1(world))):
        assert TR.main(_argv(world, tag, ["--steps", "3", *extra])) == 0, tag
        sc, sm = _sc(Path(world["root"]) / tag), json.loads((Path(world["root"]) / tag / "summary.json").read_text(encoding="utf-8"))
        assert sc["declared_vs_built"] == "PASS" and sc["nav_source_probe"]["pass"] is True
        keys = [k for k in torch.load(Path(world["root"]) / tag / "ckpt_last.pt", weights_only=False)["model"] if k.startswith("ctx_head.")]
        import strat1_check as SC                                       # the study checker reads the SAME selfcheck / checkpoint
        toks = json.loads((Path(world["root"]) / tag / "argv.json").read_text(encoding="utf-8"))["argv"]
        kind = {"c0": "C0", "n1": "N1", "s1": "S1"}[tag]
        probs = SC.gate_problems(kind, toks, sc)
        assert [x for x in probs if "steps_total" not in x and "sha 8055a321" not in x and "inherited probe" not in x and "detach-upper" not in x] == [], probs         # a 3-step synthetic run: only the steps, real-label-sha, inherited-probe and detach-upper lines differ (the synthetic argv is not the OBD-CTL recipe)
        gp, rec = SC.geval(str(Path(world["root"]) / tag / "ckpt_last.pt"), sc["levers"])
        assert gp == [] and (rec.get("regression_arm_refused") is True) == (tag == "s1")
        assert SC.gate_problems("S1" if tag != "s1" else "C0", toks, sc), "a kind mismatch must be refused"
        assert SC.hyg() == []
        if tag == "s1":
            assert sc["ctx_head_probe"]["pass"] and sc["nav_source_probe"]["n_text_ne_v9"] > 0 and set(keys) == CTX_KEYS
            assert sc["ctx_head_probe"]["control_head_grad_w0"] == 0.0 and sc["ctx_head_probe"]["control_allmasked_head_grad"] == 0.0 and sc["ctx_head_probe"]["label_staging_equal"]
            assert sc["live"]["groups_grad"]["CTX"] > 0.0 and "ctx" in sc["live"]["terms"] and all(f"ctx/{v}" in sc["live"]["terms"] for v in M.CTX_VARS)
            assert sm["ctx_heads"] == "on" and sm["nav_source"] == "text"
            ck = torch.load(Path(world["root"]) / tag / "ckpt_last.pt", weights_only=False)
            assert ck["cfg"]["ctx_heads"] == "on" and ck["cfg"]["nav_source"] == "text" and ck["nav_text_sha256"] == sha(world["nav"]) and ck["ctx_labels_sha256"] == sha(world["ctx"])
            names = {g.get("name") for g in ck["opt"]["param_groups"]}
            assert {"ctx", "ctx_nd"} <= names
            tl = [json.loads(x) for x in (Path(world["root"]) / tag / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
            assert all("ctx" in r and np.isfinite(r["ctx"]) for r in tl)
        else:
            assert keys == [] and "ctx_head_probe" not in sc
            assert sc["nav_source_probe"]["source"] == ("v9" if tag == "c0" else "text")
            assert (sc["nav_source_probe"].get("n_text_ne_v9", 0) > 0) == (tag == "n1")


def test_the_loop_trains_the_ctx_head_and_moves_it(world):
    a = torch.load(Path(world["root"]) / "s1" / "ckpt_last.pt", weights_only=False)["model"]
    r = TR.main(_argv(world, "s1b", ["--steps", "3", *S1(world)]))
    assert r == 0
    # the head moved from its seeded init (CTX_INIT_SEED), and a head that never trained would equal it exactly
    init = M.V8FMini(M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, ctx_heads="on", w_ctx=0.5), M.P10Prior.synthetic(0), *M.cons_stats(D.load_join(Path(world["root"]) / "join_train.npz")))
    assert any(not torch.equal(a[k], init.state_dict()[k]) for k in CTX_KEYS)


def mutant_trainer(old, new, tag):
    """The trainer with ONE edit, written beside the real one (so its REPO path logic holds), imported, and removed."""
    text = TRAIN_P.read_text(encoding="utf-8")
    assert text.count(old) == 1, (old[:60], text.count(old))
    p = TRAIN_P.with_name(f"_mut_{tag}.py")
    p.write_text(text.replace(old, new), encoding="utf-8")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"_mut_{tag}", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        p.unlink()


def test_mutation_deleting_the_nav_overwrite_turns_the_selfcheck_red(world):
    mod = mutant_trainer('        J["nav"], J["nav_d"], rec_ = D.nav_text_arrays(J, a.nav_text_npz, a.nav_text_sha256, "train")', '        rec_ = D.nav_text_arrays(J, a.nav_text_npz, a.nav_text_sha256, "train")[2]', "nav")
    with pytest.raises(SystemExit) as e:
        mod.main(_argv(world, "mut_nav", ["--steps", "1", "--selfcheck-only", *N1(world)]))
    assert "ZZDVB-REFUSEDZZ" in str(e.value)


def test_mutation_deleting_the_ctx_stage_attach_turns_the_loop_red(world):
    mod = mutant_trainer('                    b["ctx_y"] = ctx_lab\n', '                    pass\n', "attach")
    with pytest.raises(ValueError, match="ctx_y"):
        mod.main(_argv(world, "mut_attach", ["--steps", "2", *S1(world)]))


def test_mutation_putting_the_head_in_decay_too_turns_the_optimiser_check_red(world):
    mod = mutant_trainer('and id(p) not in ctx_ids]\n    nodecay', ']\n    nodecay', "opt")
    with pytest.raises((SystemExit, ValueError, Exception)):
        mod.main(_argv(world, "mut_opt", ["--steps", "1", "--selfcheck-only", *S1(world)]))


def test_the_ctx_head_off_argv_names_every_flag_and_the_flag_audit_lists_them():
    argv = C0(None)
    for fl in ("--nav-source", "--nav-text-npz", "--nav-text-sha256", "--ctx-heads", "--w-ctx", "--ctx-labels", "--ctx-labels-sha256", "--ctx-lr-mult", "--ctx-detach"):
        assert fl in argv
    a = TR.build_parser().parse_args(["--out", "x", *argv])
    assert a.ctx_heads == "off" and a.nav_source == "v9"


# ------------------------------------------------------------------------------------------------ T-EVAL (the evaluators)
def test_t1_nav_variants_change_the_nav_only_and_real_is_the_joins_rows(world):
    import v8f_t1_eval as T1
    _, J = make_batch(world)
    ci, t0 = 1, 20
    o, n = int(J["off"][ci]), int(J["n"][ci])
    real = T1.nav_rows_variant(J, ci, t0, "real")
    assert np.array_equal(real, np.asarray(J["nav"][o:o + n], np.float32))                     # --nav-variant real == the tool's rows, bit for bit
    zero = T1.nav_rows_variant(J, ci, t0, "zero")
    assert (zero == 0).all() and zero.shape == real.shape
    don = T1.nav_rows_variant(J, ci, t0, "donor", donor=(3, 25))
    do = int(J["off"][3])
    assert np.array_equal(don[t0], np.asarray(J["nav"][do + 25], np.float32)) and np.array_equal(don[t0 - 4], np.asarray(J["nav"][do + 21], np.float32))
    assert np.array_equal(don[t0 + 1:], real[t0 + 1:])                                          # the own clip's later rows are untouched
    mut = T1.nav_rows_variant.__code__                                                           # (the mutation 'make real read the donor' is a one-line change to the first branch: pinned by the equality above)
    assert mut is not None


def test_both_evaluators_refuse_a_nav_source_mismatch_in_their_source():
    for f in ("v8f_mini_eval.py", "v8f_t1_eval.py"):
        src = (REPO / "stack" / "scripts" / f).read_text(encoding="utf-8")
        assert 'cfg.nav_source == "text" and not (a.nav_text_npz and a.nav_text_sha256)' in src and 'cfg.nav_source == "v9" and (a.nav_text_npz or a.nav_text_sha256)' in src
        assert 'D.nav_text_arrays(' in src and '"eval")' in src

