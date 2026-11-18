"""PREREG_V8F_NAVRB2: the ROUNDABOUT state channel ``nav_rb`` (``--nav-rb``) of the route-text nav, its builder, and its reachability from the trainer.
Z0 flag-off == the 4dedac45 model (snapshot, bit for bit); Z1 flag-on at its zero init == flag-off (outputs, loss, RNG); Z2 gradient > 0 on the real channel, EXACTLY 0.0 on the zero channel;
G-CLOCK-RB (builder rule, causality, regression arm FAILS); the DAgger swap holds nav_rb as logged; mutations (forward term, join line, window item, swap partition) each turn their check RED. CPU only."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
import build_navtext_nav_rb as BRB  # noqa: E402

TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
SNAP = Path(__file__).with_name("v8f_mini_tip_snapshot_navrb2.txt")
SNAP_SHA = "776643b52b0a3286ef60d12c7c9088960ea1ab7f15030b5d4c8722d17bd217eb"        # sha256 of the 4dedac45 (= a4357e79 = 22999538) stack/tanitad/models/v8f_mini.py
MODEL_P, TRAIN_P, DATA_P = (REPO / "stack" / "tanitad" / "models" / "v8f_mini.py", REPO / "stack" / "scripts" / "v8f_mini_train.py", REPO / "stack" / "tanitad" / "data" / "v8f_mini_data.py")
RUN = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
REAL_IN = Path("F:/TanitAD-scratch/strat1/read/inputs")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_navrb_npz(path, J, seed=0, p=0.3, split="train"):
    rng = np.random.default_rng(seed)
    N = int(J["off"][-1] + J["n"][-1])
    rb = (rng.uniform(size=(N, 1)) < p).astype(np.float32)
    np.savez(path, **{f"{split}_clip": np.asarray(J["clip"]).astype(str), f"{split}_off": np.asarray(J["off"]), f"{split}_n": np.asarray(J["n"]), f"{split}_nav_rb": rb})
    return path


def write_nav_npz(path, J, seed=0):
    rng = np.random.default_rng(seed)
    N = int(J["off"][-1] + J["n"][-1])
    nav = np.zeros((N, 5), np.float32)
    nav[np.arange(N), rng.integers(0, 3, N)] = 1.0
    nav[:, 3] = rng.uniform(0, 3, N).astype(np.float32)
    nav[:, 4] = 1.0
    np.savez(path, train_clip=np.asarray(J["clip"]).astype(str), train_off=np.asarray(J["off"]), train_n=np.asarray(J["n"]), train_nav=nav, train_nav_d=(nav[:, 3] * 100).astype(np.float32))
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("navrb2world")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    return {"root": root, "rb": write_navrb_npz(root / "rb.npz", J), "nav": write_nav_npz(root / "nav.npz", J), **w}


def make_batch(world, n=16, seed=0, hmax=10, rb=True):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    if rb:
        J["nav_rb"], _ = D.nav_rb_arrays(J, world["rb"], sha(world["rb"]), "train")
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    return D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx]), J


def build(J, mod=M, seed=0, **kw):
    cfg = mod.V8FMiniConfig(**{**TINY, **kw})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    m = mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd)
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


def base_module():
    return sys.modules.get("v8f_mini_snapshot_navrb2") or mod_from_text(SNAP.read_text(encoding="utf-8"), "v8f_mini_snapshot_navrb2", SNAP)


def mutant(path, old, new, name):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (old[:60], text.count(old))
    return mod_from_text(text.replace(old, new), name, path)


# ------------------------------------------------------------------------------------------------ pins, flags
def test_the_snapshot_is_the_base_module():
    assert sha(SNAP) == SNAP_SHA


def test_flags_default_off_and_each_has_a_flag_check():
    a = TR.build_parser().parse_args(["--out", "x"])
    assert (a.nav_rb, a.nav_rb_npz, a.nav_rb_sha256, a.nav_rb_variant, a.nav_rb_lr_mult) == ("off", "", "", "real", 1.0)
    for k in ("nav_rb", "nav_rb_npz", "nav_rb_sha256", "nav_rb_variant", "nav_rb_lr_mult"):
        assert k in TR.FLAG_CHECKS and k in TR.WM_ONLY_DEFAULT_DESTS
    assert M.NAV_RB_KEY_PREFIXES[0] in M.NEW_LEVER_KEY_PREFIXES and M.NAV_RB_KEYS == frozenset({"nav_rb"}) and "nav_rb" not in M.INFERENCE_INPUT_KEYS


# ------------------------------------------------------------------------------------------------ Z0: off == base
def check_equal(mt, mn, b, extra_keys=()):
    probs = []
    if [k for k in mn.state_dict() if k not in extra_keys] != list(mt.state_dict()):
        return ["state dict keys differ"]
    if not all(torch.equal(v, mn.state_dict()[k]) for k, v in mt.state_dict().items()):
        probs.append("init tensors differ")
    ot, on = run_loss(mt, b), run_loss(mn, b)
    if set(ot["terms"]) != set(on["terms"]) or not all(torch.equal(ot["terms"][k], on["terms"][k]) for k in ot["terms"]):
        probs.append("loss terms differ")
    ot["terms"]["total"].backward(); on["terms"]["total"].backward()
    pn = dict(mn.named_parameters())
    if not all(torch.equal(p.grad, pn[n].grad) for n, p in mt.named_parameters() if p.grad is not None):
        probs.append("gradients differ")
    mt.eval(); mn.eval()
    with torch.no_grad():
        it, inn = mt.forward_infer(b), mn.forward_infer(b)
    if not all(torch.equal(it[k], inn[k]) for k in it if torch.is_tensor(it[k])):
        probs.append("forward_infer differs")
    return probs


def test_z0_off_equals_the_base_model(world):
    b, J = make_batch(world)
    assert check_equal(build(J, mod=base_module()), build(J), b) == []


def test_z0_mutation_unconditional_channel_turns_red(world):
    mut = mutant(MODEL_P, 'if cfg.nav_rb == "on":', "if True:", "v8f_mini_mut_navrb2_a")
    b, J = make_batch(world)
    assert check_equal(build(J, mod=base_module()), build(J, mod=mut), b)


# ------------------------------------------------------------------------------------------------ Z1: on at the zero init == off, RNG untouched
def test_z1_on_at_zero_init_is_the_off_model_and_leaves_the_rng_alone(world):
    b, J = make_batch(world)
    m_off = build(J); s_off = torch.get_rng_state()
    m_on = build(J, nav_rb="on"); s_on = torch.get_rng_state()
    assert torch.equal(s_off, s_on)
    assert float(m_on.e_nav_rb.weight.abs().sum()) == 0.0 and m_on.e_nav_rb.bias is None and tuple(m_on.e_nav_rb.weight.shape) == (32, 1)
    assert check_equal(m_off, m_on, b, extra_keys=("e_nav_rb.weight",)) == []
    assert m_on.built_levers()["nav_rb"] == "on" and "nav_rb" not in m_off.built_levers()


def test_z1_mutation_no_fork_rng_moves_the_global_stream(world):
    mut = mutant(MODEL_P, "            with torch.random.fork_rng(devices=[]):\n                self.e_nav_rb = nn.Linear(1, d, bias=False)\n",
                 "            if True:\n                self.e_nav_rb = nn.Linear(1, d, bias=False)\n", "v8f_mini_mut_navrb2_b")
    _, J = make_batch(world)
    build(J, mod=mut, nav_rb="on"); s_mut = torch.get_rng_state()
    build(J, nav_rb="on"); s_off = torch.get_rng_state()
    assert not torch.equal(s_mut, s_off)


# ------------------------------------------------------------------------------------------------ Z2: gradient and the exact-zero control; reachability through S, T and O
def rb_grad(m, b):
    m.zero_grad(set_to_none=True)
    run_loss(m, b)["terms"]["total"].backward()
    g = m.e_nav_rb.weight.grad
    return 0.0 if g is None else float(g.abs().sum())


def test_z2_gradient_positive_on_the_channel_and_exactly_zero_on_the_zero_channel(world):
    b, J = make_batch(world)
    assert int(b["nav_rb"].sum()) > 0
    m = build(J, nav_rb="on")
    assert rb_grad(m, b) > 0.0
    assert rb_grad(m, {**b, "nav_rb": torch.zeros_like(b["nav_rb"])}) == 0.0
    with torch.no_grad():
        m.e_nav_rb.weight.fill_(0.5)
    m.eval()
    with torch.no_grad():
        o1, o0 = m.forward_infer(b), m.forward_infer({**b, "nav_rb": torch.zeros_like(b["nav_rb"])})
    assert any(not torch.equal(o1[k], o0[k]) for k in o1 if torch.is_tensor(o1[k]))         # a trained weight moves the outputs: the channel reaches the plan
    assert "nav_rb" in o1["_keys_read"]


def test_z2_mutation_deleting_the_forward_term_turns_red(world):
    mut = mutant(MODEL_P, '            t = t + self.e_nav_rb(self._at(b["nav_rb"], i).to(self.e_nav_rb.weight.dtype))\n', "            pass\n", "v8f_mini_mut_navrb2_c")
    b, J = make_batch(world)
    assert rb_grad(build(J, mod=mut, nav_rb="on"), b) == 0.0


def test_a_built_channel_with_no_batch_key_is_a_keyerror_never_a_silent_zero(world):
    b, J = make_batch(world, rb=False)
    m = build(J, nav_rb="on")
    m.eval()
    with pytest.raises(KeyError):
        with torch.no_grad():
            m.forward_infer(b)


def test_admissibility_guard_admits_nav_rb_only_for_a_model_built_with_it(world):
    b, J = make_batch(world)
    assert "nav_rb" in M.assert_no_situation_classifier_input(build(J, nav_rb="on"), b)["read"]
    mut = mutant(MODEL_P, '        allowed = allowed | (NAV_RB_KEYS if getattr(model, "e_nav_rb", None) is not None else frozenset())', "        pass", "v8f_mini_mut_navrb2_d")
    with pytest.raises(AssertionError):
        mut.assert_no_situation_classifier_input(build(J, mod=mut, nav_rb="on"), b)


# ------------------------------------------------------------------------------------------------ data
def test_nav_rb_arrays_refuses_sha_alignment_shape_values_and_applies_the_zero_variant(world, tmp_path):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    rb, rec = D.nav_rb_arrays(J, world["rb"], sha(world["rb"]), "train")
    assert rb.shape == (int(J["off"][-1] + J["n"][-1]), 1) and rec["n_on"] == int(rb.sum()) > 0
    z, rz = D.nav_rb_arrays(J, world["rb"], sha(world["rb"]), "train", "zero")
    assert float(z.sum()) == 0.0 and rz["n_on"] == 0 and rz["n_on_real"] == rec["n_on"]
    with pytest.raises(ValueError):
        D.nav_rb_arrays(J, world["rb"], "0" * 64, "train")
    with pytest.raises(ValueError):
        D.nav_rb_arrays(J, world["rb"], "", "train")
    bad = dict(np.load(world["rb"], allow_pickle=False))
    bad["train_off"] = bad["train_off"] + 1
    np.savez(tmp_path / "shift.npz", **bad)
    with pytest.raises(ValueError):
        D.nav_rb_arrays(J, tmp_path / "shift.npz", sha(tmp_path / "shift.npz"), "train")
    bad = dict(np.load(world["rb"], allow_pickle=False))
    bad["train_nav_rb"] = bad["train_nav_rb"] * 2.0
    np.savez(tmp_path / "val.npz", **bad)
    with pytest.raises(ValueError):
        D.nav_rb_arrays(J, tmp_path / "val.npz", sha(tmp_path / "val.npz"), "train")


def test_the_window_item_carries_the_rows_of_its_frames(world):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    J["nav_rb"] = np.ascontiguousarray(J["nav"][:, 3:4]).astype(np.float32)          # a row-coded stand-in: the item must take the SAME rows as the nav's frames
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=10)
    for w in wins[:: max(1, len(wins) // 7)]:
        it = ds.item(w[0], w[1], 1, 3)
        assert it["nav_rb"].shape == (10, 1)
        np.testing.assert_array_equal(it["nav_rb"][:, 0], it["nav"][:, 3])
    assert "nav_rb" not in make_batch(world, rb=False)[0]


# ------------------------------------------------------------------------------------------------ builder: rule, G-CLOCK-RB, future reads, admissibility
def toy_table():
    vocab = np.array(["FOLLOW", "TURN_L", "RB_TAKE_EXIT", "ROTARY_EXIT", "RB_ENTER", "TURN_R"])
    cls = np.array([0, 2, 2, 2, 1, 3, 3, 4, -1, 2, 2, 5], np.int16)
    flag = np.array([0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0], np.uint8)
    age = np.array([0.1, 0.0, 0.2, 0.4, 0.1, 0.05, 0.25, 0.0, 0.0, 0.3, 0.5, 0.1], np.float32)
    off, n = np.array([0, 6]), np.array([6, 6])
    clips = np.array(["a", "b"])
    return {"vocab": vocab, "train_cls_id": cls, "train_flag": flag, "train_age_s": age, "train_off": off, "train_n": n, "train_clip": clips}


class Z(dict):
    @property
    def files(self):
        return list(self)


def test_builder_rule_matches_the_hand_values_and_g_clock_rb_fails_its_regression_arm():
    t = Z(toy_table())
    rb, src = BRB.state(t["train_cls_id"], t["train_flag"], t["vocab"], t["train_off"], t["train_n"])
    assert rb[:, 0].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]
    out, rep, ok = BRB.build(t, {"clip": t["train_clip"], "off": t["train_off"], "n": t["train_n"]}, shift_reg=1)
    assert ok and rep["train"]["G-CLOCK-RB"]["pass"] and not rep["train"]["G-CLOCK-RB_regression_arm_shift_1"]["pass"]
    np.testing.assert_array_equal(out["train_nav_rb"], rb)


def test_builder_reads_no_row_after_now_and_a_future_read_mutant_is_red():
    t = toy_table()
    rb, _ = BRB.state(t["train_cls_id"], t["train_flag"], t["vocab"], t["train_off"], t["train_n"])
    for j in range(len(rb)):
        c, f = t["train_cls_id"].copy(), t["train_flag"].copy()
        c[j + 1:] = 2; f[j + 1:] = 0                                                     # garbage after step j
        rb2, _ = BRB.state(c, f, t["vocab"], t["train_off"], t["train_n"])
        assert rb2[j, 0] == rb[j, 0]
    fut, _ = BRB.state(t["train_cls_id"], t["train_flag"], t["vocab"], t["train_off"], t["train_n"], shift=1)
    assert not np.array_equal(fut, rb)


def test_builder_refuses_a_misaligned_table():
    t = Z(toy_table())
    with pytest.raises(SystemExit):
        BRB.build(t, {"clip": t["train_clip"], "off": t["train_off"] + 1, "n": t["train_n"]})


def test_builder_and_model_path_import_no_situation_module_and_read_no_ego_quantity():
    src = Path(BRB.__file__).read_text(encoding="utf-8")
    names = {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names} | {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any("situation" in x or "sitclf" in x for x in names)
    body = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "state"))
    for w in ("track", "speed", "pose", "nav_d", "lat_cls", "v9", "arc"):
        assert w not in body


@pytest.mark.skipif(not (REAL_IN / "navtok_table.npz").exists(), reason="real navtok table not on this machine")
def test_real_table_reproduces_the_navrb_counts(tmp_path):
    assert BRB.main(["--navtok", str(REAL_IN / "navtok_table.npz"), "--align", str(REAL_IN / "navtext_nav.npz"), "--join-eval", str(REAL_IN / "join_eval141.npz"),
                     "--grid", str(REAL_IN / "eval_grid_windows.json"), "--out", str(tmp_path)]) == 0
    rep = json.loads((tmp_path / "navtext_nav_rb_report.json").read_text(encoding="utf-8"))
    ev = rep["eval"]["G-CLOCK-RB"]
    assert (ev["n_on"], ev["grid_windows_on_at_now"], ev["grid_episodes_on_at_now"]) == (1334, 235, 16) and rep["train"]["G-CLOCK-RB"]["n_on"] == 29836
    assert BRB.main(["--navtok", str(REAL_IN / "navtok_table.npz"), "--align", str(REAL_IN / "navtext_nav.npz"), "--grid", str(REAL_IN / "eval_grid_windows.json"),
                     "--expect-rb-windows", "234", "--out", str(tmp_path / "x")]) == 2          # a wrong expected count is refused (no table written)


# ------------------------------------------------------------------------------------------------ the trainer, end to end (G-DVB, G-LIVE, checkpoint)
def N1(world):
    return ["--nav-source", "text", "--nav-text-npz", str(world["nav"]), "--nav-text-sha256", sha(world["nav"]), "--ctx-heads", "off", "--w-ctx", "0", "--ctx-labels", "", "--ctx-labels-sha256", "",
            "--ctx-lr-mult", "1", "--ctx-detach", "off"]


def RB(world, variant="real", mult="10"):
    return ["--nav-rb", "on", "--nav-rb-npz", str(world["rb"]), "--nav-rb-sha256", sha(world["rb"]), "--nav-rb-variant", variant, "--nav-rb-lr-mult", mult]


def _argv(world, tag, extra):
    return ["--out", str(Path(world["root"]) / tag), *RUN, *extra]


def test_end_to_end_n1r_selfcheck_probe_groups_and_checkpoint(world):
    out = Path(world["root"]) / "n1r"
    assert TR.main(_argv(world, "n1r", ["--steps", "3", *N1(world), *RB(world)])) == 0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    p = sc["navrb_probe"]
    assert sc["declared_vs_built"] == "PASS" and p["pass"] and p["keys_read_has_nav_rb"] and p["step0_identity"] is True
    assert p["e_nav_rb_grad"] > 0.0 and p["control_e_nav_rb_grad_zero_channel"] == 0.0 and p["init_weight_abs_sum"] == 0.0
    assert sc["live"]["groups_grad"]["NAVRB"] > 0.0 and sc["levers"]["nav_rb"] == "on" and sc["info"]["nav_rb"]["sha256"] == sha(world["rb"])
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert ck["cfg"]["nav_rb"] == "on" and ck["nav_rb_sha256"] == sha(world["rb"]) and ck["nav_rb_variant"] == "real"
    assert float(ck["model"]["e_nav_rb.weight"].abs().sum()) > 0.0                                     # the loop trained it off its zero init
    g = [x for x in ck["opt"]["param_groups"] if x.get("name") == "navrb"]
    assert len(g) == 1 and g[0]["lr_mult"] == 10.0


def test_end_to_end_zero_variant_keeps_the_weight_exactly_zero(world):
    out = Path(world["root"]) / "n1z"
    assert TR.main(_argv(world, "n1z", ["--steps", "3", *N1(world), *RB(world, "zero")])) == 0
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert float(ck["model"]["e_nav_rb.weight"].abs().sum()) == 0.0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    assert "e_nav_rb.weight" in sc["live"]["declared_dead"] and sc["live"]["groups_grad"]["NAVRB"] == 0.0 and sc["info"]["nav_rb"]["n_on"] == 0


def test_an_off_arm_keeps_its_selfcheck_and_checkpoint_key_sets(world):
    out = Path(world["root"]) / "n1off"
    assert TR.main(_argv(world, "n1off", ["--steps", "2", *N1(world)])) == 0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert "navrb_probe" not in sc and "nav_rb" not in sc["info"] and "nav_rb_sha256" not in ck and not any(k.startswith("e_nav_rb") for k in ck["model"])


@pytest.mark.parametrize("extra,what", [
    (["--nav-rb-npz", "x.npz"], "off with npz"),
    (["--nav-rb-lr-mult", "10"], "off with lr mult"),
    (["--nav-rb-variant", "zero"], "off with variant"),
])
def test_g_dvb_refuses_an_off_channel_with_a_named_lever(world, extra, what):
    with pytest.raises(SystemExit) as e:
        TR.main(_argv(world, "ref_" + what.replace(" ", "_"), ["--steps", "1", "--selfcheck-only", *N1(world), *extra]))
    assert "ZZDVB-REFUSEDZZ" in str(e.value) or "nav-rb" in str(e.value) or "nav_rb" in str(e.value)


def test_g_dvb_refuses_a_wrong_sha_and_the_ego_perturbation(world):
    bad = RB(world)
    bad[bad.index("--nav-rb-sha256") + 1] = "0" * 64
    with pytest.raises((SystemExit, ValueError)):
        TR.main(_argv(world, "ref_sha", ["--steps", "1", "--selfcheck-only", *N1(world), *bad]))
    with pytest.raises(SystemExit):
        TR.main(_argv(world, "ref_ego", ["--steps", "1", "--selfcheck-only", *N1(world), *RB(world), "--ego-perturb-p", "0.1"]))


def mutant_trainer(old, new, tag):
    text = TRAIN_P.read_text(encoding="utf-8")
    assert text.count(old) == 1, (old[:60], text.count(old))
    p = TRAIN_P.with_name(f"_mut_navrb2_{tag}.py")
    p.write_text(text.replace(old, new), encoding="utf-8")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"_mut_navrb2_{tag}", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        p.unlink()


def test_mutation_deleting_the_join_line_turns_the_selfcheck_red(world):
    mod = mutant_trainer('        J["nav_rb"], info["nav_rb"] = D.nav_rb_arrays(J, a.nav_rb_npz, a.nav_rb_sha256, "train", a.nav_rb_variant)', "        pass", "join")
    with pytest.raises(SystemExit) as e:
        mod.main(_argv(world, "mut_join", ["--steps", "1", "--selfcheck-only", *N1(world), *RB(world)]))
    assert "ZZDVB-REFUSEDZZ" in str(e.value)


def test_mutation_dropping_the_item_rows_turns_the_probe_red(world, monkeypatch):
    real_item = D.V8FMiniWindows.item

    def item(self, *a, **k):
        it = real_item(self, *a, **k)
        it.pop("nav_rb", None)
        return it
    monkeypatch.setattr(D.V8FMiniWindows, "item", item)
    with pytest.raises((SystemExit, KeyError)):
        TR.main(_argv(world, "mut_item", ["--steps", "1", "--selfcheck-only", *N1(world), *RB(world)]))


# ORDER MATTERS (pod G-SUITE 2026-11-18): this test drives the trainer's --synthetic world, so it must run BEFORE the imported module-scoped
# `pack` fixture below, which patches D.make_synthetic_world to T=SYN_T until this module ends (offsets then no longer match the nav-text table).
def test_fine_tune_from_a_checkpoint_without_the_channel_passes_the_init_ckpt_gate(world):
    """PREREG_V8F_NAVRB2 (pod mint 2026-11-18): every NAVRB2 arm fine-tunes a checkpoint that has no e_nav_rb; the init-ckpt G-DVB must admit exactly that fresh key, and only when --nav-rb on."""
    base = Path(world["root"]) / "n1base_init"
    assert TR.main(_argv(world, "n1base_init", ["--steps", "2", *N1(world)])) == 0
    ck0 = torch.load(base / "ckpt_last.pt", weights_only=False)
    assert not any(k.startswith("e_nav_rb") for k in ck0["model"])
    out = Path(world["root"]) / "n1r_from_init"
    assert TR.main(_argv(world, "n1r_from_init", ["--steps", "1", "--selfcheck-only", "--init-ckpt", str(base / "ckpt_last.pt"), *N1(world), *RB(world)])) == 0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    ic = sc["info"]["init_ckpt"]
    assert sc["declared_vs_built"] == "PASS" and ic["applied"] and ic["declared_new_keys"] == ["e_nav_rb.weight"] and set(ic.get("fresh_keys", [])) <= {"e_nav_rb.weight"}


# ------------------------------------------------------------------------------------------------ the DAgger swap holds nav_rb as logged
from test_v8f_dagger_train import (DG, GATE_ARGV, _woken, pack)  # noqa: E402,F401  (pack: the module fixture: a trained checkpoint, its DAgger bank, the synthetic world)


def rb_parts(pk, tmp_path, p=0.5, n=24, seed=0):
    J = {k: v for k, v in pk.J.items()}
    rbp = write_navrb_npz(tmp_path / "rb.npz", J, p=0.4)
    a = TR.build_parser().parse_args(["--out", str(tmp_path / "o"), *GATE_ARGV, "--nav-rb", "on", "--nav-rb-npz", str(rbp), "--nav-rb-sha256", sha(rbp), "--detach-upper"])
    cfg = TR.make_config(a)
    assert cfg.nav_rb == "on"
    J["nav_rb"], _ = D.nav_rb_arrays(J, str(rbp), sha(rbp), "train")
    store = D.ClipStore(pk.world, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(pk.world / "native_tracks.pt"))
    wins = D.train_windows(store, J, D.select_train_clips(list(map(str, J["clip"])), 1.0))
    ds = D.V8FMiniWindows(store, J, wins, hmax=cfg.hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    mu, sd = M.cons_stats(J)
    model = _woken(M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd))
    return model, batch, cfg, DG.DaggerSwap(DG.DaggerBank.load(pk.bank, clip_names=J["clip"]), p=p, seed=1, kin_mode="yaw")


def test_dagger_probe_passes_with_nav_rb_and_the_swap_holds_it_as_logged(pack, tmp_path):
    model, b, cfg, dag = rb_parts(pack, tmp_path)
    res = TR.dagger_probe(model, b, cfg, dag)
    assert res["pass"] is True and "nav_rb" in res["keys_read"]
    pert = dag.apply(b, DG.DaggerSwap.PROBE_STEP) if hasattr(dag, "apply") else None
    if pert is not None and pert is not b:
        assert pert["nav_rb"] is b["nav_rb"]


def test_dagger_probe_mutation_without_the_partition_is_red(pack, tmp_path, monkeypatch):
    model, b, cfg, dag = rb_parts(pack, tmp_path)
    monkeypatch.setattr(M, "NAV_RB_KEYS", frozenset())
    with pytest.raises(AssertionError):
        TR.dagger_probe(model, b, cfg, dag)
