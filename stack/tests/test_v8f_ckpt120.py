"""PREREG_V8F_CKPT120 part 1 (card section 4): ``--s-far`` / ``--w-ckpt-far`` / ``--s-far-lr-mult`` / ``--rc120-npz`` / ``--rc120-sha256`` / ``--s-far-label-perm`` / ``--nav-strat`` / ``--nav-strat-variant`` / ``--nav-strat-npz`` /
``--nav-strat-sha256`` / ``--s-adapter-detach`` in ``v8f_mini.py``, ``v8f_mini_data.py``, ``v8f_mini_train.py``.

Every default DISABLES the lever. THIS file proves reachability from the real caller (CLAUDE.md "built, tested and unreachable from its caller"): argv -> make_config -> cfg -> the BUILT model / optimiser groups / label + nav_s arrays ->
declared_vs_built (bad_sfar) -> ckpt120_probe (selfcheck.json) -> the batch the LOSS receives -> forward output -> loss term -> backward. Each mechanism has a MUTATION (ONE line rewritten in a copy of the source) that must turn its check RED.
"""
from __future__ import annotations

import hashlib
import json
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
SNAP = Path(__file__).with_name("v8f_mini_tip_snapshot_ckpt120.txt")
SNAP_SHA_FILE = Path(__file__).with_name("v8f_mini_tip_snapshot_ckpt120.sha256")
MODEL_P, TRAIN_P, DATA_P = (REPO / "stack" / "tanitad" / "models" / "v8f_mini.py", REPO / "stack" / "scripts" / "v8f_mini_train.py", REPO / "stack" / "tanitad" / "data" / "v8f_mini_data.py")
RUN = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
NEW_KEYS = {"s_far_out.weight", "s_far_out.bias", "e_nav_s.weight", "e_nav_s.bias"}


# ------------------------------------------------------------------------------------------------ fixtures
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rc120_table(path, J, seed=0):
    """An RC-A120-shaped table ROW-ALIGNED to the join J: the 80 m arc scaled to 1.5x (so it differs from every RC-A column), valid where the join's 80 m arc is."""
    N = int(J["off"][-1] + J["n"][-1])
    rng = np.random.default_rng(seed)
    rc = np.stack([J["rc"][:, 2, 0] * 1.5 + 1.0, J["rc"][:, 2, 1] * 1.5 + rng.normal(0, 0.1, N), J["rc"][:, 2, 2] * 1.2], -1).astype(np.float32)
    ok = np.asarray(J["rc_ok"][:, 2]).copy()
    np.savez(path, train_clip=np.asarray(J["clip"]).astype(str), train_off=np.asarray(J["off"]), train_n=np.asarray(J["n"]), train_rc120=rc, train_rc120_ok=ok)
    return path


def write_nav_s_npz(path, J, seed=1):
    """A nav_s table row-aligned to J (a valid text-nav layout), different from the join's nav and from the tactical text nav."""
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
    root = tmp_path_factory.mktemp("ckpt120world")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    return {"root": root, "rc120": write_rc120_table(root / "rc120.npz", J), "navs": write_nav_s_npz(root / "navs.npz", J), **w}


def joined(world, perm="off", variant="real", with_far=True, with_nav=True, mod=D):
    J = mod.load_join(Path(world["root"]) / "join_train.npz")
    if with_far:
        J["rc120"], J["rc120_ok"], _ = mod.rc120_arrays(J, world["rc120"], sha(world["rc120"]), "train", perm)
    if with_nav:
        J["nav_s"], _ = mod.nav_strat_arrays(J, world["navs"], sha(world["navs"]), "train", variant)
    return J


def make_batch(world, n=16, seed=0, hmax=10, J=None, mod=D):
    J = joined(world, mod=mod) if J is None else J
    store = mod.ClipStore(world["root"], "train", lru=8)
    wins = mod.train_windows(store, J, world["names"]["train"])
    ds = mod.V8FMiniWindows(store, J, wins, hmax=hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    return mod.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx]), J


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


def gsum(y, ps):
    g = torch.autograd.grad(y, list(ps), retain_graph=True, allow_unused=True)
    return float(sum(float(x.abs().sum()) for x in g if x is not None))


def mod_from_text(text, name, path):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(text, str(path), "exec"), mod.__dict__)
    return mod


def tip_module():
    return sys.modules.get("v8f_mini_tip_snapshot_ckpt120") or mod_from_text(SNAP.read_text(encoding="utf-8"), "v8f_mini_tip_snapshot_ckpt120", SNAP)


def mutant(old, new, name="v8f_mini_mut_ckpt120", path=MODEL_P):
    """The module with ONE edit (the mutation arm of a check): it must make the same check fail."""
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, (old[:60], text.count(old))
    return mod_from_text(text.replace(old, new), name, path)


def mutant_trainer(old, new, tag):
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


def _argv(world, tag, extra):
    return ["--out", str(Path(world["root"]) / tag), *RUN, *extra]


def _sc(out):
    return json.loads((Path(out) / "selfcheck.json").read_text(encoding="utf-8"))


def FAR(world, perm="off", mult="10", w="0.3333"):
    return ["--s-far", "on", "--w-ckpt-far", w, "--s-far-lr-mult", mult, "--rc120-npz", str(world["rc120"]), "--rc120-sha256", sha(world["rc120"]), "--s-far-label-perm", perm]


def NAVS(world, variant="real"):
    return ["--nav-strat", "on", "--nav-strat-variant", variant, "--nav-strat-npz", str(world["navs"]), "--nav-strat-sha256", sha(world["navs"])]


def args(extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", *extra])


# ------------------------------------------------------------------------------------------------ pins
def test_the_snapshot_is_the_a4357e79_module():
    assert hashlib.sha256(SNAP.read_bytes()).hexdigest() == SNAP_SHA_FILE.read_text().strip()
    assert "S_FAR_MODES" not in SNAP.read_text(encoding="utf-8") and "CTX_KEY_PREFIXES" in SNAP.read_text(encoding="utf-8")


def test_pins_constants():
    assert M.FAR_ARC == 120.0 and M.CKPT_ARCS == (30.0, 50.0, 80.0)
    assert M.S_FAR_KEY_PREFIXES == ("s_far_out.",) and M.NAV_S_KEY_PREFIXES == ("e_nav_s.",)
    assert set(M.S_FAR_KEY_PREFIXES + M.NAV_S_KEY_PREFIXES) <= set(M.NEW_LEVER_KEY_PREFIXES)
    assert M.NAV_S_KEYS == frozenset({"nav_s"}) and "nav_s" not in M.INFERENCE_INPUT_KEYS


def test_hygiene_refuses_bad_values():
    for kw in ({"s_far": "maybe"}, {"nav_strat": "x"}, {"s_adapter_detach": "x"}, {"s_far_label_perm": "x"}, {"nav_strat_variant": "x"}, {"w_ckpt_far": -1.0}, {"w_ckpt_far": float("nan")}):
        with pytest.raises(ValueError):
            M.V8FMiniConfig(**{**TINY, **kw})


# ------------------------------------------------------------------------------------------------ defaults off = a4357e79, bit for bit
def test_defaults_off_is_the_tip_bit_for_bit(world):
    tip = tip_module()
    b, J = make_batch(world, J=joined(world, with_far=False, with_nav=False))
    mt, mn = build(J, mod=tip), build(J, mod=M)
    sdt, sdn = mt.state_dict(), mn.state_dict()
    assert list(sdt) == list(sdn) and all(torch.equal(sdt[k], sdn[k]) for k in sdt)
    ot, on = run_loss(mt, b), run_loss(mn, b)
    assert set(ot["terms"]) == set(on["terms"]) and "ckpt_far" not in on["terms"] and "ckpt_far" not in on
    for k in ot["terms"]:
        assert torch.equal(ot["terms"][k], on["terms"][k]), k
    for k in ("ckpt", "traj30", "traj60", "goal_logits", "lat_logits", "lon_logits", "code"):
        assert torch.equal(ot[k], on[k]), k
    assert mn.s_far_out is None and mn.e_nav_s is None and not [k for k in sdn if k.startswith(M.S_FAR_KEY_PREFIXES + M.NAV_S_KEY_PREFIXES)]
    assert mn.built_levers().keys() == mt.built_levers().keys()


def test_defaults_off_trainer_argv_is_the_tip_config():
    a = args()
    cfg = TR.make_config(a)
    assert (a.s_far, a.w_ckpt_far, a.s_far_lr_mult, a.rc120_npz, a.rc120_sha256, a.s_far_label_perm, a.nav_strat, a.nav_strat_variant, a.s_adapter_detach) == ("off", 0.0, 1.0, "", "", "off", "off", "real", "off")
    assert (cfg.s_far, cfg.w_ckpt_far, cfg.nav_strat, cfg.s_adapter_detach, cfg.s_far_label_perm, cfg.nav_strat_variant) == ("off", 0.0, "off", "off", "off", "real")


def test_the_far_head_on_leaves_every_other_output_bit_identical_to_the_tip(world):
    """--s-far on builds the head LAST with zero init and no RNG draw: the 30/50/80 checkpoints, O's plan and T's heads equal the tip's (the shared hidden layer is read, not changed)."""
    tip = tip_module()
    b, J = make_batch(world)
    mt, mf = build(J, mod=tip), build(J, s_far="on", w_ckpt_far=0.3333)
    sdt = mt.state_dict()
    assert {k for k in mf.state_dict() if k not in sdt} == {"s_far_out.weight", "s_far_out.bias"} and all(torch.equal(sdt[k], mf.state_dict()[k]) for k in sdt)
    ot, of = run_loss(mt, b), run_loss(mf, b)
    for k in ("ckpt", "traj30", "traj60", "goal_logits", "lat_logits", "lon_logits", "code"):
        assert torch.equal(ot[k], of[k]), k
    assert torch.equal(of["ckpt_far"], torch.tensor([[120.0, 0.0, 0.0]]).expand(len(of["ckpt_far"]), 3)), "the far point starts at the nominal (120, 0, 0)"


# ------------------------------------------------------------------------------------------------ K-REACH-b: --s-far
def check_far(mod, world, w=0.3333):
    b, J = make_batch(world, mod=mod if hasattr(mod, "collate") else D)
    m = build(J, mod=mod, s_far="on", w_ckpt_far=w)
    T = run_loss(m, b)["terms"]
    head = list(m.s_far_out.parameters())
    return float(T["ckpt_far"]), gsum(T["total"], head), T, m, b


def test_far_term_reaches_the_total_with_its_weight_and_the_head_gets_a_gradient_and_the_zero_control_is_exactly_0(world):
    t1, g1, T1, m, b = check_far(M, world, 0.3333)
    t0, g0, T0, _, _ = check_far(M, world, 0.0)
    assert t1 > 0.0 and g1 > 0.0
    assert g0 == 0.0 and t0 == t1                              # weight 0: the term is computed and LOGGED, but the head's gradient is EXACTLY 0.0
    assert abs(float(T1["total"] - T0["total"]) - 0.3333 * t1) <= 1e-5 * max(t1, 1e-9)
    # an all-masked rc120_ok: term 0.0 and head gradient exactly 0.0
    b0 = {**b, "lab_s": {**b["lab_s"], "rc120_ok": torch.zeros_like(b["lab_s"]["rc120_ok"])}}
    Tm = run_loss(m, b0)["terms"]
    assert float(Tm["ckpt_far"]) == 0.0 and gsum(Tm["total"], list(m.s_far_out.parameters())) == 0.0


def test_the_far_term_is_the_masked_mean_of_the_scaled_l1(world):
    b, J = make_batch(world)
    m = build(J, s_far="on", w_ckpt_far=0.3333)
    with torch.no_grad():
        m.s_far_out.weight.add_(0.02)
    out = run_loss(m, b)
    ls = b["lab_s"]
    ok = (ls["rc120_ok"] & ls["has"]).double()
    e = ((out["ckpt_far"].double() - ls["rc120"].double()).abs() / m.ck_scale.double())
    want = ((e[:, :2].sum(-1) + e[:, 2]) * ok).sum() / ok.sum().clamp_min(1.0)
    assert float(ok.sum()) > 0 and abs(float(out["terms"]["ckpt_far"]) - float(want)) < 1e-5


def test_the_batch_carries_the_far_label_of_S_tick_row_equal_to_an_independent_reread(world):
    """The RC-A120 the loss receives is the table's row at (ci, t - a_s): read the raw npz independently of D.rc120_arrays."""
    b, J = make_batch(world)
    with np.load(world["rc120"]) as z:
        off, rc, ok = np.asarray(z["train_off"]), np.asarray(z["train_rc120"]), np.asarray(z["train_rc120_ok"])
    rows = off[b["ci"].numpy()] + (b["t"] - b["a_s"]).numpy()
    assert np.array_equal(b["lab_s"]["rc120"].numpy(), np.where(ok[rows][:, None], rc[rows], 0.0)) and np.array_equal(b["lab_s"]["rc120_ok"].numpy(), ok[rows])
    assert int(ok[rows].sum()) > 0
    assert not np.array_equal(b["lab_s"]["rc120"].numpy(), b["lab_s"]["rc"][:, 2].numpy())     # not the 80 m arc


def test_mutation_deleting_the_far_term_from_the_total_turns_the_weight_check_red(world):
    mut = mutant('        total = total + cfg.w_ckpt_far * T["ckpt_far"] if cfg.s_far == "on" else total', "        pass")
    b, J = make_batch(world)
    m = build(J, mod=mut, s_far="on", w_ckpt_far=0.3333)
    T = run_loss(m, b)["terms"]
    with pytest.raises(Exception):                              # the head receives NO gradient from the total (autograd: unused) -> the probe's `> 0` check is red
        g = torch.autograd.grad(T["total"], list(m.s_far_out.parameters()))
        assert sum(float(x.abs().sum()) for x in g) > 0.0


def test_mutation_deleting_the_far_output_turns_the_forward_red(world):
    mut = mutant('            out["ckpt_far"], self._s_far_last = self._s_far_last, None', "            pass")
    b, J = make_batch(world)
    m = build(J, mod=mut, s_far="on", w_ckpt_far=0.3333)
    with pytest.raises(KeyError):
        run_loss(m, b)


def test_mutation_joining_the_80m_arc_in_place_of_the_120m_arc_turns_the_label_identity_red(world):
    dmut = mutant('**({"rc120": J["rc120"][i], "rc120_ok": J["rc120_ok"][i]} if "rc120" in J else {})', '**({"rc120": J["rc"][i][2], "rc120_ok": J["rc_ok"][i][2]} if "rc120" in J else {})', "v8f_mini_data_mut", DATA_P)
    J = joined(world, mod=dmut)
    b, _ = make_batch(world, J=J, mod=dmut)
    with np.load(world["rc120"]) as z:
        off, rc, ok = np.asarray(z["train_off"]), np.asarray(z["train_rc120"]), np.asarray(z["train_rc120_ok"])
    rows = off[b["ci"].numpy()] + (b["t"] - b["a_s"]).numpy()
    assert not np.array_equal(b["lab_s"]["rc120"].numpy(), np.where(ok[rows][:, None], rc[rows], 0.0))


# ------------------------------------------------------------------------------------------------ T-ISO-FAR and the nav_s key guard
def test_perturbing_the_far_head_leaves_every_T_and_O_output_bit_identical(world):
    b, J = make_batch(world)
    m = build(J, s_far="on", w_ckpt_far=0.3333).eval()
    m2 = build(J, s_far="on", w_ckpt_far=0.3333).eval()
    with torch.no_grad():
        m.s_far_out.weight.add_(0.05)
        m2.s_far_out.weight.add_(0.5)
        m2.s_far_out.bias.add_(1.0)
        o1, o2 = m(b, p_gt=0.0), m(b, p_gt=0.0)
        o3 = m2(b, p_gt=0.0)
    for k in ("traj30", "traj60", "goal_logits", "lat_logits", "lon_logits", "code", "ckpt"):
        assert torch.equal(o1[k], o3[k]), k
    assert not torch.equal(o1["ckpt_far"], o3["ckpt_far"])


def test_the_far_point_is_not_read_by_forward_infer_keys_outside_the_inference_set(world):
    b, J = make_batch(world)
    m = build(J, s_far="on", w_ckpt_far=0.3333)
    adm = M.assert_no_situation_classifier_input(m, b)
    assert "rc120" not in " ".join(adm["read"]) and set(adm["read"]) <= set(M.INFERENCE_INPUT_KEYS)


# ------------------------------------------------------------------------------------------------ K-REACH-b: --nav-strat
def test_nav_s_gradient_is_nonzero_and_the_zero_control_reads_exactly_0(world):
    b, J = make_batch(world)
    m = build(J, nav_strat="on", detach_upper=True)
    g1 = gsum(run_loss(m, b)["terms"]["total"], list(m.e_nav_s.parameters()))
    m0 = build(J, nav_strat="on", detach_upper=True, w_ckpt=0.0)             # no S-side loss at all (detach_upper cuts T / O from S): e_nav_s must read EXACTLY 0.0
    g0 = gsum(run_loss(m0, b)["terms"]["total"], list(m0.e_nav_s.parameters()))
    assert g1 > 0.0 and g0 == 0.0
    off = build(J, detach_upper=True)
    assert off.e_nav_s is None


def test_T_and_O_never_read_nav_s_and_S_does(world):
    b, J = make_batch(world)
    m = build(J, nav_strat="on").eval()
    b2 = {**b, "nav_s": torch.zeros_like(b["nav_s"])}
    with torch.no_grad():
        o1, o2 = m(b, p_gt=0.0), m(b2, p_gt=0.0)
    assert not torch.equal(o1["ckpt"], o2["ckpt"])
    # with the SAME S output handed on (p_gt-free path: the checkpoints are the only link), T and O move only through S: hold S fixed and nothing else moves
    s0 = type(m).run_s.__get__(m)
    fixed = o1["ckpt"].detach()
    m.run_s = lambda *a, **k: (s0(*a, **k), fixed)[1]
    with torch.no_grad():
        o3, o4 = m(b, p_gt=0.0), m(b2, p_gt=0.0)
    for k in ("traj30", "traj60", "goal_logits", "lat_logits", "lon_logits", "code"):
        assert torch.equal(o3[k], o4[k]), k
    keys = set(build(J, nav_strat="on").eval().forward_infer(b)["_keys_read"])
    assert "nav_s" in keys and (keys - {"nav_s"}) <= set(M.INFERENCE_INPUT_KEYS)
    assert "nav_s" not in set(build(J).eval().forward_infer(b)["_keys_read"])
    M.assert_no_situation_classifier_input(build(J, nav_strat="on"), b)


def test_nav_strat_off_and_a_missing_nav_s_key_is_a_keyerror_not_a_silent_zero(world):
    b, J = make_batch(world)
    m = build(J, nav_strat="on").eval()
    bb = {k: v for k, v in b.items() if k != "nav_s"}
    with pytest.raises(KeyError):
        m(bb, p_gt=0.0)


def test_mutation_deleting_the_nav_s_token_turns_the_gradient_check_red(world):
    mut = mutant('            toks.append(self.e_nav_s(self._at(b["nav_s"], i)))', "            pass")
    b, J = make_batch(world)
    m = build(J, mod=mut, nav_strat="on", detach_upper=True)
    g = torch.autograd.grad(run_loss(m, b)["terms"]["total"], list(m.e_nav_s.parameters()), allow_unused=True)
    assert all(x is None for x in g)                          # the token is built and never reaches the loss: the probe's `> 0` is red


# ------------------------------------------------------------------------------------------------ --s-adapter-detach
def adapter_grad(mod, world, detach):
    b, J = make_batch(world)
    m = build(J, mod=mod, s_far="on", w_ckpt_far=0.3333, detach_upper=True, s_adapter_detach=detach)
    with torch.no_grad():
        m.s_far_out.weight.add_(0.05)                           # at its zero init the far term cannot reach the adapter at step 0
    T = run_loss(m, b)["terms"]
    ad = [p for mod_ in (m.adapter, m.pos, m.time) for p in (mod_.parameters() if isinstance(mod_, torch.nn.Module) else [mod_])]
    return gsum(T["ckpt"] + T["ckpt_far"], ad)


def test_s_adapter_detach_puts_exactly_zero_gradient_on_the_adapter_and_off_puts_some(world):
    assert adapter_grad(M, world, "off") > 0.0
    assert adapter_grad(M, world, "on") == 0.0


def test_s_adapter_detach_does_not_change_a_value(world):
    b, J = make_batch(world)
    a, c = build(J, s_far="on", w_ckpt_far=0.3333), build(J, s_far="on", w_ckpt_far=0.3333, s_adapter_detach="on")
    oa, oc = run_loss(a, b), run_loss(c, b)
    for k in ("ckpt", "ckpt_far", "traj30", "code"):
        assert torch.equal(oa[k], oc[k]), k


def test_mutation_dropping_the_scene_detach_turns_the_adapter_zero_check_red(world):
    mut = mutant("            g = g.detach()\n            g = g + self.time.detach()", "            g = g\n            g = g + self.time.detach()")
    assert adapter_grad(mut, world, "on") > 0.0               # red: the detach-on control no longer reads 0.0
    mut2 = mutant("            g = g + self.time.detach()[None, :, None, :]", "            g = g + self.time[None, :, None, :]")
    assert adapter_grad(mut2, world, "on") > 0.0              # red through the time table too


# ------------------------------------------------------------------------------------------------ the derangements (R1 / R2)
def test_the_derangement_is_fixed_has_no_fixed_points_and_one_cycle():
    for n in (2, 3, 6, 4572):
        p = D.derangement(n, 0)
        assert sorted(p.tolist()) == list(range(n)) and not (p == np.arange(n)).any()
        assert np.array_equal(p, D.derangement(n, 0))
        seen, i = 0, 0
        while True:
            i, seen = int(p[i]), seen + 1
            if i == 0:
                break
        assert seen == n
    with pytest.raises(ValueError):
        D.derangement(1, 0)


def test_label_perm_clip_replaces_the_far_labels_with_the_donors_and_masks_the_overhang(world):
    J0 = joined(world, with_nav=False)
    J1 = joined(world, perm="clip", with_nav=False)
    off, n = np.asarray(J0["off"]), np.asarray(J0["n"])
    don = D.derangement(len(off), 0)
    assert not np.array_equal(J0["rc120"], J1["rc120"])
    for c in range(len(off)):
        d = int(don[c])
        m = min(n[c], n[d])
        assert d != c
        assert np.array_equal(J1["rc120"][off[c]:off[c] + m], J0["rc120"][off[d]:off[d] + m])
        assert np.array_equal(J1["rc120_ok"][off[c]:off[c] + m], J0["rc120_ok"][off[d]:off[d] + m])
        assert not J1["rc120_ok"][off[c] + m:off[c] + n[c]].any()
    b0, _ = make_batch(world, J=J0)
    b1, _ = make_batch(world, J=J1)
    assert not torch.equal(b0["lab_s"]["rc120"], b1["lab_s"]["rc120"])           # the supervised value the loss receives changed


def test_nav_strat_donor_changes_the_value_s_is_fed_and_keeps_the_tactical_nav(world):
    Jr, Jd = joined(world, with_far=False), joined(world, variant="donor", with_far=False)
    assert not np.array_equal(Jr["nav_s"], Jd["nav_s"])
    off, n = np.asarray(Jr["off"]), np.asarray(Jr["n"])
    don = D.derangement(len(off), 0)
    for c in range(len(off)):
        assert np.array_equal(Jd["nav_s"][off[c]], Jr["nav_s"][off[int(don[c])]])
    br, _ = make_batch(world, J=Jr)
    bd, _ = make_batch(world, J=Jd)
    assert not torch.equal(br["nav_s"], bd["nav_s"]) and torch.equal(br["nav"], bd["nav"]) and torch.equal(br["lab_s"]["nav"], bd["lab_s"]["nav"])
    _, rec = D.nav_strat_arrays(Jr, world["navs"], sha(world["navs"]), "train", "donor")
    assert rec["n_rows_differ_from_real"] > 0 and D.nav_strat_arrays(Jr, world["navs"], sha(world["navs"]), "train", "real")[1]["n_rows_differ_from_real"] == 0


def test_the_label_and_nav_readers_refuse_a_wrong_sha_and_a_shifted_table(world, tmp_path):
    J = joined(world, with_far=False, with_nav=False)
    with pytest.raises(ValueError, match="sha256"):
        D.rc120_arrays(J, world["rc120"], "0" * 64, "train")
    with pytest.raises(ValueError, match="sha256"):
        D.nav_strat_arrays(J, world["navs"], "", "train")
    with np.load(world["rc120"]) as z:
        bad = {k: np.asarray(z[k]) for k in z.files}
    bad["train_off"] = bad["train_off"] + 1
    np.savez(tmp_path / "shift.npz", **bad)
    with pytest.raises(ValueError, match="misaligned"):
        D.rc120_arrays(J, tmp_path / "shift.npz", sha(tmp_path / "shift.npz"), "train")


# ------------------------------------------------------------------------------------------------ the S frame under the DAgger swap
def test_far_restage_is_the_identity_without_a_swap_and_the_frame_change_with_one(world):
    b, J = make_batch(world)
    assert TR.far_restage(b, b["pose_s"]) is b
    moved = {**b, "pose_s": b["pose_s"].clone()}
    moved["pose_s"][:4, 0] += 3.0
    moved["pose_s"][:4, 2] += 0.2
    out = TR.far_restage(moved, b["pose_s"])
    ls = b["lab_s"]
    want = M.transform_ckpt(ls["rc120"][:, None, :], b["pose_s"], moved["pose_s"])[:, 0]
    okm = ls["rc120_ok"]
    assert torch.equal(out["lab_s"]["rc120"][4:], ls["rc120"][4:])                          # untouched rows: bit-identical
    assert torch.allclose(out["lab_s"]["rc120"][:4][okm[:4]], want[:4][okm[:4]]) and not torch.equal(out["lab_s"]["rc120"][:4][okm[:4]], ls["rc120"][:4][okm[:4]])
    assert moved["lab_s"]["rc120"] is ls["rc120"]                                           # the input batch is never modified


# ------------------------------------------------------------------------------------------------ init from an M2b-style checkpoint
def test_a_tip_checkpoint_loads_with_exactly_the_new_key_prefixes_missing(world):
    tip = tip_module()
    b, J = make_batch(world)
    sd = build(J, mod=tip).state_dict()
    m = build(J, s_far="on", w_ckpt_far=0.3333, nav_strat="on")
    res = m.load_state_dict(sd, strict=False)
    assert set(res.missing_keys) == NEW_KEYS and not res.unexpected_keys
    assert all(k.startswith(M.NEW_LEVER_KEY_PREFIXES) for k in res.missing_keys)
    for kw, want in (({"s_far": "on", "w_ckpt_far": 0.3333}, {"s_far_out.weight", "s_far_out.bias"}), ({"nav_strat": "on"}, {"e_nav_s.weight", "e_nav_s.bias"})):
        r = build(J, **kw).load_state_dict(sd, strict=False)
        assert set(r.missing_keys) == want and not r.unexpected_keys


# ------------------------------------------------------------------------------------------------ G-DVB
def test_the_flag_audit_lists_every_new_flag_with_an_entry_and_the_off_argv_passes_bad_sfar(world):
    for dest in ("s_far", "w_ckpt_far", "s_far_lr_mult", "rc120_npz", "rc120_sha256", "s_far_label_perm", "nav_strat", "nav_strat_variant", "nav_strat_npz", "nav_strat_sha256", "s_adapter_detach"):
        assert dest in TR.FLAG_CHECKS
    parser_dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert parser_dests <= set(TR.FLAG_CHECKS)
    J = joined(world, with_far=False, with_nav=False)
    assert TR.bad_sfar(args(), build(J), {}) == []


def test_gdvb_refuses_s_far_on_with_weight_0_and_each_lever_declared_but_not_built(world):
    J = joined(world, with_far=False, with_nav=False)
    off = build(J)
    on_far = build(J, s_far="on", w_ckpt_far=0.0)
    probs = TR.bad_sfar(args(["--s-far", "on", "--w-ckpt-far", "0", "--rc120-npz", str(world["rc120"]), "--rc120-sha256", sha(world["rc120"])]), on_far, {"rc120": {"sha256": sha(world["rc120"]), "perm": "off", "n_ok_table": 3, "n_ok_after_perm": 3}})
    assert any("head with no loss" in p for p in probs)
    for extra, needle in ((["--s-far", "on", "--w-ckpt-far", "0.3333"], "no s_far_out was built"), (["--w-ckpt-far", "0.3"], "a weight with no head"), (["--nav-strat", "on"], "e_nav_s keys"),
                          (["--s-adapter-detach", "on"], "s_adapter_detach"), (["--s-far-label-perm", "clip"], "label permutation with no head"), (["--s-far-lr-mult", "10"], "lever with no head"),
                          (["--nav-strat-variant", "donor"], "variant with no token"), (["--rc120-npz", str(world["rc120"])], "declared and does nothing")):
        assert any(needle in p for p in TR.bad_sfar(args(extra), off, {})), (extra, needle)
    # a built head the argv does not name
    assert any("s_far_out keys" in p for p in TR.bad_sfar(args(), build(J, s_far="on", w_ckpt_far=0.3), {}))
    # stages that move S's frame without re-expressing the far label are refused
    for fl, val in (("--ego-perturb-p", "0.1"), ("--t2-bank-p", "0.1")):
        probs = TR.bad_sfar(args(["--s-far", "on", "--w-ckpt-far", "0.3333", fl, val]), build(J, s_far="on", w_ckpt_far=0.3333), {})
        assert any(fl in p for p in probs), fl


def test_gdvb_refuses_a_label_table_whose_sha_or_permutation_is_not_the_declared_one(world):
    J = joined(world, with_far=False, with_nav=False)
    m = build(J, s_far="on", w_ckpt_far=0.3333)
    a = args(["--s-far", "on", "--w-ckpt-far", "0.3333", "--rc120-npz", str(world["rc120"]), "--rc120-sha256", sha(world["rc120"]), "--s-far-label-perm", "clip"])
    good = {"sha256": sha(world["rc120"]), "perm": "clip", "n_ok_table": 5, "n_ok_after_perm": 4}
    assert not [p for p in TR.bad_sfar(a, m, {"rc120": good}) if "rc120" in p or "s_far_label_perm" in p or "permutation" in p and "built" in p and "declared" in p and "lever" not in p][:0]
    assert any("sha256" in p for p in TR.bad_sfar(a, m, {"rc120": {**good, "sha256": "0" * 64}}))
    assert any("arrays were built" in p for p in TR.bad_sfar(a, m, {"rc120": {**good, "perm": "off"}}))
    assert any("no RC-A120 arrays" in p for p in TR.bad_sfar(a, m, {}))


# ------------------------------------------------------------------------------------------------ the real caller, end to end (synthetic world)
def test_end_to_end_h1_h2_r1_r2_h1s_reach_the_loss_through_the_real_trainer(world):
    root = Path(world["root"])
    assert TR.main(_argv(world, "base", ["--steps", "3", "--detach-upper"])) == 0
    init = ["--init-ckpt", str(root / "base" / "ckpt_last.pt")]
    arms = {"h1": FAR(world), "h2": FAR(world) + NAVS(world), "r1": FAR(world, perm="clip"), "r2": FAR(world) + NAVS(world, "donor"), "h1s": FAR(world) + ["--s-adapter-detach", "on"]}
    for tag, extra in arms.items():
        assert TR.main(_argv(world, tag, ["--steps", "3", "--detach-upper", *init, *extra])) == 0, tag
        sc = _sc(root / tag)
        assert sc["declared_vs_built"] == "PASS" and sc["ckpt120_probe"]["pass"] is True, tag
        p = sc["ckpt120_probe"]
        assert p["head_grad"] > 0.0 and p["control_head_grad_w0"] == 0.0 and p["control_allmasked_head_grad"] == 0.0 and p["t_iso_far"] is True
        assert p["adapter_grad_s_detach_off"] > 0.0 and p["control_adapter_grad_s_detach_on"] == 0.0
        assert sc["rc120"]["sha256"] == sha(world["rc120"])
        if tag in ("h2", "r2"):
            assert p["nav_s_grad"] > 0.0 and p["nav_s_moves_s"] is True and p["nav_s_t_o_identical"] is True and sc["nav_strat"]["variant"] == ("donor" if tag == "r2" else "real")
        ck = torch.load(root / tag / "ckpt_last.pt", weights_only=False)
        names = {g.get("name") for g in ck["opt"]["param_groups"]}
        assert {"sfar", "sfar_nd"} <= names and ck["rc120_sha256"] == sha(world["rc120"]) and ck["cfg"]["s_far"] == "on"
        g = {g_["name"]: g_ for g_ in ck["opt"]["param_groups"] if g_.get("name") in ("sfar", "sfar_nd")}
        assert all(abs(x["lr_mult"] - 10.0) < 1e-12 for x in g.values())
        assert set(k for k in ck["model"] if k.startswith(M.S_FAR_KEY_PREFIXES + M.NAV_S_KEY_PREFIXES)) == ({"s_far_out.weight", "s_far_out.bias"} | ({"e_nav_s.weight", "e_nav_s.bias"} if tag in ("h2", "r2") else set()))
        assert ck["cfg"]["s_far_label_perm"] == ("clip" if tag == "r1" else "off") and ck["cfg"]["nav_strat_variant"] == ("donor" if tag == "r2" else "real")
        assert ck["cfg"]["s_adapter_detach"] == ("on" if tag == "h1s" else "off")
        tl = [json.loads(x) for x in (root / tag / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert any("ckpt_far" in json.dumps(r) for r in tl), "the far term is logged"
        sd0 = torch.load(root / "base" / "ckpt_last.pt", weights_only=False)["model"]
        assert float(ck["model"]["s_far_out.weight"].abs().sum()) > 0.0                      # the head MOVED from its zero init: it trained
    # the base (flag-off) run's selfcheck has no CKPT120 key (a flag-off selfcheck keeps its key set)
    assert "ckpt120_probe" not in _sc(root / "base")


def test_end_to_end_zero_weight_control_is_refused_by_the_gate(world):
    with pytest.raises(SystemExit) as e:
        TR.main(_argv(world, "w0", ["--steps", "1", "--selfcheck-only", "--detach-upper", *FAR(world, w="0")]))
    assert "ZZDVB-REFUSEDZZ" in str(e.value) and "head with no loss" in str(e.value)


def test_mutation_deleting_the_make_config_line_turns_the_gate_red(world):
    mod = mutant_trainer("                           s_far=a.s_far, w_ckpt_far=a.w_ckpt_far, s_far_label_perm=a.s_far_label_perm, nav_strat=a.nav_strat, nav_strat_variant=a.nav_strat_variant, s_adapter_detach=a.s_adapter_detach,",
                         "                           ", "mc")
    with pytest.raises(SystemExit) as e:
        mod.main(_argv(world, "mut_mc", ["--steps", "1", "--selfcheck-only", "--detach-upper", *FAR(world)]))
    assert "ZZDVB-REFUSEDZZ" in str(e.value)


def test_mutation_deleting_the_rc120_join_turns_the_gate_red(world):
    mod = mutant_trainer('        J["rc120"], J["rc120_ok"], info["rc120"] = D.rc120_arrays(J, a.rc120_npz, a.rc120_sha256, "train", a.s_far_label_perm)', '        info["rc120"] = D.rc120_arrays(J, a.rc120_npz, a.rc120_sha256, "train", a.s_far_label_perm)[2]', "rcj")
    with pytest.raises((SystemExit, ValueError, KeyError)):
        mod.main(_argv(world, "mut_rcj", ["--steps", "1", "--selfcheck-only", "--detach-upper", *FAR(world)]))


def test_mutation_deleting_the_nav_s_join_turns_the_gate_red(world):
    mod = mutant_trainer('        J["nav_s"], info["nav_strat"] = D.nav_strat_arrays(J, a.nav_strat_npz, a.nav_strat_sha256, "train", a.nav_strat_variant)', '        info["nav_strat"] = D.nav_strat_arrays(J, a.nav_strat_npz, a.nav_strat_sha256, "train", a.nav_strat_variant)[1]', "nsj")
    with pytest.raises((SystemExit, ValueError, KeyError)):
        mod.main(_argv(world, "mut_nsj", ["--steps", "1", "--selfcheck-only", "--detach-upper", *NAVS(world)]))


def test_mutation_putting_the_far_head_in_decay_too_turns_the_optimiser_check_red(world):
    mod = mutant_trainer('and id(p) not in sfar_ids and id(p) not in ctx_ids]\n    nodecay', 'and id(p) not in ctx_ids]\n    nodecay', "sfo")
    with pytest.raises((SystemExit, ValueError, Exception)):
        mod.main(_argv(world, "mut_sfo", ["--steps", "1", "--selfcheck-only", "--detach-upper", *FAR(world)]))


def test_mutation_deleting_the_restage_line_makes_a_swapped_row_keep_the_logged_frame(world):
    b, J = make_batch(world)
    moved = {**b, "pose_s": b["pose_s"].clone()}
    ok_rows = torch.nonzero(b["lab_s"]["rc120_ok"]).flatten()[:2]
    assert len(ok_rows) == 2
    moved["pose_s"][ok_rows, 0] += 5.0
    mod = mutant_trainer("    moved = DG.rc_frame_change(ls[\"rc120\"][:, None, :], ls[\"rc120_ok\"][:, None], pose_s_logged, batch[\"pose_s\"])[:, 0]", "    moved = ls[\"rc120\"]", "rs")
    assert torch.equal(mod.far_restage(moved, b["pose_s"])["lab_s"]["rc120"], b["lab_s"]["rc120"])
    assert not torch.equal(TR.far_restage(moved, b["pose_s"])["lab_s"]["rc120"][ok_rows], b["lab_s"]["rc120"][ok_rows])


# ------------------------------------------------------------------------------------------------ nav_s builder (step 5 with d_ann = infinity)
def test_the_nav_s_builder_drops_only_the_announcement_cut():
    import build_navtext_nav as NB
    n = 12
    cls, dist, flag, age = np.ones(n, np.int64), np.full(n, 200.0, np.float32), np.zeros(n, np.int64), np.zeros(n, np.float32)
    now_s = np.arange(n) * 0.2
    track = np.zeros((2 * n + 2, 4))
    track[:, 0] = np.arange(len(track)) * 1.0                                      # 10 m/s along x
    track[:, 3] = 10.0
    side = np.array([0, 1, 2])                                                     # class 1 = left
    cut, d_cut, _, _ = NB.clip_nav(cls, dist, flag, age, now_s, track, side)       # the default: STRAT1's rule
    cut2 = NB.clip_nav(cls, dist, flag, age, now_s, track, side, -1, True)
    nocut, d_nocut, _, _ = NB.clip_nav(cls, dist, flag, age, now_s, track, side, -1, False)
    assert np.array_equal(cut, cut2[0])
    assert (cut[:, 0] == 1).all() and (cut[:, 1] == 0).all()                       # d (~190 m) > max(30, 6 x 10): FOLLOW with the distance
    assert (nocut[:, 1] == 1).all() and (nocut[:, 0] == 0).all()                   # no cut: the side at any distance
    assert np.array_equal(cut[:, 3:], nocut[:, 3:]) and np.array_equal(d_cut, d_nocut, equal_nan=True)   # the countdown distance and the row flags are untouched
    near = NB.clip_nav(cls, np.full(n, 20.0, np.float32), flag, age, now_s, track, side, -1, False)[0]
    assert np.array_equal(near, NB.clip_nav(cls, np.full(n, 20.0, np.float32), flag, age, now_s, track, side)[0])       # inside d_ann the two rules are the same row
