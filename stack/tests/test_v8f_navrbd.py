"""PREREG_V8F_NAVRBD: the width lever ``--nav-rb-feat state|exitdist`` of the roundabout channel (model, loader, trainer). CPU only.
Z0 the ``state`` build == the e71782a3 model (snapshot: state dict, init tensors, loss terms, gradients, outputs, built levers); Z1 the exitdist build at its zero init == the state / flag-off build (outputs, loss terms,
RNG stream); Z2 the gradient PER COLUMN GROUP (c0, c1-6, c7-12, c13, c14) > 0 where the channel has a value, EXACTLY 0.0 on the zero channel and on every group whose columns are 0; the loader refuses every layout
violation; the shuffle variant is a clip-level derangement; the trainer end to end (G-DVB, G-LIVE, checkpoint round trip); each mutation turns a NAMED check RED."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
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
import tanitad  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

assert Path(tanitad.__file__).resolve().is_relative_to(REPO.resolve()), tanitad.__file__          # the study tree's tanitad, never an editable install elsewhere

TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
SNAP = Path(__file__).with_name("v8f_mini_tip_snapshot_navrbd.txt")
SNAP_SHA = "4620671a456aeecd"                      # first 16 hex of the sha256 of the e71782a3 stack/tanitad/models/v8f_mini.py (the file the N1R arms trained from)
MODEL_P, TRAIN_P, DATA_P = (REPO / "stack" / "tanitad" / "models" / "v8f_mini.py", REPO / "stack" / "scripts" / "v8f_mini_train.py", REPO / "stack" / "tanitad" / "data" / "v8f_mini_data.py")
RUN = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
GROUPS = tuple(D.NAV_RB_GROUPS)                    # c0, c1_6, c7_12, c13, c14


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rbd_npz(path, J, seed=0, p=0.6, split="train", width=15, mutate=None):
    """A VALID 15-column channel aligned to the join: c0 on with prob p; on rows: exit_cur / exit_last one-hot (prob .7 each), dist in (0, 3] with c14 = 1 (prob .8), everything 0 where c0 is 0."""
    rng = np.random.default_rng(seed)
    N = int(J["off"][-1] + J["n"][-1])
    rb = np.zeros((N, 15), np.float32)
    on = rng.uniform(size=N) < p
    rb[:, 0] = on
    ec, el = rng.integers(0, 6, N), rng.integers(0, 6, N)
    hc, hl, hd = rng.uniform(size=N) < .7, rng.uniform(size=N) < .7, rng.uniform(size=N) < .8
    rb[np.arange(N)[on & hc], 1 + ec[on & hc]] = 1.0
    rb[np.arange(N)[on & hl], 7 + el[on & hl]] = 1.0
    rb[on & hd, 13] = rng.uniform(0.05, 3.0, N)[on & hd].astype(np.float32)
    rb[on & hd, 14] = 1.0
    if width == 1:
        rb = rb[:, :1].copy()
    if mutate:
        mutate(rb)
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
    root = tmp_path_factory.mktemp("navrbdworld")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    return {"root": root, "rbd": write_rbd_npz(root / "rbd.npz", J), "rb1": write_rbd_npz(root / "rb1.npz", J, width=1), "nav": write_nav_npz(root / "nav.npz", J), **w}


def make_batch(world, feat="exitdist", n=16, seed=0, hmax=10, npz=None):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    f = npz or (world["rbd"] if feat == "exitdist" else world["rb1"])
    J["nav_rb"], _ = D.nav_rb_arrays(J, f, sha(f), "train", "real", feat)
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
    return sys.modules.get("v8f_mini_snapshot_navrbd") or mod_from_text(SNAP.read_text(encoding="utf-8"), "v8f_mini_snapshot_navrbd", SNAP)


def mutant(path, old, new, name):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (old[:60], text.count(old))
    return mod_from_text(text.replace(old, new), name, path)


def check_equal(mt, mn, b, extra_keys=()):
    """mt (the reference) and mn (the build under test) agree on every state-dict key (bar ``extra_keys`` of mn), init tensor, loss term, gradient and forward_infer output. [] = identical."""
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


def group_grads(m, b):
    """The L1 gradient of the total loss on each column group of e_nav_rb.weight."""
    m.zero_grad(set_to_none=True)
    run_loss(m, b)["terms"]["total"].backward()
    g = m.e_nav_rb.weight.grad
    if g is None:
        return {k: 0.0 for k in GROUPS}
    return {k: float(g[:, lo:hi].abs().sum()) for k, (lo, hi) in D.NAV_RB_GROUPS.items()}


def only_group(b, grp):
    """The batch with only ONE column group's values kept (everything else 0.0)."""
    lo, hi = D.NAV_RB_GROUPS[grp]
    z = torch.zeros_like(b["nav_rb"])
    z[..., lo:hi] = TR.navrb_planted(b["nav_rb"])[..., lo:hi]
    return {**b, "nav_rb": z}


# ------------------------------------------------------------------------------------------------ pins, flags, defaults
def test_the_snapshot_is_the_e71782a3_model_and_tanitad_is_the_study_tree():
    assert sha(SNAP)[:16] == SNAP_SHA
    assert Path(M.__file__).resolve() == MODEL_P.resolve()


def test_flag_default_is_state_has_a_flag_check_and_a_wm_only_default():
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.nav_rb_feat == "state" and (a.nav_rb, a.nav_rb_variant) == ("off", "real")
    assert TR.FLAG_CHECKS["nav_rb_feat"] == "lever:nav_rb_feat" and "nav_rb_feat" in TR.WM_ONLY_DEFAULT_DESTS
    assert M.NAV_RB_FEATS == ("state", "exitdist") and M.NAV_RB_DIM == {"state": 1, "exitdist": 15} and M.V8FMiniConfig().nav_rb_feat == "state"
    assert D.NAV_RB_VARIANTS == ("real", "zero", "shuffle")
    with pytest.raises(SystemExit):
        TR.build_parser().parse_args(["--out", "x", "--nav-rb-feat", "wide"])
    with pytest.raises(ValueError):
        M.V8FMiniConfig(nav_rb_feat="wide")


# ------------------------------------------------------------------------------------------------ Z0: state == e71782a3
def test_z0_state_build_is_the_e71782a3_model_bit_for_bit(world):
    b, J = make_batch(world, "state")
    assert check_equal(build(J, mod=base_module()), build(J), b) == []                                        # flag off
    m_ref, m_new = build(J, mod=base_module(), nav_rb="on"), build(J, nav_rb="on")                            # channel on, feat = the default (state)
    assert check_equal(m_ref, m_new, b) == [] and tuple(m_new.e_nav_rb.weight.shape) == (32, 1)
    assert m_new.built_levers() == m_ref.built_levers() and "nav_rb_feat" not in m_new.built_levers()
    for m in (m_ref, m_new):
        with torch.no_grad():
            m.e_nav_rb.weight.copy_(torch.randn(m.e_nav_rb.weight.shape, generator=torch.Generator().manual_seed(5)))
    assert check_equal(m_ref, m_new, b) == []                                                                  # a non-zero weight: the term itself is the same


def test_z0_mutation_a_state_build_that_takes_the_wide_path_turns_red(world):
    mut = mutant(MODEL_P, "NAV_RB_DIM = {\"state\": 1, \"exitdist\": 15}", "NAV_RB_DIM = {\"state\": 15, \"exitdist\": 15}", "v8f_mini_mut_navrbd_z0")
    b, J = make_batch(world, "state")
    with pytest.raises(Exception):
        probs = check_equal(build(J, mod=base_module(), nav_rb="on"), build(J, mod=mut, nav_rb="on"), b)
        assert probs == []


# ------------------------------------------------------------------------------------------------ Z1: exitdist at the zero init == state / off, RNG untouched
def test_z1_exitdist_at_zero_init_is_the_off_and_the_state_model_and_leaves_the_rng_alone(world):
    b15, J = make_batch(world, "exitdist")
    b1, _ = make_batch(world, "state")
    m_off = build(J); s_off = torch.get_rng_state()
    m_st = build(J, nav_rb="on"); s_st = torch.get_rng_state()
    m_ex = build(J, nav_rb="on", nav_rb_feat="exitdist"); s_ex = torch.get_rng_state()
    assert torch.equal(s_off, s_ex) and torch.equal(s_st, s_ex)
    w = m_ex.e_nav_rb.weight
    assert tuple(w.shape) == (32, 15) and float(w.abs().sum()) == 0.0 and m_ex.e_nav_rb.bias is None
    assert check_equal(m_off, m_ex, b15, extra_keys=("e_nav_rb.weight",)) == []                              # outputs, loss terms, gradients of every other parameter
    assert [k for k in m_st.state_dict()] == [k for k in m_ex.state_dict()]
    ot, oe = run_loss(m_st, b1), run_loss(m_ex, b15)
    assert all(torch.equal(ot["terms"][k], oe["terms"][k]) for k in ot["terms"])                              # the state build on its batch == the exitdist build on its batch
    assert m_ex.built_levers()["nav_rb_feat"] == "exitdist" and "nav_rb_feat" not in m_st.built_levers() and "nav_rb_feat" not in m_off.built_levers()


def test_z1_mutation_no_fork_rng_moves_the_global_stream(world):
    mut = mutant(MODEL_P, "with torch.random.fork_rng(devices=[]):\n                self.e_nav_rb = nn.Linear(", "if True:\n                self.e_nav_rb = nn.Linear(", "v8f_mini_mut_navrbd_z1")
    _, J = make_batch(world, "exitdist")
    build(J, mod=mut, nav_rb="on", nav_rb_feat="exitdist"); s_mut = torch.get_rng_state()
    build(J, nav_rb="on", nav_rb_feat="exitdist"); s_ok = torch.get_rng_state()
    assert not torch.equal(s_mut, s_ok)


# ------------------------------------------------------------------------------------------------ Z2: gradient per column group
def check_z2(mod, world):
    """The Z2 read on a module's build: every group > 0 where the channel carries a value, EXACTLY 0.0 on the zero channel, and a group whose columns are 0 reads exactly 0.0 (the gradient is column-local)."""
    b, J = make_batch(world, "exitdist")
    m = build(J, mod=mod, nav_rb="on", nav_rb_feat="exitdist")
    planted = {**b, "nav_rb": TR.navrb_planted(b["nav_rb"])}
    assert all(v > 0.0 for v in group_grads(m, planted).values()), group_grads(m, planted)
    assert all(v == 0.0 for v in group_grads(m, {**b, "nav_rb": torch.zeros_like(b["nav_rb"])}).values())
    for g in GROUPS:                                                                                         # only ONE group non-zero: the others read EXACTLY 0.0
        gg = group_grads(m, only_group(b, g))
        assert gg[g] > 0.0 and all(v == 0.0 for k, v in gg.items() if k != g), (g, gg)
    return True


def test_z2_gradient_per_column_group_and_exact_zero_controls(world):
    assert check_z2(M, world)
    b, J = make_batch(world, "exitdist")
    m = build(J, nav_rb="on", nav_rb_feat="exitdist")
    real = group_grads(m, b)                                                                                  # the REAL batch: its own counts say which groups can read > 0
    cnt = D.nav_rb_group_n_on(b["nav_rb"].reshape(-1, 15).numpy())
    assert set(cnt) == set(GROUPS) and cnt["c0"] > 0
    assert real["c0"] > 0.0
    for g in GROUPS:
        if cnt[g] == 0:
            assert real[g] == 0.0
    with torch.no_grad():
        m.e_nav_rb.weight.fill_(0.5)
    m.eval()
    with torch.no_grad():
        o1, o0 = m.forward_infer(b), m.forward_infer({**b, "nav_rb": torch.zeros_like(b["nav_rb"])})
    assert any(not torch.equal(o1[k], o0[k]) for k in o1 if torch.is_tensor(o1[k])) and "nav_rb" in o1["_keys_read"]          # a trained weight moves the plan


def test_mutation_deleting_the_forward_term_turns_z2_red(world):
    mut = mutant(MODEL_P, '            t = t + self.e_nav_rb(self._at(b["nav_rb"], i).to(self.e_nav_rb.weight.dtype))\n', "            pass\n", "v8f_mini_mut_navrbd_c")
    with pytest.raises(AssertionError):
        check_z2(mut, world)


def test_a_built_wide_channel_with_a_one_column_batch_is_refused_and_a_missing_key_is_a_keyerror(world):
    b1, J = make_batch(world, "state")
    m = build(J, nav_rb="on", nav_rb_feat="exitdist")
    m.eval()
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            m.forward_infer(b1)                                                                               # 1 column into a 15-column weight: never a silent broadcast
    nob = {k: v for k, v in b1.items() if k != "nav_rb"}
    with pytest.raises(KeyError):
        with torch.no_grad():
            m.forward_infer(nob)


def test_model_path_and_loader_import_no_situation_module_and_read_no_label():
    for path, fns in ((DATA_P, ("nav_rb_arrays", "nav_rb_shuffle", "nav_rb_derangement", "nav_rb_check_values", "nav_rb_group_n_on")), (MODEL_P, ("_nav_tok",))):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in fns or (isinstance(n, ast.FunctionDef) and n.name in fns)):
            seg = (ast.get_source_segment(src, fn) or "")
            if isinstance(fn, ast.FunctionDef):
                for w in ("situation", "sitclf", "lat_cls", "v9", "ego_state"):
                    assert w not in seg.replace(fn.name, ""), (fn.name, w)


# ------------------------------------------------------------------------------------------------ the loader: layout refusals, shuffle derangement, window item
def load15(world, tmp_path, mutate, variant="real", feat="exitdist", name="x.npz"):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    f = write_rbd_npz(tmp_path / name, J, mutate=mutate)
    return D.nav_rb_arrays(J, f, sha(f), "train", variant, feat)


def _rows(rb, pred):
    return np.nonzero(pred(rb))[0]


def test_loader_accepts_a_valid_table_and_records_width_groups_and_counts(world):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    rb, rec = D.nav_rb_arrays(J, world["rbd"], sha(world["rbd"]), "train", "real", "exitdist")
    assert rb.shape == (int(J["off"][-1] + J["n"][-1]), 15) and rb.dtype == np.float32
    assert (rec["feat"], rec["width"], rec["variant"]) == ("exitdist", 15, "real") and rec["n_on"] == int(rb[:, 0].sum()) > 0
    assert set(rec["group_n_on"]) == set(GROUPS) and all(v > 0 for v in rec["group_n_on"].values())
    z, rz = D.nav_rb_arrays(J, world["rbd"], sha(world["rbd"]), "train", "zero", "exitdist")
    assert float(np.abs(z).sum()) == 0.0 and rz["n_on"] == 0 and rz["n_on_real"] == rec["n_on"] and all(v == 0 for v in rz["group_n_on"].values())
    s1, r1 = D.nav_rb_arrays(J, world["rb1"], sha(world["rb1"]), "train", "real", "state")
    assert s1.shape[1] == 1 and r1["width"] == 1 and set(r1["group_n_on"]) == {"c0"}


@pytest.mark.parametrize("what,mut", [
    ("c0_not_binary", lambda rb: rb.__setitem__((slice(None), 0), rb[:, 0] * 0.5)),
    ("two_ones_in_exit_cur", lambda rb: (rb.__setitem__((np.nonzero(rb[:, 0])[0][:3], 1), 1.0), rb.__setitem__((np.nonzero(rb[:, 0])[0][:3], 2), 1.0))),
    ("exit_last_value_2", lambda rb: rb.__setitem__((np.nonzero(rb[:, 0])[0][:2], 7), 2.0)),
    ("c13_negative", lambda rb: rb.__setitem__((np.nonzero(rb[:, 0])[0][:1], 13), -0.1)),
    ("c13_over_3", lambda rb: rb.__setitem__((np.nonzero(rb[:, 0])[0][:1], 13), 3.5)),
    ("c14_not_binary", lambda rb: rb.__setitem__((np.nonzero(rb[:, 0])[0][:1], 14), 0.5)),
    ("column_on_where_c0_is_0", lambda rb: rb.__setitem__((np.nonzero(rb[:, 0] == 0)[0][:1], 3), 1.0)),
    ("dist_without_dist_valid", lambda rb: (rb.__setitem__((np.nonzero(rb[:, 0])[0][:1], 13), 1.0), rb.__setitem__((np.nonzero(rb[:, 0])[0][:1], 14), 0.0))),
    ("nan", lambda rb: rb.__setitem__((np.nonzero(rb[:, 0])[0][:1], 13), np.nan)),
])
def test_loader_refuses_every_layout_violation(world, tmp_path, what, mut):
    with pytest.raises(ValueError):
        load15(world, tmp_path, mut)


def test_loader_refuses_wrong_width_sha_alignment_and_a_state_table_with_a_wide_feat(world, tmp_path):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    with pytest.raises(ValueError):
        D.nav_rb_arrays(J, world["rb1"], sha(world["rb1"]), "train", "real", "exitdist")                       # 1 column, feat exitdist
    with pytest.raises(ValueError):
        D.nav_rb_arrays(J, world["rbd"], sha(world["rbd"]), "train", "real", "state")                          # 15 columns, feat state
    with pytest.raises(ValueError):
        D.nav_rb_arrays(J, world["rbd"], "0" * 64, "train", "real", "exitdist")
    with pytest.raises(ValueError):
        D.nav_rb_arrays(J, world["rbd"], sha(world["rbd"]), "train", "real", "wide")
    with pytest.raises(ValueError):
        D.nav_rb_arrays(J, world["rbd"], sha(world["rbd"]), "train", "rotate", "exitdist")
    bad = dict(np.load(world["rbd"], allow_pickle=False))
    bad["train_off"] = bad["train_off"] + 1
    np.savez(tmp_path / "shift.npz", **bad)
    with pytest.raises(ValueError):
        D.nav_rb_arrays(J, tmp_path / "shift.npz", sha(tmp_path / "shift.npz"), "train", "real", "exitdist")


@pytest.mark.parametrize("K", [2, 3, 6, 17, 101])
@pytest.mark.parametrize("seed", [0, 1, 20261118])
def test_shuffle_derangement_has_no_fixed_point_and_is_a_permutation(K, seed):
    p = D.nav_rb_derangement(K, seed)
    assert sorted(p.tolist()) == list(range(K)) and not (p == np.arange(K)).any()
    assert np.array_equal(p, D.nav_rb_derangement(K, seed))                                                    # seeded: reproducible
    with pytest.raises(ValueError):
        D.nav_rb_derangement(1, seed)


def shuffle_check(mod, world):
    """The shuffle variant on a module: every clip reads ANOTHER clip's rows, resampled to its own length; lengths preserved; the donor map is a derangement; the rows differ from the real ones."""
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    rb, rec = mod.nav_rb_arrays(J, world["rbd"], sha(world["rbd"]), "train", "shuffle", "exitdist")
    real, _ = mod.nav_rb_arrays(J, world["rbd"], sha(world["rbd"]), "train", "real", "exitdist")
    off, n = np.asarray(J["off"]), np.asarray(J["n"])
    donor = D.nav_rb_derangement(len(off))
    assert rb.shape == real.shape and rec["variant"] == "shuffle" and rec["n_on_real"] == int(real[:, 0].sum())
    assert hashlib.sha256(donor.astype(np.int64).tobytes()).hexdigest() == rec["donor_map_sha256"]
    for c in range(len(off)):
        d = int(donor[c])
        assert d != c
        idx = np.minimum((np.arange(int(n[c])) * int(n[d])) // int(n[c]), int(n[d]) - 1)
        np.testing.assert_array_equal(rb[off[c]: off[c] + n[c]], real[off[d] + idx])
    assert not np.array_equal(rb, real)
    assert rec["digest"] != _["digest"]
    return True


def test_shuffle_variant_is_a_clip_level_derangement_resampled_to_the_receiving_clip(world):
    assert shuffle_check(D, world)


def test_mutation_shuffle_replaced_by_identity_turns_the_shuffle_check_red(world):
    text = DATA_P.read_text(encoding="utf-8")
    old = 'rb, donor = nav_rb_shuffle(rb, J["off"], J["n"])'
    assert text.count(old) == 1
    mut = mod_from_text(text.replace(old, 'donor = np.arange(len(J["off"]))'), "v8f_mini_data_mut_navrbd_shuf", DATA_P)
    with pytest.raises(AssertionError):
        shuffle_check(mut, world)


def test_the_window_item_carries_the_15_column_rows_of_its_frames(world):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    N = int(J["off"][-1] + J["n"][-1])
    J["nav_rb"] = np.tile(np.ascontiguousarray(J["nav"][:, 3:4]).astype(np.float32), (1, 15))                   # a row-coded stand-in: the item must take the SAME rows as the nav's frames
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=10)
    for w in wins[:: max(1, len(wins) // 7)]:
        it = ds.item(w[0], w[1], 1, 3)
        assert it["nav_rb"].shape == (10, 15)
        np.testing.assert_array_equal(it["nav_rb"][:, 0], it["nav"][:, 3])
    b, _ = make_batch(world, "exitdist")
    assert tuple(b["nav_rb"].shape[-1:]) == (15,) and b["nav_rb"].dtype == torch.float32
    assert N > 0


def test_mutation_dropping_the_item_rows_turns_the_batch_check_red(world, monkeypatch):
    real_item = D.V8FMiniWindows.item

    def item(self, *a, **k):
        it = real_item(self, *a, **k)
        it.pop("nav_rb", None)
        return it
    monkeypatch.setattr(D.V8FMiniWindows, "item", item)
    b, J = make_batch(world, "exitdist")
    assert "nav_rb" not in b                                                                                    # the batch check (nav_rb present, 15 wide) is RED
    m = build(J, nav_rb="on", nav_rb_feat="exitdist")
    with pytest.raises(KeyError):
        m.eval()
        with torch.no_grad():
            m.forward_infer(b)


# ------------------------------------------------------------------------------------------------ the ckpt round trip (G-CKPT / G-EVAL on the model)
def test_checkpoint_round_trip_of_the_15_column_weight_and_a_cfg_that_drops_the_feat_is_refused(world, tmp_path):
    _, J = make_batch(world, "exitdist")
    m = build(J, nav_rb="on", nav_rb_feat="exitdist")
    with torch.no_grad():
        m.e_nav_rb.weight.copy_(torch.randn(m.e_nav_rb.weight.shape, generator=torch.Generator().manual_seed(2)))
    torch.save({"model": m.state_dict(), "cfg": m.cfg.__dict__}, tmp_path / "ck.pt")
    ck = torch.load(tmp_path / "ck.pt", weights_only=False)
    assert tuple(ck["model"]["e_nav_rb.weight"].shape) == (32, 15) and ck["cfg"]["nav_rb_feat"] == "exitdist"
    m2 = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior.synthetic(0), *M.cons_stats(J))
    m2.load_state_dict(ck["model"])                                                                              # strict
    assert torch.equal(m2.e_nav_rb.weight, m.e_nav_rb.weight) and m2.built_levers()["nav_rb_feat"] == "exitdist"
    with pytest.raises(RuntimeError):
        M.V8FMini(M.V8FMiniConfig(**{**ck["cfg"], "nav_rb_feat": "state"}), M.P10Prior.synthetic(0), *M.cons_stats(J)).load_state_dict(ck["model"])      # G-GEN: the dropped feat is REFUSED
    old = {k: v for k, v in ck["cfg"].items() if k != "nav_rb_feat"}
    assert M.V8FMiniConfig(**old).nav_rb_feat == "state"                                                         # a checkpoint written before the lever loads as state


# ------------------------------------------------------------------------------------------------ the trainer, end to end
def N1(world):
    return ["--nav-source", "text", "--nav-text-npz", str(world["nav"]), "--nav-text-sha256", sha(world["nav"]), "--ctx-heads", "off", "--w-ctx", "0", "--ctx-labels", "", "--ctx-labels-sha256", "",
            "--ctx-lr-mult", "1", "--ctx-detach", "off"]


def RBD(world, variant="real", mult="10", feat="exitdist", npz="rbd"):
    return ["--nav-rb", "on", "--nav-rb-npz", str(world[npz]), "--nav-rb-sha256", sha(world[npz]), "--nav-rb-variant", variant, "--nav-rb-lr-mult", mult, *(["--nav-rb-feat", feat] if feat else [])]


def _argv(world, tag, extra):
    return ["--out", str(Path(world["root"]) / tag), *RUN, *extra]


def trainer_module(old, new, tag):
    text = TRAIN_P.read_text(encoding="utf-8")
    assert text.count(old) == 1, (old[:60], text.count(old))
    p = TRAIN_P.with_name(f"_mut_navrbd_{tag}.py")
    p.write_text(text.replace(old, new), encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location(f"_mut_navrbd_{tag}", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        p.unlink()


def check_e2e(mod, world, tag, steps="3"):
    """A 3-step exitdist arm through ``mod.main``: G-DVB PASS, the probe's per-group record, the lever / info record, the 15-column checkpoint with the feat in its cfg (read back through the evaluators' build path)."""
    out = Path(world["root"]) / tag
    assert mod.main(_argv(world, tag, ["--steps", steps, *N1(world), *RBD(world)])) == 0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    p = sc["navrb_probe"]
    assert sc["declared_vs_built"] == "PASS" and p["pass"] and p["step0_identity"] is True and p["feat"] == "exitdist" and p["width"] == 15
    assert p["e_nav_rb_grad"] > 0.0 and p["control_e_nav_rb_grad_zero_channel"] == 0.0 and p["init_weight_abs_sum"] == 0.0
    assert set(p["group_grads_planted"]) == set(GROUPS) == set(p["group_control_zero_channel"]) == set(p["group_n_on"]) == set(p["group_grads"])
    assert all(v > 0.0 for v in p["group_grads_planted"].values()) and all(v == 0.0 for v in p["group_control_zero_channel"].values())
    assert p["group_n_on"]["c0"] == p["n_on_batch"] and all(g in p["group_unread_in_batch"] or p["group_n_on"][g] == 0 or p["group_grads"][g] > 0.0 for g in GROUPS)
    assert sc["levers"]["nav_rb"] == "on" and sc["levers"]["nav_rb_feat"] == "exitdist" and sc["live"]["groups_grad"]["NAVRB"] > 0.0
    info = sc["info"]["nav_rb"]
    assert info["feat"] == "exitdist" and info["width"] == 15 and info["sha256"] == sha(world["rbd"]) and all(v > 0 for v in info["group_n_on"].values())
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert tuple(ck["model"]["e_nav_rb.weight"].shape) == (32, 15) and float(ck["model"]["e_nav_rb.weight"].abs().sum()) > 0.0
    assert ck["cfg"].get("nav_rb_feat") == "exitdist" and ck["cfg"]["nav_rb"] == "on" and ck["nav_rb_feat"] == "exitdist"
    mm = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior.synthetic(0), *M.cons_stats(D.load_join(Path(world["root"]) / "join_train.npz")))
    mm.load_state_dict(ck["model"])                                                                               # the evaluators' build path, strict: the cfg carries the feat
    g = [x for x in ck["opt"]["param_groups"] if x.get("name") == "navrb"]
    assert len(g) == 1 and g[0]["lr_mult"] == 10.0
    return True


def test_end_to_end_n1d_selfcheck_probe_groups_levers_and_checkpoint(world):
    assert check_e2e(TR, world, "n1d")


def test_end_to_end_n1dx_shuffle_arm_builds_and_records_its_donor_map(world):
    out = Path(world["root"]) / "n1dx"
    assert TR.main(_argv(world, "n1dx", ["--steps", "2", *N1(world), *RBD(world, "shuffle")])) == 0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    rec = sc["info"]["nav_rb"]
    assert sc["declared_vs_built"] == "PASS" and rec["variant"] == "shuffle" and rec["donor_map_sha256"] and rec["n_on"] > 0 and sc["levers"]["nav_rb_feat"] == "exitdist"


def test_end_to_end_state_arm_keeps_the_n1r_record_set(world):
    """An N1R argv (no --nav-rb-feat) = the NAVRB2 behaviour: no nav_rb_feat lever, no per-group probe keys, no feat key in the checkpoint beside the cfg, a [d, 1] weight."""
    out = Path(world["root"]) / "n1r_state"
    assert TR.main(_argv(world, "n1r_state", ["--steps", "3", *N1(world), *RBD(world, feat=None, npz="rb1")])) == 0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    p = sc["navrb_probe"]
    assert sc["declared_vs_built"] == "PASS" and p["pass"] and "nav_rb_feat" not in sc["levers"] and not any(k.startswith("group_") or k in ("feat", "width") for k in p)
    assert sc["info"]["nav_rb"]["feat"] == "state" and sc["info"]["nav_rb"]["width"] == 1
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert tuple(ck["model"]["e_nav_rb.weight"].shape) == (32, 1) and "nav_rb_feat" not in ck and ck["cfg"]["nav_rb_feat"] == "state"


def test_end_to_end_the_flag_off_arm_keeps_its_record_set(world):
    out = Path(world["root"]) / "n1off"
    assert TR.main(_argv(world, "n1off", ["--steps", "2", *N1(world)])) == 0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert "navrb_probe" not in sc and "nav_rb" not in sc["info"] and "nav_rb_feat" not in sc["levers"] and "nav_rb_feat" not in ck and not any(k.startswith("e_nav_rb") for k in ck["model"])


@pytest.mark.parametrize("extra,what", [
    (["--nav-rb-feat", "exitdist"], "off_with_exitdist"),
    (["--nav-rb-feat", "state", "--nav-rb-lr-mult", "10"], "off_with_lr_mult"),
])
def test_g_dvb_refuses_a_channel_off_arm_that_names_a_feat_or_dose(world, extra, what):
    with pytest.raises(SystemExit) as e:
        TR.main(_argv(world, "ref_" + what, ["--steps", "1", "--selfcheck-only", *N1(world), *extra]))
    assert "ZZDVB-REFUSEDZZ" in str(e.value) or "nav" in str(e.value)


def test_g_dvb_refuses_a_width_that_disagrees_with_the_loaded_array(world):
    with pytest.raises((SystemExit, ValueError)):
        TR.main(_argv(world, "ref_w1", ["--steps", "1", "--selfcheck-only", *N1(world), *RBD(world, npz="rb1")]))                    # feat exitdist, a 1-column table
    with pytest.raises((SystemExit, ValueError)):
        TR.main(_argv(world, "ref_w2", ["--steps", "1", "--selfcheck-only", *N1(world), *RBD(world, feat=None)]))                     # feat state (default), a 15-column table


def test_bad_navrb_names_each_width_disagreement(world):
    """bad_navrb read directly: declared == cfg == built lever == built width == loaded width."""
    _, J = make_batch(world, "exitdist")
    a = TR.build_parser().parse_args(["--out", "x", "--nav-rb", "on", "--nav-rb-feat", "exitdist", "--nav-rb-npz", str(world["rbd"]), "--nav-rb-sha256", sha(world["rbd"]), "--nav-rb-lr-mult", "10"])
    m = build(J, nav_rb="on", nav_rb_feat="exitdist")
    _, rec = D.nav_rb_arrays(J, world["rbd"], sha(world["rbd"]), "train", "real", "exitdist")
    assert TR.bad_navrb(a, m, {"nav_rb": rec}) == []
    a_state = TR.build_parser().parse_args(["--out", "x", "--nav-rb", "on", "--nav-rb-npz", str(world["rbd"]), "--nav-rb-sha256", sha(world["rbd"]), "--nav-rb-lr-mult", "10"])
    assert TR.bad_navrb(a_state, m, {"nav_rb": rec})                                                             # declared state, built exitdist + loaded 15 columns
    assert TR.bad_navrb(a, build(J, nav_rb="on"), {"nav_rb": rec})                                                # declared exitdist, built state
    assert TR.bad_navrb(a, m, {"nav_rb": {**rec, "width": 1, "feat": "state"}})                                  # declared exitdist, loaded 1 column
    assert TR.bad_navrb(a, m, {"nav_rb": {**rec, "group_n_on": {**rec["group_n_on"], "c13": 0}}})                 # a column group with no non-zero row
    off = TR.build_parser().parse_args(["--out", "x", "--nav-rb-feat", "exitdist"])
    assert TR.bad_navrb(off, build(J), None)                                                                      # off + a non-default feat


def test_mutation_deleting_the_join_line_turns_the_selfcheck_red(world):
    mod = trainer_module('        J["nav_rb"], info["nav_rb"] = D.nav_rb_arrays(J, a.nav_rb_npz, a.nav_rb_sha256, "train", a.nav_rb_variant, a.nav_rb_feat)', "        pass", "join")
    with pytest.raises(SystemExit) as e:
        mod.main(_argv(world, "mut_join", ["--steps", "1", "--selfcheck-only", *N1(world), *RBD(world)]))
    assert "ZZDVB-REFUSEDZZ" in str(e.value)


def test_mutation_feat_omitted_from_make_config_turns_the_selfcheck_red(world):
    mod = trainer_module("                           nav_rb_feat=a.nav_rb_feat,                                                                   # PREREG_V8F_NAVRBD: the line that applies --nav-rb-feat (a mutation deletes it)\n", "", "mkcfg")
    with pytest.raises(SystemExit) as e:
        mod.main(_argv(world, "mut_mkcfg", ["--steps", "1", "--selfcheck-only", *N1(world), *RBD(world)]))
    assert "ZZDVB-REFUSEDZZ" in str(e.value)


def test_mutation_join_line_without_the_feat_turns_the_selfcheck_red(world):
    mod = trainer_module('"train", a.nav_rb_variant, a.nav_rb_feat)      # PREREG_V8F_NAVRB2', '"train", a.nav_rb_variant)      # PREREG_V8F_NAVRB2', "joinfeat")
    with pytest.raises((SystemExit, ValueError)):
        mod.main(_argv(world, "mut_joinfeat", ["--steps", "1", "--selfcheck-only", *N1(world), *RBD(world)]))


def test_mutation_default_flipped_to_exitdist_turns_the_default_check_red(world):
    mod = trainer_module('"--nav-rb-feat", dest="nav_rb_feat", choices=M.NAV_RB_FEATS, default="state"', '"--nav-rb-feat", dest="nav_rb_feat", choices=M.NAV_RB_FEATS, default="exitdist"', "dflt")
    assert mod.build_parser().parse_args(["--out", "x"]).nav_rb_feat == "exitdist" != TR.build_parser().parse_args(["--out", "x"]).nav_rb_feat          # test_flag_default_is_state... asserts "state": RED for the mutant
    with pytest.raises((SystemExit, ValueError, AssertionError)):
        mod.main(_argv(world, "mut_dflt", ["--steps", "1", "--selfcheck-only", *N1(world), *RBD(world, feat=None, npz="rb1")]))                  # an N1R argv under the flipped default is REFUSED (width 15 vs a 1-column table)


def test_mutation_feat_dropped_from_the_checkpoint_cfg_turns_the_checkpoint_check_red(world):
    mod = trainer_module('"argv_sha": argv_sha(argv), "cfg": cfg.__dict__,', '"argv_sha": argv_sha(argv), "cfg": {k: v for k, v in cfg.__dict__.items() if k != "nav_rb_feat"},', "ckptcfg")
    with pytest.raises(AssertionError):
        check_e2e(mod, world, "mut_ckptcfg")


# ------------------------------------------------------------------------------------------------ the probe and bad_navrb clauses, each reached on its own (a clause another clause subsumes is not a test of it)
class _MaskedC13(M.V8FMini):
    """A mutant model whose channel term ignores the c13 column: the planted arm of the probe must read 0.0 on that group and the probe must refuse."""
    def _nav_tok(self, b, i):
        x = self._at(b["nav_rb"], i).to(self.e_nav_rb.weight.dtype)
        mask = torch.ones(x.shape[-1]); mask[13] = 0.0
        return self.e_nav(self._at(b["nav"], i)) + self.e_nav_rb(x * mask)


class _OffsetChannel(M.V8FMini):
    """A mutant model whose channel term has an offset: the zero channel then gives the weight a gradient (the zero control must refuse)."""
    def _nav_tok(self, b, i):
        return self.e_nav(self._at(b["nav"], i)) + self.e_nav_rb(self._at(b["nav_rb"], i).to(self.e_nav_rb.weight.dtype) + 1.0)


def _probe_with(cls, world):
    b, J = make_batch(world, "exitdist")
    cfg = M.V8FMiniConfig(**{**TINY, "nav_rb": "on", "nav_rb_feat": "exitdist"})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(0)
    m = cls(cfg, M.P10Prior.synthetic(0), mu, sd)
    with torch.no_grad():
        m.head_o[2].weight.copy_(torch.randn(m.head_o[2].weight.shape, generator=torch.Generator().manual_seed(11)) * 0.05)
    return TR.navrb_probe(m, b, cfg)


def test_the_probe_passes_on_the_real_model_with_the_group_record(world):
    res = _probe_with(M.V8FMini, world)
    assert res["pass"] and res["feat"] == "exitdist" and res["width"] == 15
    assert all(v > 0.0 for v in res["group_grads_planted"].values()) and all(v == 0.0 for v in res["group_control_zero_channel"].values())
    assert res["group_n_on"]["c0"] == res["n_on_batch"] > 0


def test_the_probe_refuses_a_model_whose_term_ignores_a_column_group(world):
    with pytest.raises(AssertionError, match="column groups .*c13.* no gradient even with the channel planted"):
        _probe_with(_MaskedC13, world)


def test_the_probe_refuses_a_zero_channel_that_still_gives_a_gradient_and_names_the_groups(world):
    with pytest.raises(AssertionError, match="per group"):
        _probe_with(_OffsetChannel, world)


def test_bad_navrb_width_clause_and_off_feat_clause_are_each_reached_alone(world):
    _, J = make_batch(world, "exitdist")
    a = TR.build_parser().parse_args(["--out", "x", "--nav-rb", "on", "--nav-rb-feat", "exitdist", "--nav-rb-npz", str(world["rbd"]), "--nav-rb-sha256", sha(world["rbd"]), "--nav-rb-lr-mult", "10"])
    _, rec = D.nav_rb_arrays(J, world["rbd"], sha(world["rbd"]), "train", "real", "exitdist")
    m = build(J, nav_rb="on", nav_rb_feat="exitdist")
    m.e_nav_rb = torch.nn.Linear(7, 32, bias=False)                                                              # cfg, lever and loaded array all say exitdist; only the BUILT weight is 7 wide
    bad = TR.bad_navrb(a, m, {"nav_rb": rec})
    assert any("built e_nav_rb.weight is" in x for x in bad), bad
    off = TR.build_parser().parse_args(["--out", "x", "--nav-rb-feat", "exitdist"])
    assert any("--nav-rb off with" in x and "--nav-rb-feat" in x for x in TR.bad_navrb(off, build(J, nav_rb_feat="exitdist"), None))
