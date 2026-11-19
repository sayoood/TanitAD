"""PREREG_V8F_N1DW (frozen sha256 59f2b7f9): the roundabout-window weight ``--rb-window-w w`` on O's trajectory loss (model, mask, trainer). CPU only, fp32, ONE thread (card section 6, R2: every bit-identity or exact-ratio
check runs where same-code runs are bit-identical).

Z0  w = 1.0 build == the 0c3b374f model (snapshot): state dict keys + values, every loss term, the total, the gradients, forward_infer outputs, built_levers, the RNG stream.
Z1  w = 4 vs w = 1, same init, same batch: every output and every existing term bit-identical; the total differs by exactly w_traj x 3 x rb_traj, recomputed independently from the per-knot distances.
Z2  the G-LIVE probe (``rb_window_probe``): term finite and > 0, O's head gets its gradient; with the mask zeroed the term is EXACTLY 0.0 and total's gradient is bit-equal to w = 1's.
Z-ID the per-sample trajectory gradient on RB samples scales by exactly w, on non-RB samples it is unchanged.
MASK  c0 of the nav_rb row at NOW (the last index of the batch's window), read from rows at or before NOW only; agrees with the real train table (when it is on this box).
G-DVB every refusal clause of the card section 2, the flag's FLAG_CHECKS entry, the parser default, the WM-only default list.
Each mutation (K-MUT a-e) is a source edit that must turn a NAMED test RED (``run_mutants.py`` drives them and writes ``mutation_table_n1dw.json``)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

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
CH = dict(nav_rb="on", nav_rb_feat="exitdist")
SNAP = Path(__file__).with_name("v8f_mini_tip_snapshot_n1dw.txt")
SNAP_GIT_BLOB = "15e7bef4994d4a731ee53196f83c8c53ee2f350f"          # git blob of stack/tanitad/models/v8f_mini.py at 0c3b374f (the file N1D_s0 / N1D_s1 trained from)
MODEL_P, TRAIN_P, DATA_P = (REPO / "stack" / "tanitad" / "models" / "v8f_mini.py", REPO / "stack" / "scripts" / "v8f_mini_train.py", REPO / "stack" / "tanitad" / "data" / "v8f_mini_data.py")
RUN = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
REAL_TABLE_SHA = "298049df71dc469874dd2429781c9019c4727b917bc3bd2233314542f6bbc82a"             # navtext_nav_rbd.npz (b2/BANKED_TABLES.json member_sha256)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob_sha(path):
    b = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def write_rbd_npz(path, J, seed=0, p=0.5, split="train"):
    """A VALID 15-column channel aligned to the join: c0 on with prob p; on rows: exit one-hots, dist in (0, 3] with c14 = 1; everything 0 where c0 is 0."""
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
    root = tmp_path_factory.mktemp("n1dwworld")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    return {"root": root, "rbd": write_rbd_npz(root / "rbd.npz", J), "nav": write_nav_npz(root / "nav.npz", J), **w}


def load_J(world, split_table=None):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    f = split_table or world["rbd"]
    J["nav_rb"], _ = D.nav_rb_arrays(J, f, sha(f), "train", "real", "exitdist")
    return J


def make_ds(world, J, hmax=10):
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    return D.V8FMiniWindows(store, J, wins, hmax=hmax), wins


def make_batch(world, n=16, seed=0, hmax=10, J=None, with_rb=True):
    J = J or load_J(world)
    ds, wins = make_ds(world, J, hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    b = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    if with_rb:
        b["rb"] = TR.rb_mask(b)
    return b, J, [wins[i] for i in idx]


def build(J, mod=M, seed=0, **kw):
    cfg = mod.V8FMiniConfig(**{**TINY, **CH, **kw})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    m = mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd)
    with torch.no_grad():
        w = m.head_o[2].weight
        w.copy_(torch.randn(w.shape, generator=torch.Generator().manual_seed(11)) * 0.05)
        m.e_nav_rb.weight.copy_(torch.randn(m.e_nav_rb.weight.shape, generator=torch.Generator().manual_seed(12)) * 0.05)       # the channel is read (zero-init would make it inert at step 0)
    return m


def run_loss(m, b, seed=3):
    torch.manual_seed(seed)
    out = m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(seed))
    return out


def mod_from_text(text, name, path):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(text, str(path), "exec"), mod.__dict__)
    return mod


def base_module():
    return sys.modules.get("v8f_mini_snapshot_n1dw") or mod_from_text(SNAP.read_text(encoding="utf-8"), "v8f_mini_snapshot_n1dw", SNAP)


def mutant(path, old, new, name):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (old[:60], text.count(old))
    return mod_from_text(text.replace(old, new), name, path)


def equal_runs(oa, ob):
    """Two ``loss`` outputs: every tensor output and every existing term bit-identical ([] = identical); keys in ``ob`` only are returned apart."""
    probs = []
    ta, tb = oa["terms"], ob["terms"]
    extra = sorted(set(tb) - set(ta))
    for k in ta:
        if k == "total":
            continue
        if k not in tb or not torch.equal(ta[k], tb[k]):
            probs.append(f"term {k}")
    for k, v in oa.items():
        if torch.is_tensor(v) and not torch.equal(v, ob[k]):
            probs.append(f"output {k}")
    return probs, extra


# ------------------------------------------------------------------------------------------------ pins, flags, defaults
def test_the_snapshot_is_the_0c3b374f_model_and_tanitad_is_the_study_tree():
    assert git_blob_sha(SNAP) == SNAP_GIT_BLOB
    assert Path(M.__file__).resolve() == MODEL_P.resolve()


def test_flag_default_is_one_has_a_flag_check_and_a_wm_only_default():
    assert TR.FLAG_CHECKS["rb_window_w"] == "cfg:rb_window_w"
    assert TR.build_parser().get_default("rb_window_w") == 1.0
    assert "rb_window_w" in {act.dest for act in TR.build_parser()._actions}
    assert "rb_window_w" in TR.WM_ONLY_DEFAULT_DESTS
    assert M.V8FMiniConfig(**TINY).rb_window_w == 1.0


@pytest.mark.parametrize("bad", [0.999, 0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_config_validator_refuses_a_non_finite_or_below_one_weight(bad):
    with pytest.raises(ValueError, match="rb_window_w"):
        M.V8FMiniConfig(**{**TINY, "rb_window_w": bad})
    for ok in (1.0, 2.0, 4.0, 1.5):
        assert M.V8FMiniConfig(**{**TINY, "rb_window_w": ok}).rb_window_w == ok


# ------------------------------------------------------------------------------------------------ Z0
def test_z0_w1_build_is_the_0c3b374f_model_bit_for_bit(world):
    b, J, _ = make_batch(world, with_rb=False)
    mt, mn = build(J, mod=base_module()), build(J, mod=M)
    assert list(mt.state_dict()) == list(mn.state_dict())
    assert all(torch.equal(v, mn.state_dict()[k]) for k, v in mt.state_dict().items()), "state dict values"
    assert mt.built_levers() == mn.built_levers() and "rb_window_w" not in mn.built_levers()
    ot, on = run_loss(mt, b), run_loss(mn, b)
    rng_t, rng_n = torch.get_rng_state().clone(), None
    torch.manual_seed(3); _ = mn.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(3)); rng_n = torch.get_rng_state().clone()
    torch.manual_seed(3); _ = mt.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(3)); rng_t = torch.get_rng_state().clone()
    assert torch.equal(rng_t, rng_n), "the global RNG stream after one loss call"
    probs, extra = equal_runs(ot, on)
    assert probs == [] and extra == [], (probs, extra)
    assert torch.equal(ot["terms"]["total"], on["terms"]["total"]) and set(ot["terms"]) == set(on["terms"])
    ot["terms"]["total"].backward(); on["terms"]["total"].backward()
    pn = dict(mn.named_parameters())
    assert all(torch.equal(p.grad, pn[n].grad) for n, p in mt.named_parameters() if p.grad is not None), "gradients"
    mt.eval(); mn.eval()
    with torch.no_grad():
        it, inn = mt.forward_infer(b), mn.forward_infer(b)
    assert all(torch.equal(it[k], inn[k]) for k in it if torch.is_tensor(it[k])), "forward_infer"


def test_z0_w1_ignores_a_batch_rb_key_and_adds_no_term(world):
    b, J, _ = make_batch(world)
    m = build(J)
    o = run_loss(m, b)
    assert "rb_traj" not in o["terms"] and m.cfg.rb_window_w == 1.0
    o2 = run_loss(m, {k: v for k, v in b.items() if k != "rb"})
    assert torch.equal(o["terms"]["total"], o2["terms"]["total"])


# ------------------------------------------------------------------------------------------------ Z1
def indep_rb_term(traj30, gt, gok, rb):
    """The extra term recomputed from the per-knot distances, in float64, in loops over samples: sum over RB and GT-valid samples of (mean_k d / 3 + 0.5 d_last / 6), over the GT-valid count."""
    d = ((traj30.double() - gt.double()) ** 2).sum(-1).add(1e-9).sqrt()
    n = max(float(gok.sum()), 1.0)
    ade = sum(float(d[i].mean()) for i in range(d.shape[0]) if bool(rb[i]) and float(gok[i]) > 0) / n
    fde = sum(float(d[i, -1]) for i in range(d.shape[0]) if bool(rb[i]) and float(gok[i]) > 0) / n
    return ade / 3.0 + 0.5 * fde / 6.0


def test_z1_w4_vs_w1_every_existing_term_and_output_bit_equal_total_moves_by_exactly_the_term(world):
    b, J, _ = make_batch(world)
    assert 0 < int(b["rb"].sum()) < b["rb"].shape[0], "the batch holds RB and non-RB samples"
    m1, m4 = build(J, rb_window_w=1.0), build(J, rb_window_w=4.0)
    assert all(torch.equal(v, m4.state_dict()[k]) for k, v in m1.state_dict().items())
    o1, o4 = run_loss(m1, b), run_loss(m4, b)
    probs, extra = equal_runs(o1, o4)
    assert probs == [] and extra == ["rb_traj"], (probs, extra)
    t1, t4 = o1["terms"]["total"], o4["terms"]["total"]
    want = float(m4.cfg.w_traj) * 3.0 * indep_rb_term(o4["traj30"].detach(), b["gt"], b["gt_ok"].float(), b["rb"])
    rb_term = float(o4["terms"]["rb_traj"].detach())
    assert abs(rb_term - indep_rb_term(o4["traj30"].detach(), b["gt"], b["gt_ok"].float(), b["rb"])) <= 1e-6 * rb_term, "the term itself vs the independent recompute"
    eps32 = float(torch.finfo(torch.float32).eps)
    diff = float(t4) - float(t1)
    assert want > 0.0 and abs(diff - want) <= 1e-6 * want + 4.0 * eps32 * max(abs(float(t4)), abs(float(t1))), (diff, want)
    assert abs(float(t4) - (float(t1) + want)) <= 1e-6 * abs(float(t4)), "and as totals"


def test_z1_a_batch_with_no_rb_sample_adds_exactly_zero(world):
    b, J, _ = make_batch(world)
    b0 = {**b, "rb": torch.zeros_like(b["rb"])}
    m1, m4 = build(J, rb_window_w=1.0), build(J, rb_window_w=4.0)
    o1, o4 = run_loss(m1, b0), run_loss(m4, b0)
    assert float(o4["terms"]["rb_traj"]) == 0.0 and torch.equal(o1["terms"]["total"], o4["terms"]["total"])


def test_z1_w_above_one_without_an_rb_key_is_a_value_error(world):
    b, J, _ = make_batch(world, with_rb=False)
    with pytest.raises(ValueError, match="no `rb` mask"):
        run_loss(build(J, rb_window_w=2.0), b)


# ------------------------------------------------------------------------------------------------ Z-ID
def test_zid_per_sample_trajectory_gradient_scales_by_exactly_w_on_rb_and_not_on_others(world):
    """(1) The helpers alone: the gradient of w_traj (ade/3 + fde/6/2) + w_traj (w-1) rb_traj wrt traj30, per sample. (2) The FULL model loss: the gradient wrt O's plan differs between w = 4 and w = 1 by exactly (w - 1) x
    the independent per-sample gradient on RB samples and by EXACTLY 0.0 on the others."""
    b, J, _ = make_batch(world)
    rb = b["rb"]
    gok = b["gt_ok"].float()
    m4 = build(J, rb_window_w=4.0)
    w, w_traj = 4.0, float(m4.cfg.w_traj)
    traj = run_loss(m4, b)["traj30"].detach()

    def grad_of(fn):
        x = traj.clone().requires_grad_(True)
        fn(x).backward()
        return x.grad

    base = lambda x: w_traj * (lambda a, f: a / 3.0 + 0.5 * f / 6.0)(*M.traj_terms(x, b["gt"], gok))          # noqa: E731
    both = lambda x: base(x) + w_traj * (w - 1.0) * (lambda a, f: a / 3.0 + 0.5 * f / 6.0)(*M.rb_traj_terms(x, b["gt"], gok, rb))   # noqa: E731
    g1, g4 = grad_of(base), grad_of(both)
    rbs, oth = rb & (gok > 0), ~rb
    assert rbs.any() and oth.any()
    torch.testing.assert_close(g4[rbs], w * g1[rbs], rtol=1e-6, atol=0.0)
    assert torch.equal(g4[oth], g1[oth]), "non-RB samples: unchanged, bit for bit"
    # the full model's loss
    def plan_grad(m):
        o = run_loss(m, b)
        o["traj30"].retain_grad()
        o["terms"]["total"].backward()
        return o["traj30"].grad.clone()

    gm1, gm4 = plan_grad(build(J, rb_window_w=1.0)), plan_grad(m4)
    delta = gm4 - gm1
    torch.testing.assert_close(delta[rbs], (w - 1.0) * g1[rbs], rtol=1e-5, atol=1e-9)
    assert torch.equal(gm4[oth], gm1[oth]) and float(delta[oth].abs().max()) == 0.0, "the full loss leaves a non-RB sample's plan gradient unchanged"


# ------------------------------------------------------------------------------------------------ Z2
def test_z2_probe_term_positive_head_gradient_and_exact_zero_controls(world):
    b, J, _ = make_batch(world)
    m = build(J, rb_window_w=4.0)
    r = TR.rb_window_probe(m, b, m.cfg)
    assert r["pass"] and r["term_forced"] > 0.0 and r["head_o_grad_diff"] > 0.0 and r["term_at_unit_weight"] is None
    assert r["control_allfalse_term"] == 0.0 and r["control_allfalse_total_diff"] == 0.0 and r["control_allfalse_grad_diff"] == 0.0
    assert r["control_repeat_grad_diff"] == 0.0 and r["control_prior_grad_diff"] == 0.0
    assert r["rb_n"] == int(b["rb"].sum()) > 0 and r["natural_contribution"] > 0.0 and r["natural_head_o_grad_diff"] > 0.0
    # independent re-derivation of the zero control: mask zeroed at w = 4, total's gradient == w = 1's, every parameter, bit for bit
    m1, m4 = build(J, rb_window_w=1.0), build(J, rb_window_w=4.0)
    b0 = {**b, "rb": torch.zeros_like(b["rb"])}
    run_loss(m1, b0)["terms"]["total"].backward()
    run_loss(m4, b0)["terms"]["total"].backward()
    p4 = dict(m4.named_parameters())
    assert all(torch.equal(p.grad, p4[n].grad) for n, p in m1.named_parameters() if p.grad is not None)


def test_z2_probe_is_off_at_w1_and_reads_the_loops_own_mask_function(world):
    b, J, _ = make_batch(world)
    m = build(J)
    r = TR.rb_window_probe(m, b, m.cfg)
    assert r["rb_window"] == "off" and r["pass"] is True
    src = Path(TR.__file__).read_text(encoding="utf-8")
    assert "rmask = rb_mask(b)" in src and "probe[\"rb\"] = rb_mask(probe)" in src and "nat = rb_mask(batch)" in src, "the loop and the probe call the same function"


def test_z2_probe_refuses_a_model_whose_extra_term_does_not_reach_the_total(world):
    b, J, _ = make_batch(world)
    mod = mutant(MODEL_P, '''        total = total + cfg.w_traj * (cfg.rb_window_w - 1.0) * T["rb_traj"] if cfg.rb_window_w > 1.0 else total''', '''        total = total''', "_mut_n1dw_probe_noline")
    m = build(J, mod=mod, rb_window_w=4.0)
    with pytest.raises(AssertionError, match="does not reach the loss|visibility|moved by"):
        TR.rb_window_probe(m, b, m.cfg)


def test_z2_probe_refuses_a_mask_that_is_ignored(world):
    b, J, _ = make_batch(world)
    mod = mutant(MODEL_P, '''    g = gok * rb.to(gok.dtype)
    n = gok.sum().clamp_min(1.0)''', '''    g = gok
    n = gok.sum().clamp_min(1.0)''', "_mut_n1dw_probe_allones")
    m = build(J, mod=mod, rb_window_w=4.0)
    with pytest.raises(AssertionError, match="ALL-FALSE mask must read exactly 0.0"):
        TR.rb_window_probe(m, b, m.cfg)


# ------------------------------------------------------------------------------------------------ the mask
def test_mask_is_c0_of_the_nav_rb_row_at_now_a_bool_per_sample(world):
    b, J, picked = make_batch(world, n=24, seed=1)
    ds, _ = make_ds(world, J)
    m = TR.rb_mask(b)
    assert m.dtype == torch.bool and tuple(m.shape) == (24,)
    for k, (ci, t) in enumerate(p_[:2] for p_ in picked):
        assert bool(m[k]) == bool(J["nav_rb"][ds.row(ci, t), 0] > 0.5), (k, ci, t)             # the table's c0 at (clip, NOW)
    assert torch.equal(m, b["nav_rb"][:, -1, 0] > 0.5) and 0 < int(m.sum()) < 24


def test_mask_index_is_the_one_o_reads_at_now(world):
    """O's nav token is ``_nav_tok(b, last_o)`` with ``last_o = last_now = Fn - 1`` (v8f_mini.py _forward): flipping nav_rb's LAST row changes it, flipping any other row does not -- and the mask follows the LAST row."""
    b, J, _ = make_batch(world, n=8)
    m = build(J)
    Fn = b["feats"].shape[1]
    last = torch.full((8,), Fn - 1, dtype=torch.long)
    base = m._nav_tok(b, last)
    flip = lambda i: {**b, "nav_rb": torch.cat([b["nav_rb"][:, :i], 1.0 - b["nav_rb"][:, i:i + 1], b["nav_rb"][:, i + 1:]], 1)}      # noqa: E731
    assert not torch.equal(m._nav_tok(flip(Fn - 1), last), base)
    assert all(torch.equal(m._nav_tok(flip(i), last), base) for i in range(Fn - 1))
    assert torch.equal(TR.rb_mask(flip(Fn - 1)), ~TR.rb_mask(b)), "flipping the LAST row flips every sample's mask (c0 is 0 or 1)"
    assert all(torch.equal(TR.rb_mask(flip(i)), TR.rb_mask(b)) for i in range(Fn - 1)), "an earlier row does not move the mask"


def test_mask_no_future_row_garbage_after_now_leaves_it_unchanged(world):
    """Per sample (clip, NOW t): every table row of that clip AFTER t is replaced by garbage (ones in every column, a value the loader would refuse), the item is rebuilt; the mask must not move."""
    J = load_J(world)
    ds, wins = make_ds(world, J)
    rng = np.random.default_rng(7)
    sel = [wins[i] for i in rng.choice(len(wins), 30, replace=False)]
    clean = D.collate([ds.item(ci, t, 1, 3) for ci, t, *_ in sel])
    ref = TR.rb_mask(clean)
    for k, (ci, t, *_) in enumerate(sel):
        Jg = dict(J)
        garb = np.array(J["nav_rb"], copy=True)
        garb[ds.row(ci, t + 1): ds.row(ci, 0) + int(J["n"][ci])] = 1.0
        ds.J = Jg
        Jg["nav_rb"] = garb
        item = D.collate([ds.item(ci, t, 1, 3)])
        assert bool(TR.rb_mask(item)[0]) == bool(ref[k]), (ci, t)
        assert np.array_equal(item["nav_rb"].numpy(), clean["nav_rb"][k:k + 1].numpy()), "the whole window is unchanged too"
        ds.J = J
    assert 0 < int(ref.sum()) < 30


def _real_table():
    for env in ("N1DW_RBD_NPZ", "N1D_RBD_NPZ"):
        p = os.environ.get(env)
        if p and Path(p).exists():
            return Path(p)
    return None


def test_mask_agrees_with_the_real_train_table_at_200_sampled_pairs():
    """The dev-box copy of ``navtext_nav_rbd.npz`` (sha256 298049df...), found through ``N1DW_RBD_NPZ``: for 200 sampled (clip, NOW) pairs the mask built the way ``item`` + ``collate`` build the window (``nav_rb`` rows of
    ``t - hmax + 1 .. t``, clipped, ``rb_mask`` = the last one's c0) equals the table's c0 at (clip, t). Prints p = the RB share of the table's train steps. SKIPPED, with the reason, when the table is not on this box."""
    path = _real_table()
    if path is None:
        pytest.skip("navtext_nav_rbd.npz (sha256 298049df...) is not on this box: set N1DW_RBD_NPZ; the B2 pod run does this check")
    assert sha(path) == REAL_TABLE_SHA, "the real table's sha256"
    z = np.load(path)
    off, n, rb = np.asarray(z["train_off"]), np.asarray(z["train_n"]), np.asarray(z["train_nav_rb"], np.float32)
    N = int(rb.shape[0])
    p_share = float(rb[:, 0].sum() / N)
    rng = np.random.default_rng(20261119)
    hmax, items = 10, []
    pairs = [(int(c), int(rng.integers(hmax - 1, int(n[c])))) for c in rng.integers(0, len(n), 200)]
    for ci, t in pairs:
        idc = np.clip(np.arange(t - hmax + 1, t + 1), 0, int(n[ci]) - 1)
        items.append({"nav_rb": rb[[int(off[ci]) + int(j) for j in idc]]})          # the expression of V8FMiniWindows.item (v8f_mini_data.py: J["nav_rb"][[self.row(ci, j) for j in idc]])
    mask = TR.rb_mask({"nav_rb": torch.as_tensor(np.stack([it["nav_rb"] for it in items]))})
    want = np.array([rb[int(off[ci]) + t, 0] > 0.5 for ci, t in pairs])
    assert (mask.numpy() == want).all() and 0 < int(mask.sum()) < 200
    print(f"REAL-TABLE p = {p_share:.6f} ({int(rb[:, 0].sum())} of {N} train steps), sampled-pair RB share {float(mask.float().mean()):.4f}")


# ------------------------------------------------------------------------------------------------ G-DVB
def N1(world):
    return ["--nav-source", "text", "--nav-text-npz", str(world["nav"]), "--nav-text-sha256", sha(world["nav"]), "--ctx-heads", "off", "--w-ctx", "0", "--ctx-labels", "", "--ctx-labels-sha256", "",
            "--ctx-lr-mult", "1", "--ctx-detach", "off"]


def RBD(world, nav_rb="on"):
    if nav_rb == "off":
        return []
    return ["--nav-rb", "on", "--nav-rb-npz", str(world["rbd"]), "--nav-rb-sha256", sha(world["rbd"]), "--nav-rb-variant", "real", "--nav-rb-lr-mult", "10", "--nav-rb-feat", "exitdist"]


def _argv(world, tag, extra):
    return ["--out", str(Path(world["root"]) / tag), *RUN, *extra]


def gate_parts(world, extra, tag="gp"):
    a = TR.build_parser().parse_args(_argv(world, tag, [*N1(world), *extra]))
    cfg = TR.make_config(a)
    store, J, wins, clips, info = TR.build_data(a, cfg)
    info["ds_window_start"] = D.effective_window_start(cfg.hist, cfg.window_start)
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(a.seed), mu, sd)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    return a, cfg, model, opt, info


def dvb(parts):
    a, cfg, model, opt, info = parts
    return TR.declared_vs_built(a, cfg, model, opt, info, None, TR.make_ego_perturb(a), TR.LRSchedule(a.lr, a.warmup, 10), None)


def rb_only(msgs):
    return [m for m in msgs if "rb_window_w" in m]


def test_gdvb_passes_an_honoured_argv_and_names_the_flag_in_built_levers(world):
    parts = gate_parts(world, [*RBD(world), "--rb-window-w", "4"], "ok4")
    assert rb_only(dvb(parts)) == []
    assert parts[2].built_levers()["rb_window_w"] == 4.0 and parts[2].cfg.rb_window_w == 4.0
    parts1 = gate_parts(world, [*RBD(world), "--rb-window-w", "1.0"], "ok1")
    assert rb_only(dvb(parts1)) == [] and "rb_window_w" not in parts1[2].built_levers()


@pytest.mark.parametrize("bad_v", [0.5, -2.0, float("nan"), float("inf")])
def test_gdvb_refuses_a_non_finite_or_below_one_declared_weight(world, bad_v):
    parts = gate_parts(world, [*RBD(world), "--rb-window-w", "4"], "bad_v")
    parts[0].rb_window_w = bad_v                                                                # the argv says bad_v (the config validator refuses it earlier on a real launch)
    assert any("is not a finite number >= 1" in m for m in dvb(parts)), bad_v


def test_gdvb_refuses_a_weight_with_nav_rb_off(world):
    parts = gate_parts(world, [], "off0")
    parts[0].rb_window_w = 4.0
    parts[1].rb_window_w = 4.0
    parts[2].cfg.rb_window_w = 4.0
    msgs = dvb(parts)
    assert any("rb_window_w 4.0 declared but --nav-rb is off" in m for m in msgs), msgs


def test_gdvb_refuses_a_weight_on_a_zero_w_traj(world):
    parts = gate_parts(world, [*RBD(world), "--rb-window-w", "4"], "wtraj0")
    parts[2].cfg.w_traj = 0.0
    assert any("rb_window_w 4.0 declared but w_traj is 0.0" in m and "does nothing" in m for m in dvb(parts))


def test_gdvb_refuses_a_build_that_did_not_honour_the_argv(world):
    parts = gate_parts(world, [*RBD(world), "--rb-window-w", "4"], "dropped")
    assert rb_only(dvb(parts)) == []
    parts[2].cfg.rb_window_w = 1.0                                                              # the build dropped the lever (the refcv6 class: declared, never built)
    msgs = dvb(parts)
    assert any("lever rb_window_w: declared 4.0 but built 1.0" in m for m in msgs) and any("cfg.rb_window_w: declared 4.0 but built 1.0" in m for m in msgs), msgs
    parts[2].cfg.rb_window_w = 2.0                                                              # built a different number
    assert any("declared 4.0 but built 2.0" in m for m in dvb(parts))


def test_a_flag_without_a_gdvb_entry_is_refused_by_the_gate_itself(world, monkeypatch):
    parts = gate_parts(world, [*RBD(world), "--rb-window-w", "4"], "noentry")
    assert [m for m in dvb(parts) if "declared-vs-built entry" in m] == []
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != "rb_window_w"})
    bad = dvb(parts)
    assert any("flags without a declared-vs-built entry" in m and "rb_window_w" in m for m in bad), bad


def test_main_refuses_each_bad_argv_before_anything_trains(world):
    for tag, extra in (("r_nan", [*RBD(world), "--rb-window-w", "nan"]), ("r_lt1", [*RBD(world), "--rb-window-w", "0.5"]), ("r_off", ["--rb-window-w", "4"]), ("r_wtraj0", [*RBD(world), "--rb-window-w", "4", "--w-traj", "0"])):
        with pytest.raises((SystemExit, ValueError)) as e:
            TR.main(_argv(world, tag, ["--steps", "1", "--selfcheck-only", *N1(world), *extra]))
        assert (isinstance(e.value, ValueError) and "rb_window_w" in str(e.value)) or "ZZDVB-REFUSEDZZ" in str(e.value) or "rb_window_w" in str(e.value), (tag, str(e.value)[:200])


# ------------------------------------------------------------------------------------------------ the trainer, end to end
def _read(out, name):
    return json.loads((Path(out) / name).read_text(encoding="utf-8"))


def test_end_to_end_n1dw_arm_probe_levers_counts_and_checkpoint(world):
    out = Path(world["root"]) / "n1dw4"
    assert TR.main(_argv(world, "n1dw4", ["--steps", "4", *N1(world), *RBD(world), "--rb-window-w", "4"])) == 0
    sc, sm = _read(out, "selfcheck.json"), _read(out, "summary.json")
    pr = sc["rb_window_probe"]
    assert sc["declared_vs_built"] == "PASS" and pr["pass"] and pr["term_forced"] > 0.0 and pr["head_o_grad_diff"] > 0.0 and pr["control_allfalse_grad_diff"] == 0.0
    assert sc["levers"]["rb_window_w"] == 4.0
    assert sm["rb_window_w"] == 4.0 and sm["rb_n"] > 0 and sm["rb_eligible"] > 0 and sm["rb_rows"] >= sm["rb_eligible"] and sm["rb_n_eligible"] <= sm["rb_n"]
    p = sm["rb_n_eligible"] / sm["rb_eligible"]
    assert abs(sm["rb_weight_share"] - 4.0 * p / (4.0 * p + 1.0 - p)) < 1e-12
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert ck["cfg"]["rb_window_w"] == 4.0 and ck["rb_totals"]["n"] == sm["rb_n"]
    mm = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior.synthetic(0), *M.cons_stats(D.load_join(Path(world["root"]) / "join_train.npz")))
    mm.load_state_dict(ck["model"])                                                             # the evaluators' build path, strict
    assert mm.cfg.rb_window_w == 4.0 and mm.built_levers()["rb_window_w"] == 4.0
    logs = [json.loads(x) for x in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("rb_traj" in r and "rb_n" in r for r in logs) and sum(r["rb_n"] for r in logs) == sm["rb_n"] and all(r["rb_traj"] >= 0.0 for r in logs)


def test_end_to_end_flag_off_and_explicit_one_keep_the_record_set(world):
    for tag, extra in (("n1off", []), ("n1one", ["--rb-window-w", "1.0"])):
        out = Path(world["root"]) / tag
        assert TR.main(_argv(world, tag, ["--steps", "2", *N1(world), *RBD(world), *extra])) == 0
        sc, sm = _read(out, "selfcheck.json"), _read(out, "summary.json")
        ck = torch.load(out / "ckpt_last.pt", weights_only=False)
        assert "rb_window_probe" not in sc and "rb_window_w" not in sc["levers"] and not any(k.startswith("rb_") for k in sm) and "rb_totals" not in ck
        logs = [json.loads(x) for x in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert not any(k.startswith("rb_") for r in logs for k in r)
        assert ck["cfg"]["rb_window_w"] == 1.0


def test_resume_refuses_a_checkpoint_whose_cfg_weight_differs_from_the_argv(world):
    """Card section 2: a checkpoint cfg whose rb_window_w differs from the argv is refused. The argv-sha refusal fires first for a changed argv, so the checkpoint's cfg is edited in place (same argv, same weights)."""
    argv = _argv(world, "n1dw_resume", ["--steps", "3", *N1(world), *RBD(world), "--rb-window-w", "4"])
    assert TR.main(argv) == 0
    p = Path(world["root"]) / "n1dw_resume" / "ckpt_last.pt"
    ck = torch.load(p, weights_only=False)
    assert TR.main(argv) == 0                                                    # the honest checkpoint resumes (already done: returns)
    ck["cfg"]["rb_window_w"] = 2.0
    torch.save(ck, p)
    with pytest.raises(SystemExit, match="ZZRESUME-REFUSEDZZ the checkpoint's cfg.rb_window_w is 2.0 but the argv declares 4.0"):
        TR.main(argv)
    del ck["cfg"]["rb_window_w"]                                                 # a checkpoint from before the lever reads as 1.0: refused under w = 4, accepted under the default
    torch.save(ck, p)
    with pytest.raises(SystemExit, match="ZZRESUME-REFUSEDZZ the checkpoint's cfg.rb_window_w is 1.0"):
        TR.main(argv)
