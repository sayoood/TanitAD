"""PREREG_LEVELHEAD_A2: ``--lead-head {off,on}`` + ``--w-lead`` + ``--aux-cond {add,token}`` in ``stack/tanitad/models/v8f_mini.py``, ``stack/tanitad/data/v8f_mini_data.py`` (``lead_label_arrays``) and
``stack/scripts/v8f_mini_train.py`` -- arm LLR of card ``motion_c/a2_prereg/PREREG_LEVELHEAD_A2.md``: a second vision-only head (presence of a lead within 40 m + its gap) wired into O like the level head.

Defaults OFF / ``add``. THIS file proves the lever is REACHABLE from the real caller (CLAUDE.md 2026-09-10/11: "built, tested, and unreachable from its caller" is a class), hop by hop:

  argv --lead-head/--w-lead/--aux-cond -> make_config -> cfg -> the BUILT model (lead_head + e_lead | e_aux) -> declared_vs_built (bad_lead, FLAG_CHECKS) -> lead_head_probe (G-LIVE on the loop's own label call lead_stage)
  -> ``lead_lab = lead_stage(b, LL)`` BEFORE the ego stages, ``b["lead_y"], b["lead_has"], b["lead_near"]`` AFTER them -> the batch the LOSS receives -> V8FMini.loss (BCE + Huber + ONE total line)
  -> O's cond (add: e_lead([sigmoid(logit), gap_hat].detach()) ADDED to the ego token | token: ONE extra conditioning token e_aux) -> the plan -> summary.json / selfcheck.json / train_log.jsonl / the checkpoint.

  1. LABELS: ``lead_label_arrays`` against a brute-force reader of the lead file (k = 2 j; has / near / y); ``lead_stage`` looks the row up by the batch's own clip index and step.
  2. OFF == TIP: a frozen snapshot of the tip's (219f6c7, level head included) model module (sha256 pinned) and the new one, same seed, for level off and level on: state dict, loss terms, gradients, forward_infer, bit for bit.
  3. ON adds EXACTLY the declared keys; ``add`` at step 0 equals the init's forward EXACTLY; ``token`` differs by construction (one more conditioning token) and the test pins what stays identical and that the head outputs reach O's cond.
  4. The term (hand computation), the masks, the loss weight, the W-for-10-percent formula.
  5. ADMISSIBILITY: the head and the aux token read vision only -- static, key-guard and perturbation guards, and the mutations go RED.
  6. G-LIVE (lead_head_probe, and the level probe under token): contribution, BOTH output rows get gradient, the DETACH control (exactly 0.0), the output reaches O, the all-false mask reads exactly 0.0; mutations go RED.
  7. The launch gate refuses: a head with no loss, a weight with no head, token with no head, a wiring that is not the declared one, a missing / thin label file; ``--init-ckpt`` loads every old key and leaves fresh ONLY the declared keys.
  8. End to end on the trainer's synthetic world, both wirings.
"""
from __future__ import annotations

import hashlib
import json
import math
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
SNAP = Path(__file__).with_name("v8f_mini_tip_snapshot_lead.txt")
SNAP_SHA = "a363427677c881cb7c82cf20fcfad317e864a9cde58cf58306e9fed24ef3658e"      # sha256 of the tip's (219f6c7) stack/tanitad/models/v8f_mini.py (the model WITH the level head)
STOPBAL_DEFAULTS = {"goal_pos_weight_stop": 1.0, "lon_class_weight_stop": 1.0}   # PREREG_V8F_STOPBAL (5ecdd7e6's parent 2bca1c52): built_levers gains these two keys after the snapshot was taken; off = 1.0 = the tip's loss
TRAIN_P = REPO / "stack" / "scripts" / "v8f_mini_train.py"
C0_INIT = Path("E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt")        # the C0 init checkpoint (dev box only)
RUN = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
LEAD_ADD = {"lead_head.0.weight", "lead_head.0.bias", "lead_head.2.weight", "lead_head.2.bias", "e_lead.weight", "e_lead.bias"}
LEAD_TOKEN = {"lead_head.0.weight", "lead_head.0.bias", "lead_head.2.weight", "lead_head.2.bias", "e_aux.weight", "e_aux.bias"}
LEVEL_ADD = {"lvl_head.0.weight", "lvl_head.0.bias", "lvl_head.2.weight", "lvl_head.2.bias", "e_level.weight", "e_level.bias"}
LEVEL_HEAD_ONLY = {k for k in LEVEL_ADD if k.startswith("lvl_head.")}
AUX = {"e_aux.weight", "e_aux.bias"}
BOTH_HEADS_TOKEN = LEVEL_HEAD_ONLY | {k for k in LEAD_TOKEN if k.startswith("lead_head.")} | AUX


# ------------------------------------------------------------------------------------------------ fixtures / helpers
def write_lead_npz(path, names):
    """A synthetic lead file in the real layout (sha12, k, has, lead, gap, tg, s, v): per clip a mix of no-lead, a lead beyond 40 m, a lead within 40 m (incl. a negative gap), and a stretch with NO label rows."""
    sh, kk, ld, gp = [], [], [], []
    for ci, name in enumerate(names):
        for k in range(0, 2 * 60 + 2):
            j = k // 2
            if ci in (0, 2, 4) and 15 <= j < 30:
                continue                                                  # no label rows for these frames
            lead = (j // 7 + ci) % 2 == 0 and j % 5 != 0
            gap = (-2.0 if j % 11 == 0 else 10.0 + (3 * j + ci) % 50) if lead else float("nan")
            sh.append(D.sha12(name))
            kk.append(k)
            ld.append(1 if lead else 0)
            gp.append(gap)
    n = len(kk)
    np.savez(path, sha12=np.array(sh), k=np.array(kk, np.int64), has=np.ones(n, np.int64), lead=np.array(ld, np.int64), gap=np.array(gp, np.float64), tg=np.zeros(n), s=np.zeros(n), v=np.zeros(n),
             meta=np.array(json.dumps({"synthetic": True})))
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("leadworld")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    npz = write_lead_npz(root / "lead_train.npz", w["names"]["train"])
    return {"root": root, "npz": npz, **w}


def make_batch(world, n=16, seed=0, hmax=10):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    return D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx]), J


def LLof(world, J):
    return D.lead_label_arrays(J, world["npz"])


def with_lead(b, LL):
    y, has, near = TR.lead_stage(b, LL)
    return {**b, "lead_y": y, "lead_has": has, "lead_near": near}


def with_both(b, LL):
    y, ok = TR.level_stage(b)
    return {**with_lead(b, LL), "level_y": y, "level_ok": ok}


def build(J, mod=M, seed=0, wake=True, cls=None, **kw):
    cfg = mod.V8FMiniConfig(**{**TINY, **kw})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    m = (cls or mod.V8FMini)(cfg, mod.P10Prior.synthetic(0), mu, sd)
    if wake:                                                   # the residual head is zero-init: wake it so a change at O's input can be seen at its output
        with torch.no_grad():
            w = m.head_o[2].weight
            w.copy_(torch.randn(w.shape, generator=torch.Generator().manual_seed(11)) * 0.05)
    return m


def tip_module():
    name = "v8f_mini_tip_snapshot_lead"
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__file__ = str(SNAP)
    sys.modules[name] = mod                                   # dataclasses resolves string annotations through sys.modules
    exec(compile(SNAP.read_text(encoding="utf-8"), str(SNAP), "exec"), mod.__dict__)
    return mod


def wake_lead(m, seed=17, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.e_lead.weight.copy_(torch.randn(m.e_lead.weight.shape, generator=g) * scale)
        m.e_lead.bias.copy_(torch.randn(m.e_lead.bias.shape, generator=g) * scale)
    return m


def run_loss(m, b, seed=3):
    torch.manual_seed(seed)
    return m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(seed))


def args(world, extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--lead-train", str(world["npz"]), *extra])


def lead_info(a, LL, **over):
    return {"path": os.path.abspath(a.lead_train), "sha256": "0" * 64, **{k: LL[k] for k in ("n_rows", "n_has", "n_near", "n_clips", "n_clips_with_labels")}, **over}


def built(a, J, LL=None, info_over=None, **over):
    cfg = TR.make_config(a)
    keep = {k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**cfg.__dict__, **keep, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True, "kin_mode": a.kin_mode, "p10_units": None}
    if LL is not None:
        info["lead_labels"] = lead_info(a, LL)
    info.update(info_over or {})
    return cfg, model, opt, info


def gate(a, J, LL, info_over=None, **over):
    cfg, model, opt, info = built(a, J, LL, info_over, **over)
    return TR.declared_vs_built(a, cfg, model, opt, info), model


def _sc(out):
    return json.loads((Path(out) / "selfcheck.json").read_text(encoding="utf-8"))


def _sm(out):
    return json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))


def _lg(out):
    return [json.loads(x) for x in (Path(out) / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]


def _run_argv(world):
    return [*RUN, "--lead-train", str(world["npz"])]


def refused(world, argv, tmp_path, match):
    with pytest.raises(SystemExit) as e:
        TR.main(["--out", str(tmp_path / "refused"), *_run_argv(world), *argv])
    assert "ZZDVB-REFUSEDZZ" in str(e.value) and match in str(e.value), str(e.value)[:600]


# ------------------------------------------------------------------------------------------------ 1. the labels
def brute_rows(world, J):
    z = np.load(world["npz"], allow_pickle=True)
    by = {}
    for s, k, ld, gp in zip(z["sha12"], z["k"], z["lead"], z["gap"]):
        by.setdefault(str(s), {})[int(k)] = (int(ld), float(gp))
    return by


def test_lead_label_arrays_against_a_brute_force_reader(world):
    _, J = make_batch(world)
    LL = LLof(world, J)
    by = brute_rows(world, J)
    n_has = n_near = 0
    for ci, cid in enumerate(J["clip"]):
        for j in range(int(J["n"][ci])):
            row = int(J["off"][ci]) + j
            e = by[D.sha12(str(cid))].get(2 * j)                          # cache step j reads raw frame k = 2 j
            if e is None:
                assert not LL["has"][row] and not LL["near"][row] and LL["y"][row] == 0.0
                continue
            n_has += 1
            assert LL["has"][row]
            near = e[0] == 1 and math.isfinite(e[1]) and e[1] <= 40.0
            assert bool(LL["near"][row]) == near
            if near:
                n_near += 1
                assert LL["y"][row] == pytest.approx(min(max(min(e[1], 40.0) / 40.0, 0.0), 1.0), abs=1e-7)
            else:
                assert LL["y"][row] == 0.0
    assert LL["n_has"] == n_has > 0 and LL["n_near"] == n_near > 0 and LL["n_clips_with_labels"] == len(J["clip"]) == LL["n_clips"]
    assert n_near < n_has, "the synthetic file has frames with no lead within 40 m (a lead beyond 40 m, and no lead)"
    assert (LL["y"] >= 0).all() and (LL["y"] <= 1).all()
    assert D.LEAD_RANGE_M == M.LEAD_RANGE_M == 40.0 and M.LEAD_HUBER_DELTA == 0.1


def test_lead_label_semantics_on_a_hand_built_file():
    z = {"sha12": np.array([D.sha12("c")] * 6), "k": np.array([0, 2, 4, 6, 8, 12]), "lead": np.array([1, 1, 1, 0, 1, 1]), "gap": np.array([40.0, 41.0, -3.0, np.nan, np.nan, 10.0])}
    J = {"clip": np.array(["c", "other"]), "off": np.array([0, 8]), "n": np.array([8, 2])}
    r = D.lead_label_arrays(J, z)
    # steps j = 0..7 read k = 0, 2, 4, ..., 14: labels at k = 0, 2, 4, 6, 8, (10 absent), 12, (14 absent)
    assert r["has"].tolist() == [True, True, True, True, True, False, True, False, False, False]
    assert r["near"].tolist() == [True, False, True, False, False, False, True, False, False, False], "gap 40 is within 40 m, 41 is not, -3 is a lead at 0 m, lead 0 is none, a NaN gap is no lead"
    assert r["y"].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0])
    assert r["n_clips_with_labels"] == 1 and r["n_clips"] == 2 and r["n_has"] == 6 and r["n_near"] == 3


def test_lead_stage_looks_the_row_up_by_the_batchs_own_clip_index_and_step(world):
    b, J = make_batch(world)
    LL = LLof(world, J)
    by = brute_rows(world, J)
    y, has, near = TR.lead_stage(b, LL)
    assert y.dtype == torch.float32 and has.dtype == torch.bool and near.dtype == torch.bool and y.shape == has.shape == near.shape == (16,)
    for i in range(16):
        ci, t = int(b["ci"][i]), int(b["t"][i])
        e = by[D.sha12(str(J["clip"][ci]))].get(2 * t)
        assert bool(has[i]) == (e is not None)
        if e is not None:
            assert bool(near[i]) == (e[0] == 1 and math.isfinite(e[1]) and e[1] <= 40.0)
    assert has.any() and near.any() and not near.all() and not has.all(), "the probe batch sees every case: unlabelled, no lead, a far lead, a near lead"


@pytest.mark.skipif(not Path("E:/Projects/TanitAD-artifacts/v9labels/inputs/lead_train.npz").exists(), reason="the real lead file is on the dev box only")
def test_the_real_lead_file_has_the_documented_layout():
    z = np.load("E:/Projects/TanitAD-artifacts/v9labels/inputs/lead_train.npz", allow_pickle=True)
    assert {"sha12", "k", "has", "lead", "gap"} <= set(z.files)
    assert set(np.unique(z["lead"]).tolist()) == {0, 1} and set(np.unique(z["has"]).tolist()) == {1}
    assert int(np.isfinite(z["gap"]).sum()) == int((z["lead"] == 1).sum()), "a gap exists exactly where there is a lead"


# ------------------------------------------------------------------------------------------------ 2. off == the tip
def test_the_snapshot_is_the_tip_module():
    assert hashlib.sha256(SNAP.read_bytes()).hexdigest() == SNAP_SHA


@pytest.mark.parametrize("level,wm", [("off", "off"), ("off", "b"), ("on", "off")])
def test_off_is_the_tip_bit_for_bit(world, level, wm):
    hmax = 16 if wm != "off" else 10
    b, J = make_batch(world, hmax=hmax)
    LL = LLof(world, J)
    T0 = tip_module()
    kw = dict(wm=wm, hmax=hmax, **({"level_head": "on", "w_level": 0.5} if level == "on" else {}))
    tip, new = build(J, T0, **kw), build(J, M, **kw)
    assert new.cfg.lead_head == "off" and new.cfg.w_lead == 0.0 and new.cfg.aux_cond == "add"
    assert new.lead_head is None and new.e_lead is None and new.e_aux is None
    sa, sb = tip.state_dict(), new.state_dict()
    assert list(sa) == list(sb), "state-dict keys and their order"
    assert all(torch.equal(sa[k], sb[k]) for k in sa), "every tensor"
    nl = new.built_levers()
    assert {k: nl[k] for k in STOPBAL_DEFAULTS} == STOPBAL_DEFAULTS, "PREREG_V8F_STOPBAL: both stop weights are built at their default 1.0 (the tip's loss)"
    assert tip.built_levers() == {k: v for k, v in nl.items() if k not in STOPBAL_DEFAULTS} and set(new.trainable_groups()) == set(tip.trainable_groups()), "levers equal the tip's, apart from the two STOPBAL weights the snapshot predates"
    assert not {"lead_head", "w_lead", "aux_cond", "lead_vision_only_source", "aux_vision_only_source"} & set(new.built_levers())
    bb = with_both(b, LL)
    oa, ob = run_loss(tip, bb), run_loss(new, bb)
    assert list(oa["terms"]) == list(ob["terms"]) and all(torch.equal(oa["terms"][k], ob["terms"][k]) for k in oa["terms"]), "the loss terms, same keys, bit for bit"
    oa["terms"]["total"].backward()
    ob["terms"]["total"].backward()
    ga, gb = dict(tip.named_parameters()), dict(new.named_parameters())
    assert all((ga[k].grad is None and gb[k].grad is None) or torch.equal(ga[k].grad, gb[k].grad) for k in ga), "the gradients"
    tip.eval()
    new.eval()
    with torch.no_grad():
        ia, ib = tip.forward_infer(bb), new.forward_infer(bb)
    assert set(ia) == set(ib) and "lead_logit" not in ib and "lead_gap" not in ib
    assert all(torch.equal(ia[k], ib[k]) for k in ia if torch.is_tensor(ia[k])), "forward_infer, every tensor output"


# ------------------------------------------------------------------------------------------------ 3. on: the declared keys, step 0
def test_add_adds_exactly_the_declared_keys_and_leaves_the_old_ones_identical(world):
    _, J = make_batch(world)
    off, lead, lvl = build(J), build(J, lead_head="on", w_lead=0.5), build(J, level_head="on", w_level=0.5)
    both = build(J, level_head="on", w_level=0.5, lead_head="on", w_lead=0.5)
    ko, kl, kv, kb = off.state_dict(), lead.state_dict(), lvl.state_dict(), both.state_dict()
    assert set(kl) - set(ko) == LEAD_ADD and set(ko) <= set(kl) and all(torch.equal(ko[k], kl[k]) for k in ko), "the lead head is built LAST: every pre-existing tensor draws the off model's init"
    assert set(kb) - set(ko) == LEAD_ADD | LEVEL_ADD and all(torch.equal(kv[k], kb[k]) for k in kv), "with both heads: the level head draws EXACTLY the level-only model's init (the lead head comes after it)"
    assert float(lead.e_lead.weight.detach().abs().sum()) == 0.0 and float(lead.e_lead.bias.detach().abs().sum()) == 0.0, "e_lead starts at EXACTLY zero"
    lv = lead.built_levers()
    assert lv["lead_head"] == "on" and lv["w_lead"] == 0.5 and lv["lead_token_zero_init"] is True and lv["lead_vision_only_source"] is True and lv["aux_cond"] == "add" and lv["aux_vision_only_source"] is True
    assert lv["params"]["LEAD"] == sum(p.numel() for n, p in lead.named_parameters() if n.startswith(M.LEAD_KEY_PREFIXES)) and "AUX" not in lv["params"]


def test_token_adds_exactly_the_aux_token_and_builds_no_additive_path(world):
    _, J = make_batch(world)
    off = build(J)
    for kw, want in ((dict(lead_head="on", w_lead=0.5), LEAD_TOKEN), (dict(level_head="on", w_level=0.5), LEVEL_HEAD_ONLY | AUX), (dict(level_head="on", w_level=0.5, lead_head="on", w_lead=0.5), BOTH_HEADS_TOKEN)):
        m = build(J, aux_cond="token", **kw)
        km, ko = m.state_dict(), off.state_dict()
        assert set(km) - set(ko) == want and set(ko) <= set(km) and all(torch.equal(ko[k], km[k]) for k in ko), kw
        assert m.e_level is None and m.e_lead is None and m.e_aux is not None and m.e_aux.in_features == M.AUX_TOKEN_DIM == 3
        assert float(m.e_aux.weight.detach().abs().sum()) > 0.0, "the extra token has a default (non-zero) init"
        lv = m.built_levers()
        assert lv["aux_cond"] == "token" and lv["aux_token_dim"] == 3 and lv["aux_vision_only_source"] is True and lv["params"]["AUX"] == 3 * 32 + 32
        assert "level_token_zero_init" not in lv and "lead_token_zero_init" not in lv
    assert "aux_cond" not in build(J).built_levers() and build(J, aux_cond="token").e_aux is None, "token with no head builds nothing (the gate refuses it: see the gate test)"


@pytest.mark.parametrize("kw", [dict(lead_head="on", w_lead=0.5), dict(level_head="on", w_level=0.5, lead_head="on", w_lead=0.5)])
def test_add_step_0_forward_equals_the_inits_exactly(world, kw):
    b, J = make_batch(world)
    LL = LLof(world, J)
    ref = build(J, **({"level_head": "on", "w_level": 0.5} if "level_head" in kw else {}))
    on = build(J, **kw)
    res = on.load_state_dict(ref.state_dict(), strict=False)
    assert set(res.missing_keys) == LEAD_ADD and not res.unexpected_keys
    ref.eval()
    on.eval()
    bb = with_both(b, LL)
    with torch.no_grad():
        a, c = ref.forward_infer(bb), on.forward_infer(bb)
    assert set(c) - set(a) == {"lead_logit", "lead_gap"}
    assert all(torch.equal(a[k], c[k]) for k in a if torch.is_tensor(a[k])), "inference: every output equals the init's, EXACTLY (zero-init e_lead)"
    ref.train()
    on.train()
    oa, oc = run_loss(ref, bb), run_loss(on, bb)
    assert all(torch.equal(oa["terms"][k], oc["terms"][k]) for k in oa["terms"] if k != "total"), "training: every pre-existing loss term equals the init's"
    d = float((oc["terms"]["total"] - oa["terms"]["total"] - 0.5 * oc["terms"]["lead"]).detach())
    assert d == pytest.approx(0.0, abs=1e-4 * max(1.0, float(oa["terms"]["total"].detach()))), "total = init total + w_lead x lead"


def test_token_step_0_forward_differs_by_construction_but_old_keys_load_identical_and_the_extra_token_is_present(world):
    b, J = make_batch(world)
    LL = LLof(world, J)
    ref = build(J)
    on = build(J, aux_cond="token", level_head="on", w_level=0.5, lead_head="on", w_lead=0.5)
    res = on.load_state_dict(ref.state_dict(), strict=False)
    assert set(res.missing_keys) == BOTH_HEADS_TOKEN and not res.unexpected_keys
    sd_on = on.state_dict()
    assert all(torch.equal(sd_on[k], v) for k, v in ref.state_dict().items()), "old keys load bit-identical"
    seen = {"ref": [], "on": []}
    ref.read_o.register_forward_pre_hook(lambda mod, a_: seen["ref"].append(a_[0].detach().clone()))
    on.read_o.register_forward_pre_hook(lambda mod, a_: seen["on"].append(a_[0].detach().clone()))
    ref.eval()
    on.eval()
    bb = with_both(b, LL)
    with torch.no_grad():
        a, c = ref.forward_infer(bb), on.forward_infer(bb)
    assert not torch.equal(a["traj60"], c["traj60"]), "token: the step-0 forward differs from the init's BY CONSTRUCTION (one more key, and the query is cond.mean)"
    cr, co = seen["ref"][-1], seen["on"][-1]
    assert cr.shape[1] == 6 and co.shape[1] == 7, "O's cond has ONE extra token"
    assert torch.equal(co[:, :6], cr), "the six old tokens are identical (no additive path touches the ego token)"
    with torch.no_grad():
        adapt = on._encode(bb)
        last = torch.full((16,), bb["feats"].shape[1] - 1, dtype=torch.long)
        want = on.e_aux(on._aux_in(bb, adapt, last))
    assert torch.equal(co[:, 6], want), "the extra token IS e_aux of the heads' detached outputs: the head outputs reach O's cond"
    with torch.no_grad():
        saved = on.e_aux.weight.clone()
        on.e_aux.weight.zero_()
        t_zero = on.forward_infer(bb)["traj60"]
        on.e_aux.weight.copy_(saved)
    assert not torch.equal(t_zero, c["traj60"]), "zeroing the token's weight changes O's plan"


def test_aux_input_composition_and_the_frame_mask(world):
    b, J = make_batch(world)
    last = torch.full((16,), b["feats"].shape[1] - 1, dtype=torch.long)
    for kw in (dict(level_head="on", w_level=0.5), dict(lead_head="on", w_lead=0.5), dict(level_head="on", w_level=0.5, lead_head="on", w_lead=0.5)):
        m = build(J, aux_cond="token", **kw).eval()
        with torch.no_grad():
            adapt = m._encode(b)
            x = m._aux_in(b, adapt, last)
            z = torch.zeros(16)
            lv = m._level(b, adapt, last)[1] if m.lvl_head is not None else z
            if m.lead_head is not None:
                lg, gh, li = m._lead(b, adapt, last)
                ld = torch.stack([torch.sigmoid(lg), gh], -1)
                assert torch.equal(li, ld * b["frame_ok"][:, -1].float()[:, None]) and (gh >= 0).all() and (gh <= 1).all()
            else:
                ld = torch.zeros(16, 2)
        assert x.shape == (16, 3) and not x.requires_grad
        assert torch.equal(x[:, 0], lv) and torch.equal(x[:, 1:], ld * (b["frame_ok"][:, -1].float()[:, None] if m.lead_head is not None else 1.0))
        if m.lvl_head is None:
            assert (x[:, 0] == 0).all(), "no level head: the level input is exactly 0"
        if m.lead_head is None:
            assert (x[:, 1:] == 0).all(), "no lead head: the lead inputs are exactly 0"
    fo = {**b, "frame_ok": b["frame_ok"].clone()}
    fo["frame_ok"][:, -1] = False
    m = build(J, aux_cond="token", level_head="on", w_level=0.5, lead_head="on", w_lead=0.5).eval()
    with torch.no_grad():
        assert (m._aux_in(fo, m._encode(fo), last) == 0).all(), "no real NOW frame: O is handed 0"


def test_the_lead_output_reaches_O_through_run_o_whose_signature_is_unchanged(world):
    b, J = make_batch(world)
    on = wake_lead(build(J, lead_head="on", w_lead=0.5)).eval()
    with torch.no_grad():
        adapt = on._encode(b)
        last = torch.full((16,), b["feats"].shape[1] - 1, dtype=torch.long)
        ck = on.ck_nom[None].expand(16, -1, -1)
        ok = torch.ones(16, 3)
        code = torch.zeros(16, M.CODE_DIM)
        logit, gap_hat, lead_in = on._lead(b, adapt, last)
        t_given = on.run_o(b, adapt, last, ck, ok, code)[0]                         # run_o keeps its six-argument signature and computes the lead outputs itself
        saved = (on.e_lead.weight.clone(), on.e_lead.bias.clone())
        on.e_lead.weight.zero_()
        on.e_lead.bias.zero_()
        t_zero = on.run_o(b, adapt, last, ck, ok, code)[0]
        on.e_lead.weight.copy_(saved[0])
        on.e_lead.bias.copy_(saved[1])
    assert not torch.equal(t_given, t_zero)
    okf = b["frame_ok"][:, -1].float()[:, None]
    assert torch.equal(lead_in, torch.stack([torch.sigmoid(logit), gap_hat], -1) * okf) and not lead_in.requires_grad


# ------------------------------------------------------------------------------------------------ 4. the term, the masks, the weight
def test_the_lead_term_against_a_hand_computation_and_the_masks(world):
    b, J = make_batch(world)
    LL = LLof(world, J)
    on = build(J, lead_head="on", w_lead=0.5)
    lab = with_lead(b, LL)
    with torch.no_grad():
        out = run_loss(on, lab)
    lg, gh = out["lead_logit"].double().numpy(), out["lead_gap"].double().numpy()
    has, near, fok = lab["lead_has"].numpy(), lab["lead_near"].numpy(), b["frame_ok"][:, -1].numpy()
    lhas, lnear = (has & fok).astype(float), (near & has & fok).astype(float)
    bce = np.maximum(lg, 0) - lg * near.astype(float) + np.log1p(np.exp(-np.abs(lg)))
    err = np.abs(gh - lab["lead_y"].double().numpy())
    hub = np.where(err <= M.LEAD_HUBER_DELTA, 0.5 * err ** 2, M.LEAD_HUBER_DELTA * (err - 0.5 * M.LEAD_HUBER_DELTA))
    t_bce, t_hub = float((bce * lhas).sum() / max(lhas.sum(), 1)), float((hub * lnear).sum() / max(lnear.sum(), 1))
    T = out["terms"]
    assert float(T["lead_bce"]) == pytest.approx(t_bce, rel=1e-5) and float(T["lead_huber"]) == pytest.approx(t_hub, rel=1e-4) and float(T["lead"]) == pytest.approx(t_bce + t_hub, rel=1e-4)
    assert lnear.sum() > 0 and lhas.sum() > lnear.sum() > 0 and (gh >= 0).all() and (gh <= 1).all()
    for k in ("lead_y", "lead_has", "lead_near"):
        with pytest.raises(ValueError, match="lead_y"):
            on.loss({kk: v for kk, v in lab.items() if kk != k}, p_gt=0.5, gen=torch.Generator().manual_seed(1))             # a lever-ON loss on a batch without a label is refused, never silently skipped
    with pytest.raises(ValueError, match="lead_y"):
        on.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(1))
    with torch.no_grad():
        o_all0 = run_loss(on, {**lab, "lead_has": torch.zeros_like(lab["lead_has"])})
        o_far = run_loss(on, {**lab, "lead_near": torch.zeros_like(lab["lead_near"])})
        fo = {**lab, "frame_ok": b["frame_ok"].clone()}
        fo["frame_ok"][:, -1] = False
        o_nof = run_loss(on, fo)
    assert float(o_all0["terms"]["lead_bce"]) == 0.0 and float(o_all0["terms"]["lead_huber"]) == 0.0 and float(o_all0["terms"]["lead"]) == 0.0, "no labelled frame: both parts are exactly 0.0"
    assert float(o_far["terms"]["lead_huber"]) == 0.0 and float(o_far["terms"]["lead_bce"]) > 0.0, "no frame with a lead within 40 m: the Huber is 0.0, the BCE (the negatives) remains"
    assert float(o_nof["terms"]["lead"]) == 0.0, "no real NOW frame anywhere: the term is exactly 0.0"
    assert float(T["total"].detach()) == pytest.approx(float(o_all0["terms"]["total"].detach()) + 0.5 * float(T["lead"].detach()), rel=1e-5)


def test_the_w_lead_for_10_percent_formula_gives_a_tenth_of_the_total_at_step_0(world):
    b, J = make_batch(world)
    LL = LLof(world, J)
    for kw in (dict(), dict(level_head="on", w_level=0.2027), dict(aux_cond="token")):
        on = build(J, lead_head="on", w_lead=1.0, wake=False, **kw)
        lab = with_both(b, LL)
        pr = TR.lead_head_probe(on, lab, on.cfg)
        W = pr["w_lead_for_10pct_share"]
        assert W > 0 and pr["step0_lead_share"] == pytest.approx(pr["step0_lead_term"] / pr["step0_total"], rel=1e-9)
        on2 = build(J, lead_head="on", w_lead=W, wake=False, **kw)
        pr2 = TR.lead_head_probe(on2, lab, on2.cfg)
        assert pr2["step0_lead_share"] == pytest.approx(0.1, rel=1e-4), "the formula (0.1/0.9) x rest / term puts the lead term at a tenth of the total at step 0"
        assert pr2["w_lead_for_10pct_share"] == pytest.approx(W, rel=1e-5), "and it does not depend on the weight it was read at"
        assert ("step0_level_share" in pr) == ("level_head" in kw)


def test_labels_are_looked_up_before_the_ego_stages_in_the_loop():
    src = TRAIN_P.read_text(encoding="utf-8")
    i_lab = src.index("lead_lab = lead_stage(b, LL)")
    i_ego, i_rcv, i_att, i_dev = (src.index(x, i_lab) for x in ("b = ego.apply(b, step)", "b = rcv.apply(b, step)", 'b["lead_y"], b["lead_has"], b["lead_near"] = lead_lab', "b = to_dev(b, dev)"))
    assert i_lab < i_ego < i_rcv < i_att < i_dev, "labels from the LOGGED clip index and step, attached after every stage that rebuilds the batch, before the device move"


# ------------------------------------------------------------------------------------------------ 5. admissibility: vision only
def test_vision_only_guards_pass_on_the_built_head_and_the_aux_token(world):
    b, J = make_batch(world)
    LL = LLof(world, J)
    on = build(J, lead_head="on", w_lead=0.5)
    rec = M.assert_lead_head_vision_only(on, with_lead(b, LL))
    assert rec["keys_read"] in ([], ["frame_ok"]) and M.lead_head_source_problems() == [] and M.aux_source_problems() == [] and rec["vision_sensitivity"] > 0
    tok = build(J, aux_cond="token", level_head="on", w_level=0.5, lead_head="on", w_lead=0.5)
    assert M.assert_lead_head_vision_only(tok, with_both(b, LL))["source_problems"] == []
    ra = M.assert_aux_vision_only(tok, with_both(b, LL))
    assert ra["keys_read"] in ([], ["frame_ok"]) and ra["source_problems"] == []
    on.eval()
    with torch.no_grad():
        info = M.assert_no_situation_classifier_input(on, b)
    assert not {"lead_y", "lead_has", "lead_near"} & set(info["read"]), "inference reads no label"
    with pytest.raises(AssertionError, match="no lead head"):
        M.assert_lead_head_vision_only(build(J), with_lead(b, LL))
    with pytest.raises(AssertionError, match="no aux token"):
        M.assert_aux_vision_only(on, with_lead(b, LL))


def test_mutation_head_fed_ego_goes_red(world):
    b, J = make_batch(world)
    LL = LLof(world, J)

    class EgoHead(M.V8FMini):                                 # the head reads the ego p10 features instead of the vision tokens
        def _lead(self, b, adapt, last):
            o = self.lead_head(self._ego(self._at(b["p10f"], last))).float()
            return o[:, 0], torch.sigmoid(o[:, 1]), torch.stack([torch.sigmoid(o[:, 0]), torch.sigmoid(o[:, 1])], -1)
    m = build(J, cls=EgoHead, lead_head="on", w_lead=0.5)
    with pytest.raises(AssertionError, match="non-vision"):
        M.assert_lead_head_vision_only(m, with_lead(b, LL))   # the STATIC half names p10f / _ego


def test_mutation_head_fed_ego_through_a_source_that_evades_the_scan_goes_red(world):
    b, J = make_batch(world)
    LL = LLof(world, J)

    class Sneaky(M.V8FMini):                                  # same wiring, spelled so no forbidden identifier appears: the RUNTIME halves must catch it
        def _lead(self, b, adapt, last):
            k = "p10" + "f"
            ego = getattr(self, "_e" + "go")(self._at(b[k], last))
            o = self.lead_head(adapt.mean((1, 2)) + ego).float()
            return o[:, 0], torch.sigmoid(o[:, 1]), torch.stack([torch.sigmoid(o[:, 0]), torch.sigmoid(o[:, 1])], -1)
    m = build(J, cls=Sneaky, lead_head="on", w_lead=0.5)
    assert M.lead_head_source_problems(Sneaky._lead) == [], "the static scan is evaded on purpose"
    with pytest.raises(AssertionError, match="READS batch keys"):
        M.assert_lead_head_vision_only(m, with_lead(b, LL))


def test_mutation_aux_token_fed_ego_goes_red(world):
    b, J = make_batch(world)
    LL = LLof(world, J)

    class EgoAux(M.V8FMini):
        def _aux_in(self, b, adapt, last):
            return super()._aux_in(b, adapt, last) + self._at(b["p10f"], last)[:, :3]
    m = build(J, cls=EgoAux, aux_cond="token", lead_head="on", w_lead=0.5)
    with pytest.raises(AssertionError, match="non-vision"):
        M.assert_aux_vision_only(m, with_lead(b, LL))

    class SneakyAux(M.V8FMini):
        def _aux_in(self, b, adapt, last):
            k = "p10" + "f"
            return super()._aux_in(b, adapt, last) + self._at(b[k], last)[:, :3]
    m2 = build(J, cls=SneakyAux, aux_cond="token", lead_head="on", w_lead=0.5)
    assert M.aux_source_problems(SneakyAux._aux_in) == []
    with pytest.raises(AssertionError, match="READS batch keys"):
        M.assert_aux_vision_only(m2, with_lead(b, LL))


# ------------------------------------------------------------------------------------------------ 6. G-LIVE
@pytest.mark.parametrize("kw", [dict(lead_head="on", w_lead=0.5), dict(lead_head="on", w_lead=0.5, level_head="on", w_level=0.5), dict(lead_head="on", w_lead=0.5, aux_cond="token"),
                                dict(lead_head="on", w_lead=0.5, level_head="on", w_level=0.5, aux_cond="token")])
def test_live_check_and_the_lead_probe_pass(world, kw):
    b, J = make_batch(world)
    LL = LLof(world, J)
    on = build(J, wake=False, **kw)
    lab = with_both(b, LL)
    res = TR.live_check(on, lab, on.cfg)
    assert res["pass"] and res["control_prior_grad_sum"] == 0.0 and "lead" in res["terms"] and not res["unexpected_dead"]
    pr = TR.lead_head_probe(on, lab, on.cfg)
    assert pr["pass"] is True and pr["aux_cond"] == on.cfg.aux_cond
    assert pr["head_grad_at_w0_detach_control"] == 0.0 and pr["control_allfalse_term"] == 0.0 and pr["control_allfalse_total_diff"] == 0.0 and pr["control_prior_grad_diff"] == 0.0
    assert pr["contribution"] == pytest.approx(pr["w_lead_x_term"], rel=1e-3) and pr["head_grad_diff"] > 0.0 and pr["plan_moves_with_token"] > 0.0
    assert pr["head_grad_diff_presence_row"] > 0.0 and pr["head_grad_diff_gap_row"] > 0.0 and pr["w_lead_for_10pct_share"] > 0.0 and pr["probe_n_near"] > 0
    assert ("e_aux_lead_cols_grad_at_w0" in pr) == (on.cfg.aux_cond == "token") and ("e_lead_grad_at_w0" in pr) == (on.cfg.aux_cond == "add")
    assert (pr["aux_vision_only"] is not None) == (on.cfg.aux_cond == "token")
    if "level_head" in kw:
        lp = TR.level_head_probe(on, lab, on.cfg)             # the level probe still passes next to the lead head, and under the token wiring (e_level is not built there)
        assert lp["pass"] is True and (("e_aux_level_col_grad_at_w0" in lp) == (on.cfg.aux_cond == "token")) and lp["plan_moves_with_token"] > 0.0 and lp["head_grad_at_w0_detach_control"] == 0.0
    off = build(J)
    assert TR.lead_head_probe(off, b, off.cfg) == {"lead": "off", "pass": True, "skipped": "--lead-head not named: nothing is built (the default run is the pre-change run)"}


def test_the_level_probe_passes_under_token_with_the_level_head_alone(world):
    b, J = make_batch(world)
    LL = LLof(world, J)
    m = build(J, wake=False, level_head="on", w_level=0.5, aux_cond="token")
    lab = with_both(b, LL)
    assert TR.live_check(m, lab, m.cfg)["pass"]
    lp = TR.level_head_probe(m, lab, m.cfg)
    assert lp["pass"] and lp["e_aux_level_col_grad_at_w0"] > 0.0 and lp["aux_vision_only"]["source_problems"] == [] and "e_level_grad_at_w0" not in lp


def test_mutation_head_dropped_from_Os_cond_goes_red(world):
    b, J = make_batch(world)
    LL = LLof(world, J)

    class NoLead(M.V8FMini):
        def run_o(self, *a, **k):
            keep, self.e_lead = self.e_lead, None             # O's cond is built without the lead term
            try:
                return super().run_o(*a, **k)
            finally:
                self.e_lead = keep

    class NoAux(M.V8FMini):
        def run_o(self, *a, **k):
            keep, self.e_aux = self.e_aux, None               # O's cond is built without the extra token
            try:
                return super().run_o(*a, **k)
            finally:
                self.e_aux = keep
    m = build(J, cls=NoLead, lead_head="on", w_lead=0.5, wake=False)
    with pytest.raises(AssertionError, match="does not reach O"):
        TR.lead_head_probe(m, with_lead(b, LL), m.cfg)
    m = build(J, cls=NoAux, lead_head="on", w_lead=0.5, aux_cond="token", wake=False)
    with pytest.raises(AssertionError, match="does not reach O"):
        TR.lead_head_probe(m, with_lead(b, LL), m.cfg)
    m = build(J, cls=NoAux, level_head="on", w_level=0.5, aux_cond="token", wake=False)
    with pytest.raises(AssertionError, match="does not reach O"):
        TR.level_head_probe(m, with_both(b, LL), m.cfg)


def test_mutation_total_line_deleted_goes_red(world):
    b, J = make_batch(world)
    LL = LLof(world, J)

    class NoLine(M.V8FMini):
        def loss(self, *a, **k):
            out = super().loss(*a, **k)
            out["terms"]["total"] = out["terms"]["total"] - self.cfg.w_lead * out["terms"]["lead"]         # the `total = total + w_lead * lead` line is gone
            return out
    m = build(J, cls=NoLine, lead_head="on", w_lead=0.5, wake=False)
    with pytest.raises(AssertionError, match="does not reach the loss"):
        TR.lead_head_probe(m, with_lead(b, LL), m.cfg)


def test_mutation_detach_removed_goes_red(world):
    b, J = make_batch(world)
    LL = LLof(world, J)

    class NoDetach(M.V8FMini):
        def _lead(self, b, adapt, last):
            logit, gap_hat, _ = super()._lead(b, adapt, last)
            return logit, gap_hat, torch.stack([torch.sigmoid(logit), gap_hat], -1)                        # O reads the head's outputs themselves: the plan's loss reaches the head
    m = build(J, cls=NoDetach, lead_head="on", w_lead=0.5, wake=False)
    with pytest.raises(AssertionError, match="not EXACTLY 0.0"):
        TR.lead_head_probe(m, with_lead(b, LL), m.cfg)
    m = build(J, cls=NoDetach, lead_head="on", w_lead=0.5, aux_cond="token", wake=False)
    with pytest.raises(AssertionError, match="not EXACTLY 0.0"):
        TR.lead_head_probe(m, with_lead(b, LL), m.cfg)


def test_mutation_the_label_masks_ignored_goes_red(world):
    b, J = make_batch(world)
    LL = LLof(world, J)

    class NoMask(M.V8FMini):
        def loss(self, batch, *a, **k):
            return super().loss({**batch, "lead_has": torch.ones_like(batch["lead_has"])}, *a, **k)
    m = build(J, cls=NoMask, lead_head="on", w_lead=0.5, wake=False)
    with pytest.raises(AssertionError, match="ALL-FALSE"):
        TR.lead_head_probe(m, with_lead(b, LL), m.cfg)


def test_mutation_the_huber_part_dropped_goes_red(world):
    b, J = make_batch(world)
    LL = LLof(world, J)

    class PresenceOnly(M.V8FMini):
        def loss(self, *a, **k):
            out = super().loss(*a, **k)
            T = out["terms"]
            T["total"] = T["total"] - self.cfg.w_lead * T["lead_huber"]                                      # the gap term never reaches the loss
            T["lead"] = T["lead_bce"]
            return out
    m = build(J, cls=PresenceOnly, lead_head="on", w_lead=0.5, wake=False)
    with pytest.raises(AssertionError, match="one output of the head only"):
        TR.lead_head_probe(m, with_lead(b, LL), m.cfg)


def test_the_probe_refuses_a_batch_it_cannot_read(world):
    b, J = make_batch(world)
    LL = LLof(world, J)
    m = build(J, wake=False, lead_head="on", w_lead=0.5)
    lab = with_lead(b, LL)
    with pytest.raises(AssertionError, match="cannot see the lead term"):
        TR.lead_head_probe(m, {**lab, "lead_near": torch.zeros_like(lab["lead_near"])}, m.cfg)
    with pytest.raises(AssertionError, match="carries no lead_y"):
        TR.lead_head_probe(m, b, m.cfg)


# ------------------------------------------------------------------------------------------------ 7. the gate
def test_every_flag_has_an_entry():
    assert TR.FLAG_CHECKS["lead_head"] == "lever:lead_head" and TR.FLAG_CHECKS["w_lead"] == "cfg:w_lead" and TR.FLAG_CHECKS["aux_cond"] == "lever:aux_cond"
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert {"lead_head", "w_lead", "aux_cond"} <= dests and dests == set(TR.FLAG_CHECKS)
    p = TR.build_parser()
    assert (p.get_default("lead_head"), p.get_default("w_lead"), p.get_default("aux_cond")) == ("off", 0.0, "add")


def test_the_gate_refuses_lies(world):
    _, J = make_batch(world)
    LL = LLof(world, J)
    on = ["--lead-head", "on", "--w-lead", "0.7"]
    a = args(world, on)
    bad, model = gate(a, J, LL)
    assert bad == [] and model.lead_head is not None
    for over, want in ((dict(lead_head="off", w_lead=0.0), "declared on but built off"),                     # declared on, built off
                       ):
        bad, _ = gate(a, J, LL, **over)
        assert any(want in m for m in bad), bad
    bad, _ = gate(args(world, ["--lead-head", "on", "--w-lead", "0"]), J, LL)                                  # a head with no loss
    assert any("a head with no loss" in m for m in bad)
    bad, _ = gate(args(world, ["--w-lead", "0.5"]), J, LL)                                                     # a weight with no head
    assert any("a loss weight with no head" in m for m in bad)
    bad, _ = gate(args(world), J, LL, lead_head="on", w_lead=0.5)                                              # declared off, a head built
    assert any("lead_head off declared" in m for m in bad)
    cfg, model, opt, info = built(a, J, LL)                                                                    # a non-zero e_lead at a fresh start
    wake_lead(model)
    assert any("zero-initialised" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    with pytest.raises(ValueError, match="w_lead"):
        TR.make_config(args(world, ["--lead-head", "on", "--w-lead", "-1"]))
    with pytest.raises(ValueError, match="lead_head"):
        M.V8FMiniConfig(**TINY, lead_head="maybe")
    with pytest.raises(ValueError, match="aux_cond"):
        M.V8FMiniConfig(**TINY, aux_cond="append")
    bad, _ = gate(args(world), J, None)
    assert bad == [], "the default argv still passes"
    # the wiring: token with no head, the wrong wiring built, additive paths built under token
    bad, _ = gate(args(world, ["--aux-cond", "token"]), J, None)
    assert any("both off" in m for m in bad) and any("lever aux_cond: declared token but built add" in m for m in bad), bad
    bad, _ = gate(args(world, [*on, "--aux-cond", "token"]), J, LL, aux_cond="add")
    assert any("lever aux_cond: declared token but built add" in m for m in bad) and any("cfg.aux_cond" in m for m in bad)
    bad, _ = gate(args(world, on), J, LL, aux_cond="token")
    assert any("lever aux_cond: declared add but built token" in m for m in bad) and any("aux token was built" in m for m in bad)
    bad, _ = gate(args(world, [*on, "--aux-cond", "token"]), J, LL)
    assert bad == []
    cfg, model, opt, info = built(args(world, [*on, "--aux-cond", "token"]), J, LL)
    model.e_lead = torch.nn.Linear(2, 32)                                                                      # an additive path left alive under token
    assert any("REPLACES both additive paths" in m for m in TR.declared_vs_built(args(world, [*on, "--aux-cond", "token"]), cfg, model, opt, info))


def test_the_gate_refuses_a_lead_head_with_no_usable_labels(world):
    _, J = make_batch(world)
    LL = LLof(world, J)
    a = args(world, ["--lead-head", "on", "--w-lead", "0.7"])
    bad, _ = gate(a, J, LL, info_over={"lead_labels": None})
    assert any("no lead labels were built" in m for m in bad)
    for over, want in ((dict(path="elsewhere.npz"), "declared"), (dict(sha256="short"), "sha256"), (dict(n_near=0), "has nothing to learn from"), (dict(n_has=0), "has nothing to learn from"),
                       (dict(n_clips_with_labels=1), "the wrong file")):
        bad, _ = gate(a, J, LL, info_over={"lead_labels": lead_info(a, LL, **over)})
        assert any(want in m for m in bad), (over, bad)


def test_the_launch_refuses_through_main(world, tmp_path):
    refused(world, ["--lead-head", "on"], tmp_path, "a head with no loss")
    refused(world, ["--w-lead", "0.5"], tmp_path, "a loss weight with no head")
    refused(world, ["--aux-cond", "token"], tmp_path, "both off")
    with pytest.raises(SystemExit) as e:
        TR.main(["--out", str(tmp_path / "nolabels"), *RUN, "--lead-train", str(tmp_path / "does_not_exist.npz"), "--lead-head", "on", "--w-lead", "0.5"])
    assert "ZZDVB-REFUSEDZZ" in str(e.value) and "does not exist" in str(e.value)


# ------------------------------------------------------------------------------------------------ 8. init from an older checkpoint
@pytest.fixture(scope="module")
def trained(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("leadtrain")
    R = _run_argv(world)
    off = root / "off"
    assert TR.main(["--out", str(off), *R, "--steps", "2"]) == 0
    lvl = root / "lvl"
    assert TR.main(["--out", str(lvl), *R, "--steps", "2", "--level-head", "on", "--w-level", "0.5", "--init-ckpt", str(off / "ckpt_last.pt"), "--lr", "1e-3"]) == 0
    add = root / "add"                                                                    # old (no head) checkpoint -> lead on, add
    assert TR.main(["--out", str(add), *R, "--steps", "3", "--lead-head", "on", "--w-lead", "0.5", "--init-ckpt", str(off / "ckpt_last.pt"), "--lr", "1e-3"]) == 0
    both = root / "both"                                                                  # level-on checkpoint (the CLR) -> both heads, add: only the lead keys are fresh
    assert TR.main(["--out", str(both), *R, "--steps", "3", "--level-head", "on", "--w-level", "0.5", "--lead-head", "on", "--w-lead", "0.5", "--init-ckpt", str(lvl / "ckpt_last.pt"), "--lr", "1e-3"]) == 0
    tok = root / "tok"                                                                    # old checkpoint -> both heads, token
    assert TR.main(["--out", str(tok), *R, "--steps", "3", "--level-head", "on", "--w-level", "0.5", "--lead-head", "on", "--w-lead", "0.5", "--aux-cond", "token", "--init-ckpt", str(off / "ckpt_last.pt"), "--lr", "1e-3"]) == 0
    return types.SimpleNamespace(root=root, off=off, lvl=lvl, add=add, both=both, tok=tok)


def test_init_from_an_older_checkpoint_loads_every_old_key_and_only_the_declared_ones_are_fresh(trained):
    old = torch.load(trained.off / "ckpt_last.pt", map_location="cpu", weights_only=False)
    for run, new_keys, fresh in ((trained.add, LEAD_ADD, LEAD_ADD), (trained.tok, BOTH_HEADS_TOKEN, BOTH_HEADS_TOKEN), (trained.both, LEVEL_ADD | LEAD_ADD, LEAD_ADD)):
        sc = _sc(run)
        rec = sc["levers"]["init_ckpt"]
        assert rec["applied"] is True and rec["n_mismatch_after_load"] == 0 and set(rec["fresh_keys"]) == fresh and set(rec["declared_new_keys"]) == new_keys, run.name
        assert sc["declared_vs_built"] == "PASS" and sc["lead_head_probe"]["pass"] is True and sc["levers"]["lead_head"] == "on"
    assert _sc(trained.add)["levers"]["init_ckpt"]["n_tensors"] == len(old["model"]) and not (set(old["model"]) & (LEAD_ADD | BOTH_HEADS_TOKEN))
    lvl = torch.load(trained.lvl / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert _sc(trained.both)["levers"]["init_ckpt"]["n_tensors"] == len(lvl["model"]), "the level-on checkpoint's own (trained) level keys load; only the lead keys are fresh"


def test_a_lever_on_checkpoint_into_a_narrower_argv_is_refused_and_the_old_one_into_an_off_argv_is_the_strict_path(world, trained, tmp_path):
    R = _run_argv(world)
    for ck, argv, what, msg in ((trained.add, [], "lead_head", "Unexpected key"),                                                       # no lever declared: the strict load, verbatim
                                (trained.tok, ["--lead-head", "on", "--w-lead", "0.5"], "e_aux", "carries keys the built model does not have"),
                                (trained.lvl, ["--lead-head", "on", "--w-lead", "0.5", "--aux-cond", "token", "--level-head", "on", "--w-level", "0.5"], "e_level", "carries keys the built model does not have")):
        with pytest.raises(SystemExit) as e:
            TR.main(["--out", str(tmp_path / f"x_{ck.name}"), *R, "--steps", "1", *argv, "--init-ckpt", str(ck / "ckpt_last.pt")])
        assert "ZZDVB-REFUSEDZZ" in str(e.value) and msg in str(e.value) and what in str(e.value), (ck.name, str(e.value)[:300])
    assert TR.main(["--out", str(tmp_path / "y"), *R, "--steps", "1", "--init-ckpt", str(trained.off / "ckpt_last.pt")]) == 0
    r = _sc(tmp_path / "y")["levers"]["init_ckpt"]
    assert "fresh_keys" not in r and "declared_new_keys" not in r and r["n_mismatch_after_load"] == 0
    sc = _sc(tmp_path / "y")
    assert "lead_head_probe" not in sc and "lead_labels" not in sc and not {"lead_head", "w_lead", "aux_cond"} & set(sc["levers"]) and not {"lead_head", "w_lead", "aux_cond", "lead_labels"} & set(_sm(tmp_path / "y"))


def test_init_ckpt_refuses_a_fresh_key_that_is_not_the_declared_levers(world, trained, tmp_path):
    blob = torch.load(trained.off / "ckpt_last.pt", map_location="cpu", weights_only=False)
    victim = next(k for k in blob["model"] if k.startswith("read_o.q"))
    del blob["model"][victim]                                                         # a checkpoint that lacks an OLD key: not the lever's to leave fresh
    bad = tmp_path / "missing_old_key.pt"
    torch.save(blob, bad)
    with pytest.raises(SystemExit) as e:
        TR.main(["--out", str(tmp_path / "z"), *_run_argv(world), "--steps", "1", "--lead-head", "on", "--w-lead", "0.5", "--init-ckpt", str(bad)])
    assert "ZZDVB-REFUSEDZZ" in str(e.value) and victim in str(e.value)


def test_load_init_ckpt_itself_and_the_gate_record(world, trained, tmp_path):
    """The two guards are tested one at a time (the end-to-end refusal above would hide a deleted one behind the other)."""
    ck = torch.load(trained.off / "ckpt_last.pt", map_location="cpu", weights_only=False)
    victim = next(k for k in ck["model"] if k.startswith("read_o.q"))
    for kw, fresh in ((dict(lead_head="on", w_lead=0.5), LEAD_ADD), (dict(lead_head="on", w_lead=0.5, aux_cond="token"), LEAD_TOKEN)):
        cfg = M.V8FMiniConfig(**{**ck["cfg"], **kw})
        mk = lambda: M.V8FMini(cfg, M.P10Prior.synthetic(5), ck["cons_mu"], ck["cons_sd"])      # noqa: E731
        badf = tmp_path / "lacks_an_old_key.pt"
        torch.save({**ck, "model": {k: v for k, v in ck["model"].items() if k != victim}}, badf)
        with pytest.raises(ValueError, match="not the declared lever's"):
            TR.load_init_ckpt(str(badf), mk(), cfg)
        good = TR.load_init_ckpt(str(trained.off / "ckpt_last.pt"), mk(), cfg)
        assert set(good["fresh_keys"]) == fresh == set(good["declared_new_keys"])
    a = args(world, ["--lead-head", "on", "--w-lead", "0.5", "--init-ckpt", "x.pt"])
    rec = {"path": str(Path("x.pt").resolve()), "sha256": "0" * 64, "applied": True, "n_mismatch_after_load": 0, "n_tensors": 5, "skipped": None, "fresh_keys": sorted(LEAD_ADD), "declared_new_keys": sorted(LEAD_ADD)}
    assert TR.bad_init_ckpt(a, {"init_ckpt": rec}) == []
    assert any("left fresh" in m for m in TR.bad_init_ckpt(a, {"init_ckpt": {**rec, "fresh_keys": sorted(LEAD_ADD) + [victim]}}))
    assert any("declared new keys" in m for m in TR.bad_init_ckpt(a, {"init_ckpt": {**rec, "declared_new_keys": []}}))
    assert any("declared new keys" in m for m in TR.bad_init_ckpt(args(world, ["--init-ckpt", "x.pt"]), {"init_ckpt": rec}))
    assert TR.bad_init_ckpt(args(world, ["--level-head", "on", "--w-level", "0.5", "--init-ckpt", "x.pt"]), {"init_ckpt": rec}) == [], "the level head's own declaration of new keys is unchanged"


@pytest.mark.skipif(not C0_INIT.exists(), reason="the C0 init checkpoint is on the dev box only")
@pytest.mark.parametrize("kw,fresh", [(dict(lead_head="on", w_lead=1.0), LEAD_ADD), (dict(lead_head="on", w_lead=1.0, level_head="on", w_level=1.0, aux_cond="token"), BOTH_HEADS_TOKEN)])
def test_real_c0_init_checkpoint_loads_with_only_the_declared_keys_fresh(world, kw, fresh):
    ck = torch.load(C0_INIT, map_location="cpu", weights_only=False)
    cfg_d = {**ck["cfg"]}
    b, J = make_batch(world, hmax=cfg_d["hmax"])
    LL = LLof(world, J)
    mu, sd = M.cons_stats(J)

    def mk(**k):
        torch.manual_seed(0)
        return M.V8FMini(M.V8FMiniConfig(**{**cfg_d, **k}), M.P10Prior.synthetic(0), mu, sd)
    off, on = mk(), mk(**kw)
    off.load_state_dict(ck["model"], strict=True)
    rec = TR.load_init_ckpt(str(C0_INIT), on, on.cfg)
    assert rec["n_mismatch_after_load"] == 0 and set(rec["fresh_keys"]) == fresh and rec["n_tensors"] == len(ck["model"])
    sd_on = on.state_dict()
    assert all(torch.equal(sd_on[k], v) for k, v in ck["model"].items()), "every old key bit-identical"
    off.eval()
    on.eval()
    bb = with_both(b, LL)
    with torch.no_grad():
        a, c = off.forward_infer(bb), on.forward_infer(bb)
    if on.cfg.aux_cond == "add":
        assert all(torch.equal(a[k], c[k]) for k in a if torch.is_tensor(a[k])), "add: the step-0 forward of the lever-ON model on the real C0 weights equals the C0 model's, EXACTLY"
    else:
        assert not torch.equal(a["traj60"], c["traj60"]), "token: differs by construction"


# ------------------------------------------------------------------------------------------------ 9. end to end
def test_training_moves_the_heads_and_the_checkpoints_rebuild_the_identical_models(trained):
    for run, keys, token in ((trained.add, LEAD_ADD, False), (trained.both, LEVEL_ADD | LEAD_ADD, False), (trained.tok, BOTH_HEADS_TOKEN, True)):
        log = _lg(run)
        assert len(log) == 3 and all(math.isfinite(r["lead"]) and r["lead"] > 0.0 and math.isfinite(r["lead_bce"]) and math.isfinite(r["lead_huber"]) and math.isfinite(r["total"]) for r in log), run.name
        assert all(abs(r["lead"] - r["lead_bce"] - r["lead_huber"]) < 1e-5 for r in log)
        ck = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
        assert ck["cfg"]["lead_head"] == "on" and ck["cfg"]["w_lead"] == 0.5 and ck["cfg"]["aux_cond"] == ("token" if token else "add") and keys <= set(ck["model"])
        assert float(ck["model"]["e_aux.weight" if token else "e_lead.weight"].abs().sum()) > 0.0, "the path left its init: the loss reached it"
        if not token:
            assert float(ck["model"]["e_lead.bias"].abs().sum()) > 0.0 and float(ck["model"]["e_lead.weight"].abs().sum()) > 0.0
        cfg = M.V8FMiniConfig(**ck["cfg"])                                             # G-EVAL: the evaluators build V8FMiniConfig(**ck["cfg"])
        model = M.V8FMini(cfg, M.P10Prior.synthetic(5), ck["cons_mu"], ck["cons_sd"])
        model.load_state_dict(ck["model"], strict=True)
        assert model.lead_head is not None and (model.e_aux is not None) == token and (model.e_lead is not None) == (not token)
        sm, sc = _sm(run), _sc(run)
        assert sm["lead_head"] == "on" and sm["w_lead"] == 0.5 and sm["aux_cond"] == ("token" if token else "add") and sm["done"] is True
        assert sm["lead_labels"] == sc["lead_labels"] and sc["lead_labels"]["n_near"] > 0 and len(sc["lead_labels"]["sha256"]) == 64 and sc["lead_labels"]["n_clips_with_labels"] == 6
        pr = sc["lead_head_probe"]
        assert pr["w_lead_for_10pct_share"] > 0 and pr["step0_lead_share"] > 0 and pr["head_grad_at_w0_detach_control"] == 0.0
        assert ("level_head_probe" in sc) == (run is not trained.add) and sc["levers"]["aux_cond"] == ("token" if token else "add")
    assert _sm(trained.tok)["level_head"] == "on" and _sc(trained.tok)["level_head_probe"]["e_aux_level_col_grad_at_w0"] > 0.0


def test_flag_off_records_keep_their_key_sets(trained):
    sc, sm = _sc(trained.off), _sm(trained.off)
    assert "lead_head_probe" not in sc and "lead_labels" not in sc and not {"lead_head", "w_lead", "aux_cond", "aux_vision_only_source"} & set(sc["levers"])
    assert not {"lead_head", "w_lead", "aux_cond", "lead_labels"} & set(sm) and not {"lead", "lead_bce", "lead_huber"} & set(_lg(trained.off)[0])
    cfg = torch.load(trained.off / "ckpt_last.pt", map_location="cpu", weights_only=False)["cfg"]
    assert cfg["lead_head"] == "off" and cfg["w_lead"] == 0.0 and cfg["aux_cond"] == "add", "the checkpoint's cfg carries the three fields at their defaults"
    lvl = _sm(trained.lvl)
    assert "aux_cond" not in lvl and "lead_head" not in lvl and lvl["level_head"] == "on", "a level-only add arm keeps today's record"
