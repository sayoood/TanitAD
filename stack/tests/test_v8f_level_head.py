"""PREREG_V8F_LEVEL: ``--level-head {off,on}`` + ``--w-level`` in ``stack/tanitad/models/v8f_mini.py`` and ``stack/scripts/v8f_mini_train.py`` -- lever (a) of DECISION_K2, the speed LEVEL read from VISION.

Default OFF. THIS file proves the lever is REACHABLE from the real caller (CLAUDE.md 2026-09-10/11: "built, tested, and unreachable from its caller" is a class), hop by hop:

  argv --level-head/--w-level -> make_config -> cfg -> the BUILT model (lvl_head + e_level) -> declared_vs_built (bad_level, FLAG_CHECKS) -> level_head_probe (G-LIVE on the loop's own label call level_stage)
  -> ``lvl_lab = level_stage(b)`` BEFORE the ego stages, ``b["level_y"], b["level_ok"]`` AFTER them -> the batch the LOSS receives -> V8FMini.loss (Huber term + ONE total line)
  -> O's cond (e_level(v_hat.detach() / 10) ADDED to the ego token) -> the plan -> summary.json / selfcheck.json / train_log.jsonl / the checkpoint.

  1. OFF == TIP: a frozen snapshot of the tip's model module (sha256 pinned) and the new one, built from the same seed: state-dict keys, every tensor, built_levers, the loss terms, forward_infer, the gradients -- bit for bit.
  2. ON adds EXACTLY the declared keys (lvl_head.*, e_level.*), leaves every old tensor identical, and at step 0 (e_level zero) the forward equals the off model's EXACTLY (an appended token could not: see V8FMini.run_o).
  3. The term, the mask, the loss weight; the label is the LOGGED speed taken before any ego stage.
  4. ADMISSIBILITY: the head reads vision only -- the static guard, the key guard, the perturbation guard, and the three mutations (head fed ego, evading source, detach removed) go RED.
  5. G-LIVE (level_head_probe): contribution, head gradient, the DETACH control (exactly 0.0), the token reaches O, the all-false mask reads exactly 0.0, the frozen prior reads exactly 0.0; the mutations (token dropped from O's
     cond, total line deleted, detach removed) go RED.
  6. The launch gate refuses: a head with no loss, a weight with no head, a head declared and not built, a head built and not declared.
  7. ``--init-ckpt`` from an older checkpoint: every old key bit-identical, ONLY the declared new keys fresh; a lever-ON checkpoint into an off argv is refused; the real C0 init checkpoint when present.
  8. End to end on the trainer's synthetic world: train steps, the checkpoint rebuilds the identical model (G-EVAL), flag-off records keep their key sets.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
SNAP = Path(__file__).with_name("v8f_mini_tip_snapshot_level.txt")
SNAP_SHA = "b337ccecd3abfa4250a77e233502d9d3464a280dbafdc86b1e94ecbeef682721"      # sha256 of the tip's (b07982c) stack/tanitad/models/v8f_mini.py
STOPBAL_DEFAULTS = {"goal_pos_weight_stop": 1.0, "lon_class_weight_stop": 1.0}   # PREREG_V8F_STOPBAL (5ecdd7e6's parent 2bca1c52): built_levers gains these two keys after the snapshot was taken; off = 1.0 = the tip's loss
TRAIN_P = REPO / "stack" / "scripts" / "v8f_mini_train.py"
C0_INIT = Path("E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt")        # the C0 init checkpoint (dev box only)
RUN = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
NEW_KEYS = {"lvl_head.0.weight", "lvl_head.0.bias", "lvl_head.2.weight", "lvl_head.2.bias", "e_level.weight", "e_level.bias"}


# ------------------------------------------------------------------------------------------------ fixtures / helpers
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("lvlworld")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    return {"root": root, **w}


def make_batch(world, n=8, seed=0, hmax=10):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    return D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx]), J


def with_labels(b):
    y, ok = TR.level_stage(b)
    return {**b, "level_y": y, "level_ok": ok}


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
    name = "v8f_mini_tip_snapshot_level"
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__file__ = str(SNAP)
    sys.modules[name] = mod                                   # dataclasses resolves string annotations through sys.modules
    exec(compile(SNAP.read_text(encoding="utf-8"), str(SNAP), "exec"), mod.__dict__)
    return mod


def wake_token(m, seed=17, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.e_level.weight.copy_(torch.randn(m.e_level.weight.shape, generator=g) * scale)
        m.e_level.bias.copy_(torch.randn(m.e_level.bias.shape, generator=g) * scale)
    return m


def run_loss(m, b, seed=3):
    torch.manual_seed(seed)
    return m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(seed))


def args(extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", *extra])


def built(a, J, **over):
    cfg = TR.make_config(a)
    keep = {k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**cfg.__dict__, **keep, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True, "kin_mode": a.kin_mode, "p10_units": None}
    return cfg, model, opt, info


def _sc(out):
    return json.loads((Path(out) / "selfcheck.json").read_text(encoding="utf-8"))


def _sm(out):
    return json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))


def _lg(out):
    return [json.loads(x) for x in (Path(out) / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]


def refused(argv, tmp_path, match):
    with pytest.raises(SystemExit) as e:
        TR.main(["--out", str(tmp_path / "refused"), *RUN, *argv])
    assert "ZZDVB-REFUSEDZZ" in str(e.value) and match in str(e.value), str(e.value)[:600]


# ------------------------------------------------------------------------------------------------ 1. off == the tip
def test_the_snapshot_is_the_tip_module():
    assert hashlib.sha256(SNAP.read_bytes()).hexdigest() == SNAP_SHA


@pytest.mark.parametrize("wm", ["off", "b"])
def test_off_is_the_tip_bit_for_bit(world, wm):
    hmax = 16 if wm != "off" else 10
    b, J = make_batch(world, hmax=hmax)
    T0 = tip_module()
    tip, new = build(J, T0, wm=wm, hmax=hmax), build(J, M, wm=wm, hmax=hmax)
    assert new.cfg.level_head == "off" and new.cfg.w_level == 0.0 and new.lvl_head is None and new.e_level is None
    sa, sb = tip.state_dict(), new.state_dict()
    assert list(sa) == list(sb), "state-dict keys and their order"
    assert all(torch.equal(sa[k], sb[k]) for k in sa), "every tensor"
    nl = new.built_levers()
    assert {k: nl[k] for k in STOPBAL_DEFAULTS} == STOPBAL_DEFAULTS, "PREREG_V8F_STOPBAL: both stop weights are built at their default 1.0 (the tip's loss)"
    assert tip.built_levers() == {k: v for k, v in nl.items() if k not in STOPBAL_DEFAULTS} and set(new.trainable_groups()) == set(tip.trainable_groups()), "levers equal the tip's, apart from the two STOPBAL weights the snapshot predates"
    assert "level" not in new.built_levers().get("params", {}) and "level_head" not in new.built_levers()
    oa, ob = run_loss(tip, b), run_loss(new, b)
    assert list(oa["terms"]) == list(ob["terms"]) and all(torch.equal(oa["terms"][k], ob["terms"][k]) for k in oa["terms"]), "the loss terms, same keys, bit for bit"
    oa["terms"]["total"].backward()
    ob["terms"]["total"].backward()
    ga, gb = dict(tip.named_parameters()), dict(new.named_parameters())
    assert all((ga[k].grad is None and gb[k].grad is None) or torch.equal(ga[k].grad, gb[k].grad) for k in ga), "the gradients"
    tip.eval()
    new.eval()
    with torch.no_grad():
        ia, ib = tip.forward_infer(b), new.forward_infer(b)
    assert set(ia) == set(ib) and "level_v" not in ib
    assert all(torch.equal(ia[k], ib[k]) for k in ia if torch.is_tensor(ia[k])), "forward_infer, every tensor output"


# ------------------------------------------------------------------------------------------------ 2. on: the declared keys, step 0
def test_on_adds_exactly_the_declared_keys_and_leaves_the_old_ones_identical(world):
    _, J = make_batch(world)
    off, on = build(J), build(J, level_head="on", w_level=0.5)
    ko, kn = off.state_dict(), on.state_dict()
    assert set(kn) - set(ko) == NEW_KEYS and set(ko) <= set(kn)
    assert all(k.startswith(M.LEVEL_KEY_PREFIXES) for k in set(kn) - set(ko))
    assert all(torch.equal(ko[k], kn[k]) for k in ko), "every pre-existing tensor draws the off model's init (the head is built LAST)"
    assert float(on.e_level.weight.detach().abs().sum()) == 0.0 and float(on.e_level.bias.detach().abs().sum()) == 0.0, "e_level starts at EXACTLY zero"
    lv = on.built_levers()
    assert lv["level_head"] == "on" and lv["w_level"] == 0.5 and lv["level_token_zero_init"] is True and lv["level_vision_only_source"] is True
    assert lv["params"]["LEVEL"] == sum(p.numel() for n, p in on.named_parameters() if n.startswith(M.LEVEL_KEY_PREFIXES))


def test_step_0_forward_equals_the_off_models(world):
    b, J = make_batch(world)
    off, on = build(J), build(J, level_head="on", w_level=0.5)
    res = on.load_state_dict(off.state_dict(), strict=False)
    assert set(res.missing_keys) == NEW_KEYS and not res.unexpected_keys
    off.eval()
    on.eval()
    with torch.no_grad():
        a, c = off.forward_infer(b), on.forward_infer(b)
    assert set(c) - set(a) == {"level_v"}
    assert all(torch.equal(a[k], c[k]) for k in a if torch.is_tensor(a[k])), "inference: every output equals the off model's, EXACTLY (zero-init e_level)"
    off.train()
    on.train()
    oa, oc = run_loss(off, b), run_loss(on, with_labels(b))
    assert all(torch.equal(oa["terms"][k], oc["terms"][k]) for k in oa["terms"] if k != "total"), "training: every pre-existing loss term equals the off model's"
    assert float(oc["terms"]["total"] - oa["terms"]["total"] - 0.5 * oc["terms"]["level"]) == pytest.approx(0.0, abs=1e-4 * max(1.0, float(oa["terms"]["total"]))), "total = off total + w_level x level"


def test_an_appended_token_could_not_do_this_the_reason_for_the_additive_form(world):
    """Documents the design choice (reported to the MM): a zero token APPENDED to O's cond changes cond.mean(1) (the query) and adds a softmax key, so the step-0 forward would differ from the off model's."""
    b, J = make_batch(world)
    off = build(J)
    off.eval()
    with torch.no_grad():
        adapt = off._encode(b)
        i = torch.full((8,), b["feats"].shape[1] - 1, dtype=torch.long)
        cond = torch.stack([off._ego(off._at(b["p10f"], i))] * 3, 1)
        sc, pad = off._scene(adapt, b["frame_ok"], i)
        h0 = off.read_o(cond, sc, pad)
        h1 = off.read_o(torch.cat([cond, torch.zeros_like(cond[:, :1])], 1), sc, pad)
    assert not torch.equal(h0, h1)


# ------------------------------------------------------------------------------------------------ 3. the term, the label
def test_the_token_reaches_O_through_run_o_whose_signature_is_unchanged(world):
    b, J = make_batch(world)
    on = wake_token(build(J, level_head="on", w_level=0.5)).eval()
    with torch.no_grad():
        adapt = on._encode(b)
        last = torch.full((8,), b["feats"].shape[1] - 1, dtype=torch.long)
        ck = on.ck_nom[None].expand(8, -1, -1)
        ok = torch.ones(8, 3)
        code = torch.zeros(8, M.CODE_DIM)
        v_hat, v_in = on._level(b, adapt, last)
        t_given = on.run_o(b, adapt, last, ck, ok, code)[0]                         # run_o keeps its six-argument signature and computes the level itself
        saved = (on.e_level.weight.clone(), on.e_level.bias.clone())
        on.e_level.weight.zero_()
        on.e_level.bias.zero_()
        t_zero = on.run_o(b, adapt, last, ck, ok, code)[0]
        on.e_level.weight.copy_(saved[0])
        on.e_level.bias.copy_(saved[1])
    assert not torch.equal(t_given, t_zero)
    assert torch.equal(v_in, v_hat / M.LEVEL_SCALE * b["frame_ok"][:, -1].float())


def test_the_level_term_against_a_hand_computation_and_the_masks(world):
    b, J = make_batch(world, n=8)
    on = build(J, level_head="on", w_level=0.5)
    y, ok = TR.level_stage(b)
    assert torch.equal(y, b["kin3"][:, 0].float())
    lab = {**b, "level_y": y, "level_ok": ok}
    with torch.no_grad():
        out = run_loss(on, lab)
    vh = out["level_v"].double().numpy()
    yy = np.nan_to_num(y.double().numpy())
    use = (ok & b["frame_ok"][:, -1]).numpy()
    err = np.abs(vh - yy)
    hub = np.where(err <= 1.0, 0.5 * err ** 2, err - 0.5)
    assert float(out["terms"]["level"]) == pytest.approx(float((hub * use).sum() / max(use.sum(), 1)), rel=1e-5)
    fo = {**lab, "frame_ok": b["frame_ok"].clone()}
    fo["frame_ok"][:, -1] = False                                                     # no real NOW frame anywhere: the term is exactly 0.0 and O is handed 0
    with torch.no_grad():
        o2 = run_loss(on, fo)
    assert float(o2["terms"]["level"]) == 0.0
    with pytest.raises(ValueError, match="level_y"):
        on.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(1))                    # a lever-ON loss on a batch without the label is refused, never silently skipped
    with torch.no_grad():
        o3 = run_loss(on, {**lab, "level_ok": torch.zeros_like(ok)})
    assert float(o3["terms"]["level"]) == 0.0


def test_the_label_is_taken_before_the_ego_stages_in_the_loop():
    src = TRAIN_P.read_text(encoding="utf-8")
    i_lab = src.index("lvl_lab = level_stage(b)")
    i_ego, i_rcv, i_att, i_dev = (src.index(x, i_lab) for x in ("b = ego.apply(b, step)", "b = rcv.apply(b, step)", 'b["level_y"], b["level_ok"] = lvl_lab', "b = to_dev(b, dev)"))
    assert i_lab < i_ego < i_rcv < i_att < i_dev, "label from the LOGGED kin3, attached after every stage that rebuilds it, before the device move"
    kin = torch.tensor([[3.0, 0.1, 0.0], [float("nan"), 0.0, 0.0]])
    y, ok = TR.level_stage({"kin3": kin})
    assert y[0] == 3.0 and ok.tolist() == [True, False]


# ------------------------------------------------------------------------------------------------ 4. admissibility: vision only
def test_vision_only_guard_passes_on_the_built_head(world):
    b, J = make_batch(world)
    on = build(J, level_head="on", w_level=0.5)
    rec = M.assert_level_head_vision_only(on, with_labels(b))
    assert rec["keys_read"] == [] or rec["keys_read"] == ["frame_ok"]
    assert M.level_head_source_problems() == []
    on.eval()
    with torch.no_grad():
        info = M.assert_no_situation_classifier_input(on, b)
    assert not {"level_y", "level_ok"} & set(info["read"]), "inference reads no label"


def test_mutation_head_fed_ego_goes_red(world):
    b, J = make_batch(world)

    class EgoHead(M.V8FMini):                                 # the head reads the ego p10 features instead of the vision tokens
        def _level(self, b, adapt, last):
            v_hat = self.lvl_head(self._ego(self._at(b["p10f"], last))).squeeze(-1).float()
            return v_hat, v_hat.detach() / M.LEVEL_SCALE
    m = build(J, cls=EgoHead, level_head="on", w_level=0.5)
    with pytest.raises(AssertionError, match="non-vision"):
        M.assert_level_head_vision_only(m, with_labels(b))    # the STATIC half names p10f / _ego


def test_mutation_head_fed_ego_through_a_source_that_evades_the_scan_goes_red(world):
    b, J = make_batch(world)

    class Sneaky(M.V8FMini):                                  # same wiring, spelled so no forbidden identifier appears: the RUNTIME halves must catch it
        def _level(self, b, adapt, last):
            k = "p10" + "f"
            ego = getattr(self, "_e" + "go")(self._at(b[k], last))
            v_hat = self.lvl_head(adapt.mean((1, 2)) + ego).squeeze(-1).float()
            return v_hat, v_hat.detach() / M.LEVEL_SCALE
    m = build(J, cls=Sneaky, level_head="on", w_level=0.5)
    assert M.level_head_source_problems(Sneaky._level) == [], "the static scan is evaded on purpose"
    with pytest.raises(AssertionError, match="READS batch keys"):
        M.assert_level_head_vision_only(m, with_labels(b))


# ------------------------------------------------------------------------------------------------ 5. G-LIVE
def test_live_check_and_the_probe_pass(world):
    b, J = make_batch(world)
    on = build(J, level_head="on", w_level=0.5, wake=False)
    lab = with_labels(b)
    res = TR.live_check(on, lab, on.cfg)
    assert res["pass"] and res["control_prior_grad_sum"] == 0.0 and "level" in res["terms"] and not res["unexpected_dead"]
    pr = TR.level_head_probe(on, lab, on.cfg)
    assert pr["pass"] is True
    assert pr["head_grad_at_w0_detach_control"] == 0.0 and pr["control_allfalse_term"] == 0.0 and pr["control_allfalse_total_diff"] == 0.0 and pr["control_prior_grad_diff"] == 0.0
    assert pr["contribution"] == pytest.approx(pr["w_level_x_term"], rel=1e-3) and pr["head_grad_diff"] > 0.0 and pr["plan_moves_with_token"] > 0.0 and pr["e_level_grad_at_w0"] > 0.0
    assert pr["w_level_for_10pct_share"] > 0.0
    assert TR.level_head_probe(build(J), b, build(J).cfg) == {"level": "off", "pass": True, "skipped": pr_skip()}


def pr_skip():
    return "--level-head not named: nothing is built (the default run is the pre-change run)"


def test_mutation_token_dropped_from_Os_cond_goes_red(world):
    b, J = make_batch(world)

    class NoToken(M.V8FMini):
        def run_o(self, *a, **k):
            keep, self.e_level = self.e_level, None           # O's cond is built without the level term
            try:
                return super().run_o(*a, **k)
            finally:
                self.e_level = keep
    m = build(J, cls=NoToken, level_head="on", w_level=0.5, wake=False)
    with pytest.raises(AssertionError, match="does not reach O"):
        TR.level_head_probe(m, with_labels(b), m.cfg)


def test_mutation_total_line_deleted_goes_red(world):
    b, J = make_batch(world)

    class NoLine(M.V8FMini):
        def loss(self, *a, **k):
            out = super().loss(*a, **k)
            out["terms"]["total"] = out["terms"]["total"] - self.cfg.w_level * out["terms"]["level"]       # the `total = total + w_level * level` line is gone
            return out
    m = build(J, cls=NoLine, level_head="on", w_level=0.5, wake=False)
    with pytest.raises(AssertionError, match="does not reach the loss"):
        TR.level_head_probe(m, with_labels(b), m.cfg)


def test_mutation_detach_removed_goes_red(world):
    b, J = make_batch(world)

    class NoDetach(M.V8FMini):
        def _level(self, b, adapt, last):
            v_hat, _ = super()._level(b, adapt, last)
            return v_hat, v_hat / M.LEVEL_SCALE                # O reads v_hat itself: the plan's loss reaches the head
    m = build(J, cls=NoDetach, level_head="on", w_level=0.5, wake=False)
    with pytest.raises(AssertionError, match="not EXACTLY 0.0"):
        TR.level_head_probe(m, with_labels(b), m.cfg)


def test_mutation_the_label_mask_ignored_goes_red(world):
    b, J = make_batch(world)

    class NoMask(M.V8FMini):
        def loss(self, batch, *a, **k):
            return super().loss({**batch, "level_ok": torch.ones_like(batch["level_ok"])}, *a, **k)
    m = build(J, cls=NoMask, level_head="on", w_level=0.5, wake=False)
    with pytest.raises(AssertionError, match="ALL-FALSE"):
        TR.level_head_probe(m, with_labels(b), m.cfg)


# ------------------------------------------------------------------------------------------------ 6. the gate
def test_every_flag_has_an_entry():
    assert TR.FLAG_CHECKS["level_head"] == "lever:level_head" and TR.FLAG_CHECKS["w_level"] == "cfg:w_level"
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert {"level_head", "w_level"} <= dests and dests == set(TR.FLAG_CHECKS)


def test_the_gate_refuses_lies(world):
    _, J = make_batch(world)
    a = args(["--level-head", "on", "--w-level", "0.7"])
    cfg, model, opt, info = built(a, J)
    assert TR.declared_vs_built(a, cfg, model, opt, info) == [] and model.lvl_head is not None
    cfg, model, opt, info = built(a, J, level_head="off", w_level=0.0)                  # declared on, built off
    assert any("declared on but built off" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    a0 = args(["--level-head", "on", "--w-level", "0"])                                 # a head with no loss
    cfg, model, opt, info = built(a0, J)
    assert any("a head with no loss" in m for m in TR.declared_vs_built(a0, cfg, model, opt, info))
    aw = args(["--w-level", "0.5"])                                                     # a weight with no head
    cfg, model, opt, info = built(aw, J)
    assert any("a loss weight with no head" in m for m in TR.declared_vs_built(aw, cfg, model, opt, info))
    ao = args([])                                                                       # declared off, a head built
    cfg, model, opt, info = built(ao, J, level_head="on", w_level=0.5)
    assert any("declared off but" in m or "level_head off declared" in m for m in TR.declared_vs_built(ao, cfg, model, opt, info))
    a = args(["--level-head", "on", "--w-level", "0.7"])                                # a non-zero e_level at a fresh start
    cfg, model, opt, info = built(a, J)
    wake_token(model)
    assert any("zero-initialised" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    a = args(["--level-head", "on", "--w-level", "-1"])
    with pytest.raises(ValueError, match="w_level"):
        TR.make_config(a)
    with pytest.raises(ValueError, match="level_head"):
        M.V8FMiniConfig(**TINY, level_head="maybe")
    cfg, model, opt, info = built(args([]), J)
    assert TR.declared_vs_built(args([]), cfg, model, opt, info) == []                  # the default argv still passes


def test_the_launch_refuses_through_main(tmp_path):
    refused(["--level-head", "on"], tmp_path, "a head with no loss")
    refused(["--w-level", "0.5"], tmp_path, "a loss weight with no head")


# ------------------------------------------------------------------------------------------------ 7. init from an older checkpoint
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("lvltrain")
    off = root / "off"
    assert TR.main(["--out", str(off), *RUN, "--steps", "2"]) == 0
    on = root / "on"
    assert TR.main(["--out", str(on), *RUN, "--steps", "3", "--level-head", "on", "--w-level", "0.5", "--init-ckpt", str(off / "ckpt_last.pt"), "--lr", "1e-3"]) == 0
    return types.SimpleNamespace(root=root, off=off, on=on)


def test_init_from_an_older_checkpoint_loads_every_old_key_and_only_the_declared_ones_are_fresh(trained):
    sc, sm = _sc(trained.on), _sm(trained.on)
    rec = sc["levers"]["init_ckpt"]
    assert rec["applied"] is True and rec["n_mismatch_after_load"] == 0 and set(rec["fresh_keys"]) == NEW_KEYS == set(rec["declared_new_keys"])
    old = torch.load(trained.off / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert rec["n_tensors"] == len(old["model"]) and not (set(old["model"]) & NEW_KEYS)
    assert sc["levers"]["level_head"] == "on" and sc["levers"]["w_level"] == 0.5 and sc["declared_vs_built"] == "PASS"
    assert sc["level_head_probe"]["pass"] is True and sc["levers"]["level_vision_only_source"] is True
    assert sm["level_head"] == "on" and sm["w_level"] == 0.5 and sm["done"] is True


def test_a_lever_on_checkpoint_into_an_off_argv_is_refused_and_an_old_one_into_an_on_argv_is_the_declared_path(trained, tmp_path):
    with pytest.raises(SystemExit) as e:
        TR.main(["--out", str(tmp_path / "x"), *RUN, "--steps", "1", "--init-ckpt", str(trained.on / "ckpt_last.pt")])
    assert "ZZDVB-REFUSEDZZ" in str(e.value) and "Unexpected key" in str(e.value) and "lvl_head" in str(e.value)       # the strict load, verbatim, for every argv that declares no lever
    # an old checkpoint into an OFF argv: the strict load, the record keeps the pre-change key set
    assert TR.main(["--out", str(tmp_path / "y"), *RUN, "--steps", "1", "--init-ckpt", str(trained.off / "ckpt_last.pt")]) == 0
    r = _sc(tmp_path / "y")["levers"]["init_ckpt"]
    assert "fresh_keys" not in r and "declared_new_keys" not in r and r["n_mismatch_after_load"] == 0
    assert "level_head_probe" not in _sc(tmp_path / "y") and "level_head" not in _sc(tmp_path / "y")["levers"] and "level_head" not in _sm(tmp_path / "y")


def test_init_ckpt_refuses_a_fresh_key_that_is_not_the_declared_levers(trained, tmp_path):
    blob = torch.load(trained.off / "ckpt_last.pt", map_location="cpu", weights_only=False)
    victim = next(k for k in blob["model"] if k.startswith("read_o.q"))
    del blob["model"][victim]                                                         # a checkpoint that lacks an OLD key: not the lever's to leave fresh
    bad = tmp_path / "missing_old_key.pt"
    torch.save(blob, bad)
    with pytest.raises(SystemExit) as e:
        TR.main(["--out", str(tmp_path / "z"), *RUN, "--steps", "1", "--level-head", "on", "--w-level", "0.5", "--init-ckpt", str(bad)])
    assert "ZZDVB-REFUSEDZZ" in str(e.value) and victim in str(e.value)


def test_load_init_ckpt_itself_raises_on_a_missing_old_key_and_the_gate_reads_the_record(trained, tmp_path):
    """The two guards are tested one at a time (the end-to-end refusal above would hide a deleted one behind the other)."""
    ck = torch.load(trained.off / "ckpt_last.pt", map_location="cpu", weights_only=False)
    victim = next(k for k in ck["model"] if k.startswith("read_o.q"))
    cfg = M.V8FMiniConfig(**{**ck["cfg"], "level_head": "on", "w_level": 0.5})
    model = M.V8FMini(cfg, M.P10Prior.synthetic(5), ck["cons_mu"], ck["cons_sd"])
    bad = tmp_path / "lacks_an_old_key.pt"
    blob = {**ck, "model": {k: v for k, v in ck["model"].items() if k != victim}}
    torch.save(blob, bad)
    with pytest.raises(ValueError, match="not the declared lever's"):
        TR.load_init_ckpt(str(bad), model, cfg)
    good = TR.load_init_ckpt(str(trained.off / "ckpt_last.pt"), M.V8FMini(cfg, M.P10Prior.synthetic(5), ck["cons_mu"], ck["cons_sd"]), cfg)
    assert set(good["fresh_keys"]) == NEW_KEYS == set(good["declared_new_keys"])
    a = args(["--level-head", "on", "--w-level", "0.5", "--init-ckpt", "x.pt"])
    rec = {"path": str(Path("x.pt").resolve()), "sha256": "0" * 64, "applied": True, "n_mismatch_after_load": 0, "n_tensors": 5, "skipped": None, "fresh_keys": sorted(NEW_KEYS), "declared_new_keys": sorted(NEW_KEYS)}
    assert TR.bad_init_ckpt(a, {"init_ckpt": rec}) == []
    assert any("left fresh" in m for m in TR.bad_init_ckpt(a, {"init_ckpt": {**rec, "fresh_keys": sorted(NEW_KEYS) + [victim]}}))
    assert any("declared new keys" in m for m in TR.bad_init_ckpt(a, {"init_ckpt": {**rec, "declared_new_keys": []}}))
    assert any("declared new keys" in m for m in TR.bad_init_ckpt(args(["--init-ckpt", "x.pt"]), {"init_ckpt": rec}))


@pytest.mark.skipif(not C0_INIT.exists(), reason="the C0 init checkpoint is on the dev box only")
def test_real_c0_init_checkpoint_loads_with_only_the_declared_keys_fresh_and_the_step_0_forward_is_unchanged(world):
    ck = torch.load(C0_INIT, map_location="cpu", weights_only=False)
    cfg_d = {**ck["cfg"]}
    b, J = make_batch(world, hmax=cfg_d["hmax"])
    mu, sd = M.cons_stats(J)

    def mk(**kw):
        torch.manual_seed(0)
        return M.V8FMini(M.V8FMiniConfig(**{**cfg_d, **kw}), M.P10Prior.synthetic(0), mu, sd)
    off, on = mk(), mk(level_head="on", w_level=1.0)
    off.load_state_dict(ck["model"], strict=True)
    rec = TR.load_init_ckpt(str(C0_INIT), on, on.cfg)
    assert rec["n_mismatch_after_load"] == 0 and set(rec["fresh_keys"]) == NEW_KEYS and rec["n_tensors"] == len(ck["model"])
    sd_on = on.state_dict()
    assert all(torch.equal(sd_on[k], v) for k, v in ck["model"].items()), "every old key bit-identical"
    off.eval()
    on.eval()
    with torch.no_grad():
        a, c = off.forward_infer(b), on.forward_infer(b)
    assert all(torch.equal(a[k], c[k]) for k in a if torch.is_tensor(a[k])), "the step-0 forward of the lever-ON model on the real C0 weights equals the C0 model's, EXACTLY"


# ------------------------------------------------------------------------------------------------ 8. end to end
def test_training_moves_the_head_and_the_token_and_the_checkpoint_rebuilds_the_identical_model(trained):
    log = _lg(trained.on)
    assert len(log) == 3 and all(math.isfinite(r["level"]) and r["level"] > 0.0 and math.isfinite(r["total"]) for r in log)
    ck = torch.load(trained.on / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["cfg"]["level_head"] == "on" and ck["cfg"]["w_level"] == 0.5 and NEW_KEYS <= set(ck["model"])
    assert float(ck["model"]["e_level.weight"].abs().sum()) > 0.0 and float(ck["model"]["e_level.bias"].abs().sum()) > 0.0, "the token left its zero init: the loss reached it"
    cfg = M.V8FMiniConfig(**ck["cfg"])                                                 # G-EVAL: the evaluators build V8FMiniConfig(**ck["cfg"])
    assert cfg.level_head == "on" and cfg.w_level == 0.5
    model = M.V8FMini(cfg, M.P10Prior.synthetic(5), ck["cons_mu"], ck["cons_sd"])
    model.load_state_dict(ck["model"], strict=True)
    assert model.lvl_head is not None and model.e_level is not None
    off = torch.load(trained.off / "ckpt_last.pt", map_location="cpu", weights_only=False)
    moved = [k for k in NEW_KEYS if k.startswith("lvl_head.")]
    assert moved and set(off["model"]) | NEW_KEYS == set(ck["model"])


def test_flag_off_records_keep_their_key_sets(trained):
    sc, sm = _sc(trained.off), _sm(trained.off)
    assert "level_head_probe" not in sc and "level_head" not in sc["levers"] and "w_level" not in sc["levers"]
    assert "level_head" not in sm and "w_level" not in sm and "level" not in _lg(trained.off)[0]
    assert "level_head" in torch.load(trained.off / "ckpt_last.pt", map_location="cpu", weights_only=False)["cfg"], "the checkpoint's cfg carries the field (default off), as every earlier lever's did"
