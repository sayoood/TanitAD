"""PREREG_V8F_STOPBAL (DRAFT): ``--goal-pos-weight-stop W`` and ``--lon-class-weight-stop W`` in ``stack/scripts/v8f_mini_train.py`` -- re-weighting the rare STOP class of T's goal BCE / lon CE.

Both flags default 1.0 = OFF: the loss branch is not taken and the tip's loss is reproduced bit for bit (section 1, against a PRISTINE export of the tip when STOPBAL_PRISTINE points at one).
Reachability hop by hop (CLAUDE.md: "built, tested, and unreachable from its caller" is a class):
  argv -> make_config -> cfg.<w> -> built_levers -> declared_vs_built (G-DVB) -> selfcheck.json levers -> V8FMini.loss (the two weighting lines) -> .backward() on h_goal / h_lon.
Every check has a control that must read EXACTLY the unweighted value (a W=1 run, a batch with no STOP row, the OTHER head's parameters). A mutation (delete a weighting line) must turn sections 2-3 RED.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", REPO / "stack" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_dagger_train import TINY, _run, _selfcheck, _summary, _woken, dvb, gate_parts, pack  # noqa: E402,F401

torch.set_num_threads(1)
STOP_G, STOP_L = M.STOP_I, M.LON_STOP_I


def real_batch(pk, n=24, seed=0):
    J = pk.J
    store = D.ClipStore(pk.world, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(pk.world / "native_tracks.pt"))
    wins = D.train_windows(store, J, D.select_train_clips(list(map(str, J["clip"])), 1.0))
    cfg = M.V8FMiniConfig(**{**TINY, "wm": "off"})
    ds = D.V8FMiniWindows(store, J, wins, hmax=cfg.hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    mu, sd = M.cons_stats(J)
    model = _woken(M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd))
    return model, batch, cfg


def with_stop(batch, n_pos, stop_rows=True):
    """Label surgery: STOP_POINT observed (goal_w=1) on every row, positive on the first n_pos rows; lon allowed set = {STOP} on the first n_pos rows, {KEEP} elsewhere, every row labelled."""
    b = {**batch, "lab_t": dict(batch["lab_t"])}
    lt = b["lab_t"]
    gy, gw = lt["goal_y"].clone(), lt["goal_w"].clone()
    gw[:, STOP_G] = 1.0
    gy[:, STOP_G] = 0.0
    if stop_rows:
        gy[:n_pos, STOP_G] = 1.0
    la = torch.zeros_like(lt["lon_al"])
    la[:, 6] = True
    if stop_rows:
        la[:n_pos, :] = False
        la[:n_pos, STOP_L] = True
    lt.update(goal_y=gy, goal_w=gw, lon_al=la, has=torch.ones_like(lt["has"]))
    return b


def run_loss(model, batch, gw=1.0, lw=1.0):
    model.cfg.goal_pos_weight_stop, model.cfg.lon_class_weight_stop = gw, lw
    torch.manual_seed(7)
    out = model.loss(batch, p_gt=0.5, gen=torch.Generator().manual_seed(11))
    return out["terms"], out


def hand_goal(out, lt, W):
    x, y, w = out["goal_logits"].detach().double(), lt["goal_y"].double(), (lt["goal_w"] * lt["has"][:, None].float()).double()
    tot = 0.0
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            if w[b, c] == 0:
                continue
            pos = float(y[b, c]) * math.log1p(math.exp(-float(x[b, c])))
            neg = float(1 - y[b, c]) * math.log1p(math.exp(float(x[b, c])))
            tot += float(w[b, c]) * ((W if c == STOP_G else 1.0) * pos + neg)
    return tot / max(float(w.sum()), 1.0)


def hand_lon(out, lt, W):
    x, al = out["lon_logits"].detach().double().numpy(), lt["lon_al"].numpy()
    tot, nv = 0.0, 0
    for b in range(x.shape[0]):
        if not (al[b].any() and bool(lt["has"][b])):
            continue
        nv += 1
        l = np.log(np.exp(x[b]).sum()) - np.log(np.exp(x[b][al[b]]).sum())
        tot += l * (W if al[b][STOP_L] else 1.0)
    return tot / max(nv, 1)


# ---------------------------------------------------------------- 1. default = the tip, bit for bit
def test_defaults_are_off_and_flags_have_gdvb_entries():
    ps = TR.build_parser()
    for n in ("goal_pos_weight_stop", "lon_class_weight_stop"):
        assert ps.get_default(n) == 1.0 and TR.FLAG_CHECKS[n] == f"cfg:{n}"
        assert n in {a.dest for a in ps._actions}
        assert getattr(M.V8FMiniConfig(**TINY), n) == 1.0
        for bad in (0.5, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                M.V8FMiniConfig(**{**TINY, n: bad})


def test_default_loss_and_grads_equal_the_pristine_tip_bit_for_bit(pack):
    pr = os.environ.get("STOPBAL_PRISTINE")
    if not pr or not (Path(pr) / "tanitad" / "models" / "v8f_mini.py").exists():
        pytest.skip("STOPBAL_PRISTINE (a pristine export of the tip's stack/) not set")
    spec = importlib.util.spec_from_file_location("v8f_mini_pristine", Path(pr) / "tanitad" / "models" / "v8f_mini.py")
    P = importlib.util.module_from_spec(spec)
    sys.modules["v8f_mini_pristine"] = P
    spec.loader.exec_module(P)
    model, batch, cfg = real_batch(pack)
    batch = with_stop(batch, 5)
    mu, sd = M.cons_stats(pack.J)
    ref = _woken(P.V8FMini(P.V8FMiniConfig(**{**TINY, "wm": "off"}), P.P10Prior.synthetic(0), mu, sd))
    ref.load_state_dict(model.state_dict())
    outs = []
    for m in (model, ref):
        m.zero_grad(set_to_none=True)
        torch.manual_seed(7)
        o = m.loss(batch, p_gt=0.5, gen=torch.Generator().manual_seed(11))
        o["terms"]["total"].backward()
        outs.append((o["terms"], {k: (q.grad.clone() if q.grad is not None else None) for k, q in m.named_parameters()}))
    (T0, g0), (T1, g1) = outs
    assert set(T0) == set(T1) and all(torch.equal(T0[k], T1[k]) for k in T0), "every loss term, bit for bit"
    assert all((g0[k] is None and g1[k] is None) or torch.equal(g0[k], g1[k]) for k in g0), "every gradient, bit for bit"


# ---------------------------------------------------------------- 2. the weighting, against hand values
def test_goal_stop_positive_term_scales_by_exactly_W_and_the_rest_is_unchanged(pack):
    model, batch, cfg = real_batch(pack)
    b = with_stop(batch, 6)
    T1, o1 = run_loss(model, b, 1.0, 1.0)
    T3, o3 = run_loss(model, b, 3.5, 1.0)
    lt = b["lab_t"]
    assert abs(float(T1["goal"]) - hand_goal(o1, lt, 1.0)) < 1e-5 and abs(float(T3["goal"]) - hand_goal(o3, lt, 3.5)) < 1e-5, (float(T3["goal"]), hand_goal(o3, lt, 3.5))
    pos = (lt["goal_y"][:, STOP_G] * lt["goal_w"][:, STOP_G]).double()
    sp = F.softplus(-o1["goal_logits"].detach().double()[:, STOP_G])
    want = (2.5 * (pos * sp).sum() / (lt["goal_w"] * lt["has"][:, None].float()).sum()).item()
    assert abs((float(T3["goal"]) - float(T1["goal"])) - want) < 1e-5 and want > 0, "the delta is (W-1) x the STOP-positive entries' own BCE"
    for k in T1:
        if k not in ("goal", "total"):
            assert float(T1[k]) == float(T3[k]), f"{k} unchanged by the goal lever"
    assert abs((float(T3["total"]) - float(T1["total"])) - cfg.w_tac * (float(T3["goal"]) - float(T1["goal"]))) < 1e-5
    nb = with_stop(batch, 0, stop_rows=False)                      # control: a batch with NO STOP-positive entry
    Tn, _ = run_loss(model, nb, 1.0, 1.0)
    Tn3, _ = run_loss(model, nb, 3.5, 1.0)
    assert float(Tn["goal"]) == float(Tn3["goal"]) and float(Tn["total"]) == float(Tn3["total"]), "control: no STOP-positive row -> W changes nothing, EXACTLY"


def test_lon_stop_rows_scale_by_exactly_W_and_the_rest_is_unchanged(pack):
    model, batch, cfg = real_batch(pack)
    b = with_stop(batch, 6)
    T1, o1 = run_loss(model, b, 1.0, 1.0)
    T3, o3 = run_loss(model, b, 1.0, 4.0)
    lt = b["lab_t"]
    assert abs(float(T1["lon"]) - hand_lon(o1, lt, 1.0)) < 1e-5 and abs(float(T3["lon"]) - hand_lon(o3, lt, 4.0)) < 1e-5, (float(T3["lon"]), hand_lon(o3, lt, 4.0))
    assert float(T3["lon"]) > float(T1["lon"])
    for k in T1:
        if k not in ("lon", "total"):
            assert float(T1[k]) == float(T3[k]), f"{k} unchanged by the lon lever"
    assert abs((float(T3["total"]) - float(T1["total"])) - cfg.w_tac * 0.5 * (float(T3["lon"]) - float(T1["lon"]))) < 1e-5
    nb = with_stop(batch, 0, stop_rows=False)
    Tn, _ = run_loss(model, nb, 1.0, 1.0)
    Tn4, _ = run_loss(model, nb, 1.0, 4.0)
    assert float(Tn["lon"]) == float(Tn4["lon"]) and float(Tn["total"]) == float(Tn4["total"]), "control: no STOP row -> W changes nothing, EXACTLY"


# ---------------------------------------------------------------- 3. gradients reach the heads; controls read exactly 0.0
def grads(model, batch, gw, lw):
    model.zero_grad(set_to_none=True)
    T, _ = run_loss(model, batch, gw, lw)
    T["total"].backward()
    return {k: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for k, p in model.named_parameters() if p.requires_grad}


def diff(ga, gb, prefix):
    return sum(float((ga[k] - gb[k]).abs().sum()) for k in ga if k.startswith(prefix))


def test_gradients_reach_the_goal_and_lon_heads_and_the_other_head_reads_exactly_zero(pack):
    model, batch, _ = real_batch(pack)
    b = with_stop(batch, 6)
    g0, g0b = grads(model, b, 1.0, 1.0), grads(model, b, 1.0, 1.0)
    assert diff(g0, g0b, "") == 0.0, "control: the probe is deterministic"
    gg, gl = grads(model, b, 3.5, 1.0), grads(model, b, 1.0, 4.0)
    assert diff(gg, g0, "h_goal.") > 0.0, "the goal lever moves h_goal's gradient"
    assert diff(gg, g0, "h_lon.") == 0.0, "the goal lever leaves h_lon's gradient EXACTLY 0.0 different"
    assert diff(gl, g0, "h_lon.") > 0.0, "the lon lever moves h_lon's gradient"
    assert diff(gl, g0, "h_goal.") == 0.0, "the lon lever leaves h_goal's gradient EXACTLY 0.0 different"
    nb = with_stop(batch, 0, stop_rows=False)
    n0, n1 = grads(model, nb, 1.0, 1.0), grads(model, nb, 3.5, 4.0)
    assert diff(n1, n0, "") == 0.0, "control: no STOP row -> EVERY parameter's gradient is the unweighted one"


# ---------------------------------------------------------------- 4. G-DVB and the selfcheck
def test_gdvb_declared_vs_built_and_refusals(pack):
    parts = gate_parts(pack, ["--goal-pos-weight-stop", "3.5", "--lon-class-weight-stop", "2.0"])
    a, model = parts[0], parts[2]
    assert [m for m in dvb(parts) if "stop" in m] == []
    bl = model.built_levers()
    assert bl["goal_pos_weight_stop"] == 3.5 and bl["lon_class_weight_stop"] == 2.0
    model.cfg.goal_pos_weight_stop = 1.0                    # the build dropped the lever (the refcv6 class)
    assert any("lever goal_pos_weight_stop: declared 3.5 but built 1.0" in m for m in dvb(parts)), dvb(parts)
    model.cfg.goal_pos_weight_stop = 3.5
    model.cfg.lon_class_weight_stop = 1.0
    assert any("lever lon_class_weight_stop: declared 2.0 but built 1.0" in m for m in dvb(parts)), dvb(parts)
    model.cfg.lon_class_weight_stop = 2.0
    a.goal_pos_weight_stop = 0.5
    assert any("goal_pos_weight_stop 0.5 is not a finite number >= 1" in m for m in dvb(parts))


def test_a_flag_without_a_gdvb_entry_is_refused_by_the_gate_itself(pack, monkeypatch):
    parts = gate_parts(pack, ["--lon-class-weight-stop", "2.0"])
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != "lon_class_weight_stop"})
    assert any("flags without a declared-vs-built entry" in m and "lon_class_weight_stop" in m for m in dvb(parts))


def test_a_real_run_records_both_levers_in_selfcheck_and_summary_and_default_records_one(pack, tmp_path):
    assert TR.main(_run(tmp_path / "on", ["--steps", "3", "--goal-pos-weight-stop", "3.5", "--lon-class-weight-stop", "2.0"])) == 0
    sc, s = _selfcheck(tmp_path / "on"), _summary(tmp_path / "on")
    assert sc["levers"]["goal_pos_weight_stop"] == 3.5 and sc["levers"]["lon_class_weight_stop"] == 2.0
    assert s["levers"]["goal_pos_weight_stop"] == 3.5 and s["levers"]["lon_class_weight_stop"] == 2.0
    ck = torch.load(tmp_path / "on" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["cfg"]["goal_pos_weight_stop"] == 3.5 and ck["cfg"]["lon_class_weight_stop"] == 2.0
    assert TR.main(_run(tmp_path / "off", ["--steps", "3"])) == 0
    sc0 = _selfcheck(tmp_path / "off")
    assert sc0["levers"]["goal_pos_weight_stop"] == 1.0 and sc0["levers"]["lon_class_weight_stop"] == 1.0


def test_the_two_weighting_lines_exist_exactly_once_and_the_module_is_the_export():
    src = (REPO / "stack" / "tanitad" / "models" / "v8f_mini.py").read_text(encoding="utf-8")
    assert src.count("cfg.goal_pos_weight_stop != 1.0") == 1 and src.count("if stop_w != 1.0:") == 1
    assert Path(M.__file__).resolve().is_relative_to(REPO.resolve()), M.__file__
