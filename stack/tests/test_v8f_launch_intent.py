"""PREREG_V8F_LAUNCH_INTENT: ``--launch-intent-w W`` in ``stack/scripts/v8f_mini_train.py`` -- the up-weighted ALONG-TRACK term on GT-launch TRAIN samples.

The flag is default-OFF. THIS file proves the lever is REACHABLE from the real caller (CLAUDE.md, 2026-09-10/11: "built, tested, and unreachable from its caller" is a class), hop by hop:

  argv --launch-intent-w  ->  make_config  ->  cfg.launch_intent_w  ->  built_levers  ->  declared_vs_built (G-DVB: FLAG_CHECKS entry, declared == built, w_traj > 0)
        ->  launch_intent_probe (G-LIVE on the loop's OWN mask call ``launch_stage``: the BUILT loss moves by exactly w_traj x W x term / 3 on the rows the mask names, O's head gets the gradient,
            an all-False mask reads EXACTLY 0.0)
        ->  ``lmask = launch_stage(b)`` BEFORE the ego stages, ``b["launch"] = lmask`` AFTER them  ->  the batch the LOSS receives  ->  ``V8FMini.loss``: ``launch_along_term`` + ONE total line
        ->  summary.json / selfcheck.json / train_log.jsonl / the checkpoint (levers.launch_intent_w, launch_n, launch_eligible, launch_rows, launch_weight_share, launch_totals)

and that with the flag OFF every existing argv builds and trains bit-identically (K-BUILD): ``launch_intent_golden.json`` was written by the UNMODIFIED tip (d7208328) on the three argv families below;
this tree must reproduce it key for key and bit for bit.

The mask is a function of the LOGGED v0 and the GT future only (labels), taken before any ego stage rebuilds ``kin3``; section 6 proves it on a swapped batch. ``mutation.txt`` (tools/run_mutations.py)
deletes / breaks one rule at a time in a COPY of the tree and requires this file to go RED.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", REPO / "stack" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_dagger_train import COMMON, DAG, TINY, _log, _run, _selfcheck, _summary, _woken, dvb, gate_parts, pack  # noqa: E402,F401  (the fixtures: ONE trainer-written checkpoint, the DATA generator's bank, the synthetic world)

torch.set_num_threads(1)
GOLDEN = Path(__file__).with_name("launch_intent_golden.json")
WRITE_GOLDEN = os.environ.get("LAUNCH_INTENT_WRITE_GOLDEN") == "1"          # set ONLY on the unmodified tip, to (re)write the golden
TRAIN_P = REPO / "stack" / "scripts" / "v8f_mini_train.py"
MODEL_P = REPO / "stack" / "tanitad" / "models" / "v8f_mini.py"
FT = ["--lr", "1e-4", "--p-gt-start", "0.2", "--p-gt-end", "0.2"]
ORC_LAW = ["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0"]
W_A = 1.58926                      # arm A's dose at the MEASURED f = 49,618 / 306,566 (tools/measure_f.py, f_measure.json): 1 + W = (1 - f) / (2 f), 6 significant digits
F_MEASURED = 49618 / 306566


# ------------------------------------------------------------------------------------------------ helpers
def mk_gt(n, chords, junk=True):
    """GT waypoints [n, 30, 2] whose knots 29 -> 30 (indices 28 -> 29) differ by ``chords[i]``; every other knot is junk when ``junk`` (so only the last two matter)."""
    gt = torch.zeros(n, 30, 2, dtype=torch.float32)
    if junk:
        gt += torch.linspace(-50.0, 50.0, 60).reshape(1, 30, 2)
    for i, (cx, cy) in enumerate(chords):
        gt[i, 28] = torch.tensor([10.0, 3.0])
        gt[i, 29] = gt[i, 28] + torch.tensor([cx, cy])
    return gt


def real_batch(pk, n=48, wm="off", seed=0):
    """(model, a real collated batch of the trainer's own dataset, cfg, store). The residual head is woken so a change at O's output can be seen."""
    J = pk.J
    store = D.ClipStore(pk.world, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(pk.world / "native_tracks.pt"))
    wins = D.train_windows(store, J, D.select_train_clips(list(map(str, J["clip"])), 1.0))
    cfg = M.V8FMiniConfig(**{**TINY, "wm": wm})
    ds = D.V8FMiniWindows(store, J, wins, hmax=cfg.hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    mu, sd = M.cons_stats(J)
    model = _woken(M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd))
    return model, batch, cfg, store


def terms(model, batch, W, mask):
    model.cfg.launch_intent_w = W
    torch.manual_seed(7)
    batch = batch if mask is None else {**batch, "launch": mask}
    out = model.loss(batch, p_gt=0.5, gen=torch.Generator().manual_seed(11))
    return out["terms"], out


def hand_term(traj, gt, gok, launch):
    """The term at unit weight, written again from the definition with plain loops (numpy float64)."""
    traj, gt = traj.detach().double().numpy(), gt.double().numpy()
    tot = 0.0
    for b in range(traj.shape[0]):
        if gok[b] and launch[b]:
            tot += sum(math.sqrt((traj[b, k, 0] - gt[b, k, 0]) ** 2 + 1e-9) for k in range(0, 12)) / 30.0
    return tot / max(float(sum(bool(x) for x in gok)), 1.0)


# ------------------------------------------------------------------------------------------------ 1. the GT-launch mask against hand-computed labels
#           v0,    chord_x, chord_y, gok, expected   (v6 = |chord| / 0.2 m/s: 0.8 m -> 4.0, 0.5 -> 2.5, 2.0 -> 10.0, 1.4 -> 7.0, 1.39 -> 6.95)
MASK_CASES = [
    (2.0, 0.8, 0.0, 1, True),      # v0 < 6, 4.0 >= 3.0
    (2.0, 0.5, 0.0, 1, False),     # 2.5 < 3.0: the chord speed is too low
    (6.0, 2.0, 0.0, 1, False),     # v0 == 6.0 is NOT < 6 (strict), whatever the chord
    (5.99, 1.4, 0.0, 1, True),     # 7.0 >= 6.99
    (5.99, 1.39, 0.0, 1, False),   # 6.95 < 6.99
    (0.0, 0.25, 0.0, 1, True),     # 1.25 >= 1.0 (a standstill launch; the chord is 0.25 m so float32 absolute positions cannot decide it)
    (2.0, 0.0, 0.8, 1, True),      # a purely LATERAL chord: the norm, not the x component
    (float("nan"), 5.0, 0.0, 1, False),   # an undefined v0 is no launch
    (2.0, 0.8, 0.0, 0, False),     # no GT-valid label: no launch
    (3.0, 0.8, 0.0, 1, True),      # 4.0000000596 >= 3.0 + 1.0
    (7.5, 2.4, 0.0, 1, False),     # v0 >= 6
    (1.0, 0.1, 0.0, 1, False),     # a slower GT than v0 + 1
]


def test_launch_mask_against_hand_computed_labels():
    v0 = torch.tensor([c[0] for c in MASK_CASES], dtype=torch.float32)
    gt = mk_gt(len(MASK_CASES), [(c[1], c[2]) for c in MASK_CASES])
    gok = torch.tensor([c[3] for c in MASK_CASES], dtype=torch.bool)
    got = M.launch_mask(v0, gt, gok)
    assert got.dtype == torch.bool and tuple(got.shape) == (len(MASK_CASES),)
    assert got.tolist() == [c[4] for c in MASK_CASES]
    assert M.launch_mask(v0, gt, gok.float()).tolist() == got.tolist(), "the trainer passes gt_ok as a bool tensor and the loss as a float one: both spellings give the same mask"


def test_the_inequality_is_greater_or_equal_at_the_exact_boundary():
    """The registered inequality is ``>=``: a chord speed EXACTLY v0 + 1 is a launch. v0 is set from the function's own arithmetic so the equality is exact in float64."""
    gt = mk_gt(3, [(0.8, 0.0), (0.8, 0.0), (0.8, 0.0)])
    v6 = torch.linalg.norm((gt[:, 29] - gt[:, 28]).to(torch.float64), dim=-1) / M.LAUNCH_DT
    v0 = v6 - 1.0
    assert bool(((v0 + 1.0) == v6).all()), "the fixture makes the boundary exact"
    ok = torch.ones(3, dtype=torch.bool)
    assert M.launch_mask(v0, gt, ok).tolist() == [True, True, True], "chord speed == v0 + 1 is a launch (>=)"
    assert M.launch_mask(v0 + 1e-9, gt, ok).tolist() == [False, False, False], "and a hair above is not"
    assert M.launch_mask(v0 - 1e-9, gt, ok).tolist() == [True, True, True]


def test_the_v0_bound_is_strict_and_is_the_registered_six():
    assert (M.LAUNCH_V0_MAX, M.LAUNCH_DV, M.LAUNCH_DT, M.LAUNCH_KNOT_LO, M.LAUNCH_KNOT_HI) == (6.0, 1.0, 0.2, 1, 12)
    gt = mk_gt(3, [(3.0, 0.0)] * 3)                       # 15 m/s chord: only v0 can say no
    v0 = torch.tensor([5.9999, 6.0, 6.0001], dtype=torch.float64)
    assert M.launch_mask(v0, gt, torch.ones(3, dtype=torch.bool)).tolist() == [True, False, False]


def test_only_the_last_two_knots_set_the_chord():
    """Knots 29 and 30 (indices 28, 29) and nothing else: junk on every other knot moves nothing, a change of knot 29's position does."""
    chords = [(0.8, 0.0), (0.5, 0.0)]
    a = mk_gt(2, chords, junk=True)
    b = mk_gt(2, chords, junk=False)
    v0 = torch.tensor([2.0, 2.0])
    ok = torch.ones(2, dtype=torch.bool)
    assert M.launch_mask(v0, a, ok).tolist() == M.launch_mask(v0, b, ok).tolist() == [True, False]
    c = a.clone()
    c[:, 27] += 1000.0
    assert M.launch_mask(v0, c, ok).tolist() == [True, False], "knot 28 is not part of the chord"
    d = a.clone()
    d[1, 29, 0] += 0.3                                    # 0.5 -> 0.8 m
    assert M.launch_mask(v0, d, ok).tolist() == [True, True]


def test_the_mask_agrees_with_an_independent_numpy_spelling_on_random_labels():
    rng = np.random.default_rng(3)
    n = 4000
    v0 = rng.uniform(0.0, 12.0, n)
    gt = np.cumsum(rng.normal(0.0, 0.6, (n, 30, 2)), axis=1).astype(np.float32)
    gok = rng.random(n) > 0.1
    ref = gok & (v0 < 6.0) & (np.linalg.norm(gt[:, 29].astype(np.float64) - gt[:, 28].astype(np.float64), axis=-1) / 0.2 >= v0 + 1.0)
    got = M.launch_mask(torch.from_numpy(v0), torch.from_numpy(gt), torch.from_numpy(gok)).numpy()
    assert (got == ref).all() and 100 < ref.sum() < n - 100, "agreement on every row, and the fixture is not vacuous"


# ------------------------------------------------------------------------------------------------ 2. the up-weighted term: value, normaliser, and WHERE its gradient is exactly zero
def test_the_term_equals_a_hand_computed_value():
    rng = np.random.default_rng(5)
    traj = torch.from_numpy(rng.normal(0, 2, (6, 30, 2)).astype(np.float32))
    gt = torch.from_numpy(rng.normal(0, 2, (6, 30, 2)).astype(np.float32))
    gok = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    launch = torch.tensor([True, False, True, True, True, False])
    got = float(M.launch_along_term(traj, gt, gok, launch))
    want = hand_term(traj, gt, gok.tolist(), launch.tolist())
    assert launch.tolist()[3] and gok[3] == 0.0, "the fixture has a launch row without a label"
    assert abs(got - want) < 1e-6 * max(1.0, abs(want)), (got, want)
    assert want > 0.1


def test_the_denominator_is_every_gt_valid_row_not_the_launch_rows():
    """ADE's own normaliser: the same term over 4 GT-valid rows, 1 of them launch, is a QUARTER of that row's own value."""
    traj = torch.zeros(4, 30, 2)
    gt = torch.zeros(4, 30, 2)
    gt[:, :12, 0] = 2.0                                   # |dx| = 2 on knots 1..12
    one = float(M.launch_along_term(traj, gt, torch.ones(4), torch.tensor([True, False, False, False])))
    per_row = 12 * math.sqrt(4.0 + 1e-9) / 30.0
    assert abs(one - per_row / 4.0) < 1e-6
    allr = float(M.launch_along_term(traj, gt, torch.ones(4), torch.ones(4, dtype=torch.bool)))
    assert abs(allr - per_row) < 1e-6


def test_the_gradient_is_nonzero_on_launch_rows_along_knots_1_to_12_and_EXACTLY_zero_everywhere_else():
    rng = np.random.default_rng(1)
    traj = torch.from_numpy(rng.normal(0, 2, (6, 30, 2)).astype(np.float32)).requires_grad_(True)
    gt = torch.from_numpy(rng.normal(0, 2, (6, 30, 2)).astype(np.float32))
    gok = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    launch = torch.tensor([True, False, True, True, True, False])
    M.launch_along_term(traj, gt, gok, launch).backward()
    g = traj.grad
    live = torch.tensor([True, False, True, False, True, False])             # launch AND GT-valid
    assert bool((g[live][:, :12, 0].abs() > 0).all()), "the up-weighted slice: launch rows, knots 1-12, the along axis"
    assert float(g[:, 12:, :].abs().sum()) == 0.0, "knots 13-30 get exactly 0.0 (the horizon that is never executed at m = 10)"
    assert float(g[:, :, 1].abs().sum()) == 0.0, "the cross axis gets exactly 0.0 (an ALONG-track term)"
    assert float(g[~live].abs().sum()) == 0.0, "non-launch rows, and the launch row WITHOUT a GT label, get exactly 0.0"


def test_an_all_false_mask_reads_exactly_zero_in_value_and_gradient():
    traj = torch.randn(5, 30, 2, requires_grad=True)
    t = M.launch_along_term(traj, torch.randn(5, 30, 2), torch.ones(5), torch.zeros(5, dtype=torch.bool))
    t.backward()
    assert float(t) == 0.0 and float(traj.grad.abs().sum()) == 0.0


# ------------------------------------------------------------------------------------------------ 3. the BUILT loss (V8FMini.loss) on a real batch
def test_off_adds_no_term_and_no_key(pack):
    model, batch, cfg, _ = real_batch(pack)
    T, _ = terms(model, batch, 0.0, None)
    assert "launch_along" not in T and cfg.launch_intent_w == 0.0 and model.built_levers()["launch_intent_w"] == 0.0
    T2, _ = terms(model, batch, 0.0, torch.ones(batch["gt"].shape[0], dtype=torch.bool))
    assert "launch_along" not in T2 and float(T2["total"]) == float(T["total"]), "a mask handed to an OFF lever changes nothing"


def test_on_the_total_moves_by_exactly_w_traj_W_term_over_3_and_the_term_is_the_hand_value(pack):
    model, batch, cfg, _ = real_batch(pack)
    n = batch["gt"].shape[0]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[::2] = True
    T0, _ = terms(model, batch, 0.0, mask)
    T1, out = terms(model, batch, 1.5, mask)
    want_term = hand_term(out["traj30"], batch["gt"], batch["gt_ok"].tolist(), mask.tolist())
    assert float(T1["launch_along"]) > 0.0 and abs(float(T1["launch_along"]) - want_term) < 1e-5 * max(1.0, want_term), (float(T1["launch_along"]), want_term)
    delta = float(T1["total"]) - float(T0["total"])
    want = cfg.w_traj * 1.5 * float(T1["launch_along"]) / 3.0
    assert delta > 0 and abs(delta - want) < 1e-5 * max(1.0, abs(float(T0["total"]))), (delta, want)
    T3, _ = terms(model, batch, 3.0, mask)
    assert abs((float(T3["total"]) - float(T0["total"])) - 2.0 * delta) < 1e-4 * max(1.0, abs(float(T0["total"]))), "linear in W"
    model.cfg.w_traj = 2.5                                # the term rides ADE's own scale w_traj (1.0 by default, so a deleted factor would hide at the default)
    T0b, _ = terms(model, batch, 0.0, mask)
    T1b, _ = terms(model, batch, 1.5, mask)
    assert abs((float(T1b["total"]) - float(T0b["total"])) - 2.5 * 1.5 * float(T1b["launch_along"]) / 3.0) < 1e-5 * max(1.0, abs(float(T0b["total"]))), "scaled by w_traj"
    model.cfg.w_traj = cfg.w_traj
    for k in T0:
        if k != "total":
            assert float(T0[k]) == float(T1[k]), f"{k}: only the total and the new term move"


def test_the_gradient_to_O_is_nonzero_with_the_mask_and_EXACTLY_zero_with_an_all_false_mask(pack):
    model, batch, cfg, _ = real_batch(pack)
    n = batch["gt"].shape[0]
    on = torch.zeros(n, dtype=torch.bool)
    on[: n // 2] = True

    def grads(W, mask):
        model.zero_grad(set_to_none=True)
        T, _ = terms(model, batch, W, mask)
        T["total"].backward()
        return {k: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for k, p in model.named_parameters() if p.requires_grad}

    g0, g0b = grads(0.0, on), grads(0.0, on)
    assert sum(float((g0[k] - g0b[k]).abs().sum()) for k in g0) == 0.0, "the control: the probe is deterministic"
    gw, gn = grads(2.0, on), grads(2.0, torch.zeros(n, dtype=torch.bool))
    head = lambda g: sum(float((g[k] - g0[k]).abs().sum()) for k in g if k.startswith("head_o."))      # noqa: E731
    assert head(gw) > 0.0, "the term gives O's head a gradient"
    assert sum(float((gn[k] - g0[k]).abs().sum()) for k in g0) == 0.0, "an all-False mask: EVERY parameter's gradient is exactly the W = 0 one"


def test_a_missing_launch_key_is_refused_loudly(pack):
    model, batch, _, _ = real_batch(pack)
    with pytest.raises(ValueError, match="no `launch` mask"):
        terms(model, batch, 1.0, None)


def test_the_config_refuses_a_negative_or_non_finite_weight():
    for bad in (-0.1, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="launch_intent_w"):
            M.V8FMiniConfig(**{**TINY, "launch_intent_w": bad})
    assert M.V8FMiniConfig(**TINY).launch_intent_w == 0.0 and M.V8FMiniConfig(**{**TINY, "launch_intent_w": 1.5}).launch_intent_w == 1.5


# ------------------------------------------------------------------------------------------------ 4. G-DVB
def test_the_flag_has_a_gdvb_entry_and_the_parser_default_is_off():
    assert TR.FLAG_CHECKS["launch_intent_w"] == "cfg:launch_intent_w"
    assert TR.build_parser().get_default("launch_intent_w") == 0.0
    assert "launch_intent_w" in {act.dest for act in TR.build_parser()._actions}


def test_a_flag_without_a_gdvb_entry_is_refused_by_the_gate_itself(pack, monkeypatch):
    parts = gate_parts(pack, ["--launch-intent-w", "1.5"])
    assert [m for m in dvb(parts) if "launch" in m] == []
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != "launch_intent_w"})
    bad = dvb(parts)
    assert any("flags without a declared-vs-built entry" in m and "launch_intent_w" in m for m in bad), bad


def test_gdvb_refuses_an_argv_the_build_did_not_honour(pack):
    parts = gate_parts(pack, ["--launch-intent-w", "1.5"])
    a, cfg, model = parts[0], parts[1], parts[2]
    assert dvb(parts) == [] or all("launch" not in m for m in dvb(parts))
    assert model.cfg.launch_intent_w == 1.5 and model.built_levers()["launch_intent_w"] == 1.5
    model.cfg.launch_intent_w = 0.0                       # the build dropped the lever (the refcv6 class: declared, never built)
    bad = dvb(parts)
    assert any("lever launch_intent_w: declared 1.5 but built 0.0" in m for m in bad) and any("cfg.launch_intent_w: declared 1.5 but built 0.0" in m for m in bad), bad
    model.cfg.launch_intent_w = 2.5                       # built a different number
    assert any("declared 1.5 but built 2.5" in m for m in dvb(parts))


def test_gdvb_refuses_a_non_finite_or_negative_argv_value_and_a_weight_that_w_traj_zeroes(pack):
    parts = gate_parts(pack, ["--launch-intent-w", "1.5"])
    a, model = parts[0], parts[2]
    for bad_v in (-1.0, float("nan"), float("inf")):
        a.launch_intent_w = bad_v
        assert any("is not a finite number >= 0" in m for m in dvb(parts)), bad_v
    a.launch_intent_w = 1.5
    model.cfg.w_traj = 0.0
    assert any("a lever that is declared and does nothing" in m and "w_traj" in m for m in dvb(parts))


def test_main_refuses_a_negative_weight_before_anything_is_written(pack, tmp_path):
    with pytest.raises((ValueError, SystemExit)):
        TR.main(_run(tmp_path / "neg", ["--steps", "2", "--launch-intent-w", "-1.0"]))
    assert not (tmp_path / "neg" / "summary.json").exists() and not (tmp_path / "neg" / "ckpt_last.pt").exists()


# ------------------------------------------------------------------------------------------------ 5. G-LIVE
def test_the_probe_is_off_and_silent_at_w_zero(pack):
    model, batch, cfg, _ = real_batch(pack)
    r = TR.launch_intent_probe(model, batch, cfg)
    assert r["launch_intent"] == "off" and r["pass"] is True and (r["launch_n"], r["launch_eligible"]) == (0, 0)


def test_the_probe_passes_on_a_real_batch_and_records_the_natural_counts(pack, monkeypatch):
    monkeypatch.setattr(M, "LAUNCH_V0_MAX", 40.0)         # the synthetic world is mostly fast: widen v0 so the natural mask holds rows
    model, batch, cfg, _ = real_batch(pack)
    cfg.launch_intent_w = 1.5
    r = TR.launch_intent_probe(model, batch, cfg)
    assert r["pass"] is True and r["launch_n"] > 0 and r["launch_eligible"] == int(batch["gt_ok"].sum()) and r["launch_n"] <= r["launch_eligible"]
    assert r["head_o_grad_diff"] > 0 and r["natural_contribution"] > 0 and r["natural_head_o_grad_diff"] > 0
    assert r["control_repeat_total_diff"] == r["control_repeat_grad_diff"] == r["control_allfalse_term"] == r["control_allfalse_total_diff"] == r["control_allfalse_grad_diff"] == 0.0
    assert r["control_prior_grad_diff"] == 0.0 and r["term_at_zero_weight"] is None
    assert abs(r["contribution"] - r["w_traj_x_W_x_term_over_3"]) < 1e-3 * r["w_traj_x_W_x_term_over_3"] + 1e-6


def test_the_probe_sees_the_wire_even_when_the_probe_batch_holds_no_natural_launch_row(pack, monkeypatch):
    monkeypatch.setattr(M, "LAUNCH_V0_MAX", -1.0)         # nothing is a launch: the natural mask is empty
    model, batch, cfg, _ = real_batch(pack)
    cfg.launch_intent_w = 1.5
    r = TR.launch_intent_probe(model, batch, cfg)
    assert r["launch_n"] == 0 and r["pass"] is True and r["contribution"] > 0 and r["head_o_grad_diff"] > 0, "the FORCED mask carries the proof"


def test_deliberate_regression_a_term_that_never_reaches_the_loss_is_refused(pack, monkeypatch):
    model, batch, cfg, _ = real_batch(pack)
    cfg.launch_intent_w = 1.5
    monkeypatch.setattr(M, "launch_along_term", lambda traj, gt, gok, launch: (traj * 0.0).sum() + 1.0)   # a term with value but no dependence on O
    with pytest.raises(AssertionError, match="head|does not reach|O's head"):
        TR.launch_intent_probe(model, batch, cfg)


def test_deliberate_regression_a_term_that_ignores_the_mask_is_refused_by_the_all_false_control(pack, monkeypatch):
    model, batch, cfg, _ = real_batch(pack)
    cfg.launch_intent_w = 1.5
    orig = M.launch_along_term
    monkeypatch.setattr(M, "launch_along_term", lambda traj, gt, gok, launch: orig(traj, gt, gok, torch.ones_like(launch)))
    with pytest.raises(AssertionError, match="ALL-FALSE mask must read exactly 0.0"):
        TR.launch_intent_probe(model, batch, cfg)


def test_deliberate_regression_a_deleted_total_line_reads_zero_contribution_and_is_refused(pack, monkeypatch):
    model, batch, cfg, _ = real_batch(pack)
    cfg.launch_intent_w = 1.5
    orig_loss = M.V8FMini.loss

    def loss_without_the_line(self, b, p_gt=None, gen=None):
        out = orig_loss(self, b, p_gt, gen)
        W = self.cfg.launch_intent_w
        if W > 0.0:                                        # the total WITHOUT the launch line: subtract what the line added
            out["terms"]["total"] = out["terms"]["total"] - self.cfg.w_traj * W * out["terms"]["launch_along"] / 3.0
        return out
    monkeypatch.setattr(M.V8FMini, "loss", loss_without_the_line)
    with pytest.raises(AssertionError, match="does not reach the loss with its weight|contribution|moved by"):
        TR.launch_intent_probe(model, batch, cfg)


# ------------------------------------------------------------------------------------------------ 6. the REAL trainer: reachability, labels-not-own-state, records
@pytest.fixture
def lspy(monkeypatch):
    """Record what every call of the model's `loss` receives (the gate's probes call it too: the loop's calls are the LAST `steps`)."""
    calls = []
    orig = M.V8FMini.loss

    def loss(self, batch, p_gt=None, gen=None):
        calls.append({"launch": batch["launch"].clone() if "launch" in batch else None, "kin3": batch["kin3"].clone(), "gt": batch["gt"].clone(), "gt_ok": batch["gt_ok"].clone(),
                      "t": batch["t"].clone(), "ci": batch["ci"].clone(), "W": float(self.cfg.launch_intent_w)})
        return orig(self, batch, p_gt, gen)
    monkeypatch.setattr(M.V8FMini, "loss", loss)
    return calls


def test_a_real_run_reaches_the_loss_and_records_what_the_loss_received(pack, tmp_path, lspy, monkeypatch):
    monkeypatch.setattr(M, "LAUNCH_V0_MAX", 40.0)
    assert TR.main(_run(tmp_path / "on", ["--steps", "4", "--launch-intent-w", "1.5"])) == 0
    loop = lspy[-4:]
    assert all(c["launch"] is not None and c["W"] == 1.5 and c["launch"].dtype == torch.bool for c in loop), "the batch the LOSS receives carries the mask, and the model holds the declared W"
    n = sum(int(c["launch"].sum()) for c in loop)
    elig = sum(int(c["gt_ok"].sum()) for c in loop)
    rows = sum(int(c["gt_ok"].shape[0]) for c in loop)
    s, sc, log = _summary(tmp_path / "on"), _selfcheck(tmp_path / "on"), _log(tmp_path / "on")
    assert 0 < n < elig <= rows, "the run holds launch and non-launch rows (not a switch)"
    assert (s["launch_n"], s["launch_eligible"], s["launch_rows"]) == (n, elig, rows), "the run totals are recounted from the batches the loss received"
    assert s["launch_intent_w"] == 1.5 and s["levers"]["launch_intent_w"] == sc["levers"]["launch_intent_w"] == 1.5
    assert abs(s["launch_weight_share"] - 2.5 * n / (2.5 * n + (elig - n))) < 1e-12
    assert sc["launch_intent_probe"]["pass"] is True and sc["levers"]["launch_n"] == sc["launch_intent_probe"]["launch_n"] and sc["levers"]["launch_eligible"] == sc["launch_intent_probe"]["launch_eligible"]
    assert sc["launch_intent_probe"]["contribution"] > 0 and sc["launch_intent_probe"]["head_o_grad_diff"] > 0 and sc["launch_intent_probe"]["launch_eligible"] > 0, "the gate's probe really ran on the real probe batch"
    assert sc["levers"]["launch_intent"]["knots"] == [1, 12] and sc["levers"]["launch_intent"]["v0_max"] == 40.0
    assert sum(r["launch_n"] for r in log) == n and sum(r["launch_eligible"] for r in log) == elig and sum(r["launch_rows"] for r in log) == rows
    assert all("launch_along" in r and r["launch_along"] > 0 for r in log), "the term is logged beside the others"
    ck = torch.load(tmp_path / "on" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["launch_totals"] == {"rows": rows, "eligible": elig, "n": n} and ck["cfg"]["launch_intent_w"] == 1.5


def test_a_default_run_records_the_lever_off_and_adds_no_launch_key_to_the_log_summary_or_checkpoint(pack, tmp_path):
    assert TR.main(_run(tmp_path / "d", ["--steps", "3"])) == 0
    s, sc, log = _summary(tmp_path / "d"), _selfcheck(tmp_path / "d"), _log(tmp_path / "d")
    assert s["levers"]["launch_intent_w"] == 0.0 and sc["levers"]["launch_intent_w"] == 0.0 and sc["launch_intent_probe"]["launch_intent"] == "off"
    assert (sc["levers"]["launch_n"], sc["levers"]["launch_eligible"]) == (0, 0)
    assert not any(k.startswith("launch") for k in s), "a flag-off summary.json keeps its top-level key set"
    assert not any(k.startswith("launch") for r in log for k in r), "a default run's log is the pre-change log, key for key"
    ck = torch.load(tmp_path / "d" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert "launch_totals" not in ck


def test_naming_w_zero_is_the_absent_flag_bit_for_bit(pack, tmp_path):
    assert TR.main(_run(tmp_path / "a", ["--steps", "3"])) == 0
    assert TR.main(_run(tmp_path / "b", ["--steps", "3", "--launch-intent-w", "0.0"])) == 0
    a = torch.load(tmp_path / "a" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "b" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]) and a["cfg"] == b["cfg"]
    strip = lambda rows: [{k: v for k, v in r.items() if k != "sec"} for r in rows]       # noqa: E731
    assert strip(_log(tmp_path / "a")) == strip(_log(tmp_path / "b"))


def test_an_all_false_mask_trains_bit_for_bit_like_the_unweighted_run_and_a_real_mask_does_not(pack, tmp_path, monkeypatch):
    """The analytic identity: a mask that admits nothing adds 0.0 to the total and 0.0 to every gradient, so the weights equal W = 0's. A mask that admits rows must move them (not vacuous)."""
    assert TR.main(_run(tmp_path / "w0", ["--steps", "4"])) == 0
    monkeypatch.setattr(M, "LAUNCH_V0_MAX", -1.0)
    assert TR.main(_run(tmp_path / "empty", ["--steps", "4", "--launch-intent-w", "3.0"])) == 0
    monkeypatch.setattr(M, "LAUNCH_V0_MAX", 40.0)
    assert TR.main(_run(tmp_path / "real", ["--steps", "4", "--launch-intent-w", "3.0"])) == 0
    ld = lambda n: torch.load(tmp_path / n / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]      # noqa: E731
    w0, em, re_ = ld("w0"), ld("empty"), ld("real")
    assert all(torch.equal(w0[k], em[k]) for k in w0), "an all-False mask reproduces the unweighted run bit for bit"
    assert any(not torch.equal(w0[k], re_[k]) for k in w0), "a mask that admits rows changes the weights"
    assert _summary(tmp_path / "empty")["launch_n"] == 0 and _summary(tmp_path / "empty")["launch_eligible"] > 0


def test_the_mask_is_the_LOGGED_state_not_the_swapped_own_state(pack, tmp_path, lspy, monkeypatch):
    """Labels only. With the DAgger swap ON, a swapped sample's ``kin3`` is the model's OWN state; the launch mask must still be the one of the LOGGED v0 and the GT. Recomputed from the clip store."""
    monkeypatch.setattr(M, "LAUNCH_V0_MAX", 40.0)
    assert TR.main(_run(tmp_path / "sw", ["--steps", "4", *DAG(pack, 0.5, 3), "--launch-intent-w", "1.5"])) == 0
    assert _summary(tmp_path / "sw")["dagger_n"] > 0, "the swap fired"
    store = D.ClipStore(pack.world, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(pack.world / "native_tracks.pt"))
    clips = [str(c) for c in pack.J["clip"]]
    differ = 0
    for c in lspy[-4:]:
        v0_logged = torch.tensor([store.get(clips[int(ci)])["kin3"][int(t), 0] for ci, t in zip(c["ci"], c["t"])], dtype=torch.float64)
        want = M.launch_mask(v0_logged, c["gt"], c["gt_ok"])
        assert c["launch"].tolist() == want.tolist(), "the mask the loss received is the LOGGED one"
        own = M.launch_mask(c["kin3"][:, 0], c["gt"], c["gt_ok"])
        differ += int((own != want).sum())
    assert differ > 0, "non-vacuous: on this run a mask taken AFTER the swap would have differed from the logged one"


def test_the_mask_is_taken_before_the_stages_and_attached_after_them(pack, tmp_path, lspy, monkeypatch):
    """The stages return their own batch dict: the key must be on the dict the LOSS gets, with the swap or the perturbation ON."""
    monkeypatch.setattr(M, "LAUNCH_V0_MAX", 40.0)
    assert TR.main(_run(tmp_path / "e", ["--steps", "3", "--init-ckpt", str(pack.ck), *FT, "--ego-perturb-p", "0.5", *ORC_LAW, "--ego-perturb-seed", "7", "--launch-intent-w", "1.5"])) == 0
    assert all(c["launch"] is not None for c in lspy[-3:]) and _summary(tmp_path / "e")["ego_perturb_n"] > 0 and _summary(tmp_path / "e")["launch_n"] > 0


def test_a_resumed_run_keeps_counting_and_trains_what_an_uninterrupted_run_trained(pack, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "LAUNCH_V0_MAX", 40.0)
    common = [*COMMON, "--steps", "4", "--ckpt-every", "1", "--launch-intent-w", "1.5"]
    assert TR.main(["--out", str(tmp_path / "full"), *common]) == 0
    monkeypatch.setenv("V8F_TRAIN_CRASH_AT", "2")
    with pytest.raises(RuntimeError, match="V8F_TRAIN_CRASH_AT"):
        TR.main(["--out", str(tmp_path / "res"), *common])
    monkeypatch.delenv("V8F_TRAIN_CRASH_AT")
    assert TR.main(["--out", str(tmp_path / "res"), *common]) == 0
    a = torch.load(tmp_path / "full" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "res" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]), "the final weights are those of the uninterrupted run"
    assert a["launch_totals"] == b["launch_totals"] and a["launch_totals"]["n"] > 0
    sa, sb = _summary(tmp_path / "full"), _summary(tmp_path / "res")
    assert (sa["launch_n"], sa["launch_eligible"], sa["launch_rows"]) == (sb["launch_n"], sb["launch_eligible"], sb["launch_rows"])


# ------------------------------------------------------------------------------------------------ 7. the dose rule and the K-SHARE control formula
def test_the_dose_rule_gives_one_third_at_the_measured_f_and_the_registered_W():
    f = F_MEASURED
    assert abs(f - 0.16185095542232342) < 1e-12
    w = float(f"{(1 - f) / (2 * f) - 1:.6g}")
    assert w == W_A == 1.58926
    assert abs(TR.launch_weight_share(round(f * 1_000_000), 1_000_000, W_A) - 1 / 3) < 1e-5
    wb = float(f"{(1 - f) / (5 * f) - 1:.6g}")
    assert wb == 0.0357048 and abs(TR.launch_weight_share(round(f * 1_000_000), 1_000_000, wb) - 1 / 6) < 1e-5


def test_launch_weight_share_hand_values():
    assert abs(TR.launch_weight_share(10, 100, 1.0) - 20 / 110) < 1e-15
    assert TR.launch_weight_share(0, 100, 5.0) == 0.0 and TR.launch_weight_share(100, 100, 5.0) == 1.0
    assert TR.launch_weight_share(5, 0, 1.0) is None
    assert abs(TR.launch_weight_share(30727, 191203, W_A) - 0.3314508172724677) < 1e-9, "the simulated share of the sampler at seed 0 (tools/measure_f.py)"


# ------------------------------------------------------------------------------------------------ 8. K-BUILD: the flag-off path is the unmodified tip's, bit for bit
def argv_family(pk, name):
    """The three argv families whose flag-off behaviour must not move: DAG-OACC-style (bank + swap + the acceleration weight), ORC-style (the perturbation's law flags, no bank), plain M2b."""
    if name == "dag":
        return ["--steps", "4", *FT, *DAG(pk, 0.3, 3), "--w-acc", "49.6969"]
    if name == "orc":
        return ["--steps", "4", *FT, "--init-ckpt", str(pk.ck), "--ego-perturb-p", "0.3", *ORC_LAW, "--ego-perturb-seed", "7"]
    assert name == "m2b"
    return ["--steps", "4"]


def _tensor_digest(h, name, t):
    t = t.detach().cpu().contiguous()
    h.update(name.encode())
    h.update(str(t.dtype).encode())
    h.update(t.numpy().tobytes())


COUNTER_KEYS = ("ego_perturb_n", "ego_perturb_eligible", "ego_perturb_rows", "ego_perturb_nav_kept", "dagger_n", "dagger_eligible", "dagger_rows", "dagger_nav_kept")


def digest(out):
    """Everything a run's identity is made of: every logged number (minus wall-clock), the final weights, the optimiser state, the counters and the SETS of keys of each record."""
    out = Path(out)
    ck = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    hm, ho = hashlib.sha256(), hashlib.sha256()
    for k in sorted(ck["model"]):
        _tensor_digest(hm, k, ck["model"][k])
    for i in sorted(ck["opt"]["state"]):
        for kk in sorted(ck["opt"]["state"][i]):
            _tensor_digest(ho, f"{i}.{kk}", torch.as_tensor(ck["opt"]["state"][i][kk]))
    s = _summary(out)
    return {"log": [{k: v for k, v in r.items() if k != "sec"} for r in _log(out)], "model_sha256": hm.hexdigest(), "opt_sha256": ho.hexdigest(),
            "ego_totals": ck["ego_totals"], "dagger_totals": ck["dagger_totals"], "counters": {k: s[k] for k in COUNTER_KEYS},
            "summary_keys": sorted(s), "levers_keys": sorted(s["levers"]), "ckpt_keys": sorted(ck), "cfg": json.loads(json.dumps(ck["cfg"])), "selfcheck_keys": sorted(_selfcheck(out)),
            "levers_ego": s["levers"]["ego_perturb"], "levers_dagger_p": s["levers"]["dagger_p"]}


def bank_identity(pk):
    return [s["content_sha256"] for s in json.loads((pk.bank / "bank.json").read_text(encoding="utf-8"))["shards"]]


def test_flag_off_reproduces_the_unmodified_tip_bit_for_bit(pack, tmp_path):
    """K-BUILD: the golden was written by the unmodified tip (d7208328) for (a) a DAG-OACC-style argv, (b) an ORC-style argv and (c) plain M2b -- 4 CPU steps each on the trainer's synthetic world. With the
    flag absent this tree must give the same log (every loss, every logged key), the same final weights and optimiser state, the same counters, and the same SET of keys in summary.json and in the checkpoint
    (the checkpoint's ``cfg`` gains ONE key, ``launch_intent_w`` = 0.0; levers gain the four launch keys; selfcheck.json gains ``launch_intent_probe``)."""
    got = {}
    for name in ("dag", "orc", "m2b"):
        assert TR.main(_run(tmp_path / name, argv_family(pack, name))) == 0
        got[name] = digest(tmp_path / name)
    got["_fixture"] = {"bank_content_sha256": bank_identity(pack), "torch": torch.__version__}
    if WRITE_GOLDEN:
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        pytest.skip("golden written from the unmodified tip")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert want["_fixture"]["bank_content_sha256"] == bank_identity(pack), "the fixture bank is the golden's bank, record for record"
    for name in ("dag", "orc", "m2b"):
        g, w = got[name], want[name]
        assert g["log"] == w["log"] and len(w["log"]) == 4, f"{name}: every logged number and every logged key"
        assert (g["model_sha256"], g["opt_sha256"]) == (w["model_sha256"], w["opt_sha256"]), f"{name}: final weights and optimiser state, bit for bit (torch {want['_fixture']['torch']} wrote the golden, {torch.__version__} read it)"
        assert (g["ego_totals"], g["dagger_totals"], g["counters"]) == (w["ego_totals"], w["dagger_totals"], w["counters"]), f"{name}: the counters"
        assert g["summary_keys"] == w["summary_keys"] and g["ckpt_keys"] == w["ckpt_keys"], f"{name}: a flag-off run's summary.json / checkpoint carry no new top-level key"
        assert g["levers_ego"] == w["levers_ego"] and g["levers_dagger_p"] == w["levers_dagger_p"]
        assert {k: v for k, v in g["cfg"].items() if k != "launch_intent_w"} == w["cfg"] and g["cfg"]["launch_intent_w"] == 0.0 and "launch_intent_w" not in w["cfg"], f"{name}: the config gains only launch_intent_w = 0.0"
        assert set(g["levers_keys"]) - set(w["levers_keys"]) == {"launch_intent_w", "launch_intent", "launch_n", "launch_eligible"} and set(w["levers_keys"]) <= set(g["levers_keys"]), f"{name}: levers only GAIN the four launch keys"
        assert set(g["selfcheck_keys"]) - set(w["selfcheck_keys"]) == {"launch_intent_probe"} and set(w["selfcheck_keys"]) <= set(g["selfcheck_keys"])
    assert want["dag"]["counters"]["dagger_n"] > 0 and want["orc"]["counters"]["ego_perturb_n"] > 0, "the golden is not vacuous: the DAG family swapped rows and the ORC family perturbed rows"


# ------------------------------------------------------------------------------------------------ 9. the mutation anchors
def test_the_single_applying_lines_exist_exactly_once():
    """`tools/run_mutations.py` deletes / breaks these; an anchor that is not found exactly once would make a mutation test vacuous."""
    t = TRAIN_P.read_text(encoding="utf-8")
    m = MODEL_P.read_text(encoding="utf-8")
    assert t.count("                lmask = launch_stage(b)") == 1
    assert t.count('                b["launch"] = lmask') == 1
    assert t.count("    return M.launch_mask(batch[\"kin3\"][:, 0], batch[\"gt\"], batch[\"gt_ok\"])") == 1
    assert t.count("        lip = launch_intent_probe(model, probe, cfg) ") == 1
    assert t.count('            probe["launch"] = launch_stage(probe)') == 1
    assert t.count("    launch_on = cfg.launch_intent_w > 0.0") == 1
    assert t.count("    if a.launch_intent_w > 0.0 and not model.cfg.w_traj > 0.0:") == 1
    assert t.count('    "launch_intent_w": "cfg:launch_intent_w",') == 1
    assert t.count('    if built.get("launch_intent_w") != a.launch_intent_w:') == 1
    assert t.count('        summary["launch_n"], summary["launch_eligible"], summary["launch_rows"] = launch_tot["n"], launch_tot["eligible"], launch_tot["rows"]') == 1
    assert t.count('            blob["launch_totals"] = dict(launch_tot)') == 1
    assert t.count('c_["n"] += int(lmask.sum())') == 1
    assert m.count("        total = total + cfg.w_traj * cfg.launch_intent_w * T[\"launch_along\"] / 3.0 if cfg.launch_intent_w > 0.0 else total") == 1
    assert m.count("        if cfg.launch_intent_w > 0.0:  ") == 1
    assert m.count("(v6 >= v0 + LAUNCH_DV)") == 1 and m.count("(v0 < LAUNCH_V0_MAX)") == 1
