"""ACTLEVEL (`--w-act-level V`; note PREREG_V8F_PROOF_ADD37_N6 sections 2 and 5, controls v2 and v4): a level term inside the ACTCONS block.

level = mean over the valid (sample, start s) pairs of the consistency term and the read steps k = 5 and 10 of | gap_reference(s, k) - gap_label(s + k) | in metres, the reference read WITH gradient.
Tiny models (d 32) on CPU. Tolerances 1e-5 where floats from different code paths are compared; exact 0.0 only for structural zeros (no valid entry, a parameter outside the reach)."""
from __future__ import annotations

import copy
import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import act_cons as AC  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
GRID = ["--wm-state", "grid", "--wm-cell-dim", "16"]
LS = GRID + ["--lead-sup", "on", "--w-lead-sup", "1.0"]
MAXOFF = 1.5
STARTS = AC.valid_starts(16)
KS = tuple(hi for _, hi, _ in AC.T_READS)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("actlevel")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    D.synthetic_lead_labels(J, 0)
    ds = D.V8FMiniWindows(store, J, wins, hmax=16, lead_sup=True)
    idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    return {"batch": batch, "J": J}


def _built(world, extra=(), seed=0, **over):
    a = TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *extra])
    c0 = TR.make_config(a)
    keep = {k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**c0.__dict__, **keep, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(world["J"])
    torch.manual_seed(seed)
    return a, cfg, M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)


def _wake(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.wm.lead_head.mlp:
            if isinstance(m, torch.nn.Linear):
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.3)
        for p in model.wm.stack.predictors[0].parameters():
            if float(p.abs().sum()) == 0.0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model


def _offsets(n, seed=3):
    return AC.draw_offsets(seed, 0, n, MAXOFF)


def _mutant(old, new, name="act_cons_term"):
    src = textwrap.dedent(inspect.getsource(getattr(AC, name)))
    assert src.count(old) == 1, (old, src.count(old))
    ns = dict(vars(AC))
    exec(compile(src.replace(old, new), f"<mutant {name}>", "exec"), ns)
    return ns[name]


# ------------------------------------------------------------------------------------------ a planted kinematic world: the reference read is known by hand
class _KinPredictor(torch.nn.Module):
    """Exact constant-acceleration kinematics per step: state ch0 = ego displacement p [m], ch1 = speed v [m/s]; the embedded action is the scaled action."""
    context = 1

    def forward(self, zc, e):
        last = zc[:, -1]
        a = e[:, -1, 0] * M.WM_ACC_SCALE
        dt = AC.STEP_S
        out = last.clone()
        out[:, 0] = last[:, 0] + last[:, 1] * dt + 0.5 * a * dt * dt
        out[:, 1] = last[:, 1] + a * dt
        return out[:, None, :]


class _KinHead(torch.nn.Module):
    """gap = 30 m - p (the lead keeps its place, the ego moves), in the trainer's output unit (gap / 50 m)."""

    def forward(self, x):
        out = torch.zeros(x.shape[0], 1, 3)
        out[:, 0, 1] = (30.0 - x[:, -1, 0]) / M.LEAD_SUP_GAP_SCALE_M
        return out


def _planted(world):
    _, cfg, m = _built(world, LS)
    m.wm.stack.predictors[0] = _KinPredictor()
    m.wm.stack.act_embed = torch.nn.Identity()
    m.wm.lead_head = _KinHead()
    z1 = m.wm.z1

    def zz(adapt):
        z = 0.05 * z1(adapt)
        z = z.clone()
        z[..., 1] = z[..., 1] + 4.0                                                     # ego speed about 4 m/s: the gap moves about 0.8 m per step, so a shifted frame is visible
        return z

    m.wm.z1 = zz
    return m


def _hand_reference(m, batch, adapt):
    """The reference read by an independent route (public ``HJEPAStack.rollout``, the head as a plain module, python loops): {(b, i, j): gap [m]} for i = start index, j = read index."""
    wm, st, head = m.wm, m.wm.stack, m.wm.lead_head
    out = {}
    with torch.no_grad():
        z = wm.z1(adapt)
        a = wm.scale_actions(batch["act"].float())[:, :-1]
        for b in range(z.shape[0]):
            for i, s in enumerate(STARTS):
                r = st.rollout(1, z[b, s][None], a[b, s:s + AC.ROLL_K][None], context=1)[0]
                for j, (lo, hi, _) in enumerate(AC.T_READS):
                    out[(b, i, j)] = float(head(r[lo:hi][None])[0, 0, 1]) * 50.0
    return out


def _batch_with_labels(batch, labels_by_frame):
    """A copy of the batch whose labels at the frames in ``labels_by_frame`` ({(b, f): gap}) are those gaps (presence 1, frame_ok True); all other label entries untouched."""
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for (b, f), g in labels_by_frame.items():
        b2["lead_gapm"][b, f] = g
        b2["lead_pres"][b, f] = 1.0
        b2["frame_ok"][b, f] = True
    return b2


def test_planted_level_reads_abs_c_and_the_frame_rule_is_s_plus_k(world):
    m = _planted(world)
    batch = world["batch"]
    adapt = m._encode(batch)
    d = _offsets(8)
    ref = _hand_reference(m, batch, adapt)
    base = AC.act_cons_term(m, adapt, batch, d, MAXOFF, level=True)
    nv = int(base["n_valid"])
    assert nv > 2
    for c in (2.5, -2.5):
        lab = {(b, s + k): ref[(b, i, j)] + c for b in range(8) for i, s in enumerate(STARTS) for j, k in enumerate(KS)}
        assert len({f for (_, f) in lab}) == 6                                         # the six frames s + k are distinct: one planted value per entry
        r = AC.act_cons_term(m, adapt, _batch_with_labels(batch, lab), d, MAXOFF, level=True)
        assert int(r["n_level"]) == 2 * nv                                             # every valid (pair, step) entry is valid
        assert abs(float(r["level"]) - abs(c)) < 1e-5, (float(r["level"]), c)
    # the same plant one frame EARLY (frame s + k - 1): the level does not read |c|
    c = 2.5
    early = {(b, s + k - 1): ref[(b, i, j)] + c for b in range(8) for i, s in enumerate(STARTS) for j, k in enumerate(KS)}
    r_e = AC.act_cons_term(m, adapt, _batch_with_labels(batch, early), d, MAXOFF, level=True)
    assert abs(float(r_e["level"]) - c) > 0.1, float(r_e["level"])
    # and one frame LATE (s + k + 1 where it exists): not |c| either
    late = {(b, s + k + 1): ref[(b, i, j)] + c for b in range(8) for i, s in enumerate(STARTS) for j, k in enumerate(KS) if s + k + 1 < 16}
    r_l = AC.act_cons_term(m, adapt, _batch_with_labels(batch, late), d, MAXOFF, level=True)
    assert abs(float(r_l["level"]) - c) > 0.1, float(r_l["level"])


def test_step_k_of_roll_pairs_is_frame_s_plus_k():
    """The frame rule from source: roll_pairs targets z[:, s + 1 : s + k + 1], so rolled step j (0-based) is frame s + 1 + j; the read slice [lo:hi] ends at step hi (1-based) = frame s + hi."""
    from tanitad.models.hjepa import HJEPAStack
    assert "z[:, s + 1:s + k + 1]" in inspect.getsource(HJEPAStack.roll_pairs)
    assert AC.T_READS == ((0, 5, 1.0), (5, 10, 2.0))
    gaps, ok = AC.level_label_mask({"lead_pres": torch.ones(1, 16), "lead_gapm": torch.arange(16, dtype=torch.float32)[None] + 1.0, "frame_ok": torch.ones(1, 16, dtype=torch.bool)}, STARTS, KS)
    for i, s in enumerate(STARTS):
        for j, k in enumerate(KS):
            assert float(gaps[i, j, 0]) == float(s + k) + 1.0 and bool(ok[i, j, 0])


# ------------------------------------------------------------------------------------------ gradient structure
def _level_grads(m, batch, d, **kw):
    mm = copy.deepcopy(m)
    ad = mm._encode(batch).detach().requires_grad_(True)
    mm.zero_grad(set_to_none=True)
    r = AC.act_cons_term(mm, ad, batch, d, MAXOFF, level=True, **kw)
    if r["level"].requires_grad:
        r["level"].backward()
    return mm, ad, r


def test_level_gradient_reaches_predictor_and_action_embed_only(world):
    _, cfg, m = _built(world, LS)
    _wake(m)
    batch = world["batch"]
    mm, ad, r = _level_grads(m, batch, _offsets(8))
    names = dict(mm.named_parameters())
    assert float(r["n_level"]) > 0 and float(r["level"].detach()) > 0.0 and r["level"].requires_grad
    pred = sum(float(p.grad.abs().sum()) for n, p in names.items() if n.startswith("wm.stack.predictors.0.") and p.grad is not None)
    emb = sum(float(p.grad.abs().sum()) for n, p in names.items() if n.startswith("wm.stack.act_embed.") and p.grad is not None)
    assert pred > 0.0 and emb > 0.0
    others = {n: float(p.grad.abs().sum()) for n, p in names.items() if p.grad is not None and not n.startswith(("wm.stack.predictors.0.", "wm.stack.act_embed."))}
    assert others == {}                                                                # not one other parameter holds a gradient
    assert all(p.grad is None for n, p in names.items() if n.startswith("wm.lead_head."))      # the lead head: exactly nothing
    assert ad.grad is None or float(ad.grad.abs().sum()) == 0.0                        # the start state / encoder path: exactly 0.0


def test_level_does_not_use_the_offset_and_all_invalid_control_reads_exactly_zero(world):
    _, cfg, m = _built(world, LS)
    _wake(m)
    batch = world["batch"]
    r1 = AC.act_cons_term(m, m._encode(batch), batch, _offsets(8, 3), MAXOFF, level=True)
    r2 = AC.act_cons_term(m, m._encode(batch), batch, _offsets(8, 4), MAXOFF, level=True)
    assert abs(float(r1["level"]) - float(r2["level"])) < 1e-6 and float(r1["n_level"]) == float(r2["n_level"])        # the reference rollout has no offset
    b0 = dict(batch)
    b0["lead_pres"] = torch.zeros_like(batch["lead_pres"])
    r0 = AC.act_cons_term(m, m._encode(batch), b0, _offsets(8), MAXOFF, level=True)
    assert float(r0["level"]) == 0.0 and not r0["level"].requires_grad and float(r0["n_level"]) == 0.0 and float(r0["n_valid"]) == 0.0
    assert set(r0) == {"term", "n_valid", "n_rows", "level", "n_level"}


def test_consistency_term_is_unchanged_by_the_level_flag(world):
    """The consistency term reads the reference detached: its value and its gradient with level on equal those with level off (1e-5)."""
    _, cfg, m = _built(world, LS)
    _wake(m)
    batch = world["batch"]
    d = _offsets(8)

    def run(level):
        mm = copy.deepcopy(m)
        mm.zero_grad(set_to_none=True)
        r = AC.act_cons_term(mm, mm._encode(batch), batch, d, MAXOFF, level=level)
        r["term"].backward()
        return r, torch.cat([p.grad.flatten() for n, p in mm.named_parameters() if p.grad is not None])

    r_off, g_off = run(False)
    r_on, g_on = run(True)
    assert abs(float(r_off["term"]) - float(r_on["term"])) < 1e-5 and float(r_off["n_valid"]) == float(r_on["n_valid"])
    assert g_off.shape == g_on.shape and float((g_off - g_on).abs().max()) < 1e-5 * max(1.0, float(g_off.abs().max()))


# ------------------------------------------------------------------------------------------ invalid-label masking
def test_invalid_level_labels_are_masked(world):
    _, cfg, m = _built(world, LS)
    _wake(m)
    batch = world["batch"]
    d = _offsets(8)
    adapt = m._encode(batch)
    full = AC.act_cons_term(m, adapt, batch, d, MAXOFF, level=True)
    nv = int(full["n_valid"])
    assert int(full["n_level"]) > 0
    late = slice(5, 16)                                                                # frames 5 and later: never a start frame (starts 0, 2, 4), so the consistency term keeps its pairs

    def run(**chg):
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        for k, fn in chg.items():
            b2[k][:, late] = fn(b2[k][:, late])
        return AC.act_cons_term(m, adapt, b2, d, MAXOFF, level=True)

    for tag, chg in (("no presence", {"lead_pres": torch.zeros_like}), ("gap NaN", {"lead_gapm": lambda x: torch.full_like(x, float("nan"))}), ("gap 61", {"lead_gapm": lambda x: torch.full_like(x, 61.0)}),
                     ("gap negative", {"lead_gapm": lambda x: torch.full_like(x, -0.5)}), ("frame_ok False", {"frame_ok": torch.zeros_like})):
        r = run(**chg)
        assert int(r["n_valid"]) == nv, tag                                            # the consistency pairs are untouched
        assert float(r["n_level"]) == 0.0 and float(r["level"]) == 0.0 and not r["level"].requires_grad, tag
    # one frame only: frame 14 is s + k for (s = 4, k = 10) alone
    b3 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b3["lead_pres"][:, 14] = 0.0
    vm = AC.valid_mask(b3, STARTS, MAXOFF)
    r3 = AC.act_cons_term(m, adapt, b3, d, MAXOFF, level=True)
    n_s4 = int(vm[:, STARTS.index(4)].sum())
    assert n_s4 > 0 and int(r3["n_level"]) == int(full["n_level"]) - n_s4
    # the level is the mean over the entries that remain: a NaN label at an invalid entry does not enter
    b4 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b4["lead_gapm"][:, 14] = float("nan")
    r4 = AC.act_cons_term(m, adapt, b4, d, MAXOFF, level=True)
    assert np.isfinite(float(r4["level"])) and int(r4["n_level"]) == int(r3["n_level"])
    r4.get("level").backward()                                                         # a finite gradient: no NaN reached the graph
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


# ------------------------------------------------------------------------------------------ V = 0 and the gate
def test_level_off_returns_the_old_key_set_and_record(world):
    _, cfg, m = _built(world, LS)
    _wake(m)
    batch = world["batch"]
    r = AC.act_cons_term(m, m._encode(batch), batch, _offsets(8), MAXOFF)
    assert set(r) == {"term", "n_valid", "n_rows"}
    rec = AC.build_record(1.0, 1.5, cfg, m)
    assert set(rec) == {"w_act_cons", "act_cons_max", "k", "starts", "reads", "speed_bound_mps", "gap_range_m", "acc_scale", "lead_hist", "l1_act", "gap_scale_m", "head_params_detached", "start_state_detached"}
    rec_l = AC.build_record(1.0, 1.5, cfg, m, 0.01)
    assert set(rec_l) - set(rec) == {"w_act_level", "level_reads"} and rec_l["w_act_level"] == 0.01
    a = TR.build_parser().parse_args(["--out", "x", *LS])
    assert a.w_act_level == 0.0 and "w_act_level" in TR.FLAG_CHECKS


def test_gate_refusals_of_the_level_weight(world):
    _, cfg, m = _built(world, LS)
    rec0 = AC.build_record(0.001, 1.5, cfg, m)
    rec1 = AC.build_record(0.001, 1.5, cfg, m, 0.01)
    mk = lambda *x: TR.build_parser().parse_args(["--out", "x", *LS, *x])  # noqa: E731
    a_c = mk("--w-act-cons", "0.001")
    a_cl = mk("--w-act-cons", "0.001", "--w-act-level", "0.01")
    assert TR.bad_act_cons(a_c, cfg, m, {"act_cons": rec0}) == []                      # V = 0 as before
    assert TR.bad_act_cons(a_cl, cfg, m, {"act_cons": rec1}) == []                     # declared == built
    # refusal 1: V > 0 with W == 0
    assert any("--w-act-cons 0" in b and "w_act_level" in b for b in TR.bad_act_cons(mk("--w-act-level", "0.01"), cfg, m, {}))
    # refusal 2: V < 0
    assert any("w_act_level" in b and ">= 0" in b for b in TR.bad_act_cons(mk("--w-act-cons", "0.001", "--w-act-level", "-0.1"), cfg, m, {"act_cons": rec0}))
    # declared != built, both directions
    assert any("w_act_level: declared 0.01 but built" in b for b in TR.bad_act_cons(a_cl, cfg, m, {"act_cons": rec0}))
    assert any("w_act_level: declared 0.01 but built 0.5" in b for b in TR.bad_act_cons(a_cl, cfg, m, {"act_cons": {**rec1, "w_act_level": 0.5}}))
    assert any("w_act_level 0 declared but" in b for b in TR.bad_act_cons(a_c, cfg, m, {"act_cons": rec1}))


def test_probe_reads_level_gradients_and_goes_red_when_the_head_is_attached(world, monkeypatch):
    _, cfg, m = _built(world, LS)
    a = TR.build_parser().parse_args(["--out", "x", *LS, "--w-act-cons", "1.0", "--w-act-level", "0.01"])
    rec = TR.act_cons_probe(m, world["batch"], cfg, a)
    assert rec["pass"] and rec["w_act_level"] == 0.01 and rec["n_level"] > 0 and np.isfinite(rec["level"])
    assert rec["level_grad_norm_predictor"] > 0.0 and rec["level_grad_norm_action_embed"] > 0.0
    assert rec["level_grad_norm_lead_head"] == 0.0 and rec["level_grad_norm_state_projection"] == 0.0 and rec["level_grad_abs_sum_outside_predictor_and_embed"] == 0.0
    assert rec["control_all_invalid_level"] == 0.0 and not rec["control_all_invalid_level_has_graph"] and rec["control_all_invalid_n_level"] == 0.0
    a0 = TR.build_parser().parse_args(["--out", "x", *LS, "--w-act-cons", "1.0"])
    r0 = TR.act_cons_probe(m, world["batch"], cfg, a0)
    assert r0["pass"] and not [k for k in r0 if "level" in k]                          # V = 0: no level key in the probe record
    mut = _mutant("params = {n: p.detach() for n, p in head.named_parameters()}", "params = {n: p for n, p in head.named_parameters()}")
    monkeypatch.setattr(AC, "act_cons_term", mut)
    with pytest.raises(AssertionError, match="exactly 0.0"):
        TR.act_cons_probe(m, world["batch"], cfg, a)


def test_level_mutations_in_memory_go_red(world):
    m = _planted(world)
    batch = world["batch"]
    adapt = m._encode(batch)
    d = _offsets(8)
    ref = _hand_reference(m, batch, adapt)
    lab = {(b, s + k): ref[(b, i, j)] + 2.5 for b in range(8) for i, s in enumerate(STARTS) for j, k in enumerate(KS)}
    bt = _batch_with_labels(batch, lab)
    assert abs(float(AC.act_cons_term(m, adapt, bt, d, MAXOFF, level=True)["level"]) - 2.5) < 1e-5     # the control: the real function reads |c|
    src = textwrap.dedent(inspect.getsource(AC.level_label_mask))
    assert src.count("f = s + k\n") == 1
    ns = dict(vars(AC))
    exec(compile(src.replace("f = s + k\n", "f = s + k - 1\n"), "<mutant level_label_mask>", "exec"), ns)
    old = AC.level_label_mask
    try:
        AC.level_label_mask = ns["level_label_mask"]
        assert abs(float(AC.act_cons_term(m, adapt, bt, d, MAXOFF, level=True)["level"]) - 2.5) > 0.1      # label one frame early: RED
    finally:
        AC.level_label_mask = old


def test_the_loop_lines_of_the_level_exist_once():
    src = Path(TR.__file__).read_text(encoding="utf-8")
    assert src.count('T["total"] = T["total"] + a.w_act_level * ac_r["level"]') == 1
    assert src.count('T["act_level"], T["act_level_n"] = ac_r["level"].detach(), ac_r["n_level"].detach()') == 1
    assert src.count('levers["w_act_level"] = info["act_cons"]["w_act_level"]') == 1
    assert src.count("AC.build_record(a.w_act_cons, a.act_cons_max, cfg, model, a.w_act_level)") == 1
    assert src.count("ZZACT-LEVEL-NOT-APPLIEDZZ") == 1
    assert src.count('summary["w_act_level"], summary["act_level_n_valid"]') == 1
    # every key write sits under an `if a.w_act_level > 0.0:` guard: the line above it (or its block header) names the condition
    lines = src.splitlines()
    for key in ('T["act_level"]', 'levers["w_act_level"]', 'summary["w_act_level"]'):
        i = next(n for n, ln in enumerate(lines) if key in ln)
        assert "if a.w_act_level > 0.0" in lines[i - 1] or "if a.w_act_level > 0.0" in lines[i - 2], key
