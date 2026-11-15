"""RULEOWNL1 (--rule-own-form {sq,l1}; note PREREG_V8F_PROOF_ADD34_N1 sections 2 and 4): the own-plan term in a linear form.

sq (default) is the term of card ADD34 bit for bit; l1 = per sample the mean over the 30 knots of the Euclidean distance in metres between the predicted knot and its cap, batch term = mean over the masked samples
(exactly 0.0 when none), times --w-rule-own. The gradient is finite and exactly 0 at a knot that sits on its cap. Tiny models on CPU; the mutation test deletes the line that selects the linear form."""
from __future__ import annotations

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

from tanitad.data import dist_rule as R  # noqa: E402
from tanitad.data import rule_own as ROW  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

GRID = ["--wm-state", "grid", "--wm-cell-dim", "16"]
SYN = ["--synthetic", "--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", *GRID, "--exclude", "1:3", "--threads", "1"]
TINY = ["--batch", "8", "--d", "32", "--n-q", "2"]


def _straight(n, speed):
    g = np.zeros((n, 30, 2), np.float32)
    g[:, :, 0] = np.arange(1, 31)[None, :] * 0.2 * speed
    return g


def _inp(n, gap, v0=10.0, v_L=0.0, mask=None):
    return {"mask": np.ones(n, bool) if mask is None else np.asarray(mask, bool), "v0": np.full(n, v0), "gap": np.full(n, float(gap)), "v_L": np.full(n, v_L), "n_rows": n}


def _args(*extra):
    return TR.build_parser().parse_args(["--out", "x", *extra])


# ---------------------------------------------------------------------------------------------- definition
def test_flag_default_is_sq_and_has_its_gate_entry():
    a = _args()
    assert a.rule_own_form == "sq" and TR.FLAG_CHECKS["rule_own_form"] == "data:rule_own_form"
    assert ROW.FORMS == ("sq", "l1")
    assert ROW.RuleOwn(0.0, 1.0).form == "sq"
    with pytest.raises(ValueError):
        ROW.RuleOwn(0.0, 1.0, form="l1")                                  # the form is a form of the --w-rule-own term
    with pytest.raises(ValueError):
        ROW.RuleOwn(1.0, 1.0, form="l2")


def test_sq_default_is_the_earlier_term_bit_for_bit():
    plan = torch.as_tensor(_straight(3, 10.0)) + 0.01 * torch.randn(3, 30, 2)
    inp = _inp(3, 8.0, v0=10.0, v_L=2.0)
    t_def, _ = ROW.own_term(plan, inp)
    t_sq, _ = ROW.own_term(plan, inp, "sq")
    cap = torch.as_tensor(ROW.cap_of_plan(plan.numpy(), inp["v0"], inp["gap"], inp["v_L"])).float()
    old = ((plan - cap) ** 2).sum(-1).mean(1).mean()
    assert torch.equal(t_def, t_sq) and torch.equal(t_sq, old)


def test_l1_cap_by_hand_and_term_is_the_mean_distance_in_metres():
    plan = torch.as_tensor(_straight(3, 10.0)) + 0.01 * torch.randn(3, 30, 2)
    plan.requires_grad_(True)
    inp = _inp(3, 8.0, v0=10.0, v_L=2.0)
    term, info = ROW.own_term(plan, inp, "l1")
    by_hand = R.distance_rule(plan.detach().double().numpy(), np.full(3, 10.0), np.ones(3, bool), np.full(3, 8.0), np.full(3, 8.0), np.full(3, 2.0), m=10, floor_lifts=False)
    cap = torch.as_tensor(by_hand.plan).float()
    assert torch.equal(torch.as_tensor(ROW.cap_of_plan(plan.detach().numpy(), inp["v0"], inp["gap"], inp["v_L"])).float(), cap)
    d = np.sqrt(((plan.detach().double().numpy() - cap.double().numpy()) ** 2).sum(-1))              # [3, 30] metres
    hand = d.mean(1).mean()
    assert info["n_lead"] == 3 and info["n_active"] == 3 and float(term.detach()) > 0.0
    assert abs(float(term.detach()) - hand) <= 1e-6 * max(1.0, hand)
    term.backward()
    g = plan.grad.numpy()
    diff = (plan.detach() - cap).double().numpy()
    assert np.allclose(g, diff / np.maximum(np.sqrt((diff ** 2).sum(-1, keepdims=True)), 1e-12) / (3 * 30), atol=1e-7)      # d term / d plan = unit vector / (n x knots)
    assert np.isfinite(g).all()


def test_l1_rows_without_a_lead_or_with_a_non_finite_speed_contribute_exactly_zero():
    n = 5
    ok = np.ones(n, bool)
    pres = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    gap = np.array([8.0, 8.0, np.nan, 8.0, 8.0])
    v0 = np.array([10.0, 10.0, 10.0, np.nan, 10.0])
    vl = np.array([10.0, 10.0, 10.0, 10.0, np.nan])
    inp = ROW.own_inputs(ok, ok, v0, vl, pres, gap, np.zeros(n))
    assert inp["mask"].tolist() == [True, False, False, False, False]
    plan = torch.as_tensor(_straight(n, 10.0)).requires_grad_(True)
    term, info = ROW.own_term(plan, inp, "l1")
    term.backward()
    assert info["n_lead"] == 1 and info["n_active"] == 1 and float(term) > 0.0
    assert float(plan.grad[1:].abs().sum()) == 0.0 and float(plan.grad[0].abs().sum()) > 0.0
    plan2 = torch.as_tensor(_straight(n, 10.0)).requires_grad_(True)
    t0, i0 = ROW.own_term(plan2, ROW.own_inputs(ok, ok, v0, vl, pres * 0.0, gap, np.zeros(n)), "l1")
    assert i0["n_lead"] == 0 and float(t0) == 0.0 and not t0.requires_grad


# ---------------------------------------------------------------------------------------------- reach and exact zeros
def _reach_l1(mod):
    """True when the l1 term behaves as the linear form: on a plan that runs into the lead it equals the hand mean distance (metres), not the squared one."""
    run = torch.as_tensor(_straight(4, 10.0)).requires_grad_(True)
    T = {"total": (run ** 2).sum() * 0.0 + 1.0}
    inp = _inp(4, 8.0, v0=10.0, v_L=0.0)
    info = mod.add_own_term(T, run, inp, 2.0, "l1")
    T["total"].backward()
    cap = mod.cap_of_plan(run.detach().numpy(), inp["v0"], inp["gap"], inp["v_L"])
    hand = float(np.sqrt(((run.detach().double().numpy() - cap) ** 2).sum(-1)).mean())
    return bool(info["term"] > 0.0 and abs(info["term"] - hand) <= 1e-5 * hand and float(run.grad.abs().sum()) > 0.0 and np.isfinite(run.grad.numpy()).all()
                and float(T["total"]) == pytest.approx(1.0 + 2.0 * hand, rel=1e-5))


def test_l1_reach_into_the_lead_inside_the_rule_and_without_a_lead():
    assert _reach_l1(ROW) is True
    slow = torch.as_tensor(_straight(4, 0.5)).requires_grad_(True)                          # a crawl, lead 40 m away: inside the rule
    T2 = {"total": (slow ** 2).sum() * 0.0 + 1.0}
    i2 = ROW.add_own_term(T2, slow, _inp(4, 40.0, v0=0.5, v_L=0.0), 2.0, "l1")
    T2["total"].backward()
    assert i2["n_lead"] == 4 and i2["n_active"] == 0 and i2["term"] == 0.0 and float(slow.grad.abs().sum()) == 0.0 and float(T2["total"]) == 1.0 and np.isfinite(slow.grad.numpy()).all()
    none = torch.as_tensor(_straight(4, 10.0)).requires_grad_(True)
    T3 = {"total": (none ** 2).sum() * 0.0 + 1.0}
    i3 = ROW.add_own_term(T3, none, _inp(4, 8.0, mask=np.zeros(4, bool)), 2.0, "l1")
    T3["total"].backward()
    assert i3["n_lead"] == 0 and i3["term"] == 0.0 and float(none.grad.abs().sum()) == 0.0 and float(T3["total"]) == 1.0


def test_l1_mixed_batch_is_partial():
    plan = np.concatenate([_straight(3, 12.0), _straight(3, 0.5)], 0)
    inp = {"mask": np.ones(6, bool), "v0": np.array([12.0] * 3 + [0.5] * 3), "gap": np.array([8.0] * 3 + [40.0] * 3), "v_L": np.zeros(6), "n_rows": 6}
    t = torch.as_tensor(plan).requires_grad_(True)
    term, info = ROW.own_term(t, inp, "l1")
    term.backward()
    print("l1 partial: n_lead", info["n_lead"], "n_active", info["n_active"])
    assert info["n_lead"] == 6 and info["n_active"] == 3 and float(term) > 0.0
    assert float(t.grad[:3].abs().sum()) > 0.0 and float(t.grad[3:].abs().sum()) == 0.0


def test_l1_gradient_is_finite_with_exact_zeros_at_knots_on_the_cap():
    inp = _inp(2, 8.0, v0=12.0, v_L=0.0)
    cap0 = torch.as_tensor(ROW.cap_of_plan(_straight(2, 12.0), inp["v0"], inp["gap"], inp["v_L"])).float()
    plan = cap0.clone()                                                                     # the first 20 knots are a plan that already keeps the rule ...
    plan[:, 20:, 0] += 3.0                                                                  # ... and the last 10 push forward again
    cap = torch.as_tensor(ROW.cap_of_plan(plan.numpy(), inp["v0"], inp["gap"], inp["v_L"])).float()
    on = (plan == cap).all(-1)                                                              # [2, 30] knots that sit exactly on their cap
    assert bool(on.any()) and bool((~on).any()), "the constructed plan must have knots on and off the cap"
    p = plan.clone().requires_grad_(True)
    term, info = ROW.own_term(p, inp, "l1")
    term.backward()
    g = p.grad
    assert torch.isfinite(g).all() and torch.isfinite(term)
    assert float(g[on].abs().sum()) == 0.0 and float(g[~on].abs().sum()) > 0.0
    # the same plan under a plain sqrt gives NaN at the knots on the cap: the exact construction is needed
    q = plan.clone().requires_grad_(True)
    ((q - cap) ** 2).sum(-1).sqrt().mean().backward()
    assert not torch.isfinite(q.grad).all()
    # a plan that keeps the rule (equals its cap everywhere): term 0.0, gradient 0 everywhere, finite
    z = torch.as_tensor(_straight(2, 0.5)).requires_grad_(True)
    t0, i0 = ROW.own_term(z, _inp(2, 40.0, v0=0.5, v_L=0.0), "l1")
    t0.backward()
    assert i0["n_active"] == 0 and float(t0) == 0.0 and float(z.grad.abs().sum()) == 0.0 and torch.isfinite(z.grad).all()


# ---------------------------------------------------------------------------------------------- mutation
def _load_mutated(old, new):
    src = Path(ROW.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1
    mod = types.ModuleType("rule_own_mut_l1")
    mod.__file__ = str(ROW.__file__)
    exec(compile(src.replace(old, new), "rule_own_mut_l1", "exec"), mod.__dict__)
    return mod


def test_mutation_deleting_the_line_that_selects_the_linear_form_goes_red():
    assert _reach_l1(ROW) is True                                                            # the green run
    red = _load_mutated("        per = knot_distance(pl - cap).mean(1)", "        per = ((pl - cap) ** 2).sum(-1).mean(1)")      # l1 silently computes sq
    assert _reach_l1(red) is False


# ---------------------------------------------------------------------------------------------- the gate
def test_gdvb_declared_equals_built_and_a_wrong_expectation_is_refused():
    rec = {"w_rule_own": 1.0, "n_pres": 3, "n_gap": 3, "n_clos": 3, "labels_sha256": "a", "dist_rule_sha256": "b"}
    recl = {**rec, "form": "l1"}
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0"), {"rule_own": rec}) == []                                         # default: no form key, sq
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0", "--rule-own-form", "l1"), {"rule_own": recl}) == []
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0", "--rule-own-form", "l1"), {"rule_own": rec})                        # l1 declared, sq built
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0"), {"rule_own": recl})                                               # sq declared, l1 built
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0", "--rule-own-form", "sq"), {"rule_own": recl})
    assert TR.bad_rule_own(_args("--rule-own-form", "l1"), {})                                                             # refused without --w-rule-own
    assert TR.bad_rule_own(_args("--rule-own-form", "l1"), {"rule_own": {**recl, "w_rule_own": 0.0}})
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0", "--rule-own-form", "l1", "--rule-relabel", "on"), {"rule_own": recl}) == []
    with pytest.raises(SystemExit):
        _args("--rule-own-form", "l2")


def test_loop_passes_the_built_form():
    src = Path(TR.__file__).read_text(encoding="utf-8")
    assert src.count('ROW.add_own_term(T, out["traj30"], ro_in, RO.w_own, RO.form)') == 1
    assert src.count("form=a.rule_own_form") == 1 and src.count("ROW.own_term(tr, inp, RO.form)") == 2


def test_selfcheck_strings_and_probe_form_and_the_default_has_none(tmp_path):
    for tag, extra in (("sq", ["--w-rule-own", "1.0"]), ("l1", ["--w-rule-own", "1.0", "--rule-own-form", "l1"]), ("l1w4", ["--w-rule-own", "4.0", "--rule-own-form", "l1", "--rule-relabel", "on"])):
        out = tmp_path / tag
        assert TR.main([*SYN, *extra, "--selfcheck-only", "--out", str(out)]) == 0
        sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
        assert sc["declared_vs_built"] == "PASS" and sc["rule_own_probe"]["pass"]
        if tag == "sq":
            assert "rule_own_form" not in sc["levers"] and "form" not in sc["info"]["rule_own"] and sc["rule_own_probe"]["form"] == "sq"
        else:
            assert sc["levers"]["rule_own_form"] == "l1" and sc["info"]["rule_own"]["form"] == "l1" and sc["rule_own_probe"]["form"] == "l1"
            assert np.isfinite(sc["rule_own_probe"]["term"]) and sc["rule_own_probe"]["n_lead"] > 0 and sc["rule_own_probe"]["control_all_false_term"] == 0.0


def test_refused_without_the_weight(tmp_path):
    with pytest.raises((SystemExit, ValueError)):
        rc = TR.main([*SYN, "--rule-own-form", "l1", "--selfcheck-only", "--out", str(tmp_path / "r")])
        if rc:
            raise SystemExit(rc)


def test_train_log_keys_unchanged_and_checkpoint_cfg_equals_its_base(tmp_path):
    cks, keys = {}, {}
    for tag, extra in (("sq", ["--w-rule-own", "1.0"]), ("l1", ["--w-rule-own", "1.0", "--rule-own-form", "l1"])):
        out = tmp_path / tag
        assert TR.main([*SYN, *TINY, *extra, "--steps", "3", "--log-every", "1", "--ckpt-every", "100", "--out", str(out)]) == 0
        rows = [json.loads(line) for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
        keys[tag] = {k for r in rows for k in r}
        assert all(np.isfinite(r["rule_own_term"]) for r in rows) and any(r["rule_own_n_lead"] > 0 for r in rows)
        cks[tag] = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert keys["sq"] == keys["l1"]
    assert cks["l1"]["cfg"] == cks["sq"]["cfg"] and list(cks["l1"]["model"]) == list(cks["sq"]["model"])
    cfg = M.V8FMiniConfig(**cks["l1"]["cfg"])
    m = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    assert "all keys matched" in str(m.load_state_dict(cks["l1"]["model"])).lower()
