"""RULEOWN (--w-rule-own and --rule-relabel-weight): the distance rule applied to the model's own plan as a loss term, and the weight of the samples the relabel changed
(card PREREG_V8F_PROOF_ADD34 sections 2 and 5, note PREREG_V8F_PROOF_ADD32_N1 stage 2).

Tiny models (d 32) on CPU. Reach is proved from the flag to the loss and to the gradient on the trajectory output, with controls that must read exactly 0.0, and each line that applies a lever is deleted in a
mutation test that must go red. The trainer loop lines are pinned by source checks plus the trainer runs in checks/."""
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
from tanitad.data import rule_relabel as RRL  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

GRID = ["--wm-state", "grid", "--wm-cell-dim", "16"]
SYN = ["--synthetic", "--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", *GRID, "--exclude", "1:3", "--threads", "1"]
TINY = ["--batch", "8", "--d", "32", "--n-q", "2"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("ruleown")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    D.synthetic_lead_labels(J, 0)
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=16)
    RR = RRL.RuleRelabel("on", J, store)
    RO = ROW.RuleOwn(1.0, 4.0, J, store, share=RR)
    return {"J": J, "ds": ds, "wins": wins, "RR": RR, "RO": RO, "store": store}


def _batch(world, idx):
    return D.collate([world["ds"].item(world["wins"][i][0], world["wins"][i][1], 1, 3) for i in idx])


def _model(world):
    a = TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *GRID])
    cfg = TR.make_config(a)
    mu, sd = M.cons_stats(world["J"])
    torch.manual_seed(0)
    return M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd), cfg


def _straight(n, speed):
    g = np.zeros((n, 30, 2), np.float32)
    g[:, :, 0] = np.arange(1, 31)[None, :] * 0.2 * speed
    return g


def _inp(n, gap, v0=10.0, v_L=0.0, mask=None):
    return {"mask": np.ones(n, bool) if mask is None else np.asarray(mask, bool), "v0": np.full(n, v0), "gap": np.full(n, float(gap)), "v_L": np.full(n, v_L), "n_rows": n}


# ------------------------------------------------------------------------------------------------ flag A: definition
def test_flags_default_off_and_have_their_gate_entries():
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.w_rule_own == 0.0 and a.rule_relabel_weight == 1.0
    assert TR.FLAG_CHECKS["w_rule_own"] == "data:rule_own" and TR.FLAG_CHECKS["rule_relabel_weight"] == "data:rule_relabel_weight"
    ro = ROW.RuleOwn(0.0, 1.0)
    assert not ro.own_on and not ro.weight_on and ro.src is None


def test_mask_and_inputs_equal_the_relabel_mask(world):
    b = _batch(world, list(range(0, min(64, len(world["wins"])), 2)))
    _, info = world["RR"].stage(b)
    inp = world["RO"].stage(b)
    assert np.array_equal(inp["mask"], info["mask"]) and inp["mask"].any() and not inp["mask"].all()


def test_cap_equals_distance_rule_by_hand_on_the_detached_prediction_and_term_is_the_mean_squared_distance():
    plan = torch.as_tensor(_straight(3, 10.0)) + 0.01 * torch.randn(3, 30, 2)
    plan.requires_grad_(True)
    inp = _inp(3, 8.0, v0=10.0, v_L=2.0)
    term, info = ROW.own_term(plan, inp)
    by_hand = R.distance_rule(plan.detach().double().numpy(), np.full(3, 10.0), np.ones(3, bool), np.full(3, 8.0), np.full(3, 8.0), np.full(3, 2.0), m=10, floor_lifts=False)
    cap = torch.as_tensor(by_hand.plan).float()                                                      # the term calls the rule in float64 and casts the cap back to the prediction dtype
    assert info["n_lead"] == 3 and info["n_active"] == 3
    assert torch.equal(torch.as_tensor(ROW.cap_of_plan(plan.detach().numpy(), inp["v0"], inp["gap"], inp["v_L"])).float(), cap)               # exact
    hand = (((plan.detach() - cap) ** 2).sum(-1).mean(1)).mean()
    assert torch.allclose(term.detach(), hand, rtol=0, atol=1e-7) and float(term) > 0.0
    term.backward()
    assert torch.allclose(plan.grad, (2.0 * (plan.detach() - cap)) / (3 * 30), atol=1e-7)          # no gradient through the rule: d term / d plan = 2 (plan - cap) / (n x knots)


def test_the_rule_applied_twice_equals_once():
    plan = _straight(4, 10.0)
    inp = _inp(4, 8.0, v0=10.0, v_L=2.0)
    c1 = ROW.cap_of_plan(plan, inp["v0"], inp["gap"], inp["v_L"])
    c2 = ROW.cap_of_plan(c1, inp["v0"], inp["gap"], inp["v_L"])
    assert float(np.abs(c1 - c2).max()) <= 1e-5                                                      # the float64 path the term uses
    assert not np.array_equal(c1, plan)


def test_samples_without_a_lead_or_with_a_non_finite_raw_speed_contribute_exactly_zero():
    n = 5
    g = _straight(n, 10.0)
    ok = np.ones(n, bool)
    pres = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    gap = np.array([8.0, 8.0, np.nan, 8.0, 8.0])
    clos = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    v0 = np.array([10.0, 10.0, 10.0, np.nan, 10.0])
    vl = np.array([10.0, 10.0, 10.0, 10.0, np.nan])
    inp = ROW.own_inputs(ok, ok, v0, vl, pres, gap, clos)
    assert inp["mask"].tolist() == [True, False, False, False, False]
    plan = torch.as_tensor(g, dtype=torch.float32).requires_grad_(True)
    term, info = ROW.own_term(plan, inp)
    term.backward()
    assert info["n_lead"] == 1 and info["n_active"] == 1 and float(term) > 0.0
    assert float(plan.grad[1:].abs().sum()) == 0.0 and float(plan.grad[0].abs().sum()) > 0.0                           # only the masked row gets a gradient
    plan2 = torch.as_tensor(g, dtype=torch.float32).requires_grad_(True)
    inp0 = ROW.own_inputs(ok, ok, v0, vl, pres * 0.0, gap, clos)
    t0, i0 = ROW.own_term(plan2, inp0)
    assert i0["n_lead"] == 0 and float(t0) == 0.0 and not t0.requires_grad


# ------------------------------------------------------------------------------------------------ flag A: reach (o3)
def test_reach_a_plan_into_the_lead_a_plan_inside_the_rule_and_a_batch_without_a_lead():
    W = 2.0
    run = torch.as_tensor(_straight(4, 10.0)).requires_grad_(True)                          # 10 m/s towards a lead 8 m away
    T = {"total": (run ** 2).sum() * 0.0 + 1.0}
    info = ROW.add_own_term(T, run, _inp(4, 8.0, v0=10.0, v_L=0.0), W)
    T["total"].backward()
    assert info["term"] > 0.0 and info["n_active"] == 4 and float(run.grad.abs().sum()) > 0.0 and float(T["total"]) == pytest.approx(1.0 + W * info["term"], rel=1e-6)
    slow = torch.as_tensor(_straight(4, 0.5)).requires_grad_(True)                          # a crawl, lead 40 m away: inside the rule
    T2 = {"total": (slow ** 2).sum() * 0.0 + 1.0}
    i2 = ROW.add_own_term(T2, slow, _inp(4, 40.0, v0=0.5, v_L=0.0), W)
    T2["total"].backward()
    assert i2["n_lead"] == 4 and i2["n_active"] == 0 and i2["term"] == 0.0 and float(slow.grad.abs().sum()) == 0.0 and float(T2["total"]) == 1.0
    none = torch.as_tensor(_straight(4, 10.0)).requires_grad_(True)
    T3 = {"total": (none ** 2).sum() * 0.0 + 1.0}
    i3 = ROW.add_own_term(T3, none, _inp(4, 8.0, mask=np.zeros(4, bool)), W)
    T3["total"].backward()
    assert i3["n_lead"] == 0 and i3["term"] == 0.0 and float(none.grad.abs().sum()) == 0.0 and float(T3["total"]) == 1.0


def test_reach_on_a_world_batch_is_partial_and_reaches_the_model_gradient(world):
    model, cfg = _model(world)
    model.train()
    idx = list(range(len(world["wins"])))
    b = _batch(world, idx[: min(64, len(idx))])
    inp = world["RO"].stage(b)
    # the model's own plan is far below the rule's cap for most synthetic rows, so a planted fast plan on half of the lead rows makes the term partial: some lead samples contribute, some do not
    out = model.loss(dict(b), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    tr = out["traj30"]
    n_lead = int(inp["mask"].sum())
    assert 0 < n_lead < len(inp["mask"])
    planted = tr.detach().clone()
    sel = np.nonzero(inp["mask"])[0]
    h = max(1, len(sel) // 2)
    planted[sel[:h]] = torch.as_tensor(_straight(h, 12.0))                                  # runs into a lead 8 m away
    planted[sel[h:]] = torch.as_tensor(_straight(len(sel) - h, 0.5))                         # a crawl with the lead 40 m away: inside the rule
    planted.requires_grad_(True)
    gap2 = inp["gap"].copy()
    gap2[sel[:h]], gap2[sel[h:]] = 8.0, 40.0
    inp2 = dict(inp, gap=gap2, v_L=np.where(inp["mask"], 0.0, inp["v_L"]))
    term, info = ROW.own_term(planted, inp2)
    print("partial reach: n_lead", info["n_lead"], "n_active", info["n_active"])
    assert info["n_lead"] == n_lead and 0 < info["n_active"] < info["n_lead"] and float(term) > 0.0
    for p in model.parameters():
        p.grad = None
    model.loss(dict(b), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    o2 = model.loss(dict(b), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    T = o2["terms"]
    tot0 = float(T["total"].detach())
    inf2 = ROW.add_own_term(T, o2["traj30"], inp2, 1.0)
    T["total"].backward()
    g = float(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None))
    assert g > 0.0 and np.isfinite(inf2["term"])
    assert float(T["total"].detach()) == pytest.approx(tot0 + inf2["term"], rel=1e-5, abs=1e-5)


def _load_mutated(old, new):
    src = Path(ROW.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1
    mod = types.ModuleType("rule_own_mut")
    mod.__file__ = str(ROW.__file__)
    exec(compile(src.replace(old, new), "rule_own_mut", "exec"), mod.__dict__)
    return mod


def _reach_a(mod):
    run = torch.as_tensor(_straight(4, 10.0)).requires_grad_(True)
    T = {"total": (run ** 2).sum() * 0.0 + 1.0}
    info = mod.add_own_term(T, run, _inp(4, 8.0, v0=10.0, v_L=0.0), 2.0)
    T["total"].backward()
    return info["term"] > 0.0 and float(T["total"]) > 1.0 and float(run.grad.abs().sum()) > 0.0


def test_mutation_deleting_the_line_that_adds_the_term_goes_red():
    assert _reach_a(ROW) is True                                                            # the green run
    red = _load_mutated('    T["total"] = T["total"] + W * term\n', "    pass\n")
    assert _reach_a(red) is False                                                           # the same check on the mutated copy goes red


def test_the_loop_applies_the_term_after_the_loss_and_guards_it():
    src = Path(TR.__file__).read_text(encoding="utf-8")
    assert src.count("ROW.add_own_term(T, out[\"traj30\"], ro_in, RO.w_own, RO.form)") == 1 and src.count("ZZRULE-OWN-NOT-APPLIEDZZ") == 1
    assert src.index("                ro_in = RO.stage(b)") < src.index("                b = comp.apply(b, step)" + chr(10)) < src.index("ROW.add_own_term(T, out[\"traj30\"]") < src.index("                if not torch.isfinite(T[\"total\"]):")
    assert src.count("ROW.add_weight_extra(T, out[\"traj30\"]") == 1 and src.count("ZZRULE-WEIGHT-NOT-APPLIEDZZ") == 1


# ------------------------------------------------------------------------------------------------ the gate (o4, w4)
def _args(*extra):
    return TR.build_parser().parse_args(["--out", "x", *extra])


def test_gdvb_declared_equals_built_and_a_wrong_expectation_is_refused():
    rec = {"w_rule_own": 1.0, "n_pres": 3, "n_gap": 3, "n_clos": 3, "labels_sha256": "a", "dist_rule_sha256": "b"}
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0"), {"rule_own": rec}) == []
    assert TR.bad_rule_own(_args("--w-rule-own", "4.0"), {"rule_own": rec})                          # declared 4 but built 1
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0"), {})                                           # declared but nothing built
    assert TR.bad_rule_own(_args(), {"rule_own": rec})                                                 # off declared but a stage was built
    assert TR.bad_rule_own(_args("--w-rule-own", "-1.0"), {"rule_own": {**rec, "w_rule_own": -1.0}})
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0", "--ego-perturb-p", "0.5"), {"rule_own": rec})
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0", "--dagger-p", "0.5"), {"rule_own": rec})
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0", "--recover-p", "0.5"), {"rule_own": rec})
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0", "--encoder", "lora"), {"rule_own": rec})
    assert TR.bad_rule_own(_args("--w-rule-own", "1.0", "--rule-relabel", "on"), {"rule_own": rec}) == []          # may be combined with --rule-relabel on
    wrec = {"w": 4.0}
    assert TR.bad_rule_own(_args("--rule-relabel", "on", "--rule-relabel-weight", "4"), {"rule_relabel_weight": wrec}) == []
    assert TR.bad_rule_own(_args("--rule-relabel", "on", "--rule-relabel-weight", "2"), {"rule_relabel_weight": wrec})        # declared 2 but built 4
    assert TR.bad_rule_own(_args("--rule-relabel-weight", "4"), {"rule_relabel_weight": wrec})                                 # needs --rule-relabel on
    assert TR.bad_rule_own(_args("--rule-relabel", "on", "--rule-relabel-weight", "4"), {})                                    # declared but nothing built
    assert TR.bad_rule_own(_args("--rule-relabel", "on"), {"rule_relabel_weight": wrec})                                       # default 1 but a stage was built
    assert TR.bad_rule_own(_args("--rule-relabel", "on", "--rule-relabel-weight", "-2"), {"rule_relabel_weight": {"w": -2.0}})


def test_selfcheck_on_has_the_probe_and_the_strings_and_off_has_no_key(tmp_path):
    for tag, extra in (("off", []), ("own", ["--w-rule-own", "1.0"]), ("own_rt", ["--w-rule-own", "4.0", "--rule-relabel", "on"]), ("wt", ["--rule-relabel", "on", "--rule-relabel-weight", "4"])):
        out = tmp_path / tag
        rc = TR.main([*SYN, *extra, "--selfcheck-only", "--out", str(out)])
        assert rc == 0
        sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
        assert sc["declared_vs_built"] == "PASS"
        keys = [k for k in list(sc["levers"]) + list(sc["info"]) + list(sc) if "rule_own" in k or "rule_relabel_weight" in k or k == "w_rule_own"]
        if tag == "off":
            assert not keys
        if tag.startswith("own"):
            p = sc["rule_own_probe"]
            assert sc["levers"]["w_rule_own"] == float(extra[1]) and p["pass"] and np.isfinite(p["term"]) and p["n_lead"] > 0 and "n_active" in p
            assert "grad_norm_term_on_traj30" in p and "grad_norm_traj_loss_on_traj30" in p and p["control_all_false_term"] == 0.0
        if tag == "wt":
            assert sc["levers"]["rule_relabel_weight"] == 4.0 and "rule_own_probe" not in sc and "w_rule_own" not in sc["levers"]


@pytest.mark.parametrize("extra", [["--ego-perturb-p", "0.5"], ["--recover-p", "0.5"], ["--dagger-p", "0.5"]])
def test_refused_combinations(tmp_path, extra):
    with pytest.raises(SystemExit) as e:
        TR.main([*SYN, "--w-rule-own", "1.0", *extra, "--selfcheck-only", "--out", str(tmp_path / "x")])
    assert e.value.code not in (0, None)


def test_the_weight_without_the_relabel_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        TR.main([*SYN, "--rule-relabel-weight", "4", "--selfcheck-only", "--out", str(tmp_path / "x")])
    assert e.value.code not in (0, None)


def test_train_log_keys_only_when_on_and_checkpoint_cfg_equals_its_base(tmp_path):
    cks = {}
    for tag, extra in (("off", []), ("own", ["--w-rule-own", "1.0"]), ("wt", ["--rule-relabel", "on", "--rule-relabel-weight", "4"])):
        out = tmp_path / tag
        rc = TR.main([*SYN, *TINY, *extra, "--steps", "3", "--log-every", "1", "--ckpt-every", "100", "--out", str(out)])
        assert rc == 0
        rows = [json.loads(line) for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
        keys = {k for r in rows for k in r}
        if tag == "off":
            assert not [k for k in keys if "rule_own" in k or "weighted" in k]
        if tag == "own":
            assert {"rule_own_term", "rule_own_n_lead", "rule_own_n_active"} <= keys and all(np.isfinite(r["rule_own_term"]) for r in rows)
            assert any(r["rule_own_n_lead"] > 0 for r in rows)
        if tag == "wt":
            assert "rule_relabel_weighted_n" in keys and not [k for k in keys if "rule_own" in k]
        cks[tag] = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    for tag in ("own", "wt"):
        assert cks[tag]["cfg"] == cks["off"]["cfg"] and list(cks[tag]["model"]) == list(cks["off"]["model"])
    assert "rule_own_totals" in cks["own"] and "rule_own_totals" not in cks["off"]
    a, cfg = None, M.V8FMiniConfig(**cks["own"]["cfg"])
    m = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    assert "all keys matched" in str(m.load_state_dict(cks["own"]["model"])).lower()


# ------------------------------------------------------------------------------------------------ flag B (w2, w3, w4)
def _hand_terms(traj, gt):
    d = np.sqrt(((traj - gt) ** 2).sum(-1) + 1e-9)
    return d.mean(1), d[:, -1]


def test_weighted_loss_equals_a_hand_computation_and_only_changed_samples_carry_the_weight(world):
    model, cfg = _model(world)
    model.train()
    b = _batch(world, list(range(0, min(64, len(world["wins"])), 2)))
    gt_new, info = world["RR"].stage(b)
    assert info["n_changed"] > 0 and info["n_relabel"] > info["n_changed"] - 1
    chg = ROW.changed_vector(info, int(b["gt"].shape[0]))
    assert chg.sum() == info["n_changed"] and (chg <= info["mask"]).all()
    b2 = {**b, "gt": gt_new}
    out = model.loss(b2, p_gt=0.5, gen=torch.Generator().manual_seed(11))
    T, tr = out["terms"], out["traj30"]
    base_total = float(T["total"].detach())
    W = 4.0
    tr_req = tr.detach().clone().requires_grad_(True)
    extra = ROW.weight_extra(tr_req, gt_new, b2["gt_ok"], chg, W, cfg.w_traj)
    ade_i, fde_i = _hand_terms(tr.detach().double().numpy(), gt_new.double().numpy())
    gok = b2["gt_ok"].numpy().astype(np.float64)
    den = max(gok.sum(), 1.0)
    hand = cfg.w_traj * (W - 1.0) * (((ade_i * gok * chg).sum() / den) / 3.0 + 0.5 * ((fde_i * gok * chg).sum() / den) / 6.0)
    assert float(extra.detach()) == pytest.approx(hand, rel=1e-5, abs=1e-9) and hand > 0.0
    # the total with the weight = the loss with those samples counted W times, hand-computed from the unweighted terms
    ade_all = (ade_i * gok).sum() / den
    fde_all = (fde_i * gok).sum() / den
    assert float(T["ade"].detach()) == pytest.approx(ade_all, rel=1e-4) and float(T["fde"].detach()) == pytest.approx(fde_all, rel=1e-4)
    ade_w = (ade_i * gok * (1.0 + (W - 1.0) * chg)).sum() / den
    fde_w = (fde_i * gok * (1.0 + (W - 1.0) * chg)).sum() / den
    assert float(extra.detach()) == pytest.approx(cfg.w_traj * ((ade_w - ade_all) / 3.0 + 0.5 * (fde_w - fde_all) / 6.0), rel=1e-4)       # not renormalised: the denominator is the GT-valid count
    Tc = {"total": T["total"]}
    ROW.add_weight_extra(Tc, tr.detach(), gt_new, b2["gt_ok"], chg, W, cfg.w_traj)
    assert float(Tc["total"].detach()) == pytest.approx(base_total + hand, rel=1e-5)
    extra.backward()
    g = tr_req.grad.abs().sum(dim=(1, 2)).numpy()
    assert (g[~chg] == 0.0).all() and (g[chg] > 0.0).all()                                               # only the changed samples carry the weight
    assert float(ROW.weight_extra(tr.detach(), gt_new, b2["gt_ok"], np.zeros_like(chg), W, cfg.w_traj)) == 0.0         # no changed sample: exactly 0
    assert float(ROW.weight_extra(tr.detach(), gt_new, b2["gt_ok"], chg, 1.0, cfg.w_traj)) == 0.0                       # weight 1: exactly 0
    both = ROW.weight_extra(tr.detach(), gt_new, b2["gt_ok"], chg, W, cfg.w_traj, w_acc=2.0)                             # acc enters like in the loss: w_acc x acc over the same samples
    assert float(both) > float(extra.detach())


def test_mutation_deleting_the_line_that_adds_the_weight_goes_red():
    tr = torch.as_tensor(_straight(4, 10.0)).requires_grad_(True)
    gt = torch.as_tensor(_straight(4, 9.0))
    ok, chg = np.ones(4, bool), np.array([True, False, True, False])

    def reach(mod):
        T = {"total": (tr ** 2).sum() * 0.0 + 1.0}
        mod.add_weight_extra(T, tr, gt, ok, chg, 4.0, 1.0)
        return float(T["total"]) > 1.0

    assert reach(ROW) is True
    red = _load_mutated('    T["total"] = T["total"] + weight_extra(traj30, gt, gt_ok, changed, W, w_traj, w_acc, launch, launch_w)\n', "    pass\n")
    assert reach(red) is False
