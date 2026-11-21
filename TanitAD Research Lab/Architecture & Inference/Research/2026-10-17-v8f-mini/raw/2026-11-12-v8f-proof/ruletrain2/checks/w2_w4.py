"""RULEOWN stage 2 checks w2 (the weighted loss equals the hand computation; only the changed samples carry the weight), the reach part of w3 (the mutation test runs this file on a mutated copy: it must go RED) and w4 (the gate entry).
Usage: w2_w4.py <tree_root> <out json>   (PYTHONPATH = tree stack;taniteval;stack/scripts)

w2  batch of 64 logged windows from the trainer's own build_data with --rule-relabel on --rule-relabel-weight 4. The relabel stage gives the target and the mask of CHANGED samples. The tiny V8FMini loss gives the unweighted total.
    By hand (numpy, per sample, from the model's traj30 and the relabelled target): ade_i, fde_i, the GT-valid count; the extra loss must equal w_traj x (W - 1) x (sum over CHANGED samples of ade_i / 3 + 0.5 fde_i / 6) / n_valid
    (not renormalised: the denominator is the GT-valid count). Total with the weight = total + extra. The gradient of the extra on traj30 is exactly 0 on every unchanged row and above 0 on every changed row. Weight 1 and no changed
    sample give exactly 0. The acc and launch terms enter with their own weights when on (checked against the hand computation as well).
w3  reach: the total with the weight must be above the total without it, and the weighted loss must move the model gradient.
w4  gate: bad_rule_own with declared equals built (empty), a wrong expectation (declared 2, built 4), the weight without --rule-relabel on, nothing built, default with a stage built, negative: all refused.
"""
import json
import os
import sys

import numpy as np
import torch

tree, outp = sys.argv[1], sys.argv[2]
torch.set_num_threads(2)
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower()), tanitad.__file__
sys.path.insert(0, os.path.join(tree, "stack", "scripts"))
import v8f_mini_train as TR  # noqa: E402
from tanitad.data import rule_own as ROW  # noqa: E402
from tanitad.data import rule_relabel as RRL  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

assert os.path.abspath(ROW.__file__).lower().startswith(os.path.abspath(tree).lower()), ROW.__file__
res = {"tanitad_file": tanitad.__file__, "rule_own_file": ROW.__file__}
OUT = "F:/TanitAD-scratch/v8proof/ruletrain2/work/tmp/w2w4"
os.makedirs(OUT, exist_ok=True)
W = 4.0
argv = ["--synthetic", "--batch", "64", "--d", "32", "--n-q", "2", "--threads", "1", "--seed", "0", "--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16",
        "--rule-relabel", "on", "--rule-relabel-weight", str(W), "--exclude", "1:3", "--out", OUT]
a = TR.build_parser().parse_args(argv)
cfg = TR.make_config(a)
torch.manual_seed(0)
np.random.seed(0)
store, J, wins, clips, info = TR.build_data(a, cfg)
ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
RR = RRL.RuleRelabel("on", J, store)
RO = ROW.RuleOwn(a.w_rule_own, a.rule_relabel_weight, J, store)
assert RO.weight_on and not RO.own_on
wl = list(wins)
torch.manual_seed(0)
model = M.V8FMini(cfg, M.P10Prior.synthetic(0), torch.zeros(M.N_CONS), torch.ones(M.N_CONS))
model.train()
batch = D.collate([ds.item(*w) for w in wl[:64]])
gt_new, ri = RR.stage(batch)
chg = ROW.changed_vector(ri, 64)
b2 = {**batch, "gt": gt_new}
named = [(nm, p) for nm, p in model.named_parameters() if p.requires_grad]


def run(with_weight):
    for _, p in named:
        p.grad = None
    torch.manual_seed(20261115)
    o = model.loss(dict(b2), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    T, tr = o["terms"], o["traj30"]
    if with_weight:
        ROW.add_weight_extra(T, tr, b2["gt"], b2["gt_ok"], chg, W, cfg.w_traj, cfg.w_acc, b2.get("launch"), cfg.launch_intent_w)
    T["total"].backward()
    return float(T["total"].detach()), {nm: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for nm, p in named}, tr.detach().clone(), T


tot0, g0, tr0, T0 = run(False)
tot1, g1, tr1, T1 = run(True)
# ---- w2: by hand, per sample
tr = tr0.double().numpy()
gt = gt_new.double().numpy()
d = np.sqrt(((tr - gt) ** 2).sum(-1) + 1e-9)
ade_i, fde_i = d.mean(1), d[:, -1]
gok = b2["gt_ok"].numpy().astype(np.float64)
den = max(gok.sum(), 1.0)
hand = cfg.w_traj * (W - 1.0) * sum((ade_i[i] / 3.0 + 0.5 * fde_i[i] / 6.0) * gok[i] for i in range(64) if chg[i]) / den
extra = tot1 - tot0
trq = tr0.clone().requires_grad_(True)
ex = ROW.weight_extra(trq, gt_new, b2["gt_ok"], chg, W, cfg.w_traj)
ex.backward()
grow = trq.grad.abs().sum(dim=(1, 2)).numpy()
tq2 = tr0.clone().requires_grad_(True)
ex_acc = ROW.weight_extra(tq2, gt_new, b2["gt_ok"], chg, W, cfg.w_traj, w_acc=2.0)
d2 = M.second_diff_02(torch.as_tensor(tr)) - M.second_diff_02(torch.as_tensor(gt))
acc_i = (d2 ** 2).sum(-1).mean(1).numpy()
hand_acc = 2.0 * (W - 1.0) * sum(acc_i[i] * gok[i] for i in range(64) if chg[i]) / den
res["w2"] = {"n_rows": 64, "n_changed": int(chg.sum()), "n_gt_valid": int(gok.sum()), "total_without_weight": tot0, "total_with_weight": tot1, "extra_in_total": extra, "extra_by_hand": float(hand),
             "abs_diff_extra_vs_hand": abs(extra - float(hand)), "rel_diff": abs(extra - float(hand)) / max(abs(float(hand)), 1e-30), "mean_ade_all": float((ade_i * gok).sum() / den),
             "grad_rows_unchanged_max": float(grow[~chg].max()), "grad_rows_changed_min": float(grow[chg].min()),
             "weight1_extra": float(ROW.weight_extra(tr0, gt_new, b2["gt_ok"], chg, 1.0, cfg.w_traj)), "no_changed_extra": float(ROW.weight_extra(tr0, gt_new, b2["gt_ok"], np.zeros(64, bool), W, cfg.w_traj)),
             "acc_extra_minus_ade_extra": float(ex_acc.detach() - ex.detach()), "acc_extra_by_hand": float(hand_acc),
             "denominator_note": "not renormalised: the denominator is the GT-valid count"}
w = res["w2"]
w["PASS"] = bool(w["n_changed"] > 0 and w["rel_diff"] < 1e-4 and w["grad_rows_unchanged_max"] == 0.0 and w["grad_rows_changed_min"] > 0.0 and w["weight1_extra"] == 0.0 and w["no_changed_extra"] == 0.0
                 and abs(w["acc_extra_minus_ade_extra"] - w["acc_extra_by_hand"]) <= 1e-3 * max(1.0, abs(w["acc_extra_by_hand"])))
# ---- w3 reach (the mutation test turns this RED)
gdiff = float(sum((g0[k] - g1[k]).abs().sum() for k in g0))
res["w3_reach"] = {"total_with": tot1, "total_without": tot0, "grad_diff_abs_sum": gdiff, "PASS": bool(tot1 > tot0 and gdiff > 0.0)}
# ---- w4 gate
rec = {"w": W}
A = lambda *x: TR.build_parser().parse_args(["--out", "x", *x])      # noqa: E731
w4 = {"declared_equals_built": TR.bad_rule_own(A("--rule-relabel", "on", "--rule-relabel-weight", "4"), {"rule_relabel_weight": rec}) == [],
      "wrong_expectation_refused": bool(TR.bad_rule_own(A("--rule-relabel", "on", "--rule-relabel-weight", "2"), {"rule_relabel_weight": rec})),
      "without_relabel_refused": bool(TR.bad_rule_own(A("--rule-relabel-weight", "4"), {"rule_relabel_weight": rec})),
      "nothing_built_refused": bool(TR.bad_rule_own(A("--rule-relabel", "on", "--rule-relabel-weight", "4"), {})),
      "default_with_a_stage_refused": bool(TR.bad_rule_own(A("--rule-relabel", "on"), {"rule_relabel_weight": rec})),
      "negative_refused": bool(TR.bad_rule_own(A("--rule-relabel", "on", "--rule-relabel-weight", "-2"), {"rule_relabel_weight": {"w": -2.0}})),
      "flag_check_entry": TR.FLAG_CHECKS.get("rule_relabel_weight"), "built_weight_of_the_real_stage": RO.w_weight, "built_levers": RO.built_levers()}
w4["PASS"] = bool(w4["declared_equals_built"] and w4["wrong_expectation_refused"] and w4["without_relabel_refused"] and w4["nothing_built_refused"] and w4["default_with_a_stage_refused"] and w4["negative_refused"]
                  and w4["flag_check_entry"] == "data:rule_relabel_weight" and w4["built_weight_of_the_real_stage"] == W)
res["w4"] = w4
res["PASS"] = bool(res["w2"]["PASS"] and res["w3_reach"]["PASS"] and w4["PASS"])
json.dump(res, open(outp, "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(res["w2"]))
print(json.dumps(res["w3_reach"]))
print(json.dumps(res["w4"]))
print("W2W4", "PASS" if res["PASS"] else "FAIL")
sys.exit(0 if res["PASS"] else 1)
