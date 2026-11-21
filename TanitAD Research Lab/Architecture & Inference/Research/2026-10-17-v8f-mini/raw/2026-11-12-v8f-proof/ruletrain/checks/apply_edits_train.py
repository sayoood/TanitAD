"""apply_edits_train.py: the RULETRAIN edits of stack/scripts/v8f_mini_train.py (usage: apply_edits_train.py <tree root>). Every anchor must occur exactly once."""
import sys

root = sys.argv[1]
path = f"{root}/stack/scripts/v8f_mini_train.py"
src = open(path, encoding="utf-8", newline="").read()


def put(anchor, new, where="before"):
    global src
    n = src.count(anchor)
    assert n == 1, (n, anchor[:80])
    src = src.replace(anchor, (new + anchor) if where == "before" else (anchor + new))


# 1 import
put("from tanitad.models import v8f_mini as M  # noqa: E402\n",
    "from tanitad.data import rule_relabel as RRL  # noqa: E402      PREREG_V8F_PROOF_ADD32 RULETRAIN: the label-fed distance rule applied to the training target (`--rule-relabel off` = nothing is built and nothing is applied)\n", "after")

# 2 flag
put('    ap.add_argument("--cache-dir", default="C:/Users/Admin/refav1_hj_cache")\n',
    '    ap.add_argument("--rule-relabel", dest="rule_relabel", choices=("off", "on"), default="off",\n'
    '                    help="RULETRAIN (PREREG_V8F_PROOF_ADD32): on = the 30-knot target of the executed plan (batch gt) of every sample whose CURRENT frame has a label lead (presence, finite gap, finite closing-speed label) is replaced by "\n'
    '                         "dist_rule.distance_rule(gt, v0, present, gap, gap, v_L, cap-only) with v_L = max(0, v_logged + closing). Samples without a label lead keep their target. Labels from the label join (--join-train); "\n'
    '                         "training only, nothing at inference. Refused with --encoder lora, --ego-perturb-p, --dagger-p, --recover-p. Default off = every earlier arm, bit for bit")\n')

# 3 FLAG_CHECKS
put('    "init_ckpt": "data:init_ckpt",',
    '    "rule_relabel": "data:rule_relabel",      # RULETRAIN: argv -> RuleRelabel (built from the join) -> info["rule_relabel"] (bad_rule_relabel) -> rule_relabel_probe (G-LIVE: count > 0, the relabelled target reaches the loss and gradient, all-False mask reads EXACTLY 0.0) -> `b["gt"] = rr_gt` in the loop\n')

# 4 gate clause
put("    bad += bad_lead_sup(a, cfg, model, data_info)    ",
    "    bad += bad_rule_relabel(a, data_info)                                                        # RULETRAIN: --rule-relabel read back from the BUILT stage (labels, rule file sha256) and its refused combinations\n")

# 5 functions before declared_vs_built
put("def declared_vs_built(a, cfg, model, opt, data_info,", '''def bad_rule_relabel(a, data_info) -> list:
    """RULETRAIN (G-DVB): ``--rule-relabel`` read back from the BUILT stage record ``info["rule_relabel"]``. off: no stage may exist. on: the record names the mode, the label arrays have rows with presence, gap and closing labels,
    the rule file's sha256 is recorded, and the stages that rewrite the logged state or target (ego perturbation, DAgger swap, recovery, LoRA frames) are off (the rule is applied to the LOGGED frame's target)."""
    rec = (data_info or {}).get("rule_relabel")
    if a.rule_relabel == "off":
        return ["rule_relabel off declared but a relabel stage was built"] if rec else []
    if not rec:
        return ["rule_relabel on declared but no relabel stage was built"]
    bad = []
    if rec.get("mode") != a.rule_relabel:
        bad.append(f"rule_relabel: declared {a.rule_relabel} but built {rec.get('mode')}")
    for k in ("n_pres", "n_gap", "n_clos"):
        if not (isinstance(rec.get(k), int) and rec[k] > 0):
            bad.append(f"rule_relabel labels: {k} = {rec.get(k)!r}: no label lead rows to relabel (a lever that is declared and does nothing)")
    if not rec.get("labels_sha256") or not rec.get("dist_rule_sha256"):
        bad.append("rule_relabel: the label arrays' or the rule file's sha256 is not recorded")
    for flag, val, off in (("ego_perturb_p", a.ego_perturb_p, 0.0), ("dagger_p", a.dagger_p, 0.0), ("recover_p", a.recover_p, 0.0)):
        if val != off:
            bad.append(f"rule_relabel on with --{flag.replace('_', '-')} {val}: the rule is applied to the LOGGED frame's target, which that stage rewrites")
    if a.encoder != "cache":
        bad.append("rule_relabel is not built for --encoder lora (the loop's batch path differs)")
    return bad


''')

# 6 probe + stage helpers before to_dev
put("def to_dev(b, dev):", '''def rule_relabel_probe(model, probe_logged, probe_rl, cfg, RR, info) -> dict:
    """G-LIVE for RULETRAIN, read from the BUILT model's own ``loss`` on a COPY, on the real probe batch. off: {"rule_relabel": "off"}. on:
      count        info["n_relabel"] > 0 (zero is a refusal: the lever would do nothing on this data)
      reach        the same loss with the LOGGED gt and with the RELABELLED gt (same seeds): when the rule changed at least one target (n_changed > 0) the ``ade`` term and the gradient of the total differ (> 0)
      controls     the same loss twice on the relabelled batch: term and gradient difference EXACTLY 0.0; an ALL-FALSE mask (RRL.relabel_gt, mask_override) returns the logged target itself and the loss and gradient
                   difference read EXACTLY 0.0
    Any violation is an AssertionError (the caller turns it into a refusal)."""
    if not RR.on:
        return {"rule_relabel": "off", "pass": True}
    if info["n_relabel"] <= 0:
        raise AssertionError("rule relabel probe: no sample of the probe batch has a label lead: the count is 0")
    m = copy.deepcopy(model).cpu()
    m.train()
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]

    def run(b):
        for _, p in named:
            p.grad = None
        torch.manual_seed(20261115)
        out = m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(11))
        T = out["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"rule relabel probe: loss term {k} is not finite: {float(v)}")
        T["total"].backward()
        return float(T["ade"].detach()), float(T["total"].detach()), {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}

    gdiff = lambda ga, gb: float(sum((ga[n] - gb[n]).abs().sum() for n in ga))      # noqa: E731
    a_l, t_l, g_l = run(dict(probe_logged))
    a_r, t_r, g_r = run(dict(probe_rl))
    a_r2, t_r2, g_r2 = run(dict(probe_rl))
    gt_none, i_none = RRL.relabel_gt(probe_logged["gt"], probe_logged["gt_ok"], probe_logged["frame_ok"][:, -1], probe_logged["p10f"][:, -1, 0], probe_logged["wm_speed"][:, -1],
                                     *RR.label_rows(probe_logged["ci"].numpy(), probe_logged["t"].numpy()), mask_override=np.zeros(int(probe_logged["gt"].shape[0]), bool))
    a_n, t_n, g_n = run({**probe_logged, "gt": gt_none})
    rec = {"rule_relabel": "on", "n_relabel": info["n_relabel"], "n_changed": info["n_changed"], "n_rows": info["n_rows"], "ade_logged": a_l, "ade_relabelled": a_r, "total_logged": t_l, "total_relabelled": t_r,
           "ade_diff": abs(a_r - a_l), "grad_diff": gdiff(g_l, g_r), "control_same_batch_ade_diff": abs(a_r - a_r2), "control_same_batch_grad_diff": gdiff(g_r, g_r2),
           "control_all_false_target_equal": bool(torch.equal(gt_none, probe_logged["gt"])), "control_all_false_n": i_none["n_relabel"], "control_all_false_ade_diff": abs(a_n - a_l), "control_all_false_grad_diff": gdiff(g_n, g_l)}
    if rec["control_same_batch_ade_diff"] != 0.0 or rec["control_same_batch_grad_diff"] != 0.0:
        raise AssertionError(f"rule relabel probe: the same batch twice differs ({rec['control_same_batch_ade_diff']}, {rec['control_same_batch_grad_diff']}): the probe is not deterministic")
    if not (rec["control_all_false_target_equal"] and i_none["n_relabel"] == 0 and rec["control_all_false_ade_diff"] == 0.0 and rec["control_all_false_grad_diff"] == 0.0):
        raise AssertionError(f"rule relabel probe: the all-False mask control does not read exactly 0.0: {rec}")
    if info["n_changed"] > 0 and not (rec["ade_diff"] > 0.0 and rec["grad_diff"] > 0.0):
        raise AssertionError(f"rule relabel probe: the rule changed {info['n_changed']} targets but the loss / gradient did not move: {rec}")
    rec["reach_visible"] = bool(info["n_changed"] > 0)
    rec["pass"] = True
    return rec


''')

# 7 main: build the stage after the data
put("    store, J, wins, clips, info = build_data(a, cfg)\n", '''    RR = RRL.RuleRelabel(a.rule_relabel, J, store)                          # RULETRAIN: off = nothing built; on = the lead label arrays of the join (the same rule as lead_sup_label_arrays)
    if RR.on:
        info["rule_relabel"] = {"mode": RR.mode, "labels_sha256": RR.labels["sha256"], **{k: RR.labels[k] for k in ("n_rows", "n_pres", "n_gap", "n_clos")},
                                "dist_rule_sha256": hashlib.sha256(open(RRL.R.__file__, "rb").read()).hexdigest(), "floor_lifts": False, "m": RRL.RULE_M,
                                "constants": {"D_M": RRL.R.D_M, "TAU_S": RRL.R.TAU_S, "B_MAX": RRL.R.B_MAX, "DT": RRL.R.DT}}
''', "after")

# 8 main: relabel the probe batch before the live check
put("        live = live_check(model.cpu(), probe, cfg)\n", '''        probe_logged, rr_i = dict(probe), None
        if RR.on:                                        # RULETRAIN: every probe below calls model.loss on the batch the loop's loss receives, i.e. with the relabelled target; the logged one is kept for the reach probe
            probe["gt"], rr_i = RR.stage(probe)
            info["rule_relabel_n"], info["rule_relabel_changed"], info["rule_relabel_rows"] = rr_i["n_relabel"], rr_i["n_changed"], rr_i["n_rows"]
            if rr_i["n_relabel"] == 0:
                raise SystemExit("ZZDVB-REFUSEDZZ rule-relabel: --rule-relabel on but the G-LIVE probe batch has no sample with a label lead (count 0): a lever that is declared and does nothing")
''')

# 9 probe call
put("    try:\n        lvp = level_head_probe(model, probe, cfg)", '''    try:
        rrp = rule_relabel_probe(model, probe_logged, probe, cfg, RR, rr_i or {}) if enc is None else {"rule_relabel": "off", "pass": True}      # RULETRAIN: G-LIVE (count, reach, controls); off: nothing runs
    except (AssertionError, ValueError) as e:
        raise SystemExit(f"ZZDVB-REFUSEDZZ rule-relabel: {e}")
''')

# 10 levers + sc
put('    levers["init_ckpt"] = info.get("init_ckpt")\n', '''    if RR.on:                                                         # RULETRAIN: lever-ON arms only (a flag-off selfcheck.json keeps its key set); the mode is read from the BUILT stage
        levers["rule_relabel"] = RR.mode
        levers["rule_relabel_rule"] = {k: info["rule_relabel"][k] for k in ("dist_rule_sha256", "floor_lifts", "m", "constants")}
''', "after")
put('    sc["launch_intent_probe"] = lip\n', '''    if RR.on:                                                           # RULETRAIN: lever-ON arms only
        sc["rule_relabel_probe"] = rrp
''', "after")

# 11 loop state
put('    launch_int = {"rows": 0, "eligible": 0, "n": 0}                # ... and since the last log line\n', '''    rr_tot = {"rows": 0, "n": 0, "changed": 0}                      # RULETRAIN run totals: samples seen, relabelled (label lead), of those changed by the rule; carried in the checkpoint of a lever-ON arm only
    rr_int = {"rows": 0, "n": 0, "changed": 0}                      # ... and since the last log line
''', "after")
put('        launch_tot.update(ck.get("launch_totals") or {})\n', '        rr_tot.update(ck.get("rule_relabel_totals") or {})\n', "after")
put("        if launch_on:                                                   # lever-ON arms only: a flag-off checkpoint keeps its key set\n",
    '        if RR.on:                                                       # RULETRAIN: lever-ON arms only: a flag-off checkpoint keeps its key set\n            blob["rule_relabel_totals"] = dict(rr_tot)\n')

# 12 loop: label lookup before the stages, attach after
put("            if level_on:                                           # PREREG_V8F_LEVEL: the label is the LOGGED speed at NOW",
    "            if RR.on:                                              # RULETRAIN: the relabelled target of the LOGGED frame (clip index, step, speed, label lead), taken BEFORE any ego stage\n                rr_gt, rr_i = RR.stage(b)\n")
put("            if level_on:                                           # ... and attached AFTER the stages",
    '''            if RR.on:                                              # RULETRAIN: attached AFTER the stages: the loss reads b["gt"] (ade, fde, acc, launch_along)
                b["gt"] = rr_gt
                for c_ in (rr_int, rr_tot):
                    c_["rows"] += rr_i["n_rows"]
                    c_["n"] += rr_i["n_relabel"]
                    c_["changed"] += rr_i["n_changed"]
''')

# 13 log record
put("                if launch_on:                                               # PREREG_V8F_LAUNCH_INTENT: per log interval", '''                if RR.on:                                                   # RULETRAIN: per log interval (the steps since the last line); a flag-off log line keeps its key set
                    rec["rule_relabel_n"], rec["rule_relabel_changed"], rec["rule_relabel_rows"] = rr_int["n"], rr_int["changed"], rr_int["rows"]
                    rr_int.update(rows=0, n=0, changed=0)
''')

# 14 summary
put("    if level_on:                                                           # PREREG_V8F_LEVEL: the lever-ON arm's run record", '''    if RR.on:                                                              # RULETRAIN: the lever-ON arm's run record (a flag-off summary.json keeps its key set)
        summary["rule_relabel"], summary["rule_relabel_n"], summary["rule_relabel_changed"], summary["rule_relabel_rows"] = "on", rr_tot["n"], rr_tot["changed"], rr_tot["rows"]
''')

open(path, "w", encoding="utf-8", newline="").write(src)
print("trainer edits applied")

# 15 (applied by sed in the build; recorded here): the synthetic world gets its lead track also for rule_relabel on
src2 = open(path, encoding="utf-8", newline="").read()
a_, b_ = '        if a.lead_sup == "on":  ', '        if a.lead_sup == "on" or a.rule_relabel == "on":  '
assert src2.count(a_) == 1, src2.count(a_)
if a_ in src2:
    open(path, "w", encoding="utf-8", newline="").write(src2.replace(a_, b_))

# 16 the loop guard (applied after the edits above in the build; idempotent)
src3 = open(path, encoding="utf-8", newline="").read()
a_ = "            b = to_dev(b, dev)\n"
if "ZZRULE-NOT-APPLIEDZZ" not in src3:
    assert src3.count(a_) == 1
    src3 = src3.replace(a_, a_ + '''            if RR.on and not torch.equal(b["gt"].cpu(), rr_gt):    # RULETRAIN: the batch the loss receives must carry the relabelled target (a deleted `b["gt"] = rr_gt` line is caught here, not 2,000 steps later)
                raise SystemExit(f"ZZRULE-NOT-APPLIEDZZ step {step}: --rule-relabel on but the batch the loss receives does not carry the relabelled target (n_relabel {rr_i['n_relabel']}): a lever that is declared and does nothing")
''')
    open(path, "w", encoding="utf-8", newline="").write(src3)
