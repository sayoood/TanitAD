"""Applies the RULEOWN edits to work/tree_patch/stack/scripts/v8f_mini_train.py (exact-string replacements, each must match exactly once). Re-playable on a copy of tree_base's trainer.
Usage: apply_edits_own.py <trainer path in> <trainer path out>"""
import hashlib
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8", newline="").read()
assert "\r" not in s


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# 1. import
rep("from tanitad.data import rule_relabel as RRL  # noqa: E402",
    "from tanitad.data import rule_own as ROW  # noqa: E402      PREREG_V8F_PROOF_ADD34 RULEOWN: `--w-rule-own` (the distance rule on the model's own plan as a loss term) and `--rule-relabel-weight` (weight of the changed samples); both off = nothing built\n"
    "from tanitad.data import rule_relabel as RRL  # noqa: E402")

# 1b. the synthetic join carries a lead track only when a lever that reads it is on (selfcheck only; a real launch never passes --synthetic)
rep('        if a.lead_sup == "on" or a.rule_relabel == "on":', '        if a.lead_sup == "on" or a.rule_relabel == "on" or a.w_rule_own > 0.0:')

# 2. FLAG_CHECKS
rep('    "init_ckpt": "data:init_ckpt",            # O-RECOVER',
    '    "w_rule_own": "data:rule_own",            # RULEOWN: argv -> RuleOwn (built from the join) -> info["rule_own"] (bad_rule_own) -> rule_own_probe (G-LIVE: term finite, n_lead > 0, n_active, gradient norms, zero controls) -> `ROW.add_own_term` in the loop\n'
    '    "rule_relabel_weight": "data:rule_relabel_weight",      # RULEOWN stage 2: argv -> RuleOwn.w_weight -> info["rule_relabel_weight"] (bad_rule_relabel_weight) -> `ROW.add_weight_extra` in the loop\n'
    '    "init_ckpt": "data:init_ckpt",            # O-RECOVER')

# 3. argparse
rep('    ap.add_argument("--cache-dir", default="C:/Users/Admin/refav1_hj_cache")',
    '    ap.add_argument("--w-rule-own", dest="w_rule_own", type=float, default=0.0,\n'
    '                    help="RULEOWN (PREREG_V8F_PROOF_ADD34): W > 0 adds W x term to the loss. term = mean over the samples with a label lead (the mask and inputs of --rule-relabel) of the mean over the 30 knots of the squared distance (m^2) between the "\n'
    '                         "model own predicted trajectory (out traj30, the output the loss trains against batch gt and the T1-kin tool rolls out) and its cap dist_rule.distance_rule(plan.detach(), v0, present, gap, gap, v_L, m=10, floor_lifts=False). "\n'
    '                         "A plan that already keeps the rule gives exactly 0. Training only, nothing at inference. Refused with --encoder lora, --ego-perturb-p, --dagger-p, --recover-p; may be combined with --rule-relabel on. Default 0.0 = every earlier arm, bit for bit")\n'
    '    ap.add_argument("--rule-relabel-weight", dest="rule_relabel_weight", type=float, default=1.0,\n'
    '                    help="RULEOWN stage 2 (note PREREG_V8F_PROOF_ADD32_N1): the weight, in the trajectory loss terms that read batch gt (ade, fde, and acc / launch_along when on), of the samples whose target --rule-relabel CHANGED. Not renormalised "\n'
    '                         "(4 = those samples count four times). Needs --rule-relabel on when != 1. Default 1.0 = every earlier arm, bit for bit")\n'
    '    ap.add_argument("--cache-dir", default="C:/Users/Admin/refav1_hj_cache")')

# 4. G-DVB clauses
rep("def declared_vs_built(a, cfg, model, opt, data_info, enc=None",
    'def bad_rule_own(a, data_info) -> list:\n'
    '    """RULEOWN (G-DVB): ``--w-rule-own`` and ``--rule-relabel-weight`` read back from the BUILT stage record ``info["rule_own"]`` (the weights RuleOwn holds, the rule file sha256, the label rows). W = 0 (off): no record may exist. W > 0: the record\n'
    '    equals the declared value, the label arrays have lead rows, and the stages that rewrite the logged state (ego perturbation, DAgger swap, recovery, LoRA frames) are off. ``--rule-relabel-weight`` != 1 needs ``--rule-relabel on`` and the same built value."""\n'
    '    rec = (data_info or {}).get("rule_own")\n'
    '    bad = []\n'
    '    if not (a.w_rule_own >= 0.0 and a.w_rule_own == a.w_rule_own and a.w_rule_own != float("inf")):\n'
    '        bad.append(f"w_rule_own {a.w_rule_own!r} must be finite and >= 0")\n'
    '    if not (a.rule_relabel_weight >= 0.0 and a.rule_relabel_weight == a.rule_relabel_weight and a.rule_relabel_weight != float("inf")):\n'
    '        bad.append(f"rule_relabel_weight {a.rule_relabel_weight!r} must be finite and >= 0")\n'
    '    if a.w_rule_own == 0.0:\n'
    '        if rec and rec.get("w_rule_own") != 0.0:\n'
    '            bad.append("w_rule_own 0 declared but a rule-own stage with a weight was built")\n'
    '    else:\n'
    '        if not rec:\n'
    '            return bad + ["w_rule_own > 0 declared but no rule-own stage was built"]\n'
    '        if rec.get("w_rule_own") != a.w_rule_own:\n'
    '            bad.append(f"w_rule_own: declared {a.w_rule_own} but built {rec.get(\'w_rule_own\')}")\n'
    '        for k in ("n_pres", "n_gap", "n_clos"):\n'
    '            if not (isinstance(rec.get(k), int) and rec[k] > 0):\n'
    '                bad.append(f"rule_own labels: {k} = {rec.get(k)!r}: no label lead rows (a lever that is declared and does nothing)")\n'
    '        if not rec.get("labels_sha256") or not rec.get("dist_rule_sha256"):\n'
    '            bad.append("rule_own: the label arrays\' or the rule file\'s sha256 is not recorded")\n'
    '        for flag, val in (("ego_perturb_p", a.ego_perturb_p), ("dagger_p", a.dagger_p), ("recover_p", a.recover_p)):\n'
    '            if val != 0.0:\n'
    '                bad.append(f"w_rule_own > 0 with --{flag.replace(\'_\', \'-\')} {val}: the rule is applied to the LOGGED frame, which that stage rewrites")\n'
    '        if a.encoder != "cache":\n'
    '            bad.append("w_rule_own is not built for --encoder lora (the loop\'s batch path differs)")\n'
    '    wrec = (data_info or {}).get("rule_relabel_weight")\n'
    '    if a.rule_relabel_weight == 1.0:\n'
    '        if wrec:\n'
    '            bad.append("rule_relabel_weight 1 declared but a weight stage was built")\n'
    '    else:\n'
    '        if a.rule_relabel != "on":\n'
    '            bad.append(f"rule_relabel_weight {a.rule_relabel_weight} needs --rule-relabel on (the weight is on the samples the relabel changed)")\n'
    '        if not wrec:\n'
    '            bad.append("rule_relabel_weight != 1 declared but no weight stage was built")\n'
    '        elif wrec.get("w") != a.rule_relabel_weight:\n'
    '            bad.append(f"rule_relabel_weight: declared {a.rule_relabel_weight} but built {wrec.get(\'w\')}")\n'
    '    return bad\n'
    '\n\n'
    "def declared_vs_built(a, cfg, model, opt, data_info, enc=None")
rep("    bad += bad_rule_relabel(a, data_info)                                                        # RULETRAIN:",
    "    bad += bad_rule_own(a, data_info)                                                            # RULEOWN: --w-rule-own and --rule-relabel-weight read back from the BUILT stage\n"
    "    bad += bad_rule_relabel(a, data_info)                                                        # RULETRAIN:")

# 5. G-LIVE probe function (before to_dev)
rep("def to_dev(b, dev):",
    'def rule_own_probe(model, probe_logged, RO, inp, cfg) -> dict:\n'
    '    """G-LIVE for RULEOWN flag A, read from the BUILT model on a COPY, on the real probe batch. off: {"rule_own": "off"}. on, with out["traj30"] of the model\'s own loss call on the probe batch:\n'
    '      count     n_lead (samples with a label lead) > 0, else a refusal\n'
    '      term      finite, >= 0; n_active (samples with a term above 0); the gradient norm of W x term and of the trajectory loss (w_traj x (ade / 3 + 0.5 fde / 6)) on traj30\n'
    '      reach     the loop\'s own add helper moves the total by W x term; when n_active > 0 the gradient norm of the term on traj30 is above 0\n'
    '      controls  an all-False mask gives term exactly 0.0 and gradient norm exactly 0.0; the same call twice gives the same term\n'
    '    Any violation is an AssertionError (the caller turns it into a refusal)."""\n'
    '    if not RO.own_on:\n'
    '        return {"rule_own": "off", "pass": True}\n'
    '    if int(inp["mask"].sum()) <= 0:\n'
    '        raise AssertionError("rule own probe: no sample of the probe batch has a label lead: the count is 0")\n'
    '    m = copy.deepcopy(model).cpu()\n'
    '    m.train()\n'
    '    torch.manual_seed(20261115)\n'
    '    out = m.loss(dict(probe_logged), p_gt=0.5, gen=torch.Generator().manual_seed(11))\n'
    '    tr, T = out["traj30"], out["terms"]\n'
    '    total0 = T["total"].detach().clone()\n'
    '    term, info = ROW.own_term(tr, inp)\n'
    '    term2, _ = ROW.own_term(tr, inp)\n'
    '    traj_loss = cfg.w_traj * (T["ade"] / 3.0 + 0.5 * T["fde"] / 6.0)\n'
    '    gn = lambda y: float(torch.autograd.grad(y, tr, retain_graph=True, allow_unused=True)[0].abs().sum()) if y.requires_grad else 0.0      # noqa: E731\n'
    '    g_term, g_traj = gn(RO.w_own * term), gn(traj_loss)\n'
    '    Tc = {"total": T["total"]}\n'
    '    info2 = ROW.add_own_term(Tc, tr, inp, RO.w_own)\n'
    '    none_inp = {**inp, "mask": np.zeros_like(inp["mask"])}\n'
    '    term0, info0 = ROW.own_term(tr, none_inp)\n'
    '    rec = {"rule_own": "on", "w_rule_own": RO.w_own, "term": float(term.detach()), "n_rows": info["n_rows"], "n_lead": info["n_lead"], "n_active": info["n_active"], "grad_norm_term_on_traj30": g_term,\n'
    '           "grad_norm_traj_loss_on_traj30": g_traj, "total_before": float(total0), "total_after": float(Tc["total"].detach()), "total_delta": float(Tc["total"].detach() - total0),\n'
    '           "control_same_call_term_diff": abs(float(term.detach()) - float(term2.detach())), "control_all_false_term": float(term0.detach()), "control_all_false_n_lead": info0["n_lead"],\n'
    '           "control_all_false_requires_grad": bool(term0.requires_grad)}\n'
    '    if not np.isfinite(rec["term"]) or rec["term"] < 0.0 or not np.isfinite(g_term) or not np.isfinite(g_traj):\n'
    '        raise AssertionError(f"rule own probe: the term or a gradient norm is not finite / not >= 0: {rec}")\n'
    '    if rec["control_same_call_term_diff"] != 0.0 or rec["control_all_false_term"] != 0.0 or rec["control_all_false_n_lead"] != 0 or rec["control_all_false_requires_grad"]:\n'
    '        raise AssertionError(f"rule own probe: a control does not read exactly 0.0: {rec}")\n'
    '    if info["n_active"] > 0 and not (g_term > 0.0 and abs(rec["total_delta"] - RO.w_own * rec["term"]) <= 1e-5 * (1.0 + abs(rec["total_before"]))):\n'
    '        raise AssertionError(f"rule own probe: n_active {info[\'n_active\']} > 0 but the term does not reach the total / the trajectory output: {rec}")\n'
    '    if info["n_active"] == 0 and rec["term"] != 0.0:\n'
    '        raise AssertionError(f"rule own probe: n_active 0 but the term is {rec[\'term\']}: {rec}")\n'
    '    rec["reach_visible"] = bool(info["n_active"] > 0)\n'
    '    rec["pass"] = True\n'
    '    return rec\n'
    '\n\n'
    "def to_dev(b, dev):")

# 6. build the stage after RR
rep('    info["p10_units"] = None if a.synthetic else M.p10_meta(a.p10).get("units", "steer")',
    '    RO = ROW.RuleOwn(a.w_rule_own, a.rule_relabel_weight, J, store, share=RR)       # RULEOWN: off (0.0 and 1.0) = nothing built; flag A builds from the join (or shares RR\'s label arrays)\n'
    '    if RO.own_on:\n'
    '        src = RO.src\n'
    '        info["rule_own"] = {"w_rule_own": RO.w_own, "labels_sha256": src.labels["sha256"], **{k: src.labels[k] for k in ("n_rows", "n_pres", "n_gap", "n_clos")},\n'
    '                            "dist_rule_sha256": hashlib.sha256(open(RRL.R.__file__, "rb").read()).hexdigest(), "floor_lifts": False, "m": ROW.RULE_M, "output": "traj30",\n'
    '                            "constants": {"D_M": RRL.R.D_M, "TAU_S": RRL.R.TAU_S, "B_MAX": RRL.R.B_MAX, "DT": RRL.R.DT}}\n'
    '    if RO.weight_on:\n'
    '        info["rule_relabel_weight"] = {"w": RO.w_weight, "terms": ["ade", "fde"] + (["acc"] if a.w_acc > 0.0 else []) + (["launch_along"] if a.launch_intent_w > 0.0 else []), "renormalised": False}\n'
    '    info["p10_units"] = None if a.synthetic else M.p10_meta(a.p10).get("units", "steer")')

# 7. probe: relabel stage in the live check; own inputs
rep("        live = live_check(model.cpu(), probe, cfg)\n    else:",
    '        ro_in = RO.stage(probe_logged) if RO.own_on else None          # RULEOWN: the inputs of flag A on the LOGGED probe batch (the same mask as the relabel)\n'
    '        live = live_check(model.cpu(), probe, cfg)\n    else:')
rep("    try:\n        lvp = level_head_probe(model, probe, cfg)",
    '    try:\n'
    '        rop = rule_own_probe(model, probe_logged, RO, ro_in, cfg) if enc is None else {"rule_own": "off", "pass": True}      # RULEOWN: G-LIVE (count, term, gradients, controls); off: nothing runs\n'
    '    except (AssertionError, ValueError) as e:\n'
    '        raise SystemExit(f"ZZDVB-REFUSEDZZ rule-own: {e}")\n'
    "    try:\n        lvp = level_head_probe(model, probe, cfg)")

# 8. levers + selfcheck
rep('    levers["lr_schedule"] = lr_at.built_levers()',
    '    if RO.own_on:                                                     # RULEOWN: lever-ON arms only (a flag-off selfcheck.json keeps its key set)\n'
    '        levers["w_rule_own"] = RO.w_own\n'
    '        levers["rule_own_rule"] = {k: info["rule_own"][k] for k in ("dist_rule_sha256", "floor_lifts", "m", "constants", "output")}\n'
    '    if RO.weight_on:\n'
    '        levers["rule_relabel_weight"] = RO.w_weight\n'
    '    levers["lr_schedule"] = lr_at.built_levers()')
rep("    if cfg.level_head == \"on\":                                          # lever-ON arms only: a flag-off selfcheck.json keeps its key set\n        sc[\"level_head_probe\"] = lvp",
    '    if RO.own_on:                                                       # RULEOWN: lever-ON arms only\n'
    '        sc["rule_own_probe"] = rop\n'
    "    if cfg.level_head == \"on\":                                          # lever-ON arms only: a flag-off selfcheck.json keeps its key set\n        sc[\"level_head_probe\"] = lvp")

# 9. run totals
rep("    rr_int = {\"rows\": 0, \"n\": 0, \"changed\": 0}                      # ... and since the last log line\n",
    "    rr_int = {\"rows\": 0, \"n\": 0, \"changed\": 0}                      # ... and since the last log line\n"
    "    ro_tot = {\"rows\": 0, \"n_lead\": 0, \"n_active\": 0, \"term_sum\": 0.0, \"steps\": 0}      # RULEOWN run totals (flag A); carried in the checkpoint of a lever-ON arm only\n"
    "    ro_int = {\"n_lead\": 0, \"n_active\": 0, \"term_sum\": 0.0, \"steps\": 0, \"weighted\": 0}      # ... and since the last log line (weighted = samples carrying the flag-B weight)\n"
    "    rw_tot = {\"weighted\": 0}                                      # RULEOWN stage 2 run total\n")
rep('        rr_tot.update(ck.get("rule_relabel_totals") or {})\n',
    '        rr_tot.update(ck.get("rule_relabel_totals") or {})\n'
    '        ro_tot.update(ck.get("rule_own_totals") or {})\n'
    '        rw_tot.update(ck.get("rule_relabel_weight_totals") or {})\n')
rep('            blob["rule_relabel_totals"] = dict(rr_tot)\n',
    '            blob["rule_relabel_totals"] = dict(rr_tot)\n'
    '        if RO.own_on:                                                   # RULEOWN: lever-ON arms only (the model cfg gets NO new key)\n'
    '            blob["rule_own_totals"] = dict(ro_tot)\n'
    '        if RO.weight_on:\n'
    '            blob["rule_relabel_weight_totals"] = dict(rw_tot)\n')

# 10. loop: stage before the ego stages
rep("                rr_gt, rr_i = RR.stage(b)\n",
    "                rr_gt, rr_i = RR.stage(b)\n"
    "            if RO.own_on:                                          # RULEOWN flag A: the mask and the rule inputs of the LOGGED frame, taken BEFORE any ego stage (a swapped sample's ci / t is the own one)\n"
    "                ro_in = RO.stage(b)\n")

# 11. loop: the term and the weight after the loss
rep("                T = out[\"terms\"]\n                if not torch.isfinite(T[\"total\"]):",
    "                T = out[\"terms\"]\n"
    "                ro_i, tot0 = None, T[\"total\"]\n"
    "                if RO.own_on:                                       # RULEOWN flag A: ONE call adds W x term to the total (a deleted add line is caught by the guard below)\n"
    "                    ro_i = ROW.add_own_term(T, out[\"traj30\"], ro_in, RO.w_own)\n"
    "                    if T[\"total\"] is tot0 and ro_i[\"n_active\"] > 0:\n"
    "                        raise SystemExit(f\"ZZRULE-OWN-NOT-APPLIEDZZ step {step}: --w-rule-own {RO.w_own} but the term did not reach the total (n_lead {ro_i['n_lead']}, n_active {ro_i['n_active']}): a lever that is declared and does nothing\")\n"
    "                if RO.weight_on:                                    # RULEOWN flag B: the extra loss of the changed samples; the count of weighted samples is the relabel's own changed mask\n"
    "                    rw_chg = ROW.changed_vector(rr_i, int(b[\"gt\"].shape[0]))\n"
    "                    tot1 = T[\"total\"]\n"
    "                    ROW.add_weight_extra(T, out[\"traj30\"], b[\"gt\"], b[\"gt_ok\"], rw_chg, RO.w_weight, cfg.w_traj, cfg.w_acc, b.get(\"launch\"), cfg.launch_intent_w)\n"
    "                    if T[\"total\"] is tot1 and int(rw_chg.sum()) > 0:\n"
    "                        raise SystemExit(f\"ZZRULE-WEIGHT-NOT-APPLIEDZZ step {step}: --rule-relabel-weight {RO.w_weight} but the extra loss did not reach the total ({int(rw_chg.sum())} changed samples): a lever that is declared and does nothing\")\n"
    "                    ro_int[\"weighted\"] += int(rw_chg.sum())\n"
    "                    rw_tot[\"weighted\"] += int(rw_chg.sum())\n"
    "                if ro_i is not None:\n"
    "                    for c_ in (ro_int, ro_tot):\n"
    "                        c_[\"n_lead\"] += ro_i[\"n_lead\"]\n"
    "                        c_[\"n_active\"] += ro_i[\"n_active\"]\n"
    "                        c_[\"term_sum\"] += ro_i[\"term\"]\n"
    "                        c_[\"steps\"] += 1\n"
    "                    ro_tot[\"rows\"] += ro_i[\"n_rows\"]\n"
    "                if not torch.isfinite(T[\"total\"]):")

# 12. log keys
rep("                if launch_on:                                               # PREREG_V8F_LAUNCH_INTENT: per log interval (the steps since the last line)\n                    rec[\"launch_n\"]",
    "                if RO.own_on:                                               # RULEOWN flag A: per log interval (term = mean over the steps since the last line); a flag-off log line keeps its key set\n"
    "                    rec[\"rule_own_term\"] = ro_int[\"term_sum\"] / max(ro_int[\"steps\"], 1)\n"
    "                    rec[\"rule_own_n_lead\"], rec[\"rule_own_n_active\"] = ro_int[\"n_lead\"], ro_int[\"n_active\"]\n"
    "                if RO.weight_on:                                            # RULEOWN flag B: the samples that carried the weight since the last line\n"
    "                    rec[\"rule_relabel_weighted_n\"] = ro_int[\"weighted\"]\n"
    "                ro_int.update(n_lead=0, n_active=0, term_sum=0.0, steps=0, weighted=0)\n"
    "                if launch_on:                                               # PREREG_V8F_LAUNCH_INTENT: per log interval (the steps since the last line)\n                    rec[\"launch_n\"]")

# 13. summary
rep("    if level_on:                                                           # PREREG_V8F_LEVEL: the lever-ON arm's run record",
    "    if RO.own_on:                                                          # RULEOWN: the lever-ON arm's run record (a flag-off summary.json keeps its key set)\n"
    "        summary[\"w_rule_own\"], summary[\"rule_own_n_lead\"], summary[\"rule_own_n_active\"], summary[\"rule_own_rows\"] = RO.w_own, ro_tot[\"n_lead\"], ro_tot[\"n_active\"], ro_tot[\"rows\"]\n"
    "        summary[\"rule_own_term_mean\"] = ro_tot[\"term_sum\"] / max(ro_tot[\"steps\"], 1)\n"
    "    if RO.weight_on:\n"
    "        summary[\"rule_relabel_weight\"], summary[\"rule_relabel_weighted_n\"] = RO.w_weight, rw_tot[\"weighted\"]\n"
    "    if level_on:                                                           # PREREG_V8F_LEVEL: the lever-ON arm's run record")

open(dst, "w", encoding="utf-8", newline="").write(s)
print("OK", hashlib.sha256(s.encode("utf-8")).hexdigest())
