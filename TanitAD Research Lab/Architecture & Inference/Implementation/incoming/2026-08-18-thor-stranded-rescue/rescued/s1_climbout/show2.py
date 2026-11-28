import json, sys
d = json.load(open(sys.argv[1]))
print("### arm=%s n=%d eps=%d anchors=%d wall=%.0fs" % (d["arm"], d["n_windows"], d["n_episodes"], d["n_anchors"], d["wall_s"]))
s = d["shipped"]
print("shipped ade=%.4f rank_acc=%.4f f2x=%.4f" % (s["ade_0_2s"]["mean"], s["rank_acc"]["mean"], s["frac_sel_2x_worse"]["mean"]))
hdr = "%-16s %-5s %8s %9s %-22s %-5s %8s %8s %8s"
print("\n== CE objective (REGISTERED) ==")
print(hdr % ("set","ctrl","ADE","delta","CI95","sep","rank_ac","f2x","leak_m"))
for k, v in d["arms"].items():
    p = v["paired_vs_shipped"]
    print(hdr % (k, str(v["degenerate_control"])[:5], "%.4f"%v["loeo"]["ade_0_2s"]["mean"], "%+.4f"%p["delta"],
                 "[%+.4f,%+.4f]"%(p["lo"],p["hi"]), str(p["separated"])[:5],
                 "%.4f"%v["loeo"]["rank_acc"]["mean"], "%.4f"%v["loeo"]["frac_sel_2x_worse"]["mean"], "%+.4f"%v["C-leak_gap_m"]))
print("\n== soft-ADE objective (POST-HOC) ==")
for k, v in d["objective_diagnostic"]["arms"].items():
    p = v["paired_vs_shipped"]; q = v["paired_vs_same_features_under_CE"]
    print("%-16s ADE=%.4f  vs_shipped=%+.4f [%+.4f,%+.4f] sep=%s | vs_CE=%+.4f [%+.4f,%+.4f] sep=%s  w=%s" % (
        k, v["loeo"]["ade_0_2s"]["mean"], p["delta"], p["lo"], p["hi"], p["separated"],
        q["delta"], q["lo"], q["hi"], q["separated"], [round(x,3) for x in v["weights"]["w_mean"]]))
print("\n== CONTROLS ==")
c = d["controls"]
print("C-monotone:", c["C-monotone"]["shipped_matches"], c["C-monotone"]["refined_matches"],
      "| A-shipped=%.6f shipped=%.6f | A-refined=%.6f refined=%.6f" % (
      c["C-monotone"]["A-shipped_ade"], c["C-monotone"]["shipped_ade"],
      c["C-monotone"]["A-refined_ade"], c["C-monotone"]["refined_argmax_ade"]))
pt = c["C-permuted-target"]
print("C-permuted-target: ade=%.4f floor=%.4f paired=%+.4f [%+.4f,%+.4f] sep=%s" % (
      pt["ade"], pt["uniform_floor_over_survivors"], pt["paired_vs_floor"]["delta"],
      pt["paired_vs_floor"]["lo"], pt["paired_vs_floor"]["hi"], pt["paired_vs_floor"]["separated"]))
print("C-identity dev=%.2e  C-oracle dev=%.2e" % (c["C-reproduce"]["abs_deviation"], c["C-oracle-floor"]["abs_deviation"]))
b = d["s1b"]
print("\n== S1b ==", json.dumps(b["controls"]))
print("emitted ade=%.4f rank_acc=%.4f f2x=%.4f | prefinal ade=%.4f rank_acc=%.4f f2x=%.4f" % (
    b["emitted"]["ade_0_2s"]["mean"], b["emitted"]["rank_acc"]["mean"], b["emitted"]["frac_sel_2x_worse"]["mean"],
    b["prefinal"]["ade_0_2s"]["mean"], b["prefinal"]["rank_acc"]["mean"], b["prefinal"]["frac_sel_2x_worse"]["mean"]))
pe = b["paired_emitted_minus_prefinal"]
print("paired emitted-prefinal = %+.4f [%+.4f,%+.4f] sep=%s | agree=%.4f corr=%.4f" % (
    pe["delta"], pe["lo"], pe["hi"], pe["separated"], b["agreement_emitted_vs_prefinal"], b["corr_emitted_prefinal"]))
print("\n== FOUR FAMILIES: paired vs shipped ==")
for arm in ("B-both", "C-lon", "D-lon+scores", "S1b-emitted"):
    if arm not in d["families"]: continue
    fp = d["families"][arm]["paired_vs_shipped"]
    print(" -- %s (dt=%.2f s, %s)" % (arm, fp["_dt_s"], fp["_dt_provenance"]))
    for fam in ("LONGITUDINAL", "LATERAL"):
        for k, v in fp[fam].items():
            print("    %-12s %-24s %+.4f [%+.4f,%+.4f] sep=%s" % (fam, k, v["delta"], v["lo"], v["hi"], v["separated"]))
print("\n == CEILING oracle-minus-shipped ==")
fc = d["families"]["_ceiling_oracle_minus_shipped"]
for fam in ("LONGITUDINAL", "LATERAL"):
    for k, v in fc[fam].items():
        print("    %-12s %-24s %+.4f [%+.4f,%+.4f] sep=%s" % (fam, k, v["delta"], v["lo"], v["hi"], v["separated"]))
print("\n== VERDICT =="); print(json.dumps(d["verdict"], indent=1))
