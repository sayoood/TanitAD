import json, sys
d = json.load(open(sys.argv[1]))
print("arm=%s n=%d eps=%d anchors=%d wall=%.0fs" % (d["arm"], d["n_windows"], d["n_episodes"], d["n_anchors"], d["wall_s"]))
print("reach:", json.dumps(d["reachability"]))
s = d["shipped"]
print("shipped: ade=%.4f rank_acc=%.4f f2x=%.4f sel_gap=%.4f" % (
    s["ade_0_2s"]["mean"], s["rank_acc"]["mean"], s["frac_sel_2x_worse"]["mean"], s["sel_gap"]["mean"]))
print()
print("=== ARMS: LOEO selection ADE@2s, paired vs shipped ===")
hdr = "%-16s %-5s %8s %9s %-22s %-5s %8s %8s %8s %8s"
print(hdr % ("set", "ctrl", "ADE", "delta", "CI95", "sep", "rank_ac", "f2x", "leak_m", "rho_rch"))
for k, v in d["arms"].items():
    p = v["paired_vs_shipped"]
    print(hdr % (k, str(v["degenerate_control"])[:5], "%.4f" % v["loeo"]["ade_0_2s"]["mean"],
                 "%+.4f" % p["delta"], "[%+.4f,%+.4f]" % (p["lo"], p["hi"]), str(p["separated"])[:5],
                 "%.4f" % v["loeo"]["rank_acc"]["mean"], "%.4f" % v["loeo"]["frac_sel_2x_worse"]["mean"],
                 "%+.4f" % v["C-leak_gap_m"], "%+.4f" % v["rho_reachable"]["mean"]))
print()
print("=== CONTROLS ===")
for k, v in d["controls"].items():
    print(k, "::", json.dumps(v)[:400])
print()
print("=== S1b ===")
print(json.dumps(d["s1b"], indent=1)[:2200])
print()
print("=== VERDICT ===")
print(json.dumps(d["verdict"], indent=1))
