"""Print the S1b headline numbers from the artifacts (for the RESULT.md / hand-back first line).  numpy-free; reads json only."""
import json
import os

R = "C:/Users/Admin/par18/unseen_s1b"
st = json.load(open(f"{R}/state.json")) if os.path.exists(f"{R}/state.json") else {}
print("STATE", {k: st.get(k) for k in ("stage", "updated", "stage1_wall_s", "seq1_cpu_s", "seq1_unfilled", "total_wall_s", "error")})
p = f"{R}/raw/identity_s1b.json"
if os.path.exists(p):
    r = json.load(open(p))
    f = r["frozen_filter"]
    print("IDENTITY n_logs", r["n_logs"], "n_cand", r["n_candidates"], "n_known", r["n_known"], "| recall %.4f %s | precision %.4f %s | S1 window-rule-only precision %.4f" % (
        f["recall_micro"], f["recall_ci95_log_cluster_bootstrap"], f["precision_micro"], f["precision_ci95_log_cluster_bootstrap"], f["S1_window_rule_only_precision_on_these_logs"]))
    print("CLAUSES", r["clauses"])
    print("RESIDUAL false removals", r["residual_false_removals_(known navtrain tokens our filter removes)"], r["residual_false_removals_margins"])
    print("RESIDUAL false keeps", {k: v for k, v in r["residual_false_keeps_(kept, not in navtrain)"].items() if k in ("n", "share_of_kept", "within_0.05_of_a_threshold")})
    print("INCLUSIVITY", r["inclusivity"])
    for k, v in r["red_arms"].items():
        print("ARM", k, v)
    print("CONTROL", r["control_shuffle_within_log"])
    print("BEST_F1_SWEEP (diagnostic)", r["threshold_sweep_diagnostic_best_f1"])
    print("SHARES", json.dumps(r["population_shares"]))
    print("SHARE DIFF CI", r["population_share_diff_ours_minus_navtrain_ci95"])
    c = r["cost"]
    print("COST cpu_s/token incl overheads %.3f | median warm %.3f | excl top1%% %.3f | proj" % (c["cpu_s_per_token_incl_all_overheads"], c["cpu_s_per_token_median_warm"], c["cpu_s_per_token_mean_excl_top_1pct"]), c["projection"])
p = f"{R}/raw/fetch_list_s2_s4_filtered.json"
if os.path.exists(p):
    r = json.load(open(p))
    print("FETCH", {k: r[k] for k in ("n_logs", "n_tokens", "n_jpgs", "n_unique_paths", "size_MB_ESTIMATED", "n_unfilled_logs", "tokens_scored_total", "tokens_accepted_total", "scored_per_accepted_overall", "max_dt_ms")})
    print("FETCH acceptance", r["acceptance_rate_across_logs"])
    print("FETCH U_extrapolation", r["U_extrapolation"])
    print("FETCH shares", r["accepted_token_population_shares_design_weighted_equal_log"])
    print("FETCH unfilled", r["unfilled"][:12])
