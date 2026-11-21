from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/a3_keq_compare.py"); s = p.read_text(encoding="utf-8")
rep = [
("""    added, pre_bad, nonr9_bad, addr = set(), 0, {}, set()
    if same_seq:
        for a, b in zip(Don, Doff):""",
"""    added, pre_bad, nonr9_bad, addr = set(), 0, {}, set()
    # PLAN ALIASES: outputs that ARE the emitted plan tensor (refc_v3.py:2742 `out["wp_seq"] = traj`, set AFTER the R9 re-time at :2723).
    # Established from data, not assumed: digest(k) == digest("traj") on EVERY call in BOTH runs.  Such a key carries R9's plan by
    # construction, so it is checked like `traj`: ON traj_preretime == OFF k.
    aliases = sorted(k for k in (set(Doff[0]["d"]) & set(Don[0]["d"])) - {"traj"}
                     if all(c["d"].get(k) == c["d"]["traj"] for c in Don) and all(c["d"].get(k) == c["d"]["traj"] for c in Doff)) if same_seq else []
    alias_bad = 0
    if same_seq:
        for a, b in zip(Don, Doff):"""),
("""                if k == "traj":
                    if a["d"].get("traj_preretime", "<absent>") != v:
                        pre_bad += 1""",
"""                if k == "traj":
                    if a["d"].get("traj_preretime", "<absent>") != v:
                        pre_bad += 1
                elif k in aliases:
                    if a["d"].get("traj_preretime", "<absent>") != v:
                        alias_bad += 1"""),
("""    out.update({"keys_added_by_R9": sorted(added),""", """    out.update({"plan_aliases_checked_as_traj": aliases, "plan_alias_source": "refc_v3.py:2742 out['wp_seq'] = traj (after the R9 re-time, :2723)",
                "calls_alias_ne_off": alias_bad, "keys_added_by_R9": sorted(added),"""),
("""    out["PASS"] = bool(out["subset_equal"] and out["same_tree"] and out["same_refc_v3_sha"] and same_seq and pre_bad == 0 and not nonr9_bad""",
 """    out["PASS"] = bool(out["subset_equal"] and out["same_tree"] and out["same_refc_v3_sha"] and same_seq and pre_bad == 0 and alias_bad == 0 and not nonr9_bad"""),
# K-EQ beside reads
("""    res["PASS"] = bool(n >= A.min_windows and res["n_windows_over_tol"] == 0 and dec_agree == 1.0)""",
 """    res["PASS"] = bool(n >= A.min_windows and res["n_windows_over_tol"] == 0 and dec_agree == 1.0)
    # ---- BESIDE (not the registered reading): where the residual comes from
    import torch
    R = RC.r9p()
    cfg = R.R9Config(**blob["r9_config"])
    with torch.no_grad():
        o2 = R.head_forward(h, cfg, torch.from_numpy(np.ascontiguousarray(z["z"][m], np.float32)), torch.from_numpy(np.ascontiguousarray(z["x"][m], np.float32)),
                            torch.from_numpy(np.ascontiguousarray(z["s_p"][m], np.float32)), None)
        t2 = R.retime(torch.from_numpy(np.ascontiguousarray(z["pre"][m], np.float32)), o2["r9_s_target"]).to(torch.float32).numpy().astype(np.float64)
    d2 = np.abs(fwd - t2).reshape(n, -1).max(1)
    pr = np.abs(I["S2_prior8"][ji[m]].astype(np.float64)).reshape(n, -1).max(1)
    tol_in = 2.0 * np.spacing(np.maximum(coord, pr).astype(np.float32)).astype(np.float64)
    res["BESIDE_not_registered"] = {
        "A_forward_own_s_prior": {"rule": "offline head_forward + retime with the forward's OWN s_prior (r9_s_prior) instead of arc_at_slots(scoring prior8)",
                                  "max_abs_m": float(d2.max()), "n_windows_over_2ulp_output_coord": int((d2 > tol).sum()),
                                  "decision_agreement": float((z["dec"][m] == o2["r9_decision"].numpy()).mean())},
        "B_coordinate_scope_incl_prior": {"rule": "registered offline composition; tolerance 2 ulp of max |coordinate| over the window's plan AND its prior8",
                                          "n_windows_over": int((d > tol_in).sum()), "max_ratio": float((d / tol_in).max())}}"""),
]
for a, b in rep:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8"); print("ok")
