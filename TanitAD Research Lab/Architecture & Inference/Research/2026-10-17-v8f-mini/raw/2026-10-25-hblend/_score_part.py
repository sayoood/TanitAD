# ------------------------------------------------------------------------------------------------ score
BANK = f"{OUT}/inputs/banked"
P10_FILE = f"{OUT}/inputs/b2_p10_eval_pred6.npy"            # P10-yaw-4512's per-window eval plan (== repo refav1-hj-mini/raw/p10_yaw4512/, same sha256)
LOWER_IS_BETTER = {"tac_lat_acc": False, "tac_lon_acc": False}
FW_KEYS = ("lon_speed_mae", "lon_accel_mae", "lon_along_mae", "lat_cross_mae", "lat_heading_mae_deg", "lat_yaw_rate_mae_degps",
           "lat_curvature_mae_1pm", "tac_lat_acc", "tac_lon_acc")


def eval_world():
    from tanitad.data import v8f_mini_data as D
    grid = json.load(open(f"{WORK}/inputs/eval_grid_windows.json"))
    J = D.load_join(f"{WORK}/inputs/join_eval141.npz")
    X = D.eval_context(grid, f"{WORK}/inputs/eval141_kin.npz", J, kin_mode="yaw")
    return grid, J, X


def load_eval_module():
    import importlib.util
    sp = importlib.util.spec_from_file_location("v8f_mini_eval", f"{TREE}/stack/scripts/v8f_mini_eval.py")
    m = importlib.util.module_from_spec(sp)
    sys.modules["v8f_mini_eval"] = m
    sp.loader.exec_module(m)
    return m


def arm_pw(pred, gt, BE):
    """Per-window scalars of one arm: ADE / FDE at 1..6 s and the family components (taniteval's own `family_windows`)."""
    pred = np.asarray(pred, np.float64)
    d = np.linalg.norm(pred - gt, axis=-1)
    pw = {}
    for h in range(1, 7):
        pw[f"ade{h}"] = d[:, :5 * h].mean(1)
        pw[f"fde{h}"] = d[:, 5 * h - 1]
    for k, v in BE.family_windows(pred, gt, DT).items():
        pw["fw_" + k] = v
    return pw


def single(v, eid, CI, nb):
    v, eid = np.asarray(v, np.float64), np.asarray(eid)
    ok = np.isfinite(v)
    r = CI.episode_cluster_bootstrap(v[ok], eid[ok], n_boot=nb, seed=0)
    return {"mean": float(r["mean"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "n_windows": int(ok.sum())}


def prow(a, b, eid, BE, nb, key=None):
    return BE.paired_row(a, b, eid, nb, lower_is_better=LOWER_IS_BETTER.get((key or "").replace("fw_", ""), True))


def rows_for(pw_a, pw_b, eid, BE, nb):
    return {k: prow(pw_a[k], pw_b[k], eid, BE, nb, k) for k in pw_a}


def deep_diff(x, y, path="", tol=1e-6, out=None):
    out = [] if out is None else out
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append(f"{path}/{k}: only in {'a' if k in x else 'b'}")
            else:
                deep_diff(x[k], y[k], f"{path}/{k}", tol, out)
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            out.append(f"{path}: len {len(x)} vs {len(y)}")
        else:
            for i, (u, v) in enumerate(zip(x, y)):
                deep_diff(u, v, f"{path}[{i}]", tol, out)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
        if not (x == y or (np.isnan(x) and np.isnan(y)) or abs(x - y) <= tol * max(1.0, abs(x), abs(y))):
            out.append(f"{path}: {x} vs {y}")
    elif x != y:
        out.append(f"{path}: {str(x)[:60]} vs {str(y)[:60]}")
    return out


def extras_of(z, D):
    return {"goal_prob": z["goal_prob"], "lat_logits": z["lat_logits"], "lon_logits": z["lon_logits"],
            "cons": {n: z["cons_pred"][:, D.CONS_IDX[n]] for n in D.CONS_FIELDS}}


def crossfit(tau, P, y, eid, clip_names):
    """SECONDARY (descriptive, NOT deployable): the PREREG's 2-fold episode cross-fit on eval-141 -- fold = parity of the SORTED-episode index."""
    rank = {c: i for i, c in enumerate(sorted(clip_names))}
    par = np.asarray([rank[clip_names[e]] % 2 for e in eid])
    H = np.zeros_like(tau)
    ws = {}
    for f in (0, 1):
        R = fit_weights(tau[par == f], P[par == f], y[par == f])
        ws[f"fit_on_parity_{f}"] = {"w_along": R["w"][:, 0], "w_cross": R["w"][:, 1], "n_windows": int((par == f).sum()),
                                    "n_episodes": int(len({e for e in eid[par == f]}))}
        sel = par != f
        H[sel] = apply_w(P[sel], tau[sel], R["w"])
    return H, ws


def cmd_score(a):
    import torch
    t0 = time.time()
    ev = load_eval_module()
    from taniteval import ci as CI
    from taniteval import v8f_beta_eval as BE
    from tanitad.data import v8f_mini_data as D
    seed, nb = a.seed, a.n_boot
    grid, J, X = eval_world()
    eid, gt = X["eid"], X["gt"].astype(np.float64)
    clip_names = J["clip"].tolist()
    z = np.load(f"{BANK}/eval_{seed}-kind-own.npz")
    bj = json.load(open(f"{BANK}/eval_{seed}-kind-own.json", encoding="utf-8"))
    tau = z["pred"].astype(np.float64)
    other = [s for s in SEEDS if s != seed][0]
    F_exact = abs(float(z["ade6"].mean()) - float(np.load(f"{BANK}/eval_{other}-kind-own.npz")["ade6"].mean()))
    # ---- P = this checkpoint's OWN frozen P10 (needs no image: 8 kinematic features + the kd_x base, both from the eval kin file)
    model, cfg, ck = load_model(seed)
    with torch.no_grad():
        P32 = model.prior(torch.as_tensor(X["kf8"], dtype=torch.float32), torch.as_tensor(X["base"], dtype=torch.float32)).numpy()
    P = P32.astype(np.float64)
    P_file = np.load(P10_FILE).astype(np.float64)
    fl = np.load(f"{OUT}/inputs/full_eval_floors_perwindow.npz")
    W = json.load(open(f"{OUT}/weights_{seed}.json"))
    w_fit = np.stack([np.asarray(W["w_along_by_step"]), np.asarray(W["w_cross_by_step"])], 1)          # [30, 2]
    ctl = {}
    # ---- identity controls (BEFORE anything else)
    dB = np.linalg.norm(tau - gt, axis=-1)
    ctl["banked_pred_eq_traj30"] = bool(np.array_equal(z["pred"], z["traj30"]))
    ctl["banked_ade6_recomputed_maxabs"] = float(np.abs(dB.mean(1) - z["ade6"]).max())
    ctl["banked_ade2_recomputed_maxabs"] = float(np.abs(dB[:, :10].mean(1) - z["ade2"]).max())
    ctl["banked_fde6_recomputed_maxabs"] = float(np.abs(dB[:, -1] - z["fde6"]).max())
    H1 = apply_w(P, tau, 1.0)
    H0 = apply_w(P, tau, 0.0)
    ctl["w1_vs_banked_pred_maxabs_m"] = float(np.abs(H1 - tau).max())
    ctl["w1_ade6_mean"] = float(np.linalg.norm(H1 - gt, axis=-1).mean())
    ctl["w1_ade6_vs_banked_mean_absdiff"] = abs(ctl["w1_ade6_mean"] - float(z["ade6"].mean()))
    ctl["w1_ade6_perwindow_maxabs_vs_banked"] = float(np.abs(np.linalg.norm(H1 - gt, axis=-1).mean(1) - z["ade6"]).max())
    ctl["w0_ade6_mean"] = float(np.linalg.norm(H0 - gt, axis=-1).mean())
    ctl["w0_ade6_vs_prereg_2.4870_absdiff"] = abs(ctl["w0_ade6_mean"] - 2.4870)
    ctl["w0_ade6_perwindow_maxabs_vs_floors_P10_refit"] = float(np.abs(np.linalg.norm(H0 - gt, axis=-1).mean(1) - fl["P10_refit__ade6"]).max())
    ctl["w0_ade2_perwindow_maxabs_vs_floors_P10_refit"] = float(np.abs(np.linalg.norm(H0 - gt, axis=-1)[:, :10].mean(1) - fl["P10_refit__ade2"]).max())
    ctl["model_prior_vs_P10_4512_file_maxabs_m"] = float(np.abs(P - P_file).max())
    ctl["P10_file_ade6_mean"] = float(np.linalg.norm(P_file - gt, axis=-1).mean())
    ctl["P10_file_ade6_perwindow_maxabs_vs_floors"] = float(np.abs(np.linalg.norm(P_file - gt, axis=-1).mean(1) - fl["P10_refit__ade6"]).max())
    ctl["fit_weights_in_range_and_monotone"] = bool(((w_fit >= 0) & (w_fit <= 1)).all() and (np.diff(w_fit, axis=0) >= -1e-12).all())
    ctl["F_exact_m"] = F_exact
    ctl["F_prereg_printed_m"] = F_PREREG
    ctl["PRIMARY_N_bar_m"] = F_exact / 2
    ctl["all_pass"] = bool(ctl["banked_pred_eq_traj30"] and ctl["w1_vs_banked_pred_maxabs_m"] <= 1e-4 and ctl["w1_ade6_vs_banked_mean_absdiff"] <= 1e-4
                           and ctl["w0_ade6_vs_prereg_2.4870_absdiff"] <= 1e-3 and ctl["model_prior_vs_P10_4512_file_maxabs_m"] <= 1e-4
                           and ctl["fit_weights_in_range_and_monotone"])
    print(f"[score {seed}] identity controls all_pass={ctl['all_pass']}  w1 maxabs {ctl['w1_vs_banked_pred_maxabs_m']:.2e}  "
          f"w0 ADE6 {ctl['w0_ade6_mean']:.4f}  prior-vs-file {ctl['model_prior_vs_P10_4512_file_maxabs_m']:.2e}", flush=True)
    if not ctl["all_pass"]:
        jdump({"seed": seed, "identity_controls": ctl}, f"{OUT}/score_{seed}.json")
        raise SystemExit(f"IDENTITY CONTROL FAILED -- refusing to read: {json.dumps(ctl)}")
    # ---- the arm
    H = apply_w(P, tau, w_fit)
    pwB, pwH, pwP = (arm_pw(p, gt, BE) for p in (tau, H, P_file))
    out = {"seed": seed, "ckpt_sha256": SHA[seed], "tier": "T1-class open loop (own held outputs; EVAL_DOCTRINE tier T0/T1 stamp the source eval carries)",
           "estimator": f"episode-cluster bootstrap, B={nb}, 141 episodes, seed 0; deltas PAIRED (same resampled episodes). Answers 'another draw of EPISODES' ONLY; "
                        "training-seed variance only via F = |ADE6(s0)-ADE6(s1)|; inference is deterministic (no inference-seed floor).",
           "n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))), "identity_controls": ctl,
           "weights_file": f"weights_{seed}.json", "weights_sha256": sha256_file(f"{OUT}/weights_{seed}.json"),
           "route": "BANKED predictions: tau = B's banked `pred` (mirror raw/2026-10-25-kind/run/eval, KIND run own code); P = the checkpoint's own frozen P10 "
                    "evaluated on the eval kin features (no image needed), cross-checked to the P10-yaw-4512 file; GT from the eval context (eval's own).",
           "inputs_sha256": {"banked_npz": sha256_file(f"{BANK}/eval_{seed}-kind-own.npz"), "banked_json": sha256_file(f"{BANK}/eval_{seed}-kind-own.json"),
                             "p10_4512_pred": sha256_file(P10_FILE), "ckpt": SHA[seed]}}
    keys = list(pwB)
    out["single"] = {name: {k: single(pw[k], eid, CI, nb) for k in keys} for name, pw in (("B", pwB), ("H", pwH), ("P10", pwP))}
    out["paired_H_minus_B"] = rows_for(pwH, pwB, eid, BE, nb)
    out["paired_H_minus_P10"] = rows_for(pwH, pwP, eid, BE, nb)
    out["paired_B_minus_P10"] = {k: prow(pwB[k], pwP[k], eid, BE, nb, k) for k in ("ade2", "ade6", "fw_lon_accel_mae")}
    # ---- full eval blocks via the eval's OWN `evaluate` (families, consistency, tactical, self-consistency): H, P10, and the w=1 control
    ex = extras_of(z, D)
    resH, pwH_ev = ev.evaluate(H, X, ex, nb, with_families=not a.no_families)
    resP, _ = ev.evaluate(P_file, X, None, nb, with_families=not a.no_families)
    out["blocks"] = {"H": resH, "P10": resP}
    if not a.no_families:
        res1, _ = ev.evaluate(H1, X, ex, nb, with_families=True)
        bd = deep_diff({k: v for k, v in res1.items()}, {k: bj["raw"][k] for k in res1 if k in bj["raw"]}, tol=1e-6)
        out["w1_full_block_vs_banked_json"] = {"n_mismatches": len(bd), "mismatches": bd[:20], "compared_keys": sorted(set(res1) & set(bj["raw"]))}
        out["accel_mae_control"] = {"B_mean_fw_vs_banked_family_json": [float(np.nanmean(pwB["fw_lon_accel_mae"])),
                                                                          bj["raw"]["families"]["longitudinal"]["accel_mae_mps2"]],
                                    "P10_mean_fw_vs_PREREG_0.638": [float(np.nanmean(pwP["fw_lon_accel_mae"])), 0.638]}
    # STRUCTURAL IDENTITY: T's decision heads and S's checkpoints are untouched => the tactical block (macro-F1 of T's heads) and the strategic block
    # (S's checkpoint error) are B's, bit for bit. (families.tactical_from_trajectory is read FROM THE PLAN and is NOT covered by it.)
    out["structural_identity"] = {
        "tactical_block_T_heads": {"H_equals_banked_B": json.dumps(resH["tactical"], sort_keys=True, default=float) == json.dumps(bj["raw"]["tactical"], sort_keys=True, default=float),
                                   "note": "lat/lon macro-F1 and goal macro-F1 of T's own heads (logits unchanged)"},
        "strategic_block_S_checkpoints": {"H_equals_banked_B": True, "by_construction": "H never touches S; the banked ck_pos_err per window is carried unchanged",
                                          "banked_B_block": bj["raw"]["strategic"],
                                          "status": "PAUSED per PI ruling PI_DECISION_QUEUE ITEM 25 (2026-09-19): strategic layer switched off temporarily; reported as PAUSED, not a pass"},
        "families_tactical_from_trajectory_moves_with_H": ("families" in resH and "tactical_from_trajectory" in resH["families"])}
    # ---- SECONDARY: 2-fold episode cross-fit on eval-141 (descriptive, NOT deployable)
    Hcf, wcf = crossfit(tau, P, gt, eid, clip_names)
    pwCF = arm_pw(Hcf, gt, BE)
    out["crossfit_secondary"] = {"label": "CROSS-FITTED on eval-141 (fold = parity of the sorted-episode index); descriptive, not a deployable weight",
                                 "weights": wcf, "single": {k: single(pwCF[k], eid, CI, nb) for k in ("ade2", "ade6", "fw_lon_accel_mae")},
                                 "paired_vs_B": {k: prow(pwCF[k], pwB[k], eid, BE, nb, k) for k in ("ade1", "ade2", "ade3", "ade4", "ade5", "ade6", "fw_lon_accel_mae")}}
    # ---- diagnostics (declared as such, never a bar): eval-oracle in-sample weights = the ceiling of the lever; a shuffled-residual fit = noise control
    Ro = fit_weights(tau, P, gt)
    pwO = arm_pw(apply_w(P, tau, Ro["w"]), gt, BE)
    out["diag_eval_oracle_in_sample"] = {"label": "T0 DIAGNOSTIC: w fitted ON eval-141 itself (reads GT) -- the ceiling of this lever, never a result",
                                         "w_along": Ro["w"][:, 0], "w_cross": Ro["w"][:, 1],
                                         "single": {k: single(pwO[k], eid, CI, nb) for k in ("ade2", "ade6", "fw_lon_accel_mae")},
                                         "paired_vs_B": {k: prow(pwO[k], pwB[k], eid, BE, nb, k) for k in ("ade2", "ade6", "fw_lon_accel_mae")}}
    fw = np.load(f"{OUT}/fit_windows_{seed}.npz")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(fw["p30"]))
    r_tr = fw["traj30"].astype(np.float64) - fw["p30"].astype(np.float64)
    Rs = fit_weights(fw["p30"].astype(np.float64) + r_tr[perm], fw["p30"], fw["gt"])
    pwS = arm_pw(apply_w(P, tau, Rs["w"]), gt, BE)
    out["diag_shuffled_residual_fit"] = {"label": "NOISE CONTROL: w fitted on train windows with the residual permuted across windows (seed 0): must NOT manufacture a gain",
                                         "w_along": Rs["w"][:, 0], "w_cross": Rs["w"][:, 1],
                                         "single": {k: single(pwS[k], eid, CI, nb) for k in ("ade2", "ade6")},
                                         "paired_vs_B": {k: prow(pwS[k], pwB[k], eid, BE, nb, k) for k in ("ade2", "ade6")}}
    out["seconds"] = time.time() - t0
    jdump(out, f"{OUT}/score_{seed}.json")
    np.savez_compressed(f"{OUT}/score_{seed}_perwindow.npz", H=H, P=P, tau=tau, P_file=P_file, H_crossfit=Hcf, eid=eid,
                        **{f"H__{k}": v for k, v in pwH.items()}, **{f"B__{k}": v for k, v in pwB.items()}, **{f"P10__{k}": v for k, v in pwP.items()})
    pH = out["paired_H_minus_B"]
    print(f"[score {seed}] DONE {time.time() - t0:.0f}s  ADE2 H-B {pH['ade2']['mean']:+.4f} [{pH['ade2']['lo']:+.4f},{pH['ade2']['hi']:+.4f}]  "
          f"ADE6 H-B {pH['ade6']['mean']:+.4f} [{pH['ade6']['lo']:+.4f},{pH['ade6']['hi']:+.4f}]  accel H-B {pH['fw_lon_accel_mae']['mean']:+.4f}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ read: the outcome table, verbatim
def cmd_read(a):
    S = {s: json.load(open(f"{OUT}/score_{s}.json")) for s in SEEDS}
    for s in SEEDS:
        if not S[s]["identity_controls"]["all_pass"]:
            raise SystemExit(f"{s}: identity controls did not pass")
    F_exact = S["M2b_s0"]["identity_controls"]["F_exact_m"]
    bar_n = F_exact / 2
    rows = {}
    for s in SEEDS:
        d = S[s]["paired_H_minus_B"]
        a2, a6, ac = d["ade2"], d["ade6"], d["fw_lon_accel_mae"]
        dom_pt = {"ade2": S[s]["single"]["H"]["ade2"]["mean"] <= S[s]["single"]["P10"]["ade2"]["mean"],
                  "ade6": S[s]["single"]["H"]["ade6"]["mean"] <= S[s]["single"]["P10"]["ade6"]["mean"],
                  "accel": S[s]["single"]["H"]["fw_lon_accel_mae"]["mean"] <= S[s]["single"]["P10"]["fw_lon_accel_mae"]["mean"]}
        dp6 = S[s]["paired_H_minus_P10"]["ade6"]
        rows[s] = {"PRIMARY_S": {"delta": a2["mean"], "lo": a2["lo"], "hi": a2["hi"], "separated": a2["separated"], "pass": bool(a2["mean"] < 0 and a2["separated"])},
                   "PRIMARY_N": {"delta": a6["mean"], "lo": a6["lo"], "hi": a6["hi"], "bar_hi_le": bar_n, "pass": bool(a6["hi"] <= bar_n),
                                 "within_0.0003_of_bar": bool(abs(a6["hi"] - bar_n) < 3e-4),
                                 "hi_le_prereg_printed_0.060": bool(a6["hi"] <= 0.060)},
                   "SECONDARY_A": {"delta": ac["mean"], "lo": ac["lo"], "hi": ac["hi"], "separated": ac["separated"], "pass": bool(ac["mean"] < 0 and ac["separated"])},
                   "DOMINANCE_vs_P10": {"point_estimates_le_P10": dom_pt, "ade6_H_minus_P10": {k: dp6[k] for k in ("mean", "lo", "hi", "separated")},
                                        "H_dominates_P10": bool(all(dom_pt.values()) and dp6["separated"])}}
    s_ok = all(rows[s]["PRIMARY_S"]["pass"] for s in SEEDS)
    n_ok = all(rows[s]["PRIMARY_N"]["pass"] for s in SEEDS)
    if s_ok and n_ok:
        outcome, nxt = "PASS", ("H becomes the deployed v8F output. The shrinkage weights become a model buffer in the next trained arm, and the FLAT and MIXO reads are "
                                "re-scored with H applied to each arm's own fitted weights (descriptive).")
    elif s_ok:
        outcome, nxt = "SHORT ONLY", ("the residual's 6 s signal and its short-horizon error are entangled. Next: a TRAINED horizon-weighted loss arm under its own PREREG (box 2).")
    else:
        outcome, nxt = "NO GAIN", ("the short-horizon loss is not residual noise that shrinkage can remove. Next: a diagnosis of the short-horizon error by component "
                                   "(P10's own error vs the residual vs the speed code), before any training arm.")
    R = {"outcome": outcome, "committed_next_step_verbatim": nxt, "PRIMARY_S_both_seeds": s_ok, "PRIMARY_N_both_seeds": n_ok,
         "SECONDARY_A_both_seeds": all(rows[s]["SECONDARY_A"]["pass"] for s in SEEDS),
         "H_dominates_P10_both_seeds": all(rows[s]["DOMINANCE_vs_P10"]["H_dominates_P10"] for s in SEEDS),
         "F_exact_m": F_exact, "PRIMARY_N_bar_m": bar_n, "per_seed": rows,
         "estimator": S["M2b_s0"]["estimator"], "tier": S["M2b_s0"]["tier"]}
    jdump(R, f"{OUT}/READ.json")
    print(json.dumps({k: R[k] for k in ("outcome", "PRIMARY_S_both_seeds", "PRIMARY_N_both_seeds", "SECONDARY_A_both_seeds", "H_dominates_P10_both_seeds")}, indent=1))
    for s in SEEDS:
        print(s, json.dumps({k: rows[s][k] for k in ("PRIMARY_S", "PRIMARY_N", "SECONDARY_A")}, default=float))
    return 0


def register_score_read(sub):
    p = sub.add_parser("score")
    p.add_argument("--seed", required=True, choices=SEEDS)
    p.add_argument("--n-boot", type=int, default=B_BOOT)
    p.add_argument("--no-families", action="store_true", help="smoke only")
    p = sub.add_parser("read")
    p.add_argument("--n-boot", type=int, default=B_BOOT)
