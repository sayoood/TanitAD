r"""rb_trained.py -- the r-b read of PREREG_V8F_PROOF_ADD12 (stage 0 definitions, S0_DEFS R8 / R19) on the TRAINED model of a stage-1 arm:
the lead gap at +2.0 s read from (i) the state rolled out by the arm's own level-1 predictor, (ii) the present state (copy), (iii) the true
future state (ceiling), against the ego-only floors (iv) and (iv-b).  Card question Q2 (stage 1) and question C1 of PREREG_V8F_PROOF_ADD16.
Tier T0: a world-model diagnostic on logged frames, never driving performance.  CPU, zero training of the model, no model pass here.

  python rb_trained.py run  --arm G_s0 --states <ARM>_eval.npz --latents LATENTS_GW_<ARM>.npz --out-dir DIR
  python rb_trained.py pair --a DIR/OOF_RB_<A>.npz --b DIR/OOF_RB_<B>.npz --out FILE.md

Inputs: the eval bank of stage 0 (res/cache/frames.npz: episode, grid step, ego vector, lead gap), the level-1 states of the arm on the eval
clips (extract_states.py, one state per frame), and the per-window latents the pixel read saves (z0 = state at the window start, pred_2s = the
learned rollout HJEPAStack.rollout over 2.0 s under the logged actions, true_2s = the state of the frame 2.0 s later).
Every probe, fold, label and bootstrap function is imported unchanged from spatial/s0/s0_read.py and res/resprobe.py.

Reading choices (each the most literal reading of the card and of S0_DEFS; none moves a threshold, a bar or an outcome rule):
  RB-1  Probe = the "bank" probe of S0_DEFS R19: for each fold, the gap probe fitted on the states of the training-fold bank frames with a
        finite lead gap, applied unchanged to (i), (ii), (iii) of the test-fold windows.
  RB-2  Windows = the windows of the latents file that have a bank row at the start and 2.0 s later; "a lead at both times" as in stage 0.
  RB-3  Ego floors as in stage 0: (iv) a probe on the start ego vector plus the window's actions, fitted on training-fold windows;
        (iv-b) the bank ego probe applied to the ego vector of the frame 2.0 s later.  The HIGHER of the two decides (note N4).
        The actions are the scaled actions saved with the latents (a fixed per-channel rescale of the logged ones).
  RB-4  Outcome per arm (card, Q2): IMAGINE-LEAD if (i) minus (ii) AND (i) minus the floor are above 0 with the interval excluding 0;
        IMAGINE-EGO if only (i) minus (ii) is; else IMAGINE-FLAT.  The card's name needs both seeds of the arm.
Controls (first in the verdict):
  t1 every bank row has a state of the arm; t2 the ego-only gap floor on the bank reproduces 0.133 within 0.03;
  t3 labels permuted within folds: every r-b R2 at or below 0.02; t4 the ceiling (iii) is at or above (i) and (ii);
  t5 the states of the two sources agree: the probe read from true_2s and from the extracted state of the same frame correlate at or
     above 0.999, and z0 against the extracted state of the start frame likewise;
  t6 known values: the copy put in the place of the rollout reads a difference of exactly 0 against the copy, and the true future put in
     the place of the rollout reads exactly (iii) minus (ii).
Exit: 0 valid, 2 a control failed, 3 refused.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

ROOT = "F:/TanitAD-scratch/v8proof"
for _p in (ROOT + "/spatial/s0",):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import s0_read as S  # noqa: E402

RP = S.RP
TIER = "T0 (a world-model diagnostic on logged frames, never driving performance)"
FAMILIES = "LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)"
KS = S.KSTEP


def load(states_path, latents_path, expect_dim):
    Z = np.load(S.BANK)
    beid = Z["beid"].astype(str)
    bg = Z["bg"].astype(np.int64)
    E = Z["E"].astype(np.float64)
    okE = np.isfinite(E).all(1)
    st = np.load(states_path, allow_pickle=True)
    clips = st["clips"].astype(str)
    key = {(clips[int(c)], int(t)): i for i, (c, t) in enumerate(zip(st["clip_idx"], st["t"]))}
    idx = np.array([key.get((e, int(g)), -1) for e, g in zip(beid, bg)])
    missing = int((okE & (idx < 0)).sum())
    rows = np.flatnonzero(okE & (idx >= 0))
    beid, bg, E = beid[rows], bg[rows], E[rows]
    Sb = st["Z"][idx[rows]].astype(np.float64)
    if Sb.shape[1] != expect_dim:
        raise SystemExit("REFUSED: state width %d, expected %d" % (Sb.shape[1], expect_dim))
    gap = Z["lead_gap"][rows].astype(np.float64)
    uep = sorted(set(beid))
    uidx = {e: i for i, e in enumerate(uep)}
    epi = np.array([uidx[e] for e in beid])
    pos = {(e, int(g)): i for i, (e, g) in enumerate(zip(beid, bg))}
    L = np.load(latents_path, allow_pickle=True)
    w_eid = L["eid"].astype(str)
    g0 = L["g0"].astype(np.int64)
    ri0 = np.array([pos.get((e, int(g)), -1) for e, g in zip(w_eid, g0)])
    rik = np.array([pos.get((e, int(g) + KS), -1) for e, g in zip(w_eid, g0)])
    A = np.asarray(L["act_scaled"], np.float64)
    if A.ndim == 3:
        A = A[:, :KS, :]
    A = A.reshape(len(w_eid), -1)
    lat = {k: np.asarray(L[k], np.float64) for k in ("z0", "pred_2s", "true_2s")}
    ok = (ri0 >= 0) & (rik >= 0) & np.isfinite(A).all(1) & np.all([np.isfinite(v).all(1) for v in lat.values()], axis=0)
    cnt = {"bank_rows": int(len(rows)), "bank_rows_without_state": missing, "episodes": len(uep), "windows_in_latents": int(len(w_eid)),
           "windows_used": int(ok.sum()), "windows_dropped": int((~ok).sum()), "state_width": int(Sb.shape[1])}
    sel = np.flatnonzero(ok)
    D = {"beid": beid, "bg": bg, "E": E, "Sb": Sb, "gap": gap, "uep": uep, "epi": epi, "w_eid": w_eid[sel], "w_g0": g0[sel], "ri0": ri0[sel], "rik": rik[sel],
         "A": A[sel], "z0": lat["z0"][sel], "pred": lat["pred_2s"][sel], "true": lat["true_2s"][sel], "counts": cnt,
         "w_fold_file": np.asarray(L["w_fold"])[sel] if "w_fold" in L.files else None}
    return D


def run_folds(D, log):
    N, nw = len(D["Sb"]), len(D["ri0"])
    folds_b = RP.make_folds(D["beid"])
    fold_of = {e: int(f) for e, f in zip(D["beid"], folds_b)}
    folds_w = np.array([fold_of[e] for e in D["w_eid"]])
    epi = D["epi"]
    w_epi = np.array([D["uep"].index(e) for e in D["w_eid"]])
    E, A, ri0, rik, Sb = D["E"], D["A"], D["ri0"], D["rik"], D["Sb"]
    gk = D["gap"][rik]
    both = np.isfinite(gk) & np.isfinite(D["gap"][ri0])
    gk_b = np.where(both, gk, np.nan)
    labs = {"main": {"gap": D["gap"], "gapk": gk_b},
            "perm": {"gap": S.permute_within_fold(D["gap"], folds_b, 1), "gapk": S.permute_within_fold(gk_b, folds_w, 3)}}
    out = {}
    for lab in labs:
        for nm in ("i", "ii", "iii", "iiib", "iib", "iv", "ivb"):
            out[lab + "|" + nm] = np.full(nw, np.nan)
    Fe = np.concatenate([E[ri0], A], 1)
    t0 = time.time()
    for f in range(5):
        tr = np.flatnonzero(folds_b != f)
        wtr = np.flatnonzero(folds_w != f)
        wte = np.flatnonzero(folds_w == f)
        for lab, Lb in labs.items():
            y, yk = Lb["gap"], Lb["gapk"]
            vtr = tr[np.isfinite(y[tr])]
            wv_tr = wtr[np.isfinite(yk[wtr])]
            wv_te = wte[np.isfinite(yk[wte])]
            if len(vtr) >= 12 and len(wv_te):
                fit = S.fit_probe(Sb[vtr], y[vtr], epi[vtr], S.SEED + f)
                out[lab + "|i"][wv_te] = fit(D["pred"][wv_te])
                out[lab + "|ii"][wv_te] = fit(D["z0"][wv_te])
                out[lab + "|iii"][wv_te] = fit(D["true"][wv_te])
                out[lab + "|iiib"][wv_te] = fit(Sb[rik[wv_te]])
                out[lab + "|iib"][wv_te] = fit(Sb[ri0[wv_te]])
                fit_e = S.fit_probe(E[vtr], y[vtr], epi[vtr], S.SEED + f)
                out[lab + "|ivb"][wv_te] = fit_e(E[rik[wv_te]])
            if len(wv_tr) >= 12 and len(wv_te):
                fit = S.fit_probe(Fe[wv_tr], yk[wv_tr], w_epi[wv_tr], S.SEED + f)
                out[lab + "|iv"][wv_te] = fit(Fe[wv_te])
            log("fold %d label set %s done (%.0f s)" % (f, lab, time.time() - t0))
    return out, labs, both, folds_b, folds_w, w_epi


def _corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def analyse(D, out, labs, both, folds_b, w_epi, B):
    n_ep = len(D["uep"])
    C = RP.boot_counts(n_ep, B, S.SEED)
    yk = labs["main"]["gapk"]
    res = {"rb": {}, "perm": {}}
    for nm in ("i", "ii", "iii", "iv", "ivb"):
        res["rb"][nm] = S.single_r2(w_epi, yk, out["main|" + nm], C, n_ep)
        res["perm"][nm] = S.single_r2(w_epi, labs["perm"]["gapk"], out["perm|" + nm], C, n_ep)
    fl = "ivb" if res["rb"]["ivb"]["r2"] > res["rb"]["iv"]["r2"] else "iv"
    res["floor_used"] = fl
    P = lambda a, b: S.paired(w_epi, yk, out["main|" + a], out["main|" + b], C, n_ep)   # noqa: E731
    res["i_minus_ii"] = P("i", "ii")
    res["i_minus_floor"] = P("i", fl)
    res["ii_minus_floor"] = P("ii", fl)
    res["iii_minus_i"] = P("iii", "i")
    res["iii_minus_ii"] = P("iii", "ii")
    a, b = res["i_minus_ii"]["separated_pos"], res["i_minus_floor"]["separated_pos"]
    res["outcome_this_seed"] = "IMAGINE-LEAD" if (a and b) else ("IMAGINE-EGO" if a else "IMAGINE-FLAT")
    # controls
    e_pred, e_chosen = S.e_floor_ra(D["E"], D["gap"], np.isfinite(D["gap"]), D["epi"], folds_b, False)
    e_gap = RP.r2_pooled(D["gap"], e_pred)
    ctr = {"t1_bank_rows_have_states": {"pass": bool(D["counts"]["bank_rows_without_state"] == 0), **D["counts"]},
           "t2_ego_floor_bank": {"pass": bool(abs(e_gap - S.FULLRES_E_GAP) <= 0.03), "ego_gap_r2": float(e_gap), "reference": S.FULLRES_E_GAP, "tol": 0.03, "chosen": e_chosen},
           "t3_permuted_labels": {"pass": bool(all(res["perm"][k]["r2"] <= 0.02 for k in ("i", "ii", "iii", "iv", "ivb"))), "bar": 0.02,
                                  "values": {k: res["perm"][k]["r2"] for k in res["perm"]}},
           "t4_ceiling": {"pass": bool(res["rb"]["iii"]["r2"] >= res["rb"]["i"]["r2"] and res["rb"]["iii"]["r2"] >= res["rb"]["ii"]["r2"]),
                          "r2_i": res["rb"]["i"]["r2"], "r2_ii": res["rb"]["ii"]["r2"], "r2_iii": res["rb"]["iii"]["r2"]}}
    c_fut, c_now = _corr(out["main|iii"], out["main|iiib"]), _corr(out["main|ii"], out["main|iib"])
    ctr["t5_state_sources_agree"] = {"pass": bool(c_fut >= 0.999 and c_now >= 0.999), "corr_future": c_fut, "corr_start": c_now, "bar": 0.999,
                                     "max_abs_state_diff_future": float(np.abs(D["true"] - D["Sb"][D["rik"]]).max()),
                                     "max_abs_state_diff_start": float(np.abs(D["z0"] - D["Sb"][D["ri0"]]).max())}
    k_copy = S.paired(w_epi, yk, out["main|ii"], out["main|ii"], C, n_ep)
    k_true = S.paired(w_epi, yk, out["main|iii"], out["main|ii"], C, n_ep)
    ctr["t6_known_values"] = {"pass": bool(k_copy["diff"] == 0.0 and abs(k_true["diff"] - (res["rb"]["iii"]["r2"] - res["rb"]["ii"]["r2"])) < 1e-12),
                              "copy_as_rollout_diff": k_copy["diff"], "true_as_rollout_diff": k_true["diff"]}
    if D["w_fold_file"] is not None:
        fw = np.array([int(x) for x in D["w_fold_file"]])
        fold_of = {e: int(f) for e, f in zip(D["beid"], folds_b)}
        res["fold_map_equals_latents_file"] = bool(all(fold_of[e] == f for e, f in zip(D["w_eid"], fw)))
    res["controls"] = ctr
    res["controls_pass"] = bool(all(v["pass"] for v in ctr.values()))
    res["n_lead_both_windows"] = int(np.isfinite(yk).sum())
    return res


def _ci(c):
    return "[%+.3f, %+.3f]" % (c[0], c[1])


def write_verdict(path, arm, res, meta):
    r = res["rb"]
    L = ["# VERDICT r-b on the trained model: " + arm, "", "Tier " + TIER + ". Card PREREG_V8F_PROOF_ADD12.md (Q2, stage 1), definitions of stage 0 (S0_DEFS R8, R19).",
         FAMILIES + ".", "", "## Controls (first)", ""]
    for k, v in res["controls"].items():
        L.append("- %s: %s :: %s" % (k, "PASS" if v["pass"] else "FAIL", json.dumps({a: b for a, b in v.items() if a != "pass"})))
    L += ["- controls_pass: %s" % res["controls_pass"], "",
          "## Lead gap at +2.0 s (windows with a lead at both times, n %d of %d windows, %d episodes; R2 with episode-cluster bootstrap, B %d)" % (
              res["n_lead_both_windows"], meta["counts"]["windows_used"], meta["counts"]["episodes"], meta["B"]), "",
          "- (i) rolled-out state: %+.3f %s" % (r["i"]["r2"], _ci(r["i"]["ci95"])),
          "- (ii) present state (copy): %+.3f %s" % (r["ii"]["r2"], _ci(r["ii"]["ci95"])),
          "- (iii) true future state (ceiling): %+.3f %s" % (r["iii"]["r2"], _ci(r["iii"]["ci95"])),
          "- (iv) ego only, start ego vector plus actions: %+.3f %s" % (r["iv"]["r2"], _ci(r["iv"]["ci95"])),
          "- (iv-b) ego only, bank probe on the future ego vector: %+.3f %s; floor used: %s" % (r["ivb"]["r2"], _ci(r["ivb"]["ci95"]), res["floor_used"]), "",
          "Paired over episodes:"]
    for k, t in (("i_minus_ii", "(i) minus (ii), rollout against copy"), ("i_minus_floor", "(i) minus the ego floor"), ("ii_minus_floor", "(ii) minus the ego floor"),
                 ("iii_minus_i", "(iii) minus (i), headroom of the rollout"), ("iii_minus_ii", "(iii) minus (ii), what the future state adds over the copy")):
        d = res[k]
        L.append("- %s: %+.3f %s%s" % (t, d["diff"], _ci(d["ci95"]), " (separated above 0)" if d["separated_pos"] else (" (separated below 0)" if d["separated_neg"] else "")))
    L += ["", "**Outcome at this seed (the card's name needs both seeds): %s**" % res["outcome_this_seed"], "",
          "Limits: one training run; the probe is fitted on bank frames of the training folds (139 episodes); a separated interval is necessary and not sufficient.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


def cmd_run(a):
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    logp = os.path.join(a.out_dir, "run_rb_%s.log" % a.arm)

    def log(*x):
        s = "[rb %s] " % time.strftime("%H:%M:%S") + " ".join(str(v) for v in x)
        print(s, flush=True)
        with open(logp, "a", encoding="utf-8", newline="\n") as f:
            f.write(s + "\n")
    D = load(a.states, a.latents, a.expect_dim)
    log("counts", json.dumps(D["counts"]))
    out, labs, both, folds_b, folds_w, w_epi = run_folds(D, log)
    res = analyse(D, out, labs, both, folds_b, w_epi, a.n_boot)
    meta = {"arm": a.arm, "states": a.states, "latents": a.latents, "tier": TIER, "families": FAMILIES, "B": a.n_boot, "counts": D["counts"],
            "estimator": "episode-cluster bootstrap, paired for differences (resprobe, unchanged)", "elapsed_s": round(time.time() - t0, 1)}
    res["meta"] = meta
    with open(os.path.join(a.out_dir, "RESULT_RB_%s.json" % a.arm), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez(os.path.join(a.out_dir, "OOF_RB_%s.npz" % a.arm), w_eid=D["w_eid"], w_g0=D["w_g0"], yk=labs["main"]["gapk"], w_epi=w_epi,
             uep=np.array(D["uep"]), **{"p_" + k.split("|")[1]: v for k, v in out.items() if k.startswith("main|")})
    write_verdict(os.path.join(a.out_dir, "VERDICT_RB_%s.md" % a.arm), a.arm, res, meta)
    log("outcome", res["outcome_this_seed"], "controls", "PASS" if res["controls_pass"] else "FAILED", "elapsed %.0f s" % (time.time() - t0))
    return 0 if res["controls_pass"] else 2


def cmd_pair(a):
    A, Bz = np.load(a.a, allow_pickle=True), np.load(a.b, allow_pickle=True)
    ka = {(str(e), int(g)): i for i, (e, g) in enumerate(zip(A["w_eid"], A["w_g0"]))}
    ia, ib = [], []
    for j, (e, g) in enumerate(zip(Bz["w_eid"], Bz["w_g0"])):
        i = ka.get((str(e), int(g)), -1)
        if i >= 0:
            ia.append(i)
            ib.append(j)
    ia, ib = np.array(ia), np.array(ib)
    yk = A["yk"][ia]
    if not np.allclose(np.nan_to_num(yk, nan=-1.0), np.nan_to_num(Bz["yk"][ib], nan=-1.0)):
        print("REFUSED: labels differ on the common windows")
        return 3
    uep = sorted(set(str(e) for e in A["w_eid"][ia]))
    uidx = {e: i for i, e in enumerate(uep)}
    w_epi = np.array([uidx[str(e)] for e in A["w_eid"][ia]])
    C = RP.boot_counts(len(uep), a.n_boot, S.SEED)
    na, nb = os.path.basename(a.a)[7:-4], os.path.basename(a.b)[7:-4]
    L = ["# PAIR r-b: %s minus %s" % (na, nb), "", "Tier " + TIER + ". Paired over episodes on the %d common windows (%d with a lead at both times), B %d." % (
        len(ia), int(np.isfinite(yk).sum()), a.n_boot), FAMILIES + ".", ""]
    res = {}
    for nm, t in (("i", "(i) rolled-out state"), ("ii", "(ii) present state"), ("iii", "(iii) true future state")):
        d = S.paired(w_epi, yk, A["p_" + nm][ia], Bz["p_" + nm][ib], C, len(uep))
        res[nm] = d
        L.append("- %s, %s minus %s: %+.3f [%+.3f, %+.3f]%s" % (t, na, nb, d["diff"], d["ci95"][0], d["ci95"][1],
                 " (separated above 0)" if d["separated_pos"] else (" (separated below 0)" if d["separated_neg"] else "")))
    L.append("")
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    with open(a.out[:-3] + ".json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\n".join(L))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--states", required=True)
    r.add_argument("--latents", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--expect-dim", type=int, default=640)
    r.add_argument("--n-boot", type=int, default=S.B_BOOT)
    p = sub.add_parser("pair")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=S.B_BOOT)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_pair(a)


if __name__ == "__main__":
    sys.exit(main())
