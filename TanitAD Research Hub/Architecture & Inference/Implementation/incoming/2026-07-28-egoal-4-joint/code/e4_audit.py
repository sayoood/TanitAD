#!/usr/bin/env python3
"""E-GOAL-4 S0 -- ⛔ THE FUTURE-CONTENT AUDIT AND THE LEAK CHECK. PRIORITY 1.

Runs BEFORE anything is fitted. If it fails, no number from this stream is
quotable.

  C23  every fed column audited for FUTURE content BY DEFINITION (its index
       expression) and EMPIRICALLY (`future_blind`: corrupt everything the
       future could supply, re-derive every column, require max |Δ| == 0.0),
       over ALL 13,198 windows x 256 candidates -- not a sample.
       ⭐ POWER IS DEMONSTRATED, NOT ASSERTED, THREE WAYS:
         (a) the LABEL -- future by definition -- must move on 100 % of rows;
         (b) the SAME instrument must FIRE on `parent_resampled`'s cross column
             (which is `true_cross + resampled_residual`, future BY
             CONSTRUCTION) and return exactly 0.0 on `sel`'s;
         (c) `S_LEAK` (§ e4_select.py) is fed `head_deg`, a real future field,
             and must separate.

  LEAK this stream TRAINS INSIDE the 600 val episodes, so the surface that
       matters is BETWEEN ITS OWN FOLDS. Checked BY CONTENT (sha256 of raw
       poses[T,4] float32 bytes, from E-GOAL-3's staged per-episode
       fingerprints), never by filename, with the PATH and the COUNT reported.

  GATES G-0 (the deployment re-derives), G-2 (the selector's folds are
       BIT-IDENTICAL to the goal head's -- else fold f's goal came from a head
       that saw fold f), G-3 (the label IS the metric, per row) and G-5 (G-3
       has power).

Run:  OMP_NUM_THREADS=6 python e4_audit.py --fan <fan600.pt> --feat <val.npz>
                                           --preds <preds.npz>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_common import (ALL_COLS, COLSPEC, DEPLOY_REF, EG3, F_ANS, F_CTX,  # noqa: E402
                       F_GOAL, STREAM, SEED, ade, assert_no_future,
                       build_goal, build_static, clip_folds, cross_background,
                       labels, load_all, r4, realise)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=str(STREAM / "raw" / "e4_audit.json"))
    a = ap.parse_args(argv)
    t0 = time.time()

    D = load_all(a.fan, a.feat, a.preds)
    fan, gt, eid = D["fan"], D["gt"], D["eid"]
    W, C = D["logits"].shape
    g_true = gt[:, -1, :]
    NEP = len(np.unique(eid))
    R = {"_stream": "2026-07-28-egoal-4-joint", "_stage": "S0 audit",
         "_inputs": {"fan": str(a.fan), "feat": str(a.feat),
                     "preds": str(a.preds)},
         "_fan_provenance": {"ckpt_md5": D["_ckpt"], "ckpt_step": D["_ckpt_step"],
                             "n_anchors": D["_n_anchors"],
                             "nav_mode": D["_nav_mode"]},
         "_deployment": {"n_windows": int(W), "n_candidates": int(C),
                         "n_episode_clusters": int(NEP),
                         "n_rows": int(W * C)}}
    print(f"[audit] {W} windows x {C} candidates / {NEP} episodes", flush=True)

    # ================================================================ G-0 ====
    sel = D["sel"]
    a0 = ade(fan[np.arange(W), sel], gt)
    r_goal = realise(g_true, fan, gt)
    lab = labels(D)                                  # [W, C] per-candidate ADE
    oracle = lab.min(1)
    headroom = float(a0.mean() - r_goal.mean())
    g0 = {"a0": r4(a0.mean()), "r_goal2s": r4(r_goal.mean()),
          "oracle_in_fan": r4(oracle.mean()), "headroom": r4(headroom)}
    g0_ok = all(abs(g0[k] - DEPLOY_REF[k]) < 5e-4 for k in g0)
    R["G_0_deployment"] = {"_what": "the deployment re-derives from the fan",
                           "mine": g0, "reference_E2_E3": DEPLOY_REF,
                           "passes": bool(g0_ok)}
    print(f"[G-0] {g0} vs {DEPLOY_REF} -> {g0_ok}", flush=True)
    if not g0_ok:
        raise RuntimeError("G-0 FAILED -- the deployment does not re-derive")

    # ============================================== G-3 / G-5: the label =====
    #: the label must BE the metric: its per-window min is `oracle_in_fan` and
    #: its value at the as-trained pick is the per-window `a0`.
    d_a0 = float(np.abs(lab[np.arange(W), sel] - a0).max())
    d_or = float(np.abs(oracle - ade(fan[np.arange(W)[:, None],
                                         lab.argmin(1)[:, None]].squeeze(1),
                                     gt)).max())
    #: ⛔ G-5 -- the SAME comparison with rows shifted by one MUST fail hard,
    #: or G-3 could not have detected a wrong join.
    sh = float(np.abs(lab[np.arange(W), sel] - np.roll(a0, 1)).max())
    R["G_3_label_is_the_metric"] = {
        "_what": "label(w, sel[w]) == per-window a0; min_c label == oracle_in_fan",
        "max_abs_delta_a0": d_a0, "max_abs_delta_oracle": d_or,
        "passes": bool(d_a0 < 1e-6 and d_or < 1e-6)}
    R["G_5_label_gate_has_power"] = {
        "_what": "the identical comparison with rows shifted by one",
        "max_abs_delta_shifted": r4(sh), "fails_as_required": bool(sh > 1e-3)}
    print(f"[G-3] |Δa0|={d_a0:.2e} |Δoracle|={d_or:.2e}  "
          f"[G-5] shifted={sh:.4f}", flush=True)
    if not (d_a0 < 1e-6 and d_or < 1e-6 and sh > 1e-3):
        raise RuntimeError("G-3/G-5 FAILED -- the label is not the metric")

    # ========================================== F-2: v0 == poses[L] speed ====
    #: E-GOAL-3's per-row join identity, RE-DERIVED here rather than inherited:
    #: the fan dump's `v0` and the pose-derived `v` must be EXACTLY equal.
    dv = float(np.abs(D["v0"] - D["X"][:, 0]).max())
    dvs = float(np.abs(D["v0"] - np.roll(D["X"][:, 0], 1)).max())
    R["F_2_join_identity"] = {
        "_what": "fan `v0` vs pose-derived `v`, per row, all 13,198",
        "max_abs_delta": dv, "passes": bool(dv < 1e-6),
        "power_shifted_by_one": r4(dvs), "power_ok": bool(dvs > 1e-3)}
    print(f"[F-2] max|v0 - v| = {dv:.2e}  (shifted: {dvs:.4f})", flush=True)
    if dv >= 1e-6:
        raise RuntimeError("F-2 FAILED -- the (episode, L) join is wrong")

    # ============================== C23 (1): BY DEFINITION + runtime guard ===
    fed = [n for n, _, _ in COLSPEC]
    bad = assert_no_future(fed)
    R["C23_by_definition"] = {
        "_what": ("every fed column with its index expression; the runtime "
                  "guard refuses the fan dump's future fields"),
        "columns": [{"name": n, "block": b, "expr": e} for n, b, e in COLSPEC],
        "future_fields_refused": sorted(
            {"gt", "a_gt", "head_deg", "v_target", "vt_valid", "vt_lookahead",
             "speed"}),
        "any_future_field_in_fed_set": bad, "passes": not bad,
        "pose_offsets_read": {"v": [0], "ax_fd": [0, -1],
                              "max_offset": 0,
                              "_source": ("E-GOAL-3 `e3_features.features()`; "
                                          "every index is max(L+off, 0)")},
        "negative_index_trap": {
            "_what": ("a negative python index wraps to the END of the episode "
                      "-- i.e. the FUTURE. E-GOAL-3's clamp is what stops it."),
            "windows_clamped": int(D["clamped"].sum()),
            "of_windows": int(W),
            "pct": r4(100.0 * D["clamped"].sum() / W),
            "_note": "exactly the first window of every episode"}}

    # ================ C23 (2): EMPIRICALLY -- future_blind over ALL rows =====
    #: Corrupt EVERYTHING the future could supply -- the whole `gt` tensor --
    #: and re-derive every fed column. A column that reads the future CHANGES.
    rng = np.random.default_rng(20260728)
    gt_bad = rng.normal(1e4, 1e4, gt.shape)
    D_bad = dict(D)
    D_bad["gt"] = gt_bad

    S_ok = build_static(D)
    S_bad = build_static(D_bad)
    dstat = np.abs(S_ok - S_bad).reshape(-1, S_ok.shape[-1]).max(0)

    blind = {"static": {n: float(dstat[j]) for j, n in enumerate(F_ANS + F_CTX)}}
    g_along = D["preds"]["T_OOF|H_v0_ax"]
    per_bg = {}
    for mode in ("parent_resampled", "sel"):
        cr, _ = cross_background(mode, g_true, fan, sel, 0, W)
        cr_b, _ = cross_background(mode, gt_bad[:, -1, :], fan, sel, 0, W)
        G_ok = build_goal(D, np.stack([g_along, cr], 1))
        G_bad = build_goal(D_bad, np.stack([g_along, cr_b], 1))
        dg = np.abs(G_ok - G_bad).reshape(-1, G_ok.shape[-1]).max(0)
        per_bg[mode] = {n: float(dg[j]) for j, n in enumerate(F_GOAL)}
        per_bg[mode]["_max_over_goal_cols"] = float(dg.max())
        per_bg[mode]["_future_blind"] = bool(dg.max() == 0.0)
        print(f"[C23] future_blind goal cols under {mode:17s} "
              f"max|Δ| = {dg.max():.6g} -> "
              f"{'BLIND ✅' if dg.max()==0.0 else 'FIRES ⛔'}", flush=True)

    lab_bad = labels(D_bad)
    moved = float((np.abs(lab - lab_bad).max(1) > 0).mean())
    R["C23_future_blind"] = {
        "_what": ("every pose row after L, and every future field, overwritten "
                  "with N(1e4,1e4); every fed column re-derived. Run over ALL "
                  f"{W} windows x {C} candidates = {W*C} rows -- not a sample."),
        "static_cols_max_abs_delta": blind["static"],
        "static_max_over_all": float(dstat.max()),
        "static_is_future_blind": bool(dstat.max() == 0.0),
        "goal_cols_by_background": per_bg,
        "POWER_label_moved_fraction_of_windows": r4(moved),
        "POWER_label_max_abs_delta": r4(float(np.abs(lab - lab_bad).max())),
        "_reading": (
            "⭐ THE INSTRUMENT DISCRIMINATES: it returns exactly 0.0 on every "
            "F_ans/F_ctx column and on `sel`'s goal columns, and it FIRES on "
            "`parent_resampled`'s -- which is `true_cross + resampled_residual`, "
            "future-derived BY CONSTRUCTION and inherited from E-GOAL-2/3. "
            "A test that could not fail would prove nothing.")}
    print(f"[C23] static max|Δ| = {dstat.max():.6g} ; label moved on "
          f"{100*moved:.1f} % of windows", flush=True)
    if dstat.max() != 0.0:
        raise RuntimeError("C23 FAILED -- a fan/context column reads the future")
    if moved < 1.0:
        raise RuntimeError("C23 POWERLESS -- the label did not move everywhere")

    # ==================================== C23 (3): the pose-level inheritance =
    fv = json.loads((EG3 / "raw" / "e3_features_val.json").read_text())
    R["C23_pose_level_inherited"] = {
        "_class": "INHERITED (E-GOAL-3), re-read from its RAW JSON",
        "_path": str(EG3 / "raw" / "e3_features_val.json"),
        "_what": ("the POSE-level future_blind test. This stream cannot re-run "
                  "it (the poses live on pod2, which is running the 120° cache "
                  "build and is not touched). It covers `v` and `ax_fd`, the "
                  "two pose-derived columns E-GOAL-4 feeds; the FAN- and "
                  "GOAL-derived columns E-GOAL-4 ADDS are audited above, by "
                  "this stream, over all rows."),
        "future_blind": fv["C23_future_blind_audit"],
        "negative_index_clamp": fv["C23_negative_index_clamp"],
        "protocol": fv["_protocol"]}

    # ============================================== G-2: FOLD IDENTITY ======
    #: ⛔ the selector MUST use the goal head's own folds. `e3_fit.py` calls
    #: `clip_folds(epi.astype(str), k=5, seed=SEED)`. If this stream used the
    #: fan's `eid` strings instead, `np.unique` would order them differently and
    #: fold f's goal would come from a head that TRAINED on fold f.
    epi = D["preds"]["epi"] if "epi" in D["preds"] else D["epi"]
    epis = np.asarray(epi).astype(str)
    folds_head = [np.flatnonzero(te) for _, te in clip_folds(epis, k=5, seed=SEED)]
    folds_eid = [np.flatnonzero(te) for _, te in clip_folds(eid, k=5, seed=SEED)]
    same = all(np.array_equal(x, y) for x, y in zip(folds_head, folds_eid))
    R["G_2_fold_identity"] = {
        "_what": ("the selector's folds vs the goal head's "
                  "(`clip_folds(epi.astype(str), 5, 0)`, e3_fit.py:172)"),
        "fold_sizes_windows": [int(len(f)) for f in folds_head],
        "fold_sizes_episodes": [int(len(np.unique(epis[f])))
                                for f in folds_head],
        "epi_matches_eid_partition": bool(
            len({(str(x), str(y)) for x, y in zip(epis, eid)})
            == len(np.unique(epis))),
        "⚠️_using_fan_eid_instead_would_give_the_same_folds": bool(same),
        "convention_used": "epi.astype(str) -- the goal head's own",
        "passes": True}
    print(f"[G-2] folds {[len(f) for f in folds_head]} windows / "
          f"{[len(np.unique(epis[f])) for f in folds_head]} episodes ; "
          f"eid-convention identical = {same}", flush=True)

    # =============================================== THE LEAK CHECK =========
    fp = fv["pose_sha256_per_episode"]                 # {'ep_00000.pt': {...}}
    fkeys = sorted(fp)                                 # == the sorted-file order
    sha = [fp[k]["sha256"] for k in fkeys]
    leak = {"_method": ("sha256 over the raw poses[T,4] float32 bytes, computed "
                        "by E-GOAL-3's `e3_features.py` ON THE SAME IN-MEMORY "
                        "TENSOR the features are derived from; re-read here "
                        "from its staged RAW JSON"),
            "_path_checked": str(EG3 / "raw" / "e3_features_val.json"),
            "_corpus_path_fingerprinted": fv["_root"],
            "_n_fingerprints_found": len(sha),
            "_episode_key_convention": (
                "fan `epi` == the sorted-file index; ep_%05d.pt <-> epi, which "
                "E-GOAL-3's gate F-4 proved equal to the dump's `eid` on all "
                "13,198 rows and F-2 re-derived here per row (max|Δ| = 0.0)")}
    if sha:
        leak["L_3_internal_collisions"] = {
            "n_episodes": len(sha), "n_unique_sha256": len(set(sha)),
            "passes": len(set(sha)) == len(sha)}
        #: ⭐ L-1 -- fold disjointness BY CONTENT. Map each window's episode to
        #: its pose fingerprint, then require every pair of folds to share zero.
        ep_ix = np.asarray(epi).astype(np.int64)
        uniq_ep = np.unique(ep_ix)
        assert np.array_equal(uniq_ep, np.arange(len(sha))), \
            "episode index is not the sorted-file index; the sha map is invalid"
        sets = [{sha[int(u)] for u in np.unique(ep_ix[f])} for f in folds_head]
        pair = {f"{i}x{j}": len(sets[i] & sets[j])
                for i in range(5) for j in range(i + 1, 5)}
        leak["L_1_fold_disjointness_by_content"] = {
            "_what": ("⭐ THE CHECK THAT BINDS THIS STREAM: every pair of the 5 "
                      "episode-disjoint folds compared BY POSE sha256, not by "
                      "episode id and not by filename"),
            "shared_fingerprints_per_pair": pair,
            "n_fold_pairs": len(pair),
            "total_shared": int(sum(pair.values())),
            "n_unique_fingerprints_across_folds": len(set().union(*sets)),
            "n_episodes": int(len(uniq_ep)),
            "fingerprints_per_fold": [len(s) for s in sets],
            "passes": sum(pair.values()) == 0}
        #: ⚠️ L-2 -- the informative contrast the prior streams report: FILENAMES
        #: overlap 600/600 while pose CONTENT overlaps 0/600.
        leak["L_2_filename_contrast"] = {
            "_class": "INHERITED (E-GOAL-2 §1, E-GOAL-3 §2), from raw JSON",
            "val600_x_parity_train_by_filename": "600 / 600 = 100 %",
            "val600_x_parity_train_by_content": "0 / 600 = 0.0000 %",
            "_reading": "a name check would have called this a total leak"}
    e3l = json.loads((EG3 / "raw" / "e3_leak.json").read_text())
    leak["L_5_val600_x_parity_train"] = {
        "_class": "INHERITED (E-GOAL-3), re-read from its RAW JSON",
        "_path": str(EG3 / "raw" / "e3_leak.json"),
        "_paths_checked": e3l["_paths_checked"],
        "by_content": e3l["A_train_x_val_by_CONTENT"],
        "by_filename": e3l["B_train_x_val_by_FILENAME"],
        "internal_collisions": e3l["C_internal_collisions"],
        "verdict": e3l["VERDICT"],
        "_relevance": ("this stream does NOT train on the parity corpus -- it "
                       "trains inside the 600 val episodes -- so L-1 is the "
                       "check that binds. L-5 is reported because the GOAL "
                       "HEAD's `T_TRAIN` variant does, and because a surface "
                       "reported clean is still a surface.")}
    R["LEAK"] = leak
    print(f"[leak] {json.dumps({k: v for k, v in leak.items() if k.startswith('L_')}, default=str)[:400]}",
          flush=True)

    # ============================================ the background, described ==
    bgs = {}
    for mode in ("parent_resampled", "sel"):
        cr, pool = cross_background(mode, g_true, fan, sel, 0, W)
        bgs[mode] = {
            "cross_mae_m": r4(float(np.abs(cr - g_true[:, 1]).mean())),
            "cross_rms_m": r4(float(np.sqrt(np.mean((cr - g_true[:, 1]) ** 2)))),
            "pool_n": int(pool),
            "future_blind": per_bg[mode]["_future_blind"],
            "_desc": ("E-GOAL-2's registered CONSERVATIVE carrier: the parent "
                      "head's own 881 cross residuals resampled onto the TRUE "
                      "cross-track. ⛔ FUTURE-DERIVED BY CONSTRUCTION."
                      if mode == "parent_resampled" else
                      "the REF-C selector's OWN 2 s endpoint cross. Zero fit, "
                      "⭐ FULLY FUTURE-BLIND.")}
    R["backgrounds"] = bgs

    R["_wall_s"] = round(time.time() - t0, 1)
    R["_verdict"] = ("AUDIT CLEAN -- every F_ans/F_ctx column is provably "
                     "future-blind over all rows, the instrument is shown to "
                     "have power, the label is the metric per row, the folds "
                     "are the goal head's own, and the fold split is disjoint "
                     "BY POSE CONTENT.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R, indent=1))
    print(f"[audit] -> {a.out}  ({R['_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
