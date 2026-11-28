"""pg_labels.py -- PROBEGATE / RANDGATE label builder (PREREG_PROBEGATE.md section 2-4).  CPU only.  REFUSES to run unless the prereg matches its seal.

Cross-fits lonprobe.py's probe (v1: fit_pred / folds / load_bank, unchanged) over the 136 episodes that carry S6 compliant GT-decel/GT-cruise windows
(5 folds, folds(eid, 5, 0), fit_pred seed 0) and predicts a score for EVERY window of the 4,754-window roll in the held-out episodes.
Writes (all small, in this directory):
  probe_scores_s0.npz       score [4754] in roll (extras) order + fold id per window + clip_index/ws
  labels_probegate_s0.npz   oracle format {clip_index, ws, lon_v7 float32 (forced v7 id or -100)}
  labels_randgate_s0.npz    same, RANDGATE
  LABELS_CHECK.json         counts, v1-reproduction check, descriptive flip table, sha256s
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L2 = HERE.parent
sys.path.insert(0, str(L2 / "code"))
sys.path.insert(0, str(L2 / "lonprobe"))
sys.path.insert(0, str(L2 / "roc"))
sys.path.insert(0, str(HERE))
import l2_common as C  # noqa: E402
import lonprobe as LP  # noqa: E402
from l2_roc import auc, cls_of, dv_of  # noqa: E402
import pg_policy as P  # noqa: E402

V1_AUC_PROBE_S0 = 0.7623863391074416            # lonprobe/LONPROBE.json seeds.0.auc.PROBE (copied here as a literal, NOT read back)
V1_PICKS_S0 = [[32, 0.01], [64, 0.01], [16, 0.01], [32, 1.0], [32, 0.01]]
HEAD_TRACE = L2 / "bank" / "trace_oracle_s0.npz"   # raw_lon_logits = the head's output in the os-only roll; identical across all three L2 traces (max abs 0.0)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha12(x):
    return hashlib.sha256(str(x).encode()).hexdigest()[:12]


def save_labels(path, clip, ws, lab):
    np.savez(path, clip_index=np.asarray(clip, np.int64), ws=np.asarray(ws, np.int64), lon_v7=np.asarray(lab, np.float32))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="GLUE TEST ONLY: no seal required, NO PROBE IS FITTED (a seeded random score stands in), outputs go to --outdir")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    OUT = Path(a.outdir) if a.outdir else HERE
    OUT.mkdir(parents=True, exist_ok=True)
    if a.dry and OUT == HERE:
        raise SystemExit("[pg_labels] --dry must write to a scratch --outdir, never over the real outputs")
    t0 = time.time()
    got = sha256_file(HERE / "PREREG_PROBEGATE.md") if (HERE / "PREREG_PROBEGATE.md").exists() else "ABSENT"
    if not a.dry:
        want = (HERE / "PREREG_PROBEGATE.sha256").read_text().split()[0]
        if len(want) != 64 or want != got:
            raise SystemExit(f"[pg_labels] PREREG_PROBEGATE.md does not match its seal ({got[:12]} vs {want[:12]}): refusing to score the roll")
    chk = {"prereg_sha256": got, "dry_glue_test": bool(a.dry), "tool": "pg_labels.py", "script_sha256": sha256_file(__file__), "policy_sha256": sha256_file(HERE / "pg_policy.py"),
           "lonprobe_py_sha256": sha256_file(L2 / "lonprobe" / "lonprobe.py")}
    E = C.load_eval_pf(0)
    ex = np.load(C.ARCH / C.EVAL_TREE / "dump_s0" / "refcv6_extras.npz", allow_pickle=True)
    clip_r, ws_r, v0_r = ex["clip_index"].astype(np.int64), ex["ws"].astype(np.int64), ex["v0"].astype(np.float64)
    n_roll = len(ws_r)
    assert n_roll == 4754, n_roll
    tr = np.load(HEAD_TRACE, allow_pickle=False)
    assert np.array_equal(tr["e_i"], clip_r) and np.array_equal(tr["ws"], ws_r), "head trace order != extras order"
    raw8 = tr["raw_lon_logits"].astype(np.float32)
    assert np.isfinite(raw8).all()
    # episode id of every roll window (clip -> eid is one-to-one on the S6 surface; verified at prereg time: 139 clips, 139 eids)
    c2e = {}
    for c, e in zip(E["clip"].tolist(), E["eid"].tolist()):
        assert c2e.setdefault(c, e) == e
    eid_r = np.array([c2e[int(c)] for c in clip_r], np.int64)
    # ---- v1's fit rows, constructed EXACTLY as lonprobe.main() does ----------------------------------------------------------------
    idx, Z = LP.load_bank(LP.BANK / "ztac_eval_s0.npz")
    gt = cls_of(dv_of(E["g"], E["v0"], E["dt"]))
    m = E["compliant"] & np.isin(gt, ["decel", "cruise"])
    rows = np.flatnonzero(m)
    j = np.array([idx.get((int(c), int(w)), -1) for c, w in zip(E["clip"][rows], E["ws"][rows])])
    assert (j >= 0).all()
    y = (gt[rows] == "decel").astype(int)
    eid_f = E["eid"][rows]
    Xz_f, Xs_f = Z[j], E["v0"][rows][:, None]
    n_fit = len(rows)
    # ---- roll-window features: z_tac joined on (clip, ws), v0 from the banked extras ----------------------------------------------------
    jr = np.array([idx.get((int(c), int(w)), -1) for c, w in zip(clip_r, ws_r)])
    assert (jr >= 0).all(), "roll windows without a z_tac bank entry"
    Xz = np.vstack([Xz_f, Z[jr]])
    Xs = np.vstack([Xs_f, v0_r[:, None]])
    y_all = np.concatenate([y, np.zeros(n_roll, int)])            # roll labels are never read (te rows only)
    eid_all = np.concatenate([eid_f, eid_r])
    # ---- folds: v1's folds over the fit-row episodes; episodes without fit rows (no model can have trained on them) by sorted position mod 5 ----
    of = LP.folds(eid_f, 5, 0)
    fold_of = {int(e): int(f) for e, f in zip(eid_f, of)}
    unseen = sorted(set(int(e) for e in eid_r) - set(fold_of))
    for r, e in enumerate(unseen):
        fold_of[e] = r % 5
    fw = np.array([fold_of[int(e)] for e in eid_r], np.int64)
    score = np.full(n_roll, np.nan)
    picks = []
    for f in range(5):
        trn = np.flatnonzero(of != f)
        te = n_fit + np.flatnonzero(fw == f)
        if a.dry:
            s, bk = np.random.default_rng(1000 + f).normal(size=len(te)), (0, 0.0)
        else:
            s, bk = LP.fit_pred(Xz, Xs, y_all, trn, te, eid_all, True)
        score[te - n_fit] = s
        picks.append([int(bk[0]), float(bk[1])])
    assert np.isfinite(score).all()
    # ---- self-check against v1 (already-reported numbers; same models by construction) -----------------------------------------------------
    s2idx = E["rows"][rows]                                         # S2 (roll) index of every fit row
    a_fit = float(auc(-score[s2idx][y == 1], -score[s2idx][y == 0]))
    chk["v1_reproduction"] = {"auc_probe_fit_rows": a_fit, "v1_auc_probe_s0": V1_AUC_PROBE_S0, "abs_diff": abs(a_fit - V1_AUC_PROBE_S0),
                              "picks": picks, "v1_picks": V1_PICKS_S0, "picks_equal": picks == V1_PICKS_S0,
                              "pass": bool(abs(a_fit - V1_AUC_PROBE_S0) < 1e-6 and picks == V1_PICKS_S0)}
    chk["n_fit_rows"], chk["n_fit_decel"], chk["n_fit_cruise"], chk["n_fit_episodes"] = int(n_fit), int(y.sum()), int((1 - y).sum()), int(len(np.unique(eid_f)))
    chk["n_roll"] = int(n_roll)
    chk["windows_per_fold"] = [int((fw == f).sum()) for f in range(5)]
    chk["unseen_episodes_sha12"] = [sha12(e) for e in unseen]
    chk["unseen_episode_windows"] = int(np.isin(eid_r, unseen).sum())
    # ---- policy ----------------------------------------------------------------------------------------------------------------------------
    lab_p, up_p, dn_p, info = P.probegate(score, raw8)
    lab_r, up_r, dn_r, infor = P.randgate(raw8, int(up_p.sum()), int(dn_p.sum()), seed=0)
    grp, dpick, cpick = P.head_view(raw8)
    chk["head_group_counts"] = {g: int((grp == g).sum()) for g in "DCA"}
    chk["head_decel_rate_roll"] = float((grp == "D").mean())
    chk["tau_info"] = info
    chk["probe_decel_rate"] = float(info["k"] / n_roll)
    chk["probegate"] = {"n_up": int(up_p.sum()), "n_down": int(dn_p.sum()), "n_forced": int((lab_p != P.NONE).sum()),
                        "forced_class_counts_v7": {str(k): int((lab_p == k).sum()) for k in sorted(set(lab_p.tolist()) - {P.NONE})}}
    chk["randgate"] = {"n_up": int(up_r.sum()), "n_down": int(dn_r.sum()), "n_forced": int((lab_r != P.NONE).sum()), **infor,
                       "forced_class_counts_v7": {str(k): int((lab_r == k).sum()) for k in sorted(set(lab_r.tolist()) - {P.NONE})},
                       "overlap_with_probegate_flips": int(((up_r | dn_r) & (up_p | dn_p)).sum())}
    assert int(up_p.sum()) == int(dn_p.sum()), "equal-marginal rate must give equal up/down counts"
    # head argmax from the battery's banked logits vs the roll's raw logits (a descriptive robustness number)
    gb = P.head_view(E["logits_s2"].astype(np.float32))[0]
    chk["head_group_disagreements_roll_vs_battery_logits"] = int((gb != grp).sum())
    # ---- descriptive flip table on the scored S6 population (GT is read AFTER the labels are fixed; nothing below feeds back) ----------------
    comp_s2 = np.zeros(n_roll, bool)
    gt_s2 = np.full(n_roll, "none", dtype=object)
    comp_s2[E["rows"]] = E["compliant"]
    gt_s2[E["rows"]] = gt
    desc = {}
    for nm, up, dn in (("probegate", up_p, dn_p), ("randgate", up_r, dn_r)):
        d = {}
        for dirn, mk in (("up", up), ("down", dn)):
            d[dirn] = {"total": int(mk.sum()), "in_S6_compliant": int((mk & comp_s2).sum()),
                       **{f"GT_{c}": int((mk & comp_s2 & (gt_s2 == c)).sum()) for c in ("decel", "cruise", "accel")}}
        desc[nm] = d
    chk["descriptive_flips_on_S6_compliant"] = desc
    # ---- write -----------------------------------------------------------------------------------------------------------------------------
    np.savez(OUT / "probe_scores_s0.npz", score=score, fold=fw, clip_index=clip_r, ws=ws_r)
    save_labels(OUT / "labels_probegate_s0.npz", clip_r, ws_r, lab_p)
    save_labels(OUT / "labels_randgate_s0.npz", clip_r, ws_r, lab_r)
    chk["sha256"] = {f: sha256_file(OUT / f) for f in ("probe_scores_s0.npz", "labels_probegate_s0.npz", "labels_randgate_s0.npz")}
    chk["head_trace_sha256"] = sha256_file(HEAD_TRACE)
    chk["wall_s"] = round(time.time() - t0, 1)
    chk["ALL_PASS"] = bool(chk["v1_reproduction"]["pass"])
    tmp = OUT / "LABELS_CHECK.json.tmp"
    tmp.write_text(json.dumps(chk, indent=1), encoding="utf-8")
    tmp.replace(OUT / "LABELS_CHECK.json")
    (OUT / "labels.sha256").write_text("".join(f"{v}  {k}\n" for k, v in chk["sha256"].items()), encoding="utf-8")
    print(json.dumps({k: chk[k] for k in ("v1_reproduction", "probegate", "randgate", "head_group_counts", "tau_info", "ALL_PASS")}, indent=1, default=str))
    print("ZZPGLABELSDONEZZ", chk["ALL_PASS"])
    return 0 if chk["ALL_PASS"] else 3


if __name__ == "__main__":
    sys.exit(main())
