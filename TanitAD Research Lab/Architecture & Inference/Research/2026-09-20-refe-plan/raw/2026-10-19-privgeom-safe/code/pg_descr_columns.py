"""PRIVGEOM-SAFE, POST-HOC DESCRIPTIVE table (no authority over any branch; NOT in the prereg's decision rule, NOT pinned by Amendment 1).

Question it informs (SH5 target design): which of the DAC companion's 10 columns (`DAC-nosign`: abs of dac_clear_min and of dac_clear_t00..t40) carry the jump?
Seed 0 only, the registered recipe, the registered folds and the registered paired log-cluster bootstrap (the SAME draws as the primary read).

  fit      --shard i --nshards n : refits the L+G(V) arm DROPPING one column (drop_j) and KEEPING ONE column (only_j), j = 0..9, into run/descr/fits/.
  analyze  writes run/descr/descr_columns.json and run/descr/DESCR_columns.md:
           (a) the standardised L+G(V) coefficients u of the registered fit (LGV_dac_s0.json), mean +- sd over the 5 folds;
           (b) the drop-one table: F-AUC of the 9-column arm, Delta_F vs L, and the DROP EFFECT = AUC_F(LGV) - AUC_F(LGV minus j) (paired);
           (c) the single-column table: F-AUC of L + column j, Delta_F vs L;
           (d) the raw sign-agnostic within-set F-AUC of each |column| alone (no fit) in the better direction, and the mean of each |column| by label.
Run with the registered interpreter (C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -B -W ignore), CUDA_VISIBLE_DEVICES=-1.  It reads the stored labels through the gate's `labels_sub.npy`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pg_lib as P                                                                       # noqa: E402
import pg_run as R                                                                       # noqa: E402

FL = P.FL
ROOT = R.ROOT
OUTD = f"{ROOT}/run/descr"
SEED, ROW = 0, "dac"
NAMES = ["abs_clear_min"] + [f"abs_clear_t{s:02d}" for s in (0, 5, 10, 15, 20, 25, 30, 35, 40)]   # = pg_geom.unsigned order: dac_clear_min, then the 9 key steps
K = len(NAMES)


def tasks() -> list:
    return [("drop", j) for j in range(K)] + [("only", j) for j in range(K)]


def cols_for(kind: str, j: int) -> list:
    return [c for c in range(K) if c != j] if kind == "drop" else [j]


def stage_fit(a) -> int:
    R.require_prereg()
    R.require_token()
    A = R.load_join(labels=True)
    gv = R.g_for(A, ROW, "GV")
    assert gv.shape[-1] == K
    os.makedirs(f"{OUTD}/fits", exist_ok=True)
    col = R.LABEL_COL[ROW]
    z_st, y = np.ascontiguousarray(A["z6"][..., col]), np.ascontiguousarray(A["lab"][..., col])
    T = tasks()[a.shard::a.nshards]
    R.log(f"descr fit: shard {a.shard}/{a.nshards}: {len(T)} tasks")
    for kind, j in T:
        out = f"{OUTD}/fits/{kind}{j}_dac_s{SEED}"
        if os.path.exists(out + ".npy") and os.path.exists(out + ".json"):
            continue
        t0 = time.time()
        g = np.ascontiguousarray(gv[..., cols_for(kind, j)])
        trace: list = []
        oof = P.oof_logits("LG", A["feat"], z_st, y, g, A["fold"], SEED, A["p2_W6"][col], float(A["p2_b6"][col]), ram_guard=lambda: FL.wait_for_ram(3.0, log=R.log), trace=trace)
        np.save(out + ".npy", oof)
        R.jdump({"kind": kind, "j": j, "name": NAMES[j], "cols": cols_for(kind, j), "seed": SEED, "seconds": time.time() - t0, "oof_sha256": R.sha(out + ".npy"), "folds": trace}, out + ".json")
        R.log(f"  {kind}{j} ({NAMES[j]}): {time.time() - t0:.0f} s")
    open(f"{OUTD}/fits/DONE_{a.shard}.txt", "w").write(R.now())
    R.log(f"DESCR_SHARD_DONE {a.shard}")
    return 0


def stage_analyze(a) -> int:
    R.require_prereg()
    R.require_token()
    A = R.load_join(labels=True)
    an = P.Analyzer(A["lab"], A["code"], A["log_idx"], A["n_logs"])
    Pth = R.paths()
    st = lambda p: an.per_log_stats(np.load(p), ROW)                                      # noqa: E731
    sL, sLG, sLGV = (st(f"{Pth['fits']}/{n}_dac_s{SEED}.npy") for n in ("L", "LG", "LGV"))
    base = {"F_auc_L": an.auc(sL, "Q4"), "F_auc_LG_primary": an.auc(sLG, "Q4"), "F_auc_LGV_all10": an.auc(sLGV, "Q4"), "delta_F_LGV_all10_vs_L": an.diff(sLGV, sL, "Q4")}
    # (a) standardised coefficients of the registered LGV fit
    reg = json.load(open(f"{Pth['fits']}/LGV_dac_s{SEED}.json", encoding="utf-8"))
    U = np.array([f["u"] for f in reg["folds"]])
    coef = {"names": NAMES, "mean_u": U.mean(0).tolist(), "sd_u_over_folds": U.std(0).tolist(), "u_by_fold": U.tolist()}
    # (b), (c)
    drop, only = [], []
    for j in range(K):
        sd_, so_ = st(f"{OUTD}/fits/drop{j}_dac_s{SEED}.npy"), st(f"{OUTD}/fits/only{j}_dac_s{SEED}.npy")
        drop.append({"j": j, "name": NAMES[j], "F_auc": an.auc(sd_, "Q4"), "delta_F_vs_L": an.diff(sd_, sL, "Q4"), "drop_effect": an.diff(sLGV, sd_, "Q4")})
        only.append({"j": j, "name": NAMES[j], "F_auc": an.auc(so_, "Q4"), "delta_F_vs_L": an.diff(so_, sL, "Q4"), "share_of_all10_delta": float(an.diff(so_, sL, "Q4")["point"] / base["delta_F_LGV_all10_vs_L"]["point"])})
    # (d) raw sign-agnostic within-set F-AUC of each |column| alone, and the mean by label
    gv = R.g_for(A, ROW, "GV")
    lab = A["lab"][..., R.LABEL_COL[ROW]]
    raw = []
    for j in range(K):
        x = gv[..., j].astype(np.float64)
        # higher score = safer: AUC(x) is "a larger |value| is safer"; AUC(-x) its mirror; report both and the better one
        a_hi = an.auc(an.per_log_stats(x.astype(np.float32), ROW), "Q4")["point"]
        a_lo = an.auc(an.per_log_stats((-x).astype(np.float32), ROW), "Q4")["point"]
        raw.append({"j": j, "name": NAMES[j], "F_auc_larger_is_safer": a_hi, "F_auc_smaller_is_safer": a_lo, "mean_unsafe": float(x[lab == 0.0].mean()), "mean_safe": float(x[lab == 1.0].mean()), "sd": float(x.std())})
    out = {"schema": "privgeom-safe-descr-columns-v1", "created": R.now(), "authority": "DESCRIPTIVE, post-hoc, no authority over any branch", "seed": SEED, "row": ROW, "base": base, "coef": coef, "drop": drop, "only": only, "raw": raw,
           "tier": "SELECTION, SEEN scenes, P2 in-sample; 41 logs; interval answers 'another draw of LOGS' only"}
    R.jdump(out, f"{OUTD}/descr_columns.json")
    f4 = lambda c: f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"                  # noqa: E731
    u4 = lambda c: f"{c['point']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]"                      # noqa: E731
    L = ["# DESCRIPTIVE (post-hoc, no authority over any branch): which DAC-nosign columns carry the jump (seed 0, F stratum)", "",
         f"Reference (registered fits): F-AUC L {u4(base['F_auc_L'])}; L+G(V), all 10 columns {u4(base['F_auc_LGV_all10'])}; Delta_F(L+G(V) - L) {f4(base['delta_F_LGV_all10_vs_L'])}; the primary L+G (11 columns incl. the sign) {u4(base['F_auc_LG_primary'])}.", "",
         "## (a) standardised L+G(V) coefficients u (logit per training-set SD of the column; mean and SD over the 5 folds)", "", "| column | mean u | sd over folds |", "|---|---|---|"]
    for j in range(K):
        L.append(f"| `{NAMES[j]}` | {coef['mean_u'][j]:+.3f} | {coef['sd_u_over_folds'][j]:.3f} |")
    L += ["", "## (b) drop-one: the 9-column arm without column j (drop effect = AUC_F(all 10) - AUC_F(without j), paired)", "", "| dropped | F-AUC without j | Delta_F vs L | drop effect |", "|---|---|---|---|"]
    for d in drop:
        L.append(f"| `{d['name']}` | {u4(d['F_auc'])} | {f4(d['delta_F_vs_L'])} | {f4(d['drop_effect'])} |")
    L += ["", "## (c) single-column: L + column j only", "", "| kept | F-AUC | Delta_F vs L | share of the all-10 Delta_F |", "|---|---|---|---|"]
    for o in only:
        L.append(f"| `{o['name']}` | {u4(o['F_auc'])} | {f4(o['delta_F_vs_L'])} | {o['share_of_all10_delta']:.2f} |")
    L += ["", "## (d) raw, no fit: within-set F-AUC of the |column| alone (higher score = safer) and its mean by label", "", "| column | AUC (larger = safer) | AUC (smaller = safer) | mean unsafe | mean safe |", "|---|---|---|---|---|"]
    for r in raw:
        L.append(f"| `{r['name']}` | {r['F_auc_larger_is_safer']:.4f} | {r['F_auc_smaller_is_safer']:.4f} | {r['mean_unsafe']:.3f} | {r['mean_safe']:.3f} |")
    L += ["", f"Evidence class: MEASURED, post-hoc, DESCRIPTIVE. Estimator: paired LOG-cluster bootstrap over {A['n_logs']} logs, B = {P.BOOT_B}, seed {P.BOOT_SEED} (the primary read's draws); it answers 'another draw of LOGS' only, not training variance (one seed here).", ""]
    with open(f"{OUTD}/DESCR_columns.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    R.log("descr analyze OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("fit", "analyze"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(argv)
    return {"fit": stage_fit, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
