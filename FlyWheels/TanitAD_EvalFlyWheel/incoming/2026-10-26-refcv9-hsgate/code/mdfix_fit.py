"""mdfix_fit.py -- PREREG_HSGATE_MDFIX (sha256 545ff217...) Phase A (label, K-LABEL, s1a power rule) and, iff powered, the FIT-side freeze
(MDFIX_FROZEN.json, K-TAU, K-MUT).  FIT rows ONLY (L1C fresh split, 7,147 S6 compliant rows banked by hsgate_fit.py in FIT_ROWS.npz); NO eval139 read.
NEW file, refcv9-hsgate agent, 2026-10-26.

* label  y = 1[GT-decel AND lonclass(P_F) == decel AND lonclass(P_H-OOF) != decel]   (l2_common.lon_indicators / plan_class, as the MD / FD cells)
* lambda 0.1 FIXED; held-out logits: 5-fold episode-grouped CV, fold seed 1 (= FIT_ROWS.cv_fold, the HSGATE lambda-CV folds)
* tau(c) = the held-out logit threshold minimising FIT MD s.t. FD_FIT <= FD_FIT(H2cn-OOF) + c, ties -> HIGHER tau; candidates = every distinct held-out
  logit plus +inf (no gating).  MD / FD minimised on INTEGER counts (no float ties).
* s1a power rule for c in (0.002, 0.005): FIT-side paired episode-cluster bootstrap (taniteval.ci draws, B 2000, seed 0) of MDFIX-OOF - H2cn-OOF;
  SE_FIT = std(ddof=1) of the bootstrap deltas; SE_eval_MD = SE_FIT_MD * sqrt(E_FIT_decel / 79); SE_eval_FD = SE_FIT_FD * sqrt(E_FIT / 135);
  power_G1 = Phi(|dMD| / SE_eval_MD - 2.241); c = smallest with power >= 0.50 AND dFD + 1.96 * SE_eval_FD <= 0.010; else UNDERPOWERED (terminal).
* freeze: all-FIT-rows fit at lambda 0.1 (standardised on all FIT rows), g(x) = 1[w.xs + b >= tau]; MDFIX_FROZEN.json canonical bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hsgate_common as H  # noqa: E402

import numpy as np  # noqa: E402

PREREG = H.WORK / "PREREG_HSGATE_MDFIX.md"
PREREG_SHA = H.WORK / "PREREG_HSGATE_MDFIX.sha256"
LAMBDA = 0.1
C_GRID = (0.002, 0.005)
B, BOOT_SEED = 2000, 0
E_EVAL_MD, E_EVAL_FD = 79, 135
Z_G1 = 2.241                      # one-sided 98.75 % (the s3 G1 97.5 % two-sided interval)
POWER_MIN, FD_UB_MAX = 0.50, 0.010
BASE_RATE_EXPECTED = 0.0371       # LITERAL from the sealed draft / prereg s2 K-LABEL -- not an expression over this code
FROZEN = H.WORK / "MDFIX_FROZEN.json"
FIT_INFO = H.WORK / "MDFIX_FIT_INFO.json"
POWER_JSON = H.WORK / "next" / "MDFIX_POWER_FIT.json"
DISCLOSURES = [
    "Forking paths: the MDFIX label was chosen AFTER the HSGATE8 eval139 read, on FIT-row arithmetic only (next/FDCAP_FRONTIER_FIT.json; "
    "next/DRAFT_NEXT_ARM_HSGATE_MDFIX.md). This is the SECOND eval139 read of the gate family; the multiplicity is paid in the bars (G1 at 97.5 %).",
    "No MDFIX output on eval139 existed at sealing; the draft's FIT-side numbers are 5-fold episode CV on 7,147 FIT rows, one training seed, one fit, no interval.",
]


def log(m):
    print(f"[mdfix_fit {time.strftime('%H:%M:%S')}] {m}", flush=True)


def check_prereg_mdfix() -> dict:
    a, b = H.sha256_file(PREREG), H.first_token(PREREG_SHA)
    if len(a) != 64 or len(b) != 64 or a != b:
        raise SystemExit(f"[mdfix_fit] PREREG_HSGATE_MDFIX seal check failed ({len(a)}/{len(b)} chars, equal={a == b})")
    return {"file": str(PREREG), "sha256": a}


def heldout_logits(X, y, fold, lam):
    z = np.full(len(y), np.nan)
    for j in np.unique(fold):
        tr, te = fold != j, fold == j
        mu, sd = H.standardise_fit(X[tr])
        m = H.lr_fit((X[tr] - mu) / sd, y[tr], lam)
        z[te] = H.gate_logit({"mu": mu, "sd": sd, "w": m["w"], "b": m["b"]}, X[te])
    return z


def counts(g, A):
    """integer (missed-decel, false-decel) counts of the gated plan: F on g, H off g."""
    miss = np.where(g, A["mdF"], A["mdH"])
    fals = np.where(g, A["fdF"], A["fdH"])
    return int(np.nansum(miss)), int(np.nansum(fals))


def tau_scan_prefix(z, A, c):
    """PRIMARY: prefix cumsum over distinct thresholds (descending); +inf first.  Ties in MD -> the HIGHER tau (first in descending order)."""
    o = np.argsort(-z, kind="stable")
    zs = z[o]
    dmd = np.where(A["dec"], np.nan_to_num(A["mdF"]) - np.nan_to_num(A["mdH"]), 0.0)[o]
    dfd = np.where(A["cru"], np.nan_to_num(A["fdF"]) - np.nan_to_num(A["fdH"]), 0.0)[o]
    cut = np.r_[np.where(zs[:-1] > zs[1:])[0], len(z) - 1]
    md = np.r_[A["md0"], A["md0"] + np.cumsum(dmd)[cut]].round().astype(np.int64)
    fd = np.r_[A["fd0"], A["fd0"] + np.cumsum(dfd)[cut]].round().astype(np.int64)
    taus = np.r_[np.inf, zs[cut]]
    ok = fd <= A["fd0"] + c * A["n_cru"] + 1e-9
    best = int(np.min(md[ok]))
    i = int(np.where(ok & (md == best))[0][0])          # descending order -> first == highest tau
    return {"tau": float(taus[i]), "md_count": int(md[i]), "fd_count": int(fd[i]), "n_candidates": int(len(taus)), "n_feasible": int(ok.sum())}


def tau_scan_bruteforce(z, A, c):
    """K-TAU: an independent re-derivation -- every candidate threshold evaluated by a direct mask, ascending order, ties broken explicitly."""
    best = None
    for t in np.r_[np.unique(z), np.inf]:
        g = z >= t
        mdc, fdc = counts(g, A)
        if fdc <= A["fd0"] + c * A["n_cru"] + 1e-9:
            if best is None or mdc < best[1] or (mdc == best[1] and t > best[0]):
                best = (float(t), mdc, fdc)
    return {"tau": best[0], "md_count": best[1], "fd_count": best[2]}


def boot_se(new, ref, eid):
    """paired episode-cluster bootstrap with taniteval.ci's own draws (B 2000, seed 0): point, SE (std ddof 1), 95 % interval, n."""
    from taniteval import ci
    keep = np.isfinite(new) & np.isfinite(ref)
    a, b, e = new[keep], ref[keep], [x for x, k in zip(eid, keep) if k]
    uniq, idx = ci.episode_index(e)
    d = np.array([float(a[s].mean() - b[s].mean()) for s in ci._draws(uniq, idx, B, BOOT_SEED)])
    r = ci.paired_episode_cluster_bootstrap(a, b, e, n_boot=B, seed=BOOT_SEED)
    return {"delta": float(a.mean() - b.mean()), "se_boot": float(d.std(ddof=1)), "ci95_lo": r["lo"], "ci95_hi": r["hi"], "separated95": r["separated"],
            "n_windows": int(keep.sum()), "n_episodes": int(len(uniq)), "ci_delta_rendered": r["delta"]}


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def freeze_gate(X, y, A, z_ho, c, label_name):
    tau = tau_scan_prefix(z_ho, A, c)
    mu, sd = H.standardise_fit(X)
    m = H.lr_fit((X - mu) / sd, y, LAMBDA)
    gate = {"arm": "MDFIX8", "prereg": "PREREG_HSGATE_MDFIX.md", "label": label_name, "lambda": LAMBDA, "c": c, "tau": tau["tau"],
            "decision": "g(x) = 1[ ((x - mu) / sd) . w + b >= tau ]  (per-window logit threshold, no batch statistic)",
            "features": H.FEATURE_SPEC, "mu": [float(v) for v in mu], "sd": [float(v) for v in sd], "w": [float(v) for v in m["w"]], "b": float(m["b"]),
            "fit": {"loss": m["loss"], "max_abs_grad": m["max_abs_grad"], "n_iter": m["n_iter"], "objective": H.OBJECTIVE}}
    return gate, tau, m


def main() -> int:
    t0 = time.time()
    H.env_check()
    seal = check_prereg_mdfix()
    log(f"PREREG_HSGATE_MDFIX verified {seal['sha256'][:12]}")
    if FROZEN.exists():
        raise SystemExit("[mdfix_fit] MDFIX_FROZEN.json already exists -- refusing to re-freeze")
    fit_info_h = json.loads((H.WORK / "HSGATE_FIT_INFO.json").read_text(encoding="utf-8"))
    L = H.lane()
    import hsgate_fit as HF
    tr6 = HF.load_fit_side(L)["trs"]["S6"]
    R = np.load(H.SCRATCH / "FIT_ROWS.npz")
    if not (np.array_equal(R["clip_index"], tr6["clip"]) and np.array_equal(R["ws"], tr6["ws"])):
        raise SystemExit("[mdfix_fit] FIT_ROWS row order != load_fresh S6 rows")
    sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
    import l2_common as L2
    E = {"g": np.asarray(tr6["G"], np.float64), "v0": np.asarray(tr6["v0"], np.float64), "dt": float(tr6["dt"])}
    allm = np.ones(len(R["y"]), bool)
    fdF, mdF, gt, pcF = L2.lon_indicators(R["PF6"], E, allm)
    fdH, mdH, gt2, pcH = L2.lon_indicators(R["PH6"], E, allm)
    X, fold, eid = R["X"], R["cv_fold"], R["episode_code"]
    cru, dec = gt == "cruise", gt == "decel"
    A = {"fdF": fdF, "mdF": mdF, "fdH": fdH, "mdH": mdH, "cru": cru, "dec": dec, "n_cru": int(cru.sum()), "n_dec": int(dec.sum()),
         "md0": int(np.nansum(mdH)), "fd0": int(np.nansum(fdH))}
    # ---- label + K-LABEL (two derivations: plan classes vs MD indicators; base rate vs the LITERAL 0.0371)
    y = (dec & (pcF == "decel") & (pcH != "decel")).astype(np.float64)
    y_md = (dec & (np.nan_to_num(mdH) == 1) & (np.nan_to_num(mdF) == 0)).astype(np.float64)
    k_label = {"label_equals_md_indicator_derivation_every_row": bool(np.array_equal(y, y_md)), "gt_class_equal_both_calls": bool(np.array_equal(gt, gt2)),
               "base_rate": float(y.mean()), "base_rate_4dp": round(float(y.mean()), 4), "expected_literal": BASE_RATE_EXPECTED, "n_pos": int(y.sum())}
    k_label["PASS"] = bool(k_label["label_equals_md_indicator_derivation_every_row"] and k_label["gt_class_equal_both_calls"]
                           and k_label["base_rate_4dp"] == BASE_RATE_EXPECTED)
    log(f"K-LABEL {k_label}")
    if not k_label["PASS"]:
        H.jdump({"K-LABEL": k_label, "status": "WITHHELD"}, H.WORK / "MDFIX_FIT_WITHHELD.json")
        raise SystemExit("[mdfix_fit] WITHHELD: K-LABEL failed")
    # ---- held-out logits (lambda 0.1 fixed) and the s1a power table
    z_ho = heldout_logits(X, y, fold, LAMBDA)
    e_fit = int(len(np.unique(eid)))
    e_fit_decel = int(len(np.unique(eid[dec])))
    e_fit_cruise = int(len(np.unique(eid[cru])))
    rows = []
    for c in C_GRID:
        t = tau_scan_prefix(z_ho, A, c)
        g = z_ho >= t["tau"]
        md_new = np.where(dec, np.where(g, mdF, mdH), np.nan)
        fd_new = np.where(cru, np.where(g, fdF, fdH), np.nan)
        bm, bf = boot_se(md_new, mdH, eid), boot_se(fd_new, fdH, eid)
        se_md = bm["se_boot"] * math.sqrt(e_fit_decel / E_EVAL_MD)
        se_fd = bf["se_boot"] * math.sqrt(e_fit / E_EVAL_FD)
        se_fd_cruise_beside = bf["se_boot"] * math.sqrt(e_fit_cruise / E_EVAL_FD)
        power = phi(abs(bm["delta"]) / se_md - Z_G1) if se_md > 0 else 0.0
        fd_ub = bf["delta"] + 1.96 * se_fd
        rows.append({"c": c, "tau_heldout": t, "F_rate_heldout": float(g.mean()), "dMD_FIT": bm, "dFD_FIT": bf,
                     "SE_eval_MD": se_md, "SE_eval_FD": se_fd, "SE_eval_FD_if_E_FIT_cruise_beside": se_fd_cruise_beside,
                     "power_G1": power, "FD_upper_projected": fd_ub, "FD_upper_projected_if_E_FIT_cruise_beside": bf["delta"] + 1.96 * se_fd_cruise_beside,
                     "dMD_sign_negative": bool(bm["delta"] < 0),
                     "qualifies": bool(power >= POWER_MIN and fd_ub <= FD_UB_MAX)})
        log(f"c {c}: tau {t['tau']:.4f} F {g.mean():.4f} dMD {bm['delta']:+.4f} (SE_FIT {bm['se_boot']:.4f}) dFD {bf['delta']:+.4f} (SE_FIT {bf['se_boot']:.4f}) "
            f"power {power:.3f} FD_ub {fd_ub:+.4f} qualifies {rows[-1]['qualifies']}")
    chosen = next((r for r in rows if r["qualifies"]), None)
    power_doc = {"tool": "mdfix_fit.py (Phase A, s1a)", "prereg": seal, "role": "FIT side only, no eval139 read", "lambda": LAMBDA,
                 "estimator": f"paired episode-cluster bootstrap over FIT episodes (taniteval.ci draws, B {B}, seed {BOOT_SEED}); SE = std(ddof=1) of the deltas",
                 "projection": {"E_FIT": e_fit, "E_FIT_decel": e_fit_decel, "E_FIT_cruise_beside": e_fit_cruise, "E_eval_MD": E_EVAL_MD, "E_eval_FD": E_EVAL_FD,
                                "z_G1": Z_G1, "power_min": POWER_MIN, "FD_upper_max": FD_UB_MAX,
                                "E_FIT_reading": "literal: E_FIT = all FIT episodes (the prereg names E_FIT_decel for MD and E_FIT for FD); the "
                                                 "cruise-episode reading is reported beside and does not decide"},
                 "reference_H2cn_OOF": {"MD": A["md0"] / A["n_dec"], "FD": A["fd0"] / A["n_cru"], "n_GT_decel": A["n_dec"], "n_GT_cruise": A["n_cru"]},
                 "K-LABEL": k_label, "table": rows, "chosen_c": (chosen["c"] if chosen else None),
                 "outcome": ("POWERED" if chosen else "UNDERPOWERED"), "disclosures": DISCLOSURES,
                 "fit_rows_npz_sha256": H.sha256_file(H.SCRATCH / "FIT_ROWS.npz"), "recorded": time.strftime("%FT%T%z")}
    H.jdump(power_doc, POWER_JSON)
    log(f"s1a recorded -> {POWER_JSON.name}: {power_doc['outcome']} c={power_doc['chosen_c']}")
    if chosen is None:
        verdict = {"arm": "MDFIX8", "word": "UNDERPOWERED", "prereg": seal, "disclosures": DISCLOSURES, "power_table": rows,
                   "committed_consequence": "PREREG_HSGATE_MDFIX s5: no eval139 read; the refcv8 post-hoc progress line ENDS; the lever is refcv9's trained lon "
                                            "decision (PREREG_LONHEAD_C).", "eval139_read": False, "generated": time.strftime("%FT%T%z")}
        H.jdump(verdict, H.WORK / "result" / "MDFIX_VERDICT.json")
        md = ["# MDFIX_VERDICT -- UNDERPOWERED", "", *[f"* {d}" for d in DISCLOSURES], "",
              "| c | tau | F (held-out) | dMD_FIT | SE_eval_MD | power G1 | dFD_FIT | FD upper (proj.) | qualifies |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['c']} | {r['tau_heldout']['tau']:.4f} | {r['F_rate_heldout']:.4f} | {r['dMD_FIT']['delta']:+.4f} | {r['SE_eval_MD']:.4f} | "
                      f"{r['power_G1']:.3f} | {r['dFD_FIT']['delta']:+.4f} | {r['FD_upper_projected']:+.4f} | {r['qualifies']} |")
        md += ["", verdict["committed_consequence"]]
        (H.WORK / "result").mkdir(exist_ok=True)
        (H.WORK / "result" / "MDFIX_VERDICT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        log("UNDERPOWERED -> result/MDFIX_VERDICT.{json,md}; no freeze, no eval read")
        return 0
    # ---- freeze (c chosen and recorded above)
    c = chosen["c"]
    gate, tau, m = freeze_gate(X, y, A, z_ho, c, "MDFIX: 1[GT-decel AND lonclass(P_F)==decel AND lonclass(P_H-OOF)!=decel]")
    by = H.canonical_bytes(gate)
    FROZEN.write_bytes(by)
    sha = H.sha256_file(FROZEN)
    (H.WORK / "MDFIX_FROZEN.sha256").write_text(f"{sha}  MDFIX_FROZEN.json\n", encoding="utf-8")
    reload = json.loads(FROZEN.read_text(encoding="utf-8"))
    z_in = H.gate_logit(reload, X)
    g_in = z_in >= reload["tau"]
    # K-TAU: independent brute-force scan of the held-out logits; exact integer agreement on tau, MD and FD counts
    bf = tau_scan_bruteforce(z_ho, A, c)
    g_ho = z_ho >= tau["tau"]
    mdc, fdc = counts(g_ho, A)
    k_tau = {"primary": tau, "bruteforce": bf, "direct_mask_counts": [mdc, fdc],
             "PASS": bool(bf["tau"] == tau["tau"] and bf["md_count"] == tau["md_count"] == mdc and bf["fd_count"] == tau["fd_count"] == fdc
                          and reload["tau"] == tau["tau"])}
    # K-MUT: P_H condition dropped -> a DIFFERENT frozen gate
    y_mut = (dec & (pcF == "decel")).astype(np.float64)
    z_mut = heldout_logits(X, y_mut, fold, LAMBDA)
    gate_mut, tau_mut, _ = freeze_gate(X, y_mut, A, z_mut, c, "MUTANT: 1[GT-decel AND lonclass(P_F)==decel]")
    sha_mut = hashlib.sha256(H.canonical_bytes(gate_mut)).hexdigest()
    k_mut = {"label_differs_rows": int((y_mut != y).sum()), "mutant_gate_sha256": sha_mut, "frozen_sha256": sha, "mutant_tau": tau_mut["tau"],
             "mutant_F_rate_insample": float((H.gate_logit(gate_mut, X) >= gate_mut["tau"]).mean()), "PASS": bool(sha_mut != sha and len(sha_mut) == 64)}
    k_reload = bool(np.array_equal(H.gate_logit(reload, X), H.gate_logit(gate, X)) and reload["tau"] == gate["tau"])
    fit_in = {"F_rate": float(g_in.mean()), "md_count": counts(g_in, A)[0], "fd_count": counts(g_in, A)[1],
              "MD": counts(g_in, A)[0] / A["n_dec"], "FD": counts(g_in, A)[1] / A["n_cru"]}
    controls = {"K-LABEL": k_label, "K-TAU": k_tau, "K-MUT": k_mut, "K-RELOAD": k_reload,
                "carried_from_HSGATE_phase1": {"HSGATE_FIT_INFO_sha256": H.sha256_file(H.WORK / "HSGATE_FIT_INFO.json"),
                                               "ALL_PASS": fit_info_h.get("ALL_PASS"), "note": "K-FRESH-SS / K-JOIN / K-OOF on the SAME FIT rows (FIT_ROWS.npz) -- PREREG s1 'exactly PREREG_HSGATE s1'"},
                "FIT_ROWS_row_order_equals_load_fresh": True}
    controls["ALL_PASS"] = bool(k_label["PASS"] and k_tau["PASS"] and k_mut["PASS"] and k_reload and fit_info_h.get("ALL_PASS") is True)
    info = {"tool": "mdfix_fit.py", "prereg": seal, "frozen": {"path": str(FROZEN), "sha256": sha, "bytes": len(by)}, "c": c, "tau": gate["tau"], "lambda": LAMBDA,
            "power_record": {"path": str(POWER_JSON), "sha256": H.sha256_file(POWER_JSON)}, "fit": gate["fit"],
            "heldout_at_tau": {"F_rate": float(g_ho.mean()), "MD": mdc / A["n_dec"], "FD": fdc / A["n_cru"], "md_count": mdc, "fd_count": fdc},
            "insample_FIT_beside": fit_in, "reference_H2cn_OOF": power_doc["reference_H2cn_OOF"], "controls": controls,
            "ALL_PASS": controls["ALL_PASS"], "disclosures": DISCLOSURES, "fit_rows_npz_sha256": power_doc["fit_rows_npz_sha256"],
            "generated": time.strftime("%FT%T%z"), "wall_s": round(time.time() - t0, 1)}
    H.jdump(info, FIT_INFO)
    log(f"FROZEN {sha} c {c} tau {gate['tau']:.6f} | held-out F {g_ho.mean():.4f} MD {mdc / A['n_dec']:.4f} FD {fdc / A['n_cru']:.4f} | "
        f"in-sample F {fit_in['F_rate']:.4f} MD {fit_in['MD']:.4f} FD {fit_in['FD']:.4f} | K-TAU {k_tau['PASS']} K-MUT {k_mut['PASS']} ALL_PASS {controls['ALL_PASS']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
