#!/usr/bin/env python
"""E-ARCH-TSC-2 — compute the pre-registered reads R1–R6 (+R7 parse) from the raw
train_log.jsonl of the three arms. Every number printed here is read from the raw
row at the named step; nothing is smoothed unless labelled 'supplementary'.

usage: python reads.py --bp <dir> --e <dir> --ap <dir> [--r7 <dir> --r7-log <file>] --out reads.json
"""
import argparse
import json
import math
import os
import re


def rows(d):
    p = os.path.join(d, "train_log.jsonl")
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def at(rs, step):
    for r in rs:
        if r["step"] == step:
            return r
    raise KeyError(f"no row at step {step}; have {[r['step'] for r in rs]}")


def series(rs, key):
    return [(r["step"], r[key]) for r in rs if r.get(key) is not None]


def n_down(rs, key):
    v = [r[key] for r in rs]
    d = [b - a for a, b in zip(v, v[1:])]
    return sum(1 for x in d if x < 0), len(d)


def s_per_step(rs, lo=10, hi=250):
    a, b = at(rs, lo), at(rs, hi)
    return (b["elapsed_s"] - a["elapsed_s"]) / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", required=True)
    ap.add_argument("--e", default=None, help="omit if arm E was NOT run (SPEC STOP rule after a VOID R1)")
    ap.add_argument("--ap", required=True)
    ap.add_argument("--r7", default=None, help="R7 resume run dir (copy of B')")
    ap.add_argument("--r7-log", default=None, help="R7 stdout/stderr log")
    ap.add_argument("--final", type=int, default=250)
    ap.add_argument("--ap-final", type=int, default=None,
                    help="A' read step if the arm was TRUNCATED (process killed before 250); "
                         "stated in the verdict")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = a.final
    FA = a.ap_final if a.ap_final is not None else F
    R = {}
    arms = {"B_prime": rows(a.bp), "E": rows(a.e) if a.e else None, "A_prime": rows(a.ap)}
    for nm, rs in arms.items():
        if rs is None:
            R[f"{nm}_status"] = "NOT RUN — SPEC/brief STOP rule: R1 read VOID (< 3), E uninterpretable"
            continue
        Fn = FA if nm == "A_prime" else F
        R[f"{nm}_n_rows"] = len(rs)
        R[f"{nm}_steps"] = [r["step"] for r in rs]
        R[f"{nm}_s_per_step_10_to_{Fn}"] = round(s_per_step(rs, 10, Fn), 3)
        R[f"{nm}_elapsed_s_at_{Fn}"] = at(rs, Fn)["elapsed_s"]
        if Fn != F:
            R[f"{nm}_TRUNCATED"] = (f"read at step {Fn}, not {F}: the process was killed before "
                                    f"step {F} (last banked row {rs[-1]['step']})")
        for key in ("tgt_std_tac", "tgt_std_str", "tgt_std_op", "adapter_std",
                    "loss", "loss_feat_op", "loss_feat_tac", "loss_feat_str",
                    "participation", "grad_norm", "ema_decay"):
            R[f"{nm}_{key}_series"] = series(rs, key)
    bp, e, apr = arms["B_prime"], arms["E"], arms["A_prime"]
    w_tac = 0.5   # RefAV1Config.w_feat_tac (refa_v1.py:197, HEAD blob 8264ded8)

    # ---- R1: B' tgt_std_tac(F)/tgt_std_tac(1)
    r1 = at(bp, F)["tgt_std_tac"] / at(bp, 1)["tgt_std_tac"]
    R["R1"] = {"read": "B' tgt_std_tac(F)/tgt_std_tac(1)",
               "tgt_std_tac_1": at(bp, 1)["tgt_std_tac"], f"tgt_std_tac_{F}": at(bp, F)["tgt_std_tac"],
               "ratio": r1,
               "verdict": ("CONFIRMED (> 10: rig reproduces the live inflation)" if r1 > 10
                           else "VOID (< 3: rig insensitive)" if r1 < 3
                           else "NEITHER BRANCH (3 <= ratio <= 10)")}
    def share(r):
        return w_tac * r["loss_feat_tac"] / r["loss"]
    if e is None:
        void = "VOID — arm E NOT RUN (STOP rule: R1 < 3, the rig did not reproduce the inflation)"
        R["R2"] = {"read": "E tgt_std_tac(F)/tgt_std_tac(1)", "verdict": void}
        R["R3"] = {"read": "E tactical loss share at F", "verdict": void,
                   "share_Bprime_at_1": share(at(bp, 1)), "share_Bprime_at_F": share(at(bp, F)),
                   "supplementary_mean_share_last5_Bprime": sum(share(r) for r in bp[-5:]) / 5,
                   "supplementary_max_share_Bprime_any_row": max(share(r) for r in bp)}
    else:
        # ---- R2: E tgt_std_tac(F)/tgt_std_tac(1)
        r2 = at(e, F)["tgt_std_tac"] / at(e, 1)["tgt_std_tac"]
        R["R2"] = {"read": "E tgt_std_tac(F)/tgt_std_tac(1)",
                   "tgt_std_tac_1": at(e, 1)["tgt_std_tac"], f"tgt_std_tac_{F}": at(e, F)["tgt_std_tac"],
                   "ratio": r2,
                   "verdict": ("CONFIRMED (<= 2)" if r2 <= 2
                               else "REFUTED (> 5: EMA does not pin the scale)" if r2 > 5
                               else "NEITHER BRANCH (2 < ratio <= 5)")}
        # ---- R3: E tactical loss share at F
        r3 = share(at(e, F))
        R["R3"] = {"read": "E tactical loss share at F = 0.5*loss_feat_tac/loss",
                   "loss_feat_tac": at(e, F)["loss_feat_tac"], "loss": at(e, F)["loss"],
                   "share": r3,
                   "share_Bprime_at_F": share(at(bp, F)),
                   "share_Bprime_at_1": share(at(bp, 1)), "share_E_at_1": share(at(e, 1)),
                   "supplementary_mean_share_last5_E": sum(share(r) for r in e[-5:]) / 5,
                   "supplementary_mean_share_last5_Bprime": sum(share(r) for r in bp[-5:]) / 5,
                   "verdict": ("CONFIRMED (< 10 %)" if r3 < 0.10
                               else "REFUTED (>= 25 %)" if r3 >= 0.25
                               else "NEITHER BRANCH (10 % <= share < 25 %)")}
    # ---- R4: A' adapter_std falls > 15 %
    a1, aF = at(apr, 1)["adapter_std"], at(apr, FA)["adapter_std"]
    r4 = aF / a1 - 1.0
    nd, nt = n_down(apr, "adapter_std")
    R["R4"] = {"read": f"A' adapter_std({FA})/adapter_std(1) - 1"
                       + (f"  [TRUNCATED: committed step {F}, read at {FA}]" if FA != F else ""),
               "adapter_std_1": a1, f"adapter_std_{FA}": aF, "rel_change": r4,
               "mean_last3_rows": sum(r["adapter_std"] for r in apr[-3:]) / 3,
               "deltas_down": f"{nd}/{nt}",
               "min_adapter_std": min(r["adapter_std"] for r in apr),
               "Bprime_adapter_std_1": at(bp, 1)["adapter_std"],
               f"Bprime_adapter_std_{F}": at(bp, F)["adapter_std"],
               **({"E_adapter_std_1": at(e, 1)["adapter_std"],
                   f"E_adapter_std_{F}": at(e, F)["adapter_std"]} if e is not None else {}),
               "verdict": ("CONFIRMED (falls > 15 %: collapse reproduced)" if r4 < -0.15
                           else "REFUTED (flat/rising: rig insensitive to collapse)" if r4 >= -0.02
                           else "NEITHER BRANCH (falls 2-15 %)")}
    # ---- R5: tgt_std_op ~ 1.0 throughout in B' and E
    r5 = {}
    for nm, rs in (("B_prime", bp), ("E", e)):
        if rs is None:
            continue
        v = [r["tgt_std_op"] for r in rs]
        f5, l5 = sum(v[:5]) / 5, sum(v[-5:]) / 5
        r5[nm] = {"min": min(v), "max": max(v), "first": v[0], "last": v[-1],
                  "mean": sum(v) / len(v), "mean_first5": f5, "mean_last5": l5,
                  "drift_last5_vs_first5": l5 / f5 - 1.0,
                  "max_abs_dev_from_1": max(abs(x - 1.0) for x in v)}
    worst_drift = max(abs(r5[k]["drift_last5_vs_first5"]) for k in r5)
    worst_dev = max(r5[k]["max_abs_dev_from_1"] for k in r5)
    R["R5"] = {"read": "tgt_std_op in B' (and E if run), all rows", **r5,
               "tolerance_note": "SPEC says '~ 1.0 throughout' vs 'drifts', no tolerance given. "
                                 "Operationalised post-hoc (stated): DRIFT = |mean(last 5 rows)/"
                                 "mean(first 5 rows) - 1| > 5 %; per-row deviation from 1.0 is "
                                 "reported but single-window rows (bs 1) carry scene-to-scene noise",
               "verdict": ("CONFIRMED (~1.0 throughout: no drift, last-5 vs first-5 within 5 %; "
                           f"every row within {worst_dev * 100:.1f} % of 1.0)" if worst_drift <= 0.05
                           else "VOID (drifts > 5 %: instrument fault)")}
    # ---- R6: E loss_feat_op(F) vs B'
    if e is None:
        R["R6"] = {"read": "E loss_feat_op(F) vs B'", "verdict": "VOID — arm E NOT RUN (STOP rule)",
                   "Bprime_loss_feat_op_1": at(bp, 1)["loss_feat_op"],
                   f"Bprime_loss_feat_op_{F}": at(bp, F)["loss_feat_op"],
                   "supplementary_mean_last5_Bprime": sum(r["loss_feat_op"] for r in bp[-5:]) / 5}
    else:
        lb, le = at(bp, F)["loss_feat_op"], at(e, F)["loss_feat_op"]
        r6 = le / lb - 1.0
        R["R6"] = {"read": "E loss_feat_op(F) vs B' loss_feat_op(F)",
                   "Bprime": lb, "E": le, "rel_diff_E_minus_Bprime": r6,
                   "supplementary_mean_last5_Bprime": sum(r["loss_feat_op"] for r in bp[-5:]) / 5,
                   "supplementary_mean_last5_E": sum(r["loss_feat_op"] for r in e[-5:]) / 5,
                   "verdict": ("CONFIRMED (within 10 % of B')" if abs(r6) <= 0.10
                               else "REFUTED (> 25 % worse)" if r6 > 0.25
                               else "NEITHER BRANCH")}
    # ---- extra, not pre-registered
    live = {nm: rs for nm, rs in arms.items() if rs is not None}
    R["extra"] = {
        "Bprime_tgt_std_tac_max_row": max(bp, key=lambda r: r["tgt_std_tac"])["step"],
        "Bprime_tgt_std_tac_max_ratio": max(r["tgt_std_tac"] for r in bp) / at(bp, 1)["tgt_std_tac"],
        "Bprime_tgt_std_str_ratio": at(bp, F)["tgt_std_str"] / at(bp, 1)["tgt_std_str"],
        "Bprime_adapter_std_rel_change": at(bp, F)["adapter_std"] / at(bp, 1)["adapter_std"] - 1.0,
        "Aprime_read_step": FA,
        "Aprime_tgt_std_tac_ratio": at(apr, FA)["tgt_std_tac"] / at(apr, 1)["tgt_std_tac"],
        "Aprime_tgt_std_str_ratio": at(apr, FA)["tgt_std_str"] / at(apr, 1)["tgt_std_str"],
        "Aprime_tgt_std_op_first_last": (at(apr, 1)["tgt_std_op"], at(apr, FA)["tgt_std_op"]),
        "Aprime_loss_first_last": (at(apr, 1)["loss"], at(apr, FA)["loss"]),
        "Aprime_loss_feat_op_first_last": (at(apr, 1)["loss_feat_op"], at(apr, FA)["loss_feat_op"]),
        "Bprime_loss_feat_tac_ratio": at(bp, F)["loss_feat_tac"] / at(bp, 1)["loss_feat_tac"],
        "step1_identity_Bprime_vs_Aprime_tgt_std_tac": (at(bp, 1)["tgt_std_tac"], at(apr, 1)["tgt_std_tac"]),
        "participation_min": {nm: min(r["participation"] for r in rs) for nm, rs in live.items()},
        "participation_first_last": {nm: (rs[0]["participation"], rs[-1]["participation"]) for nm, rs in live.items()},
        "grad_norm_max": {nm: max(r["grad_norm"] for r in rs) for nm, rs in live.items()},
    }
    if e is not None:
        R["extra"].update({
            "E_tgt_std_str_ratio": at(e, F)["tgt_std_str"] / at(e, 1)["tgt_std_str"],
            "E_loss_feat_tac_ratio": at(e, F)["loss_feat_tac"] / at(e, 1)["loss_feat_tac"],
            "E_ema_decay_first_last": (at(e, 10).get("ema_decay"), at(e, F).get("ema_decay")),
            "step1_identity_Bprime_vs_E": {k: (at(bp, 1)[k], at(e, 1)[k]) for k in
                                           ("loss", "loss_feat_op", "loss_feat_tac", "loss_feat_str",
                                            "tgt_std_tac", "tgt_std_str", "tgt_std_op", "adapter_std")},
        })
    # ---- R7 parse
    if a.r7 and a.r7_log:
        txt = open(a.r7_log, encoding="utf-8", errors="replace").read()
        m_res = re.search(r"resumed from step (\d+)", txt)
        m_err = re.search(r"(RuntimeError: Error\(s\) in loading state_dict[^\n]*)", txt)
        miss = re.search(r"Missing key\(s\) in state_dict: (.*)", txt)
        r7rows = rows(a.r7) if os.path.exists(os.path.join(a.r7, "train_log.jsonl")) else []
        post = [r for r in r7rows if r["step"] > F]
        R["R7"] = {"resumed_line": m_res.group(0) if m_res else None,
                   "error_line": m_err.group(1)[:400] if m_err else None,
                   "missing_keys_excerpt": (miss.group(1)[:600] if miss else None),
                   "n_missing_keys": (len(re.findall(r'"ema\.', miss.group(1))) if miss else None),
                   "rows_after_resume": post,
                   "verdict": ("ACCEPTED: OFF checkpoint loaded under --ema-targets and trained on"
                               if (m_res and post) else
                               "REFUSED on missing ema.* keys (strict load_state_dict)" if m_err
                               else "UNVERIFIED — neither a resume line nor a load error found")}
    json.dump(R, open(a.out, "w", encoding="utf-8"), indent=1)
    for k in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        if k in R:
            print(k, "->", R[k]["verdict"])
    print("s/step:", {k: v for k, v in R.items() if "s_per_step" in k})
    print("extra:", json.dumps(R["extra"], indent=1)[:1500])


if __name__ == "__main__":
    main()
