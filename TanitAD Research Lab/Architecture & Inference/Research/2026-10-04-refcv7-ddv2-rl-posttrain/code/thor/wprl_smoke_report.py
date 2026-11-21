#!/usr/bin/env python
"""Summarise the stage-2 GPU smoke (gate inputs) from its ARTIFACTS only.

    python wprl_smoke_report.py <smoke_dir> --out smoke_report.json
"""
import argparse
import json
from pathlib import Path


def rows(d):
    p = Path(d) / "metrics.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("smoke_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.smoke_dir)
    rep = {"_evidence_class": "MEASURED (ours)", "_tier": "T0 training-side (gate smoke)"}
    seg, full = rows(S / "rl_seg"), rows(S / "rl_full")
    rep["resume"] = {
        "n_steps_seg": len(seg), "n_steps_full": len(full),
        "segments_seg": [json.loads(x) for x in (S / "rl_seg" / "segments.jsonl").read_text().splitlines()]
        if (S / "rl_seg" / "segments.jsonl").exists() else None,
        "windows_identical_every_step": len(seg) == len(full) and all(
            s["windows"] == f["windows"] for s, f in zip(seg, full)),
        "loss_abs_diff_per_step": [abs(s["loss"] - f["loss"]) for s, f in zip(seg, full)],
        "param_delta_abs_diff_per_step": [abs(s["param_delta_norm"] - f["param_delta_norm"]) for s, f in zip(seg, full)],
        "reward_mean_abs_diff_per_step": [abs(s["reward_mean"] - f["reward_mean"]) for s, f in zip(seg, full)]}
    off = rows(S / "rloff")
    rep["rloff"] = {"n": len(off), "coef_rl_abs_sum": [r["coef_rl_abs_sum"] for r in off],
                    "all_exactly_zero": all(r["coef_rl_abs_sum"] == 0.0 for r in off),
                    "param_moved": [r["param_delta_norm"] for r in off]}
    sh = rows(S / "rlshuf")
    rep["rlshuf"] = {"n": len(sh), "perms": [r["shuffle_perm"] for r in sh],
                     "derangement_every_step": all(r["shuffle_perm"] and all(int(p) != i for i, p in enumerate(r["shuffle_perm"])) for r in sh)}
    tm = rows(S / "timing32")
    rep["timing32"] = {"n": len(tm), "wall_s_cum": [r["wall_s"] for r in tm],
                       "timing": [r["timing"] for r in tm],
                       "s_per_step_from_wall": ([round((tm[-1]["wall_s"] - tm[0]["wall_s"]) / (len(tm) - 1), 2)]
                                                if len(tm) > 1 else None),
                       "first_step_wall_s": tm[0]["wall_s"] if tm else None}
    for name in ("rl_full", "rloff", "rlshuf", "timing32"):
        r = rows(S / name)
        if r:
            rep.setdefault("telemetry", {})[name] = {
                k: [round(x[k], 4) if isinstance(x[k], float) else x[k] for x in r]
                for k in ("loss", "il_m", "reward_mean", "human_pdms_mean", "frac_positive_after_bar",
                          "frac_constraint_fail", "cand_dac_mean", "cand_nc_mean", "grad_norm",
                          "grad_clipped", "param_delta_norm", "chain_endpoint_spread_m",
                          "human_dac1_frac", "human_nc1_frac", "n_dac_dead")}
            rep["telemetry"][name]["grad_groups_last"] = r[-1]["grad_groups"]
    for name in ("identity", "rl_full"):
        p = S / name / "identity.json"
        rep.setdefault("identity", {})[name] = json.loads(p.read_text()) if p.exists() else None
        if rep["identity"][name]:
            rep["identity"][name] = {k: v for k, v in rep["identity"][name].items() if k != "differ"} | {
                "n_differ": len(json.loads(p.read_text())["differ"])}
    for name in ("identity", "rl_seg", "rl_full", "rloff", "rlshuf", "timing32"):
        p = S / name / "check.json"
        if p.exists():
            c = json.loads(p.read_text())
            rep.setdefault("checks", {})[name] = {"verdict": c["verdict"], "failed": c["failed"],
                                                  "selftest": c.get("selftest_each_check_fires")}
    hb, hi = S / "heldout_base_s0_stride40.json", S / "heldout_identity_s0_stride40.json"
    if hb.exists() and hi.exists():
        b, i = json.loads(hb.read_text()), json.loads(hi.read_text())
        keys = [k for k in b["rows"][0] if k not in ("sha12", "t0")]
        diff = sum(1 for rb, ri in zip(b["rows"], i["rows"]) for k in keys if rb[k] != ri[k])
        rep["forward_identity"] = {"n_windows": len(b["rows"]), "n_episodes": b["n_episodes"],
                                   "n_cells_differ": diff, "n_cells": len(keys) * len(b["rows"]),
                                   "base_means": b["means"], "max_tick_recon_abs": b["max_tick_recon_abs"],
                                   "max_sel_match_abs": b["max_sel_match_abs"]}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
