"""analyse_diag.py -- summarise raw/diag_arms.json into raw/diag_summary.json (and print it).

Re-applies wrapper_probe_r7._analyse's exact rule (floor = max rel(plain-repeat, plain-permuted); wrapper = max rel(
plain-micro13, plain-micro31); bar = max(3*floor, 1e-5)) to the CPU arms of each noise mode, and tabulates the
per-row output comparisons (vs_ref) for the keys that matter. Pure post-processing, no torch needed."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "raw" / "diag_arms.json", encoding="utf-8"))
A = D["arms"]
KEYS = ("cascade", "r8_subscore", "traj", "loss", "r8", "r8_alloc_l1", "r8_sat", "sel_v3", "law", "box3d", "agent",
        "map_hires", "goal_score_absmean")
OUTK = ("out.traj", "out.anchor_traj", "out.sel_score", "out.sel_score_v3", "out.r8_sub_logits", "out.anchor_logits",
        "out.r8_alloc", "out.r8_hyp_alloc", "out.pooled", "out.fmap_s16", "out.perception.box_slots.box3d",
        "out.perception.box_slots.presence_logit", "out.layer_u0_hat[0]", "out.layer_u0_hat[2]")


def rel(a, b):
    return abs(a - b) / max(abs(a), 1e-12)


def clause(mode):
    need = [f"{mode}_{x}" for x in ("plain", "repeat" if mode == "live" else "seed1", "perm", "micro13", "micro31")]
    if not all(n in A for n in need):
        return {"missing": [n for n in need if n not in A]}
    p = A[f"{mode}_plain"]["scalars"]
    rep = A[need[1]]["scalars"]
    per, m13, m31 = (A[f"{mode}_{x}"]["scalars"] for x in ("perm", "micro13", "micro31"))
    rows, over = {}, []
    terms = sorted(k for k, v in p.items() if abs(v) > 1e-6 and k != "goal_gate_grad"
                   and all(k in s for s in (rep, per, m13, m31)))
    for k in terms:
        fl = max(rel(p[k], rep[k]), rel(p[k], per[k]))
        wr = max(rel(p[k], m13[k]), rel(p[k], m31[k]))
        bar = max(3 * fl, 1e-5)
        rows[k] = {"plain": p[k], "floor": fl, "wrapper": wr, "bar": bar, "ok": wr <= bar}
        if wr > bar:
            over.append(k)
    return {"n_terms": len(terms), "terms_over_bar": over,
            "max_wrapper_rel": max((r["wrapper"] for r in rows.values()), default=None),
            "max_floor_rel": max((r["floor"] for r in rows.values()), default=None),
            "keys": {k: rows[k] for k in KEYS if k in rows},
            "worst5_wrapper": sorted(((r["wrapper"], k) for k, r in rows.items()), reverse=True)[:5]}


def scalar_cmp(a, b):
    if a not in A or b not in A:
        return None
    sa, sb = A[a]["scalars"], A[b]["scalars"]
    common = [k for k in sa if k in sb]
    diffs = {k: rel(sa[k], sb[k]) for k in common}
    n_exact = sum(1 for k in common if sa[k] == sb[k])
    return {"n_common": len(common), "n_bit_equal": n_exact, "max_rel": max(diffs.values()) if diffs else None,
            "worst5": sorted(((v, k) for k, v in diffs.items()), reverse=True)[:5],
            "keys": {k: [sa.get(k), sb.get(k), diffs.get(k)] for k in KEYS if k in diffs}}


S = {"source": "raw/diag_arms.json", "errors": D.get("errors"), "batch_meta": D.get("batch_meta"),
     "model": D.get("model"), "trunk_levers": D.get("trunk_levers"),
     "wall_s": {n: a.get("wall_s") for n, a in A.items()},
     "clause_rule_reapplied": {"zero": clause("zero"), "live": clause("live")},
     "control_zero_seed1_vs_zero_plain": scalar_cmp("zero_seed1", "zero_plain"),
     "control_live_repeat_vs_live_plain": scalar_cmp("live_repeat", "live_plain"),
     "noise_only_live_seed1_vs_live_plain": scalar_cmp("live_seed1", "live_plain"),
     "per_row_outputs": {}}
for n, a in A.items():
    if "vs_ref" in a:
        vr = a["vs_ref"]
        S["per_row_outputs"][n] = {"max_rel_all_keys": max((v.get("rel", 0.0) for v in vr.values()), default=None),
                                   "n_keys": len(vr), "top_rel": a.get("top_rel"),
                                   "keys": {k: vr[k] for k in OUTK if k in vr}}
json.dump(S, open(HERE / "raw" / "diag_summary.json", "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps({k: S[k] for k in ("errors", "clause_rule_reapplied", "control_zero_seed1_vs_zero_plain",
                                    "control_live_repeat_vs_live_plain", "noise_only_live_seed1_vs_live_plain")},
                 default=str, indent=0)[:9000])
for n, r in S["per_row_outputs"].items():
    print(n, "max_rel_all", r["max_rel_all_keys"], "top", [(k, f"{v:.2e}") for k, v in (r["top_rel"] or [])[:4]])
