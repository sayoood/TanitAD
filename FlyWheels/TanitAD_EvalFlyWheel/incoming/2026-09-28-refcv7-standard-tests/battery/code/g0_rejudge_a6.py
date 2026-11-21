"""POST-HOC A6 re-judge of an EXISTING 24-seed G0 artifact, with A6's measured inputs taken from a
SUPPLEMENT G0 run on the same checkpoint (`g0_refcv7.py --seeds 0 --mutations "" --diagnostic-arms ""
--skip-wrapper-control`, which runs seed 0 with the per-cell capture plus the `fp32_s0` arm).

    python g0_rejudge_a6.py --g0 D:/refcv7_eval_kit/battery/step30000/g0.json \
        --supp D:/refcv7_eval_kit/battery/g0a6supp_step30000/g0_supp.json \
        --metrics D:/refcv7_eval_kit/thor_reads/metrics_final_50400.jsonl \
        --out D:/refcv7_eval_kit/battery/step30000/g0_A6_posthoc.json

⛔ POST HOC by construction: A6 (draft `raw/AMENDMENT_A6_DRAFT.md`) was written AFTER the step-30,000
G0 numbers were seen, so this verdict is REPORTED and never gates step 30,000; that checkpoint's
registered verdicts (as registered FAIL, A2 FAIL, A5 FAIL) stand. Zero GPU (the supplement is the GPU).

Preconditions, asserted on CONTENT (else REFUSED, nothing judged):
  * the supplement ran on the SAME checkpoint md5 and the SAME window permutation (perm sha256);
  * its seed-0 row reproduces G0's stored seed-0 row on every key both carry within rel 1e-5 (5-dp
    granularity; the EXACT-match count is reported beside it) -- so A6's floor is measured on the
    replay G0 judged, not on a different one;
  * the in-run row read from `--metrics` equals the one G0 judged (every `eval_*` key);
  * the supplement carries the `fp32_s0` row and BOTH per-cell captures.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g0_refcv7 as G  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", required=True)
    ap.add_argument("--supp", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g0 = json.load(open(a.g0, encoding="utf-8"))
    sp = json.load(open(a.supp, encoding="utf-8"))
    out = {"tool": "g0_rejudge_a6.py", "POST_HOC": True,
           "note": "A6 was drafted AFTER this checkpoint's G0 numbers were seen; REPORTED, never a gate "
                   "result for this checkpoint", "g0": a.g0, "supp": a.supp,
           "a6_k": G.A6_K, "a6_threshold_terms": sorted(G.THRESHOLD_TARGET)}
    bad = []
    if g0.get("ckpt_md5") != sp.get("ckpt_md5"):
        bad.append(f"ckpt md5 {g0.get('ckpt_md5')} != supplement {sp.get('ckpt_md5')}")
    if g0.get("perm_sha256") != sp.get("perm_sha256"):
        bad.append("window permutation differs")
    s0 = ((sp.get("by_seed") or {}).get("0") or {}).get("row") or {}
    g_s0 = g0["by_seed"]["0"]["row"]
    common = [k for k in g_s0 if k in s0 and k.startswith("eval_")]

    def _rel(x, y):
        if G._isnull(x) or G._isnull(y):
            return 0.0 if (G._isnull(x) and G._isnull(y)) else float("inf")
        return abs(float(x) - float(y)) / max(abs(float(x)), 1e-9)
    exact = [k for k in common if s0[k] == g_s0[k]]
    rels = {k: _rel(g_s0[k], s0[k]) for k in common}
    diff = [k for k, r in rels.items() if r > 1e-5]
    out["s0_control"] = {"n_keys_compared": len(common), "n_exact": len(exact),
                         "n_beyond_rel_1e-5": len(diff), "max_rel": max(rels.values()) if rels else None,
                         "differ_first10": diff[:10],
                         "eval_tacv6_goal_conf_bce": [g_s0.get("eval_tacv6_goal_conf_bce"),
                                                      s0.get("eval_tacv6_goal_conf_bce")]}
    if not common or diff:
        bad.append(f"supplement seed 0 does not reproduce G0 seed 0 ({len(diff)} of {len(common)} keys "
                   f"beyond rel 1e-5)")
    a6 = sp.get("a6") or {}
    if not (a6.get(G.A6_ARM) or {}).get("row"):
        bad.append(f"supplement has no {G.A6_ARM} row ({str(a6.get(G.A6_ARM))[:200]})")
    if not ((a6.get("cells") or {}).get("s0") and (a6.get("cells") or {}).get(G.A6_ARM)):
        bad.append("supplement lacks a per-cell capture")
    step = int(g0["step"])
    ev = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in ev if r.get("step") == step and "eval_loss" in r]
    inrun = None
    if len(ev) != 1:
        bad.append(f"{len(ev)} eval rows at step {step}")
    else:
        inrun = ev[0]
        t_inrun = {k: v.get("inrun") for k, v in g0["verdict"]["terms"].items()}
        dif_in = [k for k, v in t_inrun.items() if inrun.get(k) != v]
        if dif_in:
            bad.append(f"in-run row differs from the one G0 judged on {len(dif_in)} keys")
    by_seed = {int(s): {"row": v["row"], "buffers_unchanged": v.get("buffers_unchanged", True)}
               for s, v in g0["by_seed"].items()}
    out["seeds"] = sorted(by_seed)
    if len(by_seed) < G.A5_MIN_SEEDS:
        bad.append(f"only {len(by_seed)} seeds (< {G.A5_MIN_SEEDS})")
    if bad:
        out["REFUSED"] = bad
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
        print(json.dumps({"REFUSED": bad}), flush=True)
        return 0
    rec = copy.deepcopy(g0)
    rec["a6"] = a6
    v6 = G.judge(inrun, by_seed, rec, amend="A6")
    v5 = G.judge(inrun, by_seed, rec, amend="A5")
    out["verdict_A5_recomputed"] = {"G0": v5["G0"], "reasons": v5["reasons"],
                                    "equals_banked": v5["G0"] == (g0.get("verdict") or {}).get("G0")}
    out["verdict_A6_POSTHOC"] = v6
    out["a6_arm_wall_s"] = (a6.get(G.A6_ARM) or {}).get("wall_s")
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({"G0_A6_POSTHOC": v6["G0"], "reasons": v6["reasons"][:10],
                      "a6_threshold_terms": v6.get("a6_threshold_terms"),
                      "a6_interval": v6.get("a6_interval"),
                      "a6_rescued": v6.get("a6_rescued"), "medians": v6.get("medians"),
                      "mutation_detection": {m: {k: d.get(k) for k in ("detected", "n_terms_out")}
                                             for m, d in v6["mutation_detection"].items()},
                      "A5_recomputed": out["verdict_A5_recomputed"]["G0"]},
                     indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
