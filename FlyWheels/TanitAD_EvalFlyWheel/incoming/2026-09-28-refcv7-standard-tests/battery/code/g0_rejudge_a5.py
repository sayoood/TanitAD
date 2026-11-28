"""POST-HOC A5 re-judge of an EXISTING 8-seed G0 artifact, extended with seeds from `g0_diag_r7.py`.

    python g0_rejudge_a5.py --g0 D:/refcv7_eval_kit/battery/step5000/g0.json \
        --diag D:/refcv7_eval_kit/battery/g0diag_step5000/diag.json \
        --metrics D:/refcv7_eval_kit/thor_reads/metrics_final_50400.jsonl \
        --out D:/refcv7_eval_kit/battery/step5000/g0_A5_posthoc.json

⛔ POST HOC by construction (SPEC AMENDMENT A5 item 6): A5 was written AFTER the step-5,000 numbers were
seen, so this verdict is REPORTED and never gates step 5,000; that checkpoint's registered verdicts (as
registered FAIL, A2 FAIL) stand. Zero GPU.

Preconditions, asserted on CONTENT (else REFUSED, nothing judged):
  * the diag ran on the SAME checkpoint md5 and the SAME window permutation (perm sha256);
  * its `s0` arm reproduces G0's stored seed-0 row on every key both carry within rel 1e-5 (the 5-dp
    rounding granularity of the row; the count of EXACT matches is reported beside it), so the extra
    seeds come from the same replay G0 is;
  * the in-run row read from `--metrics` equals the one G0 judged (every `eval_*` key).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g0_refcv7 as G  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g0 = json.load(open(a.g0, encoding="utf-8"))
    dg = json.load(open(a.diag, encoding="utf-8"))
    out = {"tool": "g0_rejudge_a5.py", "POST_HOC": True,
           "note": "SPEC A5 item 6: A5 was written after the step's numbers were seen; REPORTED, never a "
                   "gate result for this checkpoint", "g0": a.g0, "diag": a.diag}
    bad = []
    if g0.get("ckpt_md5") != dg.get("ckpt_md5"):
        bad.append(f"ckpt md5 {g0.get('ckpt_md5')} != diag {dg.get('ckpt_md5')}")
    if g0.get("perm_sha256") != dg.get("perm_sha256"):
        bad.append("window permutation differs")
    s0 = ((dg.get("arms") or {}).get("s0") or {}).get("row") or {}
    g_s0 = g0["by_seed"]["0"]["row"]
    common = [k for k in g_s0 if k in s0 and k.startswith("eval_")]
    def _rel(x, y):
        if G._isnull(x) or G._isnull(y):
            return 0.0 if (G._isnull(x) and G._isnull(y)) else float("inf")
        return abs(float(x) - float(y)) / max(abs(float(x)), 1e-9)
    exact = [k for k in common if s0[k] == g_s0[k]]
    rels = {k: _rel(g_s0[k], s0[k]) for k in common}
    diff = [k for k, r in rels.items() if r > 1e-5]          # 5-dp rounding granularity
    out["s0_control"] = {"n_keys_compared": len(common), "n_exact": len(exact),
                         "n_beyond_rel_1e-5": len(diff), "max_rel": max(rels.values()) if rels else None,
                         "differ_first10": diff[:10],
                         "eval_traj": [g_s0.get("eval_traj"), s0.get("eval_traj")]}
    if not common or diff:
        bad.append(f"s0 does not reproduce G0 seed 0 ({len(diff)} of {len(common)} keys beyond rel 1e-5)")
    step = int(g0["step"])
    ev = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in ev if r.get("step") == step and "eval_loss" in r]
    if len(ev) != 1:
        bad.append(f"{len(ev)} eval rows at step {step}")
    else:
        inrun = ev[0]
        t_inrun = {k: v.get("inrun") for k, v in g0["verdict"]["terms"].items()}
        dif_in = [k for k, v in t_inrun.items() if inrun.get(k) != v]
        if dif_in:
            bad.append(f"in-run row differs from the one G0 judged on {len(dif_in)} keys")
    extra = {k: v for k, v in (dg.get("arms") or {}).items() if k.startswith("seed")}
    by_seed = {int(s): {"row": v["row"], "buffers_unchanged": v.get("buffers_unchanged", True)}
               for s, v in g0["by_seed"].items()}
    for k, v in extra.items():
        sd = int(v.get("seed", int(k[4:])))
        if sd in by_seed:
            bad.append(f"diag seed {sd} duplicates a G0 seed")
        by_seed[sd] = {"row": v["row"], "buffers_unchanged": True}
    out["seeds"] = sorted(by_seed)
    if len(by_seed) < G.A5_MIN_SEEDS:
        bad.append(f"only {len(by_seed)} seeds (< {G.A5_MIN_SEEDS})")
    if bad:
        out["REFUSED"] = bad
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
        print(json.dumps({"REFUSED": bad}), flush=True)
        return 0
    v = G.judge(inrun, by_seed, g0, amend="A5")
    tr = [by_seed[s]["row"]["eval_traj"] for s in sorted(by_seed)]
    out["eval_traj_seed_means"] = {"K": len(tr), "mean": statistics.fmean(tr), "sd": statistics.stdev(tr),
                                   "sd_seeds_0_7": statistics.stdev(tr[:8]),
                                   "sd_seeds_8_23": statistics.stdev(tr[8:]),
                                   "inrun": inrun.get("eval_traj")}
    out["verdict_A5_POSTHOC"] = v
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({"G0_A5_POSTHOC": v["G0"], "reasons": v["reasons"][:10],
                      "eval_traj": out["eval_traj_seed_means"],
                      "mutation_detection": {m: {k: d.get(k) for k in ("detected", "n_terms_out")}
                                             for m, d in v["mutation_detection"].items()}},
                     indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
