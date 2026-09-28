#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the ladder: per-job step statistics -> deltas vs R7 -> attribution + lever tables.

Reads <raw>/<job>/steps.jsonl + profile_*.json. For every plain job:
  * steady steps = index >= warm, not a conflict step (k % every == 0), not the final step;
  * median and IQR of the step wall; a percentile-bootstrap 90 % CI of the MEDIAN (over steps,
    2000 draws) -- this is a TIMING interval over steps of ONE run, not a training-variance
    claim; the replicate job (R7_plain_b2_rep) is the run-to-run control and is printed beside it;
  * the amortised conflict extra: (median conflict step - median steady step) / every.
Deltas are R7 minus the rung (= what switching the addition OFF saves), at b2 and b1, and the
per-sample slope (b2 - b1) when both exist.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def load(job_dir: Path):
    st = job_dir / "steps.jsonl"
    pj = sorted(job_dir.glob("profile_*.json"))
    if not st.exists() or not pj:
        return None
    prof = json.loads(pj[0].read_text(encoding="utf-8"))
    if prof.get("error"):
        return {"error": prof["error"]}
    rows = [json.loads(l) for l in st.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"prof": prof, "rows": rows}


def boot_ci(v, n=2000, seed=0):
    if len(v) < 3:
        return None
    r = random.Random(seed)
    ms = sorted(statistics.median([r.choice(v) for _ in v]) for _ in range(n))
    return [round(ms[int(0.05 * n)], 4), round(ms[int(0.95 * n)], 4)]


def stats(d):
    prof, rows = d["prof"], d["rows"]
    a = prof["args"]
    warm, steps = int(a["warm"]), int(a["steps"])
    spec_argv = prof["spec"]["argv"]
    every, cd_off = 10, False
    for i, t in enumerate(spec_argv):
        if t == "--conflict-every":
            every = int(spec_argv[i + 1])
        if t == "--conflict-detector" and spec_argv[i + 1] == "off":
            cd_off = True
    steady, conf = [], []
    for k, r in enumerate(rows):
        if k < warm or k >= steps - 1 or "wall" not in r:
            continue
        is_conf = (not cd_off) and (("conflict_measure" in r) if prof["mode"] == "timers"
                                    else (k % every == 0))
        (conf if is_conf else steady).append(r)
    # ⭐ COMPUTE = wall - data_wait. The dev box loads in the MAIN process (--workers 0: the v2
    # cache's pickled state drops `newest_frame_only`, so spawn workers crash on Windows), and a
    # boundary sync makes the GPU idle while it loads -- so the subtraction is exact.
    w = [r["wall"] - r.get("data_wait", 0.0) for r in steady]
    wc = [r["wall"] - r.get("data_wait", 0.0) for r in conf]
    out = {"n_steady": len(w), "median_s": round(statistics.median(w), 4) if w else None,
           "iqr_s": ([round(sorted(w)[len(w) // 4], 4), round(sorted(w)[(3 * len(w)) // 4], 4)]
                     if len(w) >= 4 else None),
           "ci90_median": boot_ci(w), "data_wait_median_s":
               round(statistics.median(r.get("data_wait", 0.0) for r in steady), 4) if steady else None,
           "max_mem_gb": max((r.get("max_mem_gb") or 0.0) for r in rows) if rows else None,
           "peak_mem_gb": prof.get("peak_mem_gb"),
           "n_conflict": len(wc), "conflict_median_s": round(statistics.median(wc), 4) if wc else None,
           "every": every, "cd_off": cd_off, "batch": int(a["batch"]), "rung": prof["rung"],
           "levers": prof.get("levers") or [], "edits": [e for e in a.get("edit") or []]}
    if wc and w:
        out["conflict_extra_s"] = round(statistics.median(wc) - statistics.median(w), 4)
        out["conflict_amortised_s"] = round(out["conflict_extra_s"] / every, 4)
        out["marginal_s_incl_conflict"] = round(statistics.median(w) + out["conflict_amortised_s"], 4)
    else:
        out["marginal_s_incl_conflict"] = out["median_s"]
    if prof["mode"] == "timers":
        keys = sorted({k for r in steady for k in r if not k.startswith("_") and
                       isinstance(r.get(k), (int, float)) and k not in ("wall", "max_mem_gb")})
        out["timers_median"] = {k: round(statistics.median(r.get(k, 0.0) for r in steady), 5)
                                for k in keys}
        out["timers_conflict_median"] = {k: round(statistics.median(r.get(k, 0.0) for r in conf), 5)
                                         for k in keys} if conf else {}
        out["timers_final_step"] = {k: v for k, v in rows[-1].items()} if rows else {}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = Path(a.raw)
    S = {}
    for jd in sorted(p for p in raw.iterdir() if p.is_dir()):
        d = load(jd)
        if d is None:
            continue
        if "error" in d:
            S[jd.name] = {"error": d["error"]}
            continue
        S[jd.name] = stats(d)
    res = {"jobs": S}

    def m(name, key="marginal_s_incl_conflict"):
        j = S.get(name)
        return None if (j is None or "error" in j) else j.get(key)
    deltas = {}
    for b in (2, 1):
        r7 = m(f"R7_plain_b{b}")
        if r7 is None:
            continue
        for name, j in S.items():
            if "error" in j or j.get("batch") != b or name.startswith("R7_") or "plain" not in name:
                continue
            v = j.get("marginal_s_incl_conflict")
            if v is not None:
                deltas.setdefault(name.replace(f"_plain_b{b}", ""), {})[f"b{b}"] = round(r7 - v, 4)
    for k, v in deltas.items():
        if "b1" in v and "b2" in v:
            v["slope_per_sample"] = round(v["b2"] - v["b1"], 4)
            v["fixed"] = round(v["b1"] - v["slope_per_sample"], 4)
            v["linear_b16"] = round(v["fixed"] + 16 * v["slope_per_sample"], 4)
    res["delta_R7_minus_rung"] = deltas
    rep = m("R7_plain_b2_rep")
    if rep is not None and m("R7_plain_b2") is not None:
        res["replicate_noise_s"] = round(m("R7_plain_b2") - rep, 4)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"deltas": deltas, "replicate_noise_s": res.get("replicate_noise_s"),
                      "medians": {k: (v.get("median_s"), v.get("marginal_s_incl_conflict"),
                                      v.get("peak_mem_gb")) if "error" not in v else v["error"][:80]
                                  for k, v in S.items()}}, indent=1))


if __name__ == "__main__":
    main()
