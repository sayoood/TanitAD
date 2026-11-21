#!/usr/bin/env python3
"""SPEC_NAVTEST AMENDMENT 5 readout -- the NAVSIM v1 selection rule against the shipped rule, on tokens where no v1-rule
pick had ever been computed (the 43 logs outside W3's 200-token subset).

Input: the unchanged pipeline's run on the confirmation tokens (`eval/eval_checkpoint.py --name <name>`), which scored
the SHIPPED pick and dumped all 64 proposals with their logits (`proptable/<name>/proposals.npz`). This script
  G-A5a  recomputes the shipped rule from the dumped logits and must reproduce the dump's own pick on EVERY token;
  1      writes the v1-rule picks as their own seam and scores it with the unchanged harness (`score_navtest_refe.py`);
  G-A5b  on every token where both rules pick the same proposal the two scores must be IDENTICAL (poses equal);
  2      reads the per-token PDMS difference v1 - shipped with a paired log-cluster bootstrap over the confirmation
         logs (10,000 resamples, 95 %) and applies the rule committed in Amendment 5 before any of this was computed:
         REFUTED iff the upper bound < 0; CONFIRMED GAIN iff the lower bound > 0; else NO MEASURABLE DIFFERENCE.
  also   the same Delta for the v1 formula WITHOUT comfort, reported, not gating.
Writes proptable/<name>/a5_readout.json and eval/RESULT_A5_<name>.md; prints ZZA5 <name> <verdict> ...

    python eval/rule_confirm.py --name a5confirm_ep013 --tokens D:/Projects/TanitAD/data/refe_navtest/a5_confirm_tokens.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_checkpoint as EC  # noqa: E402
from proposal_table import HEAD_ORDER, read_csv, status_of  # noqa: E402


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def rules(logits):
    p = sig(logits.astype(np.float64))
    g = {k: p[..., HEAD_ORDER.index(k)] for k in ("NC", "DAC", "EP", "TTC", "C", "DDC")}
    return {"shipped": g["NC"] * g["DAC"] * g["DDC"] * (5 * g["EP"] + 5 * g["TTC"] + 4 * g["C"]) / 14,  # planner.py
            "v1": g["NC"] * g["DAC"] * (5 * g["EP"] + 5 * g["TTC"] + 2 * g["C"]) / 12,                  # NAVSIM v1
            "v1_noC": g["NC"] * g["DAC"] * (5 * g["EP"] + 5 * g["TTC"]) / 10}


def boot(delta, logs, n=10000, seed=20260926):
    ul = np.unique(logs)
    idx = {l: np.where(logs == l)[0] for l in ul}
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for b in range(n):
        ix = np.concatenate([idx[l] for l in rng.choice(ul, size=len(ul), replace=True)])
        means[b] = delta[ix].mean()
    return float(delta.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(lo, hi):
    return "REFUTED" if hi < 0 else ("CONFIRMED GAIN" if lo > 0 else "NO MEASURABLE DIFFERENCE")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--doc-dir", default=HERE, help="where RESULT_A5_<name>.md goes (a preflight passes a temp dir)")
    ap.add_argument("--out-json", default=None, help="default proptable/<name>/a5_readout.json")
    a = ap.parse_args()
    t0 = time.time()
    wd = os.path.join(EC.DATA, "proptable", a.name)
    D = np.load(os.path.join(wd, "proposals.npz"))
    tok = [str(t) for t in D["token"]]
    P, L, pick = D["proposals"], D["logits"], D["pick"]
    N = P.shape[0]
    ar = np.arange(N)
    R = {k: v.argmax(1) for k, v in rules(L).items()}
    ga = {"tokens": N, "reproduced": int((R["shipped"] == pick).sum())}
    ga["pass"] = ga["reproduced"] == N
    print(f"  G-A5a shipped rule reproduces the dump's pick on {ga['reproduced']}/{N} tokens", flush=True)
    if not ga["pass"]:
        print(f"ZZA5 {a.name} ABORT G-A5a"); return 1
    tl = json.load(open(a.tokens, encoding="utf-8"))["token_log"]
    logs = np.array([tl[t] for t in tok])
    landed = read_csv(f"{EC.DATA}/score/refe_{a.name}/refe_{a.name}.csv")
    shipped = np.array([landed[t][1] for t in tok], dtype=np.float64)
    out = {"name": a.name, "rule": "SPEC_NAVTEST AMENDMENT 5 (blob 41e25c57)", "N": N, "n_logs": int(len(np.unique(logs))),
           "G_A5a": ga, "shipped_pdms": round(100 * float(shipped.mean()), 3), "arms": {}}
    for arm in ("v1", "v1_noC"):
        label = f"refe_{a.name}_{arm}rule"
        seam = os.path.join(EC.DATA, "seams", f"{label}.npz")
        np.savez(seam, token=D["token"], fingerprint=D["fingerprint"], poses=P[ar, R[arm]].astype(np.float32),
                 sampling=D["sampling"], arm=np.array(f"REFe_{a.name}_{arm}rule"))
        log = os.path.join(wd, "logs", f"{arm}_score.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        rc, txt = EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", label, "--seam", seam, "--tokens", a.tokens,
                          "--out", f"{EC.DATA}/score"], HERE, dict(os.environ, PYTHONIOENCODING="utf-8"), log)
        st = status_of(txt)
        if not (st and st.get("status") == "PASS" and st.get("csv_valid_rows") == N):
            print(f"ZZA5 {a.name} ABORT scoring {arm}: {st and st.get('failures')}"); return 1
        rows = read_csv(f"{EC.DATA}/score/{label}/{label}.csv")
        mine = np.array([rows[t][1] for t in tok], dtype=np.float64)
        same = R[arm] == pick
        gb = {"same_pick_tokens": int(same.sum()),
              "max_abs_diff_same_pick": float(np.abs(mine[same] - shipped[same]).max()) if same.any() else None}
        gb["pass"] = bool(same.any() and gb["max_abs_diff_same_pick"] <= 1e-9)
        delta = 100 * (mine - shipped)
        m, ci = boot(delta, logs, a.boot)
        out["arms"][arm] = {"pdms": round(100 * float(mine.mean()), 3), "delta": round(m, 3),
                            "delta_ci95": [round(ci[0], 3), round(ci[1], 3)], "changed_tokens": int((~same).sum()),
                            "G_A5b": gb, "verdict": verdict(*ci) if arm == "v1" else "reported, not gating"}
        print(f"  {arm}: PDMS {100 * mine.mean():.2f} vs shipped {100 * shipped.mean():.2f}; delta {m:+.2f} "
              f"[{ci[0]:+.2f}, {ci[1]:+.2f}] over {(~same).sum()} changed tokens; G-A5b {gb}", flush=True)
        if not gb["pass"]:
            print(f"ZZA5 {a.name} ABORT G-A5b {arm}"); return 1
    v = out["arms"]["v1"]
    out["verdict"] = v["verdict"]
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out_json or os.path.join(wd, "a5_readout.json"), "w", encoding="utf-8"), indent=1)
    x = out["arms"]["v1_noC"]
    L_ = [f"# RESULT -- SPEC_NAVTEST AMENDMENT 5 -- `{a.name}` (match the selection rule to the benchmark, PI option 3)", "",
          "*Every number is read from the unchanged harness's per-token scores; none is typed. Written by "
          "`eval/rule_confirm.py`; the rule was fixed in Amendment 5 (blob `41e25c57`) before any v1-rule pick on these "
          "tokens was computed.*", "",
          f"**Confirmation tokens:** {N} from the {out['n_logs']} navtest logs that W3's 200-token subset does not touch; "
          f"snapshot after epoch 13. **Gates:** G-A5a the shipped rule reproduces the planner's own pick on "
          f"{ga['reproduced']}/{N} tokens; G-A5b identical picks score identically "
          f"(max |diff| {v['G_A5b']['max_abs_diff_same_pick']}).", "",
          f"## Verdict: **{out['verdict']}**", "",
          "| selection rule | PDMS | vs shipped, PDMS points [95 %] | tokens whose pick changed |", "|---|---|---|---|",
          f"| shipped (NAVSIM v2 EPDMS shape) | {out['shipped_pdms']:.2f} | -- | -- |",
          f"| **NAVSIM v1 formula (the rule under test)** | **{v['pdms']:.2f}** | **{v['delta']:+.2f} "
          f"[{v['delta_ci95'][0]:+.2f}, {v['delta_ci95'][1]:+.2f}]** | {v['changed_tokens']} |",
          f"| v1 without comfort (reported, not gating) | {x['pdms']:.2f} | {x['delta']:+.2f} "
          f"[{x['delta_ci95'][0]:+.2f}, {x['delta_ci95'][1]:+.2f}] | {x['changed_tokens']} |", "",
          "Rule committed in Amendment 5: REFUTED iff the upper bound is below 0 (keep the shipped rule, back to the PI); "
          "otherwise the v1 formula becomes REFe's selection rule for every evaluation from this amendment on -- "
          "CONFIRMED GAIN iff the lower bound is above 0, NO MEASURABLE DIFFERENCE iff the interval straddles 0. Paired "
          f"log-cluster bootstrap over the {out['n_logs']} logs, {a.boot:,} resamples."]
    open(os.path.join(a.doc_dir, f"RESULT_A5_{a.name}.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L_) + "\n")
    print(f"ZZA5 {a.name} {out['verdict']} delta={v['delta']:+.3f} [{v['delta_ci95'][0]:+.3f}, {v['delta_ci95'][1]:+.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
