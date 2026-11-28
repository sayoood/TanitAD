"""The stage-2 read of PREREG_T2_CRAWL (frozen 2c4c6557), written by the Master Mind on 2026-11-06 BEFORE any stage-2 rollout of run 2 was scored.

Inputs: SCORE2_POOLED.json (`t2v8f_run.py score`, via t2v8f_stage2_score.py: per seed X1 = C0(T2) - P10(T2), X2 = C0(T2) - C0(RR), BL = BLIND - C0(T2), M1 crawl share,
paired scene-cluster bootstrap B 2000 seed 0) and STAGE2_RESULT.json (the driver's gates and rollout status).
Rule: card section 6 as coded in t2v8f_score.outcome (h = 0.0267 / 0.0298). A gate that is not PASS -> GATE-FAIL; deciding cells not all DONE, or a contrast missing or in
error -> NOT-READ (never a result). Nothing here tunes on the data: every threshold is the card's.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1)
N_DECIDING = 420                                   # 42 launch scenes x {T2, RR} x {C0_s0, C0_s1, P10, ZERO, BLIND} (STAGE2_LAUNCH.md)


def read(pooled, result):
    gates = result.get("gates", {})
    gates_ok = all(gates.get(g) == "PASS" for g in ("preflight", "gate"))
    ro = result.get("steps", {}).get("rollouts", {})
    dec = ro.get("state", {}).get("deciding", {})
    out = {"card": "PREREG_T2_CRAWL 2c4c6557", "gates": gates, "overall": result.get("overall"), "deciding_state": dec}
    if not gates_ok:
        out.update(outcome="GATE-FAIL", reason="a gate is not PASS: the instrument is fixed first; no contrast is read")
        return out
    x1, x2, bl, why = {}, {}, {}, []
    n_dec = int(dec.get("done", 0)) + int(dec.get("skipped_done", 0))
    if dec.get("failed") or n_dec != N_DECIDING or any(p.get("pass") == "deciding" for p in ro.get("passes_skipped_disk", [])):
        why.append(f"deciding pass not complete: {n_dec} of {N_DECIDING} DONE, {len(dec.get('failed') or [])} failed")
    con = pooled.get("contrasts", {})
    for s in SEEDS:
        c = con.get(f"seed{s}")
        if not c or "error" in c:
            why.append(f"seed{s}: {'missing' if not c else c['error']}")
            continue
        x1[s], x2[s], bl[s] = c["X1"], c["X2"], c.get("BL")
    if why:
        out.update(outcome="NOT-READ", reason="; ".join(why))
        return out
    sys.path.insert(0, HERE)
    import t2v8f_score as S
    blv = bl if all(bl[s] is not None for s in SEEDS) else None
    out.update(S.outcome(x1, x2, blv, h=S.H_SEED, gates_ok=True))
    out["X1"], out["X2"], out["BL"], out["h"] = x1, x2, blv, S.H_SEED
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = read(json.load(open(a.pooled, encoding="utf-8")), json.load(open(a.result, encoding="utf-8")))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(r, f, indent=1)
    print(json.dumps({k: r.get(k) for k in ("outcome", "branch", "seed_gap", "reason", "notes")}))


if __name__ == "__main__":
    main()
