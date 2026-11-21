"""The stage-2 read of PREREG_T2_CRAWL (frozen 2c4c6557), written by the Master Mind on 2026-11-06 BEFORE any stage-2 rollout of run 2 was scored.

Inputs: SCORE2_POOLED.json (`t2v8f_run.py score`, via t2v8f_stage2_score.py: per seed X1 = C0(T2) - P10(T2), X2 = C0(T2) - C0(RR), BL = BLIND - C0(T2), M1 crawl share,
paired scene-cluster bootstrap B 2000 seed 0) and STAGE2_RESULT.json (the driver's gates and rollout status).
Rule: card section 6 as coded in t2v8f_score.outcome (h = 0.0267 / 0.0298). A gate that is not PASS -> GATE-FAIL; deciding cells not all DONE, or a contrast missing or in
error -> NOT-READ (never a result). Nothing here tunes on the data: every threshold is the card's.
Re-pinned 2026-11-06 19:3x with rule R-SCN (AMENDMENT_T2_SCENE_REFUSAL.md, frozen 2167a7eb), before any run-2 row was computed: with --exclusions (the
output of scene_exclusions.py, verdict ATTRIBUTED), the failed cells must be exactly every cell of the excluded scenes, the deciding count required is
10 x (launch scenes - excluded), and no pooled group may pair on an excluded scene. Without --exclusions the rule is the first pin's (420 of 420).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1)
N_DECIDING = 420                                   # 42 launch scenes x {T2, RR} x {C0_s0, C0_s1, P10, ZERO, BLIND} (STAGE2_LAUNCH.md)


CELLS_PER_SCENE = {"deciding": 10, "beside": 4}       # {T2, RR} x {C0_s0, C0_s1, P10, ZERO, BLIND}; {T2, RR} x {DAGOACC_s0, DAGOACC_s1}


def check_exclusions(exc, result, pooled):
    """R-SCN: -> (n_deciding_required, [problems], {scene8: reason})."""
    probs, n_man = [], int(result.get("n_scenes_manifest", 42))
    if exc.get("verdict") != "ATTRIBUTED" or exc.get("problems"):
        probs.append(f"exclusions not ATTRIBUTED: {exc.get('verdict')} {exc.get('problems')}")
    if exc.get("manifest_sha256") != result.get("manifest_sha256"):
        probs.append("exclusions were built from another stage result (manifest_sha256 differs)")
    ex = exc.get("excluded", {})
    st = result.get("steps", {}).get("rollouts", {}).get("state", {})
    for p, k in CELLS_PER_SCENE.items():
        failed = st.get(p, {}).get("failed") or []
        if {f.split("|")[0] for f in failed} - set(ex):
            probs.append(f"{p}: failed cells outside the excluded scenes: {sorted({f.split('|')[0] for f in failed} - set(ex))}")
        for s in ex:
            n = sum(1 for f in failed if f.split("|")[0] == s)
            if n != k:
                probs.append(f"{p}: excluded scene {s} has {n} failed cells, not all {k}")
    bad_idx = {int(v["idx"]) for v in ex.values() if v.get("idx") is not None}
    if len(bad_idx) != len(ex):
        probs.append("an excluded scene carries no scene index")
    for g, v in pooled.get("groups", {}).items():
        on = {int(pk[0]) for pk in v.get("pair_keys", [])} & bad_idx
        if on:
            probs.append(f"pooled group {g} pairs on excluded scenes {sorted(on)}")
    c0 = pooled.get("groups", {}).get("C0_s0|T2|objects")
    if c0 is not None and len({int(pk[0]) for pk in c0.get("pair_keys", [])}) != n_man - len(ex):
        probs.append(f"C0_s0|T2|objects covers {len({int(pk[0]) for pk in c0.get('pair_keys', [])})} scenes, not {n_man} - {len(ex)}")
    return CELLS_PER_SCENE["deciding"] * (n_man - len(ex)), probs, {s: v.get("reason") for s, v in sorted(ex.items())}


def read(pooled, result, exclusions=None):
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
    if exclusions is None:
        need, failed_ok = N_DECIDING, not dec.get("failed")
    else:
        need, probs, out["excluded"] = check_exclusions(exclusions, result, pooled)
        out["rule"] = "R-SCN (AMENDMENT_T2_SCENE_REFUSAL.md, frozen 2167a7eb)"
        why += probs
        failed_ok = not probs
    out["deciding_required"] = need
    if not failed_ok or n_dec != need or any(p.get("pass") == "deciding" for p in ro.get("passes_skipped_disk", [])):
        why.append(f"deciding pass not complete: {n_dec} of {need} DONE, {len(dec.get('failed') or [])} failed")
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
    ap.add_argument("--exclusions", default=None, help="EXCLUSIONS2.json from scene_exclusions.py (rule R-SCN)")
    a = ap.parse_args(argv)
    exc = json.load(open(a.exclusions, encoding="utf-8")) if a.exclusions else None
    r = read(json.load(open(a.pooled, encoding="utf-8")), json.load(open(a.result, encoding="utf-8")), exc)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(r, f, indent=1)
    print(json.dumps({k: r.get(k) for k in ("outcome", "branch", "seed_gap", "reason", "notes")}))


if __name__ == "__main__":
    main()
