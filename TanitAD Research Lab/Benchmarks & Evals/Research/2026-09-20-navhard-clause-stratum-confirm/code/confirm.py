"""The pre-registered analysis for `D-NAVSIM-CLAUSE-STRATUM-1` (b66e8cd), written BEFORE the
data it will consume exists, and validated against a fixture whose answers are already banked.

⭐ WHY IT IS WRITTEN FIRST: every analyst degree of freedom that survives until after the numbers
arrive is a place a result can be steered. Written now, the navhard run is a single invocation
with nothing left to choose.

⛔ SELF-TEST IS THE GATE. `--self-test` runs the identical code path on WARMUP, where the answers
are landed at 2142b55. Point estimates must match EXACTLY, CI bounds within measured Monte-Carlo
error (they are resampled quantiles, not deterministic), and the VERDICT exactly. If the
instrument cannot reproduce a known value it does not get to produce an unknown one.

⛔ AND IT CARRIES DELIBERATE-REGRESSION ARMS, because a guard that only ever passes is not
evidence: the self-test re-introduces the two defects that actually happened (joining on
`scene_token`; a one-arm clause detector) and FAILS if either is tolerated.

⛔ REFUSES a venue with fewer than 2 arms: the clause detector is "EP == 1.0 across every arm
scored", and with one arm that is not a detector, it is a filter on that arm.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import random
import re
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
HEX = re.compile(r"^[0-9a-f]{8,}$")
COL_EP = "ego_progress_stage_two"
COL_SCORE = "score"
B = 10000
SEED = 20260920            # fixed in the prereg, not chosen here

WARMUP_RAW = pathlib.Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                          "2026-09-19-navsim-refcv4b-bridge/raw")
WARMUP_ARMS = {"STOP": "score_STOP_zero.csv", "CV": "score_CV_official.csv",
               "A1": "score_A1_ego_cmd.csv", "ECHO": "score_ECHO_ha0_ext.csv",
               "A2": "score_A2_vision_pure.csv", "A4": "score_A4_blind_ego_cmd.csv"}
# banked at 2142b55 -- the known values this instrument must reproduce
EXPECT = {"n_scenes": 204, "n_fired": 37, "clusters_all": 16,
          "gap_not_fired": 0.1114, "ci_not_fired": [0.0292, 0.1821],
          "gap_fired": 0.2179, "wins_not_fired": (68, 83, 16)}

# ⛔ CI bounds are MONTE-CARLO, so the self-test asserts them within measured error and NOT to the
# dp they were printed at. MEASURED (mc_error.py, 12 independent streams at B=10,000 on the
# fixture): lo sd 0.00166, full spread 0.0052; hi sd 0.00066, spread 0.0025. The banked interval
# was therefore REPORTED to 4 dp and is stable to about 3.
# ⭐ What IS stable is the verdict: lo > 0 in 12/12 streams. So the self-test requires the bounds
# within tolerance AND the verdict to match exactly -- tightening the check that decides anything
# while loosening the one that cannot be met.
# ⛔ Tuning SEED until the bounds matched would have fitted the instrument to its own fixture.
CI_TOL = 0.006
CI_REPORT_DP = 3


def read_arm(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("token") or "")
            if HEX.match(t):
                out[t] = r
    return out


def num(row, col):
    try:
        return float(row.get(col, ""))
    except (TypeError, ValueError):
        return None


def analyse(arms: dict, table: pathlib.Path, label: str) -> dict:
    """arms: name -> {token: row}. table: the initial_token join/cluster table."""
    if len(arms) < 2:
        raise SystemExit(f"ZZREFUSE {label}: {len(arms)} arm(s). The clause detector needs >= 2 "
                         f"-- with one arm it is a filter on that arm, not a detector.")

    tbl = [ln.split(",") for ln in table.read_text().splitlines()[1:]]
    clust = {r[0]: r[2] for r in tbl}        # initial_token -> orig_scene
    toks = set.intersection(*(set(a) for a in arms.values())) & set(clust)
    toks = sorted(toks)
    if not toks:
        raise SystemExit(f"ZZABORT {label}: empty join. The key is initial_token, not scene_token.")

    fired = {t for t in toks if all(num(arms[a][t], COL_EP) == 1.0 for a in arms)}

    rng = random.Random(SEED)

    def boot(sel):
        by = collections.defaultdict(list)
        for t in sel:
            by[clust[t]].append(num(arms["STOP"][t], COL_SCORE) - num(arms["CV"][t], COL_SCORE))
        keys = list(by)
        if len(keys) < 2:
            return None, None, len(keys)
        ms = []
        for _ in range(B):
            draw = [x for k in rng.choices(keys, k=len(keys)) for x in by[k]]
            ms.append(st.mean(draw))
        ms.sort()
        return ms[int(0.025 * B)], ms[int(0.975 * B)], len(keys)

    res = {"_venue": label, "_n_arms_in_detector": len(arms),
           "_arms": sorted(arms), "n_scenes": len(toks), "n_fired": len(fired),
           "f_fired": round(len(fired) / len(toks), 4),
           "_estimator": f"paired bootstrap over orig_scene clusters, B={B}, seed={SEED}"}
    for name, sel in (("clause_NOT_fired", [t for t in toks if t not in fired]),
                      ("clause_FIRED", [t for t in toks if t in fired]),
                      ("ALL", toks)):
        d = [num(arms["STOP"][t], COL_SCORE) - num(arms["CV"][t], COL_SCORE) for t in sel]
        lo, hi, nk = boot(sel)
        res[name] = {
            "n_scenes": len(d), "n_CLUSTERS": nk,
            "mean_STOP_minus_CV": round(st.mean(d), 4) if d else None,
            "CI95": [round(lo, CI_REPORT_DP), round(hi, CI_REPORT_DP)] if lo is not None else None,
            # provenance must TRAVEL WITH ITS DATA: this was a hardcoded literal quoting the
            # 16-cluster warmup fixture, which is false on any other venue. Report the measured
            # cluster count and let the reader size the MC error, or say nothing.
            "_CI_basis": f"B={B} bootstrap over {nk} clusters",
            "separated": bool(lo is not None and (lo > 0 or hi < 0)),
            # ⛔ reported ALWAYS: on warmup the mean and the sign test DISAGREED
            "STOP_wins": sum(1 for x in d if x > 0),
            "CV_wins": sum(1 for x in d if x < 0),
            "ties": sum(1 for x in d if x == 0),
        }
    nf = res["clause_NOT_fired"]
    res["_VERDICT"] = (
        "CONFIRMED - clause is an AMPLIFIER, not the cause" if nf["separated"] and nf["mean_STOP_minus_CV"] > 0 else
        "REVERSED - the clause WAS the cause; warmup was wrong in sign" if nf["separated"] else
        "NOT REPLICATED - the warmup result does not survive; report as a failed replication")
    return res


def self_test() -> int:
    arms = {a: read_arm(WARMUP_RAW / fn) for a, fn in WARMUP_ARMS.items()}
    r = analyse(arms, HERE / "warmup_token_v0.csv", "warmup_two_stage (FIXTURE)")
    nf, fr = r["clause_NOT_fired"], r["clause_FIRED"]
    checks = [
        ("n_scenes", r["n_scenes"], EXPECT["n_scenes"]),
        ("n_fired", r["n_fired"], EXPECT["n_fired"]),
        ("clusters_ALL", r["ALL"]["n_CLUSTERS"], EXPECT["clusters_all"]),
        ("gap_not_fired", nf["mean_STOP_minus_CV"], EXPECT["gap_not_fired"]),
        ("ci_lo_not_fired", nf["CI95"][0], EXPECT["ci_not_fired"][0]),
        ("ci_hi_not_fired", nf["CI95"][1], EXPECT["ci_not_fired"][1]),
        ("gap_fired", fr["mean_STOP_minus_CV"], EXPECT["gap_fired"]),
        ("wins_not_fired", (nf["STOP_wins"], nf["CV_wins"], nf["ties"]),
         EXPECT["wins_not_fired"]),
    ]
    bad = []
    for n, got, exp in checks:
        ok = (abs(got - exp) <= CI_TOL) if n.startswith("ci_") else (got == exp)
        how = f"within {CI_TOL}" if n.startswith("ci_") else "exact"
        if not ok:
            bad.append((n, got, exp))
        print(f"  {'OK  ' if ok else 'FAIL'} {n:<18} got={got!r:<22} want={exp!r:<22} ({how})")

    # the verdict is what the prereg acts on, so it is asserted exactly
    want_v = "CONFIRMED - clause is an AMPLIFIER, not the cause"
    ok_v = r["_VERDICT"] == want_v
    print(f"  {'OK  ' if ok_v else 'FAIL'} {'VERDICT':<18} got={r['_VERDICT']!r} (exact)")
    if not ok_v:
        bad.append(("VERDICT", r["_VERDICT"], want_v))

    # ---- DELIBERATE REGRESSION ARMS: each re-introduces a defect that really happened ----
    import tempfile
    # (1) join on scene_token, the key that joined 0 of 204 and was landed at 2fc5bcc
    wrong = pathlib.Path(tempfile.gettempdir()) / "wrongkey.csv"
    rows = (HERE / "warmup_token_v0.csv").read_text().splitlines()
    wrong.write_text(rows[0] + "\n" + "\n".join(
        ",".join([r.split(",")[1]] + r.split(",")[1:]) for r in rows[1:]))
    try:
        analyse(arms, wrong, "MUTANT scene_token")
        print("  FAIL mutation-1        wrong join key was TOLERATED")
        bad.append(("mutation_wrong_join", "tolerated", "abort"))
    except SystemExit as e:
        ok = "empty join" in str(e)
        print(f"  {'OK  ' if ok else 'FAIL'} mutation-1        wrong join key -> {str(e)[:52]}")
        if not ok:
            bad.append(("mutation_wrong_join", str(e), "empty join abort"))
    # (2) a one-arm clause detector
    try:
        analyse({"CV": arms["CV"]}, HERE / "warmup_token_v0.csv", "MUTANT one-arm")
        print("  FAIL mutation-2        one-arm detector was TOLERATED")
        bad.append(("mutation_one_arm", "tolerated", "refuse"))
    except SystemExit as e:
        ok = "ZZREFUSE" in str(e)
        print(f"  {'OK  ' if ok else 'FAIL'} mutation-2        one-arm detector -> refused")
        if not ok:
            bad.append(("mutation_one_arm", str(e), "ZZREFUSE"))

    if bad:
        print("ZZSELFTEST-FAIL", bad)
        return 1
    print("ZZSELFTEST-OK -- point estimates exact, CI within measured Monte-Carlo error, "
          "verdict identical, and BOTH deliberate regressions rejected")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    d = pathlib.Path(sys.argv[1])
    arms = {p.stem: read_arm(p) for p in sorted(d.glob("*.csv"))}
    print(f"arms found: {sorted(arms)}")
    r = analyse(arms, HERE / "navhard_token_v0.csv", "navhard_two_stage")
    print(json.dumps(r, indent=1))
    (HERE / "confirm_navhard.json").write_bytes(json.dumps(r, indent=1).encode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
