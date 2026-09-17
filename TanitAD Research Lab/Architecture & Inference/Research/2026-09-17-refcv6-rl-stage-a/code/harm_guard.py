"""H-DDV2RL-2 — the pre-registered T1 harm guard, evaluated for the L1 arms.

⛔ `PREREG_DDV2_RL_VALIDATION.md` §13.3: *"FAIL-HARM if, for both RL seeds, T1 `ade_m` of
`os` vs BASE has a paired CI lower bound > 0. Otherwise 'no harm detected at this n'
(not a claim of no harm)."*

⭐ The invocation and the verdict logic are copied from the 2026-09-15 package's own
`analyse_validation.py`, not reimplemented, so the two packages' numbers are comparable
and the direction convention cannot drift. That code calls the tool with **A = the arm,
B = base**, and reads **FAIL-HARM when the UPPER bound of `B_minus_A` is < 0** for both
RL seeds. ⚠️ That is the SAME condition as the prereg's wording — `B_minus_A` is
`base − arm`, so `upper(base − arm) < 0` ⟺ `lower(arm − base) > 0` — and it is written
this way here only to match the sibling package bit for bit.

⛔ Reports all four metric families, never `ade_m` alone (the binding 2026-08-02 rule).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

WT = "C:/Users/Admin/tanitad-wt-rl-ddv2"
OUT = "C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917"
N_BOOT, SEED = 2000, 0


def run_paired(a: str, b: str) -> dict:
    tag = f"{a}__vs__{b}"
    outp = os.path.join(OUT, f"paired_{tag}.json")
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join([os.path.join(WT, "stack"),
                                           os.path.join(WT, "taniteval"), WT]))
    cmd = [sys.executable, os.path.join(WT, "taniteval", "tools", "paired_openloop.py"),
           "--a-dump", os.path.join(OUT, f"t1_{a}_dump"), "--a-name", a, "--a-arm", "os",
           "--b-dump", os.path.join(OUT, f"t1_{b}_dump"), "--b-name", b, "--b-arm", "os",
           "--floor", "ha0", "--n-boot", str(N_BOOT), "--seed", str(SEED), "--out", outp]
    p = subprocess.run(cmd, cwd=WT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    pathlib.Path(os.path.join(OUT, f"paired_{tag}.log")).write_text(
        (p.stdout or "") + "\n" + (p.stderr or ""), encoding="utf-8")
    # ⛔ THE ARTIFACT, NOT THE EXIT CODE.
    if not os.path.exists(outp) or os.path.getsize(outp) < 1000:
        return {"status": "FAILED", "returncode": p.returncode,
                "stderr_tail": (p.stderr or "")[-900:]}
    rows = []
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith(("ade_m", "fde_m", "LON_", "LAT_", "TAC_", "STR_")):
            parts = s.split()
            try:
                ci = s[s.index("["):s.index("]") + 1]
                rows.append({"metric": parts[0], "A_abs": float(parts[1]),
                             "B_abs": float(parts[2]), "floor": float(parts[3]),
                             "B_minus_A": float(parts[6]), "ci": ci, "sep": parts[-1]})
            except (ValueError, IndexError):
                continue
    n_line = [l for l in p.stdout.splitlines() if l.strip().startswith("n = ")]
    return {"status": "OK", "record": os.path.basename(outp),
            "n": n_line[0].strip() if n_line else None,
            "convention": ("B_minus_A = (B - floor) - (A - floor); for error metrics "
                           "POSITIVE means A (the ARM, listed first) is better"),
            "rows": rows}


def main() -> int:
    # ⛔ DEGRADE, DO NOT BLOCK. `L1-RL-s1`'s roll was dropped under the §14 budget order, so
    # the guard is UNEVALUABLE by construction. That is not a reason to deliver nothing:
    # whatever rolls DID complete still give a paired read, reported as a DIAGNOSTIC with a
    # direction and no criterion -- never as the verdict.
    if not os.path.isdir(os.path.join(OUT, "t1_base_dump")):
        print("ZZHARM-BLOCKED no BASE dump -- nothing is comparableZZ")
        return 2
    arms = [a for a in ("l1-rl-s0", "l1-rl-s1")
            if os.path.isdir(os.path.join(OUT, f"t1_{a}_dump"))]
    dropped = [a for a in ("l1-rl-s0", "l1-rl-s1") if a not in arms]
    res = {"_what": "PREREG_DDV2_RL_VALIDATION.md section 13.3 -- H-DDV2RL-2, the T1 harm guard.",
           "_tier": "T1 (self-action OPEN loop, PI ruling 2026-09-02 -- never 'closed loop')",
           "_evidence_class": "MEASURED (ours)",
           "_rule": ("FAIL-HARM iff, for BOTH RL seeds, the paired ade_m interval puts the arm "
                     "definitively worse than the cold start. Otherwise 'no harm detected at "
                     "this n' -- which is NOT a claim of no harm."),
           "_note_on_norl": ("L1-NORL-s0's T1 roll was DROPPED under the section 14 drop order "
                             "(item 2); it feeds only a section 13.4 no-criterion diagnostic and "
                             "the guard does not need it."),
           "paired": {}}
    res["_arms_rolled"] = arms
    res["_arms_dropped_for_budget"] = dropped
    worse = []
    for arm in arms:
        r = run_paired(arm, "base")
        res["paired"][f"{arm} vs base"] = r
        print(f"\n=== {arm} vs base ===  status {r['status']}  {r.get('n') or ''}")
        if r["status"] != "OK":
            print("   ", r.get("stderr_tail", "")[:400]); worse.append(None); continue
        for row in r["rows"]:
            print(f"   {row['metric']:26s} arm {row['A_abs']:9.4f} base {row['B_abs']:9.4f}"
                  f"  B-A {row['B_minus_A']:+9.4f} {row['ci']}  {row['sep']}")
        ade = [x for x in r["rows"] if x["metric"] == "ade_m"]
        if not ade:
            worse.append(None); continue
        hi = float(ade[0]["ci"].strip("[]").split(",")[1])
        worse.append(hi < 0)
        print(f"   -> ade_m upper bound {hi:+.4f}  => arm definitively worse: {hi < 0}")
    if dropped or any(w is None for w in worse) or len(worse) < 2:
        res["H_DDV2RL_2"] = "UNEVALUABLE"
        res["_why_unevaluable"] = (
            f"section 13.3 needs BOTH RL seeds against BASE; rolled {arms}, dropped {dropped} "
            f"under the section 14 drop order after the 2.85 h threshold was passed. "
            f"Reported UNEVALUABLE, NEVER passed.")
    elif all(worse):
        res["H_DDV2RL_2"] = "FAIL-HARM"
    else:
        res["H_DDV2RL_2"] = "no harm detected at this n"
    res["_2026_09_15_comparison"] = ("that package read FAIL-HARM on the RELEASE-form lever: "
                                     "RL-s0 +0.083 [0.056, 0.115] and NORL-s0 +0.081 "
                                     "[0.051, 0.116] m worse than the cold start.")
    pathlib.Path(os.path.join(OUT, "H_DDV2RL_2.json")).write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8")
    pathlib.Path("C:/Users/Admin/qland/pkgrl/raw/H_DDV2RL_2.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nZZH-DDV2RL-2 {res['H_DDV2RL_2']}ZZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
