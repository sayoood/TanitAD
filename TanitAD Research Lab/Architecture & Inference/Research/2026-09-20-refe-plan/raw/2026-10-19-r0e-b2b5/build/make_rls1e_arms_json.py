#!/usr/bin/env python3
"""Write eval/rls1e_arms.json: the four arms of R0e-pick (PREREG_R0E_PICK 2.1), DERIVED from eval/rls1d_arms.json (R0d, R1o) and nothing else -- NOT from refe/launch_gate.py, so that the gate's own copy of
R0d's argv (launch_gate.RLS1E_BASE_ARGV) and this file are two independent derivations that test_launch_gate.py then requires to agree.

usage: make_rls1e_arms_json.py <PK dir>     (writes <PK>/eval/rls1e_arms.json, LF; prints the argv sha256 of each arm and the self-check verdicts)

  R0e-id    = R0d's argv, own paths, + `--epo-con-temp 0.1`   (seed 0; the identity control, run to u1000: PI 1c)
  R0e-pick  = R0d's argv, own paths, + `--epo-con-temp 0.01`  (seed 0)
  R1e-pick  = R0e-pick with `--seed 1` and its own paths      (Stage 2, only on outcome O1)
  R1o       = SAFEOBJ's registered R1o, token for token (eval/rls1d_arms.json arms.R1o): `--epo-con off`, seed 1, NO --epo-con-temp (the trainer refuses it with `off`)

SELF-CHECK (PREREG 2.1), computed here and asserted:  for each e arm, the argv minus `--epo-con-temp <T>` with its paths swapped back to /arms/R0d (and its seed read as R0d's) equals arms.R0d.argv TOKEN FOR
TOKEN; `lever_tokens_not_in_R0d` is exactly {--epo-con-temp, <T>, own paths} (+ the seed token for R1e-pick); R1o equals rls1d's R1o.
"""
import hashlib
import json
import os
import sys

R = "/workspace/data/rls1"
T = {"R0e-id": "0.1", "R0e-pick": "0.01", "R1e-pick": "0.01"}
SEED = {"R0e-id": 0, "R0e-pick": 0, "R1e-pick": 1, "R1o": 1}
LEVER = "--epo-con-temp"
PREREG_SHA256 = "1be7a0301b454a21a0adcfc209ce73f63b7802a72d3155008a90ca616d5cccd0"


def main():
    pk = os.path.abspath(sys.argv[1])
    sys.path.insert(0, os.path.join(pk, "refe"))
    import gate_token as GT  # noqa: E402
    src = os.path.join(pk, "eval", "rls1d_arms.json")
    raw = open(src, "rb").read()
    d = json.loads(raw.decode("utf-8"))
    r0d, r1o = d["arms"]["R0d"]["argv"], d["arms"]["R1o"]["argv"]
    assert r0d[r0d.index("--out") + 1] == f"{R}/arms/R0d", r0d[r0d.index("--out") + 1]
    assert GT.argv_sha256(r0d) == d["arms"]["R0d"]["argv_sha256"], "rls1d_arms.json R0d.argv does not hash to its own argv_sha256"

    def own(arm):
        a = [(f"{R}/arms/{arm}" + t[len(f"{R}/arms/R0d"):]) if t in (f"{R}/arms/R0d", f"{R}/arms/R0d/scores") else t for t in r0d]
        a[a.index("--seed") + 1] = str(SEED[arm])
        return a
    arms, per_arm = {}, {}
    notes = {"R0e-id": "its OWN arm A (same pass); banked R0d is the same-seed replicate it is read against at EVERY check (u250 gate reading, u500 / u750 / u1000 floors): PREREG 5.1",
             "R0e-pick": "its OWN arm A (same pass); R0e-id's paired B/A differences are the same-run floor quoted beside every R0e-pick vs R0d contrast",
             "R1e-pick": "its OWN arm A; the seed-1 replicate of R0e-pick (Stage 2, only if R0e-pick reads O1)",
             "R1o": "its OWN arm A; the matched lever-off control (K3 OBSERVE-ONLY, K2 / K6 live); runs only after R0e-pick AND R1e-pick read bar S = PASS"}
    for arm in ("R0e-id", "R0e-pick", "R1e-pick"):
        argv = own(arm) + [LEVER, T[arm]]
        arms[arm] = {"seed": SEED[arm], "class": "R", "registered_con_temp": float(T[arm]), "control": notes[arm], "argv": argv, "args": " ".join(argv), "argv_sha256": GT.argv_sha256(argv)}
        stripped = argv[:-2]
        assert argv[-2:] == [LEVER, T[arm]]
        rebuilt = [t.replace(f"{R}/arms/{arm}", f"{R}/arms/R0d") for t in stripped]
        rebuilt[rebuilt.index("--seed") + 1] = "0"
        eq = rebuilt == r0d
        lever = [t for t in argv if t not in r0d]
        expect = sorted({f"{R}/arms/{arm}", f"{R}/arms/{arm}/scores", LEVER, T[arm]} | ({"1"} if SEED[arm] == 1 else set()))
        per_arm[arm] = {"equals_R0d_after_swap_and_lever_removal": eq, "lever_tokens_not_in_R0d": lever, "lever_tokens_expected": expect, "lever_tokens_exact": sorted(lever) == expect,
                        "length_is_R0d_plus_2": len(argv) == len(r0d) + 2}
        assert eq and sorted(lever) == expect and len(argv) == len(r0d) + 2, (arm, per_arm[arm])
    argv = list(r1o)
    arms["R1o"] = {"seed": SEED["R1o"], "class": "R", "registered_con_temp": None, "control": notes["R1o"], "argv": argv, "args": " ".join(argv), "argv_sha256": GT.argv_sha256(argv)}
    assert LEVER not in argv and argv[-2:] == ["--epo-con", "off"] and argv[argv.index("--seed") + 1] == "1"
    per_arm["R1o"] = {"equals_rls1d_arms_R1o": argv == r1o, "lacks_epo_con_temp": LEVER not in argv}
    doc = {"schema": "rls1e_arms/1",
           "prereg": "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md (FROZEN 2026-10-18; PI decisions 1a-1f; running an arm is a PI + Master Mind decision: A40 compute)",
           "prereg_sha256": PREREG_SHA256,
           "profile": "refe_rls1e",
           "lever": "--epo-con-temp <T> (the temperature of the CONSTRAINT's softmax; the parser default None inherits --epo-temp 0.1 and silently reproduces R0d, so the argv NAMES it); R1o is `--epo-con off` with no --epo-con-temp; "
                    "everything else is R0d's argv token for token",
           "derived_from": {"arm": "R0d", "file": "eval/rls1d_arms.json", "file_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(), "argv_sha256": d["arms"]["R0d"]["argv_sha256"],
                            "R1o_argv_sha256": d["arms"]["R1o"]["argv_sha256"], "file_sha256_note": "sha256 of the file with CRLF read as LF (a Windows worktree with core.autocrlf checks eval/*.json out as CRLF)"},
           "order": ["R0e-id", "R0e-pick", "R1e-pick", "R1o"],
           "stop_rules": "Stage 1 = R0e-id then R0e-pick; R1e-pick (Stage 2) only if R0e-pick reads O1; R1o only if BOTH e-pick arms read bar S = PASS (rls1_checks.py safeobj-gate --dirs $R/arms/R0e-pick $R/arms/R1e-pick); "
                         "a K3 kill of R0e-pick or R1e-pick STOPS the chain (generic killed-arm refusal)",
           "arms": arms,
           "self_check": {"R0d_argv_sha256": d["arms"]["R0d"]["argv_sha256"], "equals_rls1d_arms_R0d": GT.argv_sha256(r0d) == d["arms"]["R0d"]["argv_sha256"], "per_arm": per_arm,
                          "note": "paths are the pod chain's ($R = /workspace/data/rls1); `lever_tokens_not_in_R0d` is a token SET difference (a repeated token like '0' is not listed), so test_launch_gate.py also asserts the "
                                  "sequence equality; the gate enforces it at launch (launch_gate.rls1e_lineage_refusals)"},
           "budget": {"a40_arms_stage1": 2, "hours_per_arm_planned": 3.68, "stage1_a40_hours": 7.36, "stage2_a40_hours_max": 7.36, "max_a40_hours": 14.72,
                      "basis": "ESTIMATED from R0d's banked wall clock, PREREG_R0E_PICK 7 (R0e-id runs to u1000: PI 1c)"}}
    path = os.path.join(pk, "eval", "rls1e_arms.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(json.dumps({"wrote": path, "argv_sha256_12": {a: arms[a]["argv_sha256"][:12] for a in arms}, "R0d_sha256_12": d["arms"]["R0d"]["argv_sha256"][:12],
                      "self_check": {a: all(v for v in per_arm[a].values() if isinstance(v, bool)) for a in per_arm}}))


if __name__ == "__main__":
    main()
