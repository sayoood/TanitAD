"""make_rl_cf27.py -- MM 2026-11-22 (RECLAB). Writes the cf27dbb7 forms of the three RL pod scripts into this folder from the build's forms in ../ (sha256 in ../POD_FILES_SHA256.txt).

Why: the build staged RL at the code of record 97fe0a56 (`. rh_env.sh`, tree_cdb, all_blobs_97fe0a56.txt, no G-SUITE step), but the --dagger-label flag exists only at
cf27dbb7. The RL arms therefore run in their OWN tree (tree_cf27) under rl_env.sh, with a fresh G-SUITE at cf27dbb7 as a chain step, and the RL gate record
(rl_gate_record.py) in place of rh_gate_record.py, whose "RH" branch test sent RL tags down the B0 branch. Every edit is an exact anchor that must match the stated
number of times, or this script fails. Nothing else in the three files changes.
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..")
WANT = {"rh_rl_chain.sh": "6e2934abcdc53c7e", "rl_arm.sh": "b2648f4a44478ba7", "chk_arm_rl.sh": "7b6e4409a26a7eac"}

EDITS = {
    "rh_rl_chain.sh": [
        ("Code of record 97fe0a56, readout causal_g1x. STAGED, NOT LAUNCHED.",
         "Code of record cf27dbb7 (rl_env.sh: own tree tree_cf27 = 97fe0a56 + the --dagger-label flag), readout causal_g1x. STAGED, NOT LAUNCHED.", 1),
        ("NO G-SUITE / pilot (both are at the SAME commit and are bound by the\n# existing gsuite_result_rh.json / PILOT_RH.json, which rh_gate_record.py re-hashes at mint),",
         "a FRESH G-SUITE at cf27dbb7 (rl_gsuite.sh, step `gsuite`, CPU under the lock, about 36 min as RH's) and NO pilot (PILOT_RH.json is the\n# step-time record, not bound to the commit; rl_gate_record.py re-hashes it and gsuite_result_rl.json at mint),", 1),
        ("(rl_argv.py: RH's argv with exactly --dagger-p 0.3 and --out)", "(rl_argv.py: RH's argv with exactly --dagger-label speed_v1 and --out)", 1),
        (". $HERE/rh_env.sh\n", ". $HERE/rl_env.sh\n", 1),
        ("$R/all_blobs_97fe0a56.txt", "$BLOBS", 2),
        ("step tree chk_tree\n",
         "step tree chk_tree\n"
         "gs_passed() { [ -s $GSUITE_RL ] && [ -s $R/logs/rl_gsuite.log ] && tail -n 1 $R/logs/rl_gsuite.log | grep -q \"ZZ RL GSUITE DONE ZZ\"; }\n"
         "gs() {   # fresh G-SUITE at cf27dbb7 (0 regressions, 0 new-test failures, 0 missing); a pass from the standalone prep run (rl_gsuite.sh before the launch) is kept, not re-run under the lock\n"
         "  gs_passed && { say \"G-SUITE at ${COMMIT:0:8} passed in the prep run: kept (rl_gate_record.py re-checks its commit and counts at mint)\"; return 0; }\n"
         "  bash $HERE/rl_gsuite.sh; gs_passed; }\n"
         "step gsuite gs\n", 1),
        ("for f in $HERE/rl_arm.sh $HERE/chk_arm_rl.sh $HERE/rl_argv.py $R/gsuite_result_rh.json $R/pilot/PILOT_RH.json",
         "for f in $HERE/rl_arm.sh $HERE/chk_arm_rl.sh $HERE/rl_argv.py $HERE/rl_env.sh $HERE/rl_gsuite.sh $HERE/rl_gate_record.py $GSUITE_RL $R/pilot/PILOT_RH.json", 1),
    ],
    "rl_arm.sh": [
        (". $HERE/rh_env.sh\n", ". $HERE/rl_env.sh\n", 1),
        ("$HERE/rh_gate_record.py", "$HERE/rl_gate_record.py", 2),
        ("-> rh_gate_record.py mint + verify", "-> rl_gate_record.py mint + verify (cf27dbb7 form: RL takes the RH branch plus the label checks)", 1),
    ],
    "chk_arm_rl.sh": [
        (". $HERE/rh_env.sh\n", ". $HERE/rl_env.sh\n", 1),
        ("--gsuite $R/gsuite_result_rh.json", "--gsuite $GSUITE_RL", 1),
        ("the RL kind is pinned to RH's argv plus exactly two tokens 0.3;", "the RL kind is pinned to RH's argv plus exactly the two tokens `--dagger-label speed_v1` (cf27dbb7 form: rl_env.sh, gsuite_result_rl.json);", 1),
    ],
}

for name, edits in EDITS.items():
    raw = open(os.path.join(SRC, name), "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    assert got.startswith(WANT[name]), f"{name}: sha256 {got[:16]} is not the build's {WANT[name]}"
    s = raw.decode("utf-8")
    for old, new, n in edits:
        c = s.count(old)
        assert c == n, f"{name}: anchor matched {c} times, want {n}: {old[:70]!r}"
        s = s.replace(old, new)
    for left in ("rh_env.sh\n", "all_blobs_97fe0a56", "gsuite_result_rh", "rh_gate_record.py mint", "rh_gate_record.py verify"):
        assert left not in s or (name == "rh_rl_chain.sh" and left == "rh_env.sh\n" and ". $HERE/rh_env.sh" not in s), f"{name}: old pin {left!r} is left"
    out = os.path.join(HERE, name)
    open(out, "w", encoding="utf-8", newline="\n").write(s)
    print("wrote", name, hashlib.sha256(s.encode()).hexdigest()[:16])
