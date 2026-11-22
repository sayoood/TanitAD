"""test_rl_gate_record.py -- MM 2026-11-22. Dev-box test of rl_gate_record.py on planted files (stdlib only, no pod).

A well-formed RL_s0 record must MINT and VERIFY; each mutation must REFUSE:
  M1 argv without --dagger-label; M2 --dagger-label human; M3 built levers without speed_v1; M4 dagger probe rows changed 0; M5 the probe key missing;
  M6 G-SUITE at 97fe0a56; M7 tag RH_s0; M8 the RH tree marker only (no done_rl/tree.done); M9 bank sha256 in the argv is not bank.json's;
  M10 a bound file changes after the mint (verify must refuse).
Usage: python -I test_rl_gate_record.py <rl_gate_record.py> <scratch dir>
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

GR, ROOT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
CF27 = "cf27dbb79f2491a6db1197860bc50ca7c8cdeff4"


def w(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(o if isinstance(o, str) else json.dumps(o))


def plant(R, tag="RL_s0", mut=None):
    if os.path.exists(R):
        shutil.rmtree(R)
    seed = tag.split("_s")[1]
    w(f"{R}/bank/s{seed}/bank.json", {"bank": "planted"})
    bsha = hashlib.sha256(open(f"{R}/bank/s{seed}/bank.json", "rb").read()).hexdigest()
    w(f"{R}/bank/s{seed}/BANK_CHECK.json", {"pass": True, "readout": "causal_g1x"})
    argv = ["--dagger-p", "0.1", "--dagger-bank", f"{R}/bank/s{seed}", "--dagger-bank-sha256", bsha if mut != "M9" else "0" * 64,
            "--dagger-bank-readout", "causal_g1x", "--dagger-bank-kappa", "0.2", "--out", f"{R}/runs/{tag}/train"]
    if mut != "M1":
        argv += ["--dagger-label", "human" if mut == "M2" else "speed_v1"]
    w(f"{R}/runs/{tag}/argv.json", {"argv": argv})
    probe = {"pass": True, "n_selected": 12}
    if mut != "M5":
        probe["rl_label_rows_changed"] = 0 if mut == "M4" else 7
    levers = {"dagger": {"dagger_bank_readout": "causal_g1x", "dagger_label": "human" if mut == "M3" else "speed_v1"}}
    w(f"{R}/runs/{tag}/selfcheck/selfcheck.json", {"declared_vs_built": "PASS", "acc_probe": {"pass": True}, "dagger_probe": probe, "levers": levers})
    w(f"{R}/gsuite_result_rl.json", {"new_commit": "97fe0a56767320dbc5f4b823d67fb07a3209a745" if mut == "M6" else CF27, "n_regressions": 0, "new_test_failures": 0, "missing_in_new": 0})
    w(f"{R}/pilot/PILOT_RH.json", {"pass": True, "readout": "causal_g1x"})
    w(f"{R}/done/tree.done", "x")
    if mut != "M8":
        w(f"{R}/done_rl/tree.done", "x")
    w(f"{R}/runs/{tag}/pair_{tag}.log", "ZZ ROLLHIST-PAIR PASS\n")


def run(R, *a):
    env = dict(os.environ, RH_ROOT=R)
    p = subprocess.run([sys.executable, "-I", GR, *a], env=env, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout + p.stderr


ok = True
R = os.path.join(ROOT, "R")
plant(R)
rc, out = run(R, "mint", "--tag", "RL_s0", "--readout", "causal_g1x")
rc2, out2 = run(R, "verify", "--tag", "RL_s0")
good = rc == 0 and "MINTED" in out and rc2 == 0 and "VERIFY OK" in out2
print("WELL-FORMED", "PASS" if good else "FAIL", rc, rc2, out.strip().splitlines()[-1:], out2.strip().splitlines()[-1:])
ok &= good
for m in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"):
    tag = "RH_s0" if m == "M7" else "RL_s0"
    plant(R, tag=tag, mut=m)
    rc, out = run(R, "mint", "--tag", tag, "--readout", "causal_g1x")
    red = rc != 0 and "REFUSED" in out
    print(m, "RED" if red else "NOT-RED", rc, [l for l in out.splitlines() if l.startswith("PROBLEM")][:2])
    ok &= red
plant(R)
run(R, "mint", "--tag", "RL_s0", "--readout", "causal_g1x")
w(f"{R}/gsuite_result_rl.json", {"new_commit": CF27, "n_regressions": 0, "new_test_failures": 0, "missing_in_new": 0, "touched": 1})
rc, out = run(R, "verify", "--tag", "RL_s0")
red = rc != 0 and "REFUSED" in out
print("M10", "RED" if red else "NOT-RED", rc)
ok &= red
shutil.rmtree(R)
print("ZZ RL-GATE-RECORD TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
