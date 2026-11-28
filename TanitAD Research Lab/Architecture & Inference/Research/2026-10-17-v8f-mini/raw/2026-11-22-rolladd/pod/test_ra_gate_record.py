"""test_ra_gate_record.py -- MM 2026-11-22. Dev-box test of ra_gate_record.py on planted files (stdlib only, no pod).

A well-formed RA_s0 record must MINT and VERIFY; each mutation must REFUSE:
  M1 argv without --dagger-mode; M2 --dagger-mode swap; M11 probe grad_diff_all 0; M12 control_p0_grad_diff != 0; M13 the commit placeholder left unfilled; M3 built levers say swap; M4 rows_appended 0; M5 the probe key missing;
  M6 G-SUITE at 97fe0a56; M7 tag RH_s0; M8 the RH tree marker only (no done_ra/tree.done); M9 bank sha256 in the argv is not bank.json's;
  M10 a bound file changes after the mint (verify must refuse).
Usage: python -I test_ra_gate_record.py <ra_gate_record.py (unfilled)> <scratch dir>
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

GR0, ROOT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
GR = os.path.join(ROOT, 'ra_gate_record_filled.py')
CF27 = "df250a0032e56aae89304421918ade5c583d53a0"          # the RA commit (MM 2026-11-22: was a generator-garbled string; the well-formed fill and the G-SUITE record must name the same 40-hex commit)


def w(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(o if isinstance(o, str) else json.dumps(o))


def plant(R, tag="RA_s0", mut=None):
    if os.path.exists(R):
        shutil.rmtree(R)
    seed = tag.split("_s")[1]
    w(f"{R}/bank/s{seed}/bank.json", {"bank": "planted"})
    bsha = hashlib.sha256(open(f"{R}/bank/s{seed}/bank.json", "rb").read()).hexdigest()
    w(f"{R}/bank/s{seed}/BANK_CHECK.json", {"pass": True, "readout": "causal_g1x"})
    argv = ["--dagger-p", "0.1", "--dagger-bank", f"{R}/bank/s{seed}", "--dagger-bank-sha256", bsha if mut != "M9" else "0" * 64,
            "--dagger-bank-readout", "causal_g1x", "--dagger-bank-kappa", "0.2", "--out", f"{R}/runs/{tag}/train"]
    if mut != "M1":
        argv += ["--dagger-mode", "swap" if mut == "M2" else "add"]
    w(f"{R}/runs/{tag}/argv.json", {"argv": argv})
    probe = {"pass": True, "n_selected": 12, "grad_diff_all": 0.0 if mut == "M11" else 3.5, "control_p0_grad_diff": 0.25 if mut == "M12" else 0.0}
    if mut != "M5":
        probe["rows_appended"] = 0 if mut == "M4" else 12
    levers = {"dagger": {"dagger_bank_readout": "causal_g1x", "dagger_mode": "swap" if mut == "M3" else "add"}}
    w(f"{R}/runs/{tag}/selfcheck/selfcheck.json", {"declared_vs_built": "PASS", "acc_probe": {"pass": True}, "dagger_probe": probe, "levers": levers})
    w(f"{R}/gsuite_result_ra.json", {"new_commit": "cf27dbb79f2491a6db1197860bc50ca7c8cdeff4" if mut == "M6" else CF27, "n_regressions": 0, "new_test_failures": 0, "missing_in_new": 0})
    w(f"{R}/pilot/PILOT_RH.json", {"pass": True, "readout": "causal_g1x"})
    w(f"{R}/done/tree.done", "x")
    if mut != "M8":
        w(f"{R}/done_ra/tree.done", "x")
    w(f"{R}/runs/{tag}/pair_{tag}.log", "ZZ ROLLHIST-PAIR PASS\n")


def run(R, *a):
    env = dict(os.environ, RH_ROOT=R)
    p = subprocess.run([sys.executable, "-I", GR, *a], env=env, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout + p.stderr


os.makedirs(ROOT, exist_ok=True)
open(GR, 'w', encoding='utf-8', newline='\n').write(open(GR0, encoding='utf-8').read().replace('__RA_COMMIT__', CF27))
ok = True
R = os.path.join(ROOT, "R")
plant(R)
rc, out = run(R, "mint", "--tag", "RA_s0", "--readout", "causal_g1x")
rc2, out2 = run(R, "verify", "--tag", "RA_s0")
good = rc == 0 and "MINTED" in out and rc2 == 0 and "VERIFY OK" in out2
print("WELL-FORMED", "PASS" if good else "FAIL", rc, rc2, out.strip().splitlines()[-1:], out2.strip().splitlines()[-1:])
ok &= good
for m in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M11", "M12"):
    tag = "RH_s0" if m == "M7" else "RA_s0"
    plant(R, tag=tag, mut=m)
    rc, out = run(R, "mint", "--tag", tag, "--readout", "causal_g1x")
    red = rc != 0 and "REFUSED" in out
    print(m, "RED" if red else "NOT-RED", rc, [l for l in out.splitlines() if l.startswith("PROBLEM")][:2])
    ok &= red
plant(R)
run(R, "mint", "--tag", "RA_s0", "--readout", "causal_g1x")
w(f"{R}/gsuite_result_ra.json", {"new_commit": CF27, "n_regressions": 0, "new_test_failures": 0, "missing_in_new": 0, "touched": 1})
rc, out = run(R, "verify", "--tag", "RA_s0")
red = rc != 0 and "REFUSED" in out
print("M10", "RED" if red else "NOT-RED", rc)
ok &= red
unf = subprocess.run([sys.executable, '-I', GR0, 'mint', '--tag', 'RA_s0', '--readout', 'causal_g1x'], env=dict(os.environ, RH_ROOT=R), capture_output=True, text=True, encoding='utf-8')
red = unf.returncode != 0 and 'REFUSED' in unf.stdout
print('M13', 'RED' if red else 'NOT-RED')
ok &= red
shutil.rmtree(R)
print("ZZ RA-GATE-RECORD TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
