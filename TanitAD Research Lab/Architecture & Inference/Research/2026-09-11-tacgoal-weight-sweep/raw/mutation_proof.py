# -*- coding: utf-8 -*-
"""Deliberate-regression proof for stack/tests/test_grad_probe_tacgoal.py.

A guard that has never FAILED certifies nothing. Each mutation below
reintroduces a real historical defect; the named test must go RED. The trainer
is restored and md5-verified after every mutation, and the script REFUSES to
continue if a restore does not verify.
"""
import hashlib
import io
import os
import subprocess
import sys

REPO = r"D:\Projects\TanitAD"
P = os.path.join(REPO, "stack", "scripts", "refc_v3_train.py")
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
TEST = "tests/test_grad_probe_tacgoal.py"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


GOOD_BYTES = open(P, "rb").read()
GOOD = hashlib.md5(GOOD_BYTES).hexdigest()
print("GOOD md5 =", GOOD)


def run(nodeid):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, "stack")
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    r = subprocess.run([PY, "-m", "pytest", nodeid, "-q", "--no-header"],
                       cwd=os.path.join(REPO, "stack"), env=env,
                       capture_output=True, text=True)
    tail = (r.stdout or "")[-400:].replace("\n", " | ")
    return r.returncode, tail


def mutate(old, new):
    s = io.open(P, encoding="utf-8").read()
    assert s.count(old) == 1, (old[:60], s.count(old))
    io.open(P, "w", encoding="utf-8", newline="\r\n").write(s.replace(old, new, 1))


def restore():
    open(P, "wb").write(GOOD_BYTES)
    got = md5(P)
    assert got == GOOD, f"RESTORE FAILED {got} != {GOOD}"


MUTATIONS = [
    ("M1 merge removed -> the probe never reaches metrics.jsonl",
     "            if _gp_row:\n                row.update(_gp_row)\n",
     "            if _gp_row and False:\n                row.update(_gp_row)\n",
     TEST + "::test_trainer_merges_the_probe_after_its_rounding_comprehension"),
    ("M2 probe moved AFTER the global clip -> reading is rescaled",
     ("        _gp_row = _grad_probe_row(\n"
      "            model, _gp_names,\n"
      "            log_every_hit=bool(_gp_names) and (\n"
      "                (step + 1) % args.log_every == 0\n"
      "                or (step + 1) >= args.steps))\n"
      "        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)"),
     ("        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)\n"
      "        _gp_row = _grad_probe_row(\n"
      "            model, _gp_names,\n"
      "            log_every_hit=bool(_gp_names) and (\n"
      "                (step + 1) % args.log_every == 0\n"
      "                or (step + 1) >= args.steps))"),
     TEST + "::test_probe_is_read_after_backward_and_before_the_global_clip"),
    ("M1b merge DELETED entirely -> the probe never reaches metrics.jsonl",
     ("            if _gp_row:\n"
      "                row.update(_gp_row)\n"),
     "",
     TEST + "::test_trainer_merges_the_probe_after_its_rounding_comprehension"),
    ("M3 probe counts params instead of gradient -> cannot read the defect",
     "                g_sum += float(p.grad.detach().abs().sum().item())",
     "                g_sum += float(p.numel())",
     TEST + "::test_probe_sums_ABSOLUTE_GRADIENT_against_an_analytic_target"),
    ("M6 abs() dropped -> a cancelling gradient reads as no gradient",
     "                g_sum += float(p.grad.detach().abs().sum().item())",
     "                g_sum += float(p.grad.detach().sum().item())",
     TEST + "::test_probe_sums_ABSOLUTE_GRADIENT_against_an_analytic_target"),
    ("M4 --w-tac-goal default raised off 0.0 -> the pinned literal moves",
     '    ap.add_argument("--w-tac-goal", type=float, default=0.0,',
     '    ap.add_argument("--w-tac-goal", type=float, default=0.05,',
     TEST + "::test_flag_defaults_are_the_literals_the_defect_depended_on"),
    ("M5 a missing module reads like an unreached one (silent absence)",
     '            out["gp_%s_found" % nm] = 0.0\n            continue',
     '            out["gp_%s_found" % nm] = 1.0\n            continue',
     TEST + "::test_a_mistyped_module_reads_found_zero_rather_than_a_silent_absence"),
]

print("\n=== GREEN baseline (whole file) ===")
rc, tail = run(TEST)
print("rc", rc, "|", tail)
assert rc == 0, "baseline must be GREEN before any mutation"

rows = []
for name, old, new, nodeid in MUTATIONS:
    mutate(old, new)
    rc, tail = run(nodeid)
    restore()
    verdict = "RED (good)" if rc != 0 else "GREEN (INERT GUARD -- BAD)"
    rows.append((name, verdict, rc))
    print(f"\n--- {name}\n    -> {verdict}  rc={rc}\n    {tail[:240]}")

print("\n=== SUMMARY ===")
bad = 0
for name, verdict, rc in rows:
    print(f"{verdict:22s} {name}")
    if rc == 0:
        bad += 1
print("\nfinal md5 =", md5(P), "restored_ok =", md5(P) == GOOD)
print("INERT GUARDS =", bad)
sys.exit(1 if bad else 0)
