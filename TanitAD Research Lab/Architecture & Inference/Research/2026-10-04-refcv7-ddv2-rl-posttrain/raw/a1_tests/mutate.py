"""Deliberate-regression arms for SPEC_RL_A1: mutate a scratch copy of the A1 code, run the named
tests, and require the EXPECTED colour (RED for a real defect; GREEN for a documented equivalent
mutant). Usage: python mutate.py <run_a1 dir> <scratch dir> <python exe>"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

src, scratch, py = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
LIB = "tanitad/rl/ddv2_refcv7.py"
TRN = "scripts/ddv2_rl_refcv7.py"
CRLF = "\r\n"

# (name, file, [(old, new, count), ...], tests, expected)
MUTS = [
    ("M1 t_prev=t in the KL", LIB, [("int(t) - 1", "int(t)", 3)],
     ["test_kl_analytic_literal_and_gradient"], "RED"),
    ("M2 KL sign flipped", LIB, [("return lp_ref.detach() - lp_th", "return lp_th - lp_ref.detach()", 1)],
     ["test_kl_analytic_literal_and_gradient", "test_kl_is_nonnegative_on_the_release_table_at_every_label"],
     "RED"),
    ("M3 reference params not restored", LIB, [("                p.copy_(saved[n])", "                pass", 1)],
     ["test_swapped_params_restores_bitwise_also_on_an_exception"], "RED"),
    ("M4 median -> mean in K1/K2", LIB, [("statistics.median(", "statistics.fmean(", 2)],
     ["test_halt_one_spike_does_not_trip_the_median_rules"], "RED"),
    ("M5 K3 waits for the window", LIB,
     [('if pdn > 0 and float(cur["param_delta_norm"]) > pdn:',
       'if pdn > 0 and len(history) >= int(window) and float(cur["param_delta_norm"]) > pdn:', 1)],
     ["test_halt_k3_is_instantaneous_and_k1_k2_wait_for_the_window"], "RED"),
    ("M6 --warmup-steps ignored", TRN,
     [("if not warmup_steps else max(1, int(warmup_steps))", "if True else max(1, int(warmup_steps))", 1)],
     ["test_lr_at_literals_fraction_and_absolute_warmup"], "RED"),
    ("M7 KL gradient reaches the reference (x0_ref AND lp_ref undetached)", LIB,
     [("    x0_ref = x0_ref.detach()" + CRLF, "", 1),
      ("    return lp_ref.detach() - lp_th", "    return lp_ref - lp_th", 1)],
     ["test_kl_gradient_reaches_theta_only"], "RED"),
    ("M7e EQUIVALENT mutant: only x0_ref.detach() removed (mu_ref, lp_ref still detached)", LIB,
     [("    x0_ref = x0_ref.detach()" + CRLF, "", 1)],
     ["test_kl_gradient_reaches_theta_only"], "GREEN"),
]

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
           OMP_NUM_THREADS="4")
bad = 0
for name, rel, subs, tests, want in MUTS:
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(src, scratch)
    p = scratch / rel
    s = p.read_bytes().decode("ascii")
    ok = True
    for old, new, cnt in subs:
        c = s.count(old)
        if c != cnt:
            print(f"{name}: ANCHOR count {c} != {cnt} for {old!r} -- arm not applied")
            ok = False
            break
        s = s.replace(old, new)
    if not ok:
        bad += 1
        continue
    p.write_bytes(s.encode("ascii"))
    env["PYTHONPATH"] = str(scratch)
    r = subprocess.run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_ddv2_refcv7.py",
                        "-k", " or ".join(tests)],
                       cwd=scratch, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    last = lines[-1] if lines else r.stderr[-200:]
    got = "RED" if (r.returncode != 0 and "failed" in last) else ("GREEN" if r.returncode == 0 else "ERROR")
    print(f"{name}: {got} (expected {want}) {'OK' if got == want else 'UNEXPECTED'} | rc {r.returncode} | {last}")
    bad += 0 if got == want else 1
shutil.rmtree(scratch, ignore_errors=True)
print("ALL ARMS AS EXPECTED" if bad == 0 else f"{bad} ARM(S) UNEXPECTED")
sys.exit(1 if bad else 0)
