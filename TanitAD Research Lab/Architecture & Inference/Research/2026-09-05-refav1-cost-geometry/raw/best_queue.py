"""Launch `combined_seed1` (FIRST) and then `best`, each into a free GPU slot.

⛔ WHY THE GATE COUNTS ARMS, NOT PROCESSES: one arm is 2+ `python.exe` entries (a
parent and its children, all carrying the full command line). The identity of an arm
is its `--dump-dir`, which is unique per arm; the count is taken over those.

⛔ SELF-IMMUNE: my own dump dirs are excluded by name, so once arm 1 launches the
gate does not deadlock on itself.

⛔ GATES ON ABSENCE OF LOAD, NEVER ON A SUCCESS MARKER: an arm that dies leaves no
marker and a marker-waiter would poll forever (`wait-loops-must-match-failure`). A
probe that FAILS is not read as "no arms".

⛔ THE EMITTED MARKER IS DISJOINT FROM THE SEARCHED TOKEN (`ZZ..ZZ`), so a monitor
grepping this log cannot match its own echoed command line.

⛔ IT REFUSES TO LAUNCH IF THE ARGV AUDIT FAILS. Each new command line must differ
from the banked `combined` invocation in EXACTLY ONE token, ignoring the three
identity tokens (`--dump-dir`, `--out`, `--arm`) that necessarily name the arm.

Arms are spawned DETACHED, so they survive this launcher.  ASCII only.
"""
import os
import re
import subprocess
import sys
import time

PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
TOOL = "C:/Users/Admin/tanitad-wt/taniteval/tools/refav1_arm.py"
P4 = "C:/Users/Admin/refav1_margin/p4out"
SP = os.path.dirname(os.path.abspath(__file__))
VEND = "64.29715042415070"
WK = "15.11245"
LADDER = "0.002,0.005,0.01,0.02,0.04"

#: the banked `combined` invocation (raw/queueG.sh, commit 6b29077) -- the reference
#: every audit below is taken against. NOT retyped from memory: the flag values are
#: cross-checked against rec_combined.json's manifest in `verify_reference()`.
COMMON = [
    "--ckpt", "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt",
    "--config", "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json",
    "--cache", "C:/Users/Admin/refav1_margin/p4/fp8",
    "--episodes", "C:/Users/Admin/refav1_margin/p4/eps",
    "--labels", "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz",
    "--nav", "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz",
    "--device", "cuda", "--episodes-n", "0", "--window-stride", "16",
    "--no-navshuf", "--no-lead-block", "--cost-metric", "ccos",
]


def argv_for(tag, wkappa, seed):
    return ([TOOL] + COMMON +
            ["--cost-weights", "0.0,%s,%s" % (wkappa, VEND),
             "--plan-seed", seed,
             "--seed-kappa-ladder", LADDER, "--kamm-mu", "0.7",
             "--dump-dir", "%s/dump_%s" % (P4, tag),
             "--out", "%s/rec_%s.json" % (P4, tag),
             "--arm", "refav1-21109-p4-%s" % tag])


REF = argv_for("combined", "0.0", "0")          # the banked arm

#: ⛔ THE THREE-LEVER ARM IS `bestlad`, NOT `best`. MEASURED 2026-09-05 23:07: a
#: SIBLING agent's `queueJ.sh` launched `--arm refav1-21109-p4-best` writing to
#: `dump_best` / `rec_best.json` -- eleven seconds before this queue started -- and
#: that arm is a DIFFERENT configuration: `W_KAPPA` + `--kamm-mu 0.7` with
#: `seed_kappa_ladder=None`, i.e. NO LADDER. Had I kept the name, the two would have
#: written the same dump and record, and my "already done" skip would then have made
#: me analyse the sibling's two-lever arm believing it was my three-lever one.
#: The sibling's arm is legitimate work and is left running; it is in fact the
#: missing cell of the factorial (see RESULT). Mine is renamed.
ARMS = [("combined_seed1", "0.0", "1"),         # P1.2 -- runs FIRST
        ("bestlad", WK, "0")]                   # P1.1: W_KAPPA + cap + LADder
MINE = {t for t, _, _ in ARMS}
IDENTITY = {"--dump-dir", "--out", "--arm"}


def audit(tag, argv, fh):
    """Token-by-token diff vs the banked `combined`. Returns list of real diffs."""
    fh.write("\n==== %s\n" % tag)
    fh.write("  ref  n_tokens=%d\n  new  n_tokens=%d\n" % (len(REF), len(argv)))
    if len(REF) != len(argv):
        fh.write("  FAIL: token COUNT differs -- not one variable\n")
        return ["<token count>"]
    diffs, skip = [], False
    for i, (a, b) in enumerate(zip(REF, argv)):
        if skip:
            skip = False
            continue
        if a in IDENTITY and a == b:
            fh.write("  identity  %s  %s -> %s  (names the arm, not a variable)\n"
                     % (a, REF[i + 1], argv[i + 1]))
            skip = True
            continue
        if a != b:
            diffs.append("%s: %r -> %r" % (REF[i - 1] if i else "<pos0>", a, b))
            fh.write("  DIFF @%d  under %s: %r -> %r\n"
                     % (i, REF[i - 1] if i else "<pos0>", a, b))
    fh.write("  real differing tokens: %d  %s\n" % (len(diffs), diffs))
    return diffs


def verify_reference(fh):
    """The reference must match the RECORD the banked arm actually wrote."""
    import json
    p = "%s/rec_combined.json" % P4
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:                                        # noqa: BLE001
        fh.write("REFERENCE CHECK INCONCLUSIVE: cannot read %s (%s)\n" % (p, e))
        return False
    m = d["refav1"]["manifest"]
    got = {"W_KAPPA": d["cost"]["weights"]["W_KAPPA"],
           "W_VEND": d["cost"]["weights"]["W_VEND"],
           "kamm_mu": m["plan_cfg"]["kamm_mu"],
           "seed": m["plan_cfg"]["seed"],
           "ladder": tuple(m["goal_rule"]["seed_kappa_ladder"] or ())}
    want = {"W_KAPPA": 0.0, "W_VEND": float(VEND), "kamm_mu": 0.7, "seed": 0,
            "ladder": tuple(float(x) for x in LADDER.split(","))}
    ok = got == want
    fh.write("REFERENCE CHECK vs rec_combined.json manifest: %s\n"
             "  got  %s\n  want %s\n" % ("PASS" if ok else "FAIL", got, want))
    return ok


PS = ["powershell.exe", "-NoProfile", "-Command",
      "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "ForEach-Object { $_.CommandLine }"]


def foreign_arms():
    """Set of live arm dump-dir basenames, or None if the probe failed.

    ⛔ COUNTS MY OWN ARMS TOO. The first version excluded them (copied from a sibling
    waiter that deliberately launches PAIRS), and because this queue launches
    SEQUENTIALLY that exclusion made the gate open for arm 2 while arm 1 was still
    running -- three concurrent arms on an 8 GB card. Self-exclusion is right for a
    waiter that owns a batch; it is wrong for one that fills one slot at a time.
    """
    try:
        r = subprocess.run(PS, capture_output=True, text=True, timeout=180)
    except Exception:                                             # noqa: BLE001
        return None
    if r.returncode != 0 and not r.stdout:
        return None
    tags = set()
    for line in r.stdout.splitlines():
        if "refav1" + "_arm.py" not in line:      # split so this file never self-matches
            continue
        m = re.search(r"--dump-dir\s+(\S+)", line)
        if m:
            t = os.path.basename(m.group(1).strip('"'))
            t = t[5:] if t.startswith("dump_") else t
            tags.add(t)
    return tags


#: ⛔ NOT `DETACHED_PROCESS`. A detached process has NO CONSOLE, and the Intel Fortran
#: runtime that ships inside the MKL/torch stack then aborts at startup with
#: `forrtl: error (200): program aborting due to window-CLOSE event` -- MEASURED here,
#: 2026-09-05 23:05, which killed `combined_seed1` after its preflight and before its
#: rollout (zero GPU wasted, the preflight design working as intended).
#: `CREATE_NO_WINDOW` gives the child its OWN console with no visible window, so the
#: runtime is happy and no console-close from this launcher's shell can reach it.
DETACH = 0x08000000 | 0x00000200          # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP


def launch(tag, argv):
    env = dict(os.environ)
    env["PYTHONPATH"] = ("C:/Users/Admin/tanitad-wt/stack;"
                         "C:/Users/Admin/tanitad-wt/taniteval")
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = "6"           # torch spawns ~113 threads/proc otherwise
    log = open("%s/%s.log" % (P4, tag), "ab")
    p = subprocess.Popen([PY] + argv, stdout=log, stderr=subprocess.STDOUT,
                         env=env, creationflags=DETACH, close_fds=True)
    return p.pid


def main():
    ap = "%s/argv_audit.txt" % SP
    with open(ap, "w", encoding="ascii") as fh:
        fh.write("ARGV AUDIT -- written BEFORE launch, not after.\n"
                 "The reference is the banked `combined` invocation; each new arm must\n"
                 "differ in EXACTLY ONE token once the three identity tokens are set aside.\n\n")
        ref_ok = verify_reference(fh)
        results = {}
        for tag, wk, seed in ARMS:
            results[tag] = audit(tag, argv_for(tag, wk, seed), fh)
        fh.write("\nVERDICT\n")
        ok = ref_ok
        for tag, diffs in results.items():
            good = (len(diffs) == 1)
            ok &= good
            fh.write("  %-16s %s  (%d real diff)\n"
                     % (tag, "PASS" if good else "FAIL", len(diffs)))
        fh.write("  reference       %s\n" % ("PASS" if ref_ok else "FAIL"))
        fh.write("  OVERALL         %s\n" % ("PASS" if ok else "FAIL -- NOT LAUNCHING"))
    print("ZZAUDIT-%s-ZZ" % ("PASS" if ok else "FAIL"), flush=True)
    print(open(ap, encoding="ascii").read(), flush=True)
    if not ok:
        sys.exit(3)
    if "--dry-run" in sys.argv:
        print("ZZDRY-RUN-NO-LAUNCHZZ", flush=True)
        for tag, wk, seed in ARMS:
            print("  would run: %s %s" % (PY, " ".join(argv_for(tag, wk, seed))))
        return

    deadline = time.time() + 6 * 3600
    for idx, (tag, wk, seed) in enumerate(ARMS):
        if os.path.exists("%s/rec_%s.json" % (P4, tag)):
            print("ZZSKIP-%s-ALREADY-DONEZZ" % tag, flush=True)
            continue
        i = 0
        while time.time() < deadline:
            i += 1
            f = foreign_arms()
            if f is None:
                print("ZZPROBE-FAILED-%d-ZZ not treating as 'no arms'" % i, flush=True)
                time.sleep(60)
                continue
            if len(f) <= 1:                # <=1 foreign arm -> mine makes 2 concurrent
                print("ZZSLOT-CLEAR-%s-%d-n%d-ZZ %s"
                      % (tag, i, len(f), time.strftime("%H:%M:%S")), flush=True)
                break
            if i % 10 == 0:
                print("ZZWAIT-%s-%d-n%d-ZZ %s %s"
                      % (tag, i, len(f), sorted(f), time.strftime("%H:%M:%S")), flush=True)
            time.sleep(30)
        else:
            print("ZZDEADLINE-%s-ZZ" % tag, flush=True)
            sys.exit(4)
        # ⛔ A LAUNCH IS NOT A RUN. `combined_seed1` printed every startup banner and
        # then died on the console bug above; the launcher reported success. Assert
        # the process is ALIVE and the log has reached the rollout before moving on.
        for tries in range(1, 4):
            log = "%s/%s.log" % (P4, tag)
            if tries > 1 and os.path.isdir("%s/dump_%s" % (P4, tag)):
                import shutil
                shutil.rmtree("%s/dump_%s" % (P4, tag), ignore_errors=True)
            pid = launch(tag, argv_for(tag, wk, seed))
            print("ZZLAUNCHED-%s-PID%d-try%d-ZZ %s"
                  % (tag, pid, tries, time.strftime("%H:%M:%S")), flush=True)
            time.sleep(240)                # startup + first plan() is ~60-70 s
            alive = tag in (foreign_arms() or set())
            txt = ""
            try:
                txt = open(log, "rb").read().decode("utf-8", "replace")
            except Exception:                                     # noqa: BLE001
                pass
            bad = ("forrtl" in txt) or ("Traceback" in txt[-4000:])
            if alive and not bad:
                print("ZZALIVE-%s-try%d-ZZ rollout under way" % (tag, tries), flush=True)
                break
            print("ZZDEAD-%s-try%d-ZZ alive=%s bad_marker=%s -- last log line: %s"
                  % (tag, tries, alive, bad,
                     (txt.strip().splitlines() or ["<empty>"])[-1][:160]), flush=True)
        else:
            print("ZZGIVEUP-%s-ZZ three launches died at startup" % tag, flush=True)
            sys.exit(7)
    print("ZZQUEUE-BEST-ALL-LAUNCHEDZZ", flush=True)


if __name__ == "__main__":
    main()
