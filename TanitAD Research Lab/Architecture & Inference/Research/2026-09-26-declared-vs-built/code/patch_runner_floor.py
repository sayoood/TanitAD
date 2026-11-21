"""The Master Mind's floor for the CPU-only gate chain, per-file junit, and a taniteval
resolution check -- exact-match edits on run_suite_bounded.py (written as a file: a quoted bash
heredoc eats one backslash level)."""
import sys

P = sys.argv[1]
s = open(P, encoding="utf-8").read()


def edit(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[patch] {tag}: anchor found {n} times")
    s = s.replace(old, new)


edit('PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"\n'
     'START_GB, KILL_GB, RETRIES, FILE_TIMEOUT_S = 9.0, 8.0, 3, 1800\n'
     'MAX_WAIT_S = 3600\n',
     'PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"\n'
     '# FLOOR, CPU-ONLY CHAIN (Master Mind, 2026-09-26 ~22:30 Berlin): start only after THREE\n'
     '# consecutive samples 30 s apart read >= 7.5 GB available; kill our child below 6.5 GB.\n'
     '# Reason: the box sat below the brief\'s 8 GB floor for > 1 h because of other sessions\' jobs;\n'
     '# E1\'s NavSim scorer aborts only after 120 s below 3 GB (run_navsim_refcv6.py:317), so 6.5 GB\n'
     '# keeps a 3.5 GB margin. The 8 GB rule stays in force for GPU jobs.\n'
     'START_GB, KILL_GB, RETRIES, FILE_TIMEOUT_S = 7.5, 6.5, 3, 1800\n'
     'START_SAMPLES, START_GAP_S = 3, 30.0\n'
     'MAX_WAIT_S = 6 * 3600\n'
     'FLOOR_NOTE = ("floor: start >= 7.5 GB on 3 consecutive samples 30 s apart, kill < 6.5 GB "\n'
     '              "(Master Mind 2026-09-26: box below 8 GB for > 1 h from other sessions; the "\n'
     '              "NavSim scorer aborts only after 120 s < 3 GB, run_navsim_refcv6.py:317; the "\n'
     '              "8 GB rule is kept for GPU jobs); OMP_NUM_THREADS=4; one pytest file at a time")\n',
     "thresholds")

edit('    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", rel]\n'
     '    log = logdir / (rel.replace("/", "__").replace("\\\\", "__") + ".log")\n',
     '    stem = rel.replace("/", "__").replace("\\\\", "__")\n'
     '    log = logdir / (stem + ".log")\n'
     '    junit = logdir / (stem + ".junit.xml")\n'
     '    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE",\n'
     '           f"--junitxml={junit}", rel]\n',
     "junit")

edit('            for attempt in range(RETRIES + 1):\n'
     '                w0 = time.time()\n'
     '                while free_gb() < START_GB and time.time() - w0 < MAX_WAIT_S:\n'
     '                    time.sleep(15.0)\n',
     '            for attempt in range(RETRIES + 1):\n'
     '                w0, ok = time.time(), 0\n'
     '                while ok < START_SAMPLES and time.time() - w0 < MAX_WAIT_S:\n'
     '                    ok = ok + 1 if free_gb() >= START_GB else 0\n'
     '                    if ok < START_SAMPLES:\n'
     '                        time.sleep(START_GAP_S)\n',
     "3-sample start")

edit('        fh.write(json.dumps({"file": "__header__", "tree": tp, "tanitad": where,\n'
     '                             "t": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\\n")\n',
     '        fh.write(json.dumps({"file": "__header__", "tree": tp, "tanitad": where,\n'
     '                             "taniteval": te_where, "floor": FLOOR_NOTE,\n'
     '                             "t": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\\n")\n'
     '        fh.flush()\n',
     "header")

edit('    where = chk.stdout.strip()\n'
     '    if not where.replace("\\\\", "/").lower().startswith(tp.lower()):\n'
     '        raise SystemExit(f"tanitad resolves to {where!r}, NOT inside {tp}")\n',
     '    where = chk.stdout.strip()\n'
     '    if not where.replace("\\\\", "/").lower().startswith(tp.lower()):\n'
     '        raise SystemExit(f"tanitad resolves to {where!r}, NOT inside {tp}")\n'
     '    chk2 = subprocess.run([PY, "-c", "import taniteval.nav_compliance as m; print(m.__file__)"],\n'
     '                          env=env, capture_output=True, text=True, cwd=str(tree / rundir))\n'
     '    te_where = chk2.stdout.strip()\n'
     '    if not te_where.replace("\\\\", "/").lower().startswith(tp.lower()):\n'
     '        raise SystemExit(f"taniteval resolves to {te_where!r}, NOT inside {tp}")\n',
     "taniteval check")

open(P, "w", encoding="utf-8", newline="\n").write(s)
print("runner patched")
