"""R42: G-LIVE-LAB's L1 stall bar is judged at the LAUNCH accumulation (the G-LIVE-MEM run), never on the accum-2 smoke.
argv: <refe dir>.  Exact-anchor replacements; every anchor must occur exactly once."""
import sys
from pathlib import Path

R = Path(sys.argv[1])


def sub(name, pairs):
    p = R / name
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert t.count(old) == 1, (name, t.count(old), old[:80])
        t = t.replace(old, new)
    p.write_bytes(t.encode("utf-8"))


# ---- gate_checks.py
sub("gate_checks.py", [
    ("LAB_NEED_SETS_S, LAB_MARGIN, LAB_STALL_MAX, MEM_MAX_GB, MEM_MIN_MICRO = 4.49, 1.5, 0.05, 44.0, 200\n",
     "LAB_NEED_SETS_S, LAB_MARGIN, LAB_STALL_MAX, MEM_MAX_GB, MEM_MIN_MICRO = 4.49, 1.5, 0.05, 44.0, 200\n"
     "LAB_STALL_MIN_WINDOWS = 2           # steady-state windows (after the first, which includes pool start-up) the L1 stall bar is judged on (R42)\n"),
    ('def judge_lab(lab: dict, smoke_metrics: list, workers_launch: int, arm=None) -> list:\n'
     '    """lab = diag_p6_labeller_identity.py\'s JSON; smoke_metrics = the SMOKE trainer\'s metrics rows. Returns the failure reasons."""\n',
     'def judge_lab(lab: dict, smoke_metrics: list, workers_launch: int, arm=None, steady_metrics=None, launch_accum=None) -> list:\n'
     '    """lab = diag_p6_labeller_identity.py\'s JSON; smoke_metrics = the SMOKE trainer\'s metrics rows (the labeller RAN, lag 0, coverage);\n'
     '    steady_metrics = the G-LIVE-MEM run\'s metrics rows (the launch argv at the launch\'s --accum): the L1 stall bar (SPEC_P6 3.2) is judged THERE.\n'
     '    R42 (MEASURED 2026-10-16, P6 pod): train.py drains each accumulation window\'s LAST micro-batch before the optimizer step (D4), so one label\n'
     '    latency per window is unhidden by construction and the stall fraction scales like 1/accum.  The accum-2 smoke read 33.1 % in its only\n'
     '    post-start-up window while the launch-argv run read 2.8-5.0 %; a stall bar judged on the smoke can never pass on any pod.\n'
     '    Returns the failure reasons."""\n'),
    ('        stall = max(e["stall_frac"] for e in (ev[1:] or ev))        # the first window includes pool start-up; steady state is judged\n'
     '        if stall > LAB_STALL_MAX:\n'
     '            r.append(f"GPU stall {100 * stall:.1f} % of a window > {100 * LAB_STALL_MAX:.0f} %")\n'
     '    return r\n',
     '    # the L1 stall bar, at the LAUNCH accumulation (R42): never on the smoke, whose 2-micro-batch windows are drain-dominated\n'
     '    sm = steady_metrics or []\n'
     '    sev = [m for m in sm if m.get("event") == "labeller"]\n'
     '    s_acc = next((m.get("accum") for m in sm if m.get("event") == "bank"), None)\n'
     '    if not sm:\n'
     '        r.append("no steady-state stall run (the G-LIVE-MEM metrics): the L1 stall is judged at the launch --accum, never on the accum-2 smoke (R42)")\n'
     '    elif launch_accum is None or s_acc != launch_accum:\n'
     '        r.append(f"the stall run\'s accum {s_acc} != the launch argv\'s --accum {launch_accum}: the stall fraction scales like 1/accum (R42)")\n'
     '    elif len(sev) < 1 + LAB_STALL_MIN_WINDOWS:\n'
     '        r.append(f"the stall run logged {len(sev)} labeller windows; need >= {1 + LAB_STALL_MIN_WINDOWS} (the first includes pool start-up)")\n'
     '    else:\n'
     '        if any(set(e.get("lag_hist", {})) - {"0"} for e in sev):\n'
     '            r.append(f"label lag != 0 in the stall run: {[e.get(\'lag_hist\') for e in sev][:2]}")\n'
     '        stall = max(e["stall_frac"] for e in sev[1:])               # the first window includes pool start-up; steady state is judged\n'
     '        if stall > LAB_STALL_MAX:\n'
     '            r.append(f"GPU stall {100 * stall:.1f} % of a window > {100 * LAB_STALL_MAX:.0f} % (launch accum {launch_accum}, max over windows after the first)")\n'
     '    return r\n'),
    ('def check_live_lab(lab_json, smoke_cap, argv: list, arm=None, rehearsal=False) -> dict:\n',
     'def check_live_lab(lab_json, smoke_cap, argv: list, arm=None, rehearsal=False, mem_metrics=None) -> dict:\n'),
    ('    w = int((GT_.flag_values(argv, "--label-workers") or ["12"])[0])\n'
     '    reasons = judge_lab(lab, smoke_cap.metrics if smoke_cap is not None else [], w, arm=arm)\n'
     '    return verdict("G-LIVE-LAB", reasons, {"lab": lab, "file_sha256": GT_.sha256_file(lab_json)}, arm=arm, rehearsal=rehearsal)\n',
     '    w = int((GT_.flag_values(argv, "--label-workers") or ["12"])[0])\n'
     '    acc = (GT_.flag_values(argv, "--accum") or [None])[0]\n'
     '    launch_accum = int(acc) if acc is not None else None\n'
     '    steady = None\n'
     '    if mem_metrics and Path(mem_metrics).is_file():\n'
     '        steady = [json.loads(ln) for ln in Path(mem_metrics).read_text(encoding="utf-8").splitlines() if ln.strip()]\n'
     '    smoke = smoke_cap.metrics if smoke_cap is not None else []\n'
     '    reasons = judge_lab(lab, smoke, w, arm=arm, steady_metrics=steady, launch_accum=launch_accum)\n'
     '    details = {"lab": lab, "file_sha256": GT_.sha256_file(lab_json),\n'
     '               "stall_judged_on": {"mem_metrics_sha256": GT_.sha256_file(mem_metrics) if steady is not None else None, "launch_accum": launch_accum,\n'
     '                                   "stall_frac_by_window": [e.get("stall_frac") for e in (steady or []) if e.get("event") == "labeller"]},\n'
     '               "smoke_stall_frac_by_window_reported_only": [e.get("stall_frac") for e in smoke if e.get("event") == "labeller"]}\n'
     '    return verdict("G-LIVE-LAB", reasons, details, arm=arm, rehearsal=rehearsal)\n'),
])

# ---- launch_gate.py
sub("launch_gate.py", [
    ("        emit(C.check_live_lab(lab_json, s, argv, arm=arm, rehearsal=rehearsal))\n",
     "        emit(C.check_live_lab(lab_json, s, argv, arm=arm, rehearsal=rehearsal, mem_metrics=mem_metrics))     # the L1 stall: on the launch-accum run (R42)\n"),
])

# ---- p6_prep.py
sub("p6_prep.py", [
    ("  mem-argv        the G-LIVE-MEM trainer argv: the launch ARGS with --epochs/--resume dropped and --steps 4 --accum 50 (200 micro-batches), a scratch --out.\n",
     "  mem-argv        the G-LIVE-MEM trainer argv: the launch ARGS with --epochs/--resume dropped and --steps 4 at the LAUNCH --accum (4 x 64 = 256 micro-batches), a scratch --out.\n"),
    ("MEM_STEPS, MEM_ACCUM = 4, 50                    # 4 x 50 micro-batches of 4 = 200 micro-batches (gate_checks.MEM_MIN_MICRO)\n",
     "MEM_STEPS = 4                                   # 4 windows at the LAUNCH --accum (64: 256 micro-batches >= gate_checks.MEM_MIN_MICRO).  It was --accum 50 until R42:\n"
     "MEM_MIN_MICRO = 200                             # the same run is G-LIVE-LAB's L1 stall reading, and the stall fraction scales like 1/accum\n"),
    ('    """The launch argv with the schedule cut to MEM_STEPS x MEM_ACCUM micro-batches (--epochs derives the step count and is dropped), no --resume, a scratch --out,\n'
     '    no periodic checkpoint. Everything that sets the memory (batch, backbone, amp, compile, labeller, workers) is the LAUNCH value."""\n'
     '    argv = args.split()\n'
     '    argv = _drop(_drop(_drop(argv, "--resume"), "--epochs", 1), "--grow-scenes", 1)\n'
     '    for f, v in (("--steps", str(MEM_STEPS)), ("--accum", str(MEM_ACCUM)), ("--log-every", "1"), ("--ckpt-every-min", "999"), ("--out", out_dir)):\n',
     '    """The launch argv with the schedule cut to MEM_STEPS windows at the LAUNCH --accum (--epochs derives the step count and is dropped), no --resume,\n'
     '    a scratch --out, no periodic checkpoint. Everything that sets the memory (batch, backbone, amp, compile, labeller, workers) AND the L1 stall\n'
     '    (the accumulation: one drained micro-batch per window, R42) is the LAUNCH value."""\n'
     '    argv = args.split()\n'
     '    if "--accum" not in argv or argv.index("--accum") + 1 >= len(argv):\n'
     '        raise SystemExit("the launch ARGS carry no --accum: the stall run must use the launch accumulation (R42)")\n'
     '    if MEM_STEPS * int(argv[argv.index("--accum") + 1]) < MEM_MIN_MICRO:\n'
     '        raise SystemExit(f"{MEM_STEPS} windows x --accum {argv[argv.index(\'--accum\') + 1]} < {MEM_MIN_MICRO} micro-batches (G-LIVE-MEM)")\n'
     '    argv = _drop(_drop(_drop(argv, "--resume"), "--epochs", 1), "--grow-scenes", 1)\n'
     '    for f, v in (("--steps", str(MEM_STEPS)), ("--log-every", "1"), ("--ckpt-every-min", "999"), ("--out", out_dir)):\n'),
])

# ---- test_p6_chain.py
sub("test_p6_chain.py", [
    ('m[m.index("--steps") + 1] == "4" and m[m.index("--accum") + 1] == "50", m; n += 3\n',
     'm[m.index("--steps") + 1] == "4" and m[m.index("--accum") + 1] == "64", m; n += 3     # the LAUNCH accum (R42)\n'),
    ('"--accum 50" in mm and "refe_p6/mem_run" in mm; n += 1\n',
     '"--accum 64" in mm and "refe_p6/mem_run" in mm; n += 1\n'),
])

# ---- test_p6_levers.py
sub("test_p6_levers.py", [
    ('    ev = [{"event": "labeller", "sets": 8, "labelled": 8, "lag_hist": {"0": 8}, "stall_frac": 0.30},\n'
     '          {"event": "labeller", "sets": 8, "labelled": 8, "lag_hist": {"0": 8}, "stall_frac": 0.01}]\n'
     '    ok(C.judge_lab(lab, ev, 12) == [], C.judge_lab(lab, ev, 12))\n'
     '    for bad, frag in (({**lab, "identity_equal_sets": 63, "identity_ok": False}, "identity"), ({**lab, "stale_refused": False}, "stale"),\n'
     '                      ({**lab, "throughput_sets_s": 4.3}, "throughput"), ({**lab, "workers": 2}, "launch argv"), ({**lab, "n_sets": 32}, "identity")):\n'
     '        r = C.judge_lab(bad, ev, 12)\n'
     '        ok(r and any(frag in x for x in r), (frag, r))\n'
     '    ok(any("stall" in x for x in C.judge_lab(lab, [ev[0], {**ev[1], "stall_frac": 0.2}], 12)))\n'
     '    ok(any("lag" in x for x in C.judge_lab(lab, [{**ev[0], "lag_hist": {"0": 4, "3": 4}}, ev[1]], 12)))\n'
     '    ok(any("no `labeller` event" in x for x in C.judge_lab(lab, [], 12)))\n'
     '    ok(C.judge_lab(lab, ev, 12, arm="lab_two_workers"), "a 2-worker pool must FAIL the throughput bar")\n',
     '    # the smoke (accum 2): its post-start-up window is DRAIN-dominated -- these are the MEASURED P6-pod values of 2026-10-16 (R42), and the judge must\n'
     '    # NOT fail on them; the L1 stall is judged on the launch-accum run (`st`, the G-LIVE-MEM metrics: bank event + >= 3 labeller windows)\n'
     '    ev = [{"event": "labeller", "sets": 8, "labelled": 8, "lag_hist": {"0": 8}, "stall_frac": 0.36692370317285544},\n'
     '          {"event": "labeller", "sets": 8, "labelled": 8, "lag_hist": {"0": 8}, "stall_frac": 0.3310444063967688}]\n'
     '    st = [{"event": "bank", "accum": 64}] + [{"event": "labeller", "sets": 256, "labelled": 256, "lag_hist": {"0": 256}, "stall_frac": s}\n'
     '                                            for s in (0.1413863002634917, 0.02780238159944948, 0.04803343924789811, 0.0300)]\n'
     '\n'
     '    def J(lab_, ev_, w_=12, arm=None, st_=st, acc=64):\n'
     '        return C.judge_lab(lab_, ev_, w_, arm=arm, steady_metrics=st_, launch_accum=acc)\n'
     '\n'
     '    ok(J(lab, ev) == [], J(lab, ev))\n'
     '    for bad, frag in (({**lab, "identity_equal_sets": 63, "identity_ok": False}, "identity"), ({**lab, "stale_refused": False}, "stale"),\n'
     '                      ({**lab, "throughput_sets_s": 4.3}, "throughput"), ({**lab, "workers": 2}, "launch argv"), ({**lab, "n_sets": 32}, "identity")):\n'
     '        r = J(bad, ev)\n'
     '        ok(r and any(frag in x for x in r), (frag, r))\n'
     '    # the stall bar, on the launch-accum run: a window above 5 % FAILS (incl. the 2026-10-16 accum-50 reading 0.0500021, which is > 0.05)\n'
     '    ok(any("stall" in x for x in J(lab, ev, st_=st[:2] + [{**st[2], "stall_frac": 0.2}] + st[3:])))\n'
     '    ok(any("stall" in x for x in J(lab, ev, st_=st[:2] + [{**st[2], "stall_frac": 0.05000205494919062}] + st[3:])))\n'
     '    ok(not any("stall" in x for x in J(lab, ev, st_=st[:1] + [{**st[1], "stall_frac": 0.9}] + st[2:])), "the FIRST window (pool start-up) is not judged")\n'
     '    # ... and only there: no stall run, an accum that is not the launch accum, or too few windows FAIL; the old smoke-only call FAILS\n'
     '    ok(any("no steady-state stall run" in x for x in J(lab, ev, st_=None)))\n'
     '    ok(any("no steady-state stall run" in x for x in C.judge_lab(lab, ev, 12)), "the pre-R42 call (smoke only) must FAIL, never pass silently")\n'
     '    ok(any("1/accum" in x for x in J(lab, ev, st_=[{"event": "bank", "accum": 50}] + st[1:])))\n'
     '    ok(any("1/accum" in x for x in J(lab, ev, acc=None)))\n'
     '    ok(any("labeller windows" in x for x in J(lab, ev, st_=st[:3])))\n'
     '    ok(any("lag" in x for x in J(lab, ev, st_=st[:2] + [{**st[2], "lag_hist": {"0": 200, "1": 56}}] + st[3:])))\n'
     '    # the smoke still proves the labeller RAN at lag 0 with coverage\n'
     '    ok(any("lag" in x for x in J(lab, [{**ev[0], "lag_hist": {"0": 4, "3": 4}}, ev[1]])))\n'
     '    ok(any("no `labeller` event" in x for x in J(lab, [])))\n'
     '    ok(J(lab, ev, arm="lab_two_workers"), "a 2-worker pool must FAIL the throughput bar")\n'
     '    # check_live_lab reads the stall run from the G-LIVE-MEM metrics FILE and the accum from the launch ARGV\n'
     '    import tempfile\n'
     '    import types\n'
     '    with tempfile.TemporaryDirectory() as td_:\n'
     '        lj, mj = Path(td_) / "lab.json", Path(td_) / "mem.jsonl"\n'
     '        lj.write_text(json.dumps({**lab, "workers": 7}), encoding="utf-8")\n'
     '        mj.write_text("".join(json.dumps(x) + "\\n" for x in st), encoding="utf-8")\n'
     '        cap = types.SimpleNamespace(metrics=ev)\n'
     '        av = ["--label-workers", "7", "--batch", "4", "--accum", "64"]\n'
     '        ok(C.check_live_lab(str(lj), cap, av, mem_metrics=str(mj))["status"] == "PASS", C.check_live_lab(str(lj), cap, av, mem_metrics=str(mj)))\n'
     '        ok(C.check_live_lab(str(lj), cap, av)["status"] == "FAIL", "no mem_metrics -> FAIL")\n'
     '        ok(C.check_live_lab(str(lj), cap, av[:4] + ["--accum", "128"], mem_metrics=str(mj))["status"] == "FAIL", "launch accum != stall-run accum -> FAIL")\n'),
])
print("patched")
