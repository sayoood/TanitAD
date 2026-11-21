# ====================================================================== C2. the r2 chain script
import inspect                                          # noqa: E402

print("== C2: the r2 chain script (dry-run, check-tree, STREAM_PATTERN, real-run refusals, tampered helpers)")
FIX = R2 / "fixtures"
R1LOG = FIX / "r1_chain_v8f_dagoaccorc.out"            # r1's chain log, snapshotted (the live copy is C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc.out)
OACC_TAIL = FIX / "oacc_r2_chain_tail.out"              # the tail of the DAG-OACC r2 chain's log (it ENDED in ZZV8FMINI-DONE)
DEFAULT_CHAIN = (R2SHIP / "chain_v8f_dagoaccorc_r2.sh").read_text(encoding="utf-8")
R1_CHAIN = (R1SHIP / "chain_v8f_dagoaccorc.sh").read_text(encoding="utf-8")
SP = "stack/scripts/v8f_mini_train.py|STUB_EGO_STREAM_SEPARATED"
SR = "levers.ego_perturb_stream=true"
SET_CHAIN = MC.render(SP, SR)
PH_CHAIN = MC.render(MC.PH_PATTERN, MC.PH_RECORD)      # the fail-closed placeholder chain (what the generator makes while the Master Mind has not set the STREAM_* values)
REG_PATTERN, REG_RECORD = MC.DEFAULT_PATTERN, MC.DEFAULT_RECORD


def sh(cmdline, env=None, timeout=300):
    e = {**os.environ, "MSYS_NO_PATHCONV": "0", **(env or {})}
    try:
        p = subprocess.run([BASH, "-c", cmdline], capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, env=e)
    except subprocess.TimeoutExpired as ex:
        return 124, ((ex.stdout or b"").decode("utf-8", "replace") + (ex.stderr or b"").decode("utf-8", "replace")).replace("\r", "")
    return p.returncode, (p.stdout or b"").decode("utf-8", "replace").replace("\r", "") + (p.stderr or b"").decode("utf-8", "replace").replace("\r", "")


def bp(p):
    return str(p).replace("\\", "/")


def ship2(root, chain=None, rm=(), tamper=None):
    shutil.rmtree(root, ignore_errors=True)
    (root / "tmp").mkdir(parents=True)
    for f in ("gate_dagoaccorc_r2_assert.py", "v8f_dagoaccorc_r2_chain_checks.py", "oaf_dagoaccorc_r2_lib.sh"):
        shutil.copy(R2SHIP / f, root / f)
    (root / "chain_v8f_dagoaccorc_r2.sh").write_bytes((DEFAULT_CHAIN if chain is None else chain).encode("utf-8"))
    for f in ("oaf_lib.sh", "v8f_oaf_chain_checks.py", "v8f_guard_dagoaccorc.sh"):
        shutil.copy(INST / f, root / f)
    for f in rm:
        (root / f).unlink()
    if tamper:
        with open(root / tamper, "ab") as fh:
            fh.write(b"\n# tampered\n")
    return root


def run_in(sd, args, env=None, timeout=300, kill_after=None):
    """kill_after: coreutils `timeout N` INSIDE the shell, so a chain that waits forever is killed with its children (a python-level timeout would kill only the outer bash and leave the chain looping, holding the pipe)."""
    e = {"TMPDIR": bp(sd) + "/tmp", "OAF_SHIP_DIR": bp(sd), "DAGOACCORC_ART": bp(sd) + "/art", **(env or {})}
    pre = " ".join(f'{k}="{v}"' for k, v in e.items())
    tmo = f"timeout {kill_after} " if kill_after else ""
    return sh(f'cd "{bp(sd)}" && {pre} {tmo}bash ./chain_v8f_dagoaccorc_r2.sh {args}', timeout=timeout if not kill_after else kill_after + 60)


qf = bp(QUEUE)
R1LOG_P = bp(R1LOG)
SD = ship2(SCR / "ship")                          # the chain ON DISK: STREAM_* = the registered values
SDP = ship2(SCR / "ship_ph", chain=PH_CHAIN)       # the placeholder chain
rc, out = run_in(SD, "--dry-run", dict(V8F_QUEUE=qf, AFTER_OUT=R1LOG_P))
T("C2-1 --dry-run of the chain ON DISK (registered STREAM_* values) on the patched queue: rc 0, 4 SAME, 4 ONE-LEVER, the registered order, the r2 predecessor line, 44 aside items, the registered STREAM_PATTERN / STREAM_RECORD printed, NO placeholder warning",
  rc == 0 and out.count(": SAME") == 4 and out.count(": ONE-LEVER") == 4 and "== runs (order): dagoaccorc 0 dagoaccorcrep 0 dagoaccorc 1 dagoaccorcrep 1" in out and "== predecessor (r2)" in out
  and "box 2 (44 items" in out and "!! " not in out and f"STREAM_PATTERN={REG_PATTERN}" in out and f"STREAM_RECORD={REG_RECORD}" in out, out[-400:])
rc, out = run_in(SDP, "--dry-run", dict(V8F_QUEUE=qf, AFTER_OUT=R1LOG_P))
T("C2-1b --dry-run of the PLACEHOLDER chain: rc 0 and both STREAM placeholders flagged with '!!'",
  rc == 0 and "!! STREAM_PATTERN is still the placeholder" in out and "!! STREAM_RECORD is still the placeholder" in out, out[-400:])
SDS = ship2(SCR / "ship_set", chain=SET_CHAIN)
rc, out = run_in(SDS, "--dry-run", dict(V8F_QUEUE=qf, AFTER_OUT=R1LOG_P))
T("C2-2 --dry-run on a chain with STREAM_PATTERN / STREAM_RECORD set: rc 0 and NO placeholder warning", rc == 0 and "!! " not in out and f"STREAM_PATTERN={SP}" in out and f"STREAM_RECORD={SR}" in out, out[-300:])
T("C2-3 the dry-run touched nothing (no BOX2_BUSY.txt, no logs/, no art/ in the ship dir)", not (SDS / "BOX2_BUSY.txt").exists() and not (SDS / "logs").exists() and not (SDS / "art").exists())

G_TIP = "6cd741b63b3b629282d48f3b88d86f99264f9c9f"
MIR = "C:/Users/Admin/tanitad-push"
files = ["stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/data/v8f_dagger.py"]


def tree(root, queue_path, trainer_extra=""):
    shutil.rmtree(root, ignore_errors=True)
    for f in files:
        dst = root / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{G_TIP}:{f}"], capture_output=True, check=True).stdout)
    q = root / PK / "code" / "v8f_mini_box2_queue.cmd"
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_bytes(Path(queue_path).read_bytes())
    if trainer_extra:
        with open(root / files[0], "a", encoding="utf-8", newline="\n") as fh:
            fh.write(trainer_extra)
    return root


STUB = ('\n    ap.add_argument("--ego-perturb-with-dagger", dest="ego_perturb_with_dagger", action="store_true")\n'
        'FLAG_CHECKS_STUB = {"ego_perturb_with_dagger": "ego:ego_perturb_with_dagger"}\n_STUB = {"ego_dagger_overlap_n": 0}\n')
STREAM_STUB = '\nSTUB_EGO_STREAM_SEPARATED = True\n'
T_NOSTREAM = tree(SCR / "t_nostream", QUEUE, STUB)
T_STREAM = tree(SCR / "t_stream", QUEUE, STUB + STREAM_STUB)
rc, out = run_in(SDP, f'--check-tree "{bp(T_STREAM)}"')
T("C2-4 --check-tree with the PLACEHOLDER STREAM_PATTERN (a tree that has everything else): REFUSED, naming STREAM_PATTERN", rc == 1 and "STREAM_PATTERN is still the placeholder" in out and "lever check: FAILED" in out, out[-300:])
rc, out = run_in(SDS, f'--check-tree "{bp(T_NOSTREAM)}"')
T("C2-5 --check-tree with STREAM_PATTERN set on a tree WITHOUT the stream separation (flag, queue, overlap record all present): REFUSED, naming the missing pattern",
  rc == 1 and "MISSING in stack/scripts/v8f_mini_train.py: STUB_EGO_STREAM_SEPARATED" in out and "lever check: FAILED" in out, out[-300:])
rc, out = run_in(SDS, f'--check-tree "{bp(T_STREAM)}"')
m2 = re.search(r"every one of (\d+) patterns", out)
rc1, out1 = sh(f'cd "{bp(R1SHIP)}" && bash ./chain_v8f_dagoaccorc.sh --check-tree "{bp(T_STREAM)}"')
m1 = re.search(r"every one of (\d+) patterns", out1)
T("C2-6 --check-tree with STREAM_PATTERN set on a tree WITH it: PASS, one pattern more than r1's chain on the same tree", rc == 0 and m1 and m2 and int(m2.group(1)) == int(m1.group(1)) + 1, (out[-200:], out1[-200:]))
chain_nopipe = MC.render("STUB_EGO_STREAM_SEPARATED", SR)
rc, out = run_in(ship2(SCR / "ship_np", chain=chain_nopipe), f'--check-tree "{bp(T_STREAM)}"')
T("C2-7 a STREAM_PATTERN that is not a '<file>|<ERE>' record: REFUSED", rc == 1 and "is not a '<file>|<grep -E pattern>' record" in out, out[-300:])
# mutants of the lever grep
mut = SET_CHAIN.replace('for rec in "${LEVER[@]}" "$STREAM_PATTERN"; do', 'for rec in "${LEVER[@]}"; do')
assert mut != SET_CHAIN
rc, out = run_in(ship2(SCR / "ship_m", chain=mut), f'--check-tree "{bp(T_NOSTREAM)}"')
M("MC-s1 the lever loop forgets STREAM_PATTERN: a tree WITHOUT the stream separation is accepted (the C2-5 refusal is gone)", not (rc == 1 and "STUB_EGO_STREAM_SEPARATED" in out), out[-200:])
mut = PH_CHAIN.replace("stream_pattern_problem() {   # rc 0 = STREAM_PATTERN cannot be used (prints why)\n", "stream_pattern_problem() {   # MUTANT\n  return 1\n")
assert mut != PH_CHAIN
rc, out = run_in(ship2(SCR / "ship_m", chain=mut), f'--check-tree "{bp(T_STREAM)}"')
M("MC-s2 stream_pattern_problem always says fine: the PLACEHOLDER chain accepts a tree (the C2-4 refusal is gone)", not (rc == 1 and "lever check: FAILED" in out), out[-200:])
OLD_CASE = "    *'|'*) return 1 ;;" + chr(10) + "    *) echo " + chr(34) + "STREAM_PATTERN is not a"
NEW_CASE = "    *) return 1 ;;" + chr(10) + "    *) echo " + chr(34) + "STREAM_PATTERN is not a"
mut = chain_nopipe.replace(OLD_CASE, NEW_CASE)
assert mut != chain_nopipe
rc, out = run_in(ship2(SCR / "ship_m", chain=mut), f'--check-tree "{bp(T_STREAM)}"')
M("MC-s3 a non-record STREAM_PATTERN is accepted as a record (the C2-7 refusal is gone)", not (rc == 1 and "is not a '<file>|<grep -E pattern>' record" in out), out[-200:])

# --- the REAL trees: the r1 tip (e7ea20d1: NO stream separation) and the same tree with the staged trainer fix (dagland/compose_fix/stage).  The registered STREAM_PATTERN must refuse the first, accept the second.
E7 = "e7ea20d1287599409d1758971ded62b2fc4ef375"
FIXSTAGE = Path("E:/Projects/TanitAD-artifacts/dagland/compose_fix/stage")
QPATH = PK + "/code/v8f_mini_box2_queue.cmd"
CONST = "EGO_PERTURB_COMPOSED_STREAM = 1"


def real_tree(root, fix=False, data_edit=None):
    shutil.rmtree(root, ignore_errors=True)
    for f in files + [QPATH]:
        dst = root / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{E7}:{f}"], capture_output=True, check=True).stdout)
    if fix:
        for f in ("stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py"):
            (root / f).write_bytes((FIXSTAGE / f).read_bytes())
    if data_edit:
        d = root / "stack/tanitad/data/v8f_mini_data.py"
        txt = d.read_bytes().decode("utf-8")
        new = data_edit(txt)
        assert new != txt
        d.write_bytes(new.encode("utf-8"))
    return root


assert (FIXSTAGE / "stack/tanitad/data/v8f_mini_data.py").read_text(encoding="utf-8").count("\n" + CONST + "\n") == 1, "the staged fix must carry the constant on a line of its own"
RT_E7 = real_tree(SCR / "rt_e7")
RT_FIX = real_tree(SCR / "rt_fix", fix=True)
RT_ZERO = real_tree(SCR / "rt_zero", fix=True, data_edit=lambda s: s.replace("\n" + CONST + "\n", "\nEGO_PERTURB_COMPOSED_STREAM = 0\n", 1))
RT_COMM = real_tree(SCR / "rt_comm", fix=True, data_edit=lambda s: s.replace("\n" + CONST + "\n", "\n# " + CONST + "\n", 1))
RT_ROWS = [      # name, tree, accepted?
    ("e7ea20d1 (the r1 tip, no stream separation)", RT_E7, False),
    ("e7ea20d1 + the staged fix (trainer + data)", RT_FIX, True),
    ("the staged fix but the constant is 0", RT_ZERO, False),
    ("the staged fix but the constant line is only a COMMENT", RT_COMM, False),
]


def real_suite(chain, tag):
    """The names of the rows the chain gets wrong (an accepted tree must print 'every one of 43 patterns'; a refused one must exit 1 with the lever check FAILED)."""
    sd = ship2(SCR / f"ship_rt_{tag}", chain=chain)
    bad = []
    for name, tr, ok in RT_ROWS:
        rc, out = run_in(sd, f'--check-tree "{bp(tr)}"')
        got = rc == 0 and "every one of 43 patterns" in out
        refused = rc == 1 and "lever check: FAILED" in out
        if (ok and not got) or (not ok and not refused):
            bad.append(name)
    return bad


rc, out = run_in(SD, f'--check-tree "{bp(RT_E7)}"')
T("C2-12 the REGISTERED STREAM_PATTERN on the real e7ea20d1 tree: REFUSED, and the stream pattern is the ONLY one missing (every other lever grep is satisfied, so the refusal is about the fix)",
  rc == 1 and out.count("MISSING in ") == 1 and "MISSING in stack/tanitad/data/v8f_mini_data.py: ^EGO_PERTURB_COMPOSED_STREAM" in out and "lever check: FAILED" in out, out[-400:])
rc, out = run_in(SD, f'--check-tree "{bp(RT_FIX)}"')
T("C2-13 the REGISTERED STREAM_PATTERN on the e7ea20d1 tree + the staged fix (trainer + data): PASS, every one of 43 patterns (42 lever greps + STREAM_PATTERN)", rc == 0 and "every one of 43 patterns" in out, out[-300:])
bad = real_suite(DEFAULT_CHAIN, "reg")
T(f"C2-14 the registered pattern gives the registered answer on all {len(RT_ROWS)} real trees (constant 0 and a commented-out constant are REFUSED)", bad == [], bad)
rc, out = sh(f'cd "{bp(R1SHIP)}" && bash ./chain_v8f_dagoaccorc.sh --check-tree "{bp(RT_E7)}"')
T("C2-15 the r1 chain's own lever greps accept the e7ea20d1 tree (it is the tree r1 ran on: the r2 refusal is the ONE extra pattern)", rc == 0 and "every one of 42 patterns" in out, out[-200:])
PAT_LINE = "STREAM_PATTERN='" + REG_PATTERN + "'"
assert DEFAULT_CHAIN.count(PAT_LINE) == 1
for i, (name, newpat) in enumerate((
        ("is satisfied by e7ea20d1 (^class EgoPerturb)", "stack/tanitad/data/v8f_mini_data.py|^class EgoPerturb"),
        ("accepts the constant at 0 or 1 ([01])", "stack/tanitad/data/v8f_mini_data.py|^EGO_PERTURB_COMPOSED_STREAM = [01][[:space:]]*$"),
        ("is unanchored (a commented-out constant passes)", "stack/tanitad/data/v8f_mini_data.py|EGO_PERTURB_COMPOSED_STREAM = 1[[:space:]]*$"),
        ("greps the wrong file (v8f_dagger.py)", "stack/tanitad/data/v8f_dagger.py|^EGO_PERTURB_COMPOSED_STREAM = 1[[:space:]]*$"))):
    cm = DEFAULT_CHAIN.replace(PAT_LINE, "STREAM_PATTERN='" + newpat + "'")
    assert cm != DEFAULT_CHAIN
    bad = real_suite(cm, f"m{i}")
    M(f"MC-r{i + 1} the registered STREAM_PATTERN {name}: the real-tree table catches it", len(bad) >= 1, bad)
cm = DEFAULT_CHAIN.replace('for rec in "${LEVER[@]}" "$STREAM_PATTERN"; do', 'for rec in "${LEVER[@]}"; do')
assert cm != DEFAULT_CHAIN
bad = real_suite(cm, "m_loop")
M("MC-r5 the lever loop forgets STREAM_PATTERN (on the registered chain): the real-tree table catches it (e7ea20d1 would be accepted)", len(bad) >= 1, bad)

# the real run refuses BEFORE it waits for, takes or ships anything while a STREAM variable is the placeholder
AFTER_NONE = bp(SCR / "no_such_r1_log.out")
rc, out = run_in(SDP, "", dict(AFTER_OUT=R1LOG_P), timeout=60)
T("C2-8 a real run with STREAM_PATTERN unset STOPs at once, before any wait, naming it; nothing was taken (no BOX2_BUSY.txt, no logs/, no art/)",
  rc == 1 and "ZZV8FMINI-STOP STREAM_PATTERN is still the placeholder" in out and "waiting for the r1 chain" not in out and not (SDP / "BOX2_BUSY.txt").exists() and not (SDP / "logs").exists() and not (SDP / "art").exists(), out[-300:])
rc, out = run_in(ship2(SCR / "ship_pr", chain=MC.render(SP, MC.PH_RECORD)), "", dict(AFTER_OUT=R1LOG_P), timeout=60)
T("C2-9 a real run with STREAM_PATTERN set but STREAM_RECORD unset STOPs at once, naming STREAM_RECORD", rc == 1 and "ZZV8FMINI-STOP STREAM_RECORD is still the placeholder" in out and "waiting for the r1 chain" not in out, out[-300:])
rc, out = run_in(SD, "", dict(AFTER_OUT=""), timeout=60)
T("C2-10 a real run with AFTER_OUT unset STOPs (no default predecessor)", rc == 1 and "AFTER_OUT is not set" in out, out[-250:])
mut = PH_CHAIN.replace("stream_vars_check || exit 1      # r2:", ": # MUTANT      # r2:")
assert mut != PH_CHAIN
rc, out = run_in(ship2(SCR / "ship_m", chain=mut), "", dict(AFTER_OUT=AFTER_NONE, OAF_POLL_S="1"), kill_after=15)
M("MC-s4 the early stream_vars_check is removed: the placeholder chain goes on to WAIT for the predecessor (the C2-8 refusal is gone)", not (rc == 1 and "STREAM_PATTERN is still the placeholder" in out), out[-200:])

# a changed helper is a changed chain
for f, why in (("gate_dagoaccorc_r2_assert.py", "reviewed asserter"), ("v8f_dagoaccorc_r2_chain_checks.py", "reviewed checks"), ("oaf_dagoaccorc_r2_lib.sh", "reviewed r2 lib"), ("oaf_lib.sh", "proven file"), ("v8f_oaf_chain_checks.py", "proven file")):
    sd = ship2(SCR / "ship_m", tamper=f)
    rc, out = run_in(sd, "--dry-run", dict(V8F_QUEUE=qf, AFTER_OUT="x"))
    M(f"MC-t {f} tampered: the chain STOPs before touching anything ({why})", rc == 1 and "ZZV8FMINI-STOP" in out and why.split()[-1] in out, out[-250:])
for f in ("oaf_dagoaccorc_r2_lib.sh", "v8f_dagoaccorc_r2_chain_checks.py", "gate_dagoaccorc_r2_assert.py"):
    sd = ship2(SCR / "ship_m", rm=(f,))
    rc, out = run_in(sd, "--dry-run", dict(V8F_QUEUE=qf, AFTER_OUT="x"))
    M(f"MC-t {f} missing: the chain STOPs (nothing was touched)", rc == 1 and "ZZV8FMINI-STOP" in out, out[-250:])
# chain-level pin mutants
for name, old, new in (("w_acc pinned as 50", "WACC=49.6969", "WACC=50"), ("seed-0 bank pinned wrong", "BANK1_SHA=(197a68bb", "BANK1_SHA=(297a68bb")):
    assert DEFAULT_CHAIN.count(old) == 1, name
    rc, out = run_in(ship2(SCR / "ship_m", chain=DEFAULT_CHAIN.replace(old, new)), "--dry-run", dict(V8F_QUEUE=qf, AFTER_OUT="x"))
    M(f"MC chain mutant: {name}: the dry run STOPs (not the one-lever argv)", rc == 1 and "ZZV8FMINI-STOP the planned lines are not the one-lever" in out, out[-250:])
rc, out = run_in(ship2(SCR / "ship_m", chain=DEFAULT_CHAIN.replace('RUNS=("dagoaccorc 0" "dagoaccorcrep 0"', 'RUNS=("dagoaccorcrep 0" "dagoaccorc 0"')), "--dry-run", dict(V8F_QUEUE=qf, AFTER_OUT="x"))
M("MC chain mutant: runs in another order: the dry run prints the wrong order (the registered one is absent)", "== runs (order): dagoaccorc 0 dagoaccorcrep 0 dagoaccorc 1 dagoaccorcrep 1" not in out)

# order: the aside is the FIRST action after BOX2_BUSY is taken and the boot probe answered
txt = DEFAULT_CHAIN
LATER = ['disk_floor "$ART" "$NEED_BYTES" "the four DAG-ORC (DAG-OACC base) runs (re-checked', "mkdir -p $E/runs", "git -C $MIRROR fetch", "archive --format=tar -o $B/repo_v8fdagoaccorc_r2_tip.tar",
         "launch telemetry_long", '> "$ART/K-BUILD.txt"', "QT=$B/v8fdagoaccorc_queue_$TAG.cmd"]


def order_ok(t):
    i_pred, i_busy, i_boot, i_aside = t.find("\nwait_predecessor\n"), t.find("( set -C; echo \"v8F-mini PREREG_V8F_DAG_ORC"), t.find("BOOT0=$(boot); [ -n"), t.find("\nr2_aside_all\n")
    if min(i_pred, i_busy, i_boot, i_aside) < 0 or t.count("\nr2_aside_all\n") != 1:
        return False
    lat = [t.find(x) for x in LATER]
    return i_pred < i_busy < i_boot < i_aside < min(lat) and min(lat) > 0


T("C2-11 r2_aside_all is called exactly once, AFTER wait_predecessor, BOX2_BUSY and the boot probe and BEFORE the disk re-check, mkdir $E, the fetch, the archive, telemetry, K-BUILD and the queue copy", order_ok(txt))
mv = txt.replace("\nr2_aside_all\n", "\n", 1).replace("git -C $MIRROR fetch -q origin $BR;", "r2_aside_all\ngit -C $MIRROR fetch -q origin $BR;", 1)
M("MC-o1 r2_aside_all moved to after the code refresh starts (the fetch): the order check refuses", mv != txt and not order_ok(mv))
mv = txt.replace("\nr2_aside_all\n", "\n", 1).replace("BOOT0=$(boot); [ -n", "r2_aside_all\nBOOT0=$(boot); [ -n", 1)
M("MC-o2 r2_aside_all moved BEFORE the boot probe: the order check refuses", mv != txt and not order_ok(mv))
M("MC-o3 r2_aside_all deleted from the chain: the order check refuses", not order_ok(txt.replace("\nr2_aside_all\n", "\n", 1)))

# ====================================================================== X1. the r2 predecessor rule
print("== X1: the r2 predecessor rule (GO only after the r1 chain's K-REACH stop)")
BASE = R1LOG.read_text(encoding="utf-8", errors="replace").replace("\r", "")
DONE_LOG = OACC_TAIL.read_text(encoding="utf-8", errors="replace").replace("\r", "")
STOP = C.EXPECTED_R1_STOP
LINES = BASE.rstrip("\n").split("\n")
assert LINES[-1] == STOP and LINES[-2].startswith(C.EXPECTED_R1_CTRL_PREFIX), "the snapshot of the r1 log must end in the K-REACH stop"


def pl(*parts):
    return "\n".join(parts) + "\n"


KF = next(x for x in LINES if "K-REACH: FAIL | dagoaccorc |" in x)
CF = LINES[-2]
PRED_ROWS = [
    ("r1's real log, BOX2_BUSY released", BASE, False, "GO"),
    ("r1's real log with CRLF line ends", BASE.replace("\n", "\r\n"), False, "GO"),
    ("r1's real log, BOX2_BUSY still held", BASE, True, "BUSY"),
    ("r1's log before its end (the last two lines cut)", "\n".join(LINES[:-2]) + "\n", False, "WAIT"),
    ("an empty log", "", False, "WAIT"),
    ("r1's log + a DONE line", BASE + "ZZV8FMINI-DONE 10-28 03:00 tip x: all four\n", False, "HALT"),
    ("the DAG-OACC r2 chain's log (it ENDED in DONE)", DONE_LOG, False, "HALT"),
    ("a different STOP (gate assertions M2b_s0_dagoaccorc)", "\n".join(LINES[:-1]) + "\nZZV8FMINI-STOP gate DAG-ORC (DAG-OACC base) assertions M2b_s0_dagoaccorc\n", False, "HALT"),
    ("the K-REACH STOP of the REPLICATE (M2b_s0_dagoaccorcrep)", "\n".join(LINES[:-1]) + "\nZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorcrep\n", False, "HALT"),
    ("the K-REACH STOP of seed 1 (M2b_s1_dagoaccorc)", "\n".join(LINES[:-1]) + "\nZZV8FMINI-STOP reached-control fail M2b_s1_dagoaccorc\n", False, "HALT"),
    ("a PREFIX variant of the stop (...M2b_s0_dagoaccorc2)", "\n".join(LINES[:-1]) + "\nZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc2\n", False, "HALT"),
    ("text AFTER the stop (the log does not END in it)", BASE + "10-28 03:00 something else happened\n", False, "HALT"),
    ("TWO stop lines", BASE + STOP + "\n", False, "HALT"),
    ("a ZZGUARD-STOP in the log", pl(*LINES[:-2], "ZZGUARD-STOP TEMP 10-28 02:00 USB HDD: 5 of the last 20 samples >= 61 C", CF, STOP), False, "HALT"),
    ("a ZZDISK-FLOOR-STOP in the log", pl(*LINES[:-2], "ZZDISK-FLOOR-STOP pulling x would leave less than 1 GiB", CF, STOP), False, "HALT"),
    ("the stop PRECEDES the control line and text follows (the log does not END in the stop)", pl(*LINES[:-2], STOP, CF, "10-28 03:00 trailing"), False, "HALT"),
    ("an extra STOP line near the start of an otherwise perfect log", pl(LINES[0], STOP, *LINES[1:]), False, "HALT"),
    ("a DONE line in the middle of an otherwise perfect log", pl(*LINES[:3], "ZZV8FMINI-DONE x", *LINES[3:]), False, "HALT"),
    ("a ZZGUARD-STOP mention in the middle of an otherwise perfect log", pl(*LINES[:3], "ZZGUARD-STOP TEMP x", *LINES[3:]), False, "HALT"),
    ("the CONTROL-FAIL line removed (K-REACH FAIL still there)", pl(*LINES[:-2], STOP), False, "HALT"),
    ("the K-REACH FAIL line removed (CONTROL-FAIL still there)", pl(*[x for x in LINES if x != KF]), False, "HALT"),
    ("the CONTROL-FAIL line names another run (M2b_s1_dagoaccorc)", pl(*LINES[:-2], CF.replace("M2b_s0_dagoaccorc:", "M2b_s1_dagoaccorc:"), STOP), False, "HALT"),
    ("the stop line prefixed by a date (not anchored: not a stop)", pl(*LINES[:-1], "10-28 02:09 " + STOP), False, "WAIT"),
    ("an unrelated log that only ECHOES the marker text", "10-28 01:16 waiting for ^ZZV8FMINI-DONE / ^ZZV8FMINI-STOP\n", False, "WAIT"),
]


def pred_suite(mod):
    return [n for n, log, busy, want in PRED_ROWS if mod.pred_verdict_r2(log, busy)[0] != want]


bad = pred_suite(C)
T(f"X1-1 pred_verdict_r2 gives the registered verdict on all {len(PRED_ROWS)} logs (GO only for r1's real log)", bad == [], bad)
LIVE = INST / "logs" / "chain_v8f_dagoaccorc.out"
if LIVE.exists():
    print("     live r1 log sha256 == snapshot:", hashlib.sha256(LIVE.read_bytes()).hexdigest() == hashlib.sha256(R1LOG.read_bytes()).hexdigest())
CSRC = (R2SHIP / "v8f_dagoaccorc_r2_chain_checks.py").read_text(encoding="utf-8")
LIBSRC = (R2SHIP / "oaf_dagoaccorc_r2_lib.sh").read_text(encoding="utf-8")


def load_mod(src, tag):
    d = SCR / "mods" / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(R2SHIP / "gate_dagoaccorc_r2_assert.py", d / "gate_dagoaccorc_r2_assert.py")
    (d / f"chk_{tag}.py").write_text(src, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(f"chk_{tag}", d / f"chk_{tag}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmut(old, new):
    assert CSRC.count(old) == 1, old
    return CSRC.replace(old, new)


PRED_MUTS = [
    ("does not require the log to END in exactly the expected stop (any single stop, text after it)", cmut("if not body or body[-1] != EXPECTED_R1_STOP:", "if False:")),
    ("accepts a stop that merely STARTS with the expected text", cmut("if not body or body[-1] != EXPECTED_R1_STOP:", "if not body or not body[-1].startswith(EXPECTED_R1_STOP):")),
    ("ignores a DONE", cmut("    if done:\n        return \"HALT\"", "    if False:\n        return \"HALT\"")),
    ("ignores a guard stop", cmut("    if guard:\n        return \"HALT\"", "    if False:\n        return \"HALT\"")),
    ("ignores a disk-floor stop", cmut("    if floor:\n        return \"HALT\"", "    if False:\n        return \"HALT\"")),
    ("accepts several stop lines", cmut("if len(stop) != 1:", "if False:")),
    ("ignores BOX2_BUSY", cmut("    if busy_exists:\n        return \"BUSY\"", "    if False:\n        return \"BUSY\"")),
    ("does not require the CONTROL-FAIL line", cmut("if len(ctrl) != 1 or len(body) < 2 or body[-2] != ctrl[0] or not ctrl[0].startswith(EXPECTED_R1_CTRL_PREFIX):", "if False:")),
    ("does not require the K-REACH FAIL line", cmut("    if not kfail:\n", "    if False:\n")),
    ("expects the stop of seed 1", cmut('EXPECTED_R1_STOP = "ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc"', 'EXPECTED_R1_STOP = "ZZV8FMINI-STOP reached-control fail M2b_s1_dagoaccorc"')),
    ("treats no stop as GO", cmut('    if not stop:\n        return "WAIT", "no end marker yet"', '    if not stop:\n        return "GO", "no end marker yet"')),
]
for i, (name, src) in enumerate(PRED_MUTS):
    M(f"MX1-{i + 1:02d} pred_verdict_r2 {name}: the predecessor table catches it", len(pred_suite(load_mod(src, f"pred{i}"))) >= 1)

# the chain-level wait (--wait-only) end to end
SDW = ship2(SCR / "ship_w")
rc, out = run_in(SDW, "--wait-only", dict(AFTER_OUT=R1LOG_P, OAF_POLL_S="1", OAF_GRACE_S="1"), timeout=60)
T("X1-3 --wait-only on r1's real log: GO (ZZOAF-WAIT-GO)", rc == 0 and "ZZOAF-WAIT-GO" in out and "predecessor verdict: GO" in out, out[-300:])
rc, out = run_in(SDW, "--wait-only", dict(AFTER_OUT=bp(OACC_TAIL), OAF_POLL_S="1", OAF_GRACE_S="1"), timeout=60)
T("X1-4 --wait-only on a log that ended in DONE: HALT (STOP, no GO)", rc == 1 and "ZZOAF-WAIT-GO" not in out and "HALT" in out and "ZZV8FMINI-STOP" in out, out[-300:])
other = SCR / "other_stop.out"
other.write_text("\n".join(LINES[:-1]) + "\nZZV8FMINI-STOP gate DAG-ORC (DAG-OACC base) assertions M2b_s0_dagoaccorc\n", encoding="utf-8", newline="\n")
rc, out = run_in(SDW, "--wait-only", dict(AFTER_OUT=bp(other), OAF_POLL_S="1", OAF_GRACE_S="1"), timeout=60)
T("X1-5 --wait-only on a log that ended in ANOTHER stop: HALT (the generic oaf_lib wait would have GONE)", rc == 1 and "ZZOAF-WAIT-GO" not in out and "HALT" in out, out[-300:])
(SDW / "BOX2_BUSY.txt").write_text("held")
rc, out = run_in(SDW, "--wait-only", dict(AFTER_OUT=R1LOG_P, OAF_POLL_S="1", OAF_GRACE_S="1"), timeout=60)
T("X1-6 --wait-only on r1's real log while BOX2_BUSY.txt is held: no GO (STOP after the re-checks)", rc == 1 and "ZZOAF-WAIT-GO" not in out and "still held" in out, out[-300:])
(SDW / "BOX2_BUSY.txt").unlink()
rc, out = run_in(SDW, "--wait-only", dict(AFTER_OUT=bp(SCR / "no_log_yet.out"), OAF_POLL_S="1", OAF_GRACE_S="1"), kill_after=10)
T("X1-7 --wait-only with no r1 log / no end marker: keeps WAITING (no GO, killed by the timeout)", rc == 124 and "ZZOAF-WAIT-GO" not in out and "log present: NO" in out, (rc, out[-200:]))

# ====================================================================== X2. the stream-record clause
print("== X2: the stream-record clause of the run-record check (registered spec levers.ego_perturb_stream=1;0, the typed A / REP grammar, summary.json AND selfcheck.json)")
SPEC_T = "levers.ego_perturb_stream=true"
SPEC_ANY = "levers.ego_perturb_stream"
SPEC_S = 'levers.ego_perturb_stream="domain_separated"'
SPEC_R = REG_RECORD                      # levers.ego_perturb_stream=1;0   (A: exactly 1, REP: exactly 0)


def smry(v=None, absent=False, extra=None):
    sm = {"levers": {"w_acc": 49.6969}}
    if not absent:
        sm["levers"]["ego_perturb_stream"] = v
    return sm


STREAM_ROWS = [      # name, summary, p, spec, expect no problems
    # ---- the REGISTERED spec: A exactly the int 1, REP exactly the int 0, absent fails on both, typed
    ("REG A: 1", smry(1), 0.3, SPEC_R, True),
    ("REG A: absent (the r1 trainer at e7ea20d1 recorded no such field)", smry(absent=True), 0.3, SPEC_R, False),
    ("REG A: 0 (the original, shared stream)", smry(0), 0.3, SPEC_R, False),
    ("REG A: True (== 1 in python, not the int 1)", smry(True), 0.3, SPEC_R, False),
    ("REG A: 1.0 (a float, not the int 1)", smry(1.0), 0.3, SPEC_R, False),
    ("REG A: 2", smry(2), 0.3, SPEC_R, False),
    ("REG A: null", smry(None), 0.3, SPEC_R, False),
    ("REG A: the string '1'", smry("1"), 0.3, SPEC_R, False),
    ("REG A: a nested path that does not exist", smry(1), 0.3, "levers.nope.deeper=1;0", False),
    ("REG REP: 0", smry(0), 0.0, SPEC_R, True),
    ("REG REP: absent (a REP literal is registered: absent FAILS)", smry(absent=True), 0.0, SPEC_R, False),
    ("REG REP: False (== 0 in python, not the int 0)", smry(False), 0.0, SPEC_R, False),
    ("REG REP: 0.0 (a float, not the int 0)", smry(0.0), 0.0, SPEC_R, False),
    ("REG REP: null", smry(None), 0.0, SPEC_R, False),
    ("REG REP: 1 (the replicate carries the separated stream)", smry(1), 0.0, SPEC_R, False),
    ("REG REP: the string '0'", smry("0"), 0.0, SPEC_R, False),
    # ---- the older grammars (no REP literal) are still handled
    ("A: literal true, field true", smry(True), 0.3, SPEC_T, True),
    ("A: literal true, field absent", smry(absent=True), 0.3, SPEC_T, False),
    ("A: literal true, field false", smry(False), 0.3, SPEC_T, False),
    ("A: literal true, field null", smry(None), 0.3, SPEC_T, False),
    ("A: any, field true", smry(True), 0.3, SPEC_ANY, True),
    ("A: any, field 'domain_separated'", smry("domain_separated"), 0.3, SPEC_ANY, True),
    ("A: any, field absent", smry(absent=True), 0.3, SPEC_ANY, False),
    ("A: any, field false", smry(False), 0.3, SPEC_ANY, False),
    ("A: any, field ''", smry(""), 0.3, SPEC_ANY, False),
    ("A: any, field 'shared'", smry("shared"), 0.3, SPEC_ANY, False),
    ("A: any, field 'none'", smry("none"), 0.3, SPEC_ANY, False),
    ("A: string literal, equal", smry("domain_separated"), 0.3, SPEC_S, True),
    ("A: string literal, different", smry("other"), 0.3, SPEC_S, False),
    ("A: NONE (a trainer that records no field), field absent", smry(absent=True), 0.3, "NONE", True),
    ("A: empty spec = skipped", smry(absent=True), 0.3, "", True),
    ("A: the placeholder spec", smry(1), 0.3, MC.PH_RECORD, False),
    ("REP (no REP literal): field absent", smry(absent=True), 0.0, SPEC_T, True),
    ("REP (no REP literal): field false", smry(False), 0.0, SPEC_T, True),
    ("REP (no REP literal): field null", smry(None), 0.0, SPEC_ANY, True),
    ("REP (no REP literal): field true (the replicate carries the separated stream)", smry(True), 0.0, SPEC_T, False),
    ("REP (no REP literal): field 'domain_separated'", smry("domain_separated"), 0.0, SPEC_ANY, False),
    ("REP: NONE", smry(1), 0.0, "NONE", True),
    ("REP: the placeholder spec", smry(absent=True), 0.0, MC.PH_RECORD, False),
    ("REP: a '<<' spec with no literal at all (only the placeholder guard refuses it)", smry(absent=True), 0.0, "levers.ego_perturb_stream<<", False),
]


def stream_suite(mod):
    return [n for n, sm, p, spec, ok in STREAM_ROWS if (mod.stream_problems(sm, p, spec) == []) != ok]


T(f"X2-1 stream_problems gives the registered answer on all {len(STREAM_ROWS)} rows (registered spec: A exactly the int 1, REP exactly the int 0, typed, absent fails on both)", stream_suite(C) == [], stream_suite(C))
STREAM_MUTS = [
    ("A without the field passes", cmut('return [f"the run record carries no {path}: the separated perturbation stream is not recorded on A"]', "return []")),
    ("REP without the field passes (with a REP literal registered)", cmut('return [f"the run record carries no {path}: REP must record exactly {want!r}"]', "return []")),
    ("the A literal is ignored", cmut("        if lit:\n            try:", "        if False:\n            try:")),
    ("the REP literal is ignored (REP falls back to the absent / no-stream rule)", cmut("    if rep_sep:\n        try:", "    if False:\n        try:")),
    ("A is compared against the REP literal", cmut("                want = json.loads(lit)\n            except ValueError:\n                return [f\"the stream-record literal", "                want = json.loads(lit_rep or lit)\n            except ValueError:\n                return [f\"the stream-record literal")),
    ("the A compare is loose (1 == True == 1.0)", cmut("            if json.dumps(cur, sort_keys=True) != json.dumps(want, sort_keys=True):\n                return [f\"the run record's {path} = {cur!r} on A,", "            if cur != want:\n                return [f\"the run record's {path} = {cur!r} on A,")),
    ("the REP compare is loose (0 == False == 0.0)", cmut("        if json.dumps(cur, sort_keys=True) != json.dumps(want, sort_keys=True):\n            return [f\"the run record's {path} = {cur!r} on REP,", "        if cur != want:\n            return [f\"the run record's {path} = {cur!r} on REP,")),
    ("REP carrying the stream passes (no REP literal)", cmut("    if found and not _no_stream(cur):", "    if False:")),
    ("the placeholder spec passes", cmut('    if "<<" in spec:', "    if False:")),
    ("NONE is not skipped", cmut('    if spec in ("", "NONE"):', '    if spec == "":')),
    ("an empty field value counts as a stream (A, no literal)", cmut("        if _no_stream(cur):\n            return [f\"the run record's {path} = {cur!r} on A: that is not a separated stream\"]", "        if False:\n            return []")),
    ("'shared' counts as a separated stream", cmut('NO_STREAM_STR = ("", "none", "off", "shared")', 'NO_STREAM_STR = ("", "none", "off")')),
]
for i, (name, src) in enumerate(STREAM_MUTS):
    M(f"MX2-{i + 1:02d} stream_problems {name}: the stream table catches it", len(stream_suite(load_mod(src, f"str{i}"))) >= 1)

# at the run-record level (the clause is CALLED from run_record, for summary.json AND selfcheck.json, with the argument the chain passes)


def run_dir(root, arm, seed, mut_summary=None, mut_argv=None, mut_sc=None, train_mut=None, stream="auto"):
    """A run directory built from DAG-OACC's REAL banked records.  stream='auto': the field the FIXED trainer records, in summary.json AND selfcheck.json (1 on A, 0 on REP); None: no field in either."""
    root.mkdir(parents=True, exist_ok=True)
    base = W.OACC / f"M2b_s{seed}_dagoacc"
    train = W.arm_argv(arm, seed)
    if train_mut:
        train = train_mut(list(train))
    sha16 = hashlib.sha256(" ".join(train).encode()).hexdigest()[:16]
    sm = json.load(open(base / "summary.json", encoding="utf-8"))
    sc = json.load(open(base / "selfcheck.json", encoding="utf-8"))
    p = G.ARMS[arm]
    tseed = 10 + seed if arm == "dagoaccorcrep" else seed
    sm.update(argv_sha=sha16, seed=tseed)
    sc.update(argv_sha=sha16)
    if p > 0.0:
        sm["levers"]["ego_perturb"] = {**sm["levers"]["ego_perturb"], "ego_perturb_p": p, **G.LAW_VALUES, "ego_perturb_seed": seed}
        sm["levers"]["ego_perturb_with_dagger"] = True
        sm.update(ego_perturb_n=24000, ego_perturb_eligible=80000, ego_dagger_overlap_n=0)
    if stream is not None:
        v = (1 if p > 0.0 else 0) if stream == "auto" else stream
        sm["levers"]["ego_perturb_stream"] = v
        sc["levers"]["ego_perturb_stream"] = v
    if mut_summary:
        mut_summary(sm)
    if mut_sc:
        mut_sc(sc)
    aj = {"argv": train, "sha": sha16}
    if mut_argv:
        aj = mut_argv(aj)
    for n, o in (("summary.json", sm), ("selfcheck.json", sc), ("argv.json", aj)):
        (root / n).write_text(json.dumps(o), encoding="utf-8", newline="\n")
    for n in ("train_line.txt", "pin_line.txt"):
        (root / n).write_text(W.line(train), encoding="utf-8", newline="\n")
    return root


class NS:
    def __init__(self, **k):
        self.__dict__.update(k)


def rr(root, arm, seed, mod=None, **kw):
    d = W.build(fresh("rr_w"), arm, seed)
    return quiet((mod or C).run_record, NS(run_dir=str(root), train_line=str(root / "train_line.txt"), pin_line=str(root / "pin_line.txt"), arm=arm, seed=str(seed), banked_init_sha=d["banked_init_sha"],
                                           bank_sha=d["bank_sha"], w_acc=W.W_ACC, steps="3000", expect="", **kw))


def put_both(v):
    """mutator for run_dir: the field changed in summary.json AND (via mut_sc) selfcheck.json; v == '<absent>' deletes it."""
    def f(o):
        if v == "<absent>":
            o["levers"].pop("ego_perturb_stream", None)
        else:
            o["levers"]["ego_perturb_stream"] = v
    return f


def rr_suite(mod):
    fails = []
    # name, arm, seed, run_dir kwargs, spec, expect no problems
    for nm, arm, s, kw, spec, ok in (
            ("A with the registered stream (1 / 1)", "dagoaccorc", 0, {}, SPEC_R, True),
            ("A, the field absent from BOTH files", "dagoaccorc", 1, dict(stream=None), SPEC_R, False),
            ("A, summary.json has it, selfcheck.json LACKS it", "dagoaccorc", 0, dict(mut_sc=put_both("<absent>")), SPEC_R, False),
            ("A, selfcheck.json has it, summary.json LACKS it", "dagoaccorc", 0, dict(mut_summary=put_both("<absent>")), SPEC_R, False),
            ("A, summary.json says 0 (the shared stream)", "dagoaccorc", 0, dict(mut_summary=put_both(0)), SPEC_R, False),
            ("A, selfcheck.json says 0 (the shared stream)", "dagoaccorc", 0, dict(mut_sc=put_both(0)), SPEC_R, False),
            ("A, summary.json says True", "dagoaccorc", 1, dict(mut_summary=put_both(True)), SPEC_R, False),
            ("REP with the registered stream (0 / 0)", "dagoaccorcrep", 0, {}, SPEC_R, True),
            ("REP, the field absent from BOTH files", "dagoaccorcrep", 1, dict(stream=None), SPEC_R, False),
            ("REP, summary.json says 1 (carries the separated stream)", "dagoaccorcrep", 1, dict(mut_summary=put_both(1)), SPEC_R, False),
            ("REP, selfcheck.json says 1 (carries the separated stream)", "dagoaccorcrep", 1, dict(mut_sc=put_both(1)), SPEC_R, False),
            ("A, NONE skips (no field in either file)", "dagoaccorc", 0, dict(stream=None), "NONE", True)):
        got = rr(run_dir(fresh("rrs"), arm, s, **kw), arm, s, mod=mod, stream_record=spec)
        if (got == []) != ok:
            fails.append(nm)
    return fails


T("X2-2 run_record with the registered spec, on DAG-OACC's real records + the stream field: A (1 / 1) and REP (0 / 0) pass; absent from either file, the shared stream (0) on A, the separated stream (1) on REP, True on A FAIL; NONE skips", rr_suite(C) == [], rr_suite(C))
T("X2-3 run_record WITHOUT the argument (the K section) is r1's: passes on the real records (the clause is skipped, nothing else changed)", rr(run_dir(fresh("rr0"), "dagoaccorc", 0, stream=None), "dagoaccorc", 0) == [])
CALL_SM = '    bad += stream_problems(sm, p, getattr(a, "stream_record", ""))      # r2: the separated perturbation stream (summary.json) ...\n'
CALL_SC = '    bad += ["selfcheck.json: " + x for x in stream_problems(sc, p, getattr(a, "stream_record", ""))]      # ... and the run\'s own selfcheck.json\n'
M("MX2-13 run_record never CALLS the stream clause on summary.json: the run-record table catches it", len(rr_suite(load_mod(cmut(CALL_SM, ""), "rrcall"))) >= 1)
M("MX2-14 run_record never CALLS the stream clause on selfcheck.json: the run-record table catches it (summary-only would miss a selfcheck that lacks the field)", len(rr_suite(load_mod(cmut(CALL_SC, ""), "rrcall2"))) >= 1)
M("MX2-15 run_record passes the arm's p of the OTHER arm to the clause (p fixed at 0.3): the REP rows catch it", len(rr_suite(load_mod(cmut(CALL_SM, CALL_SM.replace("stream_problems(sm, p,", "stream_problems(sm, 0.3,")), "rrp"))) >= 1)
mut = DEFAULT_CHAIN.replace(' --stream-record "$STREAM_RECORD"', "", 1)
M("MX2-16 the chain does not pass --stream-record to run-record: the text check refuses", mut != DEFAULT_CHAIN and ' --stream-record "$STREAM_RECORD"' not in mut and ' --stream-record "$STREAM_RECORD"' in DEFAULT_CHAIN)

# ====================================================================== X3. the ASIDE, run for real on a scratch tree
print("== X3: r2_aside_all (dev box rename + the box-2 cmd job) on a scratch tree, with the REAL lib text and cmd.exe")
RUNS_CMD = r"D:\tanitad\runs\v8f_mini_full"
ARMS2 = ("dagoaccorc", "dagoaccorcrep")


def put(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


def r1_state(world, full=True):
    """The state r1 left: dev box ART and box 2's runs root (A s0 trained+evaluated, the four gates, pins). full=False: only the gates (no run, no eval)."""
    art, runs = world / "ART", world / "runs"
    put(art / "K-BUILD.txt", "k-build\n")
    put(art / "M2b_s0_dagoaccorc" / "ckpt_last.pt", os.urandom(4096))
    put(art / "M2b_s0_dagoaccorc" / "gate" / "selfcheck.json", '{"r1": true}')
    for s in (0, 1):
        for a in ARMS2:
            put(runs / f"gate_M2b-{a}_s{s}" / "selfcheck.json", f'{{"r1": "{a}{s}"}}')
            put(runs / f"gate_M2b-{a}_s{s}.log", f"gate log {a}{s}\n")
            for suf in ("", ".new", ".chk"):
                put(runs / "dagpin" / f"M2b_s{s}_{a}.train.argv.txt{suf}", f"pin {a}{s}{suf}\n")
    if full:
        put(runs / "M2b_s0_dagoaccorc" / "ckpt_last.pt", os.urandom(8192))
        put(runs / "M2b_s0_dagoaccorc" / "summary.json", '{"done": true}')
        put(runs / "M2b_s0_dagoaccorc" / "train.log", "train\n")
        for ext in ("json", "npz", "log"):
            put(runs / "eval" / f"eval_M2b-dagoaccorc_s0.{ext}", f"eval {ext}\n")
        put(runs / "eval" / "eval_M2b-dagoaccorc_s0_proj.npz", b"proj")
    # DECOYS that must never be touched
    for rel, data in (("M2b_s0/ckpt_last.pt", b"init0"), ("M2b_s1/ckpt_last.pt", b"init1"), ("M2b_s0_dagoacc/ckpt_last.pt", b"oacc"), ("dagger_bank/s0/bank.json", b"bank"), ("dagoacc/w_acc.txt", b"49.6969\n"),
                      ("dagpin/M2b_s0_dagoacc.train.argv.txt", b"oaccpin"), ("dagpin/bank_sha_dagoacc_s0.txt", b"sha"), ("eval/eval_M2b-dagoacc_s0.npz", b"oacceval"), ("gate_M2b-dagoacc_s0/selfcheck.json", b"oaccgate"),
                      ("gate_M2b-dagoacc_s0.log", b"oaccgatelog")):
        put(runs / rel, data)
    return world


def snap(root):
    """the state under test: the dev box's ART trees and box 2's runs root (the harness's own logs / jobs are not part of it)"""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root).as_posix()
        if not (rel.startswith("ART") or rel.startswith("runs")):
            continue
        out[rel] = "DIR" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
    return out


FAKE_SCP = SCR / "fake_scp.py"


def write_harness(world, lib_text):
    put(world / "lib.sh", lib_text.replace("\r", ""))
    put(FAKE_SCP, "import sys\nfrom pathlib import Path\nsrc, jobs, runs = sys.argv[1], sys.argv[2], sys.argv[3]\nb = Path(src).read_bytes()\n"
                  "old = rb'D:\\tanitad\\runs\\v8f_mini_full'\nassert b.count(old) == 1, b.count(old)\nPath(jobs, Path(src).name).write_bytes(b.replace(old, runs.encode()))\n")
    winruns = str(world / "runs").replace("/", "\\")
    h = """set -u
W=%(W)s
B=$W/B; L=$B/logs; mkdir -p $L $W/jobs
TAG=0000; ART=$W/ART; BOOT0=x; BUSY_TAKEN=0; CUR=$L/cur; SSH=false; SCP=fakescp
stop() { echo "ZZV8FMINI-STOP $*"; exit 1; }
source $W/lib.sh
fakescp() { "%(PY)s" "%(FS)s" "$1" "$W/jobs" '%(WR)s'; }
launch() { MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w $W/jobs/$1.cmd)" > $L/$1.log 2>&1 < /dev/null; echo "launched $1"; return 0; }
clean() { tr '\\r' '\\n' < "$L/$1.log"; }
waitrc() { local rc; rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\\([-0-9]*\\).*/\\1/p'); [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; stop "$1 rc=$rc"; }; }
r2_aside_all
echo HARNESS-END
""" % dict(W=bp(world), PY=PYV, FS=bp(FAKE_SCP), WR=winruns)
    put(world / "harness.sh", h)
    return world / "harness.sh"


def aside(world, lib_text=None):
    hs = write_harness(world, lib_text if lib_text is not None else LIBSRC)
    return sh(f'bash "{bp(hs)}"', timeout=180)


def items_present_r1(runs):
    """box-2 items of r1 that exist in the r1 state (relative, '/' form)"""
    return [rel for rel in (x.replace("\\", "/") for x in LIB_ITEMS) if (runs / rel).exists()]


def lib_items(lib_text):
    return [x.replace("\\", "/") for x in re.findall(r"^    '([^']+)'(?: \\)?$", lib_text, re.M)]


LIB_ITEMS = [x.replace("/", "\\") for x in lib_items(LIBSRC)]
T("X3-0 the lib lists 44 box-2 items: 4 run dirs, 4 gate dirs + 4 logs, 12 pin files, 20 eval files", len(LIB_ITEMS) == 44 and len({*LIB_ITEMS}) == 44, len(LIB_ITEMS))
SUFX = "_r1_kreach_stop"


def aside_checks(tag, lib_text=None, expect_mutant=False):
    """Run scenarios S1..S3 on a fresh world; return the list of failed expectations (empty = the registered behaviour)."""
    fails = []
    w = r1_state(fresh(f"aside_{tag}"))
    before = snap(w)
    present = items_present_r1(w / "runs")
    rc, out = aside(w, lib_text)
    after = snap(w)
    # S1: first launch
    n_aside = len(re.findall(r"^ASIDE-", out, re.M))
    if rc != 0 or "HARNESS-END" not in out:
        fails.append(f"S1 rc {rc}: {out[-300:]}")
    if not (w / "ART_r1_kreach_stop" / "K-BUILD.txt").exists() or (w / "ART" / "K-BUILD.txt").exists() or not (w / "ART").is_dir():
        fails.append("S1 the dev box ART was not renamed aside and re-created empty")
    for rel in present:                           # every r1 artifact renamed, content preserved byte for byte
        old, new = w / "runs" / rel, w / "runs" / (rel + SUFX)
        if old.exists() or not new.exists():
            fails.append(f"S1 {rel} not moved aside")
        elif before["runs/" + rel] != after.get("runs/" + rel + SUFX):
            fails.append(f"S1 {rel} content changed")
    for rel, h in before.items():                 # nothing was lost: every file of the r1 state is somewhere under its new name or untouched (decoys)
        if rel.startswith("ART/"):
            if after.get("ART_r1_kreach_stop/" + rel[4:]) != h:
                fails.append(f"S1 ART file {rel} lost / changed")
        elif rel.startswith("runs/") and h != "DIR":
            tail = rel[5:]
            is_item = any(tail == i or tail.startswith(i + "/") for i in present)
            if is_item:
                top = next(i for i in present if tail == i or tail.startswith(i + "/"))
                moved = "runs/" + top + SUFX + tail[len(top):]
                if after.get(moved) != h:
                    fails.append(f"S1 {rel} lost (expected {moved})")
            elif after.get(rel) != h:
                fails.append(f"S1 decoy {rel} touched")
    if not (w / "runs" / "r2_aside_done.txt").exists():
        fails.append("S1 no aside marker")
    if n_aside < 44 + 1:
        fails.append(f"S1 only {n_aside} ASIDE lines (44 box-2 items + the local one)")
    # r2 starts writing its own files
    put(w / "ART" / "r2_progress.txt", "r2\n")
    put(w / "runs" / "gate_M2b-dagoaccorc_s0" / "selfcheck.json", '{"r2": true}')
    put(w / "runs" / "M2b_s0_dagoaccorc" / "summary.json", '{"r2": true}')
    snap2 = snap(w)
    rc2, out2 = aside(w, lib_text)
    snap3 = snap(w)
    # S2: relaunch: idempotent, r2's progress untouched
    if rc2 != 0 or "ASIDE-ALREADY-DONE" not in out2 or "ASIDE-KEPT" not in out2:
        fails.append(f"S2 relaunch rc {rc2} / no ALREADY-DONE / no KEPT: {out2[-300:]}")
    if snap3 != snap2:
        fails.append("S2 the relaunch changed the tree: " + str([k for k in set(snap2) | set(snap3) if snap2.get(k) != snap3.get(k)][:4]))
    # S3: marker lost AND an r2 item with its r1 aside copy present: refused, nothing overwritten, no marker
    (w / "runs" / "r2_aside_done.txt").unlink(missing_ok=True)
    rc3, out3 = aside(w, lib_text)
    snap4 = snap(w)
    if not (rc3 != 0 and "ASIDE-FAIL" in out3 and "refusing to overwrite" in out3):
        fails.append(f"S3 not refused (rc {rc3}): {out3[-300:]}")
    if (w / "runs" / "r2_aside_done.txt").exists():
        fails.append("S3 a marker was written after a failure")
    if {k: v for k, v in snap4.items() if k != "runs/r2_aside_done.txt"} != {k: v for k, v in snap3.items() if k != "runs/r2_aside_done.txt"}:
        fails.append("S3 the failed job changed the tree: " + str([k for k in set(snap3) | set(snap4) if snap3.get(k) != snap4.get(k)][:4]))
    return fails


def aside_retry_checks(tag, lib_text=None):
    """S4: a job that died midway (half the items moved, no marker): the retry moves the rest, nothing twice."""
    fails = []
    w = r1_state(fresh(f"aside4_{tag}"))
    runs = w / "runs"
    pre = ["gate_M2b-dagoaccorc_s0", "gate_M2b-dagoaccorc_s0.log", "M2b_s0_dagoaccorc"]
    for rel in pre:
        os.rename(runs / rel, runs / (rel + SUFX))
    rc, out = aside(w, lib_text)
    if rc != 0 or not (runs / "r2_aside_done.txt").exists():
        fails.append(f"S4 retry rc {rc}: {out[-200:]}")
    for rel in pre:
        if not (runs / (rel + SUFX)).exists() or (runs / (rel + SUFX + SUFX)).exists():
            fails.append(f"S4 {rel} moved twice / lost")
    if (runs / "gate_M2b-dagoaccorc_s1").exists() or not (runs / ("gate_M2b-dagoaccorc_s1" + SUFX)).exists():
        fails.append("S4 the rest was not moved")
    return fails


f1 = aside_checks("real")
T("X3-1 the REAL lib: first launch renames EVERY r1 artifact (dev box and box 2) with content preserved and decoys untouched; a relaunch is idempotent and keeps r2's progress; a lost marker + an aside copy is REFUSED without overwriting", f1 == [], f1)
f4 = aside_retry_checks("real")
T("X3-2 the REAL lib: a job that died midway (no marker) is retried: the rest is moved, the moved items are not moved twice", f4 == [], f4)
T("X3-3 the box-2 job text contains no del / rmdir / move / copy / xcopy / robocopy / Remove-Item / rm (renames and the marker line only)",
  not re.search(r"(?i)\b(del|erase|rmdir|rd|move|copy|xcopy|robocopy|remove-item)\b|\brm\b", subprocess.run([BASH, "-c", f'source "{bp(R2SHIP)}/oaf_dagoaccorc_r2_lib.sh"; r2_aside_job_text X'], capture_output=True).stdout.decode("utf-8", "replace")))
import build_r2_lib as BL                            # noqa: E402
import build_r2_checks as BC                         # noqa: E402
T("X3-4 the :aside subroutine is DAG-OACC r2's (modulo the suffix): build_r2_lib.render() asserts it and reproduces the lib on disk byte for byte", BL.render() == LIBSRC)
T("X3-5 the r2 checks file on disk == build_r2_checks.derive(r1's file) byte for byte", BC.derive(BC.R1.read_text(encoding="utf-8")) == CSRC)

LIB_MUTS = [
    ("the eval npz are not in the item list (the queue would SKIP the eval and r2 would pull the refused run's)", lambda t: "\n".join(x for x in t.split("\n") if not re.match(r"^    'eval\\eval_M2b-dagoaccorc_s0\.npz'", x)), "1"),
    ("the run directories are not in the item list (the queue would SKIP the training)", lambda t: "\n".join(x for x in t.split("\n") if not re.match(r"^    'M2b_s0_dagoaccorc' ", x)), "1"),
    ("the pinned argv is not in the item list", lambda t: "\n".join(x for x in t.split("\n") if not re.match(r"^    'dagpin\\M2b_s0_dagoaccorc\.train\.argv\.txt' ", x)), "1"),
    ("the suffix is the OACC one (_r1_assert_stop)", lambda t: t.replace("R2_SUF=_r1_kreach_stop", "R2_SUF=_r1_assert_stop").replace('%~nx1_r1_kreach_stop', '%~nx1_r1_assert_stop').replace('"%~1_r1_kreach_stop"', '"%~1_r1_assert_stop"'), "1"),
    ("the job DELETES instead of renaming", lambda t: t.replace('ren "%~1" "%~nx1_r1_kreach_stop"', 'rmdir /s /q "%~1" 2>nul & del /q "%~1" 2>nul'), "1"),
    ("the job no longer refuses when the aside copy exists", lambda t: re.sub(r"if exist \"%~1_r1_kreach_stop\" \(echo ASIDE-FAIL[^\n]*\n", "", t, count=1), "1"),
    ("the job writes no marker (a relaunch would re-process r2's own files)", lambda t: t.replace("""    echo '>"%RS%\\r2_aside_done.txt" echo r2 aside done %DATE% %TIME%'\n""", ""), "1"),
    ("the dev-box rename is a delete", lambda t: t.replace('  mv "$d" "$n" || stop "r2 aside: cannot rename $d to $n"', '  rm -rf "$d" || stop "x"'), "1"),
    ("the dev-box aside ignores an existing aside copy (mv INTO it)", lambda t: t.replace('  [ -e "$n" ] && { echo "ASIDE-KEPT $d : the r1 copy is already aside as $n"; return 0; }\n', ""), "1"),
    ("$ART is not re-created after the rename", lambda t: t.replace('  mkdir -p "$ART" || stop "r2 aside: cannot create $ART"\n', ""), "1"),
]
for i, (name, fn, _) in enumerate(LIB_MUTS):
    mt = fn(LIBSRC)
    assert mt != LIBSRC, name
    fl = aside_checks(f"m{i}", mt) + aside_retry_checks(f"m{i}", mt)
    M(f"MX3-{i + 1:02d} the aside lib {name}: the scratch-tree scenarios catch it", len(fl) >= 1, fl)
# the chain-level lib pin catches a changed lib; and a lib whose wait falls back to the generic pred-verdict is caught by the --wait-only scenario (needs the lib re-pinned in the chain)
mt = LIBSRC.replace("pred-verdict-r2 --out", "pred-verdict --out", 1)
assert mt != LIBSRC
chain_mt = DEFAULT_CHAIN.replace(MC.sha(R2SHIP / "oaf_dagoaccorc_r2_lib.sh"), hashlib.sha256(mt.encode("utf-8")).hexdigest())
sdm = ship2(SCR / "ship_m", chain=chain_mt)
(sdm / "oaf_dagoaccorc_r2_lib.sh").write_bytes(mt.encode("utf-8"))
rc, out = run_in(sdm, "--wait-only", dict(AFTER_OUT=bp(other), OAF_POLL_S="1", OAF_GRACE_S="1"), timeout=60)
M("MX3-11 the r2 wait falls back to oaf_lib's generic pred-verdict (GO after ANY predecessor stop): --wait-only on a log that ended in another stop must HALT (X1-5)", not (rc == 1 and "ZZOAF-WAIT-GO" not in out and "HALT" in out), out[-250:])

# ====================================================================== X4. r2 invariants vs r1
print("== X4: what r2 must NOT have changed")


def modfile(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C1 = modfile(R1SHIP / "v8f_dagoaccorc_chain_checks.py", "r1_checks")
same_src = {n: inspect.getsource(getattr(C1, n)) == inspect.getsource(getattr(C, n)) for n in ("reached", "_count_pair", "one_lever", "findk", "argv_sha256")}
T("X4-1 K-REACH (reached, _count_pair), one_lever, findk, argv_sha256 are SOURCE-IDENTICAL to r1's checks (the bands 0.1 +- 0.02 / 0.3 +- 0.02 / overlap exactly 0 are unchanged)", all(same_src.values()) and C1.REACH_TOL == C.REACH_TOL == 0.02, same_src)
rr_src1, rr_src2 = inspect.getsource(C1.run_record), inspect.getsource(C.run_record)
T("X4-2 run_record is r1's plus exactly the two stream clauses (summary.json and the run's selfcheck.json)", rr_src2.replace(CALL_SM, "").replace(CALL_SC, "") == rr_src1)
mutm = CSRC.replace("REACH_TOL = 0.02", "REACH_TOL = 0.05")
M("MX4-1 a loosened K-REACH tolerance (0.05) in the r2 checks: the source-identity check refuses it", inspect.getsource(load_mod(mutm, "tol").reached) != inspect.getsource(C1.reached) or load_mod(mutm, "tol").REACH_TOL != C1.REACH_TOL)
import build_r2_asserter as BA                          # noqa: E402
asr_sha = hashlib.sha256((R2SHIP / "gate_dagoaccorc_r2_assert.py").read_bytes()).hexdigest()
r1_asr_txt = (R1SHIP / "gate_dagoaccorc_assert.py").read_text(encoding="utf-8")
r1_asr_sha = hashlib.sha256(r1_asr_txt.encode("utf-8")).hexdigest()
T("X4-3 the r2 asserter == build_r2_asserter.derive(r1's asserter) byte for byte (exact-count insertions only), r1's asserter is UNTOUCHED (the installed one has the same sha256) and differs from r2's",
  BA.derive(r1_asr_txt).encode("utf-8") == (R2SHIP / "gate_dagoaccorc_r2_assert.py").read_bytes() and r1_asr_sha == hashlib.sha256((INST / "gate_dagoaccorc_assert.py").read_bytes()).hexdigest() and r1_asr_sha != asr_sha)
a2 = (R2SHIP / "gate_dagoaccorc_r2_assert.py").read_text(encoding="utf-8")
r1l, r2l = r1_asr_txt.split("\n"), a2.split("\n")
removed = [x for x in r1l if x not in set(r2l)]
T("X4-3b r2's asserter keeps every line of r1's except the docstring's first line and the token dict line (the only two the derivation edits); the rest is added (the stream clause)",
  len(removed) == 2 and removed[0].startswith('"""PREREG_V8F_DAG_ORC on box 2') and '"ego_perturb_with_dagger": ARMS[a.arm] > 0.0' in removed[1] and len(r2l) > len(r1l), removed)

def fn_body(t, name):
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}\n", t, re.M | re.S)
    assert m, name
    return m.group(0)


def arr(t, name):
    m = re.search(rf"^{name}=\(\n(.*?)\n\)\n", t, re.M | re.S)
    return m.group(1) if m else None


def line_of(t, prefix):
    return next(x for x in t.split("\n") if x.startswith(prefix))


inv = {}
inv["gate_one"] = fn_body(R1_CHAIN, "gate_one") == fn_body(DEFAULT_CHAIN, "gate_one")
inv["runone"] = fn_body(R1_CHAIN, "runone").replace("--steps $DAG_STEPS ||", '--steps $DAG_STEPS --stream-record "$STREAM_RECORD" ||') == fn_body(DEFAULT_CHAIN, "runone")
inv["pinned_b"] = fn_body(R1_CHAIN, "pinned_b") == fn_body(DEFAULT_CHAIN, "pinned_b")
inv["plan"] = fn_body(R1_CHAIN, "plan") == fn_body(DEFAULT_CHAIN, "plan")
inv["ckptsha_x"] = fn_body(R1_CHAIN, "ckptsha_x") == fn_body(DEFAULT_CHAIN, "ckptsha_x")
inv["EXPORT"] = arr(R1_CHAIN, "EXPORT") == arr(DEFAULT_CHAIN, "EXPORT")
for var in ("OACC_CKPT_SHA", "OACC_ARGV_SHA", "OACC_SUMM_SHA", "M2B_SHA", "BANK1_SHA", "WACC", "WACC_TXT_SHA", "OAF_LIB_SHA", "OAF_CHECKS_SHA", "BUILD_REF", "RUNS", "DAG_STEPS", "GATE_POLLS", "FLOOR_BYTES", "NEED_BYTES", "ART", "OACC_DIR", "DAG1_DIR"):
    inv[var] = line_of(R1_CHAIN, var + "=") == line_of(DEFAULT_CHAIN, var + "=")
k1 = [x for x in R1_CHAIN.split("\n") if "reached --summary" in x]
inv["K-REACH call"] = k1 == [x for x in DEFAULT_CHAIN.split("\n") if "reached --summary" in x] and len(k1) == 1
lev1, lev2 = arr(R1_CHAIN, "LEVER") or "", arr(DEFAULT_CHAIN, "LEVER") or ""
inv["LEVER"] = lev1 == lev2 and lev1 != ""
inv["ASSERT_SHA is the r2 asserter's (and not r1's)"] = asr_sha in line_of(DEFAULT_CHAIN, "ASSERT_SHA=") and r1_asr_sha not in line_of(DEFAULT_CHAIN, "ASSERT_SHA=")
inv["ASSERT= names the r2 asserter"] = "ASSERT=$HERE_SHIP/gate_dagoaccorc_r2_assert.py" in DEFAULT_CHAIN and "ASSERT=$HERE_SHIP/gate_dagoaccorc_assert.py" not in DEFAULT_CHAIN
T("X4-4 the r2 chain carries r1's pins, RUNS, K-REACH call, LEVER greps, EXPORT, gate_one / runone (+ the one stream argument) / pinned_b / plan / ckptsha_x unchanged", all(inv.values()), [k for k, v in inv.items() if not v])
mutc = DEFAULT_CHAIN.replace("RUNS=(\"dagoaccorc 0\"", "RUNS=(\"dagoaccorcrep 9\" \"dagoaccorc 0\"", 1)
M("MX4-2 a chain whose RUNS differ from r1's: the invariant check refuses it", line_of(R1_CHAIN, "RUNS=") != line_of(mutc, "RUNS="))
mutc = DEFAULT_CHAIN.replace(asr_sha, r1_asr_sha, 1)
M("MX4-3 a chain whose ASSERT_SHA is r1's asserter's (the stream clause is not the pinned one): the invariant check refuses it", not (asr_sha in line_of(mutc, "ASSERT_SHA=") and r1_asr_sha not in line_of(mutc, "ASSERT_SHA=")))
rc, out = run_in(ship2(SCR / "ship_m", chain=mutc), "--dry-run", dict(V8F_QUEUE=qf, AFTER_OUT="x"))
M("MX4-4 a chain pinning r1's asserter sha with the r2 asserter file shipped: the chain STOPs (not the reviewed asserter)", rc == 1 and "ZZV8FMINI-STOP" in out and "is not the reviewed asserter" in out, out[-250:])
mutc = DEFAULT_CHAIN.replace("ASSERT=$HERE_SHIP/gate_dagoaccorc_r2_assert.py", "ASSERT=$HERE_SHIP/gate_dagoaccorc_assert.py", 1)
M("MX4-5 a chain whose ASSERT= names r1's asserter: the invariant check refuses it", not ("ASSERT=$HERE_SHIP/gate_dagoaccorc_r2_assert.py" in mutc and "ASSERT=$HERE_SHIP/gate_dagoaccorc_assert.py" not in mutc))
T("X4-5 the r2 chain's STREAM_* lines are the ONLY unset-able part: make_chain_r2.py --check regenerates the file on disk byte for byte (STREAM_* read from the file)", subprocess.run([PYV, "-I", str(HERE / "make_chain_r2.py"), "--check"], capture_output=True, cwd=str(HERE)).returncode == 0)
T("X4-6 the r2 runner .cmd starts the r2 chain with AFTER_OUT = r1's log and logs to the r2 log (CRLF)", (lambda b: b"\r\n" in b and b"AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc.out bash /c/Users/Admin/box2_ship/chain_v8f_dagoaccorc_r2.sh" in b
                                                                                                       and b"> /c/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out 2>&1" in b)((R2SHIP / "run_chain_v8f_dagoaccorc_r2.cmd").read_bytes()))
# the guard's anchored end test knows the r2 log's end markers
gl = SCR / "guard_probe.out"
gl.write_text("10-28 waiting for ^ZZV8FMINI-DONE\nZZV8FMINI-STOP something\n", encoding="utf-8")
rcg, _ = sh(f"grep -q -E '^(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)' \"{bp(gl)}\"")
gl2 = SCR / "guard_probe2.out"
gl2.write_text("10-28 01:16 waiting for the r1 chain (x: end markers ^ZZV8FMINI-DONE / ^ZZV8FMINI-STOP)" + chr(10), encoding="utf-8")
rcg2, _ = sh(f"grep -q -E '^(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)' \"{bp(gl2)}\"")
T("X4-7 the installed guard's anchored end test matches an r2 stop line (and the r2 chain's own 'waiting' echo does not)", rcg == 0 and rcg2 == 1 and 'grep -q -E \'^(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)\'' in (INST / "v8f_guard_dagoaccorc.sh").read_text(encoding="utf-8"))

# ====================================================================== X6. the r2 asserter's BUILT-stream clause, the token, and the REAL records of the fixed trainer's composed run
print("== X6: ego_stream_problems (gate selfcheck levers.ego_perturb_stream EXACTLY the int 1 on A / the int 0 on REP), the token field, and the fixed trainer's REAL records")
ASRC = (R2SHIP / "gate_dagoaccorc_r2_assert.py").read_text(encoding="utf-8")
SL = ["levers", "ego_perturb_stream"]
XA_ROWS = [      # name, arm, seed, mutate(sc) or None, substrings one of which a problem must contain ([] = must PASS)
    ("A s0 clean (the record carries ego_perturb_stream 1)", "dagoaccorc", 0, None, []),
    ("A s1 clean", "dagoaccorc", 1, None, []),
    ("REP s0 clean (ego_perturb_stream 0)", "dagoaccorcrep", 0, None, []),
    ("REP s1 clean", "dagoaccorcrep", 1, None, []),
    ("A: the field is ABSENT (the r1 trainer's gate record)", "dagoaccorc", 0, sc_del(SL), ["built ego_perturb_stream"]),
    ("A s1: the field is ABSENT", "dagoaccorc", 1, sc_del(SL), ["built ego_perturb_stream"]),
    ("A: 0 (the shared, original stream)", "dagoaccorc", 0, sc_set(SL, 0), ["built ego_perturb_stream"]),
    ("A: True (== 1 in python)", "dagoaccorc", 0, sc_set(SL, True), ["built ego_perturb_stream"]),
    ("A: 1.0 (a float)", "dagoaccorc", 1, sc_set(SL, 1.0), ["built ego_perturb_stream"]),
    ("A: 2", "dagoaccorc", 0, sc_set(SL, 2), ["built ego_perturb_stream"]),
    ("A: null", "dagoaccorc", 0, sc_set(SL, None), ["built ego_perturb_stream"]),
    ("A: the string '1'", "dagoaccorc", 0, sc_set(SL, "1"), ["built ego_perturb_stream"]),
    ("REP: 1 (the replicate carries the separated stream)", "dagoaccorcrep", 0, sc_set(SL, 1), ["built ego_perturb_stream"]),
    ("REP s1: 1", "dagoaccorcrep", 1, sc_set(SL, 1), ["built ego_perturb_stream"]),
    ("REP: the field is ABSENT (a registered 0 is required: absent fails)", "dagoaccorcrep", 0, sc_del(SL), ["built ego_perturb_stream"]),
    ("REP: False (== 0 in python)", "dagoaccorcrep", 0, sc_set(SL, False), ["built ego_perturb_stream"]),
    ("REP: 0.0 (a float)", "dagoaccorcrep", 1, sc_set(SL, 0.0), ["built ego_perturb_stream"]),
    ("REP: null", "dagoaccorcrep", 0, sc_set(SL, None), ["built ego_perturb_stream"]),
]


def token_of(mod, arm, seed, tag):
    d = W.build(fresh(f"xtok_{tag}"), arm, seed)
    out = SCR / f"xtok_{tag}" / "token.json"
    args = ["token", "--selfcheck", d["selfcheck"], "--gate-line", d["gate_line"], "--train-line", d["train_line"], "--pin-line", d["pin_line"], "--arm", arm, "--seed", str(seed), "--banked-init-sha", d["banked_init_sha"],
            "--bank-sha", d["bank_sha"], "--bank-json", d["bank_json"], "--w-acc", W.W_ACC, "--b-summary", d["b_summary"], "--ref-argv-json", d["ref_argv_json"], "--commit", "a" * 40, "--queue-sha", "b" * 64,
            "--key-sha-json", '{"f": "c"}', "--run", f"M2b_s{seed}_{arm}", "--out", str(out)]
    rc = quiet(mod.main, args)
    return json.load(open(out, encoding="utf-8")) if rc == 0 and out.exists() else None


def asr_suite(mod, tokens=True):
    """The names of the rows an asserter module gets wrong: the 18 gate worlds (via world.run_check), and the token's ego_perturb_stream field (A 1, REP 0)."""
    old, fails = W.G, []
    W.G = mod
    try:
        for nm, arm, s, mut, subs in XA_ROWS:
            bad = quiet(W.run_check, W.build(fresh("xa"), arm, s, mutate=mut))
            ok = bad == [] if not subs else any(sub in x for x in bad for sub in subs)
            if not ok:
                fails.append(nm)
        if tokens:
            for arm, want in (("dagoaccorc", 1), ("dagoaccorcrep", 0)):
                tk = token_of(mod, arm, 0, arm)
                if not (tk is not None and type(tk.get("ego_perturb_stream")) is int and tk["ego_perturb_stream"] == want):
                    fails.append(f"token {arm}: ego_perturb_stream {None if tk is None else tk.get('ego_perturb_stream', '<absent>')!r}, want {want}")
    finally:
        W.G = old
    return fails


bad = asr_suite(G)
T(f"X6-1 the r2 asserter gives the registered answer on all {len(XA_ROWS)} gate worlds (the four clean worlds PASS; the field absent / 0 / True / 1.0 / 2 / null / '1' on A and 1 / absent / False / 0.0 / null on REP are REFUSED by the clause built for it) and the token records the field", bad == [], bad)
tk = token_of(G, "dagoaccorc", 0, "tka")
tr = token_of(G, "dagoaccorcrep", 1, "tkr")
T("X6-2 the launch-gate token records ego_perturb_stream (A exactly the int 1, REP exactly the int 0)", tk is not None and tr is not None and type(tk["ego_perturb_stream"]) is int and tk["ego_perturb_stream"] == 1 and type(tr["ego_perturb_stream"]) is int and tr["ego_perturb_stream"] == 0, (tk, tr))


def load_asr(src, tag):
    d = SCR / "asr_mods" / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / f"asr_{tag}.py").write_text(src, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(f"asr_{tag}", d / f"asr_{tag}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def amut(old, new):
    assert ASRC.count(old) == 1, old
    return ASRC.replace(old, new)


ASR_MUTS = [
    ("never CALLS ego_stream_problems (the clause is defined but unreachable)", amut("    bad += ego_stream_problems(lv, p)      # r2: the separated perturbation stream was BUILT\n", "")),
    ("is called on A only", amut("    bad += ego_stream_problems(lv, p)      # r2: the separated perturbation stream was BUILT\n", "    if p > 0.0:\n        bad += ego_stream_problems(lv, p)\n")),
    ("drops the type check (True == 1, 1.0 == 1, False == 0, 0.0 == 0 pass)", amut("    if type(got) is not int or got != want:", "    if got != want:")),
    ("lets an ABSENT field pass (the default is the wanted value)", amut('    got = lv.get("ego_perturb_stream", "<absent>")', "    got = lv.get(\"ego_perturb_stream\", want)")),
    ("swaps the arms (A wants 0, REP wants 1)", amut('EGO_STREAM = {"dagoaccorc": 1, "dagoaccorcrep": 0}', 'EGO_STREAM = {"dagoaccorc": 0, "dagoaccorcrep": 1}')),
    ("wants the separated stream on REP too", amut('EGO_STREAM = {"dagoaccorc": 1, "dagoaccorcrep": 0}', 'EGO_STREAM = {"dagoaccorc": 1, "dagoaccorcrep": 1}')),
    ("picks the A literal for every arm", amut('    want = EGO_STREAM["dagoaccorc" if p > 0.0 else "dagoaccorcrep"]', '    want = EGO_STREAM["dagoaccorc"]')),
    ("the token does not record the field", amut(' "ego_perturb_stream": EGO_STREAM[a.arm],', "")),
]
for i, (name, src) in enumerate(ASR_MUTS):
    M(f"MX6-{i + 1:02d} the r2 asserter {name}: the gate-world / token table catches it", len(asr_suite(load_asr(src, f"m{i}"))) >= 1)

# ---- the REAL records of the fixed trainer's composed run (fixtures/compose_fix_real, copied from F:/TanitAD-scratch/compose_fix; PROVENANCE.txt)
FR = R2 / "fixtures" / "compose_fix_real"


def jl(n):
    return json.load(open(FR / n, encoding="utf-8"))


fA_sm, fA_sc, fR_sm, fR_sc, f1_sm, f1_sc = (jl(n) for n in ("A_summary.json", "A_selfcheck.json", "REPlike_summary.json", "REPlike_selfcheck.json", "r1_A_s0_summary.json", "r1_A_s0_selfcheck.json"))


def real_rows(mod_c, mod_g):
    """The names of the real-record rows a (checks, asserter) pair gets wrong."""
    bad = []
    for nm, sm, sc, p, ok in (("the fixed trainer's REAL composed A run (stream 1, p_ego 0.3)", fA_sm, fA_sc, 0.3, True), ("the fixed trainer's NON-composed run (stream 0) read as the REP arm", fR_sm, fR_sc, 0.0, True),
                              ("the fixed trainer's NON-composed run (stream 0) read as the A arm", fR_sm, fR_sc, 0.3, False), ("r1's REFUSED A s0 record (no field, ego share 0.2492) read as the A arm", f1_sm, f1_sc, 0.3, False),
                              ("the fixed trainer's REAL composed A run read as the REP arm", fA_sm, fA_sc, 0.0, False)):
        got = (mod_c.stream_problems(sm, p, SPEC_R) == [] and mod_c.stream_problems(sc, p, SPEC_R) == [] and mod_g.ego_stream_problems(sc["levers"], p) == [])
        if got != ok:
            bad.append(nm)
    return bad


bad = real_rows(C, G)
T("X6-3 the REAL records: the fixed trainer's composed A run (summary.json AND selfcheck.json: levers.ego_perturb_stream 1) PASSES the registered spec and the asserter; its non-composed run (0) passes as REP and is REFUSED as A; r1's refused A s0 record (no field) is REFUSED as A; the composed run is REFUSED as REP", bad == [], bad)
T("X6-3b the real A / REP-like records carry exactly the int 1 / 0 (so the registered values are what the fixed trainer writes, not what I assumed)", type(fA_sm["levers"]["ego_perturb_stream"]) is int and fA_sm["levers"]["ego_perturb_stream"] == 1 and fA_sc["levers"]["ego_perturb_stream"] == 1
  and type(fR_sm["levers"]["ego_perturb_stream"]) is int and fR_sm["levers"]["ego_perturb_stream"] == 0 and "ego_perturb_stream" not in f1_sm["levers"] and "ego_perturb_stream" not in f1_sc["levers"])
rA = quiet(C.reached, NS(summary=str(FR / "A_summary.json"), arm="dagoaccorc"))
r1 = quiet(C.reached, NS(summary=str(FR / "r1_A_s0_summary.json"), arm="dagoaccorc"))
T("X6-4 K-REACH (the r1 bands, unchanged) on the REAL composed A summary: PASS (the separated stream reaches ego 0.2931 of the non-swapped eligible, swap 0.0976, overlap 0); on r1's refused A s0 summary: FAIL (0.2492)", rA == 0 and r1 == 1, (rA, r1))


def rec_with(rec, v):
    """a deep copy of a real run record with levers.ego_perturb_stream set to v ('<absent>' deletes it)"""
    o = json.loads(json.dumps(rec))
    if v == "<absent>":
        o["levers"].pop("ego_perturb_stream", None)
    else:
        o["levers"]["ego_perturb_stream"] = v
    return o


for nm, v in (("stream 0 (the shared stream)", 0), ("field absent", "<absent>"), ("stream True", True), ("stream 1.0", 1.0), ("stream 2", 2)):
    M(f"MX6-R the real composed A summary with ONE thing wrong ({nm}): the registered spec refuses it", C.stream_problems(rec_with(fA_sm, v), 0.3, SPEC_R) != [])
    M(f"MX6-R the real A SELFCHECK with ONE thing wrong ({nm}): the asserter's clause refuses it", G.ego_stream_problems(rec_with(fA_sc, v)["levers"], 0.3) != [])
for nm, v in (("stream 1 (the separated stream on the replicate)", 1), ("field absent", "<absent>"), ("stream False", False), ("stream 0.0", 0.0)):
    M(f"MX6-R the real non-composed REP-like summary with ONE thing wrong ({nm}): the registered spec refuses it", C.stream_problems(rec_with(fR_sm, v), 0.0, SPEC_R) != [])
    M(f"MX6-R the real non-composed REP-like SELFCHECK with ONE thing wrong ({nm}): the asserter's clause refuses it", G.ego_stream_problems(rec_with(fR_sc, v)["levers"], 0.0) != [])

# ====================================================================== X5. the waker
print("== X5: the r2 waker")
WK = (R2SHIP / "wake_v8fdagoaccorc_r2.sh").read_text(encoding="utf-8")


def waker(tag, lines, guard=None, pre=(), wk=None):
    d = fresh(f"wk_{tag}")
    (d / "logs").mkdir()
    (d / "logs" / "chain_v8f_dagoaccorc_r2.out").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if guard:
        (d / "logs" / "v8fdagoaccorc_r2_guard.out").write_text(guard + "\n", encoding="utf-8")
    for p in pre:
        (d / "logs" / p).write_text("")
    t = (wk or WK).replace("L=/c/Users/Admin/box2_ship/logs", f"L={bp(d)}/logs")
    assert t != (wk or WK)
    put(d / "wake.sh", t)
    return sh(f'timeout 12 bash "{bp(d)}/wake.sh"', timeout=40), d


GT = ["gate token written for M2b_s%d_%s commit x argv_sha256 y" % (s, a) for s in (0, 1) for a in ARMS2]
(rc, out), d = waker("stop", ["10-28 x", "ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc"])
T("X5-1 the waker wakes on the r2 chain's STOP marker", rc == 0 and "CHAIN END" in out, out[-200:])
(rc, out), d = waker("gates", ["10-28 x", *GT], pre=(".dagoaccorc_gates_seen",))
T("X5-2 the waker wakes once on the four gate tokens, with r1's gates-seen marker already present (r2 has its own marker)", rc == 0 and "GATES PASSED" in out and (d / "logs" / ".dagoaccorc_r2_gates_seen").exists(), out[-200:])
(rc, out), d = waker("named", ["10-28 x", "ZZDAGOACCORC-CONTROL-FAIL M2b_s0_dagoaccorc: x"])
T("X5-3 the waker wakes on the K-REACH named stop", rc == 0 and "NAMED STOP" in out, out[-200:])
(rc, out), d = waker("guard", ["10-28 x"], guard="ZZGUARD-STOP TEMP x")
T("X5-4 the waker wakes on a guard stop (exit 4)", rc == 4 and "GUARD" in out, (rc, out[-200:]))
wk_m = WK.replace(".dagoaccorc_r2_gates_seen", ".dagoaccorc_gates_seen")
(rc, out), d = waker("gates_m", ["10-28 x", *GT], pre=(".dagoaccorc_gates_seen",), wk=wk_m)
M("MX5-1 a waker sharing r1's gates-seen marker stays asleep when r1's marker exists", "GATES PASSED" not in out, out[-200:])
wk_m = WK.replace("chain_v8f_dagoaccorc_r2.out", "chain_v8f_dagoaccorc.out")
(rc, out), d = waker("stop_m", ["10-28 x", "ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc"], wk=wk_m)
M("MX5-2 a waker reading r1's log name never sees the r2 chain's stop", "CHAIN END" not in out, out[-200:])

# ====================================================================== summary
npass, nfail = sum(1 for _, ok in RESULTS if ok), sum(1 for _, ok in RESULTS if not ok)
nkill, nsurv = sum(1 for _, ok in MUTANTS if ok), sum(1 for _, ok in MUTANTS if not ok)
print(f"\nSELFTEST: {npass}/{npass + nfail} tests pass; MUTATION: {nkill}/{nkill + nsurv} mutants killed, {nsurv} survivors")
json.dump({"tests": {"passed": npass, "failed": nfail, "failures": [n for n, ok in RESULTS if not ok]}, "mutants": {"killed": nkill, "survived": nsurv, "survivors": [n for n, ok in MUTANTS if not ok]},
           "groups": {"A_asserter_mutants": len(MA), "K_run_record_mutants": len(MK), "K_reach_mutants": len(MR), "X1_pred_rows": len(PRED_ROWS), "X1_pred_mutants": len(PRED_MUTS), "X2_stream_rows": len(STREAM_ROWS),
                      "X2_stream_mutants": len(STREAM_MUTS), "X3_aside_mutants": len(LIB_MUTS) + 1, "X6_gate_world_rows": len(XA_ROWS), "X6_asserter_mutants": len(ASR_MUTS), "C2_real_tree_rows": len(RT_ROWS)}}, open(HERE / "selftest_chain_result.json", "w", encoding="utf-8"), indent=1)
print("ZZSELFTEST-R2-END", "PASS" if nfail == 0 and nsurv == 0 else "FAIL")
sys.exit(0 if nfail == 0 and nsurv == 0 else 1)
