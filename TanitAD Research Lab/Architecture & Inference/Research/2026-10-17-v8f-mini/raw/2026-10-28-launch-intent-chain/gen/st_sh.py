#!/usr/bin/env python3
"""Self-test, shell layer: REAL cmd.exe / bash execution of the queue stage, the chain (--check-tree, --dry-run, --wait-only, --rep-licence, the early real run against a scratch git mirror and a STUB ssh/scp that logs every call and refuses),
the lib (atomic BOX2_BUSY), the waker and the guard, in scratch worlds under F:/TanitAD-scratch/launch_intent_chain/selftest.  Nothing here touches box 2, the installed box2_ship (read-only copies), the network or git beyond scratch repos.

EVERY bash step goes through sealed.run / sealed.popen (gen/sealed.py): a guard in the SAME shell asserts the world's stubs are the ssh/scp that resolve, the box-2 alias is unresolved, HOME is the empty scratch dir and every write path is in the scratch world
BEFORE the step's text runs (the 2026-10-28 incident: Git bash re-prepends /usr/bin, so a PATH handed over in the environment never wins).  A test below proves that no module of this harness starts bash any other way.

A VARIANT is the generated chain + helpers with ONE text edit (a mutant) and the chain RE-RENDERED so every pinned sha256 matches the edited helper (a mutant is therefore never refused merely for a sha mismatch). A PROBE is a real run with a registered
expectation; a mutant is KILLED iff some probe of its list no longer behaves as registered.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from st_common import *                              # noqa: F401,F403
import st_common as S
import sealed
from sealed import px
import make_chain_li as MC
import make_dry_run as DR

C_ = S.C
STAGED_QUEUE = (C_.QSTAGE / "v8f_mini_box2_queue.cmd").read_bytes()
TIP_QUEUE = C_.QUEUE_TIP_FILE.read_bytes()                       # the LANDED commit's queue blob (no :launchintentarm)
QPATH = C_.QUEUE_PATH
TR, MD = "stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py"
STAGED_TR, STAGED_MD = (C_.STAGE / TR).read_bytes(), (C_.STAGE / MD).read_bytes()
STAGED_SHA, STAGED_MODEL_SHA = S.C.sha_b(STAGED_TR), S.C.sha_b(STAGED_MD)
PRED_LOG = "chain_v8f_dagoaccorc_r2.out"
PIN_PREREG, PIN_READER = "e" * 64, "f" * 64                       # test values for the two pins the launch step fills
NPAT = len(MC.LEVER_ALL)
RUN_LABELS = [C_.RUN_LABEL[r] for r in C_.RUN_ORDER]


def bash(cmd: str, env=None, timeout=300, need=()):
    """The ONLY way this harness starts bash: sealed.run (the guard runs first, in the same shell).  ``env`` may carry LI_STUB_DIR (the world's stub dir) and the world's write paths."""
    return sealed.run(cmd, env=env, timeout=timeout, need=need)


def git(cwd, *args, check=True, env=None):
    e = dict(os.environ)
    e.update(env or {})
    for attempt in range(8):                                           # a scratch repo on this drive can answer "Permission denied" / "unable to write" to an object write while a scanner or another job holds the file: these git calls are idempotent, retry
        r = subprocess.run(["git", "-c", "safe.directory=*", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=str(cwd), capture_output=True, env=e)
        if r.returncode == 0 or not any(k in r.stderr for k in (b"Permission denied", b"unable to write", b"failed to write", b"unable to create", b"failed to insert")):
            break
        time.sleep(3.0)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr.decode('utf-8', 'replace')[:300]}")
    return r.stdout.decode("utf-8", "replace")


# ---------------------------------------------------------------------------------------------------------------------------------------- the tip exports and the scratch mirrors
_TIPX = {}


def tip_files(ref: str | None = None) -> dict:
    """The exported paths of ``ref`` (default: the LANDED commit, which already carries the trainer / model and not the queue stage).  ``C_.PRE_TIP`` gives the pre-lever tree."""
    ref = ref or C_.TIP
    if ref not in _TIPX:
        chain = (S.SHIP / "chain_v8f_launchintent.sh").read_text(encoding="utf-8")
        exp = re.findall(r"^  '([^']+)'$", re.search(r"^EXPORT=\(\n(.*?)\n\)\n", chain, re.M | re.S).group(1), re.M)
        paths = sorted(set(exp + [QPATH, f"{C_.PK}/raw/eval_grid_windows.json"]))
        d = S.fresh("tipx_" + ref)
        DR.export_tip(d, paths, ref)
        _TIPX[ref] = {str(p.relative_to(d)).replace("\\", "/"): p.read_bytes() for p in d.rglob("*") if p.is_file()}
    return dict(_TIPX[ref])


def overlay(**changes) -> dict:
    """The landed tree + the STAGED trainer / model / queue; ``changes`` = {path: bytes | None (delete)}."""
    f = tip_files()
    f[TR], f[MD] = STAGED_TR, STAGED_MD
    f[QPATH] = STAGED_QUEUE
    f.update({k: v for k, v in changes.items() if v is not None})
    for k, v in changes.items():
        if v is None:
            f.pop(k, None)
    return f


def write_tree(root: Path, files: dict) -> Path:
    shutil.rmtree(root, ignore_errors=True)
    for k, v in files.items():
        S.put(root / k, v)
    return root


FIXED = {"GIT_AUTHOR_DATE": "2026-10-28T00:00:00 +0000", "GIT_COMMITTER_DATE": "2026-10-28T00:00:00 +0000", "GIT_AUTHOR_NAME": "t", "GIT_COMMITTER_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_EMAIL": "t@t"}
_ROOT = []


def root_commit() -> str:
    """The test world's stand-in for the LANDED commit: a root commit of the landed tree with FIXED author / date, so its hash is the same in every scratch mirror (and a chain rendered with it as LAUNCH_COMMIT can check ancestry)."""
    if not _ROOT:
        work = S.fresh("rootcommit")
        for k, v in tip_files().items():
            S.put(work / k, v)
        git(work, "init", "-q")
        git(work, "add", "-A")
        git(work, "commit", "-q", "-m", "landed", env=FIXED)
        _ROOT.append(git(work, "rev-parse", "HEAD").strip())
    return _ROOT[0]


def make_mirror(root: Path, files: dict, orphan: bool = False) -> Path:
    """origin (bare) <- work: commit R (the landed tree, fixed metadata) then the TIP commit carrying ``files`` on top (``orphan``: the tip alone, which does NOT descend from R); mirror = a clone of origin."""
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    origin, work, mirror = root / "origin.git", root / "work", root / "mirror"
    git(root, "init", "-q", "--bare", str(origin))
    git(root, "init", "-q", str(work))
    if not orphan:
        for k, v in tip_files().items():
            S.put(work / k, v)
        git(work, "add", "-A")
        git(work, "commit", "-q", "-m", "landed", env=FIXED)
        assert git(work, "rev-parse", "HEAD").strip() == root_commit(), "the scratch landed commit is not deterministic"
    for k in [str(p.relative_to(work)).replace("\\", "/") for p in work.rglob("*") if p.is_file() and ".git" not in p.parts]:
        if k not in files:
            (work / k).unlink()
    for k, v in files.items():
        S.put(work / k, v)
    git(work, "add", "-A")
    git(work, "commit", "-q", "--allow-empty", "-m", "orphan tip" if orphan else "tip", env={**FIXED, "GIT_AUTHOR_DATE": "2026-10-28T01:00:00 +0000", "GIT_COMMITTER_DATE": "2026-10-28T01:00:00 +0000"})
    git(work, "push", "-q", str(origin), "HEAD:refs/heads/agent/arch-inf-20260803")
    git(root, "clone", "-q", str(origin), str(mirror))
    return mirror


# ---------------------------------------------------------------------------------------------------------------------------------------- a ship dir (generated files + read-only copies of the INSTALLED ones) and a chain rendered against it
HELPERS = ("gate_launchintent_assert.py", "v8f_launchintent_chain_checks.py", "oaf_launchintent_lib.sh", "v8f_guard_launchintent.sh", "wake_v8flaunchintent.sh", "run_chain_v8f_launchintent.cmd")
INSTALLED_COPIES = ("oaf_lib.sh", "v8f_oaf_chain_checks.py", "mk_v8f_job.py", "waitjob.sh")
REAL_DIR_BYTES = b"C:/Users/Admin/box2_ship/"
KINDS = {"ok": {}, "notrainer": {"trainer": MC.PH_PIN}, "nomodel": {"LAUNCH_MODEL_SHA": MC.PH_MODEL}, "nocommit": {"LAUNCH_COMMIT": "<<UNSET: commit>>"},
         "noprereg": {"PREREG_SHA": MC.PH_PREREG}, "noreader": {"READER_SHA": MC.PH_READER}, "nolaunchpins": {"PREREG_SHA": MC.PH_PREREG, "READER_SHA": MC.PH_READER}}


class Variant:
    def __init__(self, name, chain_mut=None, helper_mut=None):
        self.name, self.chain_mut, self.helper_mut = name, chain_mut or [], helper_mut or {}
        self._w = {}

    def world(self, kind="ok") -> dict:
        if kind is True:
            kind = "ok"
        elif kind is False:
            kind = "notrainer"
        if kind in self._w:
            return self._w[kind]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.name)[:40] + "_" + kind
        ship = S.fresh(f"ship/{slug}")
        for h in HELPERS:
            t = (S.SHIP / h).read_bytes()
            for (old, new, n) in self.helper_mut.get(h, []):
                assert (t.count(old.encode()) >= 1) if n == -1 else (t.count(old.encode()) == n), (h, old[:60], t.count(old.encode()), n)
                t = t.replace(old.encode(), new.encode())
            (ship / h).write_bytes(t)
        for h in INSTALLED_COPIES:
            shutil.copyfile(C_.INSTALLED / h, ship / h)
        wj = (ship / "waitjob.sh").read_bytes()                     # the installed waitjob.sh hardcodes L=/c/Users/Admin/box2_ship/logs: a scratch copy must NEVER write there
        assert wj.count(b"L=/c/Users/Admin/box2_ship/logs;") == 1
        (ship / "waitjob.sh").write_bytes(wj.replace(b"L=/c/Users/Admin/box2_ship/logs;", ("L=" + px(ship) + "/logs;").encode()))
        mk = (ship / "mk_v8f_job.py").read_bytes()                  # ... and the installed mk_v8f_job.py writes C:/Users/Admin/box2_ship/<job>.cmd: redirected too, so even a probe that reached `queue` could not write there
        assert mk.count(REAL_DIR_BYTES) == 1
        (ship / "mk_v8f_job.py").write_bytes(mk.replace(REAL_DIR_BYTES, (px(ship) + "/").encode()))
        pins = {"LAUNCH_COMMIT": root_commit(), "LAUNCH_MODEL_SHA": STAGED_MODEL_SHA, "PREREG_SHA": PIN_PREREG, "READER_SHA": PIN_READER}
        k = dict(KINDS[kind])
        trainer = k.pop("trainer", STAGED_SHA)
        pins.update(k)
        txt = MC.render(trainer, ship, **pins)
        for (old, new, n) in self.chain_mut:
            assert (txt.count(old) >= 1) if n == -1 else (txt.count(old) == n), (self.name, old[:70], txt.count(old), n)
            txt = txt.replace(old, new)
        (ship / "chain_v8f_launchintent.sh").write_bytes(txt.encode("utf-8"))
        logs = ship / "logs"
        logs.mkdir()
        stub = sealed.make_stub_dir(ship / "stub")
        w = dict(ship=ship, chain=ship / "chain_v8f_launchintent.sh", logs=logs, stub=stub, calls=ship / "stub_calls.log", art=ship / "art", kind=kind)
        self._w[kind] = w
        return w


def guard_copy(w) -> str:
    """The world's guard with its hardcoded B=/c/Users/Admin/box2_ship pointed at the world (the REAL ship dir is never read or written by a test)."""
    g = (w["ship"] / "v8f_guard_launchintent.sh").read_bytes()
    assert g.count(b"B=/c/Users/Admin/box2_ship;") == 1
    out = w["ship"] / "guard_scratch.sh"
    out.write_bytes(g.replace(b"B=/c/Users/Admin/box2_ship;", ("B=" + px(w["ship"]) + ";").encode()))
    return px(out)


def env_for(w, mirror=None, rep=None, after=True, extra=None) -> dict:
    e = {"OAF_SHIP_DIR": px(w["ship"]), "OAF_POLL_S": "1", "OAF_GRACE_S": "1", "LAUNCHINTENT_ART": px(w["art"]), "LAUNCHINTENT_NEED_BYTES": "1000", "LI_STUB_LOG": px(w["calls"]), "LI_BUSY_FILE": px(w["ship"] / "BOX2_BUSY.txt"),
         "LI_STUB_DIR": str(w["stub"]), "TMPDIR": px(w["ship"])}
    if after:
        e["AFTER_OUT"] = px(w["logs"] / PRED_LOG)
    if mirror:
        e["OAF_MIRROR"] = px(mirror)
    if rep:
        e["LAUNCHINTENT_REP_DIR"] = px(rep)
    e.update(extra or {})
    return e


CHAIN_NEED = ("OAF_SHIP_DIR", "LAUNCHINTENT_ART", "TMPDIR")


def run_chain(w, args: str, env: dict, timeout=300):
    w["calls"].write_bytes(b"")
    return bash(f'bash "{px(w["chain"])}" {args}', env, timeout, need=CHAIN_NEED)


def calls(w) -> list:
    return [x for x in w["calls"].read_text(encoding="utf-8", errors="replace").split("\n") if x.strip()]


# ---------------------------------------------------------------------------------------------------------------------------------------- scratch artifacts shared by the probes
_SH = {}


def shared() -> dict:
    if "x" in _SH:
        return _SH
    root = S.fresh("shared")
    ok_files = overlay()
    _SH["trees"] = {
        "ok": write_tree(root / "tree_ok", ok_files),
        "pre": write_tree(root / "tree_pre", tip_files(C_.PRE_TIP)),                                     # the PRE-LEVER tip (94a47f4b): no flag, no queue stage
        "landed": write_tree(root / "tree_landed", tip_files()),                                        # the LANDED commit: the flag and the model, NOT the queue stage
        "sha": write_tree(root / "tree_sha", overlay(**{TR: STAGED_TR + b"\n# one more line\n"})),
        "msha": write_tree(root / "tree_msha", overlay(**{MD: STAGED_MD + b"\n# one more line\n"})),
        "noflag": write_tree(root / "tree_noflag", overlay(**{TR: STAGED_TR.replace(b'add_argument("--launch-intent-w"', b'add_argument("--launch-intent-x"', 1)})),
        "nostage": write_tree(root / "tree_nostage", overlay(**{QPATH: TIP_QUEUE})),
        "nototal": write_tree(root / "tree_nototal", overlay(**{MD: STAGED_MD.replace(b"total = total + cfg.w_traj * cfg.launch_intent_w", b"total = total + 0.0 * cfg.launch_intent_w", 1)})),
    }
    q = STAGED_QUEUE.decode("utf-8")
    FLAG = 'set "DAGF=%DAGF% --launch-intent-w 1.58926"'
    REPSEED = 'if /i "%DAGS%"=="launchintentrep" set "DAGSEED=1%~2"\r\n'
    REPDISP = 'if /i "%DAGS%"=="launchintentrep" goto :launchintentarm\r\n'
    REPNAME = 'if /i "%~1"=="launchintentrep" set "DAGS=launchintentrep"\r\n'
    CALL = 'call :dagoaccarm %~1 %~2 0.1 || exit /b 2\r\nif /i "%DAGS%"=="launchintentrep"'      # the call line of :launchintentarm (the same call text also opens :dagoaccorcarm, so the next line makes it unique)

    def qv(old, new):
        assert q.count(old) == 1, old
        return q.replace(old, new).encode("utf-8")
    _SH["queues"] = {
        "ok": (STAGED_QUEUE, None), "tip": (TIP_QUEUE, "prints no gate"),
        "w": (qv(FLAG, FLAG.replace("1.58926", "1.58927")), "not the one-lever"),
        "wB": (qv(FLAG, FLAG.replace("1.58926", "0.726175")), "not the one-lever"),
        "extra": (qv(FLAG, FLAG[:-1] + ' --w-traj 2.0"'), "not the one-lever"),
        "egoP": (qv(FLAG, FLAG[:-1] + ' --ego-perturb-p 0.3"'), "not the one-lever"),
        "first": (qv(FLAG, 'set "DAGF=--launch-intent-w 1.58926 %DAGF%"'), "not the one-lever"),
        "twice": (qv(FLAG, FLAG[:-1] + ' --launch-intent-w 1.58926"'), "not the one-lever"),
        "nocall": (qv(CALL, 'if /i "%DAGS%"=="launchintentrep"'), "not the one-lever"),       # the stage still prints a line, but without DAG-OACC's recipe under it: the dry run's SAME / ONE-LEVER check refuses it
        "nodispatch": (qv('if /i "%DAGS%"=="launchintent" goto :launchintentarm\r\n', ""), "prints no gate"),
        "nowhitelist": (qv('if /i "%~1"=="launchintent" set "DAGS=launchintent"\r\n', ""), "prints no gate"),
        "p03": (qv(CALL, CALL.replace("0.1 ||", "0.3 ||")), "prints no gate"),
        # ---- the replicate arm (A_rep,s): its trainer seed is 10 + s, set by ONE line; its name and its dispatch are two more
        "norepseed": (qv(REPSEED, ""), "not the one-lever"),
        "repseed2": (qv(REPSEED, REPSEED.replace("DAGSEED=1%~2", "DAGSEED=2%~2")), "not the one-lever"),
        "repseed_all": (qv(REPSEED, REPSEED.replace('if /i "%DAGS%"=="launchintentrep" set', "set")), "not the one-lever"),
        "repdispatch": (qv(REPDISP, ""), "prints no gate"),
        "repwhitelist": (qv(REPNAME, ""), "prints no gate"),
    }
    _SH["queue_files"] = {}
    for k, (b, _) in _SH["queues"].items():
        p = root / f"queue_{k}.cmd"
        p.write_bytes(b)
        _SH["queue_files"][k] = p
    _SH["trees"]["norepseed"] = write_tree(root / "tree_norepseed", overlay(**{QPATH: _SH["queues"]["norepseed"][0]}))
    _SH["trees"]["repwhitelist"] = write_tree(root / "tree_repwhitelist", overlay(**{QPATH: _SH["queues"]["repwhitelist"][0]}))
    # the REP pulls (both seeds good; and variants)
    _SH["rep_ok"] = root / "rep_ok"
    for s in (0, 1):
        S.rep_world(_SH["rep_ok"], s)
    _SH["rep_compose"] = root / "rep_compose"
    S.rep_world(_SH["rep_compose"], 0, rm_compose)
    S.rep_world(_SH["rep_compose"], 1)
    _SH["rep_s1bad"] = root / "rep_s1bad"
    S.rep_world(_SH["rep_s1bad"], 0)
    S.rep_world(_SH["rep_s1bad"], 1, rm_compose)
    # mirrors: every one has R (the scratch stand-in for the LANDED commit) as the ancestor of its tip, except `orphan`
    _SH["mirrors"] = {
        "ok": make_mirror(root / "m_ok", overlay()),
        "noflag": make_mirror(root / "m_noflag", overlay(**{TR: tip_files(C_.PRE_TIP)[TR]})),
        "nostage": make_mirror(root / "m_nostage", overlay(**{QPATH: TIP_QUEUE})),
        "sha": make_mirror(root / "m_sha", overlay(**{TR: STAGED_TR + b"\n# one more line\n"})),
        "msha": make_mirror(root / "m_msha", overlay(**{MD: STAGED_MD + b"\n# one more line\n"})),
        "nolabel": make_mirror(root / "m_nolabel", overlay(**{QPATH: STAGED_QUEUE.replace(b":dagoaccwacc\r\n", b":dagoaccwacc_gone\r\n", 1)})),
        "nototal": make_mirror(root / "m_nototal", overlay(**{MD: STAGED_MD.replace(b"total = total + cfg.w_traj * cfg.launch_intent_w", b"total = total + 0.0 * cfg.launch_intent_w", 1)})),
        "noexport": make_mirror(root / "m_noexport", {k: v for k, v in overlay().items() if k != "stack/scripts/v8f_mini_eval.py"}),
        "norepseed": make_mirror(root / "m_norepseed", overlay(**{QPATH: _SH["queues"]["norepseed"][0]})),
        "orphan": make_mirror(root / "m_orphan", overlay(), orphan=True),
    }
    _SH["x"] = True
    return _SH


def rm_compose(files):
    files["summary"]["levers"]["ego_perturb_with_dagger"] = True


def go_log(w, text=None, busy: str | None = None):
    (w["logs"] / PRED_LOG).write_bytes((text if text is not None else S.r2_log()).encode("utf-8"))
    b = w["ship"] / "BOX2_BUSY.txt"
    if b.exists():
        b.unlink()
    if busy is not None:
        b.write_bytes(busy.encode())
    for f in w["logs"].glob("chain_v8f_launchintent.out"):
        f.unlink()


# ---------------------------------------------------------------------------------------------------------------------------------------- probes: each returns (ok, detail); ok = the variant behaves as registered
def ct(v, tree, kind="ok"):
    w = v.world(kind)
    return run_chain(w, f'--check-tree "{px(shared()["trees"][tree])}"', env_for(w, after=False))


def p_ct_pass(v):
    rc, o = ct(v, "ok")
    return rc == 0 and f"every one of {NPAT} patterns" in o and "the model's sha256" in o, o[-300:]


def p_ct_placeholder(v):
    rc, o = ct(v, "ok", "notrainer")
    return rc == 1 and "LAUNCH_TRAINER_SHA is still the placeholder" in o, o[-300:]


def p_ct_model_placeholder(v):
    rc, o = ct(v, "ok", "nomodel")
    return rc == 1 and "LAUNCH_MODEL_SHA is still the placeholder" in o, o[-300:]


def p_ct_prereg_free(v):
    rc, o = ct(v, "ok", "nolaunchpins")
    return rc == 0 and f"every one of {NPAT} patterns" in o, o[-300:]


def p_ct_sha(v):
    rc, o = ct(v, "sha")
    return rc == 1 and "not the reviewed one" in o and "LAUNCH_TRAINER_SHA" in o, o[-300:]


def p_ct_msha(v):
    rc, o = ct(v, "msha")
    return rc == 1 and "v8f_mini.py sha256" in o and "!= LAUNCH_MODEL_SHA" in o, o[-300:]


def p_ct_pre(v):
    rc, o = ct(v, "pre")
    return rc == 1 and "launch-intent-w" in o and "launchintentarm" in o, o[-300:]


def p_ct_landed(v):
    rc, o = ct(v, "landed")
    return rc == 1 and "^:launchintentarm" in o and "add_argument" not in o and "!= LAUNCH" not in o, o[-400:]


def p_ct_noflag(v):
    rc, o = ct(v, "noflag")
    return rc == 1 and 'MISSING in stack/scripts/v8f_mini_train.py: add_argument\\("--launch-intent-w"' in o, o[-300:]


def p_ct_nostage(v):
    rc, o = ct(v, "nostage")
    return rc == 1 and "^:launchintentarm" in o and "goto :launchintentarm" in o, o[-300:]


def p_ct_nototal(v):
    rc, o = ct(v, "nototal")
    return rc == 1 and "launch_intent_w \\* T" in o, o[-300:]


def p_ct_norepseed(v):
    rc, o = ct(v, "norepseed")
    return rc == 1 and 'launchintentrep" set "DAGSEED=1%~2"' in o and "add_argument" not in o, o[-300:]


def p_ct_repwhitelist(v):
    rc, o = ct(v, "repwhitelist")
    return rc == 1 and 'DAGS=launchintentrep"' in o, o[-300:]


def dry(v, qkey, kind="ok"):
    w = v.world(kind)
    return run_chain(w, "--dry-run", env_for(w, extra={"V8F_QUEUE": px(shared()["queue_files"][qkey])}))


def p_dry_pass(v):
    rc, o = dry(v, "ok")
    heads = [x for x in o.split("\n") if x.startswith("== ") and " = M2b_s" in x]
    ok = (rc == 0 and o.count(": SAME") == 4 and o.count(": ONE-LEVER") == 4 and "K-SHARE:" in o and "K-MASK :" in o and not any(x.startswith("!!") for x in o.split("\n")) and o.count("--launch-intent-w 1.58926") >= 8
          and [h.split(" = ")[0][3:] for h in heads] == RUN_LABELS and "pair (A_rep,0 - REP_0)" in o and "pair (A_1 - CTRL_1)" in o and "UNSET" not in o
          and "--seed 10 " in o and "--seed 11 " in o and "--dagger-seed 0" in o and "M2b_s1_launchintentrep" in o and f"LAUNCH_COMMIT: {root_commit()}" in o)
    return ok, o[-500:]


def p_dry_unset(v):
    rc, o = dry(v, "ok", "nolaunchpins")
    return rc == 0 and "PREREG_SHA: UNSET" in o and "READER_SHA: UNSET" in o and o.count(": ONE-LEVER") == 4, o[-400:]


def p_dry_queue(qkey):
    def f(v):
        rc, o = dry(v, qkey)
        want = shared()["queues"][qkey][1]
        return rc == 1 and "ZZV8FMINI-STOP" in o and want in o, o[-300:]
    f.__name__ = f"p_dry_{qkey}"
    return f


def p_tamper(fname):
    def f(v):
        w = v.world(True)
        p = w["ship"] / fname
        orig = p.read_bytes()
        p.write_bytes(orig + b"\n")
        try:
            rc, o = run_chain(w, "--dry-run", env_for(w, extra={"V8F_QUEUE": px(shared()["queue_files"]["ok"])}))
        finally:
            p.write_bytes(orig)
        return rc == 1 and "is not the" in o and "== predecessor" not in o, o[-300:]
    f.__name__ = f"p_tamper_{fname}"
    return f


def p_r_noafter(v):
    w = v.world(True)
    go_log(w)
    rc, o = run_chain(w, "", env_for(w, after=False))
    return rc == 1 and "AFTER_OUT is not set" in o and not calls(w), o[-300:]


def p_r_pin(kind, text):
    """A real run with ONE of the five pins unset is REFUSED before the predecessor is waited for: nothing taken, nothing shipped, box 2 not contacted."""
    def f(v):
        w = v.world(kind)
        go_log(w)
        sh = shared()
        rc, o = run_chain(w, "", env_for(w, mirror=sh["mirrors"]["ok"], rep=sh["rep_ok"]))
        return rc == 1 and text in o and "waiting for" not in o and not calls(w) and not (w["ship"] / "BOX2_BUSY.txt").exists(), o[-300:]
    f.__name__ = f"p_r_pin_{kind}"
    return f


p_r_placeholder = p_r_pin("notrainer", "LAUNCH_TRAINER_SHA is still the placeholder")
p_r_prereg = p_r_pin("noprereg", "PREREG_SHA is still the placeholder")
p_r_reader = p_r_pin("noreader", "READER_SHA is still the placeholder")
p_r_model = p_r_pin("nomodel", "LAUNCH_MODEL_SHA is still the placeholder")
p_r_commit = p_r_pin("nocommit", "LAUNCH_COMMIT is still the placeholder")


def p_r_predstop(v):
    w = v.world(True)
    go_log(w, S.r2_log(done=False, stop="ZZV8FMINI-STOP K-REACH something"))
    sh = shared()
    rc, o = run_chain(w, "", env_for(w, mirror=sh["mirrors"]["ok"], rep=sh["rep_ok"]))
    return rc == 1 and "predecessor verdict: HALT" in o and "carries the reviewed trainer" not in o and not calls(w) and not (w["ship"] / "BOX2_BUSY.txt").exists(), o[-300:]


def p_r_wait_go(v):
    w = v.world(True)
    go_log(w)
    rc, o = run_chain(w, "--wait-only", env_for(w))
    return rc == 0 and "ZZOAF-WAIT-GO" in o, o[-300:]


def p_r_wait_wait(v):
    w = v.world(True)
    go_log(w, S.r2_log(done=False))
    rc, o = bash(f'timeout 6 bash "{px(w["chain"])}" --wait-only', env_for(w), need=CHAIN_NEED)
    return rc == 124 and "ZZOAF-WAIT-GO" not in o and "waiting for" in o, f"rc={rc} " + o[-200:]


def p_r_busy_held(v):
    w = v.world(True)
    go_log(w, busy="another chain's lock\n")
    rc, o = run_chain(w, "--wait-only", env_for(w))
    lk = w["ship"] / "BOX2_BUSY.txt"
    held = lk.is_file() and lk.read_bytes() == b"another chain's lock\n"                       # a chain that REMOVED the foreign lock leaves no file: that is the defect, not a crash
    return rc == 1 and "still held after" in o and held and "ZZOAF-WAIT-GO" not in o, o[-300:]


def p_r_rep_bad(v):
    w = v.world(True)
    go_log(w)
    sh = shared()
    rc, o = run_chain(w, "", env_for(w, mirror=sh["mirrors"]["ok"], rep=sh["rep_compose"]))
    return rc == 1 and "REP-LICENCE failed" in o and "condition 1" in o and "carries the reviewed trainer" not in o and not calls(w) and not (w["ship"] / "BOX2_BUSY.txt").exists(), o[-400:]


def p_r_rep_s1bad(v):
    w = v.world(True)
    go_log(w)
    sh = shared()
    rc, o = run_chain(w, "", env_for(w, mirror=sh["mirrors"]["ok"], rep=sh["rep_s1bad"]))
    return rc == 1 and "REP-LICENCE failed" in o and "REP-LICENCE s1: FAIL" in o and not calls(w), o[-400:]


def p_r_tip(kind, want):
    def f(v):
        w = v.world(True)
        go_log(w)
        sh = shared()
        rc, o = run_chain(w, "", env_for(w, mirror=sh["mirrors"][kind], rep=sh["rep_ok"]))
        return rc == 1 and want in o and "carries the reviewed trainer" not in o and "box 2 unreachable" not in o and not calls(w) and not (w["ship"] / "BOX2_BUSY.txt").exists(), o[-500:]
    f.__name__ = f"p_r_tip_{kind}"
    return f


def p_r_ok(v):
    w = v.world(True)
    go_log(w)
    sh = shared()
    decoy_before = sorted(x.name for x in w["logs"].iterdir())
    rc, o = run_chain(w, "", env_for(w, mirror=sh["mirrors"]["ok"], rep=sh["rep_ok"]))
    cs = calls(w)
    boot_only = len(cs) >= 1 and all("LastBootUpTime" in c and c.startswith("ssh BUSY=yes") for c in cs)      # box 2 is contacted ONLY while this chain holds BOX2_BUSY
    new = sorted(set(x.name for x in w["logs"].iterdir()) - set(decoy_before))
    ok = (rc == 1 and f"descends from the landed flag {root_commit()}" in o and "carries the reviewed trainer" in o and "ZZV8FMINI-STOP box 2 unreachable" in o and boot_only and not (w["ship"] / "BOX2_BUSY.txt").exists()
          and "REP-LICENCE failed" not in o)
    return ok, f"{o[-300:]} calls={cs[:3]} new={new}"


def p_claim_atomic(v):
    w = v.world(True)
    lib = px(w["ship"] / "oaf_launchintent_lib.sh")
    b = w["ship"] / "BOX2_BUSY.txt"
    if b.exists():
        b.unlink()
    script = f'B="{px(w["ship"])}"; BUSY_TAKEN=0; source "{lib}"; claim_box2 "worker $1" >/dev/null && echo "WON $1"'
    ps = [sealed.popen(script, args=(i,), env=env_for(w), need=CHAIN_NEED) for i in range(10)]
    outs = [p.communicate()[0].decode() for p in ps]
    won = [x for x in outs if "WON" in x]
    return len(won) == 1 and b.exists(), f"winners={len(won)}"


def p_claim_foreign(v):
    w = v.world(True)
    lib = px(w["ship"] / "oaf_launchintent_lib.sh")
    b = w["ship"] / "BOX2_BUSY.txt"
    b.write_bytes(b"another chain's lock\n")
    rc, o = bash(f'B="{px(w["ship"])}"; BUSY_TAKEN=0; source "{lib}"; claim_box2 "mine"; echo "rc=$? busy_taken=$BUSY_TAKEN"', env_for(w), need=CHAIN_NEED)
    return "rc=1 busy_taken=0" in o and "ZZV8FMINI-STOP busy marker present: another chain's lock" in o and b.is_file() and b.read_bytes() == b"another chain's lock\n", o[-300:]


# ---- the control-selection helpers of the chain, EXECUTED (the real run's gate / run-record / dev-box checks call them; no fake box reaches those steps)
CTRL_FUNCS = ("rn()", "rt()", "rg()", "lbl()", "ctrl_dir()", "ctrl_nm()", "ctrl_ckpt()", "ctrl_argv_sha()", "ctrl_summ_sha()")


def chain_fragment(w) -> str:
    lines = (w["chain"]).read_text(encoding="utf-8").split("\n")
    out = [x for x in lines if x.startswith(CTRL_FUNCS) or x.startswith("RUNS=(")]
    assert len(out) == len(CTRL_FUNCS) + 1, len(out)
    return "\n".join(out)


def p_ctrl_funcs(v):
    w = v.world(True)
    script = ('OACC_DIR=/o; REPDIR=/r; OACC_CKPT_SHA=(oc0 oc1); OACC_ARGV_SHA=(oa0 oa1); OACC_SUMM_SHA=(os0 os1); REP_CKPT_SHA=(rc0 rc1); REP_ARGV_SHA=(ra0 ra1); REP_SUMM_SHA=(rs0 rs1)\n' + chain_fragment(w) + "\n"
              'for run in "${RUNS[@]}"; do set -- $run; echo "$(lbl $1 $2)|$(rn $1 $2)|$(rt $1 $2)|$(rg $1 $2)|$(ctrl_dir $1 $2)|$(ctrl_nm $1 $2)|$(ctrl_ckpt $1 $2)|$(ctrl_argv_sha $1 $2)|$(ctrl_summ_sha $1 $2)"; done')
    rc, o = bash(script, env_for(w), need=CHAIN_NEED)
    want = ["A_0|M2b_s0_launchintent|M2b-launchintent_s0|gate_M2b-launchintent_s0|/o/M2b_s0_dagoacc|DAG-OACC_s0|oc0|oa0|os0",
            "A_1|M2b_s1_launchintent|M2b-launchintent_s1|gate_M2b-launchintent_s1|/o/M2b_s1_dagoacc|DAG-OACC_s1|oc1|oa1|os1",
            "A_rep,0|M2b_s0_launchintentrep|M2b-launchintentrep_s0|gate_M2b-launchintentrep_s0|/r/M2b_s0_dagoaccorcrep|REP_s0|rc0|ra0|rs0",
            "A_rep,1|M2b_s1_launchintentrep|M2b-launchintentrep_s1|gate_M2b-launchintentrep_s1|/r/M2b_s1_dagoaccorcrep|REP_s1|rc1|ra1|rs1"]
    return rc == 0 and o.strip().split("\n") == want, o[-500:]


# ---- the waker / guard / static probes
def waker_run(v, setup, poll=1, n=3, bg=None, decoys=False, timeout=60):
    w = v.world(True)
    logs = S.fresh("waker/" + re.sub(r"[^A-Za-z0-9]+", "_", v.name)[:30] + "_" + re.sub(r"[^A-Za-z0-9]+", "_", getattr(setup, "__name__", "x")))
    if decoys:
        add_decoys(logs)
    setup(logs)
    if bg:
        threading.Timer(1.2, bg, args=(logs,)).start()
    rc, o = bash(f'LI_LOG_DIR="{px(logs)}" LI_WAKE_POLL_S={poll} LI_WAKE_N={n} bash "{px(w["ship"] / "wake_v8flaunchintent.sh")}"', {"LI_STUB_DIR": str(w["stub"]), "LI_STUB_LOG": px(w["calls"]), "LI_LOG_DIR": px(logs)}, timeout)
    return rc, o


def wr(logs, name, text, mode="w"):
    with open(logs / name, mode + "b") as f:
        f.write(text.encode())


def add_decoys(logs):
    """The SIBLING chains' markers and files in the shared directory (MEASURED 2026-10-28: DAG-ORC r2 read DAG-OACC r2's r2_aside_done.txt as its own)."""
    for n in (".dagoaccorc_r2_gates_seen", ".dagoaccorc_gates_seen", ".dagoacc_gates_seen", "r2_aside_done.txt", "r2_aside_done_dagoaccorc.txt", "v8fdagoaccorc_r2_current.txt", "v8f_guard_tail.jsonl", "v8f_guard_sc.json", "BOX2_BUSY.dagoaccorc_stopped.txt"):
        wr(logs, n, "decoy\n")
    wr(logs, "v8fdagoaccorc_r2_guard.out", "ZZGUARD-STOP TEMP 10-28 06:00 decoy of the sibling chain\n")
    wr(logs, "v8fdagoacc_guard.out", "ZZGUARD-STOP TEMP decoy\n")
    wr(logs, "chain_v8f_dagoaccorc_r2.out", S.r2_log() + "ZZV8FMINI-STOP decoy sibling\n")
    wr(logs, "chain_v8f_dagoacc_r2.out", "ZZV8FMINI-DONE decoy sibling\n")


GT = "gate token written for M2b_s{}_{} commit a\n"
FOUR = [("0", "launchintent"), ("1", "launchintent"), ("0", "launchintentrep"), ("1", "launchintentrep")]


def tokens(n):
    return "".join(GT.format(s, a) for s, a in FOUR[:n])


def s_end(logs):
    wr(logs, "chain_v8f_launchintent.out", "10-28 x\nZZV8FMINI-DONE 10-28 09:00 tip x: y\n")


def s_stop(logs):
    wr(logs, "chain_v8f_launchintent.out", "10-28 x\nZZV8FMINI-STOP something\n")


def s_guard(logs):
    wr(logs, "chain_v8f_launchintent.out", "10-28 x\n")
    wr(logs, "v8flaunchintent_guard.out", "ZZGUARD-STOP TEMP 10-28 USB HDD\n")


def s_named(logs):
    wr(logs, "chain_v8f_launchintent.out", "10-28 x\nZZLAUNCHINTENT-CONTROL-FAIL M2b_s0_launchintentrep: x\n")


def s_floor(logs):
    wr(logs, "chain_v8f_launchintent.out", "10-28 x\nZZDISK-FLOOR-STOP pulling x\n")


def s_idle(logs):
    wr(logs, "chain_v8f_launchintent.out", "10-28 x\n")


def s_gates_late(logs):
    wr(logs, "chain_v8f_launchintent.out", "10-28 x\n" + tokens(4))


def bg_gates(logs):
    wr(logs, "chain_v8f_launchintent.out", tokens(4), "a")


def bg_gates_partial(logs):
    wr(logs, "chain_v8f_launchintent.out", tokens(3), "a")


def bg_pull(logs):
    wr(logs, "chain_v8f_launchintent.out", "10-28 09:00 M2b_s0_launchintent pulled to /e/x (checkpoint sha256 ab)\n", "a")


def s_pull_existing(logs):
    wr(logs, "chain_v8f_launchintent.out", "10-28 x\n10-28 08:00 M2b_s0_launchintent pulled to /e/x (checkpoint sha256 ab)\n")


def p_w_end(v):
    rc, o = waker_run(v, s_end)
    return rc == 0 and "CHAIN END" in o, o[-200:]


def p_w_stop(v):
    rc, o = waker_run(v, s_stop)
    return rc == 0 and "CHAIN END" in o, o[-200:]


def p_w_guard(v):
    rc, o = waker_run(v, s_guard)
    return rc == 4 and "GUARD:" in o, f"rc={rc} " + o[-200:]


def p_w_named(v):
    rc, o = waker_run(v, s_named)
    return rc == 0 and "NAMED STOP" in o and "CONTROL-FAIL" in o, o[-200:]


def p_w_floor(v):
    rc, o = waker_run(v, s_floor)
    return rc == 0 and "NAMED STOP" in o and "DISK-FLOOR" in o, o[-200:]


def p_w_gates(v):
    rc, o = waker_run(v, s_idle, poll=1, n=6, bg=bg_gates)
    return rc == 0 and "GATES PASSED" in o, o[-200:]


def p_w_gates_partial(v):
    """THREE of the four gates passed: the waker keeps waiting (the fourth is A_rep,1's)."""
    rc, o = waker_run(v, s_idle, poll=1, n=4, bg=bg_gates_partial)
    return "GATES PASSED" not in o and "WAKER TIMEOUT" in o, o[-200:]


def p_w_gates_decoy(v):
    rc, o = waker_run(v, s_idle, poll=1, n=6, bg=bg_gates, decoys=True)
    return rc == 0 and "GATES PASSED" in o and "GUARD" not in o and "CHAIN END" not in o, o[-200:]


def p_w_gates_late(v):
    rc, o = waker_run(v, s_gates_late, poll=1, n=2)
    return "GATES PASSED" not in o and "WAKER TIMEOUT" in o, o[-200:]


def p_w_pull(v):
    rc, o = waker_run(v, s_idle, poll=1, n=6, bg=bg_pull)
    return rc == 0 and "NEW PULL" in o, o[-200:]


def p_w_pull_none(v):
    rc, o = waker_run(v, s_pull_existing, poll=1, n=2)
    return "NEW PULL" not in o and "WAKER TIMEOUT" in o, o[-200:]


def s_none(logs):
    return None


def p_w_decoys_only(v):
    rc, o = waker_run(v, s_none, poll=1, n=2, decoys=True)
    return "WAKER TIMEOUT" in o and "CHAIN END" not in o and "GUARD" not in o, o[-200:]


def guard_env(w, **x):
    return {"LI_STUB_DIR": str(w["stub"]), "LI_STUB_LOG": px(w["calls"]), **x}


def p_guard_end(v):
    w = v.world(True)
    out = S.fresh("guard/" + re.sub(r"[^A-Za-z0-9]+", "_", v.name)[:30])
    (out / "o.out").write_bytes(b"x\nZZV8FMINI-DONE 10-28 x\n")
    rc, o = bash(f'V8F_CHAIN_OUT="{px(out / "o.out")}" bash "{guard_copy(w)}"', guard_env(w, V8F_CHAIN_OUT=px(out / "o.out")), 60)
    return rc == 0 and "chain ended; guard off" in o, o[-200:]


FORBIDDEN = ["r2_aside_done", "dagoaccorc_r2_gates_seen", ".dagoacc_gates_seen", ".dagoaccorc_gates_seen", "v8fdagoaccorc", "v8fdagoacc_", "chain_v8f_dagoacc_r2.out", "chain_v8f_dagoaccorc.out", "v8f_guard_tail.jsonl", "v8f_guard_sc.json",
             "v8f_guard_rate.log", "v8f_current.txt", "chain_v8f_mini.out", "oaf_dagoaccorc_r2_lib", "gate_dagoaccorc", "v8f_dagoaccorc_r2_chain_checks", "repo_v8fdagoacc", "TIP_V8FDAGOACC", "v8f_dagoaccorc_r2/"]


def live_text(p: Path) -> str:
    t = p.read_bytes().decode("utf-8", "replace").replace("\r", "")
    return "\n".join(x for x in t.split("\n") if not x.lstrip().startswith(("#", "rem ")))


def p_static_names(v):
    w = v.world(True)
    bad = []
    for f in ("chain_v8f_launchintent.sh", "oaf_launchintent_lib.sh", "v8f_guard_launchintent.sh", "wake_v8flaunchintent.sh", "run_chain_v8f_launchintent.cmd"):
        t = live_text(w["ship"] / f)
        for tok in FORBIDDEN:
            if tok in t:
                bad.append(f"{f}: {tok}")
    runner = live_text(w["ship"] / "run_chain_v8f_launchintent.cmd")
    if ("> /c/Users/Admin/box2_ship/logs/chain_v8f_launchintent.out 2>&1" not in runner) or runner.count("chain_v8f_dagoaccorc_r2.out") != 1 or "AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out" not in runner:
        bad.append("runner: its log must be its own and the sibling's log appears ONLY as AFTER_OUT")
    chain = live_text(w["ship"] / "chain_v8f_launchintent.sh")
    for must in ("E=$L/v8f_launchintent", "GUARD_OUT=$L/v8flaunchintent_guard.out", "CUR=$L/v8flaunchintent_current.txt", "repo_v8flaunchintent_tip.tar", "TIP_V8FLAUNCHINTENT.txt", "V8F_CHAIN_OUT=$L/chain_v8f_launchintent.out"):
        if must not in chain:
            bad.append(f"chain lacks its own name {must}")
    return not bad, bad


def p_static_realdir(v):
    """The world's COPIES of the installed helpers (waitjob.sh, mk_v8f_job.py, oaf_lib.sh, the installed checks) name no real ship dir: a probe that reached `queue` or `waitrc` could not write there."""
    w = v.world(True)
    bad = [h for h in ("waitjob.sh", "mk_v8f_job.py", "oaf_lib.sh", "v8f_oaf_chain_checks.py") if b"C:/Users/Admin/box2_ship" in (w["ship"] / h).read_bytes() or b"/c/Users/Admin/box2_ship" in (w["ship"] / h).read_bytes()]
    return not bad, bad
