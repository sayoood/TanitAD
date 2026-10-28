#!/usr/bin/env python3
"""dagorct1_lib.py -- the helpers of dagorct1_chain.sh, the dev-box CPU T1-kin chain of PREREG_V8F_DAG_ORC re-based on DAG-OACC (Amendment 1, A1-2026-10-27).
CPU only, no torch, read-only on every pull (it HASHES files, it never opens a checkpoint's values or an eval's numbers).  Nothing here judges the science: that is the pre-registered reader's.

    boxlog   --log L --art ART  run ARM SEED     parse the box-2 chain log (read-only): rc 0 "PASS <ckpt sha256>" | 1 "WAIT <state>" | 3 "FAIL <why>"
    boxlog   --log L --art ART  meta             "stops=N done=0|1 pending=P pulled=K"  (P = runs of the four with no 'pulled to' line yet)
    boxlog   --log L --art ART  done             rc 0 when a ^ZZV8FMINI-DONE line follows the last 'pulled to' line, else 1
    boxlog   --log L --art ART  table            one line per run: its state
    pullcheck --dir D --tag T [--identity ARM SEED] [--expect-sha S] [--min-age N]
                                                  is the pull COMPLETE BY CONTENT?  rc 0 "COMPLETE <ckpt sha256> <why>" | 1 "INCOMPLETE <why>" | 3 "WRONG <why>"
    kbuild   --art ART --git-dir G [--launch C] [--fix-pkg P [--evidence-commit E] [--r1-evidence-commit R]]
                                                  the K-BUILD record the reader's k_build() wants, computed from git objects + the compose package's evidence (prints JSON; rc 3 if it cannot be built).
                                                  Default = the r1 mode (ONE package: the compose package).  --fix-pkg selects the r2 TWO-STEP mode (the golden writer's bytes from the r1 compose package;
                                                  the re-run suite, mutation proof and tested-file sha256 from the fix package; see kbuild's docstring)
    manifest --art ART --out OUT --ctrl-runs R [--n-dump-min N] (--dry-run | --write PATH) [--build-json P | --git-dir G [same K-BUILD options as kbuild]]
    rcevidence --reader FILE --evidence FILE...  each evidence file PASSes (differences 0) AND carries THIS reader's sha256

Layout it expects (box 2's chain_v8f_dagoaccorc.sh `runone`):  <ART>/M2b_s<S>_<arm>/{ckpt_last.pt, eval_M2b-<arm>_s<S>.{json,npz}, selfcheck.json, summary.json, argv.json, train_log.jsonl, launch_gate_token.json, SHA256SUMS.txt}
with arm in {dagoaccorc (= A), dagoaccorcrep (= REP)}.  SHA256SUMS.txt lists EXACTLY those 8 names.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True

AR = "E:/Projects/TanitAD-artifacts"
RUNS = [("dagoaccorc", 0), ("dagoaccorcrep", 0), ("dagoaccorc", 1), ("dagoaccorcrep", 1)]      # Amendment 1 section 1 / the box-2 chain's order: A s0, REP s0, A s1, REP s1
ROLE = {"dagoaccorc": "A", "dagoaccorcrep": "REP"}
TAGSTEM = {"dagoaccorc": "DAGORC", "dagoaccorcrep": "DAGORCREP"}
# pinned values, each COPIED from a banked artifact (dagorct1_selftest.py re-derives every one from its source and asserts equality)
M2B_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
OACC_CKPT_SHA = {0: "67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f", 1: "90b01474819082a6d32a9000d63a4954e0074916140abdbbcb67b353f7ac3333"}
BANK1_SHA = {0: "197a68bb540be5a7d157942e9157e1c8e60ca60d82b1ca4fb4fcb68edccfaa8f", 1: "36c7aea964dafe9423253aa479c2fc6cdfc93848395b593315cba812f5f04d4d"}
W_ACC = 49.6969
REP_SEED = {0: 10, 1: 11}
STEPS = 3000
P_A, P_SWAP = 0.3, 0.1
BASE_COMMIT = "aeb544683b1c6b70e29f241fe7133d4290a1b4a5"
BUILD_FILES = ("stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_dagger.py")
COMPOSE_FLAG = "--ego-perturb-with-dagger"
# r1 (the first box-2 chain, commit e7ea20d1) STOPPED on K-REACH (A s0: ego share 0.2492 outside 0.3 +- 0.02; the swap and the perturbation shared one RNG stream): the Master Mind REFUSED it.  Its pulled directory is renamed
# <run>_r1_kreach_stop by the r2 chain, and no run built at the r1 commit may ever be rolled out or read.  Extra refused commits: env DORT1_REFUSE_COMMITS (comma separated, 40-hex).
R1_COMMIT = "e7ea20d1287599409d1758971ded62b2fc4ef375"
REFUSED_SUFFIX = "_r1_kreach_stop"


def refused_commits() -> set:
    return {R1_COMMIT} | {c.strip().lower() for c in os.environ.get("DORT1_REFUSE_COMMITS", "").split(",") if c.strip()}


PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
COMPOSE_PKG = os.environ.get("DORT1_COMPOSE_PKG", f"{PK}/raw/2026-10-28-dag-orc-build/compose")      # where the bit-identity evidence lives in the repo (the r2 fix may move it: env or --compose-pkg)
N_WINDOWS, N_EPISODES = 2399, 141

PULLED = re.compile(r"^(?:\d\d-\d\d \d\d:\d\d )?(M2b_s([01])_(dagoaccorc(?:rep)?)) pulled to (\S+) \(checkpoint sha256 ([0-9a-f]{64})\)\s*$")
REC = re.compile(r"^DAG-ORC \(DAG-OACC base\) run record: (PASS|\[.*?\]) \| (dagoaccorc(?:rep)?) s([01]) \|")
REACH = re.compile(r"^DAG-ORC \(DAG-OACC base\) K-REACH: (PASS|FAIL) \| (dagoaccorc(?:rep)?) \|")
CFAIL = re.compile(r"^ZZDAGOACCORC-CONTROL-FAIL (M2b_s[01]_dagoaccorc(?:rep)?)")
SUMS_LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")


# ------------------------------------------------------------------------------------------------------------------------------------ small utilities
def run_name(arm: str, seed: int) -> str:
    return f"M2b_s{seed}_{arm}"


def eval_tag(arm: str, seed: int) -> str:
    return f"M2b-{arm}_s{seed}"


def sums_names(tag: str) -> list[str]:
    """THE accepted set of a run's SHA256SUMS.txt: the 8 names box 2's `runone` hashes."""
    return ["ckpt_last.pt", f"eval_{tag}.json", f"eval_{tag}.npz", "selfcheck.json", "summary.json", "argv.json", "train_log.jsonl", "launch_gate_token.json"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def npath(p: str) -> str:
    """E:/x/y for E:\\x\\y and /e/x/y alike (no trailing slash)."""
    p = str(p).replace("\\", "/").rstrip("/")
    m = re.match(r"^/([a-zA-Z])/(.*)$", p)
    if m:
        p = f"{m.group(1)}:/{m.group(2)}"
    if re.match(r"^[a-zA-Z]:/", p):
        p = p[0].upper() + p[1:]
    return p


def tail2(p: str) -> str:
    return "/".join(npath(p).split("/")[-2:])


def jload(p: str):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def flags_of(argv: list) -> dict:
    out: dict = {}
    i = 0
    while i < len(argv):
        t = str(argv[i])
        if t.startswith("--"):
            j, vals = i + 1, []
            while j < len(argv) and not str(argv[j]).startswith("--"):
                vals.append(str(argv[j]))
                j += 1
            out.setdefault(t, []).append(vals)
            i = j
        else:
            i += 1
    return out


def one(fl: dict, name: str):
    occ = fl.get(name)
    return occ[0][0] if occ and len(occ) == 1 and len(occ[0]) == 1 else None


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def find_key(obj, key):
    """every value of ``key`` anywhere in a nested dict / list."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out += find_key(v, key)
    elif isinstance(obj, list):
        for v in obj:
            out += find_key(v, key)
    return out


# ------------------------------------------------------------------------------------------------------------------------------------ the box-2 chain log
def parse_boxlog(text: str, art: str) -> dict:
    """Section-wise parse of chain_v8f_dagoaccorc.out.  A run's section starts at its '<run> pulled to <path> (checkpoint sha256 <64 hex>)' line (the LAST such line wins: a re-pull restarts it) and holds the
    run-record line and the K-REACH line that follow it.  Facts only; run_state() decides."""
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    sec: dict = {}
    cur = None
    stops, dones, floors, problems = [], [], [], []
    for i, ln in enumerate(lines):
        m = PULLED.match(ln)
        if m:
            g = m.group(1)
            cur = g
            sec[g] = {"line": i, "arm": m.group(3), "seed": int(m.group(2)), "path": npath(m.group(4)), "sha": m.group(5), "rec": None, "reach": None, "cfail": None, "problems": []}
            continue
        m = REC.match(ln)
        if m:
            g = run_name(m.group(2), int(m.group(3)))
            if cur is None or g != cur:
                problems.append(f"line {i + 1}: a run-record line for {g} outside {g}'s section (the section is {cur})")
                sec.setdefault(g, {"line": None, "arm": m.group(2), "seed": int(m.group(3)), "path": None, "sha": None, "rec": None, "reach": None, "cfail": None, "problems": []})["problems"].append("run-record line outside its section")
            else:
                sec[cur]["rec"] = (m.group(1) == "PASS")
                if m.group(1) != "PASS":
                    sec[cur]["problems"].append(f"run record not PASS: {ln[:200]}")
            continue
        m = REACH.match(ln)
        if m:
            if cur is None or sec[cur]["arm"] != m.group(2):
                problems.append(f"line {i + 1}: a K-REACH line for arm {m.group(2)} outside that arm's section (the section is {cur})")
                if cur is not None:
                    sec[cur]["problems"].append("K-REACH line of another arm inside this section")
            else:
                sec[cur]["reach"] = (m.group(1) == "PASS")
                if m.group(1) != "PASS":
                    sec[cur]["problems"].append(f"K-REACH FAIL: {ln[:200]}")
            continue
        m = CFAIL.match(ln)
        if m:
            sec.setdefault(m.group(1), {"line": None, "arm": None, "seed": None, "path": None, "sha": None, "rec": None, "reach": None, "cfail": None, "problems": []})["cfail"] = ln[:200]
            continue
        if ln.startswith("ZZV8FMINI-STOP"):
            stops.append((i, ln[:240]))
        elif ln.startswith("ZZV8FMINI-DONE"):
            dones.append((i, ln[:240]))
        elif ln.startswith("ZZDISK-FLOOR-STOP"):
            floors.append((i, ln[:240]))
    return {"sec": sec, "stops": stops, "dones": dones, "floors": floors, "problems": problems, "nlines": len(lines), "art": npath(art)}


def run_state(P: dict, arm: str, seed: int) -> tuple[str, str]:
    """-> (NOT-PULLED | AWAITING-CONTROLS | PASS | FAIL, detail).  PASS needs the pulled line AND a PASSing run record AND a PASSing K-REACH line, in that run's section, with no STOP after the section."""
    g = run_name(arm, seed)
    s = P["sec"].get(g)
    if not s:
        return "NOT-PULLED", ""
    bad = list(s["problems"])
    if s["cfail"]:
        bad.append(f"ZZDAGOACCORC-CONTROL-FAIL: {s['cfail']}")
    if s["line"] is None:
        return "FAIL", "; ".join(bad) or f"{g}: control lines without a pulled line"
    if s["path"].endswith(REFUSED_SUFFIX):
        bad.append(f"the pull line names a REFUSED r1 directory {s['path']}")
    if s["path"].lower() != npath(f"{P['art']}/{g}").lower():
        bad.append(f"the box-2 chain pulled {g} to {s['path']}, this chain reads {P['art']}/{g}")
    if s["rec"] is False or s["reach"] is False:
        bad.append("a control line is not PASS")
    if bad:
        return "FAIL", "; ".join(bad)[:600]
    if s["rec"] and s["reach"]:
        return "PASS", s["sha"]
    later = [t for (i, t) in P["stops"] + P["floors"] if i > s["line"]]
    if later:
        return "FAIL", f"the box-2 chain STOPPED after pulling {g} and before its run record and K-REACH both printed PASS: {later[0]}"
    return "AWAITING-CONTROLS", f"run record {'PASS' if s['rec'] else 'not yet'}, K-REACH {'PASS' if s['reach'] else 'not yet'}"


def log_meta(P: dict) -> dict:
    pulled = [g for (a, s) in RUNS for g in [run_name(a, s)] if g in P["sec"] and P["sec"][g]["line"] is not None]
    last = max([P["sec"][g]["line"] for g in pulled] or [-1])
    return {"stops": len(P["stops"]), "pending": len(RUNS) - len(pulled), "pulled": len(pulled),
            "done": int(len(pulled) == len(RUNS) and any(i > last for (i, _t) in P["dones"]))}


# ------------------------------------------------------------------------------------------------------------------------------------ the pull
def pullcheck(d: str, tag: str, identity: tuple | None = None, expect_sha: str | None = None, min_age: float = 3.0, now: float | None = None) -> tuple[int, str, str]:
    """-> (rc, kind, msg).  COMPLETE iff by CONTENT: SHA256SUMS.txt exists, aged, newline-terminated, lists exactly the 8 names once; every file exists, is non-empty and re-hashes to the listed value; summary.json
    says done.  WRONG when those hold but the run's own records say it is not the run asked for (``identity`` = (arm, seed)) or its checkpoint sha256 differs from ``expect_sha`` (the box-2 log's)."""
    d = d.rstrip("/\\")
    if npath(d).endswith(REFUSED_SUFFIX):
        return 3, "WRONG", f"{d} is a REFUSED r1 directory (name ends in {REFUSED_SUFFIX}): it failed K-REACH and is never rolled out or read"
    if not os.path.isdir(d):
        return 1, "INCOMPLETE", f"{d} is absent"
    sp = os.path.join(d, "SHA256SUMS.txt")
    if not os.path.isfile(sp):
        return 1, "INCOMPLETE", "no SHA256SUMS.txt yet"
    if (now or time.time()) - os.path.getmtime(sp) < min_age:
        return 1, "INCOMPLETE", "SHA256SUMS.txt is younger than %.0f s" % min_age
    raw = open(sp, "rb").read().decode("utf-8", "replace")
    if not raw.endswith("\n"):
        return 1, "INCOMPLETE", "SHA256SUMS.txt is not newline-terminated"
    listed: dict = {}
    for ln in raw.split("\n"):
        if not ln.strip():
            continue
        m = SUMS_LINE.match(ln.rstrip("\r"))
        if not m:
            return 3, "WRONG", f"SHA256SUMS.txt has a malformed line {ln[:80]!r}"
        if m.group(2) in listed:
            return 3, "WRONG", f"SHA256SUMS.txt lists {m.group(2)} twice"
        listed[m.group(2)] = m.group(1)
    want = sums_names(tag)
    if sorted(listed) != sorted(want):
        return 1, "INCOMPLETE", f"SHA256SUMS.txt lists {sorted(set(listed) ^ set(want))} differently from the 8 expected names"
    for n in want:
        p = os.path.join(d, n)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return 1, "INCOMPLETE", f"{n} is missing or empty"
        if sha256_file(p) != listed[n]:
            return 1, "INCOMPLETE", f"{n} does not hash to its listed sha256 (a pull still filling, or corrupt)"
    ck = listed["ckpt_last.pt"]
    try:
        sm, ar, tk = jload(os.path.join(d, "summary.json")), jload(os.path.join(d, "argv.json")), jload(os.path.join(d, "launch_gate_token.json"))
        jload(os.path.join(d, "selfcheck.json"))
        jload(os.path.join(d, f"eval_{tag}.json"))
    except (OSError, ValueError) as ex:
        return 1, "INCOMPLETE", f"a record does not parse: {ex}"
    if sm.get("done") is not True:
        return 1, "INCOMPLETE", "summary.json does not say done: true"
    tc = str((tk.get("binding") or {}).get("commit", "")).lower()
    if tc in refused_commits():
        return 3, "WRONG", f"its launch-gate token binds the REFUSED commit {tc[:12]} (the r1 build, stopped on K-REACH): never rolled out or read"
    if expect_sha and ck != expect_sha:
        return 3, "WRONG", f"the pulled ckpt_last.pt hashes to {ck[:16]}, the box-2 log printed {expect_sha[:16]}"
    why = "8 names re-hashed, summary done"
    if identity:
        bad = identity_problems(d, tag, identity[0], int(identity[1]), ck, sm, ar, tk)
        if bad:
            return 3, "WRONG", "; ".join(bad)[:700]
        why += f", identity {ROLE[identity[0]]} s{identity[1]} (argv / init / bank / w_acc / perturbation / token)"
    return 0, "COMPLETE", f"{ck} {why}"


def identity_problems(d: str, tag: str, arm: str, seed: int, ck: str, sm: dict, ar: dict, tk: dict) -> list[str]:
    """Is this directory the run it is named after?  (A safety net: the box-2 chain asserted all of it, and the READER re-asserts it as K-PAIR / K-INIT / K-REACH / SUMMARY / K-GATE.)"""
    bad: list[str] = []
    argv = [str(t) for t in ar.get("argv", [])]
    fl = flags_of(argv)
    g = run_name(arm, seed)
    sd = seed if arm == "dagoaccorc" else REP_SEED[seed]
    for name, want in (("--steps", STEPS), ("--dagger-p", P_SWAP), ("--w-acc", W_ACC)):
        v = fnum(one(fl, name))
        if v is None or abs(v - want) > 1e-9:
            bad.append(f"argv {name} {one(fl, name)!r}, registered {want}")
    v = one(fl, "--seed")
    if v is None or int(float(v)) != sd:
        bad.append(f"argv --seed {v!r}, expected {sd}")
    if sm.get("seed") != sd:
        bad.append(f"summary seed {sm.get('seed')!r}, expected {sd}")
    if (one(fl, "--dagger-bank-sha256") or "").lower() != BANK1_SHA[seed]:
        bad.append("argv --dagger-bank-sha256 is not DAG-OACC's iteration-1 bank")
    v = one(fl, "--init-ckpt") or ""
    if not npath(v).endswith(f"/M2b_s{seed}/ckpt_last.pt"):
        bad.append(f"argv --init-ckpt {v!r} is not .../M2b_s{seed}/ckpt_last.pt")
    ish = {str(x) for x in find_key(sm.get("levers", {}), "sha256")}
    if M2B_SHA[seed] not in ish:
        bad.append(f"summary levers do not record the init checkpoint {M2B_SHA[seed][:12]} (found {sorted(x[:12] for x in ish)})")
    o = one(fl, "--out") or ""
    if not npath(o).endswith("/" + g):
        bad.append(f"argv --out {o!r} does not end in {g}")
    ep = fnum(one(fl, "--ego-perturb-p"))
    if arm == "dagoaccorc":
        if ep is None or abs(ep - P_A) > 1e-9:
            bad.append(f"A: argv --ego-perturb-p {one(fl, '--ego-perturb-p')!r}, registered {P_A}")
        if COMPOSE_FLAG not in argv:
            bad.append(f"A: argv does not name {COMPOSE_FLAG}")
        if fnum(one(fl, "--ego-perturb-seed")) != float(seed):
            bad.append(f"A: argv --ego-perturb-seed {one(fl, '--ego-perturb-seed')!r}, expected {seed}")
    else:
        if ep is None or ep != 0.0:
            bad.append(f"REP: argv --ego-perturb-p {one(fl, '--ego-perturb-p')!r}, must be 0")
        if COMPOSE_FLAG in argv:
            bad.append(f"REP: argv names {COMPOSE_FLAG}")
    if ck in (OACC_CKPT_SHA[seed], M2B_SHA[seed]):
        bad.append("its checkpoint is DAG-OACC's (or M2b's): nothing was trained")
    if tk.get("kind") != "launch-gate-token" or tk.get("verdict") != "PASS" or tk.get("run") != g:
        bad.append(f"launch-gate token: kind {tk.get('kind')!r} verdict {tk.get('verdict')!r} run {tk.get('run')!r} (expected a PASS token for {g})")
    if not re.fullmatch(r"[0-9a-f]{40}", str((tk.get("binding") or {}).get("commit", ""))):
        bad.append("launch-gate token binds no 40-hex commit")
    return bad


# ------------------------------------------------------------------------------------------------------------------------------------ K-BUILD
def _git(g: str, *a, text: bool = False):
    r = subprocess.run(["git", "-C", g, "-c", "safe.directory=*", *a], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(a[:3])}: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return r.stdout.decode("utf-8", "replace") if text else r.stdout


def commit_of_dir(d: str) -> str:
    return str((jload(os.path.join(d, "launch_gate_token.json")).get("binding") or {}).get("commit", "")).lower()


def launch_commit_of(art: str) -> str:
    cs = set()
    for arm, s in RUNS:
        p = f"{art}/{run_name(arm, s)}/launch_gate_token.json"
        if not os.path.isfile(p):
            raise RuntimeError(f"{p} is absent: the launch commit is read from the four launch-gate tokens")
        cs.add(str((jload(p).get("binding") or {}).get("commit", "")))
    if len(cs) != 1 or not re.fullmatch(r"[0-9a-f]{40}", next(iter(cs))):
        raise RuntimeError(f"the four launch-gate tokens bind {sorted(cs)}: exactly one 40-hex launch commit is required")
    if next(iter(cs)).lower() in refused_commits():
        raise RuntimeError(f"the four launch-gate tokens bind the REFUSED commit {next(iter(cs))[:12]} (the r1 build)")
    return next(iter(cs))


MIN_COMPOSE_TESTS, MIN_MUTANTS = 26, 19      # the r1 compose package's suite size and mutant count (its RESULTS.md); a re-run for the r2 fix may only add
# --- the r2 TWO-STEP evidence (kbuild(..., fix_pkg=P)): step 1 = the r1 compose package (names the bytes the golden was WRITTEN by); step 2 = the fix package (the re-run on the launch bytes)
R1_COMPOSE_TEST = "stack/tests/test_v8f_ego_dagger_compose.py"
GOLDEN_FILE = "stack/tests/ego_dagger_compose_golden.json"
FIX_SUITE_LOG = "logs/suite_test_v8f_ego_dagger_compose.log"
LISTED_SHA = re.compile(r"^([0-9a-f]{64})(?: \*| {2})(stack/\S+)[ \t\r]*$", re.M)      # '<sha64> *path' (sha256sum binary) AND '<sha64>  path' (two spaces); a line with trailing text ('  path  (e7ea20d1)') is NOT a listing


def listed_shas(text: str):
    """The 'tested file' sha256 listing of a RESULTS.md, both '<sha> *path' and '<sha>  path' forms.  -> (path -> sha, path -> [shas] for a path listed with DIFFERENT shas: ambiguous = refused by the caller)."""
    seen: dict = {}
    for m in LISTED_SHA.finditer(text):
        seen.setdefault(m.group(2), []).append(m.group(1))
    return {p: v[0] for p, v in seen.items() if len(set(v)) == 1}, {p: sorted(set(v)) for p, v in seen.items() if len(set(v)) > 1}


def _blob_sha(g: str, commit: str, path: str):
    """sha256 of <commit>:<path>, or None when the path is not in that commit (absent != equal)."""
    try:
        return hashlib.sha256(_git(g, "cat-file", "blob", f"{commit}:{path}")).hexdigest()
    except RuntimeError:
        return None


def _fix_evidence(g: str, launch: str, evc: str, r1c: str, differing: list, files: dict, compose_pkg: str, fix_pkg: str):
    """The r2 two-step bit-identity evidence.  -> (problems, ev).  A package FILE that is absent from its commit RAISES (the evidence is not in the mirror yet: 'NOT BUILT'); a CONTENT that does not support the claim is a
    problem (the hunks are then written 'UNVERIFIED: ...' and the reader's k_build() refuses).  Everything is read from git objects with `cat-file blob` (raw bytes: no autocrlf, no textconv); prose is never the arbiter of a sha256."""
    problems: dict = {}
    efiles: list = []

    def rd(commit, path, role):
        b = _git(g, "cat-file", "blob", f"{commit}:{path}")
        efiles.append({"role": role, "commit": commit, "path": path, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)})
        return b.decode("utf-8", "replace")
    r1_results = rd(r1c, f"{compose_pkg}/RESULTS.md", "step 1: r1 compose package RESULTS.md (names the bytes the golden was written by)")
    fix_results = rd(evc, f"{fix_pkg}/RESULTS.md", "step 2: fix package RESULTS.md (suite row, mutation proof, tested-file sha256)")
    fix_log = rd(evc, f"{fix_pkg}/{FIX_SUITE_LOG}", "step 2: fix package compose-suite log")
    # --- step 2: the suite (RESULTS.md row AND the log), never fewer than the r1 package's, no failure, the two agree
    ms = re.search(r"\|\s*test_v8f_ego_dagger_compose\s*\|\s*(\d+) passed[^|]*\|\s*-\s*\|", fix_results)
    n_table = int(ms.group(1)) if ms else None
    if n_table is None or n_table < MIN_COMPOSE_TESTS:
        problems["fix:suite"] = f"fix RESULTS.md does not show test_v8f_ego_dagger_compose '>= {MIN_COMPOSE_TESTS} passed' with no failure"
    lines = [x.strip() for x in fix_log.replace("\r", "\n").split("\n") if x.strip()]
    ml = re.match(r"^=*\s*(\d+) passed\b", lines[-1]) if lines else None
    n_log = int(ml.group(1)) if ml else None
    why = []
    if n_log is None or n_log < MIN_COMPOSE_TESTS:
        why.append(f"does not END in '>= {MIN_COMPOSE_TESTS} passed' (last line {lines[-1][:60]!r})" if lines else "is empty")
    if re.search(r"\b\d+ (failed|errors?)\b", fix_log):
        why.append("reports a failure / error")
    if any(re.fullmatch(r"[.sxXFE]+(\s*\[\s*\d+%\])?", x) and re.search(r"[FE]", x) for x in lines):
        why.append("its progress line carries an F / E")
    if n_table is not None and n_log is not None and n_table != n_log:
        why.append(f"counts {n_log} passed, RESULTS.md's table {n_table}")
    if why:
        problems["fix:suite_log"] = f"{fix_pkg}/{FIX_SUITE_LOG} " + "; ".join(why)
    # --- step 2: the mutation proof (M13 = the flag-off path; every mutant killed; the unmutated controls GREEN; the SUMMARY agrees with the per-mutant lines)
    mm = re.search(r"SUMMARY: (\d+)/(\d+) mutants killed; survivors: none", fix_results)
    mut = re.findall(r"^(M\d+[a-z]?_\w+): (.+?)(?:  rule:.*)?$", fix_results, re.M)
    killed = [n for n, st in mut if st.startswith("RED (killed)")]
    ctl = re.findall(r"^CONTROL \(no mutation\)[^\n]*?: rc=(\d+)", fix_results, re.M)
    wm = []
    if not re.search(r"^M13_off_path_changed: RED \(killed\)", fix_results, re.M):
        wm.append("M13_off_path_changed is not shown killed")
    if not mm or mm.group(1) != mm.group(2) or int(mm.group(2)) < MIN_MUTANTS:
        wm.append(f"no 'SUMMARY: n/n mutants killed; survivors: none' with n >= {MIN_MUTANTS}")
    elif len(mut) != int(mm.group(2)) or len(killed) != len(mut):
        wm.append(f"the SUMMARY says {mm.group(2)} but {len(mut)} mutant lines are listed, {len(killed)} of them 'RED (killed)'")
    if not ctl or any(c != "0" for c in ctl):
        wm.append(f"the unmutated controls are not all rc=0 (GREEN): {ctl}")
    if wm:
        problems["fix:mutation"] = "fix RESULTS.md: " + "; ".join(wm)
    # --- step 2: the tested files ARE the launch bytes (the fix package's own sha256 listing == the launch commit's blobs): the changed build files AND the compose test file
    listed, conflicts = listed_shas(fix_results)
    for f in sorted(set(differing) | {R1_COMPOSE_TEST} | set(listed) | set(conflicts)):
        lb = files[f]["sha256_launch"] if f in files else _blob_sha(g, launch, f)
        if f in conflicts:
            problems[f"fix:launch_bytes:{f}"] = f"listed with different sha256 {[x[:16] for x in conflicts[f]]}: ambiguous"
        elif lb is None or listed.get(f) != lb:
            problems[f"fix:launch_bytes:{f}"] = f"the package's tested file sha256 {str(listed.get(f))[:16]} != the launch commit's blob {str(lb)[:16]}"
    # --- step 1: the golden was written by the UNMODIFIED bytes (the r1 package's 'Unmodified tip files') == the BASE blobs; and the golden BLOB is the same at the launch commit as at the r1 evidence commit
    unmod = {m.group(1): m.group(2) for m in re.finditer(r"`(v8f_mini_\w+\.py)` ([0-9a-f]{8,64})", r1_results.split("Unmodified tip files", 1)[-1])} if "Unmodified tip files" in r1_results else {}
    for f in differing:
        u = unmod.get(os.path.basename(f), "")
        if len(u) < 16 or not files[f]["sha256_base"].startswith(u):
            problems[f"r1:golden_writer:{f}"] = f"the r1 package's golden-writing file {u or 'NOT NAMED'} != the base blob {files[f]['sha256_base'][:16]}"
    gl, g1, gb = _blob_sha(g, launch, GOLDEN_FILE), _blob_sha(g, r1c, GOLDEN_FILE), _blob_sha(g, BASE_COMMIT, GOLDEN_FILE)
    if gl is None or g1 is None or gl != g1:
        problems["golden"] = f"{GOLDEN_FILE} at the launch commit {str(gl)[:16]} != at the r1 evidence commit {r1c[:8]} {str(g1)[:16]}"
    ev = {"mode": "fix-pkg: two-step (r1 golden writer + r2 fix re-run)", "package": fix_pkg, "evidence_commit": evc,
          "results_md_sha256": efiles[1]["sha256"], "suite_log_sha256": efiles[2]["sha256"],      # raw blob bytes (= `git cat-file blob | sha256sum`)
          "test": f"{R1_COMPOSE_TEST}::test_flag_off_reproduces_the_unmodified_tip_bit_for_bit ({n_table} passed in the fix package's table, {n_log} in its log)",
          "mutant": f"M13_off_path_changed killed ({len(killed)}/{len(mut)} mutants killed; {len(ctl)} unmutated controls rc=0)",
          "step1_golden_writer": {"package": compose_pkg, "evidence_commit": r1c, "unmodified_tip_files": unmod},
          "golden": {"path": GOLDEN_FILE, "sha256_launch": gl, "sha256_r1_evidence_commit": g1, "sha256_base": gb, "r1_evidence_commit": r1c},
          "tested_files_listed_in_fix_results": listed,
          "evidence_files": efiles}
    return problems, ev


def kbuild(art: str, git_dir: str, launch: str | None = None, compose_pkg: str | None = None, evidence_commit: str | None = None, fix_pkg: str | None = None, r1_evidence_commit: str | None = None) -> dict:
    """The K-BUILD record for dag_orc_read.k_build(): {"launch_commit", "base_commit", "files": {path: {sha256_launch, sha256_base}}, "changed": [{file, hunk, behind_flag, default_off, bit_identity_test}]}.
    sha256 and hunks are MEASURED from the git objects.  The hunk -> flag attribution is ATTESTED at file level, not derived hunk by hunk: it rests on the compose package's flag-off bit-identity test
    (test_flag_off_reproduces_the_unmodified_tip_bit_for_bit), whose evidence this function checks is BOUND TO THE LAUNCH BYTES (the package's own sha256 list == the launch blobs) and to the BASE bytes (the golden was
    written by the unmodified file DAG-OACC trained).  Anything it cannot verify is written as 'UNVERIFIED: <why>', which the reader's k_build() refuses.
    TWO MODES.  Default (fix_pkg None) = r1: ONE package (compose_pkg, read at evidence_commit, default the launch commit) carries all of it.
    fix_pkg = P = r2: the evidence is a TWO-STEP chain and each step is read from the commit that holds it:
      step 1 (compose_pkg at r1_evidence_commit, default R1_COMMIT e7ea20d1): its 'Unmodified tip files' name the bytes the golden was WRITTEN by == the BASE_COMMIT blobs; and the golden BLOB at the launch commit ==
              the golden blob at that r1 evidence commit (git, not prose: the fix did not touch the reference the flag-off test compares against);
      step 2 (fix_pkg at evidence_commit, default the launch commit): the flag-off test was RE-RUN on the launch bytes: the package's own '<sha256>  <path>' listing (two-space or ' *' form) == the launch blobs (the changed
              build files AND the compose test file); the compose suite row and its log END in >= MIN_COMPOSE_TESTS passed with no failure and agree; M13_off_path_changed (the flag-off path) is killed; every one of
              >= MIN_MUTANTS mutants is killed (the SUMMARY agrees with the per-mutant lines; the unmutated controls are rc 0).
    Every evidence path and its sha256 are recorded in the returned dict (evidence.evidence_files)."""
    launch = launch or launch_commit_of(art)
    compose_pkg = compose_pkg or COMPOSE_PKG
    evc = evidence_commit or launch      # the commit whose tree holds the evidence package (default: the launch commit; a later mirror tip if the fix's evidence landed after it)
    r1c = r1_evidence_commit or R1_COMMIT      # fix_pkg mode only: the commit that holds the r1 compose package and the golden's reference blob
    for c in (launch, BASE_COMMIT, evc) + ((r1c,) if fix_pkg else ()):
        _git(git_dir, "cat-file", "-e", f"{c}^{{commit}}")
    files, changed, ev = {}, [], {}
    for f in BUILD_FILES:
        bl, bb = _git(git_dir, "cat-file", "blob", f"{launch}:{f}"), _git(git_dir, "cat-file", "blob", f"{BASE_COMMIT}:{f}")
        files[f] = {"sha256_launch": hashlib.sha256(bl).hexdigest(), "sha256_base": hashlib.sha256(bb).hexdigest()}
    differing = [f for f in BUILD_FILES if files[f]["sha256_launch"] != files[f]["sha256_base"]]
    problems: dict = {}
    if differing:      # the flag-off evidence is only read when a build file differs from the base (nothing changed -> nothing to attest)
        if fix_pkg:
            problems, ev = _fix_evidence(git_dir, launch, evc, r1c, differing, files, compose_pkg, fix_pkg)
        else:
            results = _git(git_dir, "show", f"{evc}:{compose_pkg}/RESULTS.md", text=True)
            run2 = _git(git_dir, "show", f"{evc}:{compose_pkg}/logs/compose_run2.log", text=True)
            # the r1 package showed 26 passed / 19 of 19 mutants killed; a re-run for the r2 fix may add tests or mutants, so the bar is 'at least the r1 package's, no failure, no survivor'
            ms = re.search(r"\|\s*test_v8f_ego_dagger_compose\s*\|\s*(\d+) passed[^|]*\|\s*-\s*\|", results)
            if not ms or int(ms.group(1)) < MIN_COMPOSE_TESTS:
                problems["suite"] = f"RESULTS.md does not show test_v8f_ego_dagger_compose '>= {MIN_COMPOSE_TESTS} passed' with no failure"
            mm = re.search(r"SUMMARY: (\d+)/(\d+) mutants killed; survivors: none", results)
            if not re.search(r"^M13_off_path_changed: RED \(killed\)", results, re.M) or not mm or mm.group(1) != mm.group(2) or int(mm.group(2)) < MIN_MUTANTS:
                problems["mutation"] = f"RESULTS.md does not show M13_off_path_changed killed and every one of >= {MIN_MUTANTS} mutants killed with no survivor"
            mr = re.findall(r"\b(\d+) passed\b", run2)
            if not mr or int(mr[-1]) < MIN_COMPOSE_TESTS or re.search(r"\b\d+ (failed|error)", run2):
                problems["run2"] = f"compose/logs/compose_run2.log does not end in '>= {MIN_COMPOSE_TESTS} passed' with no failure"
            listed = {m.group(2): m.group(1) for m in re.finditer(r"^([0-9a-f]{64}) \*(stack/\S+)$", results, re.M)}
            unmod = {m.group(1): m.group(2) for m in re.finditer(r"`(v8f_mini_\w+\.py)` ([0-9a-f]{8,64})", results.split("Unmodified tip files", 1)[-1])}
            for f in differing:
                if listed.get(f) != files[f]["sha256_launch"]:
                    problems[f"launch_bytes:{f}"] = f"the package's tested file sha256 {str(listed.get(f))[:16]} != the launch commit's blob {files[f]['sha256_launch'][:16]}"
                if not files[f]["sha256_base"].startswith(unmod.get(os.path.basename(f), "?" * 8)[:16]):
                    problems[f"base_bytes:{f}"] = f"the golden's unmodified file {unmod.get(os.path.basename(f))} != the base blob {files[f]['sha256_base'][:16]}"
        src = _git(git_dir, "cat-file", "blob", f"{launch}:stack/scripts/v8f_mini_train.py").decode("utf-8", "replace")
        i = src.find(f'add_argument("{COMPOSE_FLAG}"')
        snip = src[i:i + 400] if i >= 0 else ""
        default_off = bool(i >= 0 and 'action="store_true"' in snip and "default=False" in snip)
        if not default_off:
            problems["flag"] = f"the launch trainer does not declare {COMPOSE_FLAG} as store_true with default=False"
        if not fix_pkg:
            ev = {"package": compose_pkg, "evidence_commit": evc, "results_md_sha256": hashlib.sha256(results.encode("utf-8")).hexdigest(), "compose_run2_log_sha256": hashlib.sha256(run2.encode("utf-8")).hexdigest(),
                  "test": "stack/tests/test_v8f_ego_dagger_compose.py::test_flag_off_reproduces_the_unmodified_tip_bit_for_bit (26 passed in the package's final run)", "mutant": "M13_off_path_changed killed (19/19)"}
        for f in differing:
            d = _git(git_dir, "diff", "-U0", "--no-color", BASE_COMMIT, launch, "--", f, text=True)
            for h in re.findall(r"^(@@ [^@]+ @@)", d, re.M):
                changed.append({"file": f, "hunk": h, "behind_flag": COMPOSE_FLAG, "default_off": default_off,
                                "bit_identity_test": "PASS" if not problems else "UNVERIFIED: " + "; ".join(f"{k}: {v}" for k, v in problems.items())[:400]})
            if not re.search(r"^@@", d, re.M):
                changed.append({"file": f, "hunk": "(no hunk header parsed)", "behind_flag": COMPOSE_FLAG, "default_off": False, "bit_identity_test": "UNVERIFIED: no diff"})
    return {"launch_commit": launch, "base_commit": BASE_COMMIT, "files": files, "changed": changed, "identical_files": [f for f in BUILD_FILES if f not in differing],
            "evidence": ev, "attested_by": "dagorct1_lib.py kbuild: sha256 and hunks MEASURED from git objects; hunk->flag attribution ATTESTED at file level by the compose package's flag-off bit-identity test bound to the launch bytes (see its docstring)"}


# ------------------------------------------------------------------------------------------------------------------------------------ the manifest
def t1_entry(out: str, role_arm: str, seed: int, limit: str) -> dict:
    t = f"{TAGSTEM[role_arm]}_s{seed}_m10_{limit}_heads"
    return {"tag": t, "ticks": f"{out}/ticks/ticks_{t}.npz", "dump_dir": f"{out}/dump_{t}", "json": f"{out}/t1v8f_{t}.json", "argv": f"{out}/argv/{t}.argv", "done": f"{out}/done_{t}.txt"}


def t0f_tag(arm: str, seed: int) -> str:
    return f"{eval_tag(arm, seed)}-fresh"


def build_manifest(art: str, out: str, ctrl_runs: str, build: dict | None) -> dict:
    arms: dict = {"A": {}, "REP": {}}
    for arm, s in RUNS:
        run, tg = run_name(arm, s), eval_tag(arm, s)
        d = f"{art}/{run}"
        arms[ROLE[arm]][str(s)] = {"ckpt_tail": f"{run}/ckpt_last.pt", "train_dir": d, "token": f"{d}/launch_gate_token.json",
                                   "t1": t1_entry(out, arm, s, "causal"), "t1_logged": t1_entry(out, arm, s, "logged"),
                                   "t0": {"json": f"{d}/eval_{tg}.json", "npz": f"{d}/eval_{tg}.npz"},
                                   "t0_fresh": {"json": f"{out}/t0fresh/eval_{t0f_tag(arm, s)}.json", "npz": f"{out}/t0fresh/eval_{t0f_tag(arm, s)}.npz"}}
    arms["CTRL"] = jload(ctrl_runs)["arms"]["DAGOACC"]
    m = {"seeds": [0, 1], "arms": arms, "provenance": {
        "prereg": "Project Steering/PREREG_V8F_DAG_ORC.md @ 3b7b404e + Amendment 1 (A1-2026-10-27, re-based on DAG-OACC), claim row H-V8F-DAG-ORC-1",
        "built_by": "dagorct1_lib.py manifest (the dev-box T1-kin chain, dagorct1_chain.sh)",
        "slots": "A := dagoaccorc (M2b_s<S>_dagoaccorc), REP := dagoaccorcrep (M2b_s<S>_dagoaccorcrep), CTRL := DAG-OACC (banked: v8f-dagoacc + v8f-dagoacc-t1, arms.DAGOACC of its runs.json; read light: t1 + t0). B is not queued.",
        "t1": "v8f_t1_eval.py run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --dump-ticks --dump-heads, plain (no post-map); lanes: --limit causal (decisive) and --limit logged (K4 only)",
        "t0": "box 2's standard eval (--ages cycle): R-T0", "t0_fresh": "dev-box CPU --ages fresh eval: K4's comparator"}}
    if build is not None:
        m["build"] = build
    return m


def verify_manifest(m: dict, n_dump_min: int) -> list[str]:
    """Every file the manifest points at exists; every entry is THIS arm's; the done records name the pulled checkpoint; no smoke output (max_windows) is in it; the CTRL slot is the banked DAG-OACC."""
    bad: list[str] = []

    def need(p, what):
        if not p or not os.path.isfile(p) or os.path.getsize(p) == 0:
            bad.append(f"{what}: {p} missing or empty")

    seen: dict = {}
    commits: set = set()
    for role in ("A", "REP"):
        for seed in ("0", "1"):
            e = (m["arms"].get(role) or {}).get(seed)
            lab = f"{role} s{seed}"
            if not e:
                bad.append(f"{lab}: no entry")
                continue
            arm = "dagoaccorc" if role == "A" else "dagoaccorcrep"
            d = e["train_dir"]
            ck = f"{d}/ckpt_last.pt"
            if REFUSED_SUFFIX in npath(d):
                bad.append(f"{lab}: train_dir {d} is a REFUSED r1 directory")
            need(ck, f"{lab} ckpt")
            try:
                cm = commit_of_dir(d)
                commits.add(cm)
                if cm in refused_commits():
                    bad.append(f"{lab}: its launch-gate token binds the REFUSED commit {cm[:12]}")
            except (OSError, ValueError) as ex:
                bad.append(f"{lab}: launch-gate token unreadable: {ex}")
            sha = sha256_file(ck) if os.path.isfile(ck) else None
            if sha in seen:
                bad.append(f"{lab}: its checkpoint {str(sha)[:12]} is also {seen[sha]}'s")
            if sha:
                seen[sha] = lab
                if sha == OACC_CKPT_SHA[int(seed)]:
                    bad.append(f"{lab}: its checkpoint IS DAG-OACC_s{seed}'s")
            for k in (e["token"], e["t0"]["json"], e["t0"]["npz"]):
                need(k, f"{lab} file")
            for f in ("argv.json", "summary.json", "selfcheck.json"):
                need(f"{d}/{f}", f"{lab} {f}")
            for kind, want_limit in (("t1", "causal"), ("t1_logged", "logged")):
                t = e[kind]
                for k in ("ticks", "json", "done", "argv"):
                    need(t[k], f"{lab} {kind} {k}")
                nd = len(glob.glob(os.path.join(t["dump_dir"], "ep*.npz")))
                if nd < n_dump_min:
                    bad.append(f"{lab} {kind}: {nd} ep*.npz in {t['dump_dir']} (< {n_dump_min})")
                if os.path.isfile(t["json"]):
                    j = jload(t["json"]).get("t1v8f", {})
                    if tail2(j.get("ckpt", "")) != e["ckpt_tail"]:
                        bad.append(f"{lab} {kind}: the rollout json scored {tail2(j.get('ckpt', ''))}, not {e['ckpt_tail']}")
                    if j.get("max_windows") is not None or j.get("limit") != want_limit or j.get("n_windows") != N_WINDOWS:
                        bad.append(f"{lab} {kind}: rollout json max_windows {j.get('max_windows')!r} limit {j.get('limit')!r} n_windows {j.get('n_windows')!r} (a full {want_limit} run is required)")
                if os.path.isfile(t["done"]):
                    dr = jload(t["done"])
                    if dr.get("ckpt_sha256") != sha or dr.get("smoke") is not False:
                        bad.append(f"{lab} {kind}: done record names {str(dr.get('ckpt_sha256'))[:12]} smoke={dr.get('smoke')!r}, the pulled checkpoint is {str(sha)[:12]}")
            tf = e["t0_fresh"]
            for k in ("json", "npz"):
                need(tf[k], f"{lab} t0_fresh {k}")
            dn = os.path.join(os.path.dirname(os.path.dirname(tf["json"])), f"done_t0f_{t0f_tag(arm, int(seed))}.txt")
            need(dn, f"{lab} fresh-T0 done record")
            if os.path.isfile(dn) and jload(dn).get("ckpt_sha256") != sha:
                bad.append(f"{lab}: the fresh-T0 done record names another checkpoint than the pulled one")
            for pj, want in ((tf["json"], "fresh"), (e["t0"]["json"], "cycle")):
                if os.path.isfile(pj):
                    jj = jload(pj)
                    if jj.get("ages") != want or tail2(str(jj.get("ckpt", ""))) != e["ckpt_tail"]:
                        bad.append(f"{lab}: {os.path.basename(pj)} ages {jj.get('ages')!r} ckpt {tail2(str(jj.get('ckpt', '')))!r} (expected {want}, {e['ckpt_tail']})")
    if len(commits) > 1:
        bad.append(f"the four runs were built at different commits {sorted(x[:12] for x in commits)}: one launch commit is required")
    c = m["arms"].get("CTRL") or {}
    for seed in ("0", "1"):
        e = c.get(seed)
        if not e:
            bad.append(f"CTRL s{seed}: no entry")
            continue
        ck = f"{e['train_dir']}/ckpt_last.pt"
        if not os.path.isfile(ck) or sha256_file(ck) != OACC_CKPT_SHA[int(seed)]:
            bad.append(f"CTRL s{seed}: {ck} is not DAG-OACC_s{seed}'s checkpoint")
        need(e["t1"]["json"], f"CTRL s{seed} t1 json")
    return bad



# ------------------------------------------------------------------------------------------------------------------------------------ does a trainer change touch the rollout harness?
ROLL_ROOTS = ("stack/scripts/v8f_t1_eval.py", "stack/scripts/v8f_mini_eval.py", "taniteval/tools/t1_eval.py")      # the three programs this chain runs (the T1 harness, the fresh-age T0 eval, the scorer the harness loads by path)


def _modfile(tree: str, dotted: str) -> str | None:
    """tree-relative path of an in-tree module (tanitad.* -> stack/tanitad, taniteval.* -> taniteval/taniteval), else None."""
    parts = dotted.split(".")
    base = {"tanitad": "stack", "taniteval": "taniteval"}.get(parts[0])
    if base is None:
        return None
    for cand in (f"{base}/{'/'.join(parts)}.py", f"{base}/{'/'.join(parts)}/__init__.py"):
        if os.path.isfile(os.path.join(tree, cand)):
            return cand
    return None


def _imports(tree: str, rel: str, src: str):
    """-> (set of module files imported anywhere in the file, incl. inside functions;  list of (module file, kind, name, alias) uses)."""
    import ast
    t = ast.parse(src)
    files, uses = set(), []
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names:
                f = _modfile(tree, a.name)
                if f:
                    files.add(f)
                    uses.append((f, "module", None, a.asname or a.name))
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            for a in n.names:
                sub = _modfile(tree, f"{n.module}.{a.name}")       # `from pkg import module`
                if sub:
                    files.add(sub)
                    uses.append((sub, "module", None, a.asname or a.name))
                    continue
                f = _modfile(tree, n.module)
                if f:
                    files.add(f)
                    uses.append((f, "star" if a.name == "*" else "name", a.name, a.asname or a.name))
    return files, uses


def _top(src: str) -> dict:
    """top-level name -> ast.dump of its definition; '<module-level>' -> the dump of every other statement (imports, side effects)."""
    import ast
    t = ast.parse(src)
    out, other = {}, []
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.name] = ast.dump(n)
        elif isinstance(n, ast.Assign):
            for tg in n.targets:
                for nm in ast.walk(tg):
                    if isinstance(nm, ast.Name):
                        out[nm.id] = ast.dump(n)
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            tg = n.target
            if isinstance(tg, ast.Name):
                out[tg.id] = ast.dump(n)
            else:
                other.append(ast.dump(n))
        elif isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant) and isinstance(n.value.value, str):
            continue      # docstring
        else:
            other.append(ast.dump(n))
    out["<module-level>"] = "\n".join(other)
    return out


def _refs_in(src: str) -> dict:
    """top-level name -> the set of top-level names its definition references (ast.Name ids)."""
    import ast
    t = ast.parse(src)
    tops = set(_top(src))
    refs: dict = {}
    for n in t.body:
        names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)} & tops
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            refs.setdefault(n.name, set()).update(names)
        elif isinstance(n, ast.Assign):
            for tg in n.targets:
                for nm in ast.walk(tg):
                    if isinstance(nm, ast.Name):
                        refs.setdefault(nm.id, set()).update(names)
    return refs


def gate_failures(r: dict, gate: list) -> list:
    """gated files with a reachable differing name, or absent at the launch commit, or not in the closure at all (a gate on a file that is never loaded is a mis-specified gate)."""
    return [rel for rel in gate if (r["differs"].get(rel) or {}).get("touched") or r["files"].get(rel) in (None, "ABSENT_AT_LAUNCH")]


def rolloutcode(tree: str, git_dir: str, launch: str, loaded: list | None = None) -> dict:
    """Which code do the rollout / fresh-T0 programs import, and does the launch commit differ in any of it?  STATIC import closure of ROLL_ROOTS inside ``tree`` (the tool tree DAG-OACC was rolled out with), every file's sha256 against
    the launch commit's blob; for a differing module, the top-level names that differ (AST, comments and layout ignored) are intersected with the names the closure REACHES in it (``alias.name`` / ``from M import name`` uses,
    followed through the module's own call graph; module-level statements that differ always count as touching).  -> {"files": {rel: IDENTICAL|DIFFERS|ABSENT_AT_LAUNCH}, "differs": {rel: {...}}, "touched": bool}.
    The rollout programs import ONLY from ``tree`` (never from the trainer's checkout), so this answers 'would the harness behave differently if it were run from the launch commit's tree'."""
    import ast
    _git(git_dir, "cat-file", "-e", f"{launch}^{{commit}}")      # a commit that is not in the mirror must not read as 'every file absent, nothing differs'
    seen, todo, src_of, uses_of = set(), list(ROLL_ROOTS), {}, {}
    if loaded is not None:      # the EXACT runtime set (trace_imports.py): the .py files of ``tree`` a real run loaded, plus the roots
        tn = npath(tree).rstrip("/") + "/"
        todo = sorted({npath(f)[len(tn):] for f in loaded if npath(f).startswith(tn) and f.endswith(".py")} | set(ROLL_ROOTS))
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        src = open(os.path.join(tree, rel), encoding="utf-8").read()
        src_of[rel] = src
        files, uses = _imports(tree, rel, src)
        uses_of[rel] = uses
        if loaded is None:
            todo += [f for f in files if f not in seen]
    res: dict = {"files": {}, "differs": {}, "touched": False, "roots": list(ROLL_ROOTS), "basis": "runtime import trace" if loaded is not None else "static import closure"}
    for rel in sorted(seen):
        try:
            blob = _git(git_dir, "cat-file", "blob", f"{launch}:{rel}")
        except RuntimeError:
            res["files"][rel] = "ABSENT_AT_LAUNCH"
            continue
        raw = open(os.path.join(tree, rel), "rb").read()
        res["files"][rel] = "IDENTICAL" if hashlib.sha256(blob).hexdigest() == hashlib.sha256(raw).hexdigest() else "DIFFERS"
        if res["files"][rel] == "DIFFERS":
            a, b = _top(src_of[rel]), _top(blob.decode("utf-8", "replace"))
            changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            refs = _refs_in(src_of[rel])
            used, dynamic = set(), False
            for other, uses in uses_of.items():
                for (f, kind, name, alias) in uses:
                    if f != rel:
                        continue
                    if kind == "star":
                        dynamic = True
                    elif kind == "name":
                        used.add(name)
                    else:      # a module alias: every alias.attr in `other`
                        t = ast.parse(src_of[other])
                        attr_bases = [x for x in ast.walk(t) if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == alias]
                        used |= {x.attr for x in attr_bases}
                        bare = [x for x in ast.walk(t) if isinstance(x, ast.Name) and x.id == alias]
                        if len(bare) > len(attr_bases):
                            dynamic = True      # the module object is passed around: any name may be reached
            reach, stack = set(), list(used)
            while stack:
                n = stack.pop()
                if n in reach:
                    continue
                reach.add(n)
                stack += list(refs.get(n, ()))
            touched = sorted(set(changed) & (reach | {"<module-level>"}))
            if dynamic:
                touched = list(changed)
            res["differs"][rel] = {"changed": changed, "reached": sorted(reach & set(a)), "touched": touched, "dynamic": dynamic, "n_used": len(used)}
            if touched:
                res["touched"] = True
    return res


# ------------------------------------------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("boxlog")
    p.add_argument("--log", required=True)
    p.add_argument("--art", required=True)
    p.add_argument("what", choices=("run", "meta", "done", "table"))
    p.add_argument("arm", nargs="?")
    p.add_argument("seed", nargs="?", type=int)
    p = sub.add_parser("pullcheck")
    p.add_argument("--dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--identity", nargs=2, metavar=("ARM", "SEED"))
    p.add_argument("--expect-sha")
    p.add_argument("--min-age", type=float, default=3.0)
    p = sub.add_parser("kbuild")
    p.add_argument("--art", required=True)
    p.add_argument("--git-dir", required=True)
    p.add_argument("--launch")
    p.add_argument("--compose-pkg")
    p.add_argument("--evidence-commit")
    p.add_argument("--fix-pkg", help="r2 TWO-STEP mode: the fix package (repo path, read at --evidence-commit); --compose-pkg is then the r1 package read at --r1-evidence-commit")
    p.add_argument("--r1-evidence-commit", help="fix-pkg mode: the commit holding the r1 compose package and the golden's reference blob (default the r1 commit e7ea20d1)")
    p = sub.add_parser("tokencommit")
    p.add_argument("--dir", required=True)
    p = sub.add_parser("rolloutcode")
    p.add_argument("--tree", required=True)
    p.add_argument("--git-dir", required=True)
    p.add_argument("--launch", required=True)
    p.add_argument("--loaded", nargs="*", help="trace_imports.py output files: use the EXACT runtime set instead of the static closure")
    p.add_argument("--gate", nargs="*", default=[], help="tree-relative files whose TOUCHED names make the exit code 3 (the model / data modules the checkpoint is built and fed by); other differing files are reported only")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("manifest")
    p.add_argument("--art", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ctrl-runs", required=True)
    p.add_argument("--git-dir")
    p.add_argument("--build-json")
    p.add_argument("--compose-pkg")
    p.add_argument("--evidence-commit")
    p.add_argument("--fix-pkg")
    p.add_argument("--r1-evidence-commit")
    p.add_argument("--n-dump-min", type=int, default=N_EPISODES)
    p.add_argument("--write")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("rcevidence")
    p.add_argument("--reader", required=True)
    p.add_argument("evidence", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "boxlog":
        P = parse_boxlog(open(a.log, "rb").read().decode("utf-8", "replace") if os.path.isfile(a.log) else "", a.art)
        if a.what == "meta":
            m = log_meta(P)
            print(" ".join(f"{k}={v}" for k, v in m.items()))
            return 0
        if a.what == "done":
            return 0 if log_meta(P)["done"] else 1
        if a.what == "table":
            for arm, s in RUNS:
                st, det = run_state(P, arm, s)
                print(f"{run_name(arm, s)} {st} {det}")
            return 0
        if a.arm not in ROLE or a.seed not in (0, 1):
            print("usage: boxlog run ARM SEED (arm dagoaccorc|dagoaccorcrep)")
            return 2
        st, det = run_state(P, a.arm, a.seed)
        if st == "PASS":
            print(f"PASS {det}")
            return 0
        if st == "FAIL":
            print(f"FAIL {det}")
            return 3
        print(f"WAIT {st} {det}".rstrip())
        return 1
    if a.cmd == "pullcheck":
        ident = (a.identity[0], int(a.identity[1])) if a.identity else None
        rc, kind, msg = pullcheck(a.dir, a.tag, ident, a.expect_sha, a.min_age)
        print(f"{kind} {msg}")
        return rc
    if a.cmd == "tokencommit":
        try:
            c = commit_of_dir(a.dir)
        except (OSError, ValueError) as ex:
            print(f"NO-TOKEN {ex}")
            return 1
        if not re.fullmatch(r"[0-9a-f]{40}", c):
            print(f"BAD-COMMIT {c!r}")
            return 3
        if c in refused_commits():
            print(f"REFUSED {c}")
            return 3
        print(c)
        return 0
    if a.cmd == "rolloutcode":
        loaded = None
        if a.loaded:
            loaded = []
            for lf in a.loaded:
                loaded += jload(lf)
        try:
            r = rolloutcode(a.tree, a.git_dir, a.launch, loaded)
        except (RuntimeError, OSError, ValueError, SyntaxError) as ex:
            print(f"ROLLOUT-CODE NOT CHECKED: {ex}")
            return 1
        if a.json:
            print(json.dumps(r, indent=1))
        else:
            n = {k: sum(1 for v in r["files"].values() if v == k) for k in ("IDENTICAL", "DIFFERS", "ABSENT_AT_LAUNCH")}
            print(f"rollout code ({r['basis']}) of {len(r['roots'])} programs: {len(r['files'])} files vs launch {a.launch[:12]}: {n['IDENTICAL']} identical, {n['DIFFERS']} differ, {n['ABSENT_AT_LAUNCH']} absent at launch")
            for rel, d in r["differs"].items():
                print(f"  DIFFERS {rel}: changed top-level names {d['changed']}; the harness reaches {len(d['reached'])} names of it; TOUCHED {d['touched']}{' (module object passed around: conservative)' if d['dynamic'] else ''}")
            gated = gate_failures(r, a.gate)
            if a.gate:
                print(("ROLLOUT-CODE GATE FAILED: " + ", ".join(f"{rel}: {(r['differs'].get(rel) or {}).get('touched') or r['files'].get(rel, 'NOT IN THE CLOSURE')}" for rel in gated)) if gated else
                      f"ROLLOUT-CODE GATE PASS: of the gated files {a.gate} none has a rollout-reachable name that differs at the launch commit")
            else:
                print("ROLLOUT-CODE TOUCHED" if r["touched"] else "ROLLOUT-CODE UNTOUCHED: no name the harness reaches differs at the launch commit")
        if a.gate:
            return 3 if gate_failures(r, a.gate) else 0
        return 3 if r["touched"] else 0
    if a.cmd == "kbuild":
        try:
            print(json.dumps(kbuild(a.art, a.git_dir, a.launch, a.compose_pkg, a.evidence_commit, a.fix_pkg, a.r1_evidence_commit), indent=1))
        except (RuntimeError, OSError, ValueError) as ex:
            print(f"K-BUILD NOT BUILT: {ex}", file=sys.stderr)
            return 3
        return 0
    if a.cmd == "manifest":
        build = jload(a.build_json) if a.build_json else None
        if build is None and a.git_dir:
            try:
                build = kbuild(a.art, a.git_dir, None, a.compose_pkg, a.evidence_commit, a.fix_pkg, a.r1_evidence_commit)
            except (RuntimeError, OSError, ValueError) as ex:
                if not a.dry_run:
                    print(f"MANIFEST REFUSED:\n  - K-BUILD cannot be built: {ex}")
                    return 1
                build = {"NOT_BUILT": str(ex)}
        m = build_manifest(a.art, a.out, a.ctrl_runs, build)
        if a.dry_run:
            print(json.dumps(m, indent=1))
            return 0
        bad = verify_manifest(m, a.n_dump_min)
        if bad:
            print("MANIFEST REFUSED:")
            for b in bad:
                print("  -", b)
            return 1
        os.makedirs(os.path.dirname(a.write) or ".", exist_ok=True)
        with open(a.write, "w", encoding="utf-8", newline="\n") as f:
            json.dump(m, f, indent=1)
        print(f"wrote {a.write} ({sum(len(v) for v in m['arms'].values())} arm-seed entries; K-BUILD {'recorded' if build else 'ABSENT'})")
        return 0
    if a.cmd == "rcevidence":
        me = sha256_file(a.reader)
        bad = []
        for p_ in a.evidence:
            try:
                d = jload(p_)
            except (OSError, ValueError) as ex:
                bad.append(f"{p_}: unreadable ({ex})")
                continue
            if d.get("reader_sha256") != me or d.get("differences") != 0 or d.get("PASS") is not True:
                bad.append(f"{p_}: reader_sha256 {str(d.get('reader_sha256'))[:12]} (this reader {me[:12]}), differences {d.get('differences')!r}, PASS {d.get('PASS')!r}")
        print("RC evidence: " + ("OK, " + str(len(a.evidence)) + " files carry this reader's sha256 and 0 differences" if not bad else "; ".join(bad)))
        return 0 if not bad else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
