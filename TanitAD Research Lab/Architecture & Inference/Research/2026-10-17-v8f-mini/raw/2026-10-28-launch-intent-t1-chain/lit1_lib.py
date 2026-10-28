#!/usr/bin/env python3
"""lit1_lib.py -- the helpers of lit1_chain.sh, the DEV-BOX CPU T1-kin chain of PREREG_V8F_LAUNCH_INTENT (the four-run redesign, MM_REDESIGN_2026-10-28 section 3).
CPU only, no torch here, read-only on every pull (it HASHES files).  It never contacts box 2 and starts no process except `git` (read-only) and, in `parsecheck` / `cfgprobe`, a python child.
Nothing here judges the science: that is the pre-registered reader's.

The four runs (box 2's chain_v8f_launchintent.sh, in its order):  A_0 = M2b_s0_launchintent, A_1 = M2b_s1_launchintent, A_rep,0 = M2b_s0_launchintentrep (trainer --seed 10), A_rep,1 = M2b_s1_launchintentrep (--seed 11).
Pull layout (its `runone`):  <ART>/M2b_s<S>_<arm>/{ckpt_last.pt, eval_M2b-<arm>_s<S>.{json,npz}, selfcheck.json, summary.json, argv.json, train_log.jsonl, launch_gate_token.json, SHA256SUMS.txt}.

    boxlog    --log L --art ART  run ARM SEED | next [--complete R,R] | meta | done | table
    pullcheck --dir D --arm ARM --seed S [--expect-sha X] [--min-age N]        COMPLETE BY CONTENT + the run record re-verified from the pulled files
    commits   --art ART                                                          the ONE launch commit / reader sha / prereg sha the pulled tokens bind
    render    KIND --ckpt C --tag T --tree TREE [--out O ...]                    one argv token per line (KIND = causal | logged | t0fresh | reader)
    guards    [--ram-min-mb N] [--disk PATH --disk-min-gb G]                     RAM and disk guards
    tree*, kbuild, manifest*, parsecheck, cfgprobe, rolloutcode                  see lit1_tree.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

sys.dont_write_bytecode = True

AR = "E:/Projects/TanitAD-artifacts"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
SCRATCH = "F:/TanitAD-scratch/launch_intent_t1"            # the only place a test world may live (LIT1_WORLD must resolve inside it)
BASE_TREE = f"{AR}/dagt1_tree_b43c7a9a"                  # the pinned tool tree that rolled CTRL (DAG-OACC) and REP (DAG-ORC r2) out: READ ONLY
BASE_DIGEST = "6a4b47c8acfadf07ffc82ab174cccce78aa0404f6a4f89231f331013bbcda3d9"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

# ------------------------------------------------------------------------------------------------------------------------------ the registered numbers (PREREG_V8F_LAUNCH_INTENT draft v2, MM_REDESIGN section 3)
RUNS = [("launchintent", 0), ("launchintent", 1), ("launchintentrep", 0), ("launchintentrep", 1)]      # the box-2 chain's order; the chain processes whichever lands first (pull order)
LABEL = {("launchintent", 0): "A_0", ("launchintent", 1): "A_1", ("launchintentrep", 0): "A_rep,0", ("launchintentrep", 1): "A_rep,1"}
CONTROL = {("launchintent", 0): "DAG-OACC_s0", ("launchintent", 1): "DAG-OACC_s1", ("launchintentrep", 0): "REP_s0", ("launchintentrep", 1): "REP_s1"}
ROLE = {"launchintent": "A", "launchintentrep": "Arep"}
TAGSTEM = {"launchintent": "LI", "launchintentrep": "LIREP"}
SEED_OFF = {"launchintent": 0, "launchintentrep": 10}      # the trainer's --seed = S for A_s, 10 + S for A_rep,s
LI_W, LI_W_STR = 1.58926, "1.58926"
STEPS, SWAP_P, W_ACC = 3000, 0.1, 49.6969
SHARE_TARGET, SHARE_TOL = 1.0 / 3.0, 0.02
F_MEASURED, MASK_TOL = 49618 / 306566, 0.02
LI_MASK = {"v0_max": 6.0, "dv": 1.0, "dt": 0.2, "knots": [1, 12]}
LAUNCH_KEYS = ("launch_n", "launch_eligible", "launch_rows", "launch_intent_w", "launch_weight_share")
WINDOW_SHA = "e144adcd9b6273f2b44ad7dfa162931b90e159f7b3911e0893b25777b2ae8168"
EXCLUDED_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
M2B_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
BANK1_SHA = {0: "197a68bb540be5a7d157942e9157e1c8e60ca60d82b1ca4fb4fcb68edccfaa8f", 1: "36c7aea964dafe9423253aa479c2fc6cdfc93848395b593315cba812f5f04d4d"}
OACC_CKPT_SHA = {0: "67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f", 1: "90b01474819082a6d32a9000d63a4954e0074916140abdbbcb67b353f7ac3333"}
REP_CKPT_SHA = {0: "b03fd6efebec285f747673fa90d093cb9202a649d560ba8f5c016475f1fb4bca", 1: "77853e564f52ff029cfc241042b733d75dba8191a99a7be0fc59778d511aeca2"}
LAUNCH_COMMIT = "b09ac89fc9d6e81baf3977954ae638919b979e71"      # the landed flag (coordinator message 2026-10-28); a token may bind a DESCENDANT, never a commit whose trainer / model bytes differ
LAUNCH_TRAINER_SHA = "0d708b3cb841bc1f1832b09fe61e3d8421a4d3f59769f4435c0b22bb77e268ce"      # stack/scripts/v8f_mini_train.py at LAUNCH_COMMIT
LAUNCH_MODEL_SHA = "b337ccecd3abfa4250a77e233502d9d3464a280dbafdc86b1e94ecbeef682721"        # stack/tanitad/models/v8f_mini.py at LAUNCH_COMMIT
TRAINER_PATH, MODEL_PATH, DATA_PATH = "stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py"
N_WINDOWS, N_EPISODES = 2399, 141
DISABLED: set = set()      # TEST HOOK (selftest mutation pass only): clause ids whose check is skipped.  Never set by the CLI.


def on(cid: str) -> bool:
    return cid not in DISABLED


# ------------------------------------------------------------------------------------------------------------------------------ small utilities
def run_name(arm: str, seed: int) -> str:
    return f"M2b_s{seed}_{arm}"


def eval_tag(arm: str, seed: int) -> str:
    return f"M2b-{arm}_s{seed}"


def t1_tag(arm: str, seed: int, limit: str) -> str:
    return f"{TAGSTEM[arm]}_s{seed}_m10_{limit}_heads"


def t0f_tag(arm: str, seed: int) -> str:
    return f"{eval_tag(arm, seed)}-fresh"


def sums_names(tag: str) -> list:
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


def fmt(args: list) -> str:
    """The shell-quoted spelling the DAG-ORC chain logged (a token with a character outside [A-Za-z0-9_./:=,@%+-] is single-quoted)."""
    o = []
    for a in args:
        o.append(f"'{a}'" if re.search(r"[^A-Za-z0-9_./:=,@%+-]", a) else a)
    return " ".join(o)


def world_ok() -> bool:
    """A test world is honoured only when LIT1_WORLD names a directory INSIDE the scratch root (real mode never sets it)."""
    w = os.environ.get("LIT1_WORLD", "")
    if not w:
        return False
    r = os.path.realpath(w).replace("\\", "/").lower()
    s = os.path.realpath(SCRATCH).replace("\\", "/").lower()
    return r.startswith(s + "/") and os.path.isdir(w)


def share_of(n: int, e: int, w: float):
    """The trainer's own formula (launch_weight_share): launch samples carry 1 + W, the other GT-valid samples 1."""
    if e <= 0:
        return None
    return ((1.0 + w) * n) / ((1.0 + w) * n + (e - n))


# ------------------------------------------------------------------------------------------------------------------------------ the box-2 chain log
_L = r"A_0|A_1|A_rep,0|A_rep,1"
PULLED = re.compile(r"^(?:\d\d-\d\d \d\d:\d\d )?(M2b_s([01])_(launchintent(?:rep)?)) pulled to (\S+) \(checkpoint sha256 ([0-9a-f]{64})\) \[(" + _L + r"), control ([^\]\s]+)\]\s*$")
REC = re.compile(r"^LAUNCH-INTENT run record: (PASS|\[.*?\]) \| (" + _L + r") \| (launchintent(?:rep)?) s([01]) \|")
KSH = re.compile(r"^LAUNCH-INTENT K-SHARE: (PASS|FAIL) \| (" + _L + r") \|")
CFAIL = re.compile(r"^ZZLAUNCHINTENT-CONTROL-FAIL (M2b_s[01]_launchintent(?:rep)?)")
DONE_RE = re.compile(r"^ZZV8FMINI-DONE \S+ \S+ tip ([0-9a-f]{40}): PREREG_V8F_LAUNCH_INTENT arm A \(launch-intent-w 1\.58926\): 4 runs")
SUMS_LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")
REFUSED_LOGS = ("chain_v8f_dagoaccorc.out", "chain_v8f_dagoaccorc_r2.out", "chain_v8f_dagoacc.out", "chain_v8f_dag2.out")      # sibling chains' logs: never this chain's


def _blank(arm, seed) -> dict:
    return {"line": None, "arm": arm, "seed": seed, "path": None, "sha": None, "label": None, "control": None, "rec": None, "ksh": None, "cfail": None, "problems": []}


def parse_boxlog(text: str, art: str) -> dict:
    """Section-wise parse of chain_v8f_launchintent.out.  A run's section starts at its '<run> pulled to <path> (checkpoint sha256 <64 hex>) [<label>, control <ctl>]' line (the LAST such line wins: a re-pull restarts it)
    and holds the run-record line and the K-SHARE line that follow it.  Facts only; run_state() decides."""
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    sec: dict = {}
    cur = None
    stops, dones, floors, problems = [], [], [], []
    for i, ln in enumerate(lines):
        m = PULLED.match(ln)
        if m:
            g = m.group(1)
            cur = g
            sec[g] = _blank(m.group(3), int(m.group(2)))
            sec[g].update({"line": i, "path": npath(m.group(4)), "sha": m.group(5), "label": m.group(6), "control": m.group(7)})
            continue
        m = REC.match(ln)
        if m:
            g = run_name(m.group(3), int(m.group(4)))
            if cur is None or g != cur:
                problems.append(f"line {i + 1}: a run-record line for {g} outside {g}'s section (the section is {cur})")
                sec.setdefault(g, _blank(m.group(3), int(m.group(4))))["problems"].append("run-record line outside its section")
            else:
                sec[cur]["rec"] = (m.group(1) == "PASS")
                if m.group(1) != "PASS":
                    sec[cur]["problems"].append(f"run record not PASS: {ln[:200]}")
                if m.group(2) != LABEL[(m.group(3), int(m.group(4)))]:
                    sec[cur]["problems"].append(f"run-record label {m.group(2)} != {LABEL[(m.group(3), int(m.group(4)))]}")
            continue
        m = KSH.match(ln)
        if m:
            if cur is None or sec[cur]["label"] != m.group(2):
                problems.append(f"line {i + 1}: a K-SHARE line for {m.group(2)} outside that run's section (the section is {cur})")
                if cur is not None:
                    sec[cur]["problems"].append("K-SHARE line of another run inside this section")
            else:
                sec[cur]["ksh"] = (m.group(1) == "PASS")
                if m.group(1) != "PASS":
                    sec[cur]["problems"].append(f"K-SHARE FAIL: {ln[:200]}")
            continue
        m = CFAIL.match(ln)
        if m:
            sec.setdefault(m.group(1), _blank(None, None))["cfail"] = ln[:200]
            continue
        if ln.startswith("ZZV8FMINI-STOP"):
            stops.append((i, ln[:240]))
        elif ln.startswith("ZZV8FMINI-DONE"):
            dones.append((i, ln[:300]))
        elif ln.startswith("ZZDISK-FLOOR-STOP"):
            floors.append((i, ln[:240]))
    return {"sec": sec, "stops": stops, "dones": dones, "floors": floors, "problems": problems, "nlines": len(lines), "art": npath(art)}


def run_state(P: dict, arm: str, seed: int) -> tuple:
    """-> (NOT-PULLED | AWAITING-CONTROLS | PASS | FAIL, detail).  PASS needs the pulled line AND a PASSing run record AND a PASSing K-SHARE line, in that run's section, with no STOP after the section."""
    g = run_name(arm, seed)
    s = P["sec"].get(g)
    if not s:
        return "NOT-PULLED", ""
    bad = list(s["problems"])
    if s["cfail"] and on("log-cfail"):
        bad.append(f"ZZLAUNCHINTENT-CONTROL-FAIL: {s['cfail']}")
    if s["line"] is None:
        return "FAIL", "; ".join(bad) or f"{g}: control lines without a pulled line"
    if s["path"].lower() != npath(f"{P['art']}/{g}").lower() and on("log-path"):
        bad.append(f"the box-2 chain pulled {g} to {s['path']}, this chain reads {P['art']}/{g}")
    if s["label"] != LABEL[(arm, seed)] and on("log-label"):
        bad.append(f"the pull line's label {s['label']} is not {LABEL[(arm, seed)]} (arm / seed confusion)")
    if s["control"] != CONTROL[(arm, seed)] and on("log-label"):
        bad.append(f"the pull line's control {s['control']} is not {CONTROL[(arm, seed)]}")
    if bad:
        return "FAIL", "; ".join(bad)[:600]
    if s["rec"] and s["ksh"]:
        return "PASS", s["sha"]
    later = [t for (i, t) in P["stops"] + P["floors"] if i > s["line"]]
    if later and on("log-stop-after"):
        return "FAIL", f"the box-2 chain STOPPED after pulling {g} and before its run record and K-SHARE both printed PASS: {later[0]}"
    return "AWAITING-CONTROLS", f"run record {'PASS' if s['rec'] else 'not yet'}, K-SHARE {'PASS' if s['ksh'] else 'not yet'}"


def log_meta(P: dict) -> dict:
    pulled = [run_name(a, s) for (a, s) in RUNS if run_name(a, s) in P["sec"] and P["sec"][run_name(a, s)]["line"] is not None]
    last = max([P["sec"][g]["line"] for g in pulled] or [-1])
    return {"stops": len(P["stops"]), "pending": len(RUNS) - len(pulled), "pulled": len(pulled),
            "done": int(len(pulled) == len(RUNS) and any(i > last and DONE_RE.match(t) for (i, t) in P["dones"]))}


def pull_order(P: dict) -> list:
    """The runs that have a pull line, in the order the pull lines appear (the order the pulls landed)."""
    have = [(P["sec"][run_name(a, s)]["line"], a, s) for (a, s) in RUNS if run_name(a, s) in P["sec"] and P["sec"][run_name(a, s)]["line"] is not None]
    return [(a, s) for (_l, a, s) in sorted(have)]


def next_ready(P: dict, complete: set) -> tuple:
    """-> ('RUN', arm, seed, sha) for the earliest-pulled run in PASS state that is not complete | ('FAIL', why) when ANY pulled, incomplete run is in FAIL | ('WAIT', summary)."""
    waits = []
    for (a, s) in pull_order(P):
        if run_name(a, s) in complete:
            continue
        st, d = run_state(P, a, s)
        if st == "FAIL":
            return ("FAIL", f"{run_name(a, s)}: {d}")
        if st == "PASS":
            return ("RUN", a, s, d)
        waits.append(f"{run_name(a, s)} {st}")
    for g, sc in P["sec"].items():      # control-fail lines of a run with no pull line (box 2 printed the failure instead of a pull)
        if sc["cfail"] and sc["line"] is None and g not in complete and on("log-cfail"):
            return ("FAIL", f"{g}: ZZLAUNCHINTENT-CONTROL-FAIL: {sc['cfail']}")
    return ("WAIT", "; ".join(waits) or "no pull line yet")


# ------------------------------------------------------------------------------------------------------------------------------ the pull, by content, and the run record re-verified from the pulled files
def _ids(bad: list, cid: str, msg: str) -> None:
    bad.append(f"[{cid}] {msg}")


def runrecord_problems(d: str, arm: str, seed: int, ck: str, sums: dict | None = None) -> list:
    """Is this directory the run it is named after AND does its own record show the lever reached training?  Every clause has an id (a test deletes one and its scenario must then pass).
    Mirrors box 2's run-record + K-SHARE / K-MASK checks on the PULLED files, plus the token binds, so a record that only the LOG vouches for is never the whole story."""
    bad: list = []
    g = run_name(arm, seed)
    tseed = seed + SEED_OFF[arm]
    try:
        sm, ar, tk, sc = (jload(os.path.join(d, f)) for f in ("summary.json", "argv.json", "launch_gate_token.json", "selfcheck.json"))
    except (OSError, ValueError) as ex:
        return [f"[records] a record does not parse: {ex}"]
    argv = [str(t) for t in ar.get("argv", [])]
    fl = flags_of(argv)
    for cid, name, want in (("argv-steps", "--steps", STEPS), ("argv-dagger-p", "--dagger-p", SWAP_P), ("argv-w-acc", "--w-acc", W_ACC)):
        v = fnum(one(fl, name))
        if on(cid) and (v is None or abs(v - want) > 1e-9):
            _ids(bad, cid, f"argv {name} {one(fl, name)!r}, registered {want}")
    v = one(fl, "--seed")
    if on("argv-seed") and (v is None or fnum(v) != float(tseed)):
        _ids(bad, "argv-seed", f"argv --seed {v!r}, expected the trainer seed {tseed} ({LABEL[(arm, seed)]})")
    if on("argv-dagger-seed") and fnum(one(fl, "--dagger-seed")) != float(seed):
        _ids(bad, "argv-dagger-seed", f"argv --dagger-seed {one(fl, '--dagger-seed')!r}, expected {seed}")
    if on("argv-bank") and (one(fl, "--dagger-bank-sha256") or "").lower() != BANK1_SHA[seed]:
        _ids(bad, "argv-bank", "argv --dagger-bank-sha256 is not DAG-OACC's iteration-1 bank")
    if on("argv-init") and not npath(one(fl, "--init-ckpt") or "").endswith(f"/M2b_s{seed}/ckpt_last.pt"):
        _ids(bad, "argv-init", f"argv --init-ckpt {one(fl, '--init-ckpt')!r} is not .../M2b_s{seed}/ckpt_last.pt")
    if on("argv-out") and not npath(one(fl, "--out") or "").endswith("/" + g):
        _ids(bad, "argv-out", f"argv --out {one(fl, '--out')!r} does not end in {g}")
    occ = fl.get("--launch-intent-w", [])
    if on("argv-lever") and (not occ or any(o != [LI_W_STR] for o in occ)):
        _ids(bad, "argv-lever", f"argv --launch-intent-w {occ!r}, registered exactly [['{LI_W_STR}']]")
    if on("argv-lever-once") and len(occ) != 1:
        _ids(bad, "argv-lever-once", f"argv names --launch-intent-w {len(occ)} times, must be exactly once")
    ep = fnum(one(fl, "--ego-perturb-p"))
    if on("argv-no-ego") and ((ep is not None and ep != 0.0) or "--ego-perturb-with-dagger" in argv):
        _ids(bad, "argv-no-ego", "argv names an ego perturbation (--ego-perturb-p > 0 or --ego-perturb-with-dagger): the arm moves ONE lever")
    sha16 = hashlib.sha256(" ".join(argv).encode()).hexdigest()[:16]
    if on("argv-sha"):
        for name, vv in (("argv.json sha", ar.get("sha")), ("summary.json argv_sha", sm.get("argv_sha")), ("selfcheck.json argv_sha", sc.get("argv_sha"))):
            if vv != sha16:
                _ids(bad, "argv-sha", f"{name} {vv} != the argv's {sha16}")
    if on("sum-done") and sm.get("done") is not True:
        _ids(bad, "sum-done", "summary.json done is not true")
    if on("sum-steps") and sm.get("steps") != STEPS:
        _ids(bad, "sum-steps", f"summary.json steps {sm.get('steps')} != {STEPS}")
    if on("sum-seed") and (sm.get("arm"), sm.get("seed")) != ("M2b", tseed):
        _ids(bad, "sum-seed", f"summary.json arm / seed {(sm.get('arm'), sm.get('seed'))} != ('M2b', {tseed})")
    if on("sc-pass") and sc.get("declared_vs_built") != "PASS":
        _ids(bad, "sc-pass", "the run's own selfcheck.json declared_vs_built is not PASS")
    lv = sm.get("levers") or {}
    for what, rec in (("summary.json", sm), ("selfcheck.json", sc)):
        l2 = rec.get("levers") if isinstance(rec.get("levers"), dict) else {}
        if on("sum-lever") and (type(l2.get("launch_intent_w")) is not float or l2.get("launch_intent_w") != LI_W):
            _ids(bad, "sum-lever", f"{what} levers.launch_intent_w {l2.get('launch_intent_w', 'ABSENT')!r} != {LI_W!r} (a run built before the lever, or a wrong dose)")
        li = l2.get("launch_intent")
        if on("sum-lever-mask") and (not isinstance(li, dict) or any(li.get(k) != w for k, w in LI_MASK.items()) or li.get("launch_intent_w") != LI_W):
            _ids(bad, "sum-lever-mask", f"{what} levers.launch_intent {li!r} is not the registered built mask {LI_MASK} at W {LI_W}")
    miss = [k for k in LAUNCH_KEYS if k not in sm]
    if on("sum-keys") and miss:
        _ids(bad, "sum-keys", f"summary.json lacks {miss}: only the new build writes them (a stale or resurrected run is REFUSED)")
    elif not miss:
        n, e, r = sm["launch_n"], sm["launch_eligible"], sm["launch_rows"]
        if not all(type(x) is int for x in (n, e, r)) or not (e > 0 and 0 < n <= e):
            _ids(bad, "sum-share", f"launch_n {n!r} / launch_eligible {e!r} / launch_rows {r!r}: not a count triple with 0 < n <= eligible")
        else:
            if on("sum-top-w") and sm.get("launch_intent_w") != LI_W:
                _ids(bad, "sum-top-w", f"summary.json launch_intent_w {sm.get('launch_intent_w')!r} != {LI_W!r}")
            sh = share_of(n, e, LI_W)
            lo, hi = SHARE_TARGET - SHARE_TOL, SHARE_TARGET + SHARE_TOL
            if on("sum-share") and not (lo - 1e-12 <= sh <= hi + 1e-12):
                _ids(bad, "sum-share", f"K-SHARE: launch_weight_share recomputed {sh:.6f} from launch_n {n}, launch_eligible {e}, W {LI_W} is outside 1/3 +- {SHARE_TOL}")
            rec = sm.get("launch_weight_share")
            if on("sum-share-rec") and (not (isinstance(rec, float) and math.isfinite(rec)) or abs(rec - sh) > 1e-9):
                _ids(bad, "sum-share-rec", f"the record's own launch_weight_share {rec!r} != the recomputed {sh!r}")
            if on("sum-rows") and r != e:
                _ids(bad, "sum-rows", f"launch_rows {r} != launch_eligible {e}: a row seen by the loop had no GT-valid along-track label")
            f = n / e
            if on("sum-mask") and not (F_MEASURED - MASK_TOL <= f <= F_MEASURED + MASK_TOL):
                _ids(bad, "sum-mask", f"K-MASK: launch_n / launch_eligible = {f:.6f} outside f = {F_MEASURED:.6f} +- {MASK_TOL}")
    if on("sum-window") and sm.get("train_window_list_sha256") != WINDOW_SHA:
        _ids(bad, "sum-window", f"K-MASK: train_window_list_sha256 {str(sm.get('train_window_list_sha256'))[:12]} != DAG-OACC's {WINDOW_SHA[:12]}")
    ic = lv.get("init_ckpt") or {}
    if on("sum-init") and ic.get("sha256") != M2B_SHA[seed]:
        _ids(bad, "sum-init", f"summary.json levers.init_ckpt.sha256 {str(ic.get('sha256'))[:16]} != M2b_s{seed}'s {M2B_SHA[seed][:16]} (K-INIT)")
    if on("sum-init-applied") and (ic.get("applied") is not True or ic.get("n_mismatch_after_load") != 0):
        _ids(bad, "sum-init-applied", f"init_ckpt not applied / read back: {ic.get('applied')} / {ic.get('n_mismatch_after_load')}")
    if on("sum-bank") and lv.get("dagger_bank_sha256") != BANK1_SHA[seed]:
        _ids(bad, "sum-bank", f"built dagger_bank_sha256 {str(lv.get('dagger_bank_sha256'))[:12]} != DAG-OACC's bank {BANK1_SHA[seed][:12]}")
    if on("sum-dagger-p") and lv.get("dagger_p") != SWAP_P:
        _ids(bad, "sum-dagger-p", f"built dagger_p {lv.get('dagger_p')!r} != {SWAP_P}")
    if on("sum-w-acc") and lv.get("w_acc") != W_ACC:
        _ids(bad, "sum-w-acc", f"built w_acc {lv.get('w_acc')!r} != {W_ACC}")
    epl = lv.get("ego_perturb") or {}
    if on("sum-ego-zero") and (epl.get("ego_perturb_p") != 0.0 or lv.get("ego_perturb_with_dagger") not in (False, None)):
        _ids(bad, "sum-ego-zero", f"built ego perturbation is not off: p {epl.get('ego_perturb_p')!r}, with_dagger {lv.get('ego_perturb_with_dagger')!r}")
    if on("tok-kind") and (tk.get("kind") != "launch-gate-token" or tk.get("verdict") != "PASS" or tk.get("prereg") != "PREREG_V8F_LAUNCH_INTENT"):
        _ids(bad, "tok-kind", f"launch-gate token: kind {tk.get('kind')!r} verdict {tk.get('verdict')!r} prereg {tk.get('prereg')!r}")
    if on("tok-run") and tk.get("run") != g:
        _ids(bad, "tok-run", f"launch-gate token names run {tk.get('run')!r}, expected {g}")
    bd = tk.get("binding") or {}
    if on("tok-commit") and not re.fullmatch(r"[0-9a-f]{40}", str(bd.get("commit", ""))):
        _ids(bad, "tok-commit", "launch-gate token binds no 40-hex commit")
    if on("tok-shas") and not all(re.fullmatch(r"[0-9a-f]{64}", str(tk.get(k, ""))) for k in ("prereg_sha256", "reader_sha256")):
        _ids(bad, "tok-shas", "launch-gate token carries no 64-hex prereg_sha256 / reader_sha256")
    kf = tk.get("key_files_sha256") or {}
    if on("tok-trainer") and kf.get(TRAINER_PATH) != LAUNCH_TRAINER_SHA:
        _ids(bad, "tok-trainer", f"token key_files_sha256 trainer {str(kf.get(TRAINER_PATH))[:12]} != the pinned launch trainer {LAUNCH_TRAINER_SHA[:12]}")
    if on("tok-model") and kf.get(MODEL_PATH) != LAUNCH_MODEL_SHA:
        _ids(bad, "tok-model", f"token key_files_sha256 model {str(kf.get(MODEL_PATH))[:12]} != the pinned launch model {LAUNCH_MODEL_SHA[:12]}")
    if on("tok-init") and tk.get("init_ckpt_sha256") != M2B_SHA[seed]:
        _ids(bad, "tok-init", f"token init_ckpt_sha256 {str(tk.get('init_ckpt_sha256'))[:12]} != M2b_s{seed}'s")
    if on("tok-bank") and tk.get("bank_sha256") != BANK1_SHA[seed]:
        _ids(bad, "tok-bank", f"token bank_sha256 {str(tk.get('bank_sha256'))[:12]} != DAG-OACC's bank")
    if on("tok-wacc") and (tk.get("w_acc") != W_ACC or tk.get("dagger_p") != SWAP_P):
        _ids(bad, "tok-wacc", f"token w_acc {tk.get('w_acc')!r} / dagger_p {tk.get('dagger_p')!r} != {W_ACC} / {SWAP_P}")
    if on("tok-excluded") and tk.get("excluded_sha256") != EXCLUDED_SHA:
        _ids(bad, "tok-excluded", f"token excluded_sha256 {str(tk.get('excluded_sha256'))[:12]} != DAG-OACC's {EXCLUDED_SHA[:12]}")
    if on("tok-arm") and tk.get("arm") is not None and tk.get("arm") != arm:
        _ids(bad, "tok-arm", f"token arm {tk.get('arm')!r} != {arm}")
    if on("ck-not-ctrl") and ck in (OACC_CKPT_SHA[seed], M2B_SHA[seed], REP_CKPT_SHA[seed]):
        _ids(bad, "ck-not-ctrl", "its checkpoint is DAG-OACC's, M2b's or REP's: nothing was trained")
    return bad


def pullcheck(d: str, arm: str, seed: int, expect_sha: str | None = None, min_age: float = 3.0, now: float | None = None) -> tuple:
    """-> (rc, kind, msg).  COMPLETE iff by CONTENT: SHA256SUMS.txt exists, aged, newline-terminated, lists exactly the 8 names once; every file exists, is non-empty and re-hashes to the listed value; summary.json says done.
    WRONG when those hold but the checkpoint sha256 differs from ``expect_sha`` (the box-2 log's) or the run's own records say it is not the run asked for."""
    d = d.rstrip("/\\")
    tag = eval_tag(arm, seed)
    if not os.path.isdir(d):
        return 1, "INCOMPLETE", f"{d} is absent"
    sp = os.path.join(d, "SHA256SUMS.txt")
    if not os.path.isfile(sp):
        return 1, "INCOMPLETE", "no SHA256SUMS.txt yet"
    if (now or time.time()) - os.path.getmtime(sp) < min_age and on("pull-age"):
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
    if sorted(listed) != sorted(want) and on("pull-names"):
        return 1, "INCOMPLETE", f"SHA256SUMS.txt lists {sorted(set(listed) ^ set(want))} differently from the 8 expected names"
    for n in want:
        p = os.path.join(d, n)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return 1, "INCOMPLETE", f"{n} is missing or empty"
        if on("pull-rehash") and sha256_file(p) != listed.get(n):
            return 1, "INCOMPLETE", f"{n} does not hash to its listed sha256 (a pull still filling, or corrupt)"
    ck = listed.get("ckpt_last.pt", "")
    try:
        sm = jload(os.path.join(d, "summary.json"))
        jload(os.path.join(d, f"eval_{tag}.json"))
    except (OSError, ValueError) as ex:
        return 1, "INCOMPLETE", f"a record does not parse: {ex}"
    if on("pull-done") and sm.get("done") is not True:
        return 1, "INCOMPLETE", "summary.json does not say done: true"
    if expect_sha and ck != expect_sha and on("pull-3way"):
        return 3, "WRONG", f"the pulled ckpt_last.pt hashes to {ck[:16]}, the box-2 log printed {expect_sha[:16]}"
    if on("pull-3way") and expect_sha and sha256_file(os.path.join(d, "ckpt_last.pt")) != expect_sha:
        return 3, "WRONG", "ckpt_last.pt re-hashed from the file is not the box-2 log's sha256"
    bad = runrecord_problems(d, arm, seed, ck)
    if bad:
        return 3, "WRONG", "; ".join(bad)[:900]
    return 0, "COMPLETE", f"{ck} 8 names re-hashed, summary done, run record re-verified ({LABEL[(arm, seed)]}: argv / lever / K-SHARE / K-MASK / K-INIT / bank / token)"


def commits_of(art: str, runs=None) -> dict:
    """The ONE launch commit, reader sha and prereg sha the pulled tokens bind (every token present must agree; at least one must exist)."""
    com, rd, pr = set(), set(), set()
    n = 0
    for (a, s) in (runs or RUNS):
        p = f"{art}/{run_name(a, s)}/launch_gate_token.json"
        if not os.path.isfile(p):
            continue
        tk = jload(p)
        com.add(str((tk.get("binding") or {}).get("commit", "")))
        rd.add(str(tk.get("reader_sha256", "")))
        pr.add(str(tk.get("prereg_sha256", "")))
        n += 1
    out = {"n_tokens": n, "commits": sorted(com), "reader_sha256": sorted(rd), "prereg_sha256": sorted(pr)}
    out["ok"] = bool(n and len(com) == 1 and len(rd) == 1 and len(pr) == 1 and re.fullmatch(r"[0-9a-f]{40}", next(iter(com))) and re.fullmatch(r"[0-9a-f]{64}", next(iter(rd))) and re.fullmatch(r"[0-9a-f]{64}", next(iter(pr))))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the commands (templates in argv/, one token per line)
SF = f"{AR}/sfresh_inputs"


def tmpl_tokens(kind: str) -> list:
    path = os.path.join(HERE, "argv", {"causal": "rollout_causal.tmpl", "logged": "rollout_logged.tmpl", "t0fresh": "t0fresh.tmpl", "reader": "reader.tmpl"}[kind])
    toks = []
    for line in open(path, encoding="utf-8").read().split("\n"):
        line = line.rstrip("\r")
        if line and not line.startswith("#"):
            toks.append(line)
    return toks


def render(kind: str, ckpt: str = "", tag: str = "", tree: str = BASE_TREE, out: str = "", **kw) -> list:
    """The argv of one step.  Inputs (grid / join / kin) are the PINNED tree's (md5-checked) for every lane, as in the banked commands; only the tool script comes from ``tree``."""
    raw = f"{BASE_TREE}/{PK}/raw"
    val = {"PY": PY, "TREE": npath(tree), "CKPT": ckpt, "TAG": tag, "TRACKS": f"{SF}/inputs/native_tracks.pt", "CACHE": f"{SF}/cache", "GRID": f"{raw}/eval_grid_windows.json", "JOIN": f"{raw}/join_eval141.npz",
           "KIN": f"{raw}/eval141_kin.npz", "OUT": out, "V9": f"{SF}/inputs/v9", "P10": f"{SF}/inputs/p10/p10_prior.pt", "T0DIR": f"{out}/t0fresh",
           "READER": kw.get("reader", ""), "PREREG": kw.get("prereg", ""), "PREREG_SHA": kw.get("prereg_sha", ""), "ORCDIR": kw.get("orc_dir", f"{AR}/dagorc_rebase/read")}
    res = []
    for t in tmpl_tokens(kind):
        for k, v in val.items():
            t = t.replace(f"@{k}@", v)
        if re.search(r"@[A-Z0-9_]+@", t):
            raise SystemExit(f"template {kind}: unresolved placeholder in {t!r}")
        res.append(t)
    return res


def cmd_sha(argv: list) -> str:
    return hashlib.sha256(b"".join(a.encode() + b"\0" for a in argv)).hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ guards
def ram_avail_mb() -> int:
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("ml", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = MS()
    m.l = ctypes.sizeof(MS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(m.ap // (1 << 20))


def guards(ram_min_mb: int, disk: str | None, disk_min_gb: float) -> tuple:
    """-> (rc, text).  rc 0 = every guard holds, 1 = LOW (the chain waits, never deletes).  The test hooks (env LIT1_TEST_RAM_MB / LIT1_TEST_DISK_GB, or the files <world>/hook_ram_mb.txt / hook_disk_gb.txt, which win)
    are honoured ONLY inside a valid test world."""
    import shutil
    ram, free = None, None
    low = []

    def hook(env_name: str, file_name: str):
        if not world_ok():
            return None
        fp = os.path.join(os.environ["LIT1_WORLD"], file_name)
        if os.path.isfile(fp):
            return open(fp).read().strip()
        return os.environ.get(env_name) or None
    if ram_min_mb:
        ram = ram_avail_mb()
        if hook("LIT1_TEST_RAM_MB", "hook_ram_mb.txt"):
            ram = int(hook("LIT1_TEST_RAM_MB", "hook_ram_mb.txt"))
        if ram < ram_min_mb and on("guard-ram"):
            low.append(f"RAM available {ram} MB < {ram_min_mb} MB")
    if disk:
        free = shutil.disk_usage(disk).free / (1 << 30)
        if hook("LIT1_TEST_DISK_GB", "hook_disk_gb.txt"):
            free = float(hook("LIT1_TEST_DISK_GB", "hook_disk_gb.txt"))
        if free < disk_min_gb and on("guard-disk"):
            low.append(f"{disk} free {free:.1f} GB < {disk_min_gb} GB")
    txt = f"ram={ram}MB disk_free={None if free is None else round(free, 1)}GB"
    return (1, "LOW " + "; ".join(low) + " | " + txt) if low else (0, "OK " + txt)


# ------------------------------------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(newline="\n", encoding="utf-8")
        except Exception:      # noqa: BLE001
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("treeplan", "treedigest", "treebuild", "treecheck", "modelproof", "cfgprobe", "rolloutcode", "kbuild", "manifest", "verify", "parsecheck"):
        sys.path.insert(0, HERE)
        import lit1_tree
        return lit1_tree.main(argv)
    ap = argparse.ArgumentParser(prog="lit1_lib.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("boxlog")
    b.add_argument("--log", required=True)
    b.add_argument("--art", required=True)
    b.add_argument("what", choices=["run", "next", "meta", "done", "table"])
    b.add_argument("arm", nargs="?")
    b.add_argument("seed", nargs="?")
    b.add_argument("--complete", default="")
    p = sub.add_parser("pullcheck")
    p.add_argument("--dir", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--expect-sha")
    p.add_argument("--min-age", type=float, default=3.0)
    c = sub.add_parser("commits")
    c.add_argument("--art", required=True)
    r = sub.add_parser("render")
    r.add_argument("kind", choices=["causal", "logged", "t0fresh", "reader"])
    r.add_argument("--ckpt", default="")
    r.add_argument("--tag", default="")
    r.add_argument("--tree", default=BASE_TREE)
    r.add_argument("--out", default="")
    r.add_argument("--reader", default="")
    r.add_argument("--prereg", default="")
    r.add_argument("--prereg-sha", default="")
    r.add_argument("--orc-dir", default=f"{AR}/dagorc_rebase/read")
    g = sub.add_parser("guards")
    g.add_argument("--ram-min-mb", type=int, default=0)
    g.add_argument("--disk")
    g.add_argument("--disk-min-gb", type=float, default=0.0)
    f = sub.add_parser("fmt")
    f.add_argument("tokens", nargs="*")
    sub.add_parser("cmdsha").add_argument("tokens", nargs="*")
    sub.add_parser("world")
    ns = ap.parse_args(argv)
    if ns.cmd == "boxlog":
        P = parse_boxlog(open(ns.log, "rb").read().decode("utf-8", "replace") if os.path.isfile(ns.log) else "", ns.art)
        if ns.what == "run":
            st, d = run_state(P, ns.arm, int(ns.seed))
            if st == "PASS":
                print("PASS", d)
                return 0
            print(("FAIL " if st == "FAIL" else "WAIT " + st + " ") + d)
            return 3 if st == "FAIL" else 1
        if ns.what == "next":
            res = next_ready(P, {x for x in ns.complete.split(",") if x})
            if res[0] == "RUN":
                print("RUN", res[1], res[2], res[3])
                return 0
            print(res[0], res[1])
            return 3 if res[0] == "FAIL" else 1
        if ns.what == "meta":
            m = log_meta(P)
            print(" ".join(f"{k}={v}" for k, v in m.items()), f"problems={len(P['problems'])}")
            return 0
        if ns.what == "done":
            return 0 if log_meta(P)["done"] else 1
        for (a, s) in RUNS:
            st, d = run_state(P, a, s)
            print(f"{run_name(a, s)} ({LABEL[(a, s)]}): {st} {d[:120]}")
        for pr in P["problems"]:
            print("log problem:", pr)
        return 0
    if ns.cmd == "pullcheck":
        rc, kind, msg = pullcheck(ns.dir, ns.arm, ns.seed, ns.expect_sha, ns.min_age)
        print(kind, msg)
        return rc
    if ns.cmd == "commits":
        o = commits_of(ns.art)
        print(json.dumps(o))
        return 0 if o["ok"] else 3
    if ns.cmd == "render":
        for t in render(ns.kind, ns.ckpt, ns.tag, ns.tree, ns.out, reader=ns.reader, prereg=ns.prereg, prereg_sha=ns.prereg_sha, orc_dir=ns.orc_dir):
            print(t)
        return 0
    if ns.cmd == "guards":
        rc, txt = guards(ns.ram_min_mb, ns.disk, ns.disk_min_gb)
        print(txt)
        return rc
    if ns.cmd == "fmt":
        print(fmt(ns.tokens))
        return 0
    if ns.cmd == "world":
        print("WORLD-OK" if world_ok() else "NO-WORLD")
        return 0 if world_ok() else 1
    if ns.cmd == "cmdsha":
        print(cmd_sha(ns.tokens))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
