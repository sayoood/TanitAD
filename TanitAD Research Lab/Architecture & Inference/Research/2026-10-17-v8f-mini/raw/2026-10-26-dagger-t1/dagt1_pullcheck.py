#!/usr/bin/env python3
"""dagt1_pullcheck.py -- is one DAgger run's pull, or one seed's bank, on the dev box COMPLETE?   (PREREG_V8F_DAGGER dev-box T1 chain; CPU only; read-only)

    python dagt1_pullcheck.py run  <ART> <dag|dag0> <seed>
      rc 0   prints  COMPLETE <ckpt sha256> <why>
      rc 1   prints  INCOMPLETE <the first reason>          (the caller waits)
      rc 3   prints  WRONG <the contradiction>              (the files verify, or the sums carry a forbidden entry, but this is not the run asked for: the caller STOPS, waiting would never fix it)
    python dagt1_pullcheck.py bank <ART> <seed> [--expect-shards N]
      rc 0   prints  COMPLETE <bank.json sha256> <why>      rc 1 prints INCOMPLETE <reason>      rc 3 prints WRONG <the contradiction>

Source of the layout: box 2's chain (dagq/box2_ship/chain_v8f_dag.sh, `publish_bank_one` and `runone`; ART = DAG_ART, default E:/Projects/TanitAD-artifacts/v8f-dagger):

    <ART>/M2b_s<S>_<dag|dag0>/   ckpt_last.pt  eval_M2b-<arm>_s<S>.{json,npz}  selfcheck.json summary.json argv.json train_log.jsonl launch_gate_token.json   SHA256SUMS.txt = EXACTLY these 8 names
                                 [gate/pin.txt, gate/selfcheck.json: the box-2 gate's pinned training argv and its record; not in the sums; pin.txt is read by the bank check below]
    <ART>/bank_s<S>/             PROVENANCE.json  bank.json  shard_KKKofNNN.provenance.json (N of them)  shard_KKKofNNN.npz (N of them)  bank_stage.log     SHA256SUMS.txt = EXACTLY these 3 + 2N names (11 for the queue's N = 4)

The four kinds of bank file are what dag_read.load_bank needs: PROVENANCE.json (`bank.{s}.report`), bank.json (`bank_file`; its sha256 IS the bank's identity), the shard provenance files, and the shard npz (their `header_json`
member is the ONLY place the bank's clips are listed by name: K-B3.1).  The box-2 chain writes SHA256SUMS.txt LAST, after every file is pulled and verified, and only for a bank that passed every control and COVERAGE.

NO FRESH-AGE EVAL FROM BOX 2 (decision O1, 2026-10-26: the fresh-age T0 comparator runs on the dev box CPU only, so K-EQ never compares a CUDA eval with a CPU tick 0).  A run's sums therefore list EXACTLY the 8 names above.
A SHA256SUMS.txt that carries a fresh-age eval (eval_*-fresh.{json,npz}: the 10-name form of a box-2 chain that still runs `evaldagfresh`) is refused as WRONG, never accepted and never waited for.

A run is COMPLETE iff, by CONTENT and not by presence (the orct1 rule):
  1. SHA256SUMS.txt exists, was not modified in the last MIN_AGE_S seconds, ends with a newline, lists exactly the 8 names (every name once);
  2. every listed file exists, is non-empty and its streamed sha256 equals the listed one;
  3. the four small JSON records parse and summary.json says done: true.
It is WRONG (rc 3) when the sums carry a box-2 fresh eval, or, with 1-3 holding, the run's own records contradict its identity: summary arm / seed, the argv's --seed / --init-ckpt / --out / --dagger-p, or the DAG-only flags
(dag names all four dagger flags, dag0 only --dagger-p 0).  A dag0 directory that holds DAG's run (or the other way round) is therefore a STOP, never a wait.

A bank is COMPLETE iff:
  1. SHA256SUMS.txt exists, aged, newline-terminated, no name twice; it lists PROVENANCE.json and bank.json and both verify;
  2. the expected name set DERIVED from those two files (bank.json n_shards / shards[].file, PROVENANCE.shard_provenance) equals the listed set EXACTLY (3 + 2N names);
  3. every listed file exists, is non-empty and hashes to the listed sha256;
  4. no verified file contradicts another (WRONG if so): PROVENANCE.bank_sha256 == sha256(bank.json); every shard npz hashes to bank.json's file_sha256 AND to its shard provenance's;
  5. bank.json's sha256 equals the `--dagger-bank-sha256` that box 2's trainer gate DECLARED in the training argv it pinned (<ART>/M2b_s<S>_dag/gate/pin.txt): the bank on disk is the bank DAG s<S> was gated on.
     (the pin not yet pulled = INCOMPLETE; a pin that names another sha256, or none, = WRONG).
It does NOT judge the science (K-B*, COVERAGE): that is the box-2 chain's and the reader's job.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

MIN_AGE_S = 3.0          # SHA256SUMS.txt is written by `sha256sum | tee`: a file younger than this may still be filling
ARMS = ("dag", "dag0")
LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")
FRESH_NAME = re.compile(r"^eval_.+-fresh\.(json|npz)$")
DAG_ONLY = ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")
BANK_KIND, BANK_SCHEMA = "bank", "v8f-dagger-bank-v1"
BANK_STAGE_LOG = "bank_stage.log"


def run_name(arm: str, seed: int) -> str:
    return f"M2b_s{seed}_{arm}"


def eval_tag(arm: str, seed: int) -> str:
    return f"M2b-{arm}_s{seed}"


def base_files(arm: str, seed: int) -> list[str]:
    """THE accepted set of a run's SHA256SUMS.txt: 8 names, no fresh-age eval (decision O1)."""
    tg = eval_tag(arm, seed)
    return ["ckpt_last.pt", f"eval_{tg}.json", f"eval_{tg}.npz", "selfcheck.json", "summary.json", "argv.json", "train_log.jsonl", "launch_gate_token.json"]


def box2_fresh_files(arm: str, seed: int) -> list[str]:
    """What a box-2 chain that still ran `evaldagfresh` would also have listed (the 10-name form).  Never accepted: used only to name the refusal."""
    tg = eval_tag(arm, seed)
    return [f"eval_{tg}-fresh.json", f"eval_{tg}-fresh.npz"]


def bank_names(n: int) -> dict[str, list[str]]:
    """{kind: [file names]} of a bank of n shards, in the generator's naming (v8f_dagger_bank.py: shard_KKKofNNN.{npz,provenance.json})."""
    return {"provenance": ["PROVENANCE.json"], "manifest": ["bank.json"],
            "shard_provenance": [f"shard_{k:03d}of{n:03d}.provenance.json" for k in range(n)],
            "shard_npz": [f"shard_{k:03d}of{n:03d}.npz" for k in range(n)], "stage_log": [BANK_STAGE_LOG]}


def bank_flat(n: int) -> list[str]:
    nm = bank_names(n)
    return [x for k in ("provenance", "manifest", "shard_provenance", "shard_npz", "stage_log") for x in nm[k]]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def flags_of(argv: list) -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = {}
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
    if not occ or len(occ) != 1 or len(occ[0]) != 1:
        return None
    return occ[0][0]


def norm(p: str) -> str:
    return str(p).replace("\\", "/").rstrip("/")


def read_sums(d: str, min_age_s: float, now: float | None) -> tuple[dict[str, str] | None, str]:
    """(listed {name: sha256}, '') or (None, why-INCOMPLETE): SHA256SUMS.txt exists, is not younger than min_age_s, ends with a newline, every line is `<sha256>  <name>`, no name twice."""
    sums = os.path.join(d, "SHA256SUMS.txt")
    if not os.path.isfile(sums):
        return None, f"{d} has no SHA256SUMS.txt yet (the box-2 chain writes it after the last scp returns)"
    age = (time.time() if now is None else now) - os.path.getmtime(sums)
    if age < min_age_s:
        return None, f"SHA256SUMS.txt was modified {age:.1f} s ago (< {min_age_s} s): it may still be filling"
    with open(sums, encoding="utf-8", errors="replace", newline="") as f:
        text = f.read()
    if not text.endswith("\n"):
        return None, "SHA256SUMS.txt does not end with a newline (its last line may be partial)"
    listed: dict[str, str] = {}
    for raw in text.splitlines():
        m = LINE.match(raw.rstrip("\r"))
        if not m:
            return None, f"SHA256SUMS.txt has a line that is not '<sha256>  <name>': {raw[:80]!r}"
        name = m.group(2).strip()
        if name in listed:
            return None, f"SHA256SUMS.txt lists {name} twice"
        listed[name] = m.group(1)
    return listed, ""


def identity_problems(arm: str, seed: int, d: str) -> list[str]:
    """The run's own records must say it is THIS arm and seed.  Called only once the files verify (so a half-pulled file cannot read as a contradiction)."""
    bad = []
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
        sm = json.load(f)
    with open(os.path.join(d, "argv.json"), encoding="utf-8") as f:
        aj = json.load(f)
    if sm.get("arm") != "M2b" or sm.get("seed") != seed:
        bad.append(f"summary.json says arm {sm.get('arm')!r} seed {sm.get('seed')!r}, this is M2b seed {seed}")
    argv = aj.get("argv")
    if not isinstance(argv, list) or not argv:
        return bad + ["argv.json carries no argv list"]
    fl = flags_of(argv)
    if one(fl, "--seed") != str(seed):
        bad.append(f"argv --seed {one(fl, '--seed')!r}, expected {seed}")
    ic = one(fl, "--init-ckpt")
    if ic is None or not norm(ic).endswith(f"M2b_s{seed}/ckpt_last.pt"):
        bad.append(f"argv --init-ckpt {ic!r} is not .../M2b_s{seed}/ckpt_last.pt")
    out = one(fl, "--out")
    if out is None or os.path.basename(norm(out)) != run_name(arm, seed):
        bad.append(f"argv --out {out!r} does not end in {run_name(arm, seed)}: this directory holds another run's record")
    p = one(fl, "--dagger-p")
    try:
        pv = float(p)
    except (TypeError, ValueError):
        pv = None
    if pv is None:
        bad.append(f"argv names --dagger-p {p!r}, not exactly one number")
    elif arm == "dag0" and pv != 0.0:
        bad.append(f"dag0's argv names --dagger-p {p}: that is a DAG run")
    elif arm == "dag" and not pv > 0.0:
        bad.append(f"dag's argv names --dagger-p {p}: that is a DAG-0 run")
    if arm == "dag0":
        for n in DAG_ONLY:
            if n in fl:
                bad.append(f"dag0's argv names {n}: DAG-0 names only --dagger-p 0")
    elif arm == "dag":
        for n in DAG_ONLY:
            if n not in fl:
                bad.append(f"dag's argv does not name {n}")
    lv = sm.get("levers") if isinstance(sm.get("levers"), dict) else {}
    if "dagger_p" in lv and pv is not None and float(lv["dagger_p"]) != pv:
        bad.append(f"summary.json built dagger_p {lv['dagger_p']} != the argv's {pv}")
    return bad


def check(art: str, arm: str, seed: int, min_age_s: float = MIN_AGE_S, now: float | None = None) -> tuple[str, str, str]:
    """-> (status COMPLETE | INCOMPLETE | WRONG, reason, ckpt sha256 ('' unless COMPLETE))"""
    if arm not in ARMS:
        return "INCOMPLETE", f"unknown arm {arm!r}", ""
    d = os.path.join(art, run_name(arm, seed))
    if not os.path.isdir(d):
        return "INCOMPLETE", f"no run directory {d} yet", ""
    listed, why = read_sums(d, min_age_s, now)
    if listed is None:
        return "INCOMPLETE", why, ""
    carried = sorted(n for n in listed if FRESH_NAME.match(n))
    if carried:                                                      # decision O1: the fresh-age eval is a dev-box CPU step; a box-2 SUMS that lists one is another build's pull, and waiting cannot fix it
        return "WRONG", (f"SHA256SUMS.txt in {run_name(arm, seed)} carries a box-2 fresh-age eval {carried}: decision O1 (2026-10-26) puts the fresh-age T0 eval on the dev box CPU only, so the only accepted "
                         f"form is the {len(base_files(arm, seed))}-name one without it; this pull came from a box-2 chain that still runs evaldagfresh"), ""
    want = base_files(arm, seed)
    if sorted(listed) != sorted(want):
        for oa in ARMS:                                              # a directory that lists ANOTHER run's file names holds that run's pull: waiting would never fix it
            for osd in (0, 1):
                if (oa, osd) != (arm, seed) and sorted(listed) == sorted(base_files(oa, osd)):
                    return "WRONG", f"SHA256SUMS.txt in {run_name(arm, seed)} lists {run_name(oa, osd)}'s files: this directory holds another run's pull", ""
        miss, extra = sorted(set(want) - set(listed)), sorted(set(listed) - set(want))
        return "INCOMPLETE", f"SHA256SUMS.txt lists {len(listed)} names, not the exact set of {len(want)} (missing {miss}, unexpected {extra})", ""
    for name in want:
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            return "INCOMPLETE", f"{name} is listed but absent", ""
        if os.path.getsize(p) == 0:
            return "INCOMPLETE", f"{name} is empty", ""
    for name in want:
        got = sha256_file(os.path.join(d, name))
        if got != listed[name]:
            return "INCOMPLETE", f"{name} sha256 {got[:12]} != the listed {listed[name][:12]} (half-pulled, or replaced after SHA256SUMS.txt was written)", ""
    for name in ("selfcheck.json", "summary.json", "argv.json", "launch_gate_token.json"):
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError) as e:
            return "INCOMPLETE", f"{name} does not parse: {type(e).__name__}: {e}", ""
        if name == "summary.json" and rec.get("done") is not True:
            return "INCOMPLETE", f"summary.json done is {rec.get('done')!r}, not true (the training did not finish)", ""
    bad = identity_problems(arm, seed, d)
    if bad:
        return "WRONG", f"{run_name(arm, seed)} verifies but is not this run: " + "; ".join(bad), ""
    return "COMPLETE", f"{len(want)} files present, non-empty, sha256-verified against SHA256SUMS.txt; summary.json done; identity of {run_name(arm, seed)} consistent", listed["ckpt_last.pt"]


# ---------------------------------------------------------------------------------------------------------------- the bank directory
def argv_of(text: str) -> list[str] | None:
    """The training argv of a pinned queue line: the tokens after v8f_mini_train.py, double quotes grouping and removed (the box-2 chain's own tokenisation).  None if the line names no trainer."""
    t = [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', text.strip())]
    k = next((i for i, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")), None)
    return None if k is None else t[k + 1:]


def declared_bank_sha(art: str, seed: int) -> tuple[str, str]:
    """-> (status, value-or-reason): the `--dagger-bank-sha256` the trainer gate of DAG s<seed> pinned (box 2's <RUNSM>\\dagpin\\M2b_s<S>_dag.train.argv.txt, pulled by the box-2 chain to <ART>/M2b_s<S>_dag/gate/pin.txt)."""
    pin = os.path.join(art, run_name("dag", seed), "gate", "pin.txt")
    if not os.path.isfile(pin) or os.path.getsize(pin) == 0:
        return "INCOMPLETE", f"{pin} is not pulled yet: the bank cannot be tied to the sha256 box 2's trainer gate declared for DAG s{seed}"
    with open(pin, encoding="utf-8", errors="replace") as f:
        argv = argv_of(f.read())
    if argv is None:
        return "WRONG", f"{pin} is not a training line (no v8f_mini_train.py in it)"
    fl = flags_of(argv)
    v = one(fl, "--dagger-bank-sha256")
    if v is None:
        return "WRONG", f"{pin} names --dagger-bank-sha256 {len(fl.get('--dagger-bank-sha256', []))} time(s) or with no single value: DAG s{seed} was not gated on a declared bank"
    if not re.fullmatch(r"[0-9a-f]{64}", v):
        return "WRONG", f"{pin} declares --dagger-bank-sha256 {v!r}, which is not a sha256"
    return "OK", v


def check_bank_status(art: str, seed: int, min_age_s: float = MIN_AGE_S, now: float | None = None, expect_shards: int | None = None) -> tuple[str, str, str]:
    """-> (COMPLETE | INCOMPLETE | WRONG, reason, bank.json sha256 ('' unless COMPLETE)).  See the module docstring for the five conditions."""
    d = os.path.join(art, f"bank_s{seed}")
    if not os.path.isdir(d):
        return "INCOMPLETE", f"no bank directory {d} yet", ""
    listed, why = read_sums(d, min_age_s, now)
    if listed is None:
        return "INCOMPLETE", why.replace("(the box-2 chain writes it after the last scp returns)", "(the box-2 chain writes it LAST: after the bank controls and COVERAGE passed and every shard npz is pulled and verified)"), ""
    for nm in ("PROVENANCE.json", "bank.json"):                     # the two files the expected set is derived from: listed, present, verified FIRST
        p = os.path.join(d, nm)
        if nm not in listed:
            return "INCOMPLETE", f"SHA256SUMS.txt does not list {nm} ({len(listed)} names listed)", ""
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return "INCOMPLETE", f"{nm} is absent or empty", ""
        got = sha256_file(p)
        if got != listed[nm]:
            return "INCOMPLETE", f"{nm} sha256 {got[:12]} != the listed {listed[nm][:12]} (half-pulled, or replaced after SHA256SUMS.txt was written)", ""
    try:
        with open(os.path.join(d, "PROVENANCE.json"), encoding="utf-8") as f:
            prov = json.load(f)
        with open(os.path.join(d, "bank.json"), encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError) as e:
        return "INCOMPLETE", f"PROVENANCE.json / bank.json does not parse: {type(e).__name__}: {e}", ""
    if not isinstance(prov, dict) or prov.get("kind") != BANK_KIND or prov.get("schema") != BANK_SCHEMA:
        return "INCOMPLETE", f"PROVENANCE.json kind/schema {prov.get('kind') if isinstance(prov, dict) else None!r}/{prov.get('schema') if isinstance(prov, dict) else None!r}, not {BANK_KIND!r}/{BANK_SCHEMA!r}", ""
    n = man.get("n_shards") if isinstance(man, dict) else None
    shards = man.get("shards") if isinstance(man, dict) else None
    if not isinstance(n, int) or isinstance(n, bool) or n < 1 or not isinstance(shards, list) or len(shards) != n:
        return "INCOMPLETE", f"bank.json lists n_shards {n!r} with {len(shards) if isinstance(shards, list) else shards!r} shard entries", ""
    if expect_shards is not None and n != expect_shards:
        return "WRONG", f"bank.json has {n} shards, the box-2 queue's bankdag runs {expect_shards}", ""
    names = bank_names(n)
    if [s.get("file") for s in sorted(shards, key=lambda s: s.get("k", -1))] != names["shard_npz"]:
        return "WRONG", f"bank.json shards name {[s.get('file') for s in shards]}, not {names['shard_npz']}", ""
    if prov.get("shard_provenance") != names["shard_provenance"] or prov.get("manifest", "bank.json") != "bank.json":
        return "WRONG", f"PROVENANCE.json names shard_provenance {prov.get('shard_provenance')!r} / manifest {prov.get('manifest')!r}, not {names['shard_provenance']} / bank.json", ""
    want = bank_flat(n)
    if sorted(listed) != sorted(want):
        miss, extra = sorted(set(want) - set(listed)), sorted(set(listed) - set(want))
        hint = " (the shard npz are what dag_read.load_bank reads the bank's clip names from, K-B3.1: a bank without them would REFUSE)" if any(x.endswith(".npz") for x in miss) else ""
        return "INCOMPLETE", f"SHA256SUMS.txt lists {len(listed)} names, not the exact set of {len(want)} (missing {miss}, unexpected {extra}){hint}", ""
    for nm in want:
        p = os.path.join(d, nm)
        if not os.path.isfile(p):
            return "INCOMPLETE", f"{nm} is listed but absent", ""
        if os.path.getsize(p) == 0:
            return "INCOMPLETE", f"{nm} is empty", ""
    for nm in want:
        got = sha256_file(os.path.join(d, nm))
        if got != listed[nm]:
            return "INCOMPLETE", f"{nm} sha256 {got[:12]} != the listed {listed[nm][:12]} (half-pulled, or replaced after SHA256SUMS.txt was written)", ""
    msha = listed["bank.json"]
    if prov.get("bank_sha256") != msha:                              # both files verified against the sums and still disagree: no wait will fix it
        return "WRONG", f"bank.json hashes to {msha[:12]}, not PROVENANCE.bank_sha256 {str(prov.get('bank_sha256'))[:12]}", ""
    by_k = {s.get("k"): s for s in shards}
    for k, (npz, pn) in enumerate(zip(names["shard_npz"], names["shard_provenance"])):
        if listed[npz] != by_k[k].get("file_sha256"):
            return "WRONG", f"shard {npz} hashes to {listed[npz][:12]}, not bank.json's file_sha256 {str(by_k[k].get('file_sha256'))[:12]}", ""
        try:
            with open(os.path.join(d, pn), encoding="utf-8") as f:
                sp = json.load(f)
        except (OSError, ValueError) as e:
            return "INCOMPLETE", f"shard provenance {pn} does not parse: {type(e).__name__}: {e}", ""
        if not isinstance(sp, dict) or sp.get("file_sha256") != listed[npz]:
            return "WRONG", f"shard provenance {pn} records file_sha256 {str(sp.get('file_sha256') if isinstance(sp, dict) else None)[:12]}, the npz hashes to {listed[npz][:12]}", ""
    st, v = declared_bank_sha(art, seed)
    if st != "OK":
        return st, v, ""
    if v != msha:
        return "WRONG", (f"bank.json in {d} hashes to {msha[:12]}, but box 2's trainer gate declared --dagger-bank-sha256 {v[:12]} for DAG s{seed} "
                         f"({os.path.join(art, run_name('dag', seed), 'gate', 'pin.txt')}): this is not the bank that run was gated on"), ""
    return "COMPLETE", (f"{len(want)} files = PROVENANCE.json + bank.json (sha256-bound, == the gate's declared --dagger-bank-sha256) + {n} shard provenance + {n} shard npz (== bank.json's file_sha256) + {BANK_STAGE_LOG}, "
                        f"all present, non-empty, sha256-verified against SHA256SUMS.txt"), msha


def check_bank(art: str, seed: int, min_age_s: float = MIN_AGE_S, now: float | None = None, expect_shards: int | None = None) -> tuple[bool, str, str]:
    """-> (complete, reason, bank.json sha256): check_bank_status with the verdict folded to a bool (a WRONG reason says so)."""
    st, why, sha = check_bank_status(art, seed, min_age_s, now, expect_shards)
    return st == "COMPLETE", (f"WRONG: {why}" if st == "WRONG" else why), sha


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    usage = "usage: dagt1_pullcheck.py run <ART> <dag|dag0> <seed>  |  bank <ART> <seed> [--expect-shards N]"
    if not a:
        print(usage, file=sys.stderr)
        return 2
    if a[0] == "run":
        if len(a) != 4 or not a[3].isdigit():                        # --b2-fresh was REMOVED with decision O1: a caller that still passes it is told, not silently served
            print(usage + ("   (--b2-fresh was removed: box 2 pulls no fresh-age eval, decision O1)" if "--b2-fresh" in a else ""), file=sys.stderr)
            return 2
        st, why, sha = check(a[1], a[2], int(a[3]))
        print(f"{st} {sha} {why}" if st == "COMPLETE" else f"{st} {why}")
        return {"COMPLETE": 0, "INCOMPLETE": 1, "WRONG": 3}[st]
    if a[0] == "bank":
        exp = None
        if "--expect-shards" in a:
            k = a.index("--expect-shards")
            if k + 1 >= len(a) or not a[k + 1].isdigit():
                print(usage, file=sys.stderr)
                return 2
            exp = int(a[k + 1])
            del a[k:k + 2]
        if len(a) != 3 or not a[2].isdigit():
            print(usage, file=sys.stderr)
            return 2
        st, why, sha = check_bank_status(a[1], int(a[2]), expect_shards=exp)
        print(f"COMPLETE {sha} {why}" if st == "COMPLETE" else f"{st} {why}")
        return {"COMPLETE": 0, "INCOMPLETE": 1, "WRONG": 3}[st]
    print(usage, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
