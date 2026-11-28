#!/usr/bin/env python3
"""dagoacc_t1_pullcheck.py -- is one DAG-OACC run's pull on the dev box COMPLETE?   (PREREG_V8F_DAG_OACC dev-box T1 chain; CPU only; read-only; it never opens an eval_* or a checkpoint's VALUES -- it HASHES them)
Sibling of flatdag_t1_pullcheck.py (which ran PREREG_V8F_FLATDAG's chain).  Changes: ONE arm (dagoacc), the 8-name sums (no oracle eval: the cut is not in this arm), the init (M2b_s<seed>, DAG's), the iteration-1 bank
(DAG's own, BANKED and read-only: dagger_bank/s<seed>), and C-O1 on the argv text: DAG-OACC's argv equals DAG's banked argv except --out and --w-acc, --w-acc named once with the w_acc stage's value.

    python dagoacc_t1_pullcheck.py run   <ART> dagoacc <seed> [--dag-art DIR]
      rc 0   prints  COMPLETE <ckpt sha256> <why>
      rc 1   prints  INCOMPLETE <the first reason>          (the caller waits)
      rc 3   prints  WRONG <the contradiction>              (the files verify but this is not the run asked for: the caller STOPS, waiting would never fix it)
    python dagoacc_t1_pullcheck.py wacc  <ART>                       the box-2 w_acc stage's record (<ART>/wacc/w_acc.json + w_acc.txt): rc 0 prints "OK <W>" | 1 INCOMPLETE | 3 WRONG
    python dagoacc_t1_pullcheck.py dagbank <DAG_ART> <seed>          DAG's BANKED iteration-1 bank (bank_s<seed>): bank.json's sha256 == the sha256 DAG's argv declared: rc 0 COMPLETE | 1 INCOMPLETE | 3 WRONG

Layout (box 2's chain_v8f_dagoacc_r2.sh `runone`; ART = DAGOACC_ART, default E:/Projects/TanitAD-artifacts/v8f-dagoacc):

    <ART>/M2b_s<S>_dagoacc/   ckpt_last.pt  eval_M2b-dagoacc_s<S>.{json,npz}  selfcheck.json  summary.json  argv.json  train_log.jsonl  launch_gate_token.json     SHA256SUMS.txt = EXACTLY these 8 names
                              (box 2 pulls train.log, the eval .log and eval_*_proj.npz too, and the gate's gate/{selfcheck.json,pin.txt,dag_pin.txt,gate_*.log}; none is in the sums)
    <ART>/wacc/               w_acc.json  w_acc.txt  waccdag.log                                                                        (the w_acc stage, pulled before any gate)

A run is COMPLETE iff, by CONTENT and not by presence: (1) SHA256SUMS.txt exists, aged, newline-terminated, lists exactly the 8 names once; (2) every listed file exists, is non-empty, and its streamed sha256 equals the
listed one; (3) the four small JSON records parse and summary.json says done: true.   It is WRONG (rc 3) when (1)-(3) hold and the run's own records contradict its identity (C-O1): summary arm / seed, the argv's --seed /
--init-ckpt / --out / --dagger-p / the iteration-1 bank flags and sha256 / --dagger-seed, --w-acc named once with the w_acc stage's value, and the ONE-LEVER rule (the argv equals DAG's banked argv except --out and --w-acc).
It does NOT judge the science (C-O2's gate record and every row are the READER's).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time

MIN_AGE_S = 3.0          # SHA256SUMS.txt is written by `sha256sum | tee`: a file younger than this may still be filling
ARMS = ("dagoacc",)
LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")
FRESH_NAME = re.compile(r"^eval_.+-fresh\.(json|npz)$")
DAGGER_FLAGS = ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")
DAGGER_P = 0.1
DAG_ART_DEFAULT = "E:/Projects/TanitAD-artifacts/v8f-dagger"
# DAG_s<S> (the control, BANKED) and its init, bank and argv: the box-2 chain's own pins (chain_v8f_dagoacc_r2.sh), each a sha256 of a banked file on this disk
DAG_CKPT_SHA = {0: "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1", 1: "1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef"}      # v8f-dagger/M2b_s<S>_dag/SHA256SUMS.txt
DAG_ARGV_SHA = {0: "50e9d01d3dee881367a4eb1433903b05cc093c83d7bccc32a518e2290d9f3ce3", 1: "38a591e3e5a05c6f72669a268aaf99591af167758296720d650c46ebba4541b3"}      # v8f-dagger/M2b_s<S>_dag/argv.json
M2B_CKPT_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}      # M2b_s<S>: DAG's and DAG-OACC's init
BANK1_SHA = {0: "197a68bb540be5a7d157942e9157e1c8e60ca60d82b1ca4fb4fcb68edccfaa8f", 1: "36c7aea964dafe9423253aa479c2fc6cdfc93848395b593315cba812f5f04d4d"}        # the ITERATION-1 bank.json sha256 (DAG's --dagger-bank-sha256)


def run_name(arm: str, seed: int) -> str:
    return f"M2b_s{seed}_{arm}"


def eval_tag(arm: str, seed: int) -> str:
    return f"M2b-{arm}_s{seed}"


def base_files(arm: str, seed: int) -> list[str]:
    """THE accepted set of a run's SHA256SUMS.txt: 8 names (box 2's chain_v8f_dagoacc_r2.sh `sha256sum` line in runone)."""
    tg = eval_tag(arm, seed)
    return ["ckpt_last.pt", f"eval_{tg}.json", f"eval_{tg}.npz", "selfcheck.json", "summary.json", "argv.json", "train_log.jsonl", "launch_gate_token.json"]


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


def drop_flag(argv: list, names) -> list:
    """argv without each flag in ``names`` and the ONE value token after it."""
    o, k = [], 0
    while k < len(argv):
        if argv[k] in names:
            k += 2
        else:
            o.append(argv[k])
            k += 1
    return o


def finite_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def one_lever_problems(argv: list, ref_argv: list, w: float | None) -> list[str]:
    """C-O1 on the argv text: ``argv`` == ``ref_argv`` (DAG_s<S>'s banked argv.json argv) except --out and --w-acc; --w-acc named exactly once, with the value ``w`` (the w_acc stage's) when it is known."""
    bad: list[str] = []
    if "--w-acc" in ref_argv:
        bad.append("the REFERENCE (DAG's banked argv) already names --w-acc: it is not the control")
    n = [i for i, t in enumerate(argv) if t == "--w-acc"]
    if len(n) != 1:
        bad.append(f"--w-acc is named {len(n)} time(s), exactly once is the lever")
    else:
        v = argv[n[0] + 1] if n[0] + 1 < len(argv) else None
        try:
            vf = float(v)
        except (TypeError, ValueError):
            vf = None
        if vf is None or not finite_num(vf) or vf <= 0.0:
            bad.append(f"--w-acc value {v!r} is not a finite positive number")
        elif w is not None and vf != w:
            bad.append(f"--w-acc {v} != the w_acc stage's value {w!r}")
    rest, ref = drop_flag(argv, ("--out", "--w-acc")), drop_flag(ref_argv, ("--out",))
    if rest != ref:
        k = next((i for i in range(min(len(rest), len(ref))) if rest[i] != ref[i]), min(len(rest), len(ref)))
        bad.append(f"the argv differs from DAG's banked argv in more than --out / --w-acc: {len(rest)} vs {len(ref)} tokens, first difference at token {k}: {rest[k:k + 2]} vs {ref[k:k + 2]}")
    return bad


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


# ---------------------------------------------------------------------------------------------------------------- the w_acc stage record
def wacc_status(art: str) -> tuple[str, str | float]:
    """-> (OK, W) | (INCOMPLETE, why) | (WRONG, why): <ART>/wacc/w_acc.json and w_acc.txt are pulled and AGREE (the json's w_acc is a finite positive number equal to the txt's number)."""
    jp, tp = os.path.join(art, "wacc", "w_acc.json"), os.path.join(art, "wacc", "w_acc.txt")
    for p in (jp, tp):
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return "INCOMPLETE", f"{p} is not pulled yet (the w_acc stage is pulled before any gate)"
    try:
        with open(jp, encoding="utf-8") as f:
            rec = json.load(f)
        txt = open(tp, encoding="utf-8").read().strip()
    except (OSError, ValueError) as e:
        return "INCOMPLETE", f"w_acc.json / w_acc.txt does not parse: {type(e).__name__}: {e}"
    w = rec.get("w_acc") if isinstance(rec, dict) else None
    if not finite_num(w) or w <= 0.0:
        return "WRONG", f"w_acc.json carries w_acc {w!r}: not a finite positive number"
    try:
        tv = float(txt)
    except ValueError:
        return "WRONG", f"w_acc.txt {txt!r} is not a number"
    if tv != w:
        return "WRONG", f"w_acc.txt {txt} != w_acc.json {w!r}"
    return "OK", float(w)


# ---------------------------------------------------------------------------------------------------------------- DAG's banked bank and argv
def dag_ref_argv(dag_art: str, seed: int) -> tuple[list | None, str]:
    """DAG_s<seed>'s banked argv (argv.json), after its sha256 is checked against the pin."""
    p = os.path.join(dag_art, f"M2b_s{seed}_dag", "argv.json")
    if not os.path.isfile(p):
        return None, f"{p} is absent: DAG's banked argv is the control of C-O1"
    if sha256_file(p) != DAG_ARGV_SHA[seed]:
        return None, f"{p} hashes to {sha256_file(p)[:12]}, pinned {DAG_ARGV_SHA[seed][:12]}"
    with open(p, encoding="utf-8") as f:
        return list(json.load(f)["argv"]), ""


def dagbank_status(dag_art: str, seed: int) -> tuple[str, str]:
    """DAG's ITERATION-1 bank (read-only): bank.json hashes to BANK1_SHA[seed], is listed with that sha256 in the bank's SHA256SUMS.txt, and PROVENANCE.json says so too."""
    d = os.path.join(dag_art, f"bank_s{seed}")
    bj = os.path.join(d, "bank.json")
    if not os.path.isfile(bj):
        return "INCOMPLETE", f"{bj} is absent"
    got = sha256_file(bj)
    if got != BANK1_SHA[seed]:
        return "WRONG", f"{bj} hashes to {got[:12]}, the sha256 DAG's argv declared (--dagger-bank-sha256) is {BANK1_SHA[seed][:12]}"
    listed, why = read_sums(d, 0.0, None)
    if listed is None:
        return "INCOMPLETE", why
    if listed.get("bank.json") != got:
        return "WRONG", f"SHA256SUMS.txt lists bank.json as {str(listed.get('bank.json'))[:12]}, it hashes to {got[:12]}"
    try:
        with open(os.path.join(d, "PROVENANCE.json"), encoding="utf-8") as f:
            prov = json.load(f)
    except (OSError, ValueError) as e:
        return "INCOMPLETE", f"PROVENANCE.json does not parse: {type(e).__name__}: {e}"
    if prov.get("bank_sha256") != got:
        return "WRONG", f"PROVENANCE.bank_sha256 {str(prov.get('bank_sha256'))[:12]} != bank.json's {got[:12]}"
    return "COMPLETE", f"DAG's iteration-1 bank_s{seed}: bank.json sha256 {got[:16]}... == the sha256 DAG's argv declared"


# ---------------------------------------------------------------------------------------------------------------- the run directory
def identity_problems(arm: str, seed: int, d: str, w: float | None, dag_argv: list | None) -> list[str]:
    """The run's own records must say it is THIS arm and seed and DAG's recipe plus the one lever (C-O1).  Called only once the files verify.
    Identity only: C-O2 (the gate shows the acceleration term running) and the built w_acc are the READER's (they route to REFUSED, not to a wait)."""
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
        bad.append(f"argv --init-ckpt {ic!r} is not .../M2b_s{seed}/ckpt_last.pt (PREREG_V8F_DAG_OACC section 1: the init is M2b_s{seed}, DAG's)")
    out = one(fl, "--out")
    if out is None or os.path.basename(norm(out)) != run_name(arm, seed):
        bad.append(f"argv --out {out!r} does not end in {run_name(arm, seed)}: this directory holds another run's record")
    p = one(fl, "--dagger-p")
    try:
        pv = float(p)
    except (TypeError, ValueError):
        pv = None
    if pv != DAGGER_P:
        bad.append(f"argv names --dagger-p {p!r}, DAG's recipe is {DAGGER_P}")
    for n in DAGGER_FLAGS:
        if n not in fl:
            bad.append(f"the argv does not name {n} (DAG's recipe names all four dagger flags)")
    bk = one(fl, "--dagger-bank")
    if bk is not None and not norm(bk).endswith(f"dagger_bank/s{seed}"):
        bad.append(f"argv --dagger-bank {bk!r} is not .../dagger_bank/s{seed}: DAG's ITERATION-1 bank is the only admissible one (a FLAT or iteration-2 bank is another arm)")
    bs = one(fl, "--dagger-bank-sha256")
    if bs != BANK1_SHA[seed]:
        bad.append(f"argv --dagger-bank-sha256 {str(bs)[:16]} != DAG's {BANK1_SHA[seed][:16]} (C-O1: the bank sha256 is identical)")
    if one(fl, "--dagger-seed") != str(seed):
        bad.append(f"argv --dagger-seed {one(fl, '--dagger-seed')!r}, expected {seed}")
    if dag_argv is None:
        bad.append("DAG's banked argv could not be read, so C-O1 (one lever) cannot be checked")
    else:
        bad += [f"C-O1: {x}" for x in one_lever_problems(argv, dag_argv, w)]
    lv = sm.get("levers") if isinstance(sm.get("levers"), dict) else {}
    if "dagger_p" in lv and pv is not None and float(lv["dagger_p"]) != pv:
        bad.append(f"summary.json built dagger_p {lv['dagger_p']} != the argv's {pv}")
    ics = (lv.get("init_ckpt") or {}).get("sha256") if isinstance(lv.get("init_ckpt"), dict) else None
    if ics is not None and ics != M2B_CKPT_SHA[seed]:
        bad.append(f"summary.json levers.init_ckpt.sha256 {str(ics)[:16]} is not M2b_s{seed}'s checkpoint {M2B_CKPT_SHA[seed][:16]}")
    if "dagger_bank_sha256" in lv and lv["dagger_bank_sha256"] != BANK1_SHA[seed]:
        bad.append(f"summary.json built dagger_bank_sha256 {str(lv['dagger_bank_sha256'])[:16]} != DAG's {BANK1_SHA[seed][:16]}")
    return bad


def check(art: str, arm: str, seed: int, min_age_s: float = MIN_AGE_S, now: float | None = None, dag_art: str = DAG_ART_DEFAULT) -> tuple[str, str, str]:
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
    if carried:
        return "WRONG", (f"SHA256SUMS.txt in {run_name(arm, seed)} carries a box-2 fresh-age eval {carried}: decision O1 (2026-10-26) puts the fresh-age T0 eval on the dev box CPU only; "
                         f"the only accepted form is the {len(base_files(arm, seed))}-name one without it"), ""
    want = base_files(arm, seed)
    if sorted(listed) != sorted(want):
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
    if listed["ckpt_last.pt"] in (DAG_CKPT_SHA[seed], M2B_CKPT_SHA[seed]):
        return "WRONG", f"{run_name(arm, seed)}/ckpt_last.pt IS {'DAG' if listed['ckpt_last.pt'] == DAG_CKPT_SHA[seed] else 'M2b'}_s{seed}'s checkpoint: nothing was fine-tuned with the lever", ""
    st, w = wacc_status(art)
    if st == "INCOMPLETE":
        return "INCOMPLETE", str(w), ""
    if st == "WRONG":
        return "WRONG", f"the w_acc stage's record: {w}", ""
    ref, why = dag_ref_argv(dag_art, seed)
    bad = identity_problems(arm, seed, d, float(w), ref)
    if ref is None:
        return "INCOMPLETE", why, ""
    if bad:
        return "WRONG", f"{run_name(arm, seed)} verifies but is not this run: " + "; ".join(bad), ""
    return "COMPLETE", (f"{len(want)} files present, non-empty, sha256-verified against SHA256SUMS.txt; summary.json done; identity of {run_name(arm, seed)} consistent; "
                        f"C-O1: DAG's argv + --w-acc {w}, the iteration-1 bank sha256 identical"), listed["ckpt_last.pt"]


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    usage = "usage: dagoacc_t1_pullcheck.py run <ART> dagoacc <seed> [--dag-art DIR]  |  wacc <ART>  |  dagbank <DAG_ART> <seed>"
    if not a:
        print(usage, file=sys.stderr)
        return 2
    code = {"COMPLETE": 0, "OK": 0, "INCOMPLETE": 1, "WRONG": 3}
    if a[0] == "run":
        dag = DAG_ART_DEFAULT
        if "--dag-art" in a:
            k = a.index("--dag-art")
            if k + 1 >= len(a):
                print(usage, file=sys.stderr)
                return 2
            dag = a[k + 1]
            del a[k:k + 2]
        if len(a) != 4 or not a[3].isdigit() or int(a[3]) not in (0, 1):
            print(usage, file=sys.stderr)
            return 2
        st, why, sha = check(a[1], a[2], int(a[3]), dag_art=dag)
        print(f"{st} {sha} {why}" if st == "COMPLETE" else f"{st} {why}")
        return code[st]
    if a[0] == "wacc":
        if len(a) != 2:
            print(usage, file=sys.stderr)
            return 2
        st, v = wacc_status(a[1])
        print(f"OK {v!r}" if st == "OK" else f"{st} {v}")
        return code[st]
    if a[0] == "dagbank":
        if len(a) != 3 or not a[2].isdigit() or int(a[2]) not in (0, 1):
            print(usage, file=sys.stderr)
            return 2
        st, why = dagbank_status(a[1], int(a[2]))
        print(f"{st} {why}")
        return code[st]
    print(usage, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
