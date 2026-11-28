#!/usr/bin/env python3
"""ra_gate_record.py -- the RA chain's launch-gate record for ONE RA stage-B arm (RA_s0 | RA_s1). A COPY of rh_gate_record.py (sha256 847019579b8ae271...), made by make_ra_gate_record.py.
Differences, and only these: COMMIT is the RA commit; the G-SUITE result is gsuite_result_ra.json; the tree marker is done_ra/tree.done; RA arms take the RH branch plus three add-mode checks
(argv names --dagger-mode add; the built levers carry dagger_mode add; the dagger probe read rows_appended > 0); any non-RA tag is refused.
Original docstring follows (RH_sk or B0_sk). stdlib only, CPU, sub-second.

WHAT IT IS NOT: it is NOT a dscale10k_check.py / launch_gate.py token. The committed chain checker (study/dscale10k_check.py: `argv_diff_problems`, `required_checks`) knows the DSCALE10K arm names only, and
the card's section 4.1 C-checks for RH live in rollhist_pair_check.py (called by render_argv_rh.py). So this record is the chain's own, minted by the chain, never by hand, and verified again at the moment of launch.
The MM decides whether a study-side RH kind must be added to the checker before the launch (listed in HANDOFF_B2.md).

  mint   --tag RH_s0 --readout TOKEN         writes $R/runs/<tag>/gate/RH_GATE_RECORD.json ONLY if every check below passes; exit 0/1
  verify --tag RH_s0                         re-hashes what the record binds (argv, bank.json, selfcheck, G-SUITE result, pilot report, tree marker); exit 0/1

Checks at mint: (1) tree blob check done-marker + commit; (2) G-SUITE result: 0 regressions, 0 new-test failures, 0 missing, new commit == 43319227; (3) PILOT_RH.json pass and the same readout;
(4) RH only: the full-bank check json passes (b,c,d,e) for this readout and the bank.json sha equals the argv's --dagger-bank-sha256; (5) the pair check printed PASS (pair_<tag>.log) ;
(6) the trainer's own --selfcheck-only record of THIS argv: declared_vs_built PASS, acc_probe pass, and for RH dagger_probe pass, and the declaration token appears in the built levers; for B0 the dagger levers read default;
(7) the argv names what the arm is: RH names --dagger-bank-readout <token> and --dagger-bank-kappa 0.2 and --dagger-p > 0; B0 names --dagger-p 0 and no bank flag."""
import argparse
import re
import hashlib
import json
import os
import sys

R = os.environ.get("RH_ROOT", "/root/rollhist")
COMMIT = "df250a0032e56aae89304421918ade5c583d53a0"                                      # filled by fill_ra_commit.sh (MM, after landing): 40 hex or this file refuses to run


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_key(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                yield v
            yield from find_key(v, key)
    elif isinstance(o, list):
        for v in o:
            yield from find_key(v, key)


def argv_map(toks):
    out, i = {}, 0
    while i < len(toks):
        if toks[i].startswith("--"):
            j = i + 1
            while j < len(toks) and not (toks[j].startswith("--") and len(toks[j]) > 2):
                j += 1
            out[toks[i]] = " ".join(toks[i + 1:j]) if j > i + 1 else True
            i = j
        else:
            i += 1
    return out


def files(tag):
    d = f"{R}/runs/{tag}"
    seed = tag.split("_s")[1]
    f = {"argv": f"{d}/argv.json", "selfcheck": f"{d}/selfcheck/selfcheck.json", "gsuite": f"{R}/gsuite_result_ra.json", "pilot": f"{R}/pilot/PILOT_RH.json",
         "tree_done": f"{R}/done_ra/tree.done", "pair_log": f"{d}/pair_{tag}.log"}
    if tag.startswith("RA"):
        f["bank_json"] = f"{R}/bank/s{seed}/bank.json"
        f["bank_check"] = f"{R}/bank/s{seed}/BANK_CHECK.json"
    return f


def mint(a):
    tag, ro = a.tag, a.readout
    if tag not in ("RA_s0", "RA_s1") or not re.fullmatch(r"[0-9a-f]{40}", COMMIT):
        print("PROBLEM ra_gate_record.py serves RA_s0 / RA_s1 only, with a filled commit (fill_ra_commit.sh)")
        print("ZZ RH-GATE-RECORD REFUSED", tag)
        return 1
    F = files(tag)
    bad, bind = [], {}
    for k, p in F.items():
        if not os.path.isfile(p):
            bad.append(f"missing {k}: {p}")
        else:
            bind[k] = {"path": p, "sha256": sha(p)}
    if bad:
        return finish(tag, bad, bind, ro)
    argv = json.load(open(F["argv"]))["argv"]
    m = argv_map([str(t) for t in argv])
    gs = json.load(open(F["gsuite"]))
    if gs.get("new") != COMMIT and gs.get("new_commit") != COMMIT:
        bad.append(f"G-SUITE new commit is {gs.get('new_commit', gs.get('new'))!r}, not {COMMIT[:8]}")
    if gs.get("n_regressions") or gs.get("new_test_failures") or gs.get("missing_in_new"):
        bad.append("G-SUITE reads regressions / new-test failures / missing tests")
    pl = json.load(open(F["pilot"]))
    if pl.get("pass") is not True or pl.get("readout") != ro:
        bad.append(f"PILOT_RH.json pass={pl.get('pass')} readout={pl.get('readout')!r} (want pass, {ro})")
    if "ZZ ROLLHIST-PAIR PASS" not in open(F["pair_log"]).read():
        bad.append("the pair check did not print ZZ ROLLHIST-PAIR PASS")
    sc = json.load(open(F["selfcheck"]))
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"selfcheck declared_vs_built = {sc.get('declared_vs_built')!r}")
    if (sc.get("acc_probe") or {}).get("pass") is not True:
        bad.append("selfcheck acc_probe did not pass")
    levers_ro = set(map(str, find_key(sc.get("levers", {}), "dagger_bank_readout")))
    if tag.startswith("RA"):
        if m.get("--dagger-mode") != "add":
            bad.append(f"RA argv --dagger-mode is {m.get('--dagger-mode')!r}, not add")
        if "add" not in set(map(str, find_key(sc.get("levers", {}), "dagger_mode"))):
            bad.append("the built levers do not carry dagger_mode add")
        pr = sc.get("dagger_probe") or {}
        rows = [v for v in find_key(pr, "rows_appended") if isinstance(v, (int, float))]
        if not rows or max(rows) <= 0:
            bad.append(f"the dagger probe read rows_appended {rows!r}: no own row was appended to the batch")
        gd = [v for v in find_key(pr, "grad_diff_all") if isinstance(v, (int, float))]
        if not gd or max(gd) <= 0:
            bad.append(f"the dagger probe read grad_diff_all {gd!r}: the appended rows do not move the gradient")
        z = [v for v in find_key(pr, "control_p0_grad_diff") if isinstance(v, (int, float))]
        if not z or max(z) != 0.0:
            bad.append(f"the dagger probe control_p0_grad_diff reads {z!r}, not exactly 0.0")
        if m.get("--dagger-bank-readout") != ro or m.get("--dagger-bank-kappa") != "0.2":
            bad.append("RH argv does not name the declaration flags for this readout")
        try:
            if float(m.get("--dagger-p")) <= 0:
                bad.append("RH --dagger-p must be > 0")
        except (TypeError, ValueError):
            bad.append("RH --dagger-p unreadable")
        if (sc.get("dagger_probe") or {}).get("pass") is not True:
            bad.append("selfcheck dagger_probe did not pass (RH)")
        if ro not in levers_ro:
            bad.append(f"the built levers do not carry dagger_bank_readout {ro!r} (read {sorted(levers_ro)})")
        bc = json.load(open(F["bank_check"]))
        if bc.get("pass") is not True or bc.get("readout") != ro:
            bad.append("full-bank check json is not a pass for this readout")
        if m.get("--dagger-bank-sha256") != bind["bank_json"]["sha256"]:
            bad.append("argv --dagger-bank-sha256 is not the sha256 of bank.json")
    else:
        if str(m.get("--dagger-p")) not in ("0", "0.0"):
            bad.append("B0 --dagger-p is not 0")
        for f in ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed", "--dagger-bank-readout", "--dagger-bank-kappa"):
            if f in m:
                bad.append(f"B0 argv names {f}")
        if levers_ro - {"pinned", "None"}:
            bad.append(f"B0 built levers carry a non-default dagger_bank_readout {sorted(levers_ro)}")
    return finish(tag, bad, bind, ro)


def finish(tag, bad, bind, ro):
    d = f"{R}/runs/{tag}/gate"
    os.makedirs(d, exist_ok=True)
    rec = {"schema": "ra-chain-gate-record-v1", "tag": tag, "readout": ro, "commit": COMMIT, "pass": not bad, "problems": bad, "binds": bind,
           "note": "chain-local record; NOT a dscale10k_check / launch_gate token"}
    if not bad:
        json.dump(rec, open(f"{d}/RH_GATE_RECORD.json", "w"), indent=1)
    else:
        json.dump(rec, open(f"{d}/RH_GATE_REFUSED.json", "w"), indent=1)
    for p in bad:
        print("PROBLEM", p)
    print("ZZ RH-GATE-RECORD", "MINTED" if not bad else "REFUSED", tag)
    return 0 if not bad else 1


def verify(a):
    p = f"{R}/runs/{a.tag}/gate/RH_GATE_RECORD.json"
    if not os.path.isfile(p):
        print("ZZ RH-GATE-VERIFY REFUSED no record")
        return 1
    rec = json.load(open(p))
    bad = [k for k, v in rec["binds"].items() if not os.path.isfile(v["path"]) or sha(v["path"]) != v["sha256"]]
    if rec.get("commit") != COMMIT or not rec.get("pass"):
        bad.append("record not a pass for this commit")
    for k in bad:
        print("PROBLEM changed or missing since the mint:", k)
    print("ZZ RH-GATE-VERIFY", "OK" if not bad else "REFUSED", a.tag)
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="c", required=True)
    m = sp.add_parser("mint")
    m.add_argument("--tag", required=True)
    m.add_argument("--readout", required=True)
    m.set_defaults(f=mint)
    v = sp.add_parser("verify")
    v.add_argument("--tag", required=True)
    v.set_defaults(f=verify)
    a = ap.parse_args()
    sys.exit(a.f(a))
