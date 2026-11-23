#!/usr/bin/env python3
"""ki_gate_record.py -- The KI chain's launch-gate record for ONE KI stage-B arm (KI_s0 | KI_s1), K-INPUT (card PREREG_V8F_KINPUT, frozen sha256 05e786a8; commit 4097a89e, `--dagger-base own|logged`).
A COPY of rk_gate_record.py revision 2 (itself of ra_gate_record.py / rh_gate_record.py), by the RK kit's pattern. Differences from rk_gate_record.py, and only these:
  * COMMIT is the K-INPUT commit 4097a89e4b4a6c9bf309cbae9dc6a8dd1e574fcb; the G-SUITE result is gsuite_result_ki.json; the tree marker is done_ki/tree.done; any non-KI tag is refused;
  * THERE IS NO LABEL CACHE in KI (the label is the human's, RH's default): every cache check, the census bar, the kin_v1 label checks, the `precheck` of a cache and the cache binds are dropped;
  * KI arms take the RH branch plus the base-lever checks (card section 3, "lever-acts check"):
      - the argv names `--dagger-base logged` (exactly), `--dagger-bank` = $R/bank/s<seed> and `--dagger-bank-sha256` = the sha256 of that bank's bank.json (= the bank the RH arms trained on), and NO `--dagger-label`, `--dagger-mode`,
        `--dagger-label-cache` or `--dagger-kin-tb` (KI = RH + `--dagger-base logged`);
      - the selfcheck's BUILT levers carry `dagger_base` exactly {logged} and `dagger_label` exactly {human} (RH's label: the swap's labels are the human's);
      - the selfcheck's `dagger_probe` passes, reads `base_source` "logged", `rows_changed.base` exactly 0 (the logged base at r' is kept bit for bit on every selected row) and `n_selected_at_p1` > 0 (selected rows > 0);
      - the dagger probe's grad_diff_all > 0 and control_p0_grad_diff exactly 0.0 (as RK);
  * `precheck --tag KI_s0` is the PRE-LOCK check of the gate-record inputs that do not need the arm's selfcheck: the G-SUITE result at the commit, PILOT_RH.json, the bank check and bank sha (writes nothing).
Original docstring (RH_sk or B0_sk) of rh_gate_record.py: it is NOT a dscale10k_check.py / launch_gate.py token; it is the chain's own record, minted by the chain, never by hand, and verified again at the moment of launch.
stdlib only, CPU, sub-second.

  mint   --tag KI_s0 --readout TOKEN         writes $R/runs/<tag>/gate/RH_GATE_RECORD.json ONLY if every check below passes; exit 0/1
  precheck --tag KI_s0 --readout TOKEN [--argv PATH|-]   the pre-lock check of the inputs (G-SUITE, pilot, bank check, bank sha, argv); writes nothing
  verify --tag KI_s0                         re-hashes what the record binds (argv, bank.json, selfcheck, G-SUITE result, pilot report, tree marker) and re-derives the probe values; exit 0/1

Checks at mint: (1) tree blob check done-marker + commit; (2) G-SUITE result: 0 regressions, 0 new-test failures, 0 missing, new commit == COMMIT; (3) PILOT_RH.json pass and the same readout; (4) the full-bank check json passes for this readout
and the bank.json sha equals the argv's --dagger-bank-sha256; (5) the pair check printed PASS (pair_<tag>.log); (6) the trainer's own --selfcheck-only record of THIS argv: declared_vs_built PASS, acc_probe pass, dagger_probe pass, the declaration token
in the built levers; (7) the argv names what the arm is (the RH declaration flags, --dagger-p > 0, `--dagger-base logged`); (8) the base-lever checks above."""
import argparse
import re
import hashlib
import json
import os
import sys

R = os.environ.get("RH_ROOT", "/root/rollhist")
COMMIT = "4097a89e4b4a6c9bf309cbae9dc6a8dd1e574fcb"                                      # the commit of record (K-INPUT): 40 hex or this file refuses to run
BASE = "logged"
TAGS = ("KI_s0", "KI_s1")
NOT_IN_KI = ("--dagger-label", "--dagger-mode", "--dagger-label-cache", "--dagger-kin-tb")   # KI = RH + --dagger-base logged: none of the other lever flags


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


def nums(o, key):
    return [v for v in find_key(o, key) if isinstance(v, (int, float)) and not isinstance(v, bool)]


def probe_problems(sc):
    """The base-lever reads of the bound selfcheck (shared by mint and verify): (problems, values)."""
    bad = []
    pr = sc.get("dagger_probe") or {}
    if pr.get("pass") is not True:
        bad.append("selfcheck dagger_probe did not pass")
    bs = [str(v) for v in find_key(pr, "base_source")]
    if set(bs) != {BASE}:
        bad.append(f"the dagger probe read base_source {sorted(set(bs))}, not exactly [{BASE!r}]: the logged base at r' was not the one kept")
    rc = [v for v in find_key(pr, "rows_changed") if isinstance(v, dict)]
    base_rows = [v.get("base") for v in rc]
    if len(base_rows) != 1 or base_rows[0] != 0 or isinstance(base_rows[0], bool):
        bad.append(f"the dagger probe read rows_changed.base {base_rows!r}, not exactly [0]: `base` must be unchanged on every selected row under --dagger-base {BASE}")
    ns = nums(pr, "n_selected_at_p1")
    if not ns or max(ns) <= 0:
        bad.append(f"the dagger probe read n_selected_at_p1 {ns!r}: no row was selected, the swap cannot be seen")
    gd = nums(pr, "grad_diff_all")
    if not gd or max(gd) <= 0:
        bad.append(f"the dagger probe read grad_diff_all {gd!r}: the swapped rows do not move the gradient")
    z = nums(pr, "control_p0_grad_diff")
    if not z or max(z) != 0.0:
        bad.append(f"the dagger probe control_p0_grad_diff reads {z!r}, not exactly 0.0")
    lb = set(map(str, find_key(sc.get("levers", {}), "dagger_base")))
    if lb != {BASE}:
        bad.append(f"the built levers carry dagger_base {sorted(lb)}, not exactly [{BASE!r}]")
    ll = set(map(str, find_key(sc.get("levers", {}), "dagger_label")))
    if ll != {"human"}:
        bad.append(f"the built levers carry dagger_label {sorted(ll)}, not exactly ['human'] (KI keeps RH's human label)")
    return bad, {"base_source": sorted(set(bs)), "rows_changed_base": base_rows, "n_selected_at_p1": ns}


def files(tag):
    d = f"{R}/runs/{tag}"
    seed = tag.split("_s")[1]
    return {"argv": f"{d}/argv.json", "selfcheck": f"{d}/selfcheck/selfcheck.json", "gsuite": f"{R}/gsuite_result_ki.json", "pilot": f"{R}/pilot/PILOT_RH.json",
            "tree_done": f"{R}/done_ki/tree.done", "pair_log": f"{d}/pair_{tag}.log", "bank_json": f"{R}/bank/s{seed}/bank.json", "bank_check": f"{R}/bank/s{seed}/BANK_CHECK.json"}


def input_problems(tag, ro, m, bind):
    """The checks that need no selfcheck (shared by mint and precheck)."""
    bad = []
    seed = tag.split("_s")[1]
    gs = json.load(open(bind["gsuite"]["path"]))
    if gs.get("new") != COMMIT and gs.get("new_commit") != COMMIT:
        bad.append(f"G-SUITE new commit is {gs.get('new_commit', gs.get('new'))!r}, not {COMMIT[:8]}")
    if gs.get("n_regressions") or gs.get("new_test_failures") or gs.get("missing_in_new"):
        bad.append("G-SUITE reads regressions / new-test failures / missing tests")
    pl = json.load(open(bind["pilot"]["path"]))
    if pl.get("pass") is not True or pl.get("readout") != ro:
        bad.append(f"PILOT_RH.json pass={pl.get('pass')} readout={pl.get('readout')!r} (want pass, {ro})")
    bc = json.load(open(bind["bank_check"]["path"]))
    if bc.get("pass") is not True or bc.get("readout") != ro:
        bad.append("full-bank check json is not a pass for this readout")
    if m.get("--dagger-bank-sha256") != bind["bank_json"]["sha256"]:
        bad.append("argv --dagger-bank-sha256 is not the sha256 of bank.json")
    if m.get("--dagger-base") != BASE:
        bad.append(f"KI argv --dagger-base is {m.get('--dagger-base')!r}, not {BASE}")
    for f in NOT_IN_KI:
        if f in m:
            bad.append(f"KI argv names {f} {m.get(f)!r}: KI is RH + --dagger-base {BASE}, nothing else")
    if m.get("--dagger-bank") != f"{R}/bank/s{seed}":
        bad.append(f"KI argv --dagger-bank is {m.get('--dagger-bank')!r}, not the RH bank {R}/bank/s{seed}")
    if m.get("--dagger-bank-readout") != ro or m.get("--dagger-bank-kappa") != "0.2":
        bad.append("KI argv does not name the declaration flags for this readout")
    try:
        if float(m.get("--dagger-p")) <= 0:
            bad.append("KI --dagger-p must be > 0")
    except (TypeError, ValueError):
        bad.append("KI --dagger-p unreadable")
    return bad


def mint(a):
    tag, ro = a.tag, a.readout
    if tag not in TAGS or not re.fullmatch(r"[0-9a-f]{40}", COMMIT):
        print("PROBLEM ki_gate_record.py serves KI_s0 / KI_s1 only, with a filled 40-hex commit")
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
        return finish(tag, bad, bind, ro, None)
    argv = json.load(open(F["argv"]))["argv"]
    m = argv_map([str(t) for t in argv])
    bad += input_problems(tag, ro, m, bind)
    if "ZZ ROLLHIST-PAIR PASS" not in open(F["pair_log"]).read():
        bad.append("the pair check did not print ZZ ROLLHIST-PAIR PASS")
    sc = json.load(open(F["selfcheck"]))
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"selfcheck declared_vs_built = {sc.get('declared_vs_built')!r}")
    if (sc.get("acc_probe") or {}).get("pass") is not True:
        bad.append("selfcheck acc_probe did not pass")
    levers_ro = set(map(str, find_key(sc.get("levers", {}), "dagger_bank_readout")))
    if ro not in levers_ro:
        bad.append(f"the built levers do not carry dagger_bank_readout {ro!r} (read {sorted(levers_ro)})")
    pb, vals = probe_problems(sc)
    bad += pb
    return finish(tag, bad, bind, ro, vals)


def precheck(a):
    """Pre-lock check of ONE seed's gate-record inputs (no selfcheck exists yet): the files, the G-SUITE result at the commit, the pilot, the bank check and the bank sha against the KI argv. Writes nothing."""
    tag, ro = a.tag, a.readout
    if tag not in TAGS:
        print("PROBLEM precheck serves KI_s0 / KI_s1 only")
        print("ZZ KI-INPUTS-PRECHECK REFUSED", tag)
        return 1
    F = files(tag)
    F.pop("selfcheck"), F.pop("pair_log"), F.pop("tree_done")                                  # not yet written before the lock (the tree marker is the chain's step; the selfcheck is the arm's)
    ext = getattr(a, "argv", None)                                                             # `--argv -` = the KI argv JSON on stdin (ki_argv.py json): the pre-lock walk before runs/KI_sK/argv.json exists
    if ext:
        F.pop("argv")
    bad, bind = [], {}
    for k, p in F.items():
        if not os.path.isfile(p):
            bad.append(f"missing {k}: {p}")
        else:
            bind[k] = {"path": p, "sha256": sha(p)}
    if not bad:
        try:
            o = json.load(sys.stdin if ext == "-" else open(ext or F["argv"], encoding="utf-8"))
            argv = o["argv"] if isinstance(o, dict) else o
        except (ValueError, KeyError, OSError) as e:
            argv, bad = [], bad + [f"the argv is unreadable: {type(e).__name__}: {str(e)[:120]}"]
        if not bad:
            bad += input_problems(tag, ro, argv_map([str(t) for t in argv]), bind)
    for q in bad:
        print("PROBLEM", q)
    print("ZZ KI-INPUTS-PRECHECK", "OK" if not bad else "REFUSED", tag)
    return 0 if not bad else 1


def finish(tag, bad, bind, ro, vals):
    d = f"{R}/runs/{tag}/gate"
    os.makedirs(d, exist_ok=True)
    rec = {"schema": "ki-chain-gate-record-v1", "tag": tag, "readout": ro, "commit": COMMIT, "base": BASE, "pass": not bad, "problems": bad, "binds": bind, "probe": vals,
           "note": "chain-local record; NOT a dscale10k_check / launch_gate token"}
    if not bad:
        json.dump(rec, open(f"{d}/RH_GATE_RECORD.json", "w"), indent=1)
        print(f"probe: base_source {vals['base_source']} rows_changed.base {vals['rows_changed_base']} n_selected_at_p1 {vals['n_selected_at_p1']}")
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
    if rec.get("commit") != COMMIT or not rec.get("pass") or rec.get("base") != BASE or a.tag not in TAGS or rec.get("tag") != a.tag:
        bad.append("record not a pass for this commit, base and tag")
    sp = (rec.get("binds", {}).get("selfcheck") or {}).get("path")
    if sp and os.path.isfile(sp):
        pb, vals = probe_problems(json.load(open(sp)))
        if pb or rec.get("probe") != vals:
            bad.append(f"probe: the bound selfcheck reads {vals!r} (problems {pb}), the record {rec.get('probe')!r}")
    else:
        bad.append("probe: no bound selfcheck")
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
    pc = sp.add_parser("precheck")
    pc.add_argument("--tag", required=True)
    pc.add_argument("--readout", required=True)
    pc.add_argument("--argv", default=None, help="a KI argv JSON path, or - for stdin (default: runs/<tag>/argv.json)")
    pc.set_defaults(f=precheck)
    a = ap.parse_args()
    sys.exit(a.f(a))
