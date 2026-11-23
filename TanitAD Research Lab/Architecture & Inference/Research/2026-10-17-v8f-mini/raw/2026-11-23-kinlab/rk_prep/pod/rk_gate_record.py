#!/usr/bin/env python3
"""rk_gate_record.py -- REVISION 2. The RK chain's launch-gate record for ONE RK stage-B arm (RK_s0 | RK_s1), KINLAB (card PREREG_V8F_KINLAB + Amendment 1, commit 1a8b11a1).
Revision 2 adds, and only adds (the rev1 checks all stay): the LABEL-CACHE checks of Amendment 1 section 4 --
  * the argv's `--dagger-label-cache` equals exactly $R/kin_cache/s<K>_kin_v1.npz (this seed's bank, never the other seed's);
  * that file exists and is BOUND (sha256 re-hashed at `verify`), as is the precompute's selfcheck $R/kin_cache/s<K>_selfcheck_cpu.json;
  * the selfcheck's BUILT levers `dagger_label_cache_sha256` equal the cache file's sha256 NOW (and `dagger_label_cache` equals the argv path); the CPU precompute's levers carry the same sha;
  * the full-bank census read from the bound selfcheck equals the CPU precompute's census (same counts), so the bar of 0.5 is read from the bank the cache was computed on;
  * the G-SUITE result and the record's commit are 1a8b11a1. A COPY of ra_gate_record.py (itself of rh_gate_record.py), by the RA kit's pattern.
Differences from ra_gate_record.py, and only these:
  * COMMIT is the KINLAB commit 0ca6f49dd076c035c1cd42da5e4e89a6f6961b9a; the G-SUITE result is gsuite_result_rk.json; the tree marker is done_rk/tree.done; any non-RK tag is refused;
  * RK arms take the RH branch plus the kinematic-label checks, in place of RA's add-mode checks:
      - the argv names `--dagger-label kin_v1`, `--dagger-bank` = $R/bank/s<seed> and `--dagger-bank-sha256` = the sha256 of that bank's bank.json (= the bank the RH and RL arms trained on), and no `--dagger-mode`;
      - the selfcheck's BUILT levers carry `dagger_label` kin_v1 (and no other value);
      - `dagger_probe.n_label_kin` > 0 (the kin label reached the probe batch) and `dagger_probe.rl_label_rows_changed` > 0;
      - THE CENSUS BAR of card section 4: from `dagger_probe.kin_status_counts` (the FULL bank's per-reason counts over its ok records) the relabelled share of ok records = counts["kin_emitted"] / sum(counts) is >= 0.5.
        The share and the counts are written into the gate record ("census"); `verify` recomputes the share from the bound selfcheck and refuses a record whose share is < 0.5 or differs;
      - the dagger probe's grad_diff_all > 0 and control_p0_grad_diff exactly 0.0 (as RA).
Original docstring (RH_sk or B0_sk) of rh_gate_record.py: it is NOT a dscale10k_check.py / launch_gate.py token; it is the chain's own record, minted by the chain, never by hand, and verified again at the moment of launch.
stdlib only, CPU, sub-second.

  mint   --tag RK_s0 --readout TOKEN         writes $R/runs/<tag>/gate/RH_GATE_RECORD.json ONLY if every check below passes; exit 0/1
  precheck --tag RK_s0                       (revision 2) the pre-lock check of the label cache + the CPU precompute's selfcheck; writes nothing
  verify --tag RK_s0                         re-hashes what the record binds (argv, bank.json, selfcheck, G-SUITE result, pilot report, tree marker) and re-derives the census share; exit 0/1

Checks at mint: (1) tree blob check done-marker + commit; (2) G-SUITE result: 0 regressions, 0 new-test failures, 0 missing, new commit == COMMIT; (3) PILOT_RH.json pass and the same readout; (4) the full-bank check json passes for this readout
and the bank.json sha equals the argv's --dagger-bank-sha256; (5) the pair check printed PASS (pair_<tag>.log); (6) the trainer's own --selfcheck-only record of THIS argv: declared_vs_built PASS, acc_probe pass, dagger_probe pass, the declaration token
in the built levers; (7) the argv names what the arm is (the RH declaration flags, --dagger-p > 0, the kin_v1 label); (8) the kin label checks and the census bar above."""
import argparse
import re
import hashlib
import json
import os
import sys

R = os.environ.get("RH_ROOT", "/root/rollhist")
COMMIT = "1a8b11a1abd50c09636da8b7425e97573cdf1d04"                                      # the commit of record (KINLAB + the label cache): 40 hex or this file refuses to run
LABEL = "kin_v1"
CENSUS_BAR = 0.5                                                                         # card section 4: KINLAB-BLOCKED if the relabelled share of ok records is below 0.5 at either seed's bank


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


def census(sc):
    """(share, n_kin, n_ok, counts) from the selfcheck's dagger_probe.kin_status_counts; share None when the counts are missing / empty / not integers."""
    cs = [v for v in find_key(sc.get("dagger_probe") or {}, "kin_status_counts") if isinstance(v, dict)]
    if len(cs) != 1 or not cs[0] or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in cs[0].values()):
        return None, 0, 0, cs[0] if len(cs) == 1 else None
    c = cs[0]
    n_ok = sum(c.values())                                                               # the labeller visits the ok records only: n_done = their count = the sum of the per-reason counts
    n_kin = int(c.get("kin_emitted", 0))
    return (n_kin / n_ok if n_ok else None), n_kin, n_ok, c


def cache_path(seed):
    return f"{R}/kin_cache/s{seed}_kin_v1.npz"


def files(tag):
    d = f"{R}/runs/{tag}"
    seed = tag.split("_s")[1]
    f = {"argv": f"{d}/argv.json", "selfcheck": f"{d}/selfcheck/selfcheck.json", "gsuite": f"{R}/gsuite_result_rk.json", "pilot": f"{R}/pilot/PILOT_RH.json",
         "tree_done": f"{R}/done_rk/tree.done", "pair_log": f"{d}/pair_{tag}.log"}
    if tag.startswith("RK"):
        f["label_cache"] = cache_path(seed)
        f["cache_selfcheck"] = f"{R}/kin_cache/s{seed}_selfcheck_cpu.json"
        f["bank_json"] = f"{R}/bank/s{seed}/bank.json"
        f["bank_check"] = f"{R}/bank/s{seed}/BANK_CHECK.json"
    return f


def mint(a):
    tag, ro = a.tag, a.readout
    if tag not in ("RK_s0", "RK_s1") or not re.fullmatch(r"[0-9a-f]{40}", COMMIT):
        print("PROBLEM rk_gate_record.py serves RK_s0 / RK_s1 only, with a filled 40-hex commit")
        print("ZZ RH-GATE-RECORD REFUSED", tag)
        return 1
    seed = tag.split("_s")[1]
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
    # -- the RK branch: the RH checks, then the kinematic-label checks
    if m.get("--dagger-label") != LABEL:
        bad.append(f"RK argv --dagger-label is {m.get('--dagger-label')!r}, not {LABEL}")
    if "--dagger-mode" in m:
        bad.append(f"RK argv names --dagger-mode {m.get('--dagger-mode')!r}: RK is a swap-mode arm")
    if m.get("--dagger-bank") != f"{R}/bank/s{seed}":
        bad.append(f"RK argv --dagger-bank is {m.get('--dagger-bank')!r}, not the RH bank {R}/bank/s{seed}")
    lv = set(map(str, find_key(sc.get("levers", {}), "dagger_label")))
    if lv != {LABEL}:
        bad.append(f"the built levers carry dagger_label {sorted(lv)}, not exactly [{LABEL!r}]")
    pr = sc.get("dagger_probe") or {}
    nk = [v for v in find_key(pr, "n_label_kin") if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nk or max(nk) <= 0:
        bad.append(f"the dagger probe read n_label_kin {nk!r}: no kin_v1 label reached the probe batch")
    rr = [v for v in find_key(pr, "rl_label_rows_changed") if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not rr or max(rr) <= 0:
        bad.append(f"the dagger probe read rl_label_rows_changed {rr!r}: the label does not change the batch's gt")
    gd = [v for v in find_key(pr, "grad_diff_all") if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not gd or max(gd) <= 0:
        bad.append(f"the dagger probe read grad_diff_all {gd!r}: the swapped rows do not move the gradient")
    z = [v for v in find_key(pr, "control_p0_grad_diff") if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not z or max(z) != 0.0:
        bad.append(f"the dagger probe control_p0_grad_diff reads {z!r}, not exactly 0.0")
    # -- Amendment 1 section 4: the label cache. The file is bound (a missing one refused above, as "missing label_cache"); the built levers must read ITS sha256 now
    want_cache = cache_path(seed)
    if m.get("--dagger-label-cache") != want_cache:
        bad.append(f"RK argv --dagger-label-cache is {m.get('--dagger-label-cache')!r}, not exactly {want_cache!r} (this seed's bank's cache)")
    csha = bind["label_cache"]["sha256"]
    lev_sha = [str(v) for v in find_key(sc.get("levers", {}), "dagger_label_cache_sha256")]
    if set(lev_sha) != {csha}:
        bad.append(f"the built levers carry dagger_label_cache_sha256 {sorted(set(lev_sha))}, not exactly the sha256 of {want_cache} now ({csha}): the cache read is not the file on disk")
    lev_path = [str(v) for v in find_key(sc.get("levers", {}), "dagger_label_cache")]
    if set(lev_path) != {want_cache}:
        bad.append(f"the built levers carry dagger_label_cache {sorted(set(lev_path))}, not exactly [{want_cache!r}]")
    csc = json.load(open(F["cache_selfcheck"]))
    cl = [str(v) for v in find_key(csc.get("levers", {}), "dagger_label_cache_sha256")]
    if set(cl) != {csha}:
        bad.append(f"the CPU precompute's selfcheck carries dagger_label_cache_sha256 {sorted(set(cl))}, not the cache file's {csha}: it is not the run that wrote this cache")
    share_c, _, n_ok_c, counts_c = census(csc)
    share, n_kin, n_ok, counts = census(sc)
    if counts_c != counts or share_c != share:
        bad.append(f"census: the CPU precompute's kin_status_counts {counts_c!r} differ from the bound selfcheck's {counts!r}")
    if share is None:
        bad.append(f"the dagger probe has no usable kin_status_counts ({counts!r}): the census bar cannot be read")
    elif share < CENSUS_BAR:
        bad.append(f"CENSUS BAR: the relabelled share of ok records is {share:.4f} ({n_kin} of {n_ok}), below {CENSUS_BAR} (card section 4: KINLAB-BLOCKED, never tuned)")
    cen = {"relabelled_share_of_ok": share, "n_kin_emitted": n_kin, "n_ok": n_ok, "status_counts": counts, "bar": CENSUS_BAR}
    if m.get("--dagger-bank-readout") != ro or m.get("--dagger-bank-kappa") != "0.2":
        bad.append("RK argv does not name the declaration flags for this readout")
    try:
        if float(m.get("--dagger-p")) <= 0:
            bad.append("RK --dagger-p must be > 0")
    except (TypeError, ValueError):
        bad.append("RK --dagger-p unreadable")
    if pr.get("pass") is not True:
        bad.append("selfcheck dagger_probe did not pass")
    if ro not in levers_ro:
        bad.append(f"the built levers do not carry dagger_bank_readout {ro!r} (read {sorted(levers_ro)})")
    bc = json.load(open(F["bank_check"]))
    if bc.get("pass") is not True or bc.get("readout") != ro:
        bad.append("full-bank check json is not a pass for this readout")
    if m.get("--dagger-bank-sha256") != bind["bank_json"]["sha256"]:
        bad.append("argv --dagger-bank-sha256 is not the sha256 of bank.json")
    return finish(tag, bad, bind, ro, cen)


def precheck(a):
    """Pre-lock check of ONE seed's label cache (Amendment 1 section 3): the file and the CPU precompute's selfcheck exist; the selfcheck's built levers read THIS file's sha256 now and the cache's path; its census
    (the full bank's kin_status_counts) clears the bar. Writes nothing. The chain runs it before the lock; the gate record (mint) repeats every check on the selfcheck of the arm under the lock."""
    tag = a.tag
    if tag not in ("RK_s0", "RK_s1"):
        print("PROBLEM precheck serves RK_s0 / RK_s1 only")
        print("ZZ RK-CACHE-PRECHECK REFUSED", tag)
        return 1
    seed = tag.split("_s")[1]
    p, sp = cache_path(seed), f"{R}/kin_cache/s{seed}_selfcheck_cpu.json"
    bad = [f"missing {k}: {q}" for k, q in (("label_cache", p), ("cache_selfcheck", sp)) if not os.path.isfile(q)]
    if not bad:
        csha = sha(p)
        csc = json.load(open(sp))
        cl = [str(v) for v in find_key(csc.get("levers", {}), "dagger_label_cache_sha256")]
        if set(cl) != {csha}:
            bad.append(f"the CPU precompute's selfcheck carries dagger_label_cache_sha256 {sorted(set(cl))}, not the cache file's {csha} now: the file changed since the run that wrote it")
        lp = [str(v) for v in find_key(csc.get("levers", {}), "dagger_label_cache")]
        if set(lp) != {p}:
            bad.append(f"the CPU precompute's selfcheck carries dagger_label_cache {sorted(set(lp))}, not exactly [{p!r}]")
        lv = set(map(str, find_key(csc.get("levers", {}), "dagger_label")))
        if lv != {LABEL}:
            bad.append(f"the CPU precompute's built levers carry dagger_label {sorted(lv)}, not exactly [{LABEL!r}]")
        share, n_kin, n_ok, counts = census(csc)
        if share is None:
            bad.append(f"the CPU precompute's selfcheck has no usable kin_status_counts ({counts!r})")
        elif share < CENSUS_BAR:
            bad.append(f"CENSUS BAR: the relabelled share of ok records is {share:.4f} ({n_kin} of {n_ok}), below {CENSUS_BAR} (card section 4: KINLAB-BLOCKED, never tuned)")
        else:
            print(f"cache {p} sha256 {csha}; census {share:.4f} ({n_kin} of {n_ok}), bar {CENSUS_BAR}")
    for q in bad:
        print("PROBLEM", q)
    print("ZZ RK-CACHE-PRECHECK", "OK" if not bad else "REFUSED", tag)
    return 0 if not bad else 1


def finish(tag, bad, bind, ro, cen):
    d = f"{R}/runs/{tag}/gate"
    os.makedirs(d, exist_ok=True)
    rec = {"schema": "rk-chain-gate-record-v1", "tag": tag, "readout": ro, "commit": COMMIT, "label": LABEL, "pass": not bad, "problems": bad, "binds": bind, "census": cen, "label_cache_sha256": (bind.get("label_cache") or {}).get("sha256"),
           "note": "chain-local record; NOT a dscale10k_check / launch_gate token"}
    if not bad:
        json.dump(rec, open(f"{d}/RH_GATE_RECORD.json", "w"), indent=1)
        print(f"census: relabelled share of ok records {cen['relabelled_share_of_ok']:.4f} ({cen['n_kin_emitted']} of {cen['n_ok']}), bar {CENSUS_BAR}")
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
    if rec.get("commit") != COMMIT or not rec.get("pass") or rec.get("label") != LABEL or a.tag not in ("RK_s0", "RK_s1") or rec.get("tag") != a.tag:
        bad.append("record not a pass for this commit, label and tag")
    sp = (rec.get("binds", {}).get("selfcheck") or {}).get("path")
    if sp and os.path.isfile(sp):
        share, n_kin, n_ok, _ = census(json.load(open(sp)))
        c = rec.get("census") or {}
        if share is None or share < CENSUS_BAR or c.get("relabelled_share_of_ok") != share or c.get("n_ok") != n_ok:
            bad.append(f"census: the bound selfcheck reads share {share!r}, the record {c.get('relabelled_share_of_ok')!r} (bar {CENSUS_BAR})")
    else:
        bad.append("census: no bound selfcheck")
    cp = (rec.get("binds", {}).get("label_cache") or {}).get("path")
    if not cp or cp != cache_path(a.tag.split("_s")[1]) or rec.get("label_cache_sha256") != (rec.get("binds", {}).get("label_cache") or {}).get("sha256"):
        bad.append("label cache: the record does not bind this seed's cache")
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
    pc.set_defaults(f=precheck)
    a = ap.parse_args()
    sys.exit(a.f(a))
