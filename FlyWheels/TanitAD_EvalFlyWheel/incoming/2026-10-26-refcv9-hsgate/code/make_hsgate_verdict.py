"""make_hsgate_verdict.py -- builds ship_r8_hsgate_verdict.py from ship_r8_sub/ship_r8_4c_verdict.py (READ-ONLY) by anchored replacements, each asserted to
match EXACTLY ONCE, + the unified diff (ship_r8_hsgate_verdict.DIFF.txt).  PREREG_HSGATE G3: SHIP-R8 (1)-(7)+(1b) on HSGATE8, Amendment 4's clauses with the
arm swapped.  Clause (1) for HSGATE8 = Amendment 4's clause-1 function (UNEDITED, run on the --hsgate forward) AND the PREREG_HSGATE seal re-hashed now == the
forward's record AND the gate file's sha256 now == REGISTERED_HSGATE.sha256 == the forward's record AND K_REG_HSGATE, K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT,
K_LANDED, K_GATE_SURF, K_TRACE AND both SELFSHARP/os-only input traces re-hashed now == the forward's record.  (1b) = Amendment 4's (1b) (the battery rolls'
registered FEED_PARITY) AND Amendment 5's (1b) (the os-only + SELFSHARP rolls' own manifests perframe_full): the FACTORED half of HSGATE8 rides on them.
"""
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub/ship_r8_4c_verdict.py")
DST = HERE / "ship_r8_hsgate_verdict.py"
DIFF = HERE / "ship_r8_hsgate_verdict.DIFF.txt"
R = []


def rep(old, new, count=1):
    R.append((old, new, count))


rep('"""ship_r8_4c_verdict.py -- SHIP-R8 AMENDMENT 4c: a COPY of',
    '"""ship_r8_hsgate_verdict.py -- PREREG_HSGATE (2331e1f3...) bar G3: a COPY of ship_r8_sub/ship_r8_4c_verdict.py (left untouched; built by\n'
    'make_hsgate_verdict.py, diff ship_r8_hsgate_verdict.DIFF.txt) with the arm HSGATE8 added.  Clause (1) for HSGATE8 = Amendment 4\'s clause-1 function\n'
    '(UNEDITED, run on the --hsgate forward) AND the PREREG_HSGATE seal re-hashed now == the forward\'s record AND the gate file sha256 now ==\n'
    'REGISTERED_HSGATE.sha256 == the forward\'s record AND K_REG_HSGATE, K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE AND the A5 input\n'
    'traces re-hashed now == the forward\'s record.  (1b) = Amendment 4\'s (1b) AND Amendment 5\'s (1b).  Writes only under refcv9_hsgate.\n\n'
    'ship_r8_4c_verdict.py -- SHIP-R8 AMENDMENT 4c: a COPY of')
rep('HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE))\nimport ship_r8_sub_common as C  # noqa: E402\n',
    'HERE = Path(__file__).resolve().parent\nSUBDIR = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub")                # READ-ONLY\n'
    'sys.path.insert(0, str(SUBDIR))\nsys.path.insert(0, str(HERE))\nimport ship_r8_sub_common as C  # noqa: E402\nimport hsgate_common as HG  # noqa: E402\n')
rep('CL1B_A5 = "(1b) [AMENDMENT 5] FEED PARITY: the os-only rolls use the per-frame feed (perframe_full), asserted from the rolls\' own manifests; not shown -> INCOMPLETE"\n',
    'CL1B_A5 = "(1b) [AMENDMENT 5] FEED PARITY: the os-only rolls use the per-frame feed (perframe_full), asserted from the rolls\' own manifests; not shown -> INCOMPLETE"\n'
    'CL1_TEXT["HSGATE8"] = ("(1) [PREREG_HSGATE] everything Amendment 4\'s clause 1 states (H2cn stage sha256 == registered; registration precedes the first eval-side "\n'
    '                       "output; fresh-clip disjoint; H2CN_PERSURFACE and RETIME_DEPLOY8 identities) AND the PREREG_HSGATE seal == its file (re-hashed now) AND the "\n'
    '                       "gate sha256 == REGISTERED_HSGATE.sha256 (re-hashed now) and its registration precedes the first eval-side output AND K_FIT_HSGATE, K_ID0, "\n'
    '                       "K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE AND the Amendment 5 input traces unchanged since the forward recorded them")\n'
    'CL1B_HSG = "(1b) [PREREG_HSGATE] FEED PARITY of BOTH halves: Amendment 4\'s (1b) (the registered FEED_PARITY of the battery rolls) AND Amendment 5\'s (1b) (the os-only and SELFSHARP rolls\' own manifests perframe_full)"\n'
    'from deviations_hsgate import DEVIATIONS as DEVIATIONS_HSG  # noqa: E402   (confirmed by the Master Mind in the PHASE 2 GO)\n')
rep('def overall(clauses) -> str:',
    'def clause1_hsgate(fd, rid) -> dict:\n'
    '    """PREREG_HSGATE clause 1 (see the module docstring).  Amendment 4\'s own conjunction is NOT edited; its status/reason are carried into this one."""\n'
    '    base = clause1_a4(fd, rid)\n'
    '    ctl = (fd or {}).get("controls") or {}\n'
    '    bad, unk = [], []\n'
    '    if base["status"] == "FAIL":\n'
    '        bad.append("Amendment 4 clause-1 conjunction: " + base["reason"])\n'
    '    elif base["status"] != "PASS":\n'
    '        unk.append("Amendment 4 clause-1 conjunction: " + base["reason"])\n'
    '    now = HG.check_prereg()                                             # SystemExit if the sealed text is not the file (never a silent pass)\n'
    '    rec = ((fd or {}).get("amendments") or {}).get("PREREG_HSGATE") or {}\n'
    '    hrec = (fd or {}).get("hsgate") or {}\n'
    '    gfile, regf = HG.WORK / "HSGATE_FROZEN.json", HG.WORK / "REGISTERED_HSGATE.sha256"\n'
    '    reg_tok = C.read_seal(regf)\n'
    '    gate_now = C.sha256_file(gfile) if gfile.exists() else ""\n'
    '    ev = dict(base["evidence"])\n'
    '    ev.update(prereg_seal_now=now, prereg_seal_forward_record=rec, gate_sha256_now=gate_now, gate_registered_sha256=reg_tok, gate_forward_record=hrec.get("gate_sha256"))\n'
    '    if fd is None:\n'
    '        unk.append("the forward file is absent")\n'
    '    elif not hrec:\n'
    '        bad.append("the forward is not an --hsgate run")\n'
    '    if not rec:\n'
    '        unk.append("the forward carries no PREREG_HSGATE seal record")\n'
    '    elif rec.get("sha256") != now["sha256"]:\n'
    '        bad.append("PREREG_HSGATE hashed now != the sha256 the forward recorded")\n'
    '    if len(reg_tok) != 64 or len(gate_now) != 64:\n'
    '        unk.append("the registered gate sha256 or the gate file is not readable as 64 hex")\n'
    '    elif not (gate_now == reg_tok == hrec.get("gate_sha256") == hrec.get("registered")):\n'
    '        bad.append("the gate file / forward record / registration disagree on the sha256")\n'
    '    for nm, key in (("K_REG_HSGATE (gate registration precedes the first eval-side output)", "K_REG_HSGATE"), ("K_FIT_HSGATE (fit controls ALL_PASS)", "K_FIT_HSGATE"),\n'
    '                    ("K_ID0 (g := 0 reproduces H2CN_DEPLOY8 bit for bit)", "K_ID0"), ("K_ID1 (g := 1 reproduces FACTORED_DEPLOY8 bit for bit)", "K_ID1"),\n'
    '                    ("K_EVALFEAT (eval features from the K-LABSRC extras keys; joins exact, mutation RED)", "K_EVALFEAT"),\n'
    '                    ("K_LANDED (H2CN / FACTORED positions == the landed packages)", "K_LANDED"), ("K_GATE_SURF (S2 / S6 decisions agree)", "K_GATE_SURF"),\n'
    '                    ("K_TRACE (Amendment 5 traces final / 4,754 / counters)", "K_TRACE")):\n'
    '        v = ctl.get(key)\n'
    '        ev[key] = v\n'
    '        if v is False:\n'
    '            bad.append(nm + " is false")\n'
    '        elif v is not True:\n'
    '            unk.append(nm + " not shown")\n'
    '    trs = {}\n'
    '    for s, r0 in ((ctl.get("K_TRACE_detail") or {}).get("inputs") or {}).items():\n'
    '        for nm in ("l1_base", "selfsharp"):\n'
    '            r = r0.get(nm) or {}\n'
    '            f = r.get("file")\n'
    '            nw = C.sha256_file(f) if f and Path(f).exists() else None\n'
    '            trs[f"s{s}|{nm}"] = {"sha256_forward_record": r.get("sha256"), "sha256_now": nw}\n'
    '            if r.get("sha256") is None or nw is None:\n'
    '                unk.append(f"s{s} {nm}: sha256 not recorded / file unreadable")\n'
    '            elif nw != r["sha256"]:\n'
    '                bad.append(f"s{s} {nm}: the trace changed since it was recorded")\n'
    '    if not trs:\n'
    '        unk.append("no input-trace record in the forward")\n'
    '    ev["traces"] = trs\n'
    '    ev["K_ID_detail"] = ctl.get("K_ID_detail")\n'
    '    ev["K_REG_HSGATE_detail"] = ctl.get("K_REG_HSGATE_detail")\n'
    '    st = _status(bad, unk)\n'
    '    return {"id": 1, "text": CL1_TEXT["HSGATE8"], "status": st, "reason": "; ".join(bad + unk) if (bad or unk) else\n'
    '            "Amendment 4\'s clause-1 conjunction holds on the --hsgate forward; PREREG_HSGATE seal == its file (re-hashed now) and == the forward\'s record; the gate "\n'
    '            "sha256 == registered (re-hashed now); K_REG_HSGATE, K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE all true; traces unchanged",\n'
    '            "evidence": ev}\n'
    '\n\n'
    'def clause1b_hsgate(reg_ship, fd) -> dict:\n'
    '    a, b = clause1b_a4(reg_ship), clause1b_a5(fd)\n'
    '    sts = (a["status"], b["status"])\n'
    '    st = "FAIL" if "FAIL" in sts else ("INCOMPLETE" if "INCOMPLETE" in sts else "PASS")\n'
    '    return {"id": "1b", "text": CL1B_HSG, "status": st, "reason": f"A4 (1b): {a[\'status\']} -- {a[\'reason\']} || A5 (1b): {b[\'status\']} -- {b[\'reason\']}",\n'
    '            "evidence": {"A4_1b": a, "A5_1b": b}}\n'
    '\n\n'
    'def overall(clauses) -> str:')
rep('choices=("H2CN_DEPLOY8", "H2CN_CAP8", "FACTORED_DEPLOY8", "H2CNL_DEPLOY8"))',
    'choices=("H2CN_DEPLOY8", "H2CN_CAP8", "FACTORED_DEPLOY8", "H2CNL_DEPLOY8", "HSGATE8"))')
rep('    amend = {"H2CN_DEPLOY8": "4", "H2CN_CAP8": "4b", "FACTORED_DEPLOY8": "5", "H2CNL_DEPLOY8": "4c"}[arm]\n    h2 = amend in ("4", "4b", "4c")\n',
    '    amend = {"H2CN_DEPLOY8": "4", "H2CN_CAP8": "4b", "FACTORED_DEPLOY8": "5", "H2CNL_DEPLOY8": "4c", "HSGATE8": "HSG"}[arm]\n    h2 = amend in ("4", "4b", "4c", "HSG")\n'
    '    if amend == "HSG":\n'
    '        HG.check_prereg()                                                 # refuses before anything is read\n'
    '        if fd is None or not fd.get("hsgate"):\n'
    '            raise SystemExit("[hsgate-verdict] HSGATE8 needs an --hsgate forward")\n'
    '        rod = od.resolve()\n'
    '        if not (rod == HG.WORK.resolve() or HG.WORK.resolve() in rod.parents):\n'
    '            raise SystemExit(f"[hsgate-verdict] REFUSING: --out-dir {od} is outside {HG.WORK}")\n')
rep('    c1 = clause1_4b(fd, rid) if amend == "4b" else (clause1_a4(fd, rid) if amend == "4" else (clause1_4c(fd, rid, arm) if amend == "4c" else clause1_a5(fd, rid, reg_ship, pins)))\n'
    '    c1b = clause1b_a4(reg_ship) if h2 else clause1b_a5(fd)\n',
    '    c1 = clause1_hsgate(fd, rid) if amend == "HSG" else (clause1_4b(fd, rid) if amend == "4b" else (clause1_a4(fd, rid) if amend == "4" else (clause1_4c(fd, rid, arm) if amend == "4c" else clause1_a5(fd, rid, reg_ship, pins))))\n'
    '    c1b = clause1b_hsgate(reg_ship, fd) if amend == "HSG" else (clause1b_a4(reg_ship) if h2 else clause1b_a5(fd))\n')
rep('"4c": ("H2CN_DEPLOY8", "H2CNL_MUT8", "H2CN_NEAR", "H2CN_PERSURFACE", "RETIME_DEPLOY8")}[amend]',
    '"4c": ("H2CN_DEPLOY8", "H2CNL_MUT8", "H2CN_NEAR", "H2CN_PERSURFACE", "RETIME_DEPLOY8"),\n'
    '                   "HSG": ("H2CN_DEPLOY8", "FACTORED_DEPLOY8", "RETIME_DEPLOY8")}[amend]')
rep('    if amend == "4b":\n        bes["clamp_touched_windows"]',
    '    if amend == "HSG":\n'
    '        bes["hsgate_F_rate"] = ((fd or {}).get("hsgate") or {}).get("F_rate")\n'
    '        bes["hsgate_per_window"] = {k: {kk: vv for kk, vv in (v.get("hsgate") or {}).items() if kk not in ("join", "join_mutation_next_window", "landed")}\n'
    '                                    for k, v in ((fd or {}).get("per_window_extra") or {}).items()}\n'
    '    if amend == "4b":\n        bes["clamp_touched_windows"]')
rep('    elif amend == "4c":\n        fails = [str(c["id"]) for c in clauses if c["status"] == "FAIL"]\n',
    '    elif amend == "HSG":\n'
    '        fails = [str(c["id"]) for c in clauses if c["status"] == "FAIL"]\n'
    '        nxt = ("this verdict is bar G3 of PREREG_HSGATE (SHIP-R8 OVERALL " + ov + (": clause(s) " + ", ".join(fails) + " named" if fails else "") +\n'
    '               "); the committed next step reads G1 + G2 + G3 + G4 together -> result/HSGATE_VERDICT.json (hsgate_bars.py)")\n'
    '    elif amend == "4c":\n        fails = [str(c["id"]) for c in clauses if c["status"] == "FAIL"]\n')
rep('"tool": "ship_r8_4c_verdict.py"', '"tool": "ship_r8_hsgate_verdict.py"')
rep('    if amend == "4c":\n        v["disclosed_deviations"] = DEVIATIONS_4C\n',
    '    if amend == "4c":\n        v["disclosed_deviations"] = DEVIATIONS_4C\n    if amend == "HSG":\n        v["disclosed_deviations"] = DEVIATIONS_HSG\n')


def main():
    src = SRC.read_text(encoding="utf-8")
    out = src
    for old, new, count in R:
        n = out.count(old)
        if n != count:
            raise SystemExit(f"anchor matched {n} times (want {count}): {old[:90]!r}")
        out = out.replace(old, new)
    DST.write_text(out, encoding="utf-8", newline="\n")
    DIFF.write_text("".join(difflib.unified_diff(src.splitlines(True), out.splitlines(True), fromfile="ship_r8_sub/ship_r8_4c_verdict.py",
                                                 tofile="ship_r8_hsgate_verdict.py")), encoding="utf-8", newline="\n")
    print(f"wrote {DST} ({len(out)} bytes) + {DIFF}; {len(R)} anchored replacements")


if __name__ == "__main__":
    main()
