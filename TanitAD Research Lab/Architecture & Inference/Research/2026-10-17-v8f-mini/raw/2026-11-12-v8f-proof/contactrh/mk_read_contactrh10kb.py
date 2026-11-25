"""mk_read_contactrh10kb.py -- writes read_contactrh10kb.py (card PREREG_V8F_PROOF_ADD142) from the landed read_contactrh10k.py (card ADD141) by asserted text replacement.

Changes, all of card ADD142 sections 3 to 5: card and name prefix; run tags H_RQ50Kx / U_RQ50Kx; bootstrap seed 142; Q3 of record = v0 - v_L (field clos2), -c_src beside;
control c5 (-c_src paired difference 0 within 1e-9 and v0 - v_L not identical across the arms); control c0 (argv of every run equals the ADD142 queue's table; RH and UE dumps differ);
names SAME-READ / MISREAD / BOTH / UNRESOLVED with the minimum of 10 windows in X and in S.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
s = io.open(os.path.join(HERE, "read_contactrh10k.py"), encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:80])
    s = s.replace(a, b)


rep('CARD = "PREREG_V8F_PROOF_ADD141"', 'CARD = "PREREG_V8F_PROOF_ADD142"')
rep("import hashlib\nimport json\n", "import hashlib\nimport io\nimport json\n")
rep('PRE = "CONTACTRH10K"', 'PRE = "CONTACTRH10KB"')
rep('TAG_H, TAG_U = "H_RQ50K", "U_RQ50K"', 'TAG_H, TAG_U = "H_RQ50Kx", "U_RQ50Kx"')
rep('N_BOOT, BOOT_SEED = 2000, 141', 'N_BOOT, BOOT_SEED = 2000, 142')
rep('N_MISREAD, N_SAME, N_UNRES = PRE + "-MISREAD", PRE + "-SAME-READ", PRE + "-UNRESOLVED"',
    'N_MISREAD, N_SAME, N_UNRES = PRE + "-MISREAD", PRE + "-SAME-READ", PRE + "-UNRESOLVED"\nN_BOTH = PRE + "-BOTH"')
rep('    out["Q3"] = paired_closing(tabs, "clos", n_ep, idx)\n', '    out["Q3"] = paired_closing(tabs, "clos2", n_ep, idx)\n')
rep('    out["beside_Q3_ego_minus_imagined_lead_speed"] = paired_closing(tabs, "clos2", n_ep, idx)\n',
    '    out["beside_Q3_minus_c_src"] = paired_closing(tabs, "clos", n_ep, idx)\n')
i = s.index("def derive_name(v):")
j = s.index("# ============================================================================================ controls")
s = s[:i] + '''def derive_name(v):
    q1, q2, q3 = v["Q1"], v["Q2"], v["Q3"]
    if v["n_X"] < MIN_X or v["n_S"] < MIN_X:
        return N_UNRES, "X has %d and S %d windows over the model seeds; below %d: too small to read" % (v["n_X"], v["n_S"], MIN_X)
    mis = _pos(q1) or _pos(q2)
    if _pos(q3) and mis:
        return N_BOTH, "Q3 above zero and Q1 or Q2 above zero, intervals excluding zero"
    if _pos(q3) and _notsep(q1) and _notsep(q2):
        return N_SAME, "Q3 (v0 - v_L) above zero with an interval excluding zero; Q1 and Q2 not separated"
    if mis and not _pos(q3):
        return N_MISREAD, "Q1 or Q2 above zero with an interval excluding zero; Q3 not above zero"
    return N_UNRES, "no name condition met; X %d, S %d" % (v["n_X"], v["n_S"])


def ctl_c5(tabs):
    """-c_src paired RH minus UE in S is 0 within 1e-9 (the structural identity of row 207); v0 - v_L differs between the arms in at least one window"""
    n, max_c, n_diff2 = 0, 0.0, 0
    for m in sorted(tabs):
        a, b = tabs[m]["S"], tabs[m]["S_U"]
        ka = {int(i): q for q, i in enumerate(a["i"])}
        for q2, i in enumerate(b["i"]):
            q = ka.get(int(i))
            if q is None:
                continue
            d1, d2 = a["clos"][q] - b["clos"][q2], a["clos2"][q] - b["clos2"][q2]
            if math.isfinite(d1):
                n += 1
                max_c = max(max_c, abs(d1))
            if math.isfinite(d2) and abs(d2) > 1e-9:
                n_diff2 += 1
    return {"n_pairs": n, "max_abs_minus_c_src_diff": max_c, "n_pairs_v0_minus_vL_differ": n_diff2, "PASS": bool(n > 0 and max_c <= 1e-9 and n_diff2 > 0)}


def ctl_c0(root, seeds):
    """argv of every run equals the ADD142 queue's table (box2_crhb_queue.build_runs, paths mapped to the box-2 root); RH and UE dumps differ"""
    import hashlib
    import json as _json
    import sys as _sys
    kit = os.path.join(os.path.dirname(os.path.abspath(root)), "kit")
    _sys.path.insert(0, kit)
    import box2_crhb_queue as Qb
    want = {r["arm"]: r["argv"] for r in Qb.build_runs()}
    bad = []
    for arm, argv in sorted(want.items()):
        p = os.path.join(root, arm, "argv.json")
        if not os.path.isfile(p) or _json.load(io.open(p, encoding="utf-8")).get("argv") != argv:
            bad.append("argv " + arm)
        if not os.path.isfile(os.path.join(root, arm, "RUN_DONE.json")):
            bad.append("done " + arm)
    for m in seeds:
        hs = []
        for tag in (TAG_H, TAG_U):
            d = dump_folder(root, tag, m)
            h = hashlib.sha256()
            for f in sorted(os.listdir(d)):
                h.update(io.open(os.path.join(d, f), "rb").read())
            hs.append(h.hexdigest())
        if hs[0] == hs[1]:
            bad.append("dumps identical seed %d" % m)
    return {"problems": bad, "n_runs": len(want), "PASS": not bad}


''' + s[j:]
rep('    ctl = {"c1": ctl_c1(runs, seeds), "c2": ctl_c2(runs, seeds, tabs, J)}\n',
    '    ctl = {"c0": ctl_c0(a.root, seeds), "c1": ctl_c1(runs, seeds), "c2": ctl_c2(runs, seeds, tabs, J)}\n')
rep('    ctl["c4"] = ctl_c4(a.tests_out)\n', '    ctl["c4"] = ctl_c4(a.tests_out)\n    ctl["c5"] = ctl_c5(tabs)\n')
rep('"* Q3 closing rate S, RH minus UE, paired by window (m/s, positive = RH closes faster; -c_src): %s" % fmt_iv(v["Q3"]),',
    '"* Q3 of record, own rolled speed minus the rule lead speed (v0 - v_L), S, RH minus UE, paired by window (m/s, positive = RH closes faster): %s" % fmt_iv(v["Q3"]),')
rep('"* beside, Q3 with ego speed minus the rule lead speed: %s" % fmt_iv(v["beside_Q3_ego_minus_imagined_lead_speed"]), ""]',
    '"* beside, -c_src (identical across the arms by construction, control c5): %s" % fmt_iv(v["beside_Q3_minus_c_src"]), ""]')
rep('print("ZZ READ-CONTACTRH10K all_pass', 'print("ZZ READ-CONTACTRH10KB all_pass')
s = s.replace('"""read_contactrh10k.py -- read tool of card PREREG_V8F_PROOF_ADD141 (CONTACTRH10K)',
              '"""read_contactrh10kb.py -- read tool of card PREREG_V8F_PROOF_ADD142 (CONTACTRH10KB), written by mk_read_contactrh10kb.py from read_contactrh10k.py (card ADD141); the text below is the parent\'s', 1)
io.open(os.path.join(HERE, "read_contactrh10kb.py"), "w", encoding="utf-8", newline="\n").write(s)
print("wrote read_contactrh10kb.py")
