"""Build t1kf_read_a1.py from t1kf_read.py: PREREG_V8F_T1_FAMILIES Amendment 1 (C6 -> C6A, post-data, disclosed). Exact anchors only."""
src = open("C:/Users/Admin/t1fam/t1kf_read.py", encoding="utf-8").read()
E = [
    ('    c6["PASS"] = bool(all(c6["per_seed"]["s0"]["must_be_separated_worse"].values()))\n'
     '    controls["C6_deliberate_regression"] = c6\n'
     '    if not c6["PASS"]:\n'
     '        refused.append("C6")\n',
     '    c6["PASS"] = bool(all(c6["per_seed"]["s0"]["must_be_separated_worse"].values()))\n'
     '    c6["SUPERSEDED"] = "by C6A (PREREG Amendment 1, written AFTER the decisive rows were seen): reported, NOT gating"\n'
     '    controls["C6_deliberate_regression_AS_WRITTEN"] = c6\n'
     '\n'
     '    # ---- C6A (Amendment 1): speed-matched CROSS-EPISODE donor tracks vs the arm itself ----\n'
     '    c6a = {"what": "window i scored with the decisive HACC track of the window from a DIFFERENT episode whose dump v0 is nearest (ties: lowest index); "\n'
     '                   "vs the unshuffled HACC tracks; TAC-LAT, TAC-LON, DK-HW must each be separated-worse at s0", "per_seed": {}}\n'
     '    for s in (0, 1):\n'
     '        d = dumps[("decisive", s)]\n'
     '        v0d, eidd = np.asarray(d["v0"], np.float64), np.asarray(d["eid"])\n'
     '        donor = np.empty(len(v0d), dtype=np.int64)\n'
     '        for i in range(len(v0d)):\n'
     '            cost = np.where(eidd != eidd[i], np.abs(v0d - v0d[i]), np.inf)\n'
     '            donor[i] = int(np.argmin(cost))                    # argmin returns the lowest index on ties\n'
     '        assert (eidd[donor] != eidd).all(), "a donor from the same episode"\n'
     '        Pd = prep(f"HACC_s{s}_xepisode", np.asarray(d["cl"])[donor], cx)\n'
     '        Pu = prepared[("decisive", s)]["arm"]\n'
     '        rw = rows(Pd, Pu, eid)\n'
     '        c6a["per_seed"][f"s{s}"] = {"rows_donor_minus_HACC": rw, "gating": s == 0,\n'
     '                                    "must_be_separated_worse": {r: rw[r]["separated_worse"] for r in ("TAC-LAT", "TAC-LON", "DK-HW")},\n'
     '                                    "donor_v0_abs_diff_mps": {"median": float(np.median(np.abs(v0d[donor] - v0d))), "max": float(np.abs(v0d[donor] - v0d).max())},\n'
     '                                    "n_distinct_donors": int(len(set(donor.tolist()))),\n'
     '                                    "BESIDE_donor_minus_P10": rows(Pd, prepared[("decisive", s)]["p10"], eid)}\n'
     '    c6a["PASS"] = bool(all(c6a["per_seed"]["s0"]["must_be_separated_worse"].values()))\n'
     '    controls["C6A_deliberate_regression_amendment1"] = c6a\n'
     '    if not c6a["PASS"]:\n'
     '        refused.append("C6A")\n'),
]
for a, b in E:
    assert src.count(a) == 1, (src.count(a), a[:80])
    src = src.replace(a, b)
open("C:/Users/Admin/t1fam/t1kf_read_a1.py", "w", encoding="utf-8").write(src)
print("built t1kf_read_a1.py")
