"""A3.3 tests for p5c_rj5_path.py (PI rulings 2026-11-16): the `fallback` step (the ONE ruled row takes its own Tc20s label), its provenance record, and the tb0r tolerance (--tc20s-tol, default unchanged).
Fixtures and helpers come from test_p5c_rj5_path.py (the A3 suite, unchanged).  Every A3.3 behaviour is a named check on a module; the REAL module reads all True and each MUTANT (the module text edited) at least one False."""
import copy
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5c_rj5_path as P  # noqa: E402
from test_p5c_rj5_path import (W, G, GATES_MOD, P5R_ERR, _feed_and_label, refused, read_map)  # noqa: E402,F401

RULED = ("logB", "b1")          # the row that stands for the ruled row in the fixture (an RJ5 rank-0 row)
OTHER = ("logC", "c2")          # another RJ5 rank-0 row that ALSO fails S2: not ruled, never a fallback
SHA_ZERO = "0" * 64


def jl(p):
    return [json.loads(ln) for ln in open(p, encoding="utf-8")]


def wjl(p, rows):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return p


def s2_fail(rows, keys):
    out = []
    for r in rows:
        if (r["log"], r["token"]) in keys and r["rank"] == 0 and "p5r_error" not in r:
            r = dict(r, p5r_S2_yaw_branch=False)
        out.append(r)
    return out


def prep(W, tag, fail_keys):
    """feed + stub labels with S2 false on `fail_keys` + rekey -> (map path, rekeyed path, map dict, bank, sub dir)"""
    man, mp, lab = _feed_and_label(W) if not (W["tmp"] / "feed" / "feed_map.jsonl.gz").exists() else (None, str(W["tmp"] / "feed" / "feed_map.jsonl.gz"), str(W["tmp"] / "feed" / "labels" / "labels_s0of1.jsonl"))
    msha = P.sha256_file(mp)
    sub = W["tmp"] / tag
    sub.mkdir(exist_ok=True)
    lab2 = wjl(str(sub / "labels_s0of1.jsonl"), s2_fail(jl(lab), set(fail_keys)))
    rk = str(sub / "rekeyed.jsonl")
    P.rekey(mp, [lab2], rk, msha)
    return mp, rk, read_map(mp), sub


def allow_of(mp_dict, key):
    return (key[1], mp_dict[(key[0], key[1], 0)]["label_traj_sha256"])


def a33_checks(Pm, W, tag):
    c = {}
    mp, rk, md, sub = prep(W, tag + "_a", [RULED, OTHER])
    A = allow_of(md, RULED)
    src = jl(rk)
    # ---- one allowed row, one unlisted candidate
    out = str(sub / "fb.jsonl")
    man = Pm.label_fallback(rk, out, [A])
    new = jl(out)
    byk = {(r["log"], r["token"], r["rank"]): r for r in new}
    old = {(r["log"], r["token"], r["rank"]): r for r in src}
    kr = (RULED[0], RULED[1], 0)
    ko = (OTHER[0], OTHER[1], 0)
    c["R1 the ruled row: p5r == its tc20s, marker tb5_s2_tc20s, the P5-R label kept as p5r_discarded"] = (byk[kr]["p5r"] == old[kr]["tc20s"] and byk[kr].get("p5r_fallback") == "tb5_s2_tc20s"
                                                                                                         and byk[kr]["p5r_discarded"] == old[kr]["p5r"] and old[kr]["p5r"] != old[kr]["tc20s"])
    c["R2 every OTHER record is byte-identical (the unlisted S2 failure is untouched and unmarked)"] = (all(byk[k] == old[k] for k in old if k != kr) and "p5r_fallback" not in byk[ko] and byk[ko]["p5r"] == old[ko]["p5r"])
    c["R3 manifest: n_fallback 1, 1 unlisted S2 candidate, counts and feed-map sha carried over, out sha bound"] = (man["n_fallback"] == 1 and man["n_unlisted_s2_candidates"] == 1 and man["counts"] == json.load(open(rk + ".manifest.json"))["counts"]
                                                                                                                    and man["out_sha256"] == P.sha256_file(out) and man["feed_map_sha256"] == P.sha256_file(mp)
                                                                                                                    and man["fallback_rows"][0]["token"] == RULED[1])
    # ---- the gates see the marked row as intended, and the unlisted failure stays a FAULT
    bank = G.load_bank([W["bank"]])
    recs, n, dup = G.load_label_recs([out])
    M = G.merge_p5r(bank, recs, dup)
    R = G.gate_tb5_p5r(bank, M, (A,))
    c["R4 TB-5 on the output: FAIL only because of the UNLISTED row (S2 count exactly 1), the fallback row itself is clean"] = R["status"] == "FAIL" and R["detail"]["S1_S2_S5_not_true"]["S2"] == 1 and R["detail"]["label_fallback"]["violations"] == []
    # ---- only the ruled row failing: TB-5 passes
    mp2, rk2, md2, sub2 = prep(W, tag + "_b", [RULED])
    out2 = str(sub2 / "fb.jsonl")
    man2 = Pm.label_fallback(rk2, out2, [allow_of(md2, RULED)])
    recs2, _, dup2 = G.load_label_recs([out2])
    R2 = G.gate_tb5_p5r(bank, G.merge_p5r(bank, recs2, dup2), (allow_of(md2, RULED),))
    c["R5 only the ruled row failing S2: after the fallback TB-5 PASSES with 1 disclosed fallback"] = R2["status"] == "PASS" and R2["detail"]["label_fallback"]["n"] == 1 and man2["n_unlisted_s2_candidates"] == 0
    # ---- default: no allow-list = identity
    out3 = str(sub / "fb_none.jsonl")
    man3 = Pm.label_fallback(rk, out3, [])
    c["R6 empty allow-list: the output is the input record for record, n_fallback 0 (the two S2 failures are reported, not touched)"] = jl(out3) == src and man3["n_fallback"] == 0 and man3["n_unlisted_s2_candidates"] == 2
    # ---- cap
    try:
        Pm.label_fallback(rk, str(sub / "fb2.jsonl"), [A, allow_of(md, OTHER)])
        c["R7 an allow-list of 2 rows is REFUSED (cap 1)"] = False
    except Pm.Refused:
        c["R7 an allow-list of 2 rows is REFUSED (cap 1)"] = True
    # ---- the ruled row must exist on the allow-list AND fail: a non-failing allowed row is not rewritten
    mp4, rk4, md4, sub4 = prep(W, tag + "_c", [OTHER])
    man4 = Pm.label_fallback(rk4, str(sub4 / "fb.jsonl"), [allow_of(md4, RULED)])
    c["R8 the allowed row does NOT fail S2: nothing is rewritten (n_fallback 0), the other failure stays unlisted"] = man4["n_fallback"] == 0 and man4["n_unlisted_s2_candidates"] == 1
    # ---- provenance
    gm = str(sub / "gm.json")
    json.dump({"label_name": "P5R", "rank_policy": "rank0", "n_rows": 12}, open(gm, "w"))
    a3 = Pm.provenance(mp2, out2 + ".manifest.json", gm, str(sub / "meta_a3.json"), str(sub / "sc.jsonl.gz"))
    side = {s["table_key"]: s for s in P.read_jsonl(str(sub / "sc.jsonl.gz"), "sc")}
    tk = man2["fallback_rows"][0]["table_key"]
    c["P1 sidecar: the ruled row carries fallback_reason tb5_s2_tc20s and p5r_label_replaced_by tc20s; all other rows' fallback_reason is the feed's"] = (
        side[tk]["fallback_reason"] == "tb5_s2_tc20s" and side[tk].get("p5r_label_replaced_by") == "tc20s"
        and all(s["fallback_reason"] in (None, "missing", "rank1") for k, s in side.items() if k != tk))
    c["P2 meta a3 block counts the fallback (n 1, reason, key)"] = a3.get("label_fallback_rows") == {"n": 1, "reasons": ["tb5_s2_tc20s"], "table_keys": [tk]}
    # pre-A3.3 manifest (plain rekey manifest): the old provenance, no new key, no new field
    a3_old = Pm.provenance(mp2, rk2 + ".manifest.json", gm, str(sub / "meta_a3_old.json"), str(sub / "sc_old.jsonl.gz"))
    side_old = list(P.read_jsonl(str(sub / "sc_old.jsonl.gz"), "sc"))
    c["P3 a manifest without fallback: no label_fallback_rows key and no new sidecar field (the old provenance)"] = "label_fallback_rows" not in a3_old and all("p5r_label_replaced_by" not in s for s in side_old)
    # ---- refusals
    refusals = []
    for allow in ([("tok", "zz")], [("", SHA_ZERO)]):
        try:
            Pm.label_fallback(rk, str(sub / "x.jsonl"), allow)
            refusals.append(False)
        except Pm.Refused:
            refusals.append(True)
    tampered = str(sub / "tampered.jsonl")
    wjl(tampered, src[:-1])
    open(tampered + ".manifest.json", "w").write(open(rk + ".manifest.json").read())
    try:
        Pm.label_fallback(tampered, str(sub / "y.jsonl"), [A])
        refusals.append(False)
    except Pm.Refused:
        refusals.append(True)
    nof = [dict(r, tc20s=None) if (r["log"], r["token"], r["rank"]) == kr else r for r in src]
    nofp = wjl(str(sub / "nofinite.jsonl"), nof)
    m = json.load(open(rk + ".manifest.json"))
    m["out_sha256"] = P.sha256_file(nofp)
    json.dump(m, open(nofp + ".manifest.json", "w"))
    try:
        Pm.label_fallback(nofp, str(sub / "z.jsonl"), [A])
        refusals.append(False)
    except Pm.Refused:
        refusals.append(True)
    c["R9 refusals: bad allow-list entries (2), a rekeyed file that is not the manifest's, a ruled row with no Tc20s label"] = all(refusals) and len(refusals) == 4
    # ---- tb0r tolerance
    fb = str(W["tmp"] / "feed" / "bank")
    lab = str(W["tmp"] / "feed" / "labels" / "labels_s0of1.jsonl")
    dev = [r for r in jl(lab) if r["rank"] == 0 and r["log"] == "logA" and (r["log"], r["token"], r["rank"]) != P5R_ERR]
    tc_off = [dict(r, tc20s=[[p[0] + 5e-9, p[1], p[2]] for p in r["tc20s"]]) if j == 0 else r for j, r in enumerate(dev)]
    dp = wjl(str(sub / "dev_off.jsonl"), tc_off)
    c["T1 tb0r: a 5e-9 Tc20s offset FAILS at the default tolerance"] = Pm.tb0r(GATES_MOD, fb, [lab], [dp], len(dev), str(sub / "t1.json"))["status"] == "FAIL"
    r2 = Pm.tb0r(GATES_MOD, fb, [lab], [dp], len(dev), str(sub / "t2.json"), 1e-8)
    c["T2 tb0r: the same offset PASSES at tol 1e-8 and the bar names 1e-08"] = r2["status"] == "PASS" and "1e-08" in r2["bar"]
    tc_big = [dict(r, tc20s=[[p[0] + 2e-8, p[1], p[2]] for p in r["tc20s"]]) if j == 0 else r for j, r in enumerate(dev)]
    c["T3 tb0r: a 2e-8 offset still FAILS at tol 1e-8"] = Pm.tb0r(GATES_MOD, fb, [lab], [wjl(str(sub / "dev_big.jsonl"), tc_big)], len(dev), str(sub / "t3.json"), 1e-8)["status"] == "FAIL"
    rc = Pm.main(["tb0r", "--gates-module", GATES_MOD, "--feed-bank", fb, "--pod-labels", lab, "--dev-labels", dp, "--expect-n", str(len(dev)), "--out", str(sub / "t4.json"), "--tc20s-tol", "1e-8"])
    rc0 = Pm.main(["tb0r", "--gates-module", GATES_MOD, "--feed-bank", fb, "--pod-labels", lab, "--dev-labels", dp, "--expect-n", str(len(dev)), "--out", str(sub / "t5.json")])
    c["T4 CLI tb0r --tc20s-tol 1e-8 exits 0 on it; without the flag it exits 1 (default unchanged)"] = rc == 0 and rc0 == 1
    # ---- CLI fallback
    rcf = Pm.main(["fallback", "--rekeyed", rk2, "--out", str(sub / "cli_fb.jsonl"), "--allow-row", f"{A[0]}:{A[1]}"])
    rcb = Pm.main(["fallback", "--rekeyed", rk2, "--out", str(sub / "cli_fb2.jsonl"), "--allow-row", "nope"])
    c["C1 CLI fallback exits 0 with the allow-row, 2 on a malformed one"] = rcf == 0 and rcb == 2 and jl(str(sub / "cli_fb.jsonl")) == jl(out2)
    return c


def test_real_module_reads_all_true(W):
    c = a33_checks(P, W, "real")
    bad = [k for k, v in c.items() if not v]
    assert not bad, bad


MUT = [
    ("fallback applied to an unlisted row", 'and (r["token"], r.get("label_traj_sha256")) in aset:', ":"),
    ("fallback applied to 2 rows (cap lifted)", "MAX_LABEL_FALLBACK_ROWS = 1  ", "MAX_LABEL_FALLBACK_ROWS = 9  "),
    ("fallback leaves the P5-R label in place", 'o["p5r"] = r["tc20s"]', "pass"),
    ("fallback marker not written", 'o["p5r_fallback"] = LABEL_FALLBACK_REASON', "pass"),
    ("fallback not recorded in the provenance sidecar", "if tk in fbrows:", "if False:"),
    ("fallback not counted in the provenance meta", "    if fbrows:\n        a3[", "    if False:\n        a3["),
    ("tb0r tolerance ignored (function)", "g_t = G.gate_tb0(bank, M_t, dev, tol, pod)", "g_t = G.gate_tb0(bank, M_t, dev, TC20S_TOL, pod)"),
    ("tb0r tolerance ignored (CLI flag not passed)", "a.expect_n, a.out, a.tc20s_tol)", "a.expect_n, a.out)"),
    ("rekeyed file integrity not checked", 'if rman.get("out_sha256") != sha256_file(rekeyed_path):', "if False:"),
]


@pytest.mark.parametrize("name,old,new", MUT, ids=[m[0] for m in MUT])
def test_mutant_goes_red(W, name, old, new):
    src = open(P.__file__, encoding="utf-8").read()
    assert src.count(old) == 1, ("mutation target", src.count(old), old[:70])
    Pm = types.ModuleType("rj5_mut_a33")
    Pm.__file__ = P.__file__
    exec(compile(src.replace(old, new), "rj5_mut_a33", "exec"), Pm.__dict__)
    try:
        c = a33_checks(Pm, W, "mut_" + str(abs(hash(name)))[:12])
        red = [k for k, v in c.items() if not v]
    except Exception as e:                                    # a crash is also RED
        red = ["raised " + type(e).__name__]
    assert red, "MUTANT SURVIVED: " + name


def test_defaults_reproduce_the_a3_module(W):
    """rekey / provenance / tb0r with no A3.3 argument give the A3 module's outputs byte for byte"""
    import importlib.util
    base = os.environ.get("P5C_A3_RJ5_BASE", os.path.join(HERE, "..", "..", "base", "p5c_rj5_path.py"))
    spec = importlib.util.spec_from_file_location("p5c_rj5_path_a3", base)
    P0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P0)
    man, mp, lab = _feed_and_label(W)
    gm = str(W["tmp"] / "gm.json")
    json.dump({"label_name": "P5R", "rank_policy": "rank0", "n_rows": 12}, open(gm, "w"))
    outs = {}
    for nm, M_ in (("new", P), ("old", P0)):
        rk = str(W["tmp"] / f"rk_{nm}.jsonl")
        M_.rekey(mp, [lab], rk, man["feed_map_sha256"])
        M_.provenance(mp, rk + ".manifest.json", gm, str(W["tmp"] / f"meta_{nm}.json"), str(W["tmp"] / f"sc_{nm}.jsonl.gz"))
        fb = str(W["tmp"] / "feed" / "bank")
        dev = [r for r in jl(lab) if r["rank"] == 0 and r["log"] == "logA" and (r["log"], r["token"], r["rank"]) != P5R_ERR]
        res = M_.tb0r(GATES_MOD, fb, [lab], [wjl(str(W["tmp"] / f"dev_{nm}.jsonl"), dev)], len(dev), str(W["tmp"] / f"tb0r_{nm}.json"))
        outs[nm] = (open(rk, "rb").read(), open(str(W["tmp"] / f"sc_{nm}.jsonl.gz"), "rb").read(),
                    json.dumps({k: v for k, v in json.load(open(str(W["tmp"] / f"meta_{nm}.json"))).items() if k != "a3"}, sort_keys=True),
                    json.dumps({k: v for k, v in json.load(open(str(W["tmp"] / f"meta_{nm}.json")))["a3"].items() if k not in ("rekey_manifest_sha256", "provenance_sidecar", "feed_map_sha256", "rekeyed_records_sha256")}, sort_keys=True),
                    json.dumps({k: v for k, v in res.items() if k != "gates_module_sha256"}, sort_keys=True))
    assert outs["new"][0] == outs["old"][0] and outs["new"][1] == outs["old"][1]
    assert outs["new"][2] == outs["old"][2] and outs["new"][3].replace("_new", "_old") == outs["old"][3].replace("_new", "_old")
    assert outs["new"][4] == outs["old"][4]
