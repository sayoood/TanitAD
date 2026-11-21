"""Tests for p5c_rj5_path.py (P5-C Amendment A3: C1 labels from the RJ5 traj, keyed to the bank row).  CPU only, no DB, no navtest.

The pinned table reader (refe/traj_label_table.py, A2 pin) and the pinned gates (a1/p5c_table_gates.py, A1 pin) are imported AS THEY ARE, so the
end-to-end tests prove the rekeyed records pass the unchanged sha checks and build the unchanged table format.  Their paths default to the repo layout
(this file at PK/raw/2026-11-11-p5c-readiness/a3/) and can be overridden with P5C_A3_TABLE_MOD / P5C_A3_GATES_MOD.  A missing module FAILS, never skips.
The labeller is a deterministic STUB with the pinned labeller's record format (the real one needs nuPlan DBs): the tests prove the PATH, not the label math.
"""
import glob
import gzip
import importlib.util
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5c_rj5_path as P  # noqa: E402

PK = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TABLE_MOD = os.environ.get("P5C_A3_TABLE_MOD", os.path.join(PK, "refe", "traj_label_table.py"))
GATES_MOD = os.environ.get("P5C_A3_GATES_MOD", os.path.join(HERE, "..", "a1", "p5c_table_gates.py"))


def _load(name, path):
    assert os.path.exists(path), f"pinned module {name} not found at {path} (set P5C_A3_TABLE_MOD / P5C_A3_GATES_MOD)"
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TL = _load("traj_label_table_pinned", TABLE_MOD)
G = _load("p5c_table_gates_pinned", GATES_MOD)


# ------------------------------------------------------------------ fixtures
def mk_traj(seed, n=20):
    return [[0.5 * (i + 1) + 0.01 * seed, 0.002 * seed * (i + 1), 0.001 * seed + 0.0005 * i] for i in range(n)]


def moved(t, d=0.3):
    return [[p[0] + d * math.sin(0.3 * i + 1.0), p[1] + 0.5 * d * math.cos(0.2 * i), p[2] + 0.01 * d] for i, p in enumerate(t)]


LOGS = {"logA": ["a1", "a2", "a3", "a4"], "logB": ["b1", "b2", "b3"], "logC": ["c1", "c2"]}
TWINS = [("logA", "a1"), ("logB", "b2"), ("logC", "c1")]
MISSING = {("logB", "b3")}                       # status "missing": declared fallback
SAME = {("logA", "a4")}                          # an RJ5 roll that reproduced the bank traj exactly
SRC = {"logA": "A", "logB": "B", "logC": "C"}
P5R_ERR = ("logA", "a2", 0)                      # the stub labeller raises on this row: raw fallback in BOTH tables


def write_bank(d):
    os.makedirs(d, exist_ok=True)
    r0, r1 = [], []
    s = 0
    for lg, toks in LOGS.items():
        for t in toks:
            s += 1
            r0.append({"log_name": lg, "token": t, "rank": 0, "sim_hz": 10, "step": 0, "traj": mk_traj(s), "extra": "ignored"})
    for lg, t in TWINS:
        s += 1
        r1.append({"log_name": lg, "token": t, "rank": 1, "sim_hz": 10, "step": 0, "traj": mk_traj(s)})
    for name, rows in (("targets_rank0.jsonl", r0), ("targets_rank1.jsonl", r1)):
        with open(os.path.join(d, name), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return {(r["log_name"], r["token"]): r["traj"] for r in r0}, r1


def join_rows(bank0, mutate=None):
    out = []
    for (lg, t), tr in sorted(bank0.items()):
        sha = P.traj_sha256(tr)
        o = {"log_name": lg, "token": t, "rank": 0, "bank_key": f"{lg}|{t}|0|{sha}", "bank_traj_sha256": sha}
        if (lg, t) in MISSING:
            o.update(status="missing", source=None)
        else:
            rt = tr if (lg, t) in SAME else moved(tr)
            o.update(status="rj5", source=SRC[lg], rj5_traj_sha256=P.traj_sha256(rt), rj5_traj=rt)
        out.append(o)
    if mutate:
        out = mutate(out)
    return out


def write_join(path, rows):
    with gzip.GzipFile(path, "wb", mtime=0) as g:
        for o in rows:
            g.write((json.dumps(o) + "\n").encode("utf-8"))
    return P.sha256_file(path)


def write_r0(d, bank0):
    os.makedirs(d, exist_ok=True)
    rows = [{"log_name": "logA", "token": "a1", "rank": 0, "arm": "R0", "traj": bank0[("logA", "a1")]},
            {"log_name": "logA", "token": "a3", "rank": 0, "arm": "R0", "traj": [[p[0] + 2e-6, p[1], p[2]] for p in bank0[("logA", "a3")]]},
            {"log_name": "logB", "token": "b1", "rank": 0, "arm": "R0", "traj": bank0[("logB", "b1")]},
            {"log_name": "logB", "token": "b2", "rank": 0, "arm": "RJ5", "traj": moved(bank0[("logB", "b2")])}]       # not an R0 row: ignored
    with open(os.path.join(d, "rj5_ctl.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


PLAT = ["A=devbox", "B=devbox", "C=pod"]


@pytest.fixture
def W(tmp_path):
    bank = str(tmp_path / "bank")
    bank0, r1 = write_bank(bank)
    jp = str(tmp_path / "join.jsonl.gz")
    jsha = write_join(jp, join_rows(bank0))
    r0 = str(tmp_path / "r0")
    write_r0(r0, bank0)
    return {"tmp": tmp_path, "bank": bank, "bank0": bank0, "r1": r1, "join": jp, "jsha": jsha, "r0": [f"A={r0}"]}


def do_feed(W, join=None, jsha=None, plat=PLAT, cov=0.5, out="feed", r0=None):
    return P.feed(W["bank"], join or W["join"], jsha or W["jsha"], plat, W["r0"] if r0 is None else r0, cov, str(W["tmp"] / out))


def stub_label(bank_dir, out_dir, err=(P5R_ERR,), tag="s0of1"):
    """the pinned labeller's record format: {"log","token","rank","traj_sha256" (of the traj it READ), "tc20s", "p5r", p5r_* flags}"""
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"labels_{tag}.jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for fp in sorted(os.listdir(bank_dir)):
            if not fp.startswith("targets_rank"):
                continue
            for ln in open(os.path.join(bank_dir, fp), encoding="utf-8"):
                r = json.loads(ln)
                t = r["traj"]
                lab = [[q[0] + 0.02 * math.cos(0.4 * i), q[1] - 0.01 * math.sin(0.5 * i), q[2] + 0.002] for i, q in enumerate(t)]
                tc = [[(a + b) / 2 for a, b in zip(q, l)] for q, l in zip(t, lab)]
                rec = {"log": r["log_name"], "token": r["token"], "rank": r["rank"], "traj_sha256": P.traj_sha256(t), "tc20s": tc, "knot_maxdev": 0.0, "knot_checks": 1}
                if (r["log_name"], r["token"], int(r["rank"])) in err:
                    rec["p5r_error"] = "RuntimeError: stub"
                else:
                    rec.update(p5r=lab, p5r_sim8=False, p5r_S3_dev=0.0, p5r_S1_finite=True, p5r_S2_yaw_branch=True, p5r_S5_lambda_reached=True)
                f.write(json.dumps(rec) + "\n")
    return p


def refused(fn, *a, code=2, match=None):
    with pytest.raises(P.Refused) as e:
        fn(*a)
    assert e.value.code == code, (e.value.code, str(e.value))
    if match:
        assert match in str(e.value), str(e.value)
    return str(e.value)


def read_map(path):
    return {(m["log_name"], m["token"], m["rank"]): m for m in P.read_jsonl(path, "map")}


# ------------------------------------------------------------------ feed: happy path
def test_feed_happy_path_provenance_and_determinism(W):
    man = do_feed(W)
    mp = read_map(str(W["tmp"] / "feed" / "feed_map.jsonl.gz"))
    assert len(mp) == 9 + 3
    assert man["n_rj5"] == 8 and abs(man["rj5_coverage_rank0"] - 8 / 9) < 1e-12
    fb = {}
    for fp in ("targets_rank0.jsonl", "targets_rank1.jsonl"):
        for ln in open(W["tmp"] / "feed" / "bank" / fp, encoding="utf-8"):
            r = json.loads(ln)
            fb[(r["log_name"], r["token"], r["rank"])] = r["traj"]
    for (lg, t, rk), m in mp.items():
        assert m["label_traj_sha256"] == P.traj_sha256(fb[(lg, t, rk)])
        if rk == 1:
            assert m["label_source"] == "bank_fallback" and m["fallback_reason"] == "rank1" and m["roll_platform"] is None
            assert m["label_traj_sha256"] == m["bank_traj_sha256"]
        elif (lg, t) in MISSING:
            assert m["label_source"] == "bank_fallback" and m["fallback_reason"] == "missing" and m["rj5_source"] is None
            assert fb[(lg, t, 0)] == W["bank0"][(lg, t)]           # the bank traj itself, never another row's RJ5 traj
        else:
            assert m["label_source"] == "rj5" and m["rj5_source"] == SRC[lg]
            assert m["roll_platform"] == {"A": "devbox", "B": "devbox", "C": "pod"}[SRC[lg]]
            assert (m["label_traj_sha256"] == m["bank_traj_sha256"]) == ((lg, t) in SAME)
    # TB-8 disclosure: logA has a 2e-6 m R0 row -> diverged; logB exact; logC has no R0 row -> uncontrolled
    assert mp[("logA", "a2", 0)]["r0_log_status"] == "diverged" and mp[("logA", "a1", 0)]["r0_row_dxy_m"] == 0.0
    assert abs(mp[("logA", "a3", 0)]["r0_row_dxy_m"] - 2e-6) < 1e-9 and mp[("logA", "a2", 0)]["r0_row_dxy_m"] is None
    assert mp[("logB", "b1", 0)]["r0_log_status"] == "exact" and mp[("logC", "c1", 0)]["r0_log_status"] == "uncontrolled"
    assert set(m["label_source"] for m in mp.values()) <= set(P.LABEL_SOURCES)
    man2 = do_feed(W, out="feed2")
    assert man2["feed_map_sha256"] == man["feed_map_sha256"] and man2["feed_bank_files_sha256"] == man["feed_bank_files_sha256"]


# ------------------------------------------------------------------ feed: RED arms (wrong-key override, integrity)
def _mut_join(W, fn, name):
    jp = str(W["tmp"] / name)
    return jp, write_join(jp, join_rows(W["bank0"], fn))


def _first_rj5(rows):
    return next(i for i, o in enumerate(rows) if o["status"] == "rj5" and (o["log_name"], o["token"]) not in SAME)


def test_red_wrong_key_override_refused(W):
    def other_sha(rows):                                        # the join row claims another bank traj's sha: the label would be keyed to another traj
        i = _first_rj5(rows)
        rows[i]["bank_traj_sha256"] = rows[i + 1]["bank_traj_sha256"]
        rows[i]["bank_key"] = "|".join(rows[i]["bank_key"].split("|")[:3] + [rows[i + 1]["bank_traj_sha256"]])
        return rows
    jp, js = _mut_join(W, other_sha, "j_wrongkey.jsonl.gz")
    refused(do_feed, W, jp, js, match="wrong-key override REFUSED")

    def key_only(rows):                                         # sha field right, bank_key string wrong
        i = _first_rj5(rows)
        rows[i]["bank_key"] = rows[i]["bank_key"][:-1] + ("0" if rows[i]["bank_key"][-1] != "0" else "1")
        return rows
    jp, js = _mut_join(W, key_only, "j_keyonly.jsonl.gz")
    refused(do_feed, W, jp, js, match="wrong-key override REFUSED")

    def foreign(rows):                                          # a token the bank does not have
        rows.append(dict(rows[_first_rj5(rows)], token="zz_not_in_bank"))
        return rows
    jp, js = _mut_join(W, foreign, "j_foreign.jsonl.gz")
    refused(do_feed, W, jp, js, match="not a rank-0 key of this bank")


@pytest.mark.parametrize("case", ["rank1", "duplicate", "conflict", "bad_rj5_sha", "yaw_wrap", "short", "nan", "status"])
def test_red_join_integrity_refused(W, case):
    def mut(rows):
        i = _first_rj5(rows)
        if case == "rank1":
            rows[i]["rank"] = 1
        elif case == "duplicate":
            rows.append(dict(rows[i]))
        elif case == "conflict":
            rows[i]["status"] = "conflict"
        elif case == "bad_rj5_sha":
            rows[i]["rj5_traj"] = moved(rows[i]["rj5_traj"], 1e-9)
        elif case == "yaw_wrap":
            rows[i]["rj5_traj"] = [[p[0], p[1], p[2] + 2 * math.pi] for p in rows[i]["rj5_traj"]]
            rows[i]["rj5_traj_sha256"] = P.traj_sha256(rows[i]["rj5_traj"])
        elif case == "short":
            rows[i]["rj5_traj"] = rows[i]["rj5_traj"][:19]
            rows[i]["rj5_traj_sha256"] = P.traj_sha256(rows[i]["rj5_traj"])
        elif case == "nan":
            rows[i]["rj5_traj"] = [list(p) for p in rows[i]["rj5_traj"]]
            rows[i]["rj5_traj"][3][0] = float("nan")
        elif case == "status":
            rows[i]["status"] = "maybe"
        return rows
    jp, js = _mut_join(W, mut, f"j_{case}.jsonl.gz")
    refused(do_feed, W, jp, js)


def test_red_absent_key_refused_missing_status_falls_back(W):
    # a bank key ABSENT from the join is REFUSED (stale join); only an explicit "missing" falls back
    jp, js = _mut_join(W, lambda rows: [o for o in rows if (o["log_name"], o["token"]) not in MISSING], "j_absent.jsonl.gz")
    refused(do_feed, W, jp, js, match="ABSENT from the join")


def test_red_pins_platform_and_r0_required(W):
    refused(do_feed, W, None, "0" * 64, match="pinned")                                        # join sha pin
    refused(do_feed, W, None, None, ["A=devbox", "B=devbox"], match="no declared roll platform")  # C undeclared
    refused(do_feed, W, None, None, ["A=devbox", "B=box2", "C=pod"], match="platform must be one of")
    refused(do_feed, W, None, None, PLAT, 0.5, "f_r0", [], match="--r0-rows is required")
    bad = str(W["tmp"] / "r0bad")
    os.makedirs(bad)
    with open(os.path.join(bad, "rj5_x.jsonl"), "w") as f:
        f.write(json.dumps({"log_name": "logZ", "token": "q", "rank": 0, "arm": "R0", "traj": mk_traj(1)}) + "\n")
    refused(do_feed, W, None, None, PLAT, 0.5, "f_r0b", [f"A={bad}"], match="not a bank rank-0 key")


def test_coverage_bar_refuses_with_code_3_and_writes_no_feed_bank(W):
    refused(do_feed, W, None, None, PLAT, 0.99, "f_cov", code=3, match="coverage")
    assert not os.path.exists(W["tmp"] / "f_cov" / "bank")
    assert json.load(open(W["tmp"] / "f_cov" / "feed_manifest.json"))["refused"]
    assert do_feed(W, cov=8 / 9, out="f_cov_ok")["n_rank0_bank_fallback"] == 1        # exactly at the bar: passes


def test_torn_join_line_refused(W):
    jp = str(W["tmp"] / "j_torn.jsonl.gz")
    rows = join_rows(W["bank0"])
    with gzip.GzipFile(jp, "wb", mtime=0) as g:
        for o in rows[:-1]:
            g.write((json.dumps(o) + "\n").encode())
        g.write(json.dumps(rows[-1]).encode()[:40])
    refused(do_feed, W, jp, P.sha256_file(jp), match="torn")


# ------------------------------------------------------------------ rekey
def _feed_and_label(W, out="feed"):
    man = do_feed(W, out=out)
    lab = stub_label(str(W["tmp"] / out / "bank"), str(W["tmp"] / out / "labels"))
    return man, str(W["tmp"] / out / "feed_map.jsonl.gz"), lab


def test_rekey_keys_to_bank_and_passes_unchanged_gates(W):
    man, mp, lab = _feed_and_label(W)
    out = str(W["tmp"] / "rekeyed.jsonl")
    rm = P.rekey(mp, [lab], out, man["feed_map_sha256"])
    assert rm["n_records"] == 12
    bank = G.load_bank([W["bank"]])
    recs, n, dup = G.load_label_recs([out])
    M = G.merge_p5r(bank, recs, dup)
    assert int(M["sha_mismatch"].sum()) == 0 and not (M["source"] == "absent").any() and not M["extra_keys"]
    assert int((M["source"] == "raw_fallback").sum()) == 1                       # the stub's p5r_error row, label := the BANK traj
    for d in recs.values():
        assert d["label_source"] in P.LABEL_SOURCES and "label_traj_sha256" in d and "roll_platform" in d


def test_red_rekey_skipped_makes_unchanged_gates_drop_the_rj5_rows(W):
    man, mp, lab = _feed_and_label(W)
    bank = G.load_bank([W["bank"]])
    recs, _, dup = G.load_label_recs([lab])                                     # NOT rekeyed
    M = G.merge_p5r(bank, recs, dup)
    n_moved = 8 - len(SAME)
    assert int(M["sha_mismatch"].sum()) == n_moved                              # every RJ5 row whose traj moved is refused by the gates' own sha check
    os.makedirs(W["tmp"] / "g_norekey")
    wr = G.write_merged(bank, M, str(W["tmp"] / "g_norekey"))
    rows = TL.read_bank_rows(W["bank"])
    labels = {}
    for ln in open(wr["rank0only"], encoding="utf-8"):
        d = json.loads(ln)
        labels[f"{d['log_name']}|{d['token']}|{d['rank']}"] = (d["label"], d["traj_sha256"])
    with pytest.raises(TL.TableRefused):
        TL.build_table(rows, labels, {"label_name": "P5R"}, 1.0, rank_policy="rank0")


def test_red_rekey_refusals(W):
    man, mp, lab = _feed_and_label(W)
    recs = [json.loads(l) for l in open(lab, encoding="utf-8")]

    def write(name, rs):
        p = str(W["tmp"] / name)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        return p
    i = next(k for k, r in enumerate(recs) if (r["log"], r["token"]) == ("logA", "a1") and r["rank"] == 0)
    # the labeller saw the BANK traj, not the fed RJ5 traj (the override did not reach it)
    bad = [dict(r) for r in recs]
    bad[i]["traj_sha256"] = P.traj_sha256(W["bank0"][("logA", "a1")])
    refused(P.rekey, mp, [write("r_bank.jsonl", bad)], str(W["tmp"] / "o1.jsonl"), man["feed_map_sha256"], match="did not see the fed traj")
    refused(P.rekey, mp, [write("r_miss.jsonl", recs[:i] + recs[i + 1:])], str(W["tmp"] / "o2.jsonl"), man["feed_map_sha256"], match="NO labeller record")
    assert not os.path.exists(W["tmp"] / "o2.jsonl") and not os.path.exists(str(W["tmp"] / "o2.jsonl") + ".partial")
    refused(P.rekey, mp, [write("r_dup.jsonl", recs + [recs[i]])], str(W["tmp"] / "o3.jsonl"), man["feed_map_sha256"], match="twice")
    refused(P.rekey, mp, [write("r_x.jsonl", recs + [dict(recs[i], token="nope")])], str(W["tmp"] / "o4.jsonl"), man["feed_map_sha256"], match="not a row of the feed map")
    refused(P.rekey, mp, [lab], str(W["tmp"] / "o5.jsonl"), "f" * 64, match="pinned")


# ------------------------------------------------------------------ the A3 table differs from the A2 table exactly where the RJ5 traj differs
def _table(W, recs_path, out_name):
    bank = G.load_bank([W["bank"]])
    recs, _, dup = G.load_label_recs([recs_path])
    M = G.merge_p5r(bank, recs, dup)
    os.makedirs(W["tmp"] / out_name)
    wr = G.write_merged(bank, M, str(W["tmp"] / out_name))
    rows = TL.read_bank_rows(W["bank"])
    labels = {}
    for ln in open(wr["rank0only"], encoding="utf-8"):
        d = json.loads(ln)
        labels[f"{d['log_name']}|{d['token']}|{d['rank']}"] = (d["label"], d["traj_sha256"])
    tab, meta, rep = TL.build_table(rows, labels, {"label_name": "P5R"}, 1.0, rank_policy="rank0")
    sha = TL.save(str(W["tmp"] / f"{out_name}.json.gz"), tab, meta)
    t = TL.load(str(W["tmp"] / f"{out_name}.json.gz"), sha)
    TL.apply_to_rows(t, rows)                                  # the trainer's reader: coverage 1.0, rank0 policy enforced
    return t


def _a3_table(W, join=None, jsha=None, out="a3"):
    man = P.feed(W["bank"], join or W["join"], jsha or W["jsha"], PLAT, W["r0"], 0.5, str(W["tmp"] / out))
    lab = stub_label(str(W["tmp"] / out / "bank"), str(W["tmp"] / out / "labels"))
    rk = str(W["tmp"] / f"{out}_rekeyed.jsonl")
    P.rekey(str(W["tmp"] / out / "feed_map.jsonl.gz"), [lab], rk, man["feed_map_sha256"])
    return _table(W, rk, f"{out}_g"), read_map(str(W["tmp"] / out / "feed_map.jsonl.gz"))


def _diff_keys(t2, t3):
    assert set(t2.labels) == set(t3.labels)                   # SAME keys: the trainer looks the C1 label up exactly as the A2 label
    return {k for k in t2.labels if t2.labels[k] != t3.labels[k]}


def test_a3_table_differs_from_a2_exactly_where_rj5_differs(W):
    lab2 = stub_label(W["bank"], str(W["tmp"] / "a2_labels"))               # A2: the same labeller on the bank traj
    t2 = _table(W, lab2, "a2_g")
    t3, mp = _a3_table(W)
    diff = _diff_keys(t2, t3)
    expect = {f"{lg}|{t}|0|{m['bank_traj_sha256']}" for (lg, t, rk), m in mp.items()
              if rk == 0 and m["label_source"] == "rj5" and m["label_traj_sha256"] != m["bank_traj_sha256"] and (lg, t, rk) != P5R_ERR}
    assert diff == expect and len(expect) == 6
    for (lg, t, rk), m in mp.items():                          # rank-1 rows: identity in both (rank0 policy)
        if rk == 1:
            k = f"{lg}|{t}|1|{m['bank_traj_sha256']}"
            assert t3.labels[k] == t2.labels[k] and TL.same_as_raw(t3.labels[k], next(r["traj"] for r in W["r1"] if (r["log_name"], r["token"]) == (lg, t)))


def test_red_override_that_never_reaches_the_labeller_gives_zero_differences(W):
    # mutation: every RJ5 row demoted to "missing" (the override is OFF) -> the A3 table IS the A2 table; the test above would go RED on it
    def off(rows):
        for o in rows:
            o.update(status="missing", source=None)
            o.pop("rj5_traj", None)
            o.pop("rj5_traj_sha256", None)
        return rows
    jp, js = _mut_join(W, off, "j_off.jsonl.gz")
    lab2 = stub_label(W["bank"], str(W["tmp"] / "a2_labels_off"))
    t2 = _table(W, lab2, "a2_off_g")
    man = P.feed(W["bank"], jp, js, PLAT, W["r0"], 0.0, str(W["tmp"] / "a3_off"))
    lab = stub_label(str(W["tmp"] / "a3_off" / "bank"), str(W["tmp"] / "a3_off" / "labels"))
    rk = str(W["tmp"] / "a3_off_rekeyed.jsonl")
    P.rekey(str(W["tmp"] / "a3_off" / "feed_map.jsonl.gz"), [lab], rk, man["feed_map_sha256"])
    t3 = _table(W, rk, "a3_off_g")
    assert _diff_keys(t2, t3) == set()


# ------------------------------------------------------------------ repair-combine
def _rep(path, rows, **kw):
    d = {"rank": 0, "rule": "lowest sha256('log|token') among rows of that rank whose log has a local DB", "n_requested": 300, "n": len(rows), "n_errors": 0, "rows": rows}
    d.update(kw)
    json.dump(d, open(path, "w"))
    return path


def test_repair_combine_uses_bank_raw_and_rj5_labels(W):
    b = [{"log": "logA", "token": "a1", "raw_sim8": True, "tc20s_sim8": False, "p5r_sim8": False}, {"log": "logB", "token": "b1", "raw_sim8": False, "tc20s_sim8": False, "p5r_sim8": True}]
    f = [{"log": "logA", "token": "a1", "raw_sim8": False, "tc20s_sim8": True, "p5r_sim8": True}, {"log": "logB", "token": "b1", "raw_sim8": True, "tc20s_sim8": False, "p5r_sim8": False}]
    out = str(W["tmp"] / "rep_a3.json")
    res = P.repair_combine(_rep(str(W["tmp"] / "rb.json"), b), _rep(str(W["tmp"] / "rf.json"), f), out)
    r = {(x["log"], x["token"]): x for x in res["rows"]}
    assert r[("logA", "a1")]["raw_sim8"] is True and r[("logA", "a1")]["p5r_sim8"] is True and r[("logA", "a1")]["rj5_raw_sim8"] is False
    assert res["raw_sim8_fail_rate"] == 0.5 and res["p5r_sim8_fail_rate"] == 0.5 and res["rj5_raw_sim8_fail_rate"] == 0.5
    assert G.prep_repair_p5r(json.load(open(out)))["rows"][0]["tc20s_sim8"] == res["rows"][0]["p5r_sim8"]       # the pinned gates read it
    refused(P.repair_combine, _rep(str(W["tmp"] / "rb2.json"), b[:1]), str(W["tmp"] / "rf.json"), out, match="row sets differ")
    refused(P.repair_combine, _rep(str(W["tmp"] / "rb3.json"), b, rank=1), str(W["tmp"] / "rf.json"), out, match="'rank'")
    refused(P.repair_combine, _rep(str(W["tmp"] / "rb4.json"), b, n_errors=2), str(W["tmp"] / "rf.json"), out, match="errors")


# ------------------------------------------------------------------ provenance
def test_provenance_sidecar_and_sha_bound_meta(W):
    man, mp, lab = _feed_and_label(W)
    rk = str(W["tmp"] / "rekeyed.jsonl")
    P.rekey(mp, [lab], rk, man["feed_map_sha256"])
    gm = str(W["tmp"] / "table_meta_fields.json")
    json.dump({"label_name": "P5R", "rank_policy": "rank0", "n_rows": 12, "bank_files_sha256": {os.path.basename(p): TL.sha256_file(p) for p in sorted(
        glob.glob(os.path.join(W["bank"], "targets_rank*.jsonl")))}}, open(gm, "w"))
    om, sc = str(W["tmp"] / "meta_a3.json"), str(W["tmp"] / "C1_provenance.jsonl.gz")
    a3 = P.provenance(mp, rk + ".manifest.json", gm, om, sc)
    keep, _ = TL.load_labeller_meta(om, W["bank"], 12, "P5R", "rank0")        # the pinned builder accepts the a3 block (sha-bound)
    assert keep["a3"]["provenance_sidecar_sha256"] == P.sha256_file(sc) == a3["provenance_sidecar_sha256"]
    side = list(P.read_jsonl(sc, "sidecar"))
    assert len(side) == 12 and all(s["label_source"] in P.LABEL_SOURCES for s in side)
    refused(P.provenance, mp, rk + ".manifest.json", om, str(W["tmp"] / "m2.json"), str(W["tmp"] / "s2.jsonl.gz"), match="'a3' block")


# ------------------------------------------------------------------ tb0r: A3 TB-0 on RJ5 inputs, through the pinned gate functions
def test_tb0r_pass_and_red_arms(W):
    man, mp, lab = _feed_and_label(W)
    fb = str(W["tmp"] / "feed" / "bank")
    recs = [json.loads(l) for l in open(lab, encoding="utf-8")]
    dev = [r for r in recs if r["rank"] == 0 and r["log"] == "logA" and (r["log"], r["token"], r["rank"]) != P5R_ERR]   # the "local" rows

    def write(name, rs):
        p = str(W["tmp"] / name)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        return p
    ok = P.tb0r(GATES_MOD, fb, [lab], [write("dev.jsonl", dev)], len(dev), str(W["tmp"] / "tb0r.json"))
    assert ok["status"] == "PASS", json.dumps(ok)[:400]
    tc_bad = [dict(r, tc20s=[[p[0] + 2e-9, p[1], p[2]] for p in r["tc20s"]]) if j == 0 else r for j, r in enumerate(dev)]
    assert P.tb0r(GATES_MOD, fb, [lab], [write("dev_tc.jsonl", tc_bad)], len(dev), str(W["tmp"] / "t1.json"))["status"] == "FAIL"     # R-A 1e-9
    p_bad = [dict(r, p5r=[[p[0] + 0.01, p[1], p[2]] for p in r["p5r"]]) for r in dev]
    assert P.tb0r(GATES_MOD, fb, [lab], [write("dev_p.jsonl", p_bad)], len(dev), str(W["tmp"] / "t2.json"))["status"] == "FAIL"     # R-TOL-2
    assert P.tb0r(GATES_MOD, fb, [lab], [write("dev_n.jsonl", dev)], len(dev) + 1, str(W["tmp"] / "t3.json"))["status"] == "FAIL"   # coverage n
    sha_bad = [dict(r, traj_sha256=P.traj_sha256(W["bank0"][(r["log"], r["token"])])) if j == 0 else r for j, r in enumerate(dev)]
    assert P.tb0r(GATES_MOD, fb, [lab], [write("dev_s.jsonl", sha_bad)], len(dev), str(W["tmp"] / "t4.json"))["status"] == "FAIL"   # dev row not on the fed traj


def test_cli_exit_codes(W):
    assert P.main(["feed", "--bank-dir", W["bank"], "--join", W["join"], "--expect-join-sha256", W["jsha"], "--platform", "A=devbox", "--platform", "B=devbox",
                   "--platform", "C=pod", "--r0-rows", W["r0"][0], "--min-rj5-coverage", "0.5", "--out-dir", str(W["tmp"] / "cli")]) == 0
    assert P.main(["feed", "--bank-dir", W["bank"], "--join", W["join"], "--expect-join-sha256", W["jsha"], "--platform", "A=devbox", "--platform", "B=devbox",
                   "--platform", "C=pod", "--r0-rows", W["r0"][0], "--min-rj5-coverage", "0.99", "--out-dir", str(W["tmp"] / "cli2")]) == 3
    assert P.main(["feed", "--bank-dir", W["bank"], "--join", W["join"], "--expect-join-sha256", "0" * 64, "--platform", "A=devbox", "--r0-rows", W["r0"][0],
                   "--min-rj5-coverage", "0.5", "--out-dir", str(W["tmp"] / "cli3")]) == 2
