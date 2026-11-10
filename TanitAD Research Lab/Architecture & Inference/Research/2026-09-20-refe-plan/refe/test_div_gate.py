#!/usr/bin/env python3
"""DIV launch-gate tests (CPU, no data).  Prints ZZTEST_PASS.  PREREG_DIV_P6 (frozen 2026-11-10, sha256 e5fb2f8e...): DV0 = C0's argv + `--split-half 0` (seed 0), DV1 = C0r's argv + `--split-half 1` (seed 1); the split keeps only the
training sets whose LOG hashes to the half, h = int(sha256(log name), 16) % 2; a half below 25,000 sets is a launch blocker; the held-out monitor is unchanged.

  V1 THE SPLIT (scorer_finetune.split_half_of / split_training_index / split_refusals): the hash rule pinned by literals, a hand-made log list -> the exact halves (disjoint, covering, every set follows its log), the audit re-hashes every kept row,
     the 25,000 floor at 24,999 / 25,000, a missing log name raises, the flag defaults OFF (-1) and takes only -1 / 0 / 1.
  V2 MUTATION: a trainer copy with the filter REMOVED, and one with the audit's off-half count zeroed, are both still refused by split_refusals; the NOFILTER regression (no_filter=True) is refused.
  V3 REACHABILITY from the caller: main() applies split_training_index to tr_idx under a.split_half, before the data event and before the DataLoader, and writes info["split"]; `wired()` is True on the real source and FALSE on a copy with the call deleted
     and on one with the data-event line deleted; the namespace read of split_half is seen by G-DVB's dead-lever check.
  V4 G-DVB (gate_checks): the new flag has a registry entry (a copy of the table WITHOUT it is refused as an unknown flag), its rows are green on a good data event and RED on each of 8 mutated events (no block, wrong half, filter not built, rows off the half,
     sets != train_sets, a half below 25,000, wrong rule, a block with the flag off); a trainer whose --split-half DEFAULT is 0 turns the control (C0's own argv, no flag) RED.
  V5 PROFILE refe_div_p6: DV0 / DV1 argvs legal; argv diff vs C0 / C0r is exactly the split flag + --out (DV1 vs C0: + --seed); refused: split flag missing / wrong half / wrong seed / wrong --out / C0's own argv / extra lever / lane flag / wrong ckpt.
  V6 FINGERPRINTS: the 24 older profile digests unmoved, the new one stable, 25 profiles.
  V7 LAYER: div_layer.peel(trainer) is the tip blob; s1zsm_layer.peel (which peels DIV first) still gives the S1ZSM tip blob; one changed character inside any reversal makes peel REFUSE.
  V8 SCRIPTS: bash -n x3; div_pod.sh vs launch_gate.div_argv vs scorer1p6_pod.sh; the chain's split audit / run-check python run on synthetic logs (good passes, off-half rows / a small half / a wrong half fail); chain and pause structure; MUTATIONS RED.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402,F401
import div_layer as DL  # noqa: E402
import gate_checks as C  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402
import s1zsm_layer as ZSM  # noqa: E402
import scorer_finetune as SF  # noqa: E402

PD = L.PROFILES["refe_div_p6"]
SHA = {L.SCORER1P6_CKPT: L.SCORER1P6_CKPT_SHA256}
for _root in (L.SCORER1P6_ROOT, L.DIV_ROOT):
    for _k, _f in (("--navtest-logs", _root + "/navtest_ids/navtest_logs.json"), ("--navtest-tokens", _root + "/navtest_ids/navtest_tokens.txt.gz")):
        SHA[_f] = L.SCORER1_NAVTEST_SHA256[_k]


def fake_sha(path):
    if path in SHA:
        return SHA[path]
    raise FileNotFoundError(path)


L.sha256_of_file = fake_sha


def refusals(argv, prof=PD):
    return L.profile_refusals(prof, argv)


def has(rs, needle):
    return any(needle in r for r in rs)


def swap(argv, flag, *vals):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            out += [flag] + list(vals)
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


def drop(argv, flag):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


def pairs(toks):
    return sorted((f, tuple(v)) for f, v in GT.flag_pairs(list(toks)))


def differing_flags(a, b):
    da, db = {}, {}
    for f, v in GT.flag_pairs(list(a)):
        da.setdefault(f, []).append(tuple(v))
    for f, v in GT.flag_pairs(list(b)):
        db.setdefault(f, []).append(tuple(v))
    return {f for f in set(da) | set(db) if da.get(f) != db.get(f)}


def blob(src: str) -> str:
    b = src.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def load_copy(src_text, name, directory):
    p = os.path.join(directory, name + ".py")
    open(p, "w", encoding="utf-8", newline="").write(src_text)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TRAINER = open(os.path.join(HERE, "scorer_finetune.py"), "rb").read().decode("utf-8").replace("\r\n", "\n")
tmp = os.environ.get("SCORER1_TMP") or None

# ---------------------------------------------------------------- V1 the split
assert SF.SPLIT_MIN_SETS == 25000 and C.SPLIT_MIN_SETS_REGISTERED == 25000 and L.DIV_MIN_SETS == 25000
assert SF.SPLIT_RULE == C.SPLIT_RULE_REGISTERED
for nm, want in (("a", 1), ("log_0001", 0), ("2021.05.12.22.00.38_veh-35_01008_01518", 0)):          # pinned literals: a changed rule (md5, sha1, a prefix, a different modulus) moves one of them
    assert SF.split_half_of(nm) == want == int(hashlib.sha256(nm.encode("utf-8")).hexdigest(), 16) % 2, nm
for bad in (None, "", 7, b"x"):
    try:
        SF.split_half_of(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("a training row without a usable log name was hashed: %r" % (bad,))
names = ["log_%04d" % i for i in range(300)] + ["2021.05.12.22.00.38_veh-35_01008_01518", "a"]
expect = {n: int(hashlib.sha256(n.encode()).hexdigest(), 16) % 2 for n in names}
assert 100 < sum(expect.values()) < 200, "both halves must be populated"
rows = [{"log_name": n, "token": "t%d" % k} for n in names for k in range(3)]               # every log has 3 sets
idx = list(range(len(rows)))
lo = lambda i: rows[i]["log_name"]  # noqa: E731
kept = {}
for h in (0, 1):
    kept[h], facts = SF.split_training_index(lo, idx, h)
    assert kept[h] == [i for i in idx if expect[rows[i]["log_name"]] == h] and all(expect[lo(i)] == h for i in kept[h])
    assert facts["rows_off_half"] == 0 and facts["filter_built"] is True and facts["half"] == h and facts["rule"] == SF.SPLIT_RULE
    assert facts["sets"] == len(kept[h]) == 3 * sum(1 for n in names if expect[n] == h) and facts["logs"] == sum(1 for n in names if expect[n] == h)
    assert facts["pre_split_sets"] == len(rows) and facts["half_sets"] == {"0": 3 * sum(1 for n in names if expect[n] == 0), "1": 3 * sum(1 for n in names if expect[n] == 1)}
    assert facts["half_logs"] == {"0": sum(1 for n in names if expect[n] == 0), "1": sum(1 for n in names if expect[n] == 1)}
    assert all(sum(1 for i in kept[h] if lo(i) == n) in (0, 3) for n in names), "every set of a log follows its log"
    json.dumps(facts)                                                                            # the data event is JSON
assert not set(kept[0]) & set(kept[1]) and sorted(kept[0] + kept[1]) == idx, "the halves are disjoint and cover the index"
sub_idx = idx[::2]                                                                               # a training index that is a SUBSET of the rows (the filters before the split)
k2, f2 = SF.split_training_index(lo, sub_idx, 1)
assert k2 == [i for i in sub_idx if expect[lo(i)] == 1] and f2["pre_split_sets"] == len(sub_idx) and f2["rows_off_half"] == 0
# the 25,000 floor: exact boundary, both halves count
n0 = next(n for n in names if expect[n] == 0)
n1 = next(n for n in names if expect[n] == 1)


def facts_for(c0, c1, half):
    r = [{"log_name": n0}] * c0 + [{"log_name": n1}] * c1
    _, f = SF.split_training_index(lambda i: r[i]["log_name"], list(range(len(r))), half)
    return f


assert SF.split_refusals(facts_for(25000, 25000, 0)) == [] and SF.split_refusals(facts_for(25000, 25000, 1)) == []
r_lo = SF.split_refusals(facts_for(24999, 30000, 0))
assert has(r_lo, "fewer than 25000") and has(r_lo, "launch blocker"), r_lo
assert has(SF.split_refusals(facts_for(30000, 24999, 0)), "fewer than 25000"), "the OTHER half below the floor blocks the arm too (both arms must run)"
assert has(SF.split_refusals(facts_for(30000, 24999, 1)), "fewer than 25000")
# the audit: forged facts are refused
good = facts_for(25000, 26000, 0)
assert SF.split_refusals(good) == []
assert has(SF.split_refusals({**good, "rows_off_half": 1}), "other half") and has(SF.split_refusals({**good, "sets": good["sets"] + 5}), "not half 0")
ap = SF.build_parser()
ns0 = ap.parse_args(["--ckpt", "c", "--targets", "t", "--images", "i", "--calib", "k", "--onpolicy", "o", "--heldout", "h", "--heldout-logs", "l", "--out", "x"])
assert ns0.split_half == -1, "the split defaults OFF"
for bad in ("2", "-2", "x"):
    try:
        with open(os.devnull, "w", encoding="utf-8") as dn:
            old, sys.stderr = sys.stderr, dn
            try:
                ap.parse_args(["--ckpt", "c", "--targets", "t", "--images", "i", "--calib", "k", "--onpolicy", "o", "--heldout", "h", "--heldout-logs", "l", "--out", "x", "--split-half", bad])
            finally:
                sys.stderr = old
    except SystemExit:
        pass
    else:
        raise AssertionError("--split-half %s was accepted" % bad)
print("V1 ok (the hash rule by literals; exact halves; the audit re-hashes; the 25,000 floor at 24,999 / 25,000; default OFF)")

# ---------------------------------------------------------------- V2 mutation of the filter and of the audit
FILT = "    kept = list(tr_idx) if no_filter else [i for i in tr_idx if split_half_of(log_of(i)) == half]\n"
assert TRAINER.count(FILT) == 1
AUDIT = '"rows_off_half": sum(1 for i in kept if split_half_of(log_of(i)) != half)'
assert TRAINER.count(AUDIT) == 1
with tempfile.TemporaryDirectory(dir=tmp) as td:
    m_nof = load_copy(TRAINER.replace(FILT, "    kept = list(tr_idx)\n"), "sf_mut_nofilter", td)                  # the filter line REMOVED from the function
    k_, f_ = m_nof.split_training_index(lo, idx, 1)
    assert k_ == idx and f_["rows_off_half"] > 0 and has(m_nof.split_refusals(f_), "other half"), "MUTATION: a removed filter must be refused by the audit"
    m_aud = load_copy(TRAINER.replace(AUDIT, '"rows_off_half": 0'), "sf_mut_noaudit", td)                       # the audit's count zeroed AND the filter removed: the second rule (kept != the half's sets) still refuses
    m_aud2 = load_copy(TRAINER.replace(AUDIT, '"rows_off_half": 0').replace(FILT, "    kept = list(tr_idx)\n"), "sf_mut_both", td)
    k_, f_ = m_aud2.split_training_index(lo, idx, 1)
    assert f_["rows_off_half"] == 0 and k_ == idx and has(m_aud2.split_refusals(f_), "not half 1"), "MUTATION: filter removed AND the off-half count zeroed is still refused"
    k_, f_ = m_aud.split_training_index(lo, idx, 1)
    assert k_ == kept[1] and [r for r in m_aud.split_refusals(f_) if "fewer than" not in r] == [], "(the audit-only mutant keeps a correct filter: nothing to refuse)"
kn, fn = SF.split_training_index(lo, idx, 1, no_filter=True)                                       # THE NOFILTER REGRESSION (REFE_SPLIT_MUTATE_NOFILTER)
assert kn == idx and fn["filter_built"] is False and fn["rows_off_half"] == len(idx) - len(kept[1]) > 0 and has(SF.split_refusals(fn), "other half")
assert SF.SPLIT_MUTATE_ENV == "REFE_SPLIT_MUTATE_NOFILTER"
print("V2 ok (a removed filter, a removed filter + zeroed audit, and the NOFILTER regression are all refused)")


# ---------------------------------------------------------------- V3 reachability from the caller
def wired(src: str) -> bool:
    """True iff main() applies the split to the TRAINING index before the data event and the loader, under a.split_half and the mutate env, and the block reaches the data event"""
    m = re.search(r"^def main\(\) -> int:\n(.*?)^\S", src + "\nX", re.S | re.M)
    body = m.group(1) if m else ""
    call = body.find("tr_idx, split_info = split_training_index(")
    gate = body.find("a.split_half >= 0")
    event = body.find('info = {"event": "data"')
    loader = body.find("dl = DataLoader(make_train_set(ds, tr_idx")
    put = body.find('info["split"] = split_info')
    logi = body.find("    log(info)\n")
    ref = body.find("bad_split = split_refusals(split_info)")
    return (0 <= gate < call < ref < event < put < logi < loader and "SPLIT_MUTATE_ENV" in body[call:call + 400] and "a.split_half" in body[call:call + 400] and "return 4" in body[ref:ref + 300]
            and body.find("ho_idx = [i for i") < call)


assert wired(TRAINER), "main() does not apply the split to the training index"
assert not wired(TRAINER.replace("tr_idx, split_info = split_training_index(", "_unused, split_info = split_training_index(")), "MUTATION: the result never reaches tr_idx"
assert not wired(TRAINER.replace('        info["split"] = split_info\n', "        pass\n")), "MUTATION: the data event loses the split block"
assert not wired(TRAINER.replace("        if bad_split:\n", "        if False:\n").replace("bad_split = split_refusals(split_info)", "bad_split = []")), "MUTATION: the refusal is never raised"
assert "split_half" in C.namespace_reads(TRAINER, C.SCORER_NAMESPACE_FUNCS), "the dead-lever check must see the trainer read a.split_half"
assert "split_half" not in C.namespace_reads(TRAINER.replace("a.split_half >= 0", "False").replace("a.split_half, os.environ", "0, os.environ"), C.SCORER_NAMESPACE_FUNCS), "a trainer that never reads the flag is a dead lever"
print("V3 ok (main() applies the split to tr_idx before the event and the loader; wired() RED on 3 mutants; the dead-lever check sees the read)")

# ---------------------------------------------------------------- V4 G-DVB
DA = {a: L.div_argv(a) for a in ("DV0", "DV1")}
C0A, C0RA = L.scorer1p6_argv("C0"), L.scorer1p6_argv("C0r")


def parse(argv):
    ns, err = C.parse_scorer_argv(SF.build_parser(), argv)
    assert err is None or not err, err
    return ns


def nest(d, dotted, v):
    cur = d
    ks = dotted.split(".")
    for k in ks[:-1]:
        cur = cur.setdefault(k, {})
    cur[ks[-1]] = v


def good_data(half, n=31000, other=29000):
    d = {"b_mode": "expreward", "b_compw": "1,1,1,1,1,1", "train_sets": n}
    for k, v in PD["data_expect"].items():
        nest(d, k, v)
    hs = {"0": n if half == 0 else other, "1": n if half == 1 else other}
    d["split"] = {"half": half, "rule": SF.SPLIT_RULE, "min_sets": 25000, "filter_built": True, "sets": n, "logs": 410, "rows_off_half": 0, "pre_split_sets": n + other,
                  "half_sets": hs, "half_logs": {str(half): 410, str(1 - half): 400}}
    return d


def split_rows(argv, data):
    return [r for r in C.scorer_dvb_rows(parse(argv), data, PD) if r["dest"] == "split_half"]


for a, h in (("DV0", 0), ("DV1", 1)):
    rr = split_rows(DA[a], good_data(h))
    assert len(rr) == 7 and all(r["ok"] for r in rr), [r for r in rr if not r["ok"]]
muts = {
    "no block": lambda d: {k: v for k, v in d.items() if k != "split"},
    "wrong half": lambda d: {**d, "split": {**d["split"], "half": 1 - d["split"]["half"]}},
    "filter not built": lambda d: {**d, "split": {**d["split"], "filter_built": False}},
    "rows off the half": lambda d: {**d, "split": {**d["split"], "rows_off_half": 12}},
    "sets != train_sets": lambda d: {**d, "train_sets": d["train_sets"] + 1},
    "a half below 25,000": lambda d: {**d, "split": {**d["split"], "half_sets": {"0": 24999, "1": 31000}}},
    "wrong rule": lambda d: {**d, "split": {**d["split"], "rule": "int(md5(log_name)) % 2"}},
}
for nm, f in muts.items():
    rr = split_rows(DA["DV0"], f(good_data(0)))
    assert any(not r["ok"] for r in rr), "MUTATION %r must turn a split row RED" % nm
rr = split_rows(C0A, good_data(0))                                                                   # a block with the flag OFF: RED (`iff`)
assert any(not r["ok"] for r in rr), "a split block under an argv without the flag"
rr = split_rows(C0A, {k: v for k, v in good_data(0).items() if k != "split"})
assert rr == [], "flag OFF and no block: no split row at all (every other profile's G-DVB is unchanged)"
# the registry entry: a copy of the table WITHOUT the new flag is refused as an unknown flag by check_dvb_scorer (a new flag without a G-DVB entry is refused by the gate itself)
assert C.DVB_KINDS_SCORER["split_half"] == "built"
dests = {x.dest for x in SF.build_parser()._actions if x.dest != "help"}
assert set(C.DVB_KINDS_SCORER) == dests, (dests ^ set(C.DVB_KINDS_SCORER))
v_ok = C.check_dvb_scorer(DA["DV0"], {"data": good_data(0)}, {**PD, "declared_arm": "DV0"})
print("   G-DVB verdict on a good DV0 data event:", v_ok.get("status") or v_ok.get("verdict"), "; reasons:", [r[:140] for r in (v_ok.get("reasons") or [])][:4])
assert not [r for r in (v_ok.get("reasons") or []) if "split" in r or "unknown" in r or "NO declared-vs-built" in r or "never read" in r], v_ok.get("reasons")
saved = dict(C.DVB_KINDS_SCORER)
del C.DVB_KINDS_SCORER["split_half"]
try:
    v_bad = C.check_dvb_scorer(DA["DV0"], {"data": good_data(0)}, {**PD, "declared_arm": "DV0"})
finally:
    C.DVB_KINDS_SCORER.clear()
    C.DVB_KINDS_SCORER.update(saved)
assert any("split_half" in r and "NO declared-vs-built" in r for r in (v_bad.get("reasons") or [])), "MUTATION: a flag without a registry entry must be refused: %s" % (v_bad.get("reasons"),)
v_mut = C.check_dvb_scorer(DA["DV0"], {"data": muts["rows off the half"](good_data(0))}, {**PD, "declared_arm": "DV0"})
assert any("--split-half" in r for r in (v_mut.get("reasons") or [])), "the full G-DVB check must go RED on the NOFILTER-shaped data event"
# the DEFAULT mutation: a trainer whose --split-half default is 0 would split C0 silently: C0's own argv (no flag) must read RED against a data event with no block
DEF = 'ap.add_argument("--split-half", type=int, default=-1,'
assert TRAINER.count(DEF) == 1
with tempfile.TemporaryDirectory(dir=tmp) as td:
    m_def = load_copy(TRAINER.replace(DEF, 'ap.add_argument("--split-half", type=int, default=0,'), "sf_mut_default", td)
    ns_def, err = C.parse_scorer_argv(m_def.build_parser(), C0A)
    assert not err and ns_def.split_half == 0
    rr = [r for r in C.scorer_dvb_rows(ns_def, {k: v for k, v in good_data(0).items() if k != "split"}, L.PROFILES["refe_scorer1p6"]) if r["dest"] == "split_half"]
    assert rr and any(not r["ok"] for r in rr), "MUTATION: a default of 0 must turn the control (C0's argv, no flag) RED"
assert parse(C0A).split_half == -1, "the real default leaves C0 unsplit"
print("V4 ok (7 rows green on both arms; 7 mutated data events + a block under flag OFF RED; a table without the flag refused; a default of 0 turns the control RED)")

# ---------------------------------------------------------------- V5 the profile
for a, parent, half, seed in (("DV0", "C0", 0, "0"), ("DV1", "C0r", 1, "1")):
    A, P = DA[a], L.scorer1p6_argv(parent)
    assert refusals(A) == [], (a, refusals(A))
    assert differing_flags(A, P) == {"--split-half", "--out"} and len(A) == len(P) + 2, differing_flags(A, P)
    i = A.index("--seed")
    assert A[i + 1] == seed and A[i + 2:i + 4] == ["--split-half", str(half)] and A.count("--split-half") == 1 == A.count("--out") and A.count("--max-updates") == 1
    assert A[A.index("--out") + 1] == L.DIV_ROOT + "/" + a.lower() and "--pair-w" not in A and "--twin-w" not in A
assert differing_flags(DA["DV1"], C0A) == {"--split-half", "--out", "--seed"}, "DV1 vs C0: the split flag, --out and the seed"
assert differing_flags(DA["DV0"], C0A) == {"--split-half", "--out"}
A0, A1 = DA["DV0"], DA["DV1"]
assert has(refusals(drop(A0, "--split-half")), "--split-half") and has(refusals(drop(A1, "--split-half")), "--split-half"), "the split flag must be NAMED"
assert has(refusals(swap(A0, "--split-half", "1")), "--split-half") and has(refusals(swap(A1, "--split-half", "0")), "--split-half"), "a wrong half"
assert has(refusals(swap(A0, "--split-half", "-1")), "--split-half"), "--split-half -1 is OFF: refused"
assert has(refusals(swap(A1, "--seed", "0")), "requires --seed") and has(refusals(swap(A0, "--seed", "1")), "requires --seed")
assert refusals(swap(A0, "--out", L.DIV_ROOT + "/dv1")) and refusals(swap(A0, "--out", L.SCORER1P6_ROOT + "/c0")) and has(refusals(swap(A0, "--max-updates", "800")), "--max-updates")
assert refusals(C0A) and refusals(C0RA), "C0's / C0r's own argvs are not DIV arms"
assert refusals(A0 + ["--twin-w", "1.0", "--twin-heads", "NC"]) and refusals(A0 + ["--pair-w", "0.5", "--pair-heads", "NC"]) and has(refusals(A0 + ["--lane-head"]), "--lane-head")
assert refusals(A0 + ["--split-half", "0"]), "the split flag twice"
assert refusals(A0, L.PROFILES["refe_scorer1p6"]) and refusals(A0, L.PROFILES["refe_rep_p6"]), "the DIV argvs are not arms of the parent's or the replicate profile"
SHA[L.SCORER1P6_CKPT] = "0" * 64
assert has(refusals(A0), "registered P6 checkpoint")
SHA[L.SCORER1P6_CKPT] = L.SCORER1P6_CKPT_SHA256
assert refusals(A0) == [] and refusals(A1) == []
for bad in ("C0", "C0r", "S1Z", "DV2", "", "dv0"):
    try:
        L.div_argv(bad)
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown arm %r was accepted" % bad)
assert L.div_arm_refusals({k: dict(v) for k, v in L.DIV_ARMS.items()}) == []
assert has(L.div_arm_refusals({"DV0": {"parent": "C0", "family": "p6", "half": 2}}), "not 0 or 1") and has(L.div_arm_refusals({"DV0": {"parent": "S1Z", "family": "p6", "half": 0}}), "not a C0 arm")
assert has(L.div_arm_refusals({"C0": {"parent": "C0", "family": "p6", "half": 0}}), "registered arm") and has(L.div_arm_refusals({"S1ZD": {"parent": "C0", "family": "p6", "half": 0}}), "registered arm")
assert has(L.div_arm_refusals({"DV0": {"parent": "C0", "family": "p6", "half": 0}, "DV1": {"parent": "C0r", "family": "p6", "half": 0}}), "same half")
assert PD["data_expect_min"] == {"train_sets": 25000} and "train_sets" not in PD["data_expect"] and PD["data_expect"]["train_sets_in_bank"] == 60000 and PD["data_expect_updates"] is None
assert PD["required_flags"] == {**L.PROFILES["refe_scorer1p6"]["required_flags"], "--split-half": None} and PD["arm_seed"] == {"DV0": 0, "DV1": 1} and PD["pair_spec"] == {}
assert PD["suite"][:-1] == L.SCORER1P6_SUITE and PD["suite"][-1] == {"script": "test_div_gate.py", "token": "ZZTEST_PASS"}
print("V5 ok (DV0 / DV1 legal; argv diff vs C0 / C0r = the split flag + --out (DV1 vs C0 also the seed); %d refusal cases)" % 20)

# ---------------------------------------------------------------- V6 fingerprints


def fp(n):
    return hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest()


TIP_FP = {"refe_p2": "627341286aea3506", "refe_p6": "80fa4d7ec757ad46", "refe_field": "5ddc49053a99fa78", "refe_field_ctrl": "0b6281b1bfa996ed", "refe_rehearsal": "821cd1ef3bf55d4e",
          "refe_rls1": "bce7e25ee8718c08", "refe_aux_s1": "f95f48727b62ccc6", "refe_p7": "94cca167ce121a94", "refe_p2rep": "44ca107f53aceee8", "refe_sft4r": "8972e3c348206b94",
          "refe_scorer1": "88d02fb7b34d4fd8", "refe_scorer1p6": "e74d64617f514d35", "refe_s1z_p6": "5c3bff457aa3bf1e", "refe_rep_p6": "dd348f44c9638f1a", "refe_rep_s1z_p6": "2356afd4efeeb52a",
          "refe_rls1b": "5d6bfbc10152ae96", "refe_rls1c": "26214cf954099d51", "refe_rls1d": "fcc2585f2a5d9b94", "refe_rls1e": "54d96289b19c3cd6", "refe_hnscreen": "d30126bbb5a32957",
          "refe_aux_s1c": "dcdf2dbd2a674f1c", "refe_s1zsm_p6": "86380fbb786daa75", "refe_rep_s1zsm_p6": "27fe236a2fd68ca4", "refe_dose_s1z_p6": "9085f9e5cf59c016"}      # MEASURED at the tip ee1ba225 (the 24 profiles before DIV)
moved = [n for n, want in TIP_FP.items() if not fp(n).startswith(want)]
assert not moved, f"existing profile fingerprints moved: {moved}"
assert set(L.PROFILES) - set(TIP_FP) == {"refe_div_p6"} and len(L.PROFILES) == 25
r = repr(sorted(PD.items()))
assert fp("refe_div_p6") == fp("refe_div_p6") and "function" not in r and "0x" not in r and fp("refe_div_p6") != fp("refe_scorer1p6")
base = L.PROFILES["refe_scorer1p6"]
assert PD is not base and PD["recipe"] is not base["recipe"] and PD["required_flags"] is not base["required_flags"] and PD["data_expect"] is not base["data_expect"]
for k in ("data_pin", "ckpt_sha256", "navtest_sha256", "forbidden_flags", "recipe", "data_flags"):
    assert PD[k] == base[k], k
print("V6 ok (the 24 older digests unmoved; NEW refe_div_p6 = %s...)" % fp("refe_div_p6")[:16])

# ---------------------------------------------------------------- V7 the layer
assert DL.has_layer(TRAINER) and DL.MARKER == "SPLIT_MIN_SETS = 25000"
pre = DL.peel(TRAINER)
assert not DL.has_layer(pre) and blob(pre) == DL.TIP_TRAINER_BLOB == "d665ed995d064f9e1cae956c3c48867cd348cd3d", blob(pre)
assert DL.peel(pre) == pre and "split_half" not in pre and "--split-half" not in pre
zp = ZSM.peel(TRAINER)
assert blob(zp) == ZSM.TIP_TRAINER_BLOB == "6a16d35c1818422ff514e97c4b292c32dde81795", "S1ZSM's pin still peels (DIV first, then S1ZSM)"
assert ZSM.peel(pre) == zp
for k, (new, old) in enumerate(DL.REVERSALS):
    bad = TRAINER.replace(new, new.replace("e", "E", 1), 1)
    try:
        DL.peel(bad)
    except AssertionError:
        continue
    raise AssertionError("DIV reversal %d: a drifted trainer was accepted" % k)
print("V7 ok (peel -> the tip blob d665ed99; S1ZSM's pin 6a16d35c unchanged; every reversal refuses a drifted trainer)")

# ---------------------------------------------------------------- V8 the scripts
bash = None
for cand in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
    if os.path.exists(cand):
        bash = cand
        break
bash = bash or shutil.which("bash")
pod, chain, pause = (os.path.join(HERE, n) for n in ("div_pod.sh", "div_chain.sh", "div_pause_p7.sh"))
chain_dose, ppod = os.path.join(HERE, "dose_chain.sh"), os.path.join(HERE, "scorer1p6_pod.sh")
for f in (pod, chain, pause, chain_dose, ppod):
    assert os.path.exists(f), f
    assert b"\r" not in open(f, "rb").read(), "LF only: " + f
    assert os.path.basename(f).split(".")[0].lower() not in ("aux", "con", "prn", "nul"), f
ctext = open(chain, "rb").read().decode("utf-8")
joined = "\n".join(ln for ln in ctext.split("\n") if not ln.lstrip().startswith("#"))
assert "PROFILE=refe_div_p6" in joined and "--profile $PROFILE" in joined and "ROOT=$DATA/refe_div" in joined and 'PKG=${DIV_PKG:-$WS/refe-plan-div}' in joined and "SPLIT_MIN=25000" in joined
assert L.SCORER1P6_CARD_SHA256 in ctext and "e5fb2f8e4123fcfe3a599b0cc71537bab187716c6a5d3e46735ce094bacf6fac" in ctext and L.SCORER1P6_CKPT_SHA256 in ctext
assert "DV0) PARENT=C0;  HALF=0;" in joined and "DV1) PARENT=C0r; HALF=1;" in joined and "ZZDIV_CHAIN_DONE" in joined and '--prefix "div/$LA"' in joined and "arm_div_$LA" in joined and "lfs_sha256" in joined
assert "MODE=mutate_nofilter" in joined and '[ "$mrc" = 4 ] && grep -q "^REFUSED: split:"' in joined and "the NOFILTER mutant was NOT refused" in joined and "ZZDIV_SPLIT" in joined and "split_is $OUT/preflight.log $HALF" in joined
assert "--max-updates" not in joined and "--seed" not in joined and "MODE=calibrate" not in joined and "--pair-calibrate" not in joined and "--twin" not in joined and "refe-plan-rep" not in joined and "refe-plan-dose" not in joined
assert not re.search(r"\bkill -|SIGSTOP|SIGCONT|pkill|taskkill|ZZP7|P7_PAUSE|pause_p7|resume_p7|lane_pause|lane_restart|refe_p7/", joined), "the chain neither pauses nor resumes P7"
assert ctext.index("mutate_nofilter") < ctext.index("launch_gate.py run") < ctext.index("ZZDIV_TRAIN_START") < ctext.index("hf_egress.py push"), "the mutant is checked BEFORE the gate and the run"


def fn_body(text, name):
    m1 = re.search(r"^%s\(\) \{.*$" % re.escape(name), text, re.M)
    if m1 and "; }" in m1.group(0):
        return m1.group(0)
    m = re.search(r"^%s\(\) \{.*?^\}" % re.escape(name), text, re.S | re.M)
    assert m, name
    return m.group(0)


dt = open(chain_dose, "rb").read().decode("utf-8")
nr = lambda t: t.replace("DOSE_", "DIV_")          # noqa: E731
for name in ("gpu_busy", "lock_held", "wait_clear", "gate_is", "stage_done", "stage_mark", "tree_sha", "refuse", "token_ok"):
    assert nr(fn_body(dt, name)) == fn_body(ctext, name), name                       # the wait / marker functions are dose_chain's, byte for byte (DOSE_ -> DIV_ only)
assert nr(fn_body(dt, "others_hold")).replace("refe_rep; do", "refe_rep refe_dose; do") == fn_body(ctext, "others_hold"), "others_hold differs by exactly the one added lock (refe_dose)"
ps = open(pause, "rb").read().decode("utf-8")
assert "trap 'resume_p7' EXIT" in ps and "HOLD_P7" in ps and "div_chain.sh" in ps and "STOPPED=1" in ps and "P7_RESUMED" in ps and "timeout 9000" in ps and "refe_div" in ps and "DV0|DV1) ;;" in ps
assert ps.index("STOPPED=1") < ps.index("kill -TERM $CHAIN") < ps.index("for A in $ARMS") < ps.rindex("resume_p7") and ps.count("exit 1") >= 3 and "PAUSE_STOPPED_EARLY" in ps
assert 'grep -q "ZZDIV_CHAIN_DONE $A"' in ps and "break" in ps and "refe-plan-dose" not in ps and "dose_chain" not in ps, "DV1 does not run after a failed DV0"
ptxt = open(pod, encoding="utf-8").read()
assert "scorer1p6_pod.sh" in ptxt and "DV0) PARENT=C0; HALF=0; SEED=0" in ptxt and "DV1) PARENT=C0r; HALF=1; SEED=1" in ptxt and "REFE_SPLIT_MUTATE_NOFILTER=1" in ptxt and "--pair-calibrate" not in ptxt.replace("# ", "")

# the chain's own python: split_is and the run check, run on synthetic logs
def heredoc(text, anchor):
    i = text.index(anchor)
    j = text.index("<<'PYEOF'\n", i) + len("<<'PYEOF'\n")
    return text[j:text.index("\nPYEOF", j)]


SPLIT_PY = heredoc(ctext, "split_is() {")
RUN_PY = heredoc(ctext, '> "$OUT/run_check.log"')


def run_py(code, *args):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=tmp, encoding="utf-8") as f:
        f.write(code)
    try:
        return subprocess.run([sys.executable, f.name, *map(str, args)], capture_output=True, encoding="utf-8")
    finally:
        os.unlink(f.name)


with tempfile.TemporaryDirectory(dir=tmp) as td:
    def write_log(name, data):
        p = os.path.join(td, name)
        open(p, "w", encoding="utf-8").write("noise line\n" + json.dumps({"event": "data", **data}) + "\n")
        return p
    for half in (0, 1):
        g = good_data(half)
        r = run_py(SPLIT_PY, write_log("g%d.log" % half, g), half, 25000)
        assert r.returncode == 0 and "half=%d sets=31000" % half in r.stdout and "half0 sets=" in r.stdout, (r.returncode, r.stdout, r.stderr[:200])
        for nm, f in muts.items():
            if nm in ("sets != train_sets", "wrong rule", "no block", "wrong half", "filter not built", "rows off the half", "a half below 25,000"):
                r = run_py(SPLIT_PY, write_log("m.log", f(g)), half, 25000)
                assert r.returncode != 0, "MUTATION %r must fail the chain's split audit (half %d)" % (nm, half)
    r = run_py(SPLIT_PY, write_log("w.log", good_data(0)), 1, 25000)
    assert r.returncode != 0, "a half-0 run audited as half 1 fails"
    # the run check
    def make_out(updates, n, half=0, off=0, models=True, done=True):
        o = os.path.join(td, "out_%d_%d_%d_%d" % (updates, n, half, off))
        os.makedirs(o, exist_ok=True)
        d = good_data(half, n=n)
        d["split"]["rows_off_half"] = off
        rows = [{"event": "data", **d}, {"event": "start", "updates": updates, "train_sets": n}] + ([{"event": "done", "updates": updates}] if done else [])
        open(os.path.join(o, "sft.jsonl"), "w", encoding="utf-8").write("\n".join(json.dumps(x) for x in rows) + "\n")
        open(os.path.join(o, "sft.log"), "w", encoding="utf-8").write("ZZSFT_DONE\n")
        if models:
            for m in ("model_sft_A.pt", "model_sft_B.pt"):
                open(os.path.join(o, m), "wb").write(b"x")
        return o
    r = run_py(RUN_PY, make_out(400, 31000), 0, 25000)
    assert r.returncode == 0 and "ZZDIV_RUN_OK updates 400 half 0 sets 31000" in r.stdout, (r.stdout, r.stderr[:300])
    r = run_py(RUN_PY, make_out(391, 25000), 0, 25000)
    assert r.returncode == 0 and "stops at update 391" in r.stdout, (r.stdout, r.stderr[:300])                 # a half of 25,000 sets: its epoch is 391 updates (stated)
    for nm, o in (("too few updates", make_out(399, 31000)), ("off-half rows", make_out(400, 31000, off=3)), ("wrong half", make_out(400, 31000, half=1)), ("no model", make_out(400, 31001, models=False)),
                  ("no done", make_out(400, 31002, done=False)), ("half below the floor", make_out(400, 24999)), ("391 where 400 is due", make_out(391, 31003))):
        r = run_py(RUN_PY, o, 0, 25000)
        assert r.returncode != 0, "the run check must fail: " + nm
print("V8a ok (the chain's split audit and run check: good logs pass, 12 mutated ones fail)")

if bash:
    for sh in (pod, chain, pause):
        r = subprocess.run([bash, "-n", sh], capture_output=True, encoding="utf-8")
        assert r.returncode == 0, (sh, r.stderr[:300])

    def run_pod(arm, ws, script=pod, mode="argv", extra=None):
        env = dict(os.environ, ARM=arm, MODE=mode, DIV_WS=ws, SCORER1P6_WS=ws)
        env.update(extra or {})
        return subprocess.run([bash, script], env=env, capture_output=True, encoding="utf-8")

    def norm(out, ws):
        return [t.replace(ws, "/workspace") for t in out.strip().split()]

    with tempfile.TemporaryDirectory(dir=tmp) as td:
        wsx = td.replace(chr(92), "/")
        for a, parent in (("DV0", "C0"), ("DV1", "C0r")):
            r = run_pod(a, wsx)
            assert r.returncode == 0, r.stderr[:300]
            out = norm(r.stdout, wsx)
            assert pairs(out) == pairs(L.div_argv(a)), set(pairs(out)) ^ set(pairs(L.div_argv(a)))                      # the pod script's argv IS launch_gate's
            pr = subprocess.run([bash, ppod], env=dict(os.environ, ARM=parent, MODE="argv", SCORER1P6_WS=wsx), capture_output=True, encoding="utf-8")
            assert pr.returncode == 0, pr.stderr[:300]
            pout = norm(pr.stdout, wsx)
            assert differing_flags(out, pout) == {"--split-half", "--out"} and len(out) == len(pout) + 2, differing_flags(out, pout)       # the pod scripts' own output: the split flag and --out only
            assert out[out.index("--seed") + 1] == pout[pout.index("--seed") + 1] == ("0" if a == "DV0" else "1") and out[out.index("--split-half") + 1] == ("0" if a == "DV0" else "1")
            assert " ".join(out).count("--split-half") == 1 and refusals(out) == []
        for bad in ("C0", "C0r", "S1ZD", "", "dv0"):
            r = run_pod(bad, "/workspace")
            assert r.returncode == 2 and "unknown ARM" in r.stderr, (bad, r.returncode, r.stderr[:200])
            r = subprocess.run([bash, chain] + ([bad] if bad else []), env=dict(os.environ, DIV_WS="/nonexistent_ws"), capture_output=True, encoding="utf-8")
            assert r.returncode == 2 and "REFUSED" in r.stdout and "unknown arm" in r.stdout, (bad, r.returncode, r.stdout[:200])
        for m in ("calibrate", "mutate_noradius", "mutate_nobalance", "bogus"):
            r = run_pod("DV0", wsx, mode=m)
            assert r.returncode == 2 and "unknown MODE" in r.stderr, (m, r.returncode, r.stderr[:200])
        # ---- MUTATIONS of the pod script: each must turn a check RED
        os.makedirs(os.path.join(td, "red"))
        for n in ("div_pod.sh", "scorer1p6_pod.sh"):
            shutil.copyfile(os.path.join(HERE, n), os.path.join(td, "red", n))
        rp = os.path.join(td, "red", "div_pod.sh")
        txt = open(rp, encoding="utf-8", newline="").read()
        a1 = '"s# --out [^ ]+# --split-half $HALF --out $OUT#"'
        assert txt.count(a1) == 1
        for label, new in (("the split flag dropped", '"s# --out [^ ]+# --out $OUT#"'), ("the wrong half", '"s# --out [^ ]+# --split-half 1 --out $OUT#"')):
            open(rp, "w", encoding="utf-8", newline="").write(txt.replace(a1, new))
            r = run_pod("DV0", wsx, script=rp)
            assert r.returncode == 0
            mo = norm(r.stdout, wsx)
            assert pairs(mo) != pairs(L.div_argv("DV0")), "MUTATION %s: the pod argv must differ from launch_gate's" % label
            assert refusals(mo), "MUTATION %s: the real profile must refuse that argv" % label
        # a parent whose seed is not the card's is refused (DV1 must carry C0r's seed 1)
        sp = os.path.join(td, "red", "scorer1p6_pod.sh")
        st = open(sp, encoding="utf-8", newline="").read()
        assert "  C0r) SEED=1 ;;" in st
        open(sp, "w", encoding="utf-8", newline="").write(st.replace("  C0r) SEED=1 ;;", "  C0r) SEED=5 ;;"))
        open(rp, "w", encoding="utf-8", newline="").write(txt)
        r = run_pod("DV1", wsx, script=rp)
        assert r.returncode == 2 and "not the 1 the card gives" in r.stderr, (r.returncode, r.stderr[:200])
        open(sp, "w", encoding="utf-8", newline="").write(st.replace("--max-updates 400 --eval-every", "--split-half 0 --max-updates 400 --eval-every"))
        r = run_pod("DV0", wsx, script=rp)
        assert r.returncode == 2 and "already names --split-half" in r.stderr, (r.returncode, r.stderr[:200])
    # launch_gate mutations (a mutated COPY of the module, loaded under another name; the real module is untouched)
    src = open(os.path.join(HERE, "launch_gate.py"), encoding="utf-8", newline="").read()

    def load_mut(a, b):
        assert src.count(a) == 1, a
        with tempfile.TemporaryDirectory(dir=tmp) as tdm:
            p = os.path.join(tdm, "launch_gate_mut.py")
            open(p, "w", encoding="utf-8", newline="").write(src.replace(a, b))
            spec = importlib.util.spec_from_file_location("launch_gate_mut", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m

    mut_nosplit = load_mut('    toks[i + 2:i + 2] = ["--split-half", str(sp["half"])]\n', '')                      # div_argv never adds the flag: the argv IS C0's
    assert differing_flags(mut_nosplit.div_argv("DV0"), C0A) == {"--out"}, "MUTATION: no split flag in div_argv must turn the argv-diff RED"
    mut_nosplit.sha256_of_file = fake_sha
    mut_nosplit.HERE = L.HERE                                                                          # the mutant copy lives in a temp dir; its sibling sources are the real ones
    assert has(mut_nosplit.profile_refusals(mut_nosplit.PROFILES["refe_div_p6"], mut_nosplit.div_argv("DV0")), "--split-half"), "the profile refuses an argv without the flag"
    mut_half = load_mut('"DV0": {"parent": "C0", "family": "p6", "half": 0}, "DV1": {"parent": "C0r", "family": "p6", "half": 1}}', '"DV0": {"parent": "C0", "family": "p6", "half": 1}, "DV1": {"parent": "C0r", "family": "p6", "half": 0}}')
    assert pairs(mut_half.div_argv("DV0")) != pairs(L.div_argv("DV0")) and differing_flags(mut_half.div_argv("DV0"), L.div_argv("DV0")) == {"--split-half"}
    mut_req = load_mut('"required_flags": {**PROFILES["refe_scorer1p6"]["required_flags"], "--split-half": None}}', '"required_flags": {**PROFILES["refe_scorer1p6"]["required_flags"]}}')
    assert "--split-half" not in mut_req.PROFILES["refe_div_p6"]["required_flags"] and "--split-half" in PD["required_flags"]
    assert hashlib.sha256(repr(sorted(mut_req.PROFILES["refe_div_p6"].items())).encode()).hexdigest() != fp("refe_div_p6"), "the digest moves with the required flag"
    mut_min = load_mut('"data_expect_min": {"train_sets": DIV_MIN_SETS}', '"data_expect_min": {}')
    assert PD["data_expect_min"] == {"train_sets": 25000} and mut_min.PROFILES["refe_div_p6"]["data_expect_min"] == {}
    print("V8 ok (bash -n x3; div_pod.sh vs launch_gate vs scorer1p6_pod.sh; chain + pause structure; pod-script and launch_gate mutations RED)")
else:
    print("V8 SKIPPED (no bash)")
print("ZZTEST_PASS")
