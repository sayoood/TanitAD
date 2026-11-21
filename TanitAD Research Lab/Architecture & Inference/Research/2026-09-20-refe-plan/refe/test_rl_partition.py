#!/usr/bin/env python3
"""Tests for refe/rl_partition.py (RL-S1 priority 2).  Run: python refe/test_rl_partition.py  ->  ZZTEST_PASS <n> (prints the mutation table).

The expected values are HAND-DERIVED from the synthetic pool below (the arithmetic is next to each), or recomputed here with hashlib,
never read back from the module under test.  Each mutation edits the module's SOURCE (the snippet must occur exactly once, else the run
errors: a vacuous mutation proves nothing) and the suite must then report >= 1 failing check.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "rl_partition.py")


def load(src: str, name: str):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


# ---------------------------------------------------------------------------------------------- the synthetic pool
# 40 logs L00..L39, log k has 8 + 2*(k % 7) scenes:  per 7-log cycle 8+10+12+14+16+18+20 = 98;  k = 0..34 is 5 cycles = 490;
# k = 35..39 adds the first five sizes 8+10+12+14+16 = 60;  total 550.
SZ = lambda k: 8 + 2 * (k % 7)                                                                # noqa: E731
LOGS = [f"L{k:02d}" for k in range(40)]
SCENES = [(f"tok_{k:02d}_{j:02d}", LOGS[k]) for k in range(40) for j in range(SZ(k))]
ORPHANS = [(f"orphan_{i}", "") for i in range(3)]                                              # no known log
NAVTEST_TOKENS = {"tok_00_00", "tok_00_01"}                                                    # 2 tokens inside an otherwise ordinary log
NAVTEST_LOGS = {"L01", "NT_ONLY_LOG"}                                                          # L01 has SZ(1) = 10 scenes
HELDOUT = {"L02", "L03"}                                                                       # 12 + 14 = 26 scenes
TRAINED = {f"tok_04_{j:02d}" for j in range(5)} | {f"tok_05_{j:02d}" for j in range(3)}       # 5 + 3 = 8 tokens
EXPECT_TALLY = {"input_scenes": 550 + 3, "duplicate_row": 1, "log_unknown": 3, "navtest_token": 2, "navtest_log": 10, "sft_heldout_log": 26,
                "trained_token": 8,
                "eligible_scenes": 550 - 2 - 10 - 26 - 8,                                      # = 504
                "eligible_logs": 40 - 3}                                                       # L01, L02, L03 gone = 37
N_TRAIN, N_VAL, MIN_VAL_LOGS, SEED = 100, 32, 6, 7                                             # cap = 32 // 6 = 5; 6 logs x 5 = 30 < 32 -> a 7th log is trimmed to 2
KW = dict(navtest_tokens=NAVTEST_TOKENS, navtest_logs=NAVTEST_LOGS, heldout_logs=HELDOUT, trained_tokens=TRAINED)


def H(seed, kind, key):
    return hashlib.sha256(f"{seed}|{kind}|{key}".encode()).hexdigest()


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def write_inputs(d, with_trained=True):
    pool = os.path.join(d, "pool.tsv.gz")
    with gzip.open(pool, "wt", encoding="utf-8", newline="") as f:
        f.write("token\tlog\textra\n")
        for t, lg in SCENES + ORPHANS + [SCENES[0]]:
            f.write(f"{t}\t{lg}\tx\n")
    nt = os.path.join(d, "navtest.json")
    json.dump({"tokens": sorted(NAVTEST_TOKENS), "token_log": {"nt_a": "L01", "nt_b": "NT_ONLY_LOG"}}, open(nt, "w"))
    ho = os.path.join(d, "heldout.txt")
    open(ho, "w").write("\n".join(sorted(HELDOUT)) + "\n")
    tr = os.path.join(d, "trained.tsv")
    with open(tr, "w", newline="") as f:
        f.write("token\tstep\trank\n")
        for t in sorted(TRAINED):
            f.write(f"{t}\t0\t0\n")
    return pool, nt, ho, (tr if with_trained else None)


def suite(P) -> list:
    """The checks, against module `P`.  Returns the FAILING check names (empty = all green)."""
    bad: list = []

    def t(name, fn):
        try:
            ok = bool(fn())
        except Exception:                                                                      # noqa: BLE001 -- an exception IS a failed check
            ok = False
        if not ok:
            bad.append(name)

    def raises(fn, exc=ValueError):
        try:
            fn()
        except exc:
            return True
        return False

    data = SCENES + ORPHANS + [SCENES[0]]                                                      # + one exact duplicate row
    run = lambda sc=None, **o: P.partition(data if sc is None else sc, **{**dict(KW, n_train=N_TRAIN, n_val=N_VAL, min_val_logs=MIN_VAL_LOGS, seed=SEED), **o})  # noqa: E731
    r = run()
    tr, va = r["train"], r["val"]
    # ---- the bookkeeping, derived by hand above
    t("tally_exact", lambda: r["tally"] == EXPECT_TALLY)
    t("val_exact_count", lambda: len(va) == N_VAL)
    t("train_exact_count", lambda: len(tr) == N_TRAIN)
    t("val_cap_and_log_span", lambda: len({lg for _, lg in va}) >= MIN_VAL_LOGS and max(sum(1 for _, g in va if g == lg) for lg in {x for _, x in va}) <= 5)
    t("val_last_log_trimmed", lambda: sorted(sum(1 for _, g in va if g == lg) for lg in {x for _, x in va})[0] == 2 and r["val_logs"] == 7)
    t("no_shortfall_flags_when_filled", lambda: not r["shortfall_val"] and not r["shortfall_train"])
    # ---- log disjointness and exclusions, computed here from scratch
    tl, vl = {lg for _, lg in tr}, {lg for _, lg in va}
    t("logs_disjoint", lambda: not (tl & vl) and tl and vl)
    t("tokens_disjoint_and_unique", lambda: not ({a for a, _ in tr} & {a for a, _ in va}) and len({a for a, _ in tr}) == len(tr) and len({a for a, _ in va}) == len(va))
    allsc = tr + va
    t("navtest_absent", lambda: not any(a in NAVTEST_TOKENS or g in NAVTEST_LOGS for a, g in allsc))
    t("heldout_absent", lambda: not any(g in HELDOUT for _, g in allsc))
    t("trained_absent", lambda: not any(a in TRAINED for a, _ in allsc))
    t("no_blank_log", lambda: all(g for _, g in allsc))
    t("verify_control_clean", lambda: P.verify(tr, va, min_val_logs=MIN_VAL_LOGS, n_val=N_VAL, n_train=N_TRAIN, **KW) == [])
    # ---- the ordering is the registered recipe (recomputed with hashlib, not read from the module)
    elig_logs = [lg for lg in LOGS if lg not in NAVTEST_LOGS | HELDOUT]
    order = sorted(elig_logs, key=lambda lg: (H(SEED, "log", lg), lg))
    t("val_logs_are_the_first_hash_ordered_logs", lambda: [lg for lg in order if lg in vl] == order[:7] and vl == set(order[:7]))
    first = order[0]
    ok_toks = [a for a, g in SCENES if g == first and a not in NAVTEST_TOKENS and a not in TRAINED]
    exp_first = sorted(ok_toks, key=lambda a: (H(SEED, "scene", a), a))[:5]
    t("val_scenes_are_the_first_hash_ordered_scenes", lambda: [a for a, g in va if g == first] == exp_first)
    t("train_follows_val_in_log_order", lambda: [lg for lg in order[7:] if lg in tl] == [lg for lg in order if lg in tl])
    # ---- determinism: input order must not matter, repeats must agree, the seed must matter
    shuf = list(data)
    random.Random(5).shuffle(shuf)
    r2 = run(shuf)
    t("input_order_free", lambda: r2["train"] == tr and r2["val"] == va)
    t("repeat_identical", lambda: run()["val"] == va and run()["train"] == tr)
    t("seed_changes_the_split", lambda: run(seed=SEED + 1)["val"] != va)
    # ---- take EVERYTHING eligible: now an admitted excluded scene cannot hide behind a random selection
    big = run(n_train=10_000)
    ball = big["train"] + big["val"]
    t("all_eligible_taken_exactly", lambda: len(ball) + big["val_unused_scenes_of_val_logs"] == EXPECT_TALLY["eligible_scenes"] and big["unused_scenes_not_assigned"] == 0)
    t("exclusions_absent_when_all_taken", lambda: not any(a in NAVTEST_TOKENS | TRAINED or g in NAVTEST_LOGS | HELDOUT or not g for a, g in ball))
    t("all_taken_still_log_disjoint", lambda: not ({g for _, g in big["train"]} & {g for _, g in big["val"]}))
    # ---- shortfalls are reported, not hidden
    r3 = run(n_train=1000)
    t("short_train_flagged", lambda: r3["shortfall_train"] and not r3["shortfall_val"] and len(r3["train"]) > N_TRAIN)
    r4 = run(n_val=900, min_val_logs=3)
    t("short_val_flagged", lambda: r4["shortfall_val"] and len(r4["train"]) == 0)
    # ---- a token under two logs is refused
    t("conflicting_logs_refused", lambda: raises(lambda: run(SCENES + [("tok_00_05", "L39")])))
    # ---- verify() sees every violation class (each built by hand from a clean output)
    v = lambda trn, val, **o: P.verify(trn, val, min_val_logs=MIN_VAL_LOGS, n_val=N_VAL, n_train=N_TRAIN, **{**KW, **o})   # noqa: E731
    t("verify_log_on_both_sides", lambda: "log_on_both_sides" in v(tr, va + [("zz_probe_token", tr[0][1])]))
    t("verify_heldout", lambda: "heldout_log_admitted" in v(tr + [("tok_02_00", "L02")], va))
    t("verify_navtest_token", lambda: "navtest_token_admitted" in v(tr + [("tok_00_00", "L00")], va))
    t("verify_navtest_log", lambda: "navtest_log_admitted" in v(tr + [("tok_01_00", "L01")], va))
    t("verify_trained", lambda: "trained_token_admitted" in v(tr + [("tok_04_00", "L04")], va))
    t("verify_blank_log", lambda: "scene_without_log" in v(tr + [("orphan_0", "")], va))
    t("verify_duplicate", lambda: "duplicate_token" in v(tr + [tr[0]], va))
    t("verify_val_too_few_logs", lambda: "val_spans_too_few_logs" in P.verify(tr, va[:3], min_val_logs=6, **KW))
    t("verify_over_n", lambda: "val_over_n" in v(tr, va + [("zz_extra", "ZZ_LOG")]))
    # ---- the CLI, end to end
    def cli():
        with tempfile.TemporaryDirectory() as d:
            pool, nt, ho, trn = write_inputs(d)
            base = ["--pool", pool, "--navtest", nt, "--heldout-logs", ho, "--n-train", str(N_TRAIN), "--n-val", str(N_VAL),
                    "--min-val-logs", str(MIN_VAL_LOGS), "--seed", str(SEED)]
            o1, o2, o3, o4, o5 = (os.path.join(d, x) for x in ("o1", "o2", "o3", "o4", "o5"))
            rc1 = P.main(base + ["--trained-keys", trn, "--out", o1])
            rc2 = P.main(base + ["--trained-keys", trn, "--out", o2])
            files = ("rl_train_scenes.tsv", "rl_val_scenes.tsv", "rl_partition_manifest.json")
            res = {"rc_ok": rc1 == 0 and rc2 == 0, "bytes_identical": all(sha(os.path.join(o1, f)) == sha(os.path.join(o2, f)) for f in files)}
            man = json.load(open(os.path.join(o1, "rl_partition_manifest.json")))
            res["manifest_hashes_are_the_files'"] = (man["train"]["sha256"] == sha(os.path.join(o1, files[0])) and man["val"]["sha256"] == sha(os.path.join(o1, files[1])))
            res["manifest_input_hashes"] = (man["inputs"]["pool"]["sha256"] == sha(pool) and man["inputs"]["trained_keys"]["sha256"] == sha(trn)
                                            and man["inputs"]["heldout_logs"]["sha256"] == sha(ho))
            res["manifest_flags_trained"] = man["trained_exclusion_applied"] is True and man["verify"] == "clean"
            res["manifest_counts"] = man["train"]["scenes"] == N_TRAIN and man["val"]["scenes"] == N_VAL and man["tally"] == EXPECT_TALLY
            rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(o1, files[1]), encoding="utf-8")][1:]
            res["file_matches_partition"] = [(a, b) for a, b in rows] == va
            rc3 = P.main(base + ["--out", o3])                                                     # no trained list: said so, not silently ok
            man3 = json.load(open(os.path.join(o3, "rl_partition_manifest.json")))
            res["no_list_is_flagged"] = rc3 == 0 and man3["trained_exclusion_applied"] is False and man3["inputs"]["trained_keys"] is None
            short = [x if x != str(N_VAL) else "900" for x in base]
            rc4 = P.main(short + ["--trained-keys", trn, "--out", o4])
            res["short_pool_refused_and_nothing_written"] = rc4 == 3 and not os.path.exists(os.path.join(o4, "rl_val_scenes.tsv"))
            rc5 = P.main(short + ["--trained-keys", trn, "--out", o5, "--allow-short"])
            res["allow_short_writes_and_flags"] = rc5 == 0 and json.load(open(os.path.join(o5, "rl_partition_manifest.json")))["shortfall"]["val"] is True
            return res
    try:
        with contextlib.redirect_stdout(io.StringIO()):                                        # the CLI's ZZPARTITION_* lines are not this suite's output
            res = cli()
    except Exception as e:                                                                     # noqa: BLE001
        res = {f"raised_{type(e).__name__}": False}
    for k, ok in res.items():
        t("cli_" + k, lambda ok=ok: ok)
    return bad


# (name, class, old snippet, new snippet) -- each snippet must occur EXACTLY once in rl_partition.py
MUTATIONS = [
    ("train_reuses_val_logs", "log_split", "for lg in logs[i:]:", "for lg in logs:"),
    ("heldout_log_admitted", "heldout", "if log in heldout_logs:", "if False:"),
    ("navtest_token_admitted", "navtest", "if token in navtest_tokens:", "if False:"),
    ("navtest_log_admitted", "navtest", "if log in navtest_logs:", "if False:"),
    ("trained_token_admitted", "trained", "if token in trained_tokens:", "if False:"),
    ("unknown_log_admitted", "unknown", "if not log:", "if False:"),
    ("seed_is_urandom", "seed", 'f"{seed}|{kind}|{key}"', 'f"{os.urandom(4).hex()}|{kind}|{key}"'),
    ("seed_ignored", "seed", 'f"{seed}|{kind}|{key}"', 'f"0|{kind}|{key}"'),
    ("log_order_is_input_order", "order", 'logs = sorted(pool, key=lambda lg: (_h(seed, "log", lg), lg))', "logs = list(pool)"),
    ("val_cap_removed", "cap", "cap = max(1, n_val // min_val_logs)", "cap = 10 ** 9"),
    ("val_not_trimmed", "count", "[:cap][: n_val - len(val)]", "[:cap]"),
    ("train_not_trimmed", "count", "in_scene_order(lg)[: n_train - len(train)]", "in_scene_order(lg)"),
    ("shortfall_silent", "shortfall", '"shortfall_val": len(val) < n_val, "shortfall_train": len(train) < n_train}',
     '"shortfall_val": False, "shortfall_train": False}'),
    ("verify_blind", "verify", "    bad = []\n    tl, vl =", "    bad = []\n    return bad\n    tl, vl ="),
    ("conflicting_logs_allowed", "input", "if token in seen and seen[token] != log:", "if False:"),
    ("duplicate_rows_kept", "input", '            tally["duplicate_row"] += 1\n            continue\n', '            tally["duplicate_row"] += 1\n'),
    ("manifest_val_hash_is_train_hash", "manifest", 'os.path.basename(pva), "sha256": sha256_file(pva)}', 'os.path.basename(pva), "sha256": sha256_file(ptr)}'),
    ("manifest_claims_trained_always", "manifest", '"trained_exclusion_applied": bool(a.trained_keys)', '"trained_exclusion_applied": True'),
]


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    n = 0
    real = suite(load(src, "rl_partition_real"))
    assert not real, f"the unmutated module FAILS its own checks: {real}"
    n += 1
    print(f"{'mutation':36s} {'class':10s} caught-by (failing checks)")
    for i, (name, cls, old, new) in enumerate(MUTATIONS):
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        try:
            mod = load(src.replace(old, new), f"rl_partition_mut{i}")
        except SyntaxError as e:
            raise AssertionError(f"mutation {name} does not compile: {e}")
        try:
            failing = suite(mod)
        except Exception as e:                                                                 # noqa: BLE001
            failing = [f"suite raised {type(e).__name__}: {str(e)[:60]}"]
        assert failing, f"mutation {name} ({cls}) stayed GREEN: the suite cannot see it"
        print(f"{name:36s} {cls:10s} {', '.join(failing[:4])}")
        n += 1
    need = {"log_split", "heldout", "navtest", "trained", "seed", "order", "cap", "count", "shortfall", "verify", "manifest"}
    assert {c for _, c, _, _ in MUTATIONS} >= need, "a required mutation class is missing"
    n += 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations + coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
