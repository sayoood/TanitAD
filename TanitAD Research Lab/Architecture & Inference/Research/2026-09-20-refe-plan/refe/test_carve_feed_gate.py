#!/usr/bin/env python3
"""Gate half of B8 (raw/2026-10-18-r0e-carvefeed): the G-DVB entries, the `carve_feed_ignored` regression arm, the G7 carve re-judge and the launch-gate tables for the carve feed of refe/scorer_finetune.py.
Run:  PYTHONIOENCODING=utf-8 python refe/test_carve_feed_gate.py   ->   ZZTEST_PASS <n>.  CPU only.

THIS FILE LANDS WITH gdvb_patch.diff (refe/gate_checks.py + refe/launch_gate.py + refe/test_launch_gate.py), AFTER the P6 gate fix: on a gate without the patch every check below is RED by design
(a trainer flag without a G-DVB entry is refused by the gate itself).  test_carve_feed.py is the trainer half and has no gate dependency.

  G  the registry classifies epo_con_feed / epo_carve_sha256 / epo_carve_micro `built` and epo_carve_logs `data`; G-DVB PASSES the default argv (no carve block) and a carve argv with the matching BUILT block;
     it FAILS (each for its named reason) a declared carve with no block (the trainer ignored the switch), a declared train with a block, another sha256 / micro than the argv's, a training row in a carve log,
     a carve row in an RL-val log, an empty carve, and a profile whose data_flags would not fingerprint the list.  The regression arm `carve_feed_ignored` FAILS on a train argv AND a carve argv.
  J  judge_g7_carve re-judges the REAL G7 of a real carve preflight from its raw fields (PASS) and FAILS each negative twin; a train-fed argv carrying carve fields FAILS.
  T  the launch-gate tables: RLS1_DATA lists --epo-carve-logs; the aux / sft4r profiles forbid all four flags and AUXS1_EPO_FLAGS equals the parser's --epo-* flags; consume_scorer_preflight calls judge_g7_carve.
Every mutation edits SOURCE (each snippet must occur EXACTLY once) and must turn the named check RED.
"""
from __future__ import annotations

import copy
import hashlib
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_carve_feed as TCF  # noqa: E402
import test_rl_s1b as S1B  # noqa: E402
import test_rl_safeobj as SO  # noqa: E402

GC_PATH, LG_PATH = (os.path.join(HERE, f) for f in ("gate_checks.py", "launch_gate.py"))
SHA = hashlib.sha256(TCF.CARVE_LOGS.encode()).hexdigest()
check = TCF.check


def suite(gc_src: str, lg_src: str, tag: str, real_g7: dict) -> list:
    bad: list = []
    GC = S1B.load(gc_src, f"gc_cf_{tag}", GC_PATH)
    real = sys.modules.get("gate_checks")
    sys.modules["gate_checks"] = GC          # launch_gate imports gate_checks lazily: the module under test must BE sys.modules["gate_checks"]
    try:
        L = S1B.load(lg_src, f"lg_cf_{tag}", LG_PATH)
        import scorer_finetune as S
        Pd = {**L.PROFILES["refe_rls1d"], "declared_arm": "R0d"}
        r0d = L.parse_args_string(SO.ARM_ARGS["R0d"])
        CV = ["--epo-con-feed", "carve", "--epo-carve-logs", "c.txt", "--epo-carve-sha256", SHA]
        ns_d, ns_c = S.build_parser().parse_args(r0d), S.build_parser().parse_args(r0d + CV)
        BLK = {"feed": "carve", "logs": 13, "sets": 700, "sha256": SHA, "micro": 1, "train_rows_in_carve_logs": 0, "carve_rows_in_heldout_logs": 0}

        def data(ns, blk=None):
            e = {k: getattr(ns, "epo_" + k) for k in GC.SCORER_EPO_KEYS if k != "ref"}
            e["ref"] = "base"
            d = {"event": "data", "b_mode": ns.b_mode, "b_compw": ns.b_compw, "epo": e, "epo_con": S.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.0, temp=0.1).built_facts()}
            if blk is not None:
                d["epo_con_carve"] = blk
            return d
        ev = lambda argv, d, prof=Pd, arm=None: GC.check_dvb_scorer(argv, {"data": d}, prof, arm=arm)          # noqa: E731
        why = lambda e: " ".join(e["reasons"])          # noqa: E731

        # ---- G
        check(bad, "G_1_registry_classifies_the_four_dests", lambda: [GC.DVB_KINDS_SCORER.get(k) for k in ("epo_con_feed", "epo_carve_sha256", "epo_carve_micro", "epo_carve_logs")] == ["built", "built", "built", "data"])
        check(bad, "G_2_PASS_the_default_argv_with_no_carve_block_27_rows", lambda: (lambda e: e["status"] == "PASS" and e["details"]["observed_rows"] == 27)(ev(r0d, data(ns_d))))
        check(bad, "G_3_PASS_a_carve_argv_with_the_matching_BUILT_block_28_rows", lambda: (lambda e: e["status"] == "PASS" and e["details"]["observed_rows"] == 28)(ev(r0d + CV, data(ns_c, BLK))))
        check(bad, "G_4_FAIL_a_declared_carve_with_NO_block_the_trainer_ignored_the_switch", lambda: (lambda e: e["status"] == "FAIL" and "--epo-con-feed: declared 'carve' != built 'train'" in why(e))(ev(r0d + CV, data(ns_c))))
        check(bad, "G_5_FAIL_a_declared_train_with_a_carve_block", lambda: (lambda e: e["status"] == "FAIL" and "--epo-con-feed: declared 'train' != built 'carve'" in why(e))(ev(r0d, data(ns_d, BLK))))
        check(bad, "G_6_FAIL_sha_micro_leak_and_empty_carve_each_for_its_named_reason", lambda: all(ev(r0d + CV, data(ns_c, {**BLK, **mut}))["status"] == "FAIL" and frag in why(ev(r0d + CV, data(ns_c, {**BLK, **mut})))
                                                                                               for mut, frag in (({"sha256": "0" * 64}, "--epo-carve-sha256"), ({"micro": 3}, "--epo-carve-micro"), ({"train_rows_in_carve_logs": 3}, "--epo-carve-logs"),
                                                                                                                 ({"carve_rows_in_heldout_logs": 2}, "--epo-carve-logs"), ({"sets": 0}, "--epo-carve-logs"))))
        check(bad, "G_7_REGRESSION_ARM_carve_feed_ignored_FAILS_on_a_train_argv_AND_a_carve_argv_and_the_same_evidence_PASSES_without_the_arm",
              lambda: ev(r0d, data(ns_d), arm="carve_feed_ignored")["status"] == "FAIL" and ev(r0d + CV, data(ns_c, BLK), arm="carve_feed_ignored")["status"] == "FAIL"
              and ev(r0d, data(ns_d))["status"] == "PASS" and ev(r0d + CV, data(ns_c, BLK))["status"] == "PASS")
        Pnb = {**Pd, "data_flags": tuple(x for x in Pd["data_flags"] if x != "--epo-carve-logs")}
        check(bad, "G_8_FAIL_a_profile_whose_data_flags_would_not_fingerprint_the_carve_list", lambda: (lambda e: e["status"] == "FAIL" and "--epo-carve-logs" in why(e) and "would NOT fingerprint" in why(e))(ev(r0d, data(ns_d), Pnb)))

        # ---- J
        j = lambda g7, ns=ns_c: GC.judge_g7_carve(g7, ns)          # noqa: E731
        check(bad, "J_1_the_REAL_G7_of_a_real_carve_preflight_is_re_judged_clean_at_its_own_argv", lambda: j(real_g7, S.build_parser().parse_args(r0d + ["--epo-con-feed", "carve", "--epo-carve-logs", "c", "--epo-carve-sha256", SHA, "--batch", "2"])) == [])
        ns_r = S.build_parser().parse_args(r0d + ["--epo-con-feed", "carve", "--epo-carve-logs", "c", "--epo-carve-sha256", SHA, "--batch", "2"])
        twins = {"sha": {"carve_sha256": "0" * 64}, "train_leak": {"train_rows_in_carve_logs": 1}, "val_leak": {"carve_rows_in_heldout_logs": 1}, "tiny": {"carve_sets": 1}, "micro": {"carve_probe_n": 4},
                 "nan": {"carve_probe_g": {"ttc": float("nan"), "nc": 0.0}}, "missing": {"carve_probe_g": {}}}
        check(bad, "J_2_each_negative_twin_of_the_real_G7_FAILS", lambda: all(j({**copy.deepcopy(real_g7), **mut}, ns_r) != [] for mut in twins.values()))
        check(bad, "J_3_a_train_fed_argv_whose_G7_carries_carve_fields_FAILS_and_a_clean_G7_passes", lambda: j(real_g7, ns_d) != [] and j({k: v for k, v in real_g7.items() if not (k.startswith("carve_") or k == "train_rows_in_carve_logs")}, ns_d) == [])

        # ---- T
        check(bad, "T_1_RLS1_DATA_lists_the_carve_list_so_the_token_fingerprints_it", lambda: "--epo-carve-logs" in L.RLS1_DATA)
        check(bad, "T_2_aux_and_sft4r_profiles_forbid_all_four_flags_and_AUXS1_EPO_FLAGS_equals_the_parsers_epo_flags", lambda: all(f in L.AUXS1_EPO_FLAGS and f in L.AUXS1_FORBIDDEN and f in L.SFT4R_FORBIDDEN for f in ("--epo-con-feed", "--epo-carve-logs", "--epo-carve-sha256", "--epo-carve-micro"))
              and set(o for x in S.build_parser()._actions for o in x.option_strings if o == "--epo" or o.startswith("--epo-")) == set(L.AUXS1_EPO_FLAGS))
        check(bad, "T_3_consume_scorer_preflight_re_judges_G7_through_judge_g7_carve", lambda: "reasons += C.judge_g7_carve(g7, ns)" in lg_src)
    finally:
        if real is not None:
            sys.modules["gate_checks"] = real
        else:
            sys.modules.pop("gate_checks", None)
    return bad


MUTATIONS = [
    ("GM1 the registry entry for epo_con_feed dropped", "gc", [('    "epo_con_feed": "built", "epo_carve_logs": "data", "epo_carve_sha256": "built", "epo_carve_micro": "built",\n', '    "epo_carve_logs": "data", "epo_carve_sha256": "built", "epo_carve_micro": "built",\n')],
     ["G_1_registry_classifies_the_four_dests"]),
    ("GM2 the feed row echoes the namespace instead of the BUILT block", "gc", [('"carve" if cb.get("feed") == "carve" else "train", "data event: epo_con_carve.feed', 'feed, "data event: epo_con_carve.feed')],
     ["G_4_FAIL_a_declared_carve_with_NO_block_the_trainer_ignored_the_switch", "G_5_FAIL_a_declared_train_with_a_carve_block"]),
    ("GM3 the regression arm is inert", "gc", [('if arm_is("carve_feed_ignored", arm) and', 'if arm_is("carve_feed_ignored_off", arm) and')], ["G_7_REGRESSION_ARM_carve_feed_ignored_FAILS_on_a_train_argv_AND_a_carve_argv_and_the_same_evidence_PASSES_without_the_arm"]),
    ("GM4 G-DVB does not check the leak counters", "gc", [('add("epo_carve_logs", [0, 0, True]', 'add("epo_carve_logs", [cb.get("train_rows_in_carve_logs"), cb.get("carve_rows_in_heldout_logs"), True]')],
     ["G_6_FAIL_sha_micro_leak_and_empty_carve_each_for_its_named_reason"]),
    ("GM5 judge_g7_carve does not require zero training rows in the carve logs", "gc", [('    if g7.get("train_rows_in_carve_logs") != 0:\n', "    if False:\n")], ["J_2_each_negative_twin_of_the_real_G7_FAILS"]),
    ("GM6 judge_g7_carve ignores carve fields on a train-fed argv", "gc", [('        return [f"G7: carve fields {carve_keys} on an argv whose --epo-con-feed is train"] if carve_keys else []\n', "        return []\n")],
     ["J_3_a_train_fed_argv_whose_G7_carries_carve_fields_FAILS_and_a_clean_G7_passes"]),
    ("GM7 the carve list is not in RLS1_DATA", "lg", [('"--aux-drv-labels", "--epo-carve-logs")', '"--aux-drv-labels")')], ["T_1_RLS1_DATA_lists_the_carve_list_so_the_token_fingerprints_it", "G_2_PASS_the_default_argv_with_no_carve_block_27_rows"]),
    ("GM8 the aux profiles no longer forbid the carve flags", "lg", [('"--epo-con-feed", "--epo-carve-logs", "--epo-carve-sha256", "--epo-carve-micro")', '"--epo-con-feed")')], ["T_2_aux_and_sft4r_profiles_forbid_all_four_flags_and_AUXS1_EPO_FLAGS_equals_the_parsers_epo_flags"]),
    ("GM9 the launch gate does not call judge_g7_carve", "lg", [("        reasons += C.judge_g7_carve(g7, ns)\n", "        pass\n")], ["T_3_consume_scorer_preflight_re_judges_G7_through_judge_g7_carve"]),
]


def main() -> int:
    gc_src, lg_src = (open(p, encoding="utf-8").read() for p in (GC_PATH, LG_PATH))
    sf_src = open(TCF.SF_PATH, encoding="utf-8").read()
    r = TCF.run_smoke(sf_src, "cfg_real", TCF.EXTRA)
    assert TCF.ok_run(r), f"the real carve preflight did not pass: {r.get('exc')} {r.get('out_pre', '')[-300:]}"
    real_g7 = r["pass_json"]["gates"]["G7_heldout_disjoint"]
    bad = suite(gc_src, lg_src, "real", real_g7)
    assert not bad, f"the unmutated gate FAILS its own checks: {bad}"
    n, problems = 1, []
    print(f"{'mutation':100s} caught-by (failing checks)")
    for i, (name, which, edits, expect) in enumerate(MUTATIONS):
        g, l = gc_src, lg_src
        for old, new in edits:
            src = g if which == "gc" else l
            assert src.count(old) == 1, f"mutation {name}: the snippet occurs {src.count(old)} times (must be exactly 1): {old[:70]!r}"
            if which == "gc":
                g = g.replace(old, new)
            else:
                l = l.replace(old, new)
        try:
            failing = suite(g, l, f"m{i}", real_g7)
        except (Exception, SystemExit) as e:                            # noqa: BLE001
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:80]}"]
        miss = [c for c in expect if c not in failing]
        if not failing or (miss and not any(f.startswith("suite setup raised") for f in failing)):
            problems.append(f"mutation {name}: must turn {expect} RED; RED were {failing[:6]}")
            continue
        print(f"{name[:100]:100s} {', '.join(failing[:3])}", flush=True)
        n += 1
    if problems:
        print("\n".join("PROBLEM " + p_ for p_ in problems))
        print(f"ZZTEST_FAIL {len(problems)} problem(s)")
        return 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
