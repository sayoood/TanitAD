#!/usr/bin/env python3
"""patch_genuine_tests: the exact-anchor edit lists of patch_genuine.py for the two EXISTING tests whose fixtures were hand-written S files (imported by patch_genuine.py).

Both tests fed `rls1_checks.safeobj_gate` a hand-written `{"S": "PASS"}`.  The gate now refuses exactly that, so the fixtures become what the gate exists to accept (a record the REAL reader wrote over
an arm directory the REAL producers made: eval/test_safeobj_gate.py `genuine()` / `copy_arm()`), and the hand-written / edited / stale ones become cases the gate must refuse FOR A NAMED REASON.
All strings are raw: they contain backslashes (shell line continuations) that must reach the anchor unchanged."""

# ================================================================================================================================================ refe/test_rl_safeobj.py (the CH block)
OLD_CH_GATE = r'''    # ---- the S gate
    with tempfile.TemporaryDirectory() as td:
        def sdir(name, content):
            d = os.path.join(td, name)
            os.makedirs(d, exist_ok=True)
            if content is not None:
                with open(os.path.join(d, "safeobj_S.json"), "w", encoding="utf-8") as f:
                    f.write(content)
            return d
        d_pass, d_pass2, d_fail, d_none, d_junk, d_weak = (sdir("p", '{"S": "PASS"}'), sdir("p2", '{"S": "PASS", "checks": {}}'), sdir("f", '{"S": "FAIL"}'), sdir("n", None), sdir("j", "{not json"),
                                                          sdir("w", '{"S": "PASS "}'))
        check(bad, "CH_safeobj_gate_passes_only_when_every_constrained_arm_reads_S_PASS", lambda: RC.safeobj_gate([d_pass, d_pass2])[0] is True)
        check(bad, "CH_safeobj_gate_refuses_a_FAIL_a_missing_file_junk_and_an_almost_PASS", lambda: all(RC.safeobj_gate([d_pass, x])[0] is False for x in (d_fail, d_none, d_junk, d_weak))
              and all(RC.safeobj_gate([x, d_pass])[0] is False for x in (d_fail, d_none, d_junk, d_weak)))
'''
NEW_CH_GATE = r'''    # ---- the S gate.  SAFEOBJ round 3: it opens only on a GENUINE, unmodified reader record, so the PASS / FAIL / NOT_READ fixtures below are arm directories made by the REAL producers with the record
    # written by the REAL reader (eval/test_safeobj_gate.py: genuine / copy_arm); a hand-written `{"S": "PASS"}` -- what the gate used to open on -- is now one of the refused cases.
    # (The full battery of mutations of the genuineness check lives in eval/test_safeobj_gate.py.)
    sys.path.insert(0, os.path.join(HERE, "..", "eval"))
    import test_safeobj_gate as TG
    with tempfile.TemporaryDirectory() as td:
        def own(label, kind, arm):                                          # a private copy of the genuine arm directory (its basename IS the arm)
            return str(TG.copy_arm(kind, arm, os.path.join(td, label)))

        def rewrite(d, fn):                                                 # edit the record in d (the reader's file, json) with fn(record) and write it back unstamped
            p = os.path.join(d, "safeobj_S.json")
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            out = fn(rec)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1, sort_keys=True)
            return d
        restamp = lambda rec: {**rec, "record_sha256": TG.oracle_digest(rec)}                      # noqa: E731  (what a forger who knows the canonicalisation writes)
        d_pass, d_pass2 = own("p", "PASS", "R0d"), own("p2", "PASS", "R1d")
        d_fail, d_nr = own("f", "FAIL", "R1d"), own("nr", "NOT_READ", "R1d")
        d_none = os.path.join(td, "n")
        os.makedirs(d_none)
        d_junk = os.path.join(td, "j")
        os.makedirs(d_junk)
        with open(os.path.join(d_junk, "safeobj_S.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        d_weak = rewrite(own("w", "PASS", "R1d"), lambda r: restamp({**r, "S": "PASS "}))        # an almost-PASS whose stamp is right: the exactness of S is the only barrier
        d_hand = rewrite(own("h", "FAIL", "R1d"), lambda r: {"S": "PASS"})                         # hand-written
        d_edit = rewrite(own("e", "FAIL", "R1d"), lambda r: {**r, "S": "PASS"})                    # a genuine FAIL with S edited to PASS, nothing else touched
        d_stale = own("s", "PASS", "R1d")                                                          # a genuine PASS whose input changed after the read
        with open(os.path.join(d_stale, "kill_replay.json"), "ab") as f:
            f.write(b" ")
        check(bad, "CH_safeobj_gate_passes_only_when_every_constrained_arm_reads_S_PASS", lambda: RC.safeobj_gate([d_pass, d_pass2])[0] is True)
        check(bad, "CH_safeobj_gate_refuses_a_FAIL_a_NOT_READ_a_missing_file_junk_and_an_almost_PASS", lambda: all(RC.safeobj_gate([d_pass, x])[0] is False for x in (d_fail, d_nr, d_none, d_junk, d_weak))
              and all(RC.safeobj_gate([x, d_pass])[0] is False for x in (d_fail, d_nr, d_none, d_junk, d_weak)))
        check(bad, "CH_safeobj_gate_refuses_a_hand_written_PASS_an_edited_record_and_a_record_whose_input_changed_each_for_its_reason", lambda: all(
            RC.safeobj_gate([d_pass, x])[0] is False and RC.safeobj_gate([x, d_pass])[0] is False and why in RC.safeobj_gate([d_pass, x])[1] for x, why in (
                (d_hand, "tool is"), (d_edit, "record_sha256 does not recompute"), (d_stale, "input 'kill_replay.json' changed since the read"))))
'''
OLD_CH_MUT = r'''    ("CH14 the pod precheck does not ship test_rl_safeobj.py", "CH", [(POD_PATH, "refe/test_rl_r0c.py refe/test_rl_safeobj.py \\\n", "refe/test_rl_r0c.py \\\n")], None),
]
'''
NEW_CH_MUT = r'''    ("CH14 the pod precheck does not ship test_rl_safeobj.py", "CH", [(POD_PATH, "refe/test_rl_r0c.py refe/test_rl_safeobj.py \\\n", "refe/test_rl_r0c.py \\\n")], None),
    ("CH15 safeobj_gate does not verify the record's genuineness", "CH", [(RC_PATH, "            why = safeobj_not_genuine(d, rec)\n", '            why = ""\n')], None),
]
'''
OLD_CH_EXPECT10 = r'''    "CH10 safeobj_gate accepts anything but FAIL": ["CH_safeobj_gate_refuses_a_FAIL_a_missing_file_junk_and_an_almost_PASS"],
'''
NEW_CH_EXPECT10 = r'''    "CH10 safeobj_gate accepts anything but FAIL": ["CH_safeobj_gate_refuses_a_FAIL_a_NOT_READ_a_missing_file_junk_and_an_almost_PASS"],
'''
OLD_CH_EXPECT14 = r'''    "CH14 the pod precheck does not ship test_rl_safeobj.py": ["CH_the_precheck_ships_test_rl_safeobj_the_refe_rls1d_suite_runs_it_on_the_pod"],
}
'''
NEW_CH_EXPECT14 = r'''    "CH14 the pod precheck does not ship test_rl_safeobj.py": ["CH_the_precheck_ships_test_rl_safeobj_the_refe_rls1d_suite_runs_it_on_the_pod"],
    "CH15 safeobj_gate does not verify the record's genuineness": ["CH_safeobj_gate_refuses_a_hand_written_PASS_an_edited_record_and_a_record_whose_input_changed_each_for_its_reason"],
}
'''
CH_TEST_EDITS = [(OLD_CH_GATE, NEW_CH_GATE), (OLD_CH_MUT, NEW_CH_MUT), (OLD_CH_EXPECT10, NEW_CH_EXPECT10), (OLD_CH_EXPECT14, NEW_CH_EXPECT14)]

# ================================================================================================================================================ refe/test_rls1_queue.py (rls1d_checks)
OLD_Q_FIX = r'''    PASS = '{"S": "PASS"}'

    def s_files(*pairs):
        def prep(R):
            for arm, content in pairs:
                (R / "arms" / arm).mkdir(parents=True, exist_ok=True)
                (R / "arms" / arm / "safeobj_S.json").write_text(content, encoding="utf-8")
        return prep
'''
NEW_Q_FIX = r'''    sys.path.insert(0, str(PKGROOT / "eval"))
    import test_safeobj_gate as TG          # SAFEOBJ round 3: the real S gate opens only on a GENUINE reader record, so the fixtures are the REAL producers' arm directories + the REAL reader's record

    def s_files(*pairs):
        """(arm, "PASS" | "FAIL" | "NOT_READ") -> a copy of the genuine arm directory (the real producers' files, the real reader's safeobj_S.json); (arm, other text) -> that text as the record"""
        def prep(R):
            for arm, what in pairs:
                if what in ("PASS", "FAIL", "NOT_READ"):
                    shutil.copytree(TG.genuine(what, arm), R / "arms" / arm)
                else:
                    (R / "arms" / arm).mkdir(parents=True, exist_ok=True)
                    (R / "arms" / arm / "safeobj_S.json").write_text(what, encoding="utf-8")
        return prep
'''
OLD_Q_PREP = r'''    def prep_all(R):
        with_c0(R)
        s_files(("R0d", PASS), ("R1d", PASS))(R)
'''
NEW_Q_PREP = r'''    def operator_read(R):                                                                                   # the operator's step between R1d and R1o (BUILD.md section 4 step 6): the reader has written R0d's and R1d's records
        with_c0(R)
        s_files(("R0d", "PASS"), ("R1d", "PASS"))(R)
'''
OLD_Q_ORDER = r'''    if want("order"):                                                                                       # R0d -> R1d -> R1o, each under refe_rls1d, each read against its OWN arm A
        r = armloop_run(text, tmp, "al_so_order", "R0d R1d R1o", prepare=prep_all, timeout=240)
        assert r["rc"] == 0 and r["done"] == ["R0d", "R1d", "R1o"] and "ZZRLS1_DONE arms: R0d R1d R1o" in r["log"], (r["rc"], r["done"], r["log"][-1500:], r["out"][-500:])
        assert re.findall(r"launch_gate\.py run --profile (\S+) --arm-name (\S+)", r["calls"]) == [("refe_rls1d", "R0d"), ("refe_rls1d", "R1d"), ("refe_rls1d", "R1o")], r["calls"]
        ctl = {a: re.search(rf"arm {a}: trainer pid/pgid .*\(control: ([^)]*)\)", r["log"]).group(1) for a in ("R0d", "R1d", "R1o")}
        assert all(ctl[a].endswith(f"/arms/{a}/scores") for a in ctl), ctl                                  # NOT .../arms/C0/scores, even for R0d with C0's dumps there
        trn = [ln for ln in calls_of(r, "scorer_finetune.py") if "--preflight" not in ln]
        pre = [ln for ln in calls_of(r, "scorer_finetune.py") if "--preflight" in ln]
        assert len(trn) == 3 and len(pre) == 3, (trn, pre)
        assert trn[0].endswith(CON_ON) and trn[1].endswith(CON_ON) and trn[2].endswith("--epo-con off"), [t[-90:] for t in trn]             # the levers reach the TRAINER ...
        assert CON_ON in pre[0] and CON_ON in pre[1] and "--epo-con off" in pre[2] and "--seed 1" in pre[2], [p_[-110:] for p_ in pre]      # ... and the PREFLIGHT
        mon = [ln for ln in calls_of(r, "rls1_checks.py") if " monitor " in ln]
        rep = [ln for ln in calls_of(r, "rls1_checks.py") if " replay " in ln]
        assert len(mon) == 3 and len(rep) == 3, (mon, rep)
        assert all(re.search(rf"--arm {a} .*--refdir \S+/arms/{a}/scores ", mon[i]) for i, a in enumerate(("R0d", "R1d", "R1o"))), mon
        assert "--observe-only" not in mon[0] and "--observe-only" not in mon[1] and "--observe-only" in mon[2], mon                       # M9: R1o's K3 monitor is OBSERVE-ONLY, the constrained arms' are not
        assert "--observe-only" not in rep[0] and "--observe-only" not in rep[1] and "--observe-only" in rep[2], rep
        gate = [ln for ln in calls_of(r, "rls1_checks.py") if " safeobj-gate " in ln]
        assert len(gate) == 1 and re.search(r"--dirs \S+/arms/R0d \S+/arms/R1d$", gate[0]), gate                                           # only R1o is gated, on BOTH constrained arms
        rv = calls_of(r, "rl_val_eval.py")
        assert len(rv) == 4 and "rlval_vs_C0.json" in rv[1] and "/arms/C0/scores/scores_update1000_A.npz" in rv[1] and all("rlval_vs_C" not in x for x in (rv[0], rv[2], rv[3])), rv      # R0d: C0 is the second control
        assert r["log"].count("ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL") == 2 and "ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL arm R1d: C1's" in r["log"] and "ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL arm R1o: C1's" in r["log"]
        assert all(f"arm {a}: kill monitor READY" in r["log"] for a in ("R0d", "R1d", "R1o")); n += 30
'''
NEW_Q_ORDER = r'''    if want("order"):                                                                                       # R0d -> R1d -> R1o, each under refe_rls1d, each read against its OWN arm A
        # TWO chain runs, because the real S gate leaves no other way (round 3).  Run 1 is the one-list form "R0d R1d R1o" with NO reader step: the chain trains R0d and R1d and R1o's gate REFUSES -- nothing
        # the chain writes is a reader record.  Run 2 is "R1o" alone after the operator's step (genuine records for R0d and R1d): the gate opens and R1o runs.  (Before round 3 one run passed because the
        # test pre-seeded hand-written S files, which the reader never wrote and the gate now refuses; BUILD.md section 4 step 7 no longer offers the one-list form for the three arms.)
        r = armloop_run(text, tmp, "al_so_order", "R0d R1d R1o", prepare=with_c0, timeout=240)
        assert r["rc"] == 1 and r["done"] == ["R0d", "R1d"] and "REFUSED: R1o runs only if R0d and R1d both read bar S = PASS" in r["log"] and "FileNotFoundError" in r["log"] and "ZZRLS1_DONE" not in r["log"], (r["rc"], r["done"], r["log"][-1500:], r["out"][-500:])
        assert not any("arms/R1o" in ln for ln in r["calls"].splitlines()) and "R1o" not in r["arms_dirs"] and "ZZRLS1_ARM_DONE R1o" not in r["log"], (r["calls"], r["arms_dirs"])      # refused before anything of R1o's
        assert re.findall(r"launch_gate\.py run --profile (\S+) --arm-name (\S+)", r["calls"]) == [("refe_rls1d", "R0d"), ("refe_rls1d", "R1d")], r["calls"]
        ctl = {a: re.search(rf"arm {a}: trainer pid/pgid .*\(control: ([^)]*)\)", r["log"]).group(1) for a in ("R0d", "R1d")}
        assert all(ctl[a].endswith(f"/arms/{a}/scores") for a in ctl), ctl                                  # NOT .../arms/C0/scores, even for R0d with C0's dumps there
        trn = [ln for ln in calls_of(r, "scorer_finetune.py") if "--preflight" not in ln]
        pre = [ln for ln in calls_of(r, "scorer_finetune.py") if "--preflight" in ln]
        assert len(trn) == 2 and len(pre) == 2, (trn, pre)
        assert trn[0].endswith(CON_ON) and trn[1].endswith(CON_ON), [t[-90:] for t in trn]                  # the levers reach the TRAINER ...
        assert CON_ON in pre[0] and CON_ON in pre[1], [p_[-110:] for p_ in pre]                             # ... and the PREFLIGHT
        mon = [ln for ln in calls_of(r, "rls1_checks.py") if " monitor " in ln]
        rep = [ln for ln in calls_of(r, "rls1_checks.py") if " replay " in ln]
        assert len(mon) == 2 and len(rep) == 2, (mon, rep)
        assert all(re.search(rf"--arm {a} .*--refdir \S+/arms/{a}/scores ", mon[i]) for i, a in enumerate(("R0d", "R1d"))), mon
        assert "--observe-only" not in mon[0] and "--observe-only" not in mon[1] and "--observe-only" not in rep[0] and "--observe-only" not in rep[1], (mon, rep)       # the constrained arms' K3 is a KILL
        gate = [ln for ln in calls_of(r, "rls1_checks.py") if " safeobj-gate " in ln]
        assert len(gate) == 1 and re.search(r"--dirs \S+/arms/R0d \S+/arms/R1d$", gate[0]), gate           # only R1o is gated, on BOTH constrained arms
        rv = calls_of(r, "rl_val_eval.py")
        assert len(rv) == 3 and "rlval_vs_C0.json" in rv[1] and "/arms/C0/scores/scores_update1000_A.npz" in rv[1] and all("rlval_vs_C" not in x for x in (rv[0], rv[2])), rv      # R0d: C0 is the second control
        assert r["log"].count("ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL") == 1 and "ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL arm R1d: C1's" in r["log"]
        assert all(f"arm {a}: kill monitor READY" in r["log"] for a in ("R0d", "R1d")); n += 22
        r = armloop_run(text, tmp, "al_so_order_r1o", "R1o", prepare=operator_read, timeout=240)            # after the reader: R0d and R1d carry genuine PASS records
        assert r["rc"] == 0 and r["done"] == ["R1o"] and "ZZRLS1_DONE arms: R1o" in r["log"], (r["rc"], r["done"], r["log"][-1500:], r["out"][-500:])
        assert re.findall(r"launch_gate\.py run --profile (\S+) --arm-name (\S+)", r["calls"]) == [("refe_rls1d", "R1o")], r["calls"]
        ctl = re.search(r"arm R1o: trainer pid/pgid .*\(control: ([^)]*)\)", r["log"]).group(1)
        assert ctl.endswith("/arms/R1o/scores"), ctl
        trn = [ln for ln in calls_of(r, "scorer_finetune.py") if "--preflight" not in ln]
        pre = [ln for ln in calls_of(r, "scorer_finetune.py") if "--preflight" in ln]
        assert len(trn) == 1 and len(pre) == 1 and trn[0].endswith("--epo-con off") and "--epo-con off" in pre[0] and "--seed 1" in pre[0], (trn, pre)
        mon = [ln for ln in calls_of(r, "rls1_checks.py") if " monitor " in ln]
        rep = [ln for ln in calls_of(r, "rls1_checks.py") if " replay " in ln]
        assert len(mon) == 1 and len(rep) == 1 and "--observe-only" in mon[0] and "--observe-only" in rep[0], (mon, rep)              # M9: R1o's K3 monitor is OBSERVE-ONLY
        gate = [ln for ln in calls_of(r, "rls1_checks.py") if " safeobj-gate " in ln]
        assert len(gate) == 1 and re.search(r"--dirs \S+/arms/R0d \S+/arms/R1d$", gate[0]) and "ZZRLS1_SAFEOBJ_S_PASS" in (r["R"] / "safeobj_gate_R1o.log").read_text(), gate     # the REAL gate opened on the genuine records
        rv = calls_of(r, "rl_val_eval.py")
        assert len(rv) == 1 and "rlval_vs_C" not in rv[0], rv
        assert r["log"].count("ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL") == 1 and "ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL arm R1o: C1's" in r["log"] and "arm R1o: kill monitor READY" in r["log"]; n += 12
'''
OLD_Q_NEEDS = r'''    if want("r1o_needs_S"):                                                                                 # R1o is REFUSED until BOTH constrained arms read S = PASS: nothing is created, nothing runs
        for nm, prep in (("none", None), ("one", s_files(("R0d", PASS))), ("fail", s_files(("R0d", PASS), ("R1d", '{"S": "FAIL"}'))), ("junk", s_files(("R0d", PASS), ("R1d", "{nope")))):
            r = armloop_run(text, tmp, f"al_so_noS_{nm}", "R1o", prepare=prep)
            assert r["rc"] == 1 and "REFUSED: R1o runs only if R0d and R1d both read bar S = PASS" in r["log"], (nm, r["rc"], r["log"][-800:])
            assert "scorer_finetune.py" not in r["calls"] and "launch_gate.py" not in r["calls"] and "R1o" not in r["arms_dirs"] and "ZZRLS1_ARM_DONE" not in r["log"], (nm, r["calls"], r["arms_dirs"]); n += 3
'''
NEW_Q_NEEDS = r'''    if want("r1o_needs_S"):                                                                                 # R1o is REFUSED until BOTH constrained arms carry a GENUINE reader record reading S = PASS: nothing is created, nothing runs, the refusal NAMES its reason
        def stale(R):                                                                                       # a genuine PASS record whose input changed after the read
            s_files(("R0d", "PASS"), ("R1d", "PASS"))(R)
            with open(R / "arms" / "R1d" / "kill_replay.json", "ab") as f_:
                f_.write(b" ")

        def hand(R):                                                                                        # the record R1d should have got from the reader, written by hand
            s_files(("R0d", "PASS"), ("R1d", "FAIL"))(R)
            (R / "arms" / "R1d" / "safeobj_S.json").write_text('{"S": "PASS"}', encoding="utf-8")

        def edited(R):                                                                                      # a genuine FAIL record with its S edited to PASS
            s_files(("R0d", "PASS"), ("R1d", "FAIL"))(R)
            p_ = R / "arms" / "R1d" / "safeobj_S.json"
            assert '"S": "FAIL"' in p_.read_text(encoding="utf-8")
            p_.write_text(p_.read_text(encoding="utf-8").replace('"S": "FAIL"', '"S": "PASS"'), encoding="utf-8")
        for nm, prep, why in (("none", None, "FileNotFoundError"), ("one", s_files(("R0d", "PASS")), "FileNotFoundError"), ("fail", s_files(("R0d", "PASS"), ("R1d", "FAIL")), "S='FAIL'"),
                              ("junk", s_files(("R0d", "PASS"), ("R1d", "{nope")), "JSONDecodeError"), ("hand_written", hand, "tool is"), ("edited", edited, "record_sha256 does not recompute"),
                              ("stale_input", stale, "changed since the read")):
            r = armloop_run(text, tmp, f"al_so_noS_{nm}", "R1o", prepare=prep)
            assert r["rc"] == 1 and "REFUSED: R1o runs only if R0d and R1d both read bar S = PASS" in r["log"] and why in r["log"], (nm, why, r["rc"], r["log"][-800:])
            assert "scorer_finetune.py" not in r["calls"] and "launch_gate.py" not in r["calls"] and "R1o" not in r["arms_dirs"] and "ZZRLS1_ARM_DONE" not in r["log"], (nm, r["calls"], r["arms_dirs"]); n += 4
'''
OLD_Q_STOP = (r"""        \"\"\"no `arm X:` log line, no gate / trainer / S-gate call that names X, no ZZRLS1_DONE, and arms/X absent or holding only the fixture's own safeobj_S.json (prep_all pre-seeds R1d's, so
        `arms_dirs` can never equal [C0, R0d]: the first full-form run of this test failed there for that reason, not because the chain went on)\"\"\"
        for a in later:
            d = r["R"] / "arms" / a
            held = sorted(p_.name for p_ in d.iterdir()) if d.exists() else None
            assert not re.search(rf"arm {a}:", r["log"]) and not re.search(rf"--arm-name {a}""" + "\x08" + r"""|--arm {a}""" + "\x08" + r"""|arms/{a}/(sft|preflight|monitor|launch_gate)", r["calls"] + r["log"]), ("KILL-STOP: the chain went on to " + a, r["log"][-1500:])
            assert held in (None, ["safeobj_S.json"]), ("KILL-STOP: the chain wrote into arms/" + a, held)
""").replace('\\"', '"')
# NOTE the old text above carries a literal BACKSPACE (0x08) where round 2's build wrote `\b` in a NON-raw string: the regex half of the KILL-STOP check never matched anything.
NEW_Q_STOP = r"""        \"\"\"no `arm X:` log line, no gate / trainer call that names X, no S-gate call when R1o is among them (its gate is asked only when R1o is next), no ZZRLS1_DONE, and arms/X ABSENT.  Round 3: the
        scenarios pre-seed no S files (the real gate would refuse a hand-written one, and a genuine one would be wiped by the chain's own arm start), so anything under arms/X is the chain going on.
        (Round 2 tolerated a pre-seeded safeobj_S.json there, because prep_all wrote one: its first full-form run failed on `arms_dirs == [C0, R0d]` for that reason, not because the chain went on.
        Round 2 also lost the word boundaries of the regex below: a word boundary written as backslash-b in a non-raw string of the build script became a BACKSPACE character, so that half of the check matched nothing.)\"\"\"
        for a in later:
            d = r["R"] / "arms" / a
            held = sorted(p_.name for p_ in d.iterdir()) if d.exists() else None
            assert not re.search(rf"arm {a}:", r["log"]) and not re.search(rf"--arm-name {a}\b|--arm {a}\b|arms/{a}/(sft|preflight|monitor|launch_gate)", r["calls"] + r["log"]), ("KILL-STOP: the chain went on to " + a, r["log"][-1500:])
            assert held is None, ("KILL-STOP: the chain wrote into arms/" + a, held)
        assert "R1o" not in later or "safeobj-gate" not in r["calls"], ("KILL-STOP: R1o's S gate was asked after the kill", r["calls"][-600:])
""".replace('\\"', '"').replace("`\\b`", "`\\b`")
OLD_Q_R0D = r'''        r = armloop_run(text, tmp, "al_so_r0d_kill", "R0d R1d R1o", env={"FAKE_TRAINER_R0d": "long", "FAKE_MONITOR_R0d": "kill_rule"}, prepare=prep_all)
        stopped_after(r, "R1d", "R1o")                                                                      # FIRST: the stop itself (a mutation that removes the stop must go RED HERE, tag KILL-STOP)
        assert r["arms_dirs"] == ["C0", "R0d", "R1d"] and sorted(p_.name for p_ in (r["R"] / "arms" / "R1d").iterdir()) == ["safeobj_S.json"], r["arms_dirs"]     # R1d is the fixture's S file, nothing else
'''
NEW_Q_R0D = r'''        r = armloop_run(text, tmp, "al_so_r0d_kill", "R0d R1d R1o", env={"FAKE_TRAINER_R0d": "long", "FAKE_MONITOR_R0d": "kill_rule"}, prepare=with_c0)
        stopped_after(r, "R1d", "R1o")                                                                      # FIRST: the stop itself (a mutation that removes the stop must go RED HERE, tag KILL-STOP)
        assert r["arms_dirs"] == ["C0", "R0d"], r["arms_dirs"]                                              # R1d and R1o were never created
'''
OLD_Q_R1D = r'''        r = armloop_run(text, tmp, "al_so_r1d_kill", "R0d R1d R1o", env={"FAKE_TRAINER_R1d": "long", "FAKE_MONITOR_R1d": "kill_rule"}, prepare=prep_all)
'''
NEW_Q_R1D = r'''        r = armloop_run(text, tmp, "al_so_r1d_kill", "R0d R1d R1o", env={"FAKE_TRAINER_R1d": "long", "FAKE_MONITOR_R1d": "kill_rule"}, prepare=with_c0)
'''
OLD_Q_RERUN = r'''        def killed(R):
            prep_all(R)
'''
NEW_Q_RERUN = r'''        def killed(R):
            with_c0(R)
'''
QUEUE_EDITS = [(OLD_Q_FIX, NEW_Q_FIX), (OLD_Q_PREP, NEW_Q_PREP), (OLD_Q_ORDER, NEW_Q_ORDER), (OLD_Q_NEEDS, NEW_Q_NEEDS), (OLD_Q_STOP, NEW_Q_STOP), (OLD_Q_R0D, NEW_Q_R0D), (OLD_Q_R1D, NEW_Q_R1D),
               (OLD_Q_RERUN, NEW_Q_RERUN)]
