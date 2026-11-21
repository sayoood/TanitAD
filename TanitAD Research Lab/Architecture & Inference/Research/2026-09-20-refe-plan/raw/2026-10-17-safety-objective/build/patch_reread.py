#!/usr/bin/env python3
"""patch_reread: round 4 -- `rls1_checks.safeobj_gate` RE-READS the arm directory and requires the stored record to equal the fresh one; the pod precheck requires the S-reader and its two tests.
usage: patch_reread.py <PK dir>

Why.  Round 3's gate (patch_genuine.py) checks (a) the tool, (b) the stamp, (c) the listed inputs' sha256s.  Two things get through (a)-(c) (BUILD.md section 8.6, Master Mind decision 8):
  * a forger who re-implements the canonicalisation, hashes the real inputs himself and writes a record whose verdict contradicts them;
  * an input that did not exist at the read and appears later (a KILLED.json, kill_checks.jsonl, scores/) -- the record lists only what existed, so (c) cannot see it.
Round 4 adds (d): when (a)-(c) found nothing wrong, the gate re-runs the reader's OWN judgement in-process on the arm directory as it is now (`safeobj_read.read_arm`) and requires the stored record to equal the
fresh one (`safeobj_read.differing_fields`: every field except the stamp and `VOLATILE_FIELDS`, which is EMPTY -- the record is a pure function of the directory's files: no clock, no path, no host).
Refusal: "record is stale or not the reader's output (a fresh read of the directory differs in: <fields>)".  The gate stays read-only.
Second, the pod precheck (`rls1_pod.sh`) lists `eval/safeobj_read.py`, `eval/test_safeobj_read.py` and `eval/test_safeobj_gate.py` (the gate loads the first BY PATH, the launch gate's suite runs the other two),
so a package without one is refused in seconds, before the queue, instead of failing R1o's gate or the suite later; `refe/test_rls1_chain.py` pins the list (static, derived from the helper's own EVAL_DIR loads,
and by RUNNING the chain against a package lacking each one), and the header comment of `rls1_pod.sh` now says the run is TWO steps.

What it edits (each anchor occurs exactly once, each base blob is the round-3 tree's -- C:/so/t6 -- and is checked; a tree that is not the round-3 tree is refused, never half-patched):
  eval/safeobj_read.py     +VOLATILE_FIELDS (empty) +comparable +differing_fields (docstring)
  refe/rls1_checks.py      safeobj_not_genuine: (d) the re-read; docstrings; the OK message
  refe/rls1_pod.sh         the precheck list (+3 eval files); the header comment (the run is two steps)
  refe/test_rl_safeobj.py  CH: +2 checks, +CH16 (no re-read) +CH17 (the precheck does not ship the S-reader)
  refe/test_rls1_chain.py  SAFEOBJ_FILES: the static pin, the derived pin, 4 mutations, the RUN against a package lacking each file
  refe/test_rls1_queue.py  R1o's gate through the real chain: +2 refusal variants (a forged verdict, a KILLED marker after the read)
  eval/test_safeobj_read.py  one expected-RED set loses a catcher that the stronger gate made redundant (see READER_TEST_EDITS)
  eval/test_safeobj_gate.py  replaced by build/test_safeobj_gate_r4.py (round 3's file + the round-4 cases, mutations and checks; the base blob is checked)
Order: last, after patch_genuine.py."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patchlib as PL  # noqa: E402

# the blobs of the round-3 tree C:/so/t6 (TEST_TABLE.md "Patched-tree blobs" of round 3)
BASES = {"eval/safeobj_read.py": "a0884c84484a5f9597ccdc7c9adf0d094f8e803e", "refe/rls1_checks.py": "f853bcc437d5bd1d1b19ad3c52666c27cbe3a5f2", "refe/rls1_pod.sh": "8fe00f9f78f730f4a2b3d7296b90b4d1bc42887e",
         "refe/test_rl_safeobj.py": "a2570f134e148df4af385e541e3d88cbc5af212c", "refe/test_rls1_chain.py": "c3c361b22ee9fa9bd2052bde5d1b5408c63d4934",
         "refe/test_rls1_queue.py": "2d4087902757034d4137d706c32be308c06cfe8d", "eval/test_safeobj_gate.py": "89f3356b21aec469dfc6cf0dee155a9bdcaab755",
         "eval/test_safeobj_read.py": "0b91698b488f49b7f023995483d14be40816b343"}

# ------------------------------------------------------------------------------------------------------------------------------------------------ eval/safeobj_read.py
READER_EDITS = [
    ("or whose listed input sha256s no longer match the files in the arm directory.",
     "whose listed input sha256s no longer match the files in the arm directory, or which a fresh read of the arm directory does not reproduce\n"
     "(`differing_fields` below: the gate runs `read_arm` again on the directory as it is NOW and the two records must be equal)."),
    ('REQUIRED_INPUTS = ("kill_replay.json", "rlval_vs_A0.json")       # a PASS record always lists these two (the reader fails closed without either); the gate refuses a record that lists fewer\n',
     'REQUIRED_INPUTS = ("kill_replay.json", "rlval_vs_A0.json")       # a PASS record always lists these two (the reader fails closed without either); the gate refuses a record that lists fewer\n'
     'VOLATILE_FIELDS = ()                             # the record fields a fresh read may legitimately change: NONE.  The record is a pure function of the arm directory\'s files (no clock, no path, no host name), so a\n'
     '                                                 # fresh read of an unchanged directory reproduces it EXACTLY.  A field that ever carries a time or a host must be named HERE, or every genuine record turns "stale".\n'),
    ("def _num(x):\n",
     'def comparable(rec):\n'
     '    """the part of a record that a fresh read of the same directory must reproduce: every field except the stamp (it is derived from the rest, and the gate checks it on its own) and VOLATILE_FIELDS (none)."""\n'
     '    return {k: v for k, v in rec.items() if k != "record_sha256" and k not in VOLATILE_FIELDS}\n'
     '\n\n'
     'def differing_fields(stored, fresh):\n'
     '    """-> the sorted names of the fields in which `stored` differs from `fresh` (empty = `fresh`, a read of the directory as it is now, reproduces `stored`).  Each field is compared as its canonical JSON, i.e.\n'
     '    as the file holds it (a tuple is a list, a float is its repr), and a field present in one record only differs.  `rls1_checks.safeobj_gate` calls this with the stored record and `read_arm(<dir>)`."""\n'
     '    a, b = comparable(stored), comparable(fresh)\n'
     '    dump = lambda v: json.dumps(v, sort_keys=True)                   # noqa: E731\n'
     '    return sorted(k for k in set(a) | set(b) if k not in a or k not in b or dump(a[k]) != dump(b[k]))\n'
     '\n\n'
     'def _num(x):\n'),
]

# ------------------------------------------------------------------------------------------------------------------------------------------------ refe/rls1_checks.py
GATE_EDITS = [
    ("TOOL (its identity), record_digest (the canonicalisation of record_sha256), sha256_file, REQUIRED_INPUTS.  The gate calls them and keeps no copy of any.\"\"\"",
     "TOOL (its identity), record_digest (the canonicalisation of record_sha256), sha256_file, REQUIRED_INPUTS, and read_arm / differing_fields (the reader's own judgement, re-run).  The gate calls them and keeps\n"
     "    no copy of any.\"\"\""),
    ("    and every input it lists still has the listed sha256 in <d>.  A CHECKSUM, not a signature: it refuses a hand-written or edited record and a record whose inputs changed after the read; it\n"
     "    cannot refuse someone who re-implements the reader's canonicalisation and hashes the files himself (BUILD.md section 8 says what would).\"\"\"\n",
     "    and every input it lists still has the listed sha256 in <d>; (d) -- only for a record that (a)-(c) found nothing wrong with -- the reader's OWN judgement, re-run in-process on <d> as it is NOW\n"
     "    (`SO.read_arm`), reproduces the stored record (`SO.differing_fields`: every field but the stamp and the reader's VOLATILE_FIELDS, which is empty).  (a)-(c) give the specific reasons; (d) is the\n"
     "    sufficient check: a record equal to a fresh read IS what the reader writes for this directory now, so it refuses a forger who re-implements the canonicalisation and hashes the inputs himself, and a\n"
     "    KILLED.json / kill_checks.jsonl / scores/ that appeared after the read.  Read-only: it writes nothing.  Still trusted, and said so in BUILD.md section 8.12: the reader's and the gate's own source, and\n"
     "    the producers' files in <d> (a producer file edited BEFORE the reader ran is read as it stands).\"\"\"\n"),
    ("                why.append(f\"input {name!r} changed since the read (sha256 differs)\")\n        return \"; \".join(why)\n",
     "                why.append(f\"input {name!r} changed since the read (sha256 differs)\")\n"
     "        if not why:                  # (d) the re-read: the stored record must be what the reader writes for <d> NOW (a forger's PASS, or an input that appeared after the read, is not)\n"
     "            diff = SO.differing_fields(rec, SO.read_arm(d))\n"
     "            if diff:\n"
     "                why.append(\"record is stale or not the reader's output (a fresh read of the directory differs in: \" + \", \".join(diff) + \")\")\n"
     "        return \"; \".join(why)\n"),
    ("(safeobj_not_genuine: its tool, its record_sha256, its inputs' sha256s)",
     "(safeobj_not_genuine: its tool, its record_sha256, its inputs' sha256s, and a fresh read of the directory reproducing it)"),
    ("each a genuine reader record with its inputs unchanged\"",
     "each a genuine reader record: its inputs unchanged and a fresh read of the arm directory reproducing it\""),
]

# ------------------------------------------------------------------------------------------------------------------------------------------------ refe/rls1_pod.sh
POD_EDITS = [
    ("eval/rl_terms_join_identity.py SHIP_MANIFEST.json \\\n",
     "eval/rl_terms_join_identity.py eval/safeobj_read.py eval/test_safeobj_read.py eval/test_safeobj_gate.py SHIP_MANIFEST.json \\\n"),
    ('# Run as  RLS1_ARMS="R0d R1d R1o"  (a K3 kill of R0d or R1d STOPS the chain; R1o refuses unless R0d and R1d both read S = PASS: rls1_checks.py safeobj-gate; R1o\'s monitor and replay are --observe-only).\n',
     '# Run in TWO STEPS, never as the one list "R0d R1d R1o":  (1) RLS1_ARMS="R0d R1d"  (a K3 kill of R0d or R1d STOPS the chain);  (2) the operator runs the reader  python eval/safeobj_read.py --arm-dir $R/arms/R0d\n'
     '# --arm-dir $R/arms/R1d  (this chain never runs it: a bar-S record is the READER\'s output, never hand-written, and it can exist only once both arms have finished -- the chain itself rewrites an arm\'s kill_replay.json,\n'
     '# rlval_vs_A0.json and scores/, so a record made earlier would be refused as stale);  (3) RLS1_ARMS="R1o".  R1o refuses (before any GPU work) unless R0d and R1d both carry a GENUINE safeobj_S.json reading S = PASS, every input unchanged since the read and a fresh\n'
     '# read of the directory reproducing the record: rls1_checks.py safeobj-gate.  The one list trains R0d and R1d and then R1o\'s gate refuses (nothing of R1o is created).  R1o\'s monitor and replay are --observe-only.\n'),
]

# ------------------------------------------------------------------------------------------------------------------------------------------------ refe/test_rl_safeobj.py
SAFEOBJ_TEST_EDITS = [
    ('    check(bad, "CH_a_missing_second_control_is_ANNOUNCED_not_refused", lambda:',
     r'''    check(bad, "CH_the_precheck_ships_the_S_reader_and_the_two_tests_that_pin_the_S_gate_a_missing_one_is_refused_before_the_queue",
          lambda: " eval/safeobj_read.py eval/test_safeobj_read.py eval/test_safeobj_gate.py SHIP_MANIFEST.json \\\n" in pod_src)
    check(bad, "CH_a_missing_second_control_is_ANNOUNCED_not_refused", lambda:'''),
    ('        check(bad, "CH_safeobj_gate_refuses_a_hand_written_PASS_an_edited_record_and_a_record_whose_input_changed_each_for_its_reason", lambda: all(\n',
     r'''        d_forged = rewrite(own("fg", "FAIL", "R1d"), lambda r: restamp({**r, "S": "PASS", "breaches": []}))          # a forger who re-implements the canonicalisation: tool, stamp and inputs are right, the verdict is a lie
        d_late = own("late", "PASS", "R1d")                                                                           # a genuine PASS read, and then a K3 marker appears in the arm directory
        with open(os.path.join(d_late, "KILLED.json"), "w", encoding="utf-8") as f_:
            json.dump({"arm": "R1d", "code": "K3", "why": "two consecutive breaches (this marker appeared after the read)"}, f_)
        check(bad, "CH_safeobj_gate_refuses_a_forger_who_recomputes_the_stamp_and_a_KILLED_marker_that_appears_after_the_read_each_as_stale", lambda: all(
            RC.safeobj_gate([d_pass, x])[0] is False and RC.safeobj_gate([x, d_pass])[0] is False and "record is stale or not the reader's output" in RC.safeobj_gate([d_pass, x])[1] for x in (d_forged, d_late)))
        check(bad, "CH_safeobj_gate_refuses_a_hand_written_PASS_an_edited_record_and_a_record_whose_input_changed_each_for_its_reason", lambda: all(
'''),
    (r'''    ("CH15 safeobj_gate does not verify the record's genuineness", "CH", [(RC_PATH, "            why = safeobj_not_genuine(d, rec)\n", '            why = ""\n')], None),
''',
     r'''    ("CH15 safeobj_gate does not verify the record's genuineness", "CH", [(RC_PATH, "            why = safeobj_not_genuine(d, rec)\n", '            why = ""\n')], None),
    ("CH16 safeobj_gate does not re-read the arm directory", "CH", [(RC_PATH, "            diff = SO.differing_fields(rec, SO.read_arm(d))\n", "            diff = []\n")], None),
    ("CH17 the pod precheck does not ship the S-reader and its two tests", "CH", [(POD_PATH, " eval/safeobj_read.py eval/test_safeobj_read.py eval/test_safeobj_gate.py SHIP_MANIFEST.json \\\n", " SHIP_MANIFEST.json \\\n")], None),
'''),
    (r'''    "CH15 safeobj_gate does not verify the record's genuineness": ["CH_safeobj_gate_refuses_a_hand_written_PASS_an_edited_record_and_a_record_whose_input_changed_each_for_its_reason"],
''',
     r'''    "CH15 safeobj_gate does not verify the record's genuineness": ["CH_safeobj_gate_refuses_a_hand_written_PASS_an_edited_record_and_a_record_whose_input_changed_each_for_its_reason"],
    "CH16 safeobj_gate does not re-read the arm directory": ["CH_safeobj_gate_refuses_a_forger_who_recomputes_the_stamp_and_a_KILLED_marker_that_appears_after_the_read_each_as_stale"],
    "CH17 the pod precheck does not ship the S-reader and its two tests": ["CH_the_precheck_ships_the_S_reader_and_the_two_tests_that_pin_the_S_gate_a_missing_one_is_refused_before_the_queue"],
'''),
]

# ------------------------------------------------------------------------------------------------------------------------------------------------ refe/test_rls1_chain.py
CHAIN_TEST_EDITS = [
    ("KILL_HEAD_RX = re.compile(",
     'SAFEOBJ_FILES = ("eval/safeobj_read.py", "eval/test_safeobj_read.py", "eval/test_safeobj_gate.py")      # the gate loads the first BY PATH (rls1_checks._safeobj_reader), the launch gate\'s suite runs the other two: absent = a refusal AFTER the queue wait\n'
     "KILL_HEAD_RX = re.compile("),
    ('        if "SHIP_MANIFEST.json" not in fl.group(1).split():\n            v.append("SHIP_MANIFEST_NOT_REQUIRED")\n',
     '        if "SHIP_MANIFEST.json" not in fl.group(1).split():\n            v.append("SHIP_MANIFEST_NOT_REQUIRED")\n'
     '        v += ["SAFEOBJ_FILE_NOT_REQUIRED:" + f for f in SAFEOBJ_FILES if f not in fl.group(1).split()]\n'
     "        gate_loads = sorted(set(re.findall(r'EVAL_DIR / \"([A-Za-z0-9_]+\\.py)\"', (HERE / \"rls1_checks.py\").read_text(encoding=\"utf-8\"))))          # every file the chain's helper loads from eval/ BY PATH\n"
     '        v += ["GATE_LOADS_A_FILE_THE_PRECHECK_DOES_NOT_REQUIRE:eval/" + f for f in gate_loads if "eval/" + f not in fl.group(1).split()]\n'),
    (r'''        ("the manifest is not required", sw(" SHIP_MANIFEST.json \\", " \\"), "SHIP_MANIFEST_NOT_REQUIRED"),
''',
     r'''        ("the manifest is not required", sw(" SHIP_MANIFEST.json \\", " \\"), "SHIP_MANIFEST_NOT_REQUIRED"),
        ("the S-reader is not required by the precheck", sw(" eval/safeobj_read.py", ""), "SAFEOBJ_FILE_NOT_REQUIRED:eval/safeobj_read.py"),
        ("the S-reader's own test is not required by the precheck", sw(" eval/test_safeobj_read.py", ""), "SAFEOBJ_FILE_NOT_REQUIRED:eval/test_safeobj_read.py"),
        ("the S-gate's test is not required by the precheck", sw(" eval/test_safeobj_gate.py", ""), "SAFEOBJ_FILE_NOT_REQUIRED:eval/test_safeobj_gate.py"),
        ("the helper loads eval/safeobj_read.py by path and the precheck does not require it", sw(" eval/safeobj_read.py", ""), "GATE_LOADS_A_FILE_THE_PRECHECK_DOES_NOT_REQUIRE:eval/safeobj_read.py"),
'''),
    ("    # a package with every required file, then the env stage, then model_final: absent -> refused; a wrong file -> refused\n",
     "    # the S-reader and the two tests that pin the S gate are REQUIRED by the precheck (SAFEOBJ round 4): with every other listed file present, a package lacking any ONE of them is refused by the chain\n"
     "    # itself, in seconds, naming it -- before the queue, the disk test and the GPU\n"
     "    listed = re.search(r'for f in (.*?); do\\n  \\[ -f', text, re.S).group(1).replace(\"\\\\\\n\", \" \").split()\n"
     "    for skip in SAFEOBJ_FILES:\n"
     "        assert skip in listed, skip; n += 1\n"
     "        for f in listed:\n"
     "            if f != skip:\n"
     "                (pk / f).parent.mkdir(parents=True, exist_ok=True)\n"
     "                (pk / f).write_text(\"\")\n"
     "        if (pk / skip).exists():\n"
     "            (pk / skip).unlink()\n"
     "        rc, out = run(); assert rc == 1 and f\"/{skip} missing: ship\" in out and \"ZZRLS1_QUEUE_CLEAR\" not in out, (skip, out); n += 1\n"
     "    # a package with every required file, then the env stage, then model_final: absent -> refused; a wrong file -> refused\n"),
]

# ------------------------------------------------------------------------------------------------------------------------------------------------ refe/test_rls1_queue.py
QUEUE_TEST_EDITS = [
    ('        for nm, prep, why in (("none", None, "FileNotFoundError"),',
     r'''        def forged(R):                                                                                      # a forger who re-implements the canonicalisation: tool, stamp and inputs are right, the verdict is a lie
            s_files(("R0d", "PASS"), ("R1d", "FAIL"))(R)
            rec_ = TG.jload(R / "arms" / "R1d" / "safeobj_S.json")
            assert rec_["S"] == "FAIL" and rec_["breaches"]
            TG.jsave(R / "arms" / "R1d", TG.restamp({**rec_, "S": "PASS", "breaches": []}))

        def late_kill(R):                                                                                   # a genuine PASS read, and a K3 marker appears in the arm directory afterwards
            s_files(("R0d", "PASS"), ("R1d", "PASS"))(R)
            (R / "arms" / "R1d" / "KILLED.json").write_text('{"arm": "R1d", "code": "K3", "why": "this marker appeared after the read"}', encoding="utf-8")
        for nm, prep, why in (("none", None, "FileNotFoundError"),'''),
    ('                              ("stale_input", stale, "changed since the read")):\n',
     '                              ("stale_input", stale, "changed since the read"), ("forged_verdict", forged, "record is stale or not the reader\'s output"),\n'
     '                              ("kill_marker_after_the_read", late_kill, "record is stale or not the reader\'s output")):\n'),
]

# ------------------------------------------------------------------------------------------------------------------------------------------------ eval/test_safeobj_read.py
# One expectation changes, and it changes because the gate got STRONGER: the mutation "a NOT_READ arm is written as PASS" used to be caught (among others) by the e2e check that the REAL gate refuses the
# mutated reader's PASS.  The gate now re-reads the directory with the REAL reader (not the mutant) and refuses that PASS as stale, so the check stays green; the mutation is still RED in the reader's own
# scenarios and in the CLI's exit code.
READER_TEST_EDITS = [
    ('     {"scenario:missing_u750", "scenario:no_replay", "e2e:cli_missing_u750_real_gate_refuses"}),' + chr(10),
     '     {"scenario:missing_u750", "scenario:no_replay", "e2e:cli_missing_u750_rc_2"}),     # round 4: the real gate is no longer a third catcher here -- it re-reads the directory with the REAL reader and refuses the mutant PASS as stale' + chr(10)),
]

EDITS = {"eval/safeobj_read.py": READER_EDITS, "refe/rls1_checks.py": GATE_EDITS, "refe/rls1_pod.sh": POD_EDITS, "refe/test_rl_safeobj.py": SAFEOBJ_TEST_EDITS,
         "refe/test_rls1_chain.py": CHAIN_TEST_EDITS, "refe/test_rls1_queue.py": QUEUE_TEST_EDITS, "eval/test_safeobj_read.py": READER_TEST_EDITS}


def main():
    pk = PL.pk_arg()
    for rel, edits in EDITS.items():
        r = PL.patch_file(os.path.join(pk, rel), edits, expect_before=BASES[rel], label=rel)
        print({k: r[k] for k in ("file", "blob_before", "blob_after", "edits")})
    dst = os.path.join(pk, "eval", "test_safeobj_gate.py")
    before = PL.git_blob(PL.read(dst))
    if before != BASES["eval/test_safeobj_gate.py"]:
        raise SystemExit(f"[eval/test_safeobj_gate.py] base blob {before} != expected {BASES['eval/test_safeobj_gate.py']}: the tree is not the round-3 tree")
    text = PL.read(os.path.join(HERE, "test_safeobj_gate_r4.py"))
    PL.write(dst, text)
    print({"file": "eval/test_safeobj_gate.py", "blob_before": before, "blob_after": PL.git_blob(text), "replaced_by": "build/test_safeobj_gate_r4.py", "bytes": len(text.encode("utf-8"))})


if __name__ == "__main__":
    main()
