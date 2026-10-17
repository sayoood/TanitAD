#!/usr/bin/env python3
"""Self-test of the GENUINENESS check of `rls1_checks.safeobj_gate` (SAFEOBJ round 3).  Run: python eval/test_safeobj_gate.py -> ZZTEST_PASS <n>.   No pod, no GPU, no network.

THE RULE UNDER TEST.  A gate input is never hand-written or hand-edited.  `safeobj_gate` (which R1o's launch hangs on) used to open on ANY file that said `"S": "PASS"`.  It now opens only on a record that is
a genuine, unmodified output of `eval/safeobj_read.py`:  (a) `tool` is the reader's identity;  (b) `record_sha256` recomputes by the reader's OWN `record_digest` (one shared canonicalisation, no copy);
(c) the record lists the inputs every PASS read has and every listed input's sha256 still matches the file in the arm directory.  Every refusal names its reason.

THE FIXTURES ARE NOT HAND-MADE.  Every arm directory comes from the REAL producers (`test_safeobj_read.build_arm`: RlValChecker, monitor, replay, rl_val_eval over real score files) and every record is
written by the REAL reader (`safeobj_read.read_arm` + `write_record`, and its CLI for the end-to-end case).  What the cases then do to them is exactly what the rule forbids: write one by hand, edit one,
change an input after the read, remove an input.  Each such case is a RED-BY-DESIGN scenario in the package's `must_fail_with` style: the gate must refuse AND the refusal must carry the reason that case
exists for (the EXACT set of reason classes -- a case that is refused for a different or an extra reason fails).

THEN THE GATE ITSELF IS MUTATED.  Each barrier is removed in a copy of the gate's source (a mutant tree: `refe/rls1_checks.py` + `eval/safeobj_read.py` side by side, as on the pod) and the cases that
exist for that barrier -- and ONLY those -- must flip from refused to open (or, for the drift mutations, from open to refused).  A mutation that flips nothing, or something else, fails the test.

WHAT THIS DOES NOT PROVE (stated, not hidden).  `record_sha256` is a checksum, not a signature.  Someone who re-implements the canonicalisation, hashes the real inputs himself and writes a record
whose verdict contradicts them passes (a)-(c); an input that did not exist at the read and appears later is not noticed either.  Closing both is one more check (the gate re-reads the directory and compares
the fresh record's digest); BUILD.md section 8 states it as an open decision.
"""
from __future__ import annotations

import atexit
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent                    # eval/
PKGROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PKGROOT / "refe"))
import safeobj_read as SO  # noqa: E402  (the REAL reader: it writes every genuine record below)
import test_safeobj_read as TS  # noqa: E402  (the producers' fixtures: Fx, build_arm, ALL)

GATE_SRC = PKGROOT / "refe" / "rls1_checks.py"
READER_SRC = HERE / "safeobj_read.py"
KINDS = {"PASS": TS.ALL, "FAIL": {**TS.ALL, "update500": "ttc"}, "NOT_READ": {**TS.ALL, "update750": None}}   # B's variants per tag (test_safeobj_read.VARIANTS)

# the reason classes of a refusal (substrings of the gate's messages; each class is one barrier)
R_TOOL, R_HASH, R_INS, R_NOIN = "tool is", "record_sha256 does not recompute", "inputs_sha256 is missing or not an object", "lists no input"
R_CHG, R_MISS, R_LOAD, R_CRASH = "changed since the read", "' is missing", "cannot be loaded", "cannot verify the record"
CLASSES = (R_TOOL, R_HASH, R_INS, R_NOIN, R_CHG, R_MISS, R_LOAD, R_CRASH)


# ---------------------------------------------------------------------------------------------------------------------------------------------- genuine arm directories (also used by other tests)
def make_fx(root):
    """the producers' fixture (250 sets, 5 proposals; test_safeobj_read.Fx) under <root>/fx."""
    return TS.Fx(Path(root) / "fx")


def genuine_arm(fx, root, kind, arm=TS.ARM):
    """<root>/<kind>_<arm>/<arm>: an arm directory the way the chain leaves it (the REAL producers) with its safeobj_S.json written by the REAL reader.  kind = PASS | FAIL | NOT_READ."""
    d = TS.build_arm(fx, Path(root), f"{kind}_{arm}", KINDS[kind], arm=arm)
    rec = SO.read_arm(str(d))
    SO.write_record(str(d), rec)
    assert rec["S"] == kind, (kind, rec["S"], rec["breaches"], rec["problems"])
    return d


_CACHE: dict = {}


def genuine(kind, arm=TS.ARM):
    """a genuine arm directory for (kind, arm), cached for the life of the process.  READ-ONLY for the caller: copy it (`copy_arm`) before changing anything."""
    if "root" not in _CACHE:
        root = tempfile.mkdtemp(prefix="so_genuine_")
        atexit.register(shutil.rmtree, root, True)
        _CACHE["root"], _CACHE["fx"] = Path(root), make_fx(root)
    if (kind, arm) not in _CACHE:
        _CACHE[(kind, arm)] = genuine_arm(_CACHE["fx"], _CACHE["root"] / "arms", kind, arm)
    return _CACHE[(kind, arm)]


def copy_arm(kind, arm, parent):
    """a private copy of the genuine (kind, arm) directory at <parent>/<arm> (the basename IS the arm: the record and the replay carry it)."""
    dst = Path(parent) / arm
    shutil.copytree(genuine(kind, arm), dst)
    return dst


# ---------------------------------------------------------------------------------------------------------------------------------------------- helpers
def oracle_digest(rec):
    """an INDEPENDENT statement of the record's stamp (compact, key-sorted JSON without the field): the test's own, so a change of the reader's canonicalisation cannot go unseen."""
    return hashlib.sha256(json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsave(d, rec, name="safeobj_S.json"):
    Path(d, name).write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8", newline="\n")


def restamp(rec):
    """the record with a record_sha256 that recomputes (by the test's oracle): what a forger who knows the canonicalisation would write."""
    rec = dict(rec)
    rec["record_sha256"] = oracle_digest(rec)
    return rec


def classes_of(why):
    return {c for c in CLASSES if c in why}


class Case:
    def __init__(self, name, dirs, want, need=(), says=(), reader="real"):
        """want = the gate opens (True) or refuses (False).  need = the EXACT set of reason classes a genuineness refusal carries (empty for an S/IO-layer refusal).  says = substrings the reason must contain."""
        self.name, self.dirs, self.want, self.need, self.says, self.reader = name, [str(x) for x in dirs], want, set(need), tuple(says), reader


# ---------------------------------------------------------------------------------------------------------------------------------------------- the trees the gate runs from
def read_text(p):
    return Path(p).read_text(encoding="utf-8")


def build_tree(root, name, gate_edits=(), reader_edits=(), reader="real"):
    """<root>/<name>/refe/rls1_checks.py (+ eval/safeobj_read.py unless reader == "missing"): the gate and the reader side by side, as on the pod, from the real sources plus the given edits.
    reader = real | missing (no eval/safeobj_read.py) | poisoned (sha256_file raises).  Every edit anchor must occur exactly once."""
    t = Path(root) / name
    (t / "refe").mkdir(parents=True)
    (t / "eval").mkdir()
    g, r = read_text(GATE_SRC), read_text(READER_SRC)
    for old, new in gate_edits:
        assert g.count(old) == 1, f"gate anchor x{g.count(old)}: {old[:80]!r}"
        g = g.replace(old, new)
    for old, new in reader_edits:
        assert r.count(old) == 1, f"reader anchor x{r.count(old)}: {old[:80]!r}"
        r = r.replace(old, new)
    if reader == "poisoned":
        old = "def sha256_file(path):\n    h = hashlib.sha256()\n"
        assert r.count(old) == 1
        r = r.replace(old, 'def sha256_file(path):\n    raise RuntimeError("the reader is poisoned for this test")\n    h = hashlib.sha256()\n')
    (t / "refe" / "rls1_checks.py").write_text(g, encoding="utf-8", newline="\n")
    if reader != "missing":
        (t / "eval" / "safeobj_read.py").write_text(r, encoding="utf-8", newline="\n")
    return t


_GATES: dict = {}


def gate_module(tree):
    p = str(Path(tree) / "refe" / "rls1_checks.py")
    if p not in _GATES:
        spec = importlib.util.spec_from_file_location(f"rls1_checks_under_test_{len(_GATES)}", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _GATES[p] = m
    return _GATES[p]


def run_inproc(tree, dirs):
    return gate_module(tree).safeobj_gate(dirs)


def run_cli(tree, dirs):
    """-> (ok, why, rc, first stdout line): the gate through its own CLI, exactly as rls1_pod.sh calls it."""
    r = subprocess.run([sys.executable, "-X", "utf8", str(Path(tree) / "refe" / "rls1_checks.py"), "safeobj-gate", "--dirs", *dirs], capture_output=True, text=True, timeout=120)
    line = (r.stdout.strip().splitlines() or [""])[0]
    marker, _, why = line.partition(" ")
    return (r.returncode == 0 and marker == "ZZRLS1_SAFEOBJ_S_PASS"), why, r.returncode, line


# ---------------------------------------------------------------------------------------------------------------------------------------------- the cases
def make_cases(tmp):
    tmp = Path(tmp)
    cases = []

    def new_dir(label):
        d = tmp / "cases" / label
        d.mkdir(parents=True)
        return d

    def arm(label, kind="PASS", name=TS.ARM):
        return copy_arm(kind, name, new_dir(label))

    def add(*a, **k):
        cases.append(Case(*a, **k))

    # ---- genuine PASS records open the gate (the reader wrote them; nothing was touched)
    add("genuine_pass", [arm("genuine_pass")], True)
    add("genuine_pass_both_arms", [arm("gp2_r0d", "PASS", "R0d"), arm("gp2_r1d", "PASS", "R1d")], True)
    # ---- 1. a hand-written {"S": "PASS"}
    d = arm("hand_written", "FAIL")
    jsave(d, {"S": "PASS"})
    add("hand_written_S_PASS", [d], False, {R_TOOL, R_HASH, R_INS, R_NOIN}, ("not a genuine reader record",))
    # ... and a cleverer one: the right tool, a stamp that recomputes, an empty list of inputs (vacuous input check)
    d = arm("forged_no_inputs", "FAIL")
    jsave(d, restamp({"tool": SO.TOOL, "arm": d.name, "S": "PASS", "breaches": [], "problems": [], "inputs_sha256": {}}))
    add("forged_stamp_no_inputs", [d], False, {R_NOIN})
    # ---- 2. a genuine FAIL record with S edited to PASS (nothing else touched)
    d = arm("fail_edited", "FAIL")
    rec = jload(d / "safeobj_S.json")
    assert rec["S"] == "FAIL" and rec["breaches"]
    rec["S"] = "PASS"
    jsave(d, rec)
    add("fail_edited_to_PASS", [d], False, {R_HASH})
    # ---- 3. record_sha256 recomputed but `tool` changed
    d = arm("tool_changed")
    rec = jload(d / "safeobj_S.json")
    rec["tool"] = "hand_written/1"
    jsave(d, restamp(rec))
    add("tool_changed_stamp_recomputed", [d], False, {R_TOOL}, ("hand_written/1",))
    # ---- 4. an input modified after the read (a whitespace byte: still valid JSON, so only the hash can tell)
    d = arm("input_modified")
    with open(d / "kill_replay.json", "ab") as f:
        f.write(b" ")
    add("input_modified_after_read", [d], False, {R_CHG}, ("'kill_replay.json'",))
    d = arm("scores_modified")
    with open(d / "scores" / "scores_update1000_B.npz", "ab") as f:
        f.write(b"\0")
    add("scores_modified_after_read", [d], False, {R_CHG}, ("'scores/scores_update1000_B.npz'",))
    # ---- 5. a missing input file
    d = arm("input_missing")
    os.remove(d / "rlval_vs_A0.json")
    add("input_missing_rlval", [d], False, {R_MISS}, ("'rlval_vs_A0.json'",))
    d = arm("scores_missing")
    shutil.rmtree(d / "scores")
    add("input_missing_scores", [d], False, {R_MISS}, ("'scores/scores_update1000_A.npz'",))
    # ---- a genuine record copied from one arm's directory into another's
    d = arm("copied_across", "FAIL", "R0d")
    shutil.copyfile(genuine("PASS", "R1d") / "safeobj_S.json", d / "safeobj_S.json")
    add("record_copied_into_another_arms_dir", [d], False, {R_CHG})
    # ---- shapes: inputs_sha256 not an object (stamped, so only the shape is wrong)
    d = arm("inputs_list")
    rec = jload(d / "safeobj_S.json")
    rec["inputs_sha256"] = []
    jsave(d, restamp(rec))
    add("PASS_with_inputs_not_an_object", [d], False, {R_INS, R_NOIN})
    # ---- the S layer (as before): FAIL, NOT_READ, "PASS " (stamped, so S's exactness is the only barrier), missing, junk, not an object
    add("genuine_FAIL_record", [arm("g_fail", "FAIL")], False, (), ("S='FAIL'",))
    add("genuine_NOT_READ_record", [arm("g_nr", "NOT_READ")], False, (), ("S='NOT_READ'",))
    d = arm("weak_S")
    rec = jload(d / "safeobj_S.json")
    rec["S"] = "PASS "
    jsave(d, restamp(rec))
    add("PASS_with_a_trailing_space", [d], False, (), ("S='PASS '",))
    add("no_file", [new_dir("no_file")], False, (), ("FileNotFoundError",))
    d = new_dir("junk")
    (d / "safeobj_S.json").write_text("{not json", encoding="utf-8")
    add("junk_json", [d], False, (), ("JSONDecodeError",))
    d = new_dir("array")
    (d / "safeobj_S.json").write_text("[1, 2]", encoding="utf-8")
    add("json_but_not_an_object", [d], False, (), ("AttributeError",))
    # ---- the gate needs EVERY arm, in either order
    add("needs_every_arm_fail_second", [arm("ev1_p", "PASS", "R0d"), arm("ev1_f", "FAIL", "R1d")], False, (), ("S='FAIL'",))
    add("needs_every_arm_fail_first", [arm("ev2_f", "FAIL", "R0d"), arm("ev2_p", "PASS", "R1d")], False, (), ("S='FAIL'",))
    # ---- the reader the gate depends on is not there / blows up: fail closed
    add("reader_not_on_disk", [arm("no_reader")], False, {R_LOAD}, reader="missing")
    add("verification_raises", [arm("poisoned")], False, {R_CRASH}, reader="poisoned")
    return cases


def judge(case, ok, why):
    """-> the list of ways this outcome differs from what the case exists for (empty = as designed)."""
    out = []
    if ok is not case.want:
        out.append(f"{'opened' if ok else 'refused'} (wanted {'open' if case.want else 'refusal'}): {why[:200]}")
    elif not case.want:
        if classes_of(why) != case.need:
            out.append(f"refused for the WRONG reason class(es): got {sorted(classes_of(why))}, wanted {sorted(case.need)}: {why[:300]}")
        out.extend(f"reason lacks {s!r}: {why[:300]}" for s in case.says if s not in why)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- the mutations of the gate / reader
G_TOOL, G_HASH = '        if rec.get("tool") != SO.TOOL:\n', '        if rec.get("record_sha256") != SO.record_digest(rec):\n'
MUTATIONS = [   # (name, gate edits, reader edits, the cases that must FLIP (verdict) and nothing else, exact?)
    ("the gate looks at S alone (the old gate)", [("            why = safeobj_not_genuine(d, rec)\n", '            why = ""\n')], [],
     {"hand_written_S_PASS", "forged_stamp_no_inputs", "fail_edited_to_PASS", "tool_changed_stamp_recomputed", "input_modified_after_read", "scores_modified_after_read", "input_missing_rlval",
      "input_missing_scores", "record_copied_into_another_arms_dir", "PASS_with_inputs_not_an_object", "reader_not_on_disk", "verification_raises"}),
    ("`tool` is not compared", [(G_TOOL, "        if False:\n")], [], {"tool_changed_stamp_recomputed"}),
    ("record_sha256 is not recomputed", [(G_HASH, "        if False:\n")], [], {"fail_edited_to_PASS"}),
    ("the listed input sha256s are not compared", [("            elif SO.sha256_file(fp) != want:\n", "            elif False:\n")], [],
     {"input_modified_after_read", "scores_modified_after_read", "record_copied_into_another_arms_dir",
      "verification_raises"}),            # the last one only because its poison IS sha256_file: with the call gone there is nothing left to raise
    ("a missing input file is tolerated", [("                why.append(f\"input {name!r} is missing\")\n", "                pass\n")], [], {"input_missing_rlval", "input_missing_scores"}),
    ("the inputs every PASS read has are not demanded", [("            if name not in ins:\n", "            if False:\n")], [], {"forged_stamp_no_inputs"}),
    ("an exception while verifying opens the gate", [("        return f\"cannot verify the record ({type(e).__name__}: {str(e)[:120]})\"\n", '        return ""\n')], [], {"verification_raises"}),
    ("a reader that cannot be loaded opens the gate", [("        return f\"the reader eval/safeobj_read.py cannot be loaded ({type(e).__name__}: {str(e)[:120]})\"\n", '        return ""\n')], [], {"reader_not_on_disk"}),
    ("S is judged by `== FAIL` instead of `!= PASS`", [('        if s != "PASS":\n', '        if s == "FAIL":\n')], [], {"genuine_NOT_READ_record", "PASS_with_a_trailing_space"}),
    ("the gate keeps its OWN canonicalisation and it has drifted (default separators)",
     [(G_HASH, '        if rec.get("record_sha256") != __import__("hashlib").sha256(json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True).encode("utf-8")).hexdigest():\n')],
     [], {"genuine_pass", "genuine_pass_both_arms"}),
    ("the READER's canonicalisation changes (the gate follows it: one definition)",
     [], [('    body = json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True, separators=(",", ":"))\n',
           '    body = json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True, separators=(", ", ": "))\n')], {"genuine_pass", "genuine_pass_both_arms"}),
    ("the reader no longer names its REQUIRED_INPUTS", [], [('REQUIRED_INPUTS = ("kill_replay.json", "rlval_vs_A0.json")', "REQUIRED_INPUTS = ()")], {"forged_stamp_no_inputs"}),
]


def main() -> int:
    bad: list = []
    n = 0

    def check(name, ok, detail=""):
        nonlocal n
        n += 1
        if not ok:
            bad.append(f"{name} {detail}".strip())

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # ---- the fixtures are what they say: genuine records of the three kinds, each stamped as the INDEPENDENT oracle stamps it
        for kind in KINDS:
            rec = jload(genuine(kind) / "safeobj_S.json")
            check(f"fixture_{kind}_is_a_real_reader_record", rec["S"] == kind and rec["tool"] == SO.TOOL and rec["record_sha256"] == oracle_digest(rec) == SO.record_digest(rec))
        check("fixture_PASS_lists_the_required_inputs", set(SO.REQUIRED_INPUTS) <= set(jload(genuine("PASS") / "safeobj_S.json")["inputs_sha256"]) and len(SO.REQUIRED_INPUTS) == 2)
        # the stamp's canonicalisation is ONE function: both stamping sites of the reader (a normal read and the 'could not interpret' record) agree with the oracle
        d = copy_arm("PASS", "R0d", tmp / "garb")
        jsave(d, {"arm": "R0d", "kill": None, "checks": [{"tag": ["update250"]}]}, "kill_replay.json")                 # an unhashable tag makes the reader RAISE inside: the except-path record
        rec = SO.read_arm(str(d))
        check("fixture_except_path_record_is_stamped_by_the_same_function", "could not interpret" in " ".join(rec["problems"]) and rec["record_sha256"] == oracle_digest(rec) == SO.record_digest(rec))
        src = read_text(READER_SRC)
        check("pin_reader_has_exactly_one_canonicalisation_and_two_stamping_sites", src.count('separators=(",", ":")') == 1 and src.count("record_digest(out)") == 2)
        gsrc = inspect.getsource(gate_module(PKGROOT).safeobj_not_genuine)
        check("pin_gate_keeps_no_copy_of_the_canonicalisation_or_the_tool_name", all(x in gsrc for x in ("SO.record_digest(", "SO.TOOL", "SO.sha256_file(", "SO.REQUIRED_INPUTS"))
              and not any(x in gsrc for x in ("json.dumps", "separators", "sha256(", "safeobj_read/1")))

        n_fix = n
        cases = make_cases(tmp)
        real = PKGROOT
        pod = build_tree(tmp / "trees", "base", reader="real")                                  # the gate and the reader side by side in a tree of their own (the shape the mutants have)
        trees = {"real": real, "missing": build_tree(tmp / "trees", "base_noreader", reader="missing"), "poisoned": build_tree(tmp / "trees", "base_poisoned", reader="poisoned")}

        # ---- 1. the REAL gate (the deliverable's own files), in-process AND through its CLI, on every case: opens on a genuine PASS, refuses the rest for the right reason
        for c in cases:
            tree = trees[c.reader]
            ok, why = run_inproc(tree, c.dirs)
            probs = judge(c, ok, why)
            check(f"inproc:{c.name}", not probs, "; ".join(probs))
            cok, cwhy, rc, line = run_cli(tree, c.dirs)
            cprobs = judge(c, cok, cwhy)
            check(f"cli:{c.name}", not cprobs and rc == (0 if c.want else 1) and line.startswith("ZZRLS1_SAFEOBJ_S_PASS" if c.want else "ZZRLS1_SAFEOBJ_S_REFUSED"), "; ".join(cprobs) + f" rc={rc} {line[:120]}")
        # the tree built from the real sources behaves exactly as the files in place (the mutants below start from it)
        for c in cases[:3]:
            check(f"copy_of_the_sources_behaves_like_the_files_in_place:{c.name}", run_inproc(pod, c.dirs)[0] is c.want)

        n_e2e = n
        # ---- 2. end to end from nothing: the reader's CLI writes the record, then the gate opens (in-process and CLI); the same directory with one input touched afterwards is refused
        d = copy_arm("PASS", "R0d", tmp / "e2e")
        os.remove(d / "safeobj_S.json")
        r = subprocess.run([sys.executable, "-X", "utf8", str(READER_SRC), "--arm-dir", str(d)], capture_output=True, text=True, timeout=120)
        check("e2e_reader_cli_rc_0_and_writes_the_record", r.returncode == 0 and (d / "safeobj_S.json").exists() and "ZZSAFEOBJ_S R0d PASS" in r.stdout, r.stdout[-200:] + r.stderr[-200:])
        check("e2e_gate_opens_on_the_record_the_reader_cli_wrote_in_process", run_inproc(real, [str(d)])[0] is True)
        check("e2e_gate_opens_on_the_record_the_reader_cli_wrote_through_its_cli", run_cli(real, [str(d)])[2] == 0)
        with open(d / "kill_replay.json", "ab") as f:
            f.write(b"\n")
        ok, why = run_inproc(real, [str(d)])
        check("e2e_the_same_directory_is_refused_once_an_input_is_touched", ok is False and classes_of(why) == {R_CHG}, why[:200])

        n_e2e = n - n_e2e
        # ---- 3. the barriers: remove each in a copy of the source; exactly the cases that exist for it flip
        for i, (name, gedits, redits, flips) in enumerate(MUTATIONS):
            mt = {k: build_tree(tmp / "trees", f"m{i}_{k}", gedits, redits, reader=("real" if k == "real" else k)) for k in ("real", "missing", "poisoned")}
            got = set()
            for c in cases:
                ok, why = run_inproc(mt[c.reader], c.dirs)
                if ok is not c.want:
                    got.add(c.name)
            check(f"mutation:{name}", got == flips, f"flipped {sorted(got)}, expected exactly {sorted(flips)}")
            print(f"  RED  {name}  ->  {len(got)} case(s) flip: {', '.join(sorted(got))[:200]}", flush=True)

    if bad:
        raise AssertionError(f"{len(bad)} of {n} checks FAILED:\n  " + "\n  ".join(bad))
    print(f"ZZTEST_PASS {n}  ({len(cases)} cases x (in-process + CLI) on the real gate, {n_e2e} end-to-end checks, {len(MUTATIONS)} mutations of the gate / reader that must flip exactly their cases, {n_fix} fixture/pin checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
