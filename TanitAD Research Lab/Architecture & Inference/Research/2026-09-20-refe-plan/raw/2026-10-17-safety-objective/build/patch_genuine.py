#!/usr/bin/env python3
"""patch_genuine: round 3 -- `rls1_checks.safeobj_gate` opens only on a GENUINE, UNMODIFIED output of eval/safeobj_read.py.  usage: patch_genuine.py <PK dir>

Why.  At the end of round 2 the gate that guards R1o opened on ANY file saying `"S": "PASS"`: a hand-written one passed (BUILD.md section 5 item 5, section 7.2 "Open").  Programme rule: a gate input is
never hand-written or hand-edited.  The reader already stamps every record with `tool` and `record_sha256` and lists the sha256 of every input it read; this step makes the gate verify them:
  (a) `tool` is the reader's identity (safeobj_read.TOOL);
  (b) `record_sha256` recomputes over the record by the reader's OWN function (safeobj_read.record_digest -- ONE canonicalisation: the reader stamps with it, the gate imports it, there is no copy);
  (c) the record lists the inputs every PASS read has (safeobj_read.REQUIRED_INPUTS) and every input it lists still has the listed sha256 in the arm directory.
Every refusal names its reason.  FAIL / NOT_READ / missing / unparsable files are refused exactly as before; the gate still needs EVERY arm to pass.

What it edits (each anchor occurs exactly once, each base blob is checked; a tree that is not the round-2 tree is refused, never half-patched):
  eval/safeobj_read.py     +record_digest (the one canonicalisation, used at both stamping sites), +REQUIRED_INPUTS; the output of the reader is byte-identical (the test pins it)
  refe/rls1_checks.py      safeobj_gate +_safeobj_reader +safeobj_not_genuine
  refe/test_rl_safeobj.py  the CH block: the S-gate fixtures were hand-written `{"S": "PASS"}`; they are now genuine records (the real producers + the real reader); +1 mutation (CH15)
  refe/test_rls1_queue.py  the SAFEOBJ scenarios: the pre-seeded hand-written S files are gone (the real gate would refuse them); R1o's gate is exercised with genuine records in a second chain run
and adds the new file eval/test_safeobj_gate.py (build/test_safeobj_gate.py).
Order: after add_reader.py (it needs eval/safeobj_read.py and eval/test_safeobj_read.py) and gen_safeobj_test.py (it edits the generated refe/test_rl_safeobj.py)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patchlib as PL  # noqa: E402

# the blobs of the round-2 tree C:/so/t4 (TEST_TABLE.md "Patched-tree blobs")
BASES = {"eval/safeobj_read.py": "2fb534a1186497ec9483f4b627dde41aa1235867", "refe/rls1_checks.py": "f9548d8c125eb0b751ae9a457042ffd7ae751a4c",
         "refe/test_rl_safeobj.py": "648b824d9d49b84ea510b515b03fd0e64804189c", "refe/test_rls1_queue.py": "2b17ef67ce20d5095b83f4ea78cc3d91331013a9"}

# ------------------------------------------------------------------------------------------------------------------------------------------------ eval/safeobj_read.py
READER_EDITS = [
    ('That file is written by THIS tool and by nothing else; a hand-written one defeats the gate\'s purpose (the gate only looks at "S").',
     'That file is written by THIS tool and by nothing else, and the gate checks that it was: `rls1_checks.safeobj_gate` refuses a record whose `tool` is not TOOL, whose `record_sha256` does not\n'
     'recompute (`record_digest` below -- the ONE canonicalisation, imported by the gate, never copied), or whose listed input sha256s no longer match the files in the arm directory.'),
    ('OUT_NAME = "safeobj_S.json"\n',
     'OUT_NAME = "safeobj_S.json"\n'
     'REQUIRED_INPUTS = ("kill_replay.json", "rlval_vs_A0.json")       # a PASS record always lists these two (the reader fails closed without either); the gate refuses a record that lists fewer\n'),
    ('def _num(x):\n',
     'def record_digest(rec):\n'
     '    """THE canonicalisation of `record_sha256`: sha256 of the compact, key-sorted JSON of the record WITHOUT that field.  The reader stamps every record with it and `rls1_checks.safeobj_gate`\n'
     '    recomputes it by calling THIS function (it keeps no copy), so the stamp and the check cannot drift apart."""\n'
     '    body = json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True, separators=(",", ":"))\n'
     '    return hashlib.sha256(body.encode("utf-8")).hexdigest()\n'
     '\n\n'
     'def _num(x):\n'),
    ('        out["record_sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()\n',
     '        out["record_sha256"] = record_digest(out)\n'),
    ('    body = json.dumps({k: v for k, v in out.items() if k != "record_sha256"}, sort_keys=True, separators=(",", ":"))\n'
     '    out["record_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()\n',
     '    out["record_sha256"] = record_digest(out)                                # stamped last; the gate recomputes it with the same function\n'),
]

# ------------------------------------------------------------------------------------------------------------------------------------------------ refe/rls1_checks.py
OLD_GATE = '''def safeobj_gate(dirs):
    """SAFEOBJ (PREREG_SAFEOBJ.md section 4 / outcome table): R1o runs only if EVERY constrained arm read bar S = PASS.  Each <dir>/safeobj_S.json must exist, parse, and carry "S": "PASS"
    (written by the reader, never by hand).  -> (ok, why); a missing, unparsable or non-PASS file is a refusal."""
    bad = []
    for d in dirs:
        p = os.path.join(d, "safeobj_S.json")
        try:
            with open(p, encoding="utf-8") as f:
                s = json.load(f).get("S")
        except (OSError, ValueError, AttributeError) as e:
            bad.append(f"{p}: {type(e).__name__}")
            continue
        if s != "PASS":
            bad.append(f"{p}: S={s!r}")
    return (not bad), (f"S = PASS on {len(dirs)} constrained arm(s)" if not bad else "; ".join(bad))
'''
NEW_GATE = '''def _safeobj_reader():
    """eval/safeobj_read.py of THIS checkout, loaded by path (never through sys.path or sys.modules, so a gate loaded from one tree cannot use another tree's reader): the one place that says what a
    genuine bar-S record is -- TOOL (its identity), record_digest (the canonicalisation of record_sha256), sha256_file, REQUIRED_INPUTS.  The gate calls them and keeps no copy of any."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("safeobj_read_of_the_gate", str(EVAL_DIR / "safeobj_read.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def safeobj_not_genuine(d, rec):
    """-> "" when `rec` (the parsed <d>/safeobj_S.json) is a genuine, unmodified output of eval/safeobj_read.py, else the reasons joined by "; ".  Fail closed: anything unexpected is a reason.
    (a) `tool` is the reader's identity; (b) `record_sha256` recomputes over the record by the reader's own `record_digest`; (c) the record lists the inputs every PASS read has (REQUIRED_INPUTS)
    and every input it lists still has the listed sha256 in <d>.  A CHECKSUM, not a signature: it refuses a hand-written or edited record and a record whose inputs changed after the read; it
    cannot refuse someone who re-implements the reader's canonicalisation and hashes the files himself (BUILD.md section 8 says what would)."""
    try:
        SO = _safeobj_reader()
    except Exception as e:                                                                                                   # noqa: BLE001
        return f"the reader eval/safeobj_read.py cannot be loaded ({type(e).__name__}: {str(e)[:120]})"
    try:
        why = []
        if rec.get("tool") != SO.TOOL:
            why.append(f"tool is {rec.get('tool')!r}, not the reader's {SO.TOOL!r}")
        if rec.get("record_sha256") != SO.record_digest(rec):
            why.append("record_sha256 does not recompute (the record was edited, or not written by the reader)")
        ins = rec.get("inputs_sha256")
        if not isinstance(ins, dict):
            why.append("inputs_sha256 is missing or not an object")
            ins = {}
        for name in SO.REQUIRED_INPUTS:
            if name not in ins:
                why.append(f"lists no input {name!r}")
        for name, want in sorted(ins.items()):
            fp = os.path.join(d, *str(name).split("/"))
            if not os.path.isfile(fp):
                why.append(f"input {name!r} is missing")
            elif SO.sha256_file(fp) != want:
                why.append(f"input {name!r} changed since the read (sha256 differs)")
        return "; ".join(why)
    except Exception as e:                                                                                                   # noqa: BLE001
        return f"cannot verify the record ({type(e).__name__}: {str(e)[:120]})"


def safeobj_gate(dirs):
    """SAFEOBJ (PREREG_SAFEOBJ.md section 4 / outcome table): R1o runs only if EVERY constrained arm read bar S = PASS.  Each <dir>/safeobj_S.json must exist, parse, carry "S": "PASS", AND be a
    genuine, unmodified output of eval/safeobj_read.py (safeobj_not_genuine: its tool, its record_sha256, its inputs' sha256s) -- a gate input is never hand-written.  -> (ok, why); a missing,
    unparsable, non-PASS or non-genuine file is a refusal, and the reason is named."""
    bad = []
    for d in dirs:
        p = os.path.join(d, "safeobj_S.json")
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            s = rec.get("S")
        except (OSError, ValueError, AttributeError) as e:
            bad.append(f"{p}: {type(e).__name__}")
            continue
        if s != "PASS":
            bad.append(f"{p}: S={s!r}")
        else:
            why = safeobj_not_genuine(d, rec)
            if why:
                bad.append(f"{p}: not a genuine reader record: {why}")
    return (not bad), (f"S = PASS on {len(dirs)} constrained arm(s), each a genuine reader record with its inputs unchanged" if not bad else "; ".join(bad))
'''
GATE_EDITS = [(OLD_GATE, NEW_GATE)]

EDITS = {"eval/safeobj_read.py": READER_EDITS, "refe/rls1_checks.py": GATE_EDITS}

try:                                                    # the two test edit lists live in their own files (long); a missing one is a hard error once they exist
    from patch_genuine_tests import CH_TEST_EDITS, QUEUE_EDITS     # noqa: E402
    EDITS["refe/test_rl_safeobj.py"] = CH_TEST_EDITS
    EDITS["refe/test_rls1_queue.py"] = QUEUE_EDITS
except ImportError as e:                                # noqa: BLE001
    if os.environ.get("PATCH_GENUINE_PARTIAL") != "1":
        raise SystemExit(f"patch_genuine_tests.py is missing ({e}); set PATCH_GENUINE_PARTIAL=1 only while developing")


def main():
    pk = PL.pk_arg()
    for rel, edits in EDITS.items():
        r = PL.patch_file(os.path.join(pk, rel), edits, expect_before=BASES[rel], label=rel)
        print({k: r[k] for k in ("file", "blob_before", "blob_after", "edits")})
    dst = os.path.join(pk, "eval", "test_safeobj_gate.py")
    if os.path.exists(dst):
        raise SystemExit("eval/test_safeobj_gate.py already exists: refusing to overwrite (is the tree the round-2 tree?)")
    text = PL.read(os.path.join(HERE, "test_safeobj_gate.py"))
    PL.write(dst, text)
    print({"wrote": "eval/test_safeobj_gate.py", "blob": PL.git_blob(text), "bytes": len(text.encode("utf-8"))})


if __name__ == "__main__":
    main()
