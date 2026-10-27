"""Scope test_v8f_orc_chain.py's two landing-moment queue tests to ORC's own region. They asserted that ORC's four labels were the LAST
edit of the queue ("the only new labels", "the tail of the file", "the whole new region"), so every later, insertion-only edit (DAGGER's
stages, b43c7a9; DAGGER2's, 1d4f4fb) turned them RED although no ORC line changed. Later edits are now DECLARED here by label and PREREG,
and every ORC invariant is still asserted on ORC's region.  python patch_orc_chain_test.py <test file>"""
import sys

p = sys.argv[1]
s = open(p, "rb").read().decode("utf-8")
assert "\r" not in s


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('ORC_STAGES = ("gateorc", "trainorc", "evalorc")\n',
    'ORC_STAGES = ("gateorc", "trainorc", "evalorc")\n'
    '# Labels appended AFTER the ORC edit by later, insertion-only edits, each declared with its PREREG. They are tested by their own builds\n'
    '# (raw/2026-10-26-dagger-queue/, raw/2026-10-27-dagger2-queue/); here they only have to sit after the ORC region and leave it untouched.\n'
    '# A label that is in none of PRE_LABELS, ORC_LABELS and LATER_LABELS still turns the test below RED: a new edit declares itself here.\n'
    'LATER_LABELS = {\n'
    '    "dagname": "PREREG_V8F_DAGGER", "prepdag": "PREREG_V8F_DAGGER", "dagarm": "PREREG_V8F_DAGGER", "dagarm_dry": "PREREG_V8F_DAGGER",\n'
    '    "dagarm_have": "PREREG_V8F_DAGGER", "bankdag": "PREREG_V8F_DAGGER", "gatedag": "PREREG_V8F_DAGGER", "traindag": "PREREG_V8F_DAGGER",\n'
    '    "evaldag": "PREREG_V8F_DAGGER",\n'
    '    "dag2arm": "PREREG_V8F_DAGGER2", "dag2arm_dry": "PREREG_V8F_DAGGER2", "dag2arm_have": "PREREG_V8F_DAGGER2", "bankdag2": "PREREG_V8F_DAGGER2",\n'
    '}\n'
    'LATER_FIRST = "dagname"                                                       # the first label after the ORC region\n'
    'LATER_HEADER = tuple(f"rem   v8f_mini_box2_queue.cmd {st} " for st in ("bankdag", "gatedag", "traindag", "evaldag", "bankdag2")) \\\n'
    '    + ("rem   gatedag / traindag / evaldag also take ARM dag2 ",)           # the later edits\' usage lines in the header\n')

rep('def _block(label):\n',
    'def _orc_region():\n'
    '    """ORC\'s own region: from :orcarm up to the first label a later edit appended (LATER_FIRST)."""\n'
    '    return _queue_text().split("\\n:orcarm\\n", 1)[1].split(f"\\n:{LATER_FIRST}\\n", 1)[0]\n'
    '\n'
    '\n'
    'def _block(label):\n')

rep('    assert set(blocks) == set(PRE_LABELS) | ORC_LABELS and len(ORC_LABELS) == 4, "the only new labels are the four of this edit"\n',
    '    assert len(ORC_LABELS) == 4 and ORC_LABELS <= set(blocks), "the four labels of this edit are present"\n'
    '    assert set(blocks) == set(PRE_LABELS) | ORC_LABELS | set(LATER_LABELS), \\\n'
    '        f"a label that is neither older, ORC\'s nor a declared later edit: {sorted(set(blocks) - set(PRE_LABELS) - ORC_LABELS - set(LATER_LABELS))}"\n')

rep('    rest = [x for x in rest if not (x.startswith(\'if /i "%~1"==\') and any(s in x for s in ORC_STAGES))]\n',
    '    rest = [x for x in rest if not (x.startswith(\'if /i "%~1"==\') and any(s in x for s in ORC_STAGES))]\n'
    '    # the later edits\' own preamble lines: their usage lines (DAGGER also added one "rem" separator after them) and their dispatch lines\n'
    '    k = next((n for n, x in enumerate(rest) if x.startswith(LATER_HEADER)), None)\n'
    '    if k is not None:\n'
    '        m = k\n'
    '        while m < len(rest) and rest[m].startswith(LATER_HEADER):\n'
    '            m += 1\n'
    '        assert rest[k - 1] == "rem" and rest[m] == "rem", "the later usage lines sit as one block between two separators"\n'
    '        rest = rest[:k] + rest[m + 1:]\n'
    '    later_disp = {f\'if /i "%~1"=="{st}" goto :{st}\' for st in LATER_LABELS}\n'
    '    rest = [x for x in rest if x not in later_disp]\n'
    '    assert not any(x.startswith(LATER_HEADER) for x in rest), "a later usage line outside the block"\n')

rep('        "the new stages start right after the last old label, which keeps its text"\n'
    '    assert q.rstrip().endswith("echo [ok] eval %TG%\\nexit /b 0"), "the new stages are the tail of the file"\n',
    '        "the new stages start right after the last old label, which keeps its text"\n'
    '    assert ("\\n:orcarm\\n" + _orc_region()).rstrip().endswith("echo [ok] eval %TG%\\nexit /b 0"), "ORC\'s region ends with evalorc"\n'
    '    tail = re.findall(r"(?m)^:([A-Za-z0-9_]+)$", q.split("\\n:evalorc\\n", 1)[1])\n'
    '    assert tail and tail[0] == LATER_FIRST and set(tail) == set(LATER_LABELS), "every label after ORC\'s region is a declared later edit"\n')

rep('    assert all(sum(x.count(fl) for x in _code(_queue_text().split("\\n:orcarm\\n")[1].split("\\n"))) == 1 for fl in LAW), '
    '"each law flag is written once in the whole new region, inside ORC\'s ORCL"\n',
    '    assert all(sum(x.count(fl) for x in _code(_orc_region().split("\\n"))) == 1 for fl in LAW), "each law flag is written once in ORC\'s region, inside ORC\'s ORCL"\n')
rep('    assert sum(x.count("--ego-perturb-p") for x in _code(_queue_text().split("\\n:orcarm\\n")[1].split("\\n"))) == 1, '
    '"--ego-perturb-p appears once in the CODE of the whole new region"\n',
    '    assert sum(x.count("--ego-perturb-p") for x in _code(_orc_region().split("\\n"))) == 1, "--ego-perturb-p appears once in the CODE of ORC\'s region"\n')
open(p, "wb").write(s.encode("utf-8"))
print("patched", p)
