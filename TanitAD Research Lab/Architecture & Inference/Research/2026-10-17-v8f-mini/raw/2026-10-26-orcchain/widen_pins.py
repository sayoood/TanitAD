"""Widen the five older queue-structure tests for O-RECOVER's four new labels (the programme's own convention: 'widened here, nothing else'). Usage: widen_pins.py <stack/tests dir>. Every replacement must match exactly once."""
import sys
from pathlib import Path

d = Path(sys.argv[1])
ORC = '{"orcarm", "gateorc", "trainorc", "evalorc"}'
NL = chr(92) + "n"          # a backslash and an n: the two characters the older tests carry in their source text
edits = {
    "test_v8f_kind.py": [
        ('OLD_DISPATCH = (', 'ORC_LABELS = ' + ORC + '                                                   # PREREG_V8F_ORECOVER appended these (test_v8f_orc_chain.py pins them); widened here, nothing else\nOLD_DISPATCH = ('),
        ('assert set(blocks) - MIXO_LABELS - FLAT_LABELS - OACC_LABELS == set(PRE_EXISTING)', 'assert set(blocks) - MIXO_LABELS - FLAT_LABELS - OACC_LABELS - ORC_LABELS == set(PRE_EXISTING)'),
    ],
    "test_v8f_lora_ar2.py": [
        ('    assert set(blocks) - {"kind", "kindseed", "kindown", "kindorc"} - mixo - flat - oacc == ',
         '    orc = ' + ORC + '          # PREREG_V8F_ORECOVER appended these (test_v8f_orc_chain.py pins them); widened here, nothing else\n    assert set(blocks) - {"kind", "kindseed", "kindown", "kindorc"} - mixo - flat - oacc - orc == '),
    ],
    "test_v8f_mixo.py": [
        ('assert set(blocks) - FLAT_LABELS - OACC_LABELS == set(_K.PRE_EXISTING)', 'assert set(blocks) - FLAT_LABELS - OACC_LABELS - ORC_LABELS == set(_K.PRE_EXISTING)'),
        ('OACC_LABELS = {"waccoacc", "oaccwacc", "gateoacc", "trainoacc", "evaloacc", "fitoacc"}', 'ORC_LABELS = ' + ORC + '           # PREREG_V8F_ORECOVER appended these (test_v8f_orc_chain.py pins them); widened here, nothing else\nOACC_LABELS = {"waccoacc", "oaccwacc", "gateoacc", "trainoacc", "evaloacc", "fitoacc"}'),
    ],
    "test_v8f_flat.py": [
        ('assert set(blocks) - OACC_LABELS == set(_K.PRE_EXISTING)', 'assert set(blocks) - OACC_LABELS - ORC_LABELS == set(_K.PRE_EXISTING)'),
        ('OACC_LABELS = {"waccoacc", "oaccwacc", "gateoacc", "trainoacc", "evaloacc", "fitoacc"}', 'ORC_LABELS = ' + ORC + '           # PREREG_V8F_ORECOVER appended these (test_v8f_orc_chain.py pins them); widened here, nothing else\nOACC_LABELS = {"waccoacc", "oaccwacc", "gateoacc", "trainoacc", "evaloacc", "fitoacc"}'),
    ],
    "test_v8f_oacc.py": [
        ('OACC_LABELS = set(OSTAGES) | {"oaccwacc"}', 'OACC_LABELS = set(OSTAGES) | {"oaccwacc"}\nORC_LABELS = ' + ORC + '           # PREREG_V8F_ORECOVER appended these after OACC\'s (test_v8f_orc_chain.py pins them); widened here, nothing else'),
        ('assert set(blocks) == set(_K.PRE_EXISTING)', 'assert set(blocks) - ORC_LABELS == set(_K.PRE_EXISTING)'),
        ('    assert q.rstrip().endswith("echo [ok] HX weights M2b-oacc_s%~2' + NL + 'exit /b 0"), "the new stages are the tail of the file"',
         '    assert q.split("' + NL + ':orcarm' + NL + '")[0].rstrip().endswith("echo [ok] HX weights M2b-oacc_s%~2' + NL + 'exit /b 0"), "the new stages are the tail of the file (PREREG_V8F_ORECOVER\'s follow: widened)"'),
        ('    assert _queue_text().rstrip().endswith("echo [ok] HX weights M2b-oacc_s%~2' + NL + 'exit /b 0")\n',
         '    assert _queue_text().split("' + NL + ':orcarm' + NL + '")[0].rstrip().endswith("echo [ok] HX weights M2b-oacc_s%~2' + NL + 'exit /b 0")      # PREREG_V8F_ORECOVER\'s stages follow (widened)\n'),
    ],
}
for name, reps in edits.items():
    p = d / name
    with open(p, "rb") as fh:
        raw = fh.read()
    assert b"\r\n" not in raw, name
    s = raw.decode("utf-8")
    for old, new in reps:
        assert s.count(old) == 1, (name, old[:70], s.count(old))
        s = s.replace(old, new)
    with open(p, "wb") as fh:
        fh.write(s.encode("utf-8"))
    print("widened", name, len(reps))
