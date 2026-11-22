"""make_rl_gate_record.py SRC OUT -- MM 2026-11-22 (RECLAB). Writes rl_gate_record.py from the pod's rh_gate_record.py (sha256 847019579b8ae271...).

Why: rl_arm.sh called rh_gate_record.py, whose branch test is tag.startswith("RH"). "RL_s0" fails that test, so the record ran the B0 branch
("B0 --dagger-p is not 0", "B0 argv names --dagger-bank") and every RL mint would have been REFUSED. The copy serves RL arms only:
  * COMMIT = cf27dbb7 (the RL code of record), G-SUITE result = gsuite_result_rl.json, tree marker = done_rl/tree.done (the RL chain's own tree step);
  * RL arms take the RH branch (bank, bank check, declaration flags, dose > 0, dagger probe), PLUS: the argv names --dagger-label speed_v1,
    the built levers carry dagger_label speed_v1, and the selfcheck's dagger probe read rl_label_rows_changed > 0 (the card's lever-acts check);
  * any non-RL tag is refused.
Every edit is an exact anchor replace that must match exactly once, or this script fails.
"""
import hashlib
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
raw = open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest().startswith("847019579b8ae271"), "unexpected rh_gate_record.py"
s = raw.decode("utf-8")

EDITS = [
    ('"""rh_gate_record.py -- the ROLLHIST chain\'s own launch-gate record for ONE stage-B arm (RH_sk or B0_sk). stdlib only, CPU, sub-second.',
     '"""rl_gate_record.py -- the RL chain\'s launch-gate record for ONE RL stage-B arm (RL_s0 | RL_s1). A COPY of rh_gate_record.py (sha256 847019579b8ae271...), made by make_rl_gate_record.py.\n'
     'Differences, and only these: COMMIT is cf27dbb7; the G-SUITE result is gsuite_result_rl.json; the tree marker is done_rl/tree.done; RL arms take the RH branch plus three label checks\n'
     '(argv names --dagger-label speed_v1; the built levers carry dagger_label speed_v1; the dagger probe read rl_label_rows_changed > 0); any non-RL tag is refused.\n'
     'Original docstring follows (RH_sk or B0_sk). stdlib only, CPU, sub-second.'),
    ('COMMIT = "97fe0a56767320dbc5f4b823d67fb07a3209a745"', 'COMMIT = "cf27dbb79f2491a6db1197860bc50ca7c8cdeff4"'),
    ('"gsuite": f"{R}/gsuite_result_rh.json"', '"gsuite": f"{R}/gsuite_result_rl.json"'),
    ('"tree_done": f"{R}/done/tree.done"', '"tree_done": f"{R}/done_rl/tree.done"'),
    ('    if tag.startswith("RH"):\n        f["bank_json"]', '    if tag.startswith("RL"):\n        f["bank_json"]'),
    ('    tag, ro = a.tag, a.readout\n    F = files(tag)\n    bad, bind = [], {}\n',
     '    tag, ro = a.tag, a.readout\n    if tag not in ("RL_s0", "RL_s1"):\n        print("PROBLEM rl_gate_record.py serves RL_s0 / RL_s1 only")\n        print("ZZ RH-GATE-RECORD REFUSED", tag)\n        return 1\n    F = files(tag)\n    bad, bind = [], {}\n'),
    ('    if tag.startswith("RH"):\n        if m.get("--dagger-bank-readout")', '    if tag.startswith("RL"):\n        if m.get("--dagger-label") != "speed_v1":\n'
     '            bad.append(f"RL argv --dagger-label is {m.get(\'--dagger-label\')!r}, not speed_v1")\n'
     '        if "speed_v1" not in set(map(str, find_key(sc.get("levers", {}), "dagger_label"))):\n'
     '            bad.append("the built levers do not carry dagger_label speed_v1")\n'
     '        rows = [v for v in find_key(sc.get("dagger_probe") or {}, "rl_label_rows_changed") if isinstance(v, (int, float))]\n'
     '        if not rows or max(rows) <= 0:\n'
     '            bad.append(f"the dagger probe read rl_label_rows_changed {rows!r}: the speed-consistent label did not reach the batch")\n'
     '        if m.get("--dagger-bank-readout")'),
    ('"schema": "rh-chain-gate-record-v1"', '"schema": "rl-chain-gate-record-v1"'),
]
for old, new in EDITS:
    n = s.count(old)
    assert n == 1, f"anchor matched {n} times: {old[:70]!r}"
    s = s.replace(old, new)
assert '"97fe0a56' not in s and "gsuite_result_rh" not in s and 'startswith("RH")' not in s, "an old pin is left"
open(OUT, "w", encoding="utf-8", newline="\n").write(s)
print("wrote", OUT, hashlib.sha256(s.encode()).hexdigest()[:16])
