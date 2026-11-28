"""MM 2026-11-19: C's pod gate FAILED G-LIVE: "declared loss term 'field_L' ABSENT from the trainer's metrics on steps [0, 1]".
The trainer computes and logs field_L only when --w-field > 0 (train.py: `if a.w_field > 0.0`); C / Cs run at 0.0, so their
profiles must not declare it. Edits the hold's land copies in place (LF, utf-8)."""
H = 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_p7/det/land/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe'
def edit(fn, pairs):
    p = H + '/' + fn; s = open(p, 'rb').read().decode('utf-8'); assert '\r\n' not in s
    for old, new in pairs:
        assert s.count(old) == 1, (fn, old[:60], s.count(old)); s = s.replace(old, new)
    open(p, 'wb').write(s.encode('utf-8')); print('edited', fn)
edit('launch_gate.py', [(
    '"live_terms": P6_TERMS + ("field_L",), "field_min_coverage": 0.999,\n',
    '"live_terms": P6_TERMS + (("field_L",) if FIELD_P7_ARM_W[arm] > 0.0 else ()), "field_min_coverage": 0.999,\n'
    '                 # field_L is a declared live term ONLY where the arm trains it: train.py computes and logs it only when --w-field > 0, so a\n'
    '                 # control at 0.0 that declared it FAILED G-LIVE on the pod ("field_L ABSENT from the trainer\'s metrics", C, 2026-11-19 06:15Z).\n')])
edit('test_field_p7_profile.py', [
    ('        ok(pr["live_terms"] == L.P6_TERMS + ("field_L",) and pr["field_min_coverage"] == 0.999)\n',
     '        ok(pr["live_terms"] == L.P6_TERMS + (("field_L",) if tw == "F" else ()) and pr["field_min_coverage"] == 0.999,\n'
     '           (nm, "field_L is declared live exactly where it is trained (F, Fs); a control (C, Cs) that declares it fails G-LIVE (pod, 2026-11-19)"))\n'),
    ('    ("launch_gate.py", "\\"live_terms\\": P6_TERMS + (\\"field_L\\",)", "\\"live_terms\\": P6_TERMS", "registry"),                                           # field_L is not a declared live term\n',
     '    ("launch_gate.py", "\\"live_terms\\": P6_TERMS + ((\\"field_L\\",) if FIELD_P7_ARM_W[arm] > 0.0 else ())", "\\"live_terms\\": P6_TERMS", "registry"),     # F does not declare field_L\n'
     '    ("launch_gate.py", "((\\"field_L\\",) if FIELD_P7_ARM_W[arm] > 0.0 else ())", "(\\"field_L\\",)", "registry"),     # the control declares field_L again (the 2026-11-19 pod G-LIVE failure)\n'),
])
