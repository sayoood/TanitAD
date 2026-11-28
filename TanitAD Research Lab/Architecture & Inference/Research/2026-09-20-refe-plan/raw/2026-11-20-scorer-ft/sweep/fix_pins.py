"""MM 2026-11-20: SCORER-FT adds 4 profiles (33 -> 37). Seven older gate tests pin the profile set / count.
Add the four names to each pinned set and move 33 -> 37. Each edit must hit exactly once."""
import sys, pathlib
R = pathlib.Path(sys.argv[1])
N4 = ', "refe_scorerft_t", "refe_scorerft_c", "refe_scorerft_t_s1", "refe_scorerft_c_s1"'
TAG = "  # + the 4 SCORER-FT profiles (PREREG_SCORER_FT, PI 2026-11-20)"
EDITS = {
    "test_div_gate.py": [('"refe_field_p7_s1"} and len(L.PROFILES) == 33', '"refe_field_p7_s1"' + N4 + '} and len(L.PROFILES) == 37')],
    "test_dose_gate.py": [('"refe_field_p7_s1"} and len(L.PROFILES) == 33', '"refe_field_p7_s1"' + N4 + '} and len(L.PROFILES) == 37')],
    "test_vdac_gate.py": [('"refe_field_p7_s1"} and len(L.PROFILES) == 33', '"refe_field_p7_s1"' + N4 + '} and len(L.PROFILES) == 37')],
    "test_field_p7_profile.py": [('L.FIELD_P7_REP_BASE == {"Cs": "C", "Fs": "F"} and len(L.PROFILES) == 33)', 'L.FIELD_P7_REP_BASE == {"Cs": "C", "Fs": "F"} and len(L.PROFILES) == 37)')],
    "test_rep_gate.py": [('assert len(L.PROFILES) == 33 and', 'assert len(L.PROFILES) == 37 and'),
                         ('"refe_div_p6", "refe_vdac_p6"}) == 19, len(L.PROFILES)', '"refe_div_p6", "refe_vdac_p6"' + N4 + '}) == 19, len(L.PROFILES)')],
    "test_s1zsm_gate.py": [('"refe_field_p7_s1"}, set(L.PROFILES) - set(TIP_FP)', '"refe_field_p7_s1"' + N4 + '}, set(L.PROFILES) - set(TIP_FP)')],
    "test_scorer1p6_gate.py": [('"refe_field_p7_s1"}, set(L.PROFILES) - set(PINNED_FP)', '"refe_field_p7_s1"' + N4 + '}, set(L.PROFILES) - set(PINNED_FP)')],
}
for fn, eds in EDITS.items():
    p = R / fn
    b = p.read_text(encoding="utf-8")
    assert "\r" not in b, fn
    for old, new in eds:
        assert b.count(old) == 1, (fn, old, b.count(old))
        b = b.replace(old, new)
    lines = b.split("\n")
    # append the provenance tag to each edited line exactly once
    for i, ln in enumerate(lines):
        if ("refe_scorerft_t_s1" in ln or "len(L.PROFILES) == 37" in ln) and TAG not in ln:
            lines[i] = ln + TAG
    p.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("ZZPIN_OK", fn)
