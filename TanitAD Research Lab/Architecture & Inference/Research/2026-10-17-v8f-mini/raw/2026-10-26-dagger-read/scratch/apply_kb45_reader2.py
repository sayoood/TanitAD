import pathlib
P = pathlib.Path("C:/Users/Admin/dagread/dag_read.py")
s = P.read_text(encoding="utf-8")
def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)
sub('''    PROVISIONAL (not yet written by the trainer / generator; each is a CLI override or listed in NOTES.md under "needs ... to add"): the built-record key names dagger_p / dagger_bank_sha256,
    the trainer's TRAIN window-list sha key (train_window_list_sha256), and the generator's `controls` K-B4 / K-B5 records.
''', '''    WRITTEN since the K-B4/K-B5 build (2026-10-26; MEASURED on fixtures/dagger_smoke and fixtures/trainer_kb45): the generator's PROVENANCE.json `controls` K-B4 / K-B5 records and the trainer's
    TRAIN window-list sha key (train_window_list_sha256, summary.json beside train_sha256). STILL PROVISIONAL (read from the trainer's source, no run output exists; each is a CLI override): the built-record
    key names dagger_p / dagger_bank_sha256 (levers) and the totals dagger_n / dagger_eligible.
''')
sub('''# NOT written by the generator (the reader REFUSES until it is; see NOTES.md "needs the bank generator to add"): PROVENANCE.json `controls` {K-B4{PASS, cases{..}}, K-B5{PASS, max_pose_now_err_m,
# max_rel_transform_err_m}} (those two are unit tests of the build, not per-bank records), and, on the TRAINER side, a TRAIN window-list sha256 (windows_sha256(J["clip"], wins)).''',
'''# PROVENANCE.json `controls` {K-B4{PASS, cases{..}, ..}, K-B5{PASS, max_pose_now_err_m, max_rel_transform_err_m, tol_m, ..}} is written by the generator since the K-B4/K-B5 build (fixtures/dagger_smoke); a bank from
# the older generator lacks it and the reader REFUSES it ("needs the bank generator to add"). On the TRAINER side the TRAIN window-list sha256 (windows_sha256(J["clip"], wins)) is `train_window_list_sha256`.''')
P.write_text(s, encoding="utf-8", newline="\n")
