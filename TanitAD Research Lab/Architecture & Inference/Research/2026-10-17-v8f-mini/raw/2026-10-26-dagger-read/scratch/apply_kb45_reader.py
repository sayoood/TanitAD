import pathlib
def edit(path, pairs):
    P = pathlib.Path(path)
    s = P.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:80])
        s = s.replace(old, new)
    P.write_text(s, encoding="utf-8", newline="\n")

edit("C:/Users/Admin/dagread/bank_audit.py", [
    ('''f"K-B3.2:s{a.seed}": "needs the trainer's summary.json (TRAIN window-list sha256; the tip's trainer does not record one)",''',
     '''f"K-B3.2:s{a.seed}": "needs the trainer's summary.json on DAG and DAG-0 (`train_window_list_sha256`, recorded on every run by v8f_mini_train.py)",'''),
    ('''does not write today. Exit code''', '''does not write (an older generator's bank lacks `controls` K-B4 / K-B5). Exit code'''),
])
edit("C:/Users/Admin/dagread/dag_read.py", [
    ('''help="PROVISIONAL: the trainer's TRAIN WINDOW-list sha256 (G.windows_sha256(J['clip'], wins)); the tip's trainer does not record one yet")''',
     '''help="the trainer's TRAIN WINDOW-list sha256 (G.windows_sha256(J['clip'], wins)); v8f_mini_train.py records it on every run, in summary.json beside train_sha256 (:1663)")'''),
])
