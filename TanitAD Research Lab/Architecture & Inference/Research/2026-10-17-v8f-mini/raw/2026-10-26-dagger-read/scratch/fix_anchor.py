import pathlib
P = pathlib.Path("C:/Users/Admin/dagread/test_dag_read.py")
s = P.read_text(encoding="utf-8")
tail = '''
    assert not dag.g_bank_config(B, ev)[0] and dag.g_bank_config({**B, "man": {**B["man"], "common": {**B["man"]["common"], "partial": False, "limit_clips": None,
                                                                                                  "train": {**B["man"]["common"]["train"], "n_clips": 4512, "n_windows": 306566}}}}, ev)[0]
'''
assert s.count(tail) == 1
s = s.replace(tail, "")
anchor = '''(the PREREG's sha is the FILE's, not the constants dict's\n'''
anchor = '''   # the PREREG's sha is the FILE's, not the constants dict's\n'''
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + tail.lstrip("\n"))
P.write_text(s, encoding="utf-8", newline="\n")
