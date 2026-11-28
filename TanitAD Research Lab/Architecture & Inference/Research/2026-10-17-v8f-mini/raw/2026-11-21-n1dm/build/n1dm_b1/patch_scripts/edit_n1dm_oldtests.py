"""edit_n1dm_oldtests.py -- the census / literal-list tests that pin the strat1_check KINDS tuple, DCHAN_KINDS and the AUX_ROUTE_MODES tuple move with the new kind. Anchored, counted."""
import sys
TREE = sys.argv[1]


def sub(rel, old, new, count=1):
    p = f"{TREE}/stack/tests/{rel}"
    s = open(p, encoding="utf-8", newline="").read()
    n = s.count(old)
    assert n == count, (rel, n, count, old[:100])
    open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))


sub("test_strat1_n1dc_check.py", 'OLD_KINDS + ("N1DC", "N1DH") and', 'OLD_KINDS + ("N1DC", "N1DH", "N1DM") and')
sub("test_strat1_n1dc_check.py", "REAL.C_KINDS + REAL.H_KINDS and", "REAL.C_KINDS + REAL.H_KINDS + REAL.M_KINDS and")
sub("test_strat1_n1dw_check.py", '("N1DW4", "N1DW2", "N1DC", "N1DH") and', '("N1DW4", "N1DW2", "N1DC", "N1DH", "N1DM") and')
sub("test_strat1_n1dw_check.py", "REAL.C_KINDS + REAL.H_KINDS and", "REAL.C_KINDS + REAL.H_KINDS + REAL.M_KINDS and")
sub("test_strat1_navrb2_check.py", '"N1DW2", "N1DC", "N1DH")      #', '"N1DW2", "N1DC", "N1DH", "N1DM")      #')
sub("test_strat1_navrb2_check.py", '"N1DC": "N1", "N1DH": "N1"} and', '"N1DC": "N1", "N1DH": "N1", "N1DM": "N1"} and')
sub("test_strat1_navrbd_check.py", '"N1DW2", "N1DC", "N1DH") and', '"N1DW2", "N1DC", "N1DH", "N1DM") and')
sub("test_strat1_s1d.py", '"N1DC", "N1DH")', '"N1DC", "N1DH", "N1DM")', 2)
sub("test_strat1_n1dh_check.py", 'OLD_KINDS + ("N1DH",) and', 'OLD_KINDS + ("N1DH", "N1DM") and')
sub("test_strat1_n1dh_check.py", "REAL.C_KINDS + REAL.H_KINDS and", "REAL.C_KINDS + REAL.H_KINDS + REAL.M_KINDS and")
sub("test_v8f_n1dh.py", 'M.AUX_ROUTE_MODES == ("off", "cond")', 'M.AUX_ROUTE_MODES == ("off", "cond", "modes")')
print("old tests ok")
