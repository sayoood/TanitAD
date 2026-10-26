import sys, numpy as np
sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_mapsel/code')
import mapsel_data as D
idx=D.decisions_index(0); keys=sorted(idx); print(len(keys))
f,chk=D.lc_windows(keys); print(chk, 'LC windows', int(f.sum()), 'of', len(keys))
