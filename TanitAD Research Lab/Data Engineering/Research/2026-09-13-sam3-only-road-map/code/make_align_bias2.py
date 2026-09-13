import sys
from pathlib import Path
B = Path(__file__).resolve().parent
s = (B / "align_bias.py").read_text(encoding="utf-8")
old_ref = """            r = np.hypot(w[:, 0] - cw[0], w[:, 1] - cw[1]); ref.append(w[r <= 8.0])"""
new_ref = """            r = np.hypot(w[:, 0] - cw[0], w[:, 1] - cw[1]); ref.append(w[r <= 10.0])"""
assert s.count(old_ref) == 1
s = s.replace(old_ref, new_ref)
old_sel = """                v = w - cw; r = np.hypot(v[:, 0], v[:, 1])
                sel = r > 8.0"""
new_sel = """                v = w - cw; r = np.hypot(v[:, 0], v[:, 1])
                oj = F[j]["T_world_rig"][:2, 3]
                sel = (r > 8.0) & (np.hypot(w[:, 0] - oj[0], w[:, 1] - oj[1]) <= 7.0)   # true match must lie inside the reference disc"""
assert s.count(old_sel) == 1
s = s.replace(old_sel, new_sel)
s = s.replace("signed along-ray offset of far paint vs near paint", "MARGIN-CORRECTED signed along-ray offset (ref <= 10 m of rig j, test located <= 7 m of rig j)")
s = s.replace("ALL CAMERAS 8-16 m", "MARGIN ALL CAMERAS 8-16 m")
(B / "align_bias2.py").write_text(s, encoding="utf-8", newline="\n")
print("ok")
