from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/a3_bars.py"); s = p.read_text(encoding="utf-8")
a1 = 'TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"'
b1 = 'TIER = "T1 self-action OPEN LOOP on logged eval139 frames, not driving performance"'
i0 = s.index("EST = ("); i1 = s.index("\n\n", i0)
b2 = ('EST = ("paired episode-cluster bootstrap over EPISODES, B 2000 seed 0 \u2014 blind to training and inference variance; fit seeds 0/1/2 = "\n'
      '       "training-variance read. (pair_read_r8.paired_cell -> taniteval.ci, models fixed; inference seeds 0 / 1 reported separately, never pooled; "\n'
      '       "ONE refcv8 trunk seed)")')
assert s.count(a1) == 1
s = s.replace(a1, b1)
s = s[:i0 - (len(a1) - len(b1))] + b2 + s[i1 - (len(a1) - len(b1)):] if False else s
i0 = s.index("EST = ("); i1 = s.index("\n\n", i0)
s = s[:i0] + b2 + s[i1:]
p.write_text(s, encoding="utf-8"); print("ok")
