import sys
D = "C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/c3diag"
k = sys.argv[1]
a = open("C:/Users/Admin/t2v8f_pulls/ladder_R35_s%s/argv.txt" % k, encoding="utf-8").read().strip()
old = r"--out D:\tanitad\runs\v8f_ladder\R35_s%s" % k
assert a.count(old) == 1, "out token"
assert "--selfcheck-only" not in a and "--wm-idm-gamma" not in a and "--wm-sampler" not in a
a = a.replace(old, r"--out D:\tanitad\actsens\diag\REF_s%s\out" % k)
R = r"D:\tanitad\actsens\diag\REF_s%s" % k
lines = ["@echo off", r"call D:\tanitad\env.cmd", "if not exist %s mkdir %s" % (R, R),
         r"echo START %%DATE%% %%TIME%% > %s\started.txt" % R,
         r'"%%PY%%" D:\tanitad\actsens\repo\stack\scripts\v8f_mini_train.py %s --selfcheck-only > %s\ref.log 2>&1' % (a, R),
         r"echo rc=%%ERRORLEVEL%% %%DATE%% %%TIME%% > %s\finished.txt" % R, ""]
open(D + "/ref_s%s.cmd" % k, "w", encoding="ascii", newline="\r\n").write("\n".join(lines))
print(open(D + "/ref_s%s.cmd" % k, encoding="ascii").read())
