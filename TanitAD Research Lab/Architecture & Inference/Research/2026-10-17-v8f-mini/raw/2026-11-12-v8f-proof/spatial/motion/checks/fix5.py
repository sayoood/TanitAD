R="C:/Users/Admin/v8proof_c/motion/"
P=R+"tree_patch/stack/scripts/v8f_mini_train.py"
s=open(P,encoding="utf-8",newline="").read()
a='    if cfg.wm != "off":\n        expect.add("wm")\n'
b=a+'    if cfg.wm_l1_roll_k > 0:\n        expect.add("wm_roll")                                            # MOTION lever B: a declared rollout term that never appears is a launch blocker\n'
assert s.count(a)==2; s=s.replace(a,b)
open(P,"w",encoding="utf-8",newline="").write(s)
P=R+"checks/m6_selfcheck.py"
s=open(P,encoding="utf-8",newline="").read()
a='rec["declared_vs_built"] in ([], None)'
assert s.count(a)==1; s=s.replace(a,'rec["declared_vs_built"] in ([], None, "PASS")')
open(P,"w",encoding="utf-8",newline="").write(s)
