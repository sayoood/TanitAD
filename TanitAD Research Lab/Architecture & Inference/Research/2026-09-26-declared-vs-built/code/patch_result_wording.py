import sys
P = sys.argv[1]
s = open(P, encoding="utf-8").read()
reps = [
 ("| id | what argv / config.json said | what was built | mechanism (tip `5de9363` lines) | class |",
  "| id | what argv / config.json said | what was built | mechanism (line numbers identical at `59f0d46`, `5de9363`, `da8400b`) | class |"),
 ("open. And the two existing guards could not see it: the argv guards in `_pin_trainer_cfg` run BEFORE the loss, and",
  "open. And the two existing guards could not see it: the argv guards in `_pin_trainer_cfg` read argv, so they cannot see a value dropped after them, and"),
 ("Of **90** config assignments in the four helpers, **2** targeted an undeclared field; both are declared now",
  "Of the **90** attribute assignments in the four helpers (86 on config objects, 4 on the argparse Namespace), **2** targeted an undeclared field; both are declared now"),
 ("(static census on the fixed tree: 0). The runtime recorder over ~60 % of the related suite",
  "(static census on the fixed tree: 0). The runtime recorder over the first ~60 % of the related suite (the run was stopped there to keep the box above its RAM floor)"),
]
for a, b in reps:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(P, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
