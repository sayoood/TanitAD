P="C:/Users/Admin/v8proof_c/motion/checks/m2345_checks.py"
s=open(P,encoding="utf-8",newline="").read()
a="    s0.cfg[0].roll_weight = 0.0\n    z, a = seq()\n"
b="    s0.cfg[0].roll_weight = 0.0\n    s1, s0 = s1.double(), s0.double()      # float64: the difference of two totals of size ~10 is then exact to ~1e-14\n    z, a = (t.double() for t in seq())\n"
assert s.count(a)==1; s=s.replace(a,b)
s=s.replace('real["abs_err"] < 1e-5','real["abs_err"] < 1e-9').replace('mutant["abs_err"] > 1e-3','mutant["abs_err"] > 1e-3')
open(P,"w",encoding="utf-8",newline="").write(s)
