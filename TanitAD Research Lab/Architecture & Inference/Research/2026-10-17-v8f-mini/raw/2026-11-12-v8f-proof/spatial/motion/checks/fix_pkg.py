import re
s=open("C:/Users/Admin/v8proof_c/motion/checks/package.py",encoding="utf-8").read()
s=s.replace('S = "F:/TanitAD-scratch/v8proof/spatial/cellpred"','S = "C:/Users/Admin/v8proof_c/motion"\nDEST = "F:/TanitAD-scratch/v8proof/spatial/motion"')
open("C:/Users/Admin/v8proof_c/motion/checks/package_src.py","w",encoding="utf-8",newline="\n").write(s)
