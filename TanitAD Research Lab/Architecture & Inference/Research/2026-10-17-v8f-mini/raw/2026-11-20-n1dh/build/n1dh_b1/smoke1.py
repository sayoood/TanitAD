import sys, torch
sys.path.insert(0,"F:/TanitAD-scratch/n1dc/build/n1dh_b1/tree/stack/scripts")
import v8f_mini_train as TR
from tanitad.models import v8f_mini as M
p=TR.build_parser()
print(p.parse_args(["--out","x"]).aux_route)
a=p.parse_args(["--out","x","--aux-route","cond","--w-aux-route","0.5"]); print(TR.make_config(a).aux_route)
try: p.parse_args(["--aux-route","ce"])
except SystemExit as e: print("exit",e.code)
