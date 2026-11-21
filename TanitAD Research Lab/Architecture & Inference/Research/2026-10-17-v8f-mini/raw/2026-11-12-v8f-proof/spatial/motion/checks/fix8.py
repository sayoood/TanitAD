P="C:/Users/Admin/v8proof_c/motion/tree_patch/stack/tanitad/models/v8f_mini.py"
s=open(P,encoding="utf-8",newline="").read()
a='"on" if type(self.wm.stack.predictors[0]).__name__ == "ResidualPredictor" else "off"'
b='"on" if isinstance(self.wm.stack.predictors[0], _hj.ResidualPredictor) else "off"'
assert s.count(a)==1; s=s.replace(a,b)
a='            pp = self.wm.probe_pred()  '
assert s.count(a)==1; s=s.replace(a,'            from tanitad.models import hjepa as _hj\n'+a)
open(P,"w",encoding="utf-8",newline="").write(s)
