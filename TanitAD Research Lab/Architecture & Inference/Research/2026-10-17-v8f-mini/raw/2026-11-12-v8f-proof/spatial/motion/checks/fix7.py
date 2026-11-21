P="C:/Users/Admin/v8proof_c/motion/tree_patch/stack/tanitad/models/v8f_mini.py"
s=open(P,encoding="utf-8",newline="").read()
a='            lv["wm_l1_residual"] = pp["residual"]                      # MOTION lever A: read from the BUILT predictor module (is it the residual wrapper?)\n'
b='            lv["wm_l1_residual"] = "on" if type(self.wm.stack.predictors[0]).__name__ == "ResidualPredictor" else "off"      # MOTION lever A: read from the BUILT predictor module (is it the residual wrapper?)\n'
assert s.count(a)==1; s=s.replace(a,b)
open(P,"w",encoding="utf-8",newline="").write(s)
