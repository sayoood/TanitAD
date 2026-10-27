"""Widen test_v8f_kind.py's, test_v8f_mixo.py's and test_v8f_lora_ar2.py's landing-moment label-set tests by the labels DAGGER (b43c7a9)
and DAGGER2 (1d4f4fb) appended, exactly as ORC's edit widened them (their ORC_LABELS / `orc` convention). Nothing else changes.
python patch_kind_mixo_lora_later.py <tests dir>"""
import sys

D = sys.argv[1].rstrip("/") + "/"
SET = ('{"dagname", "prepdag", "dagarm", "dagarm_dry", "dagarm_have", "bankdag", "gatedag", "traindag", "evaldag",'
       ' "dag2arm", "dag2arm_dry", "dag2arm_have", "bankdag2"}')
CMT = "# PREREG_V8F_DAGGER / PREREG_V8F_DAGGER2 appended these after ORC's (widened here, nothing else)"


def patch(fn, anchor, decl, old, new):
    p = D + fn
    s = open(p, "rb").read().decode("utf-8")
    assert "\r" not in s and s.count(anchor) == 1 and s.count(old) == 1 and "DAG_LABELS" not in s and " dag = {" not in s, fn
    i = s.index(anchor)
    j = s.index("\n", i) + 1
    s = s[:j] + decl + "\n" + s[j:]
    s = s.replace(old, new)
    open(p, "wb").write(s.encode("utf-8"))
    print("patched", fn)


for fn in ("test_v8f_kind.py", "test_v8f_mixo.py"):
    patch(fn, 'ORC_LABELS = {"orcarm", "gateorc", "trainorc", "evalorc"}', f"DAG_LABELS = {SET}      {CMT}",
          "- ORC_LABELS ==", "- ORC_LABELS - DAG_LABELS ==")
patch("test_v8f_lora_ar2.py", '    orc = {"orcarm", "gateorc", "trainorc", "evalorc"}', f"    dag = {SET}      {CMT}",
      "- mixo - flat - oacc - orc ==", "- mixo - flat - oacc - orc - dag ==")
