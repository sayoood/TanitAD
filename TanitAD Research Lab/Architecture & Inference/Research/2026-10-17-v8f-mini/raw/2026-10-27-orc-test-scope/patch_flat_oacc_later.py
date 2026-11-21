"""Widen test_v8f_flat.py's and test_v8f_oacc.py's landing-moment label-set tests by the labels DAGGER (b43c7a9) and DAGGER2 (1d4f4fb)
appended after them, exactly as ORC's edit widened them before (their ORC_LABELS / OACC_LABELS convention). Nothing else changes.
python patch_flat_oacc_later.py <tests dir>"""
import sys

D = sys.argv[1].rstrip("/") + "/"
DECL = ('DAG_LABELS = {"dagname", "prepdag", "dagarm", "dagarm_dry", "dagarm_have", "bankdag", "gatedag", "traindag", "evaldag",      # PREREG_V8F_DAGGER'
        ' appended these after ORC\'s; widen\n'
        '              "dag2arm", "dag2arm_dry", "dag2arm_have", "bankdag2"}                                                  # PREREG_V8F_DAGGER2,'
        ' after DAGGER\'s\n')


def patch(fn, anchor, old, new):
    p = D + fn
    s = open(p, "rb").read().decode("utf-8")
    assert "\r" not in s and s.count(anchor) == 1 and s.count(old) == 1 and "DAG_LABELS" not in s, fn
    i = s.index(anchor) + len(anchor)
    j = s.index("\n", i) + 1
    s = s[:j] + DECL + s[j:]
    s = s.replace(old, new)
    open(p, "wb").write(s.encode("utf-8"))
    print("patched", fn)


patch("test_v8f_flat.py", 'ORC_LABELS = {"orcarm", "gateorc", "trainorc", "evalorc"}',
      "    assert set(blocks) - OACC_LABELS - ORC_LABELS == set(_K.PRE_EXISTING)",
      "    assert set(blocks) - OACC_LABELS - ORC_LABELS - DAG_LABELS == set(_K.PRE_EXISTING)")
patch("test_v8f_oacc.py", 'ORC_LABELS = {"orcarm", "gateorc", "trainorc", "evalorc"}',
      "    assert set(blocks) - ORC_LABELS == set(_K.PRE_EXISTING)",
      "    assert set(blocks) - ORC_LABELS - DAG_LABELS == set(_K.PRE_EXISTING)")
