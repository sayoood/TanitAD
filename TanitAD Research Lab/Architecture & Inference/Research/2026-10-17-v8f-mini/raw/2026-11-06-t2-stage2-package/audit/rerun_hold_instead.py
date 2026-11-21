import sys
sys.path.insert(0, "tests")
import test_t2v8f_stage2 as T
print("baseline", T.check_base_equals_direct())
try:
    T.check_base_equals_direct("hold_instead")
    print("hold_instead: GREEN <-- not caught")
except AssertionError as e:
    print("hold_instead: RED", str(e)[:140])
