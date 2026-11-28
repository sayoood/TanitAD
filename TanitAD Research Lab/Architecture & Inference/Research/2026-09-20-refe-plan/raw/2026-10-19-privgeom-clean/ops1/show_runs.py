import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../code")
import test_ops1 as T
for tag, path, plan, kw in (("OLD bytes, clean completion", T.ORIG, T.CLEAN2, {}), ("FIX, clean completion", T.NEW, T.CLEAN2, {}), ("FIX, relaunch 4/4 extracted", T.NEW, T.CLEAN2, dict(prefill=4, stale_done=5)),
                            ("FIX, rc0 + logs missing", T.NEW, {"modes": [], "default": "silent0"}, {})):
    r = T.run_chain(path, plan, tag="show", **kw)
    print("=====", tag); print(r["log"]); print("STOP:", r["stop"].strip()); T.cleanup(r)
