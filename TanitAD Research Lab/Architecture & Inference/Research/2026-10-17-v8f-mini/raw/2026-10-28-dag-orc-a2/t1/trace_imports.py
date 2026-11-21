#!/usr/bin/env python3
"""trace_imports.py OUT.json TOOL.py [tool args...] -- run TOOL.py exactly as `python TOOL.py args` would (runpy, __main__), and at exit write the sorted list of every module file the process loaded to OUT.json.
Used by dagorct1_chain.sh --smoke to record the EXACT runtime import set of the T1 harness and of the fresh-age T0 eval (dagorct1_lib.py rolloutcode --loaded ...).  A diagnostic wrapper only: the real chain never uses it."""
import atexit
import json
import os
import runpy
import sys

out, tool = sys.argv[1], sys.argv[2]
sys.argv = [tool] + sys.argv[3:]
sys.path.insert(0, os.path.dirname(os.path.abspath(tool)))      # what `python TOOL.py` does


def dump():
    files = {os.path.abspath(m.__file__).replace("\\", "/") for m in list(sys.modules.values()) if getattr(m, "__file__", None)}
    files.add(os.path.abspath(tool).replace("\\", "/"))
    # taniteval/tools/t1_eval.py is loaded by PATH (importlib.util.spec_from_file_location, never registered in sys.modules): add it for the T1 harness, found from the tool's own tree root (<tree>/stack/scripts/<tool>)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(tool))))
    c = os.path.join(root, "taniteval", "tools", "t1_eval.py").replace("\\", "/")
    if os.path.isfile(c) and os.path.basename(tool).startswith("v8f_t1"):
        files.add(c)
    json.dump(sorted(files), open(out, "w", encoding="utf-8"), indent=0)


atexit.register(dump)
runpy.run_path(tool, run_name="__main__")
