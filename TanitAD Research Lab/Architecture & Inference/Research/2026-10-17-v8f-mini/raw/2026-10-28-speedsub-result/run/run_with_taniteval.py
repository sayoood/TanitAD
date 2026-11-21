import os, runpy, sys
tree, script = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(tree, "taniteval"))
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
