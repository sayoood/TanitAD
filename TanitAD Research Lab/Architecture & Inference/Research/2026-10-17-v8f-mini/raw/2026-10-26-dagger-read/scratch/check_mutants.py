import importlib.util, sys, pathlib
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("mutate_dag", "C:/Users/Admin/dagread/mutate_dag.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
src = pathlib.Path("C:/Users/Admin/dagread/dag_read.py").read_text(encoding="utf-8")
ids = [x[0] for x in m.MUTANTS]
assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
bad = 0
for mid, what, edit, k in m.MUTANTS:
    try:
        mut = edit(src)
        assert mut != src
        compile(mut, mid, "exec")
    except Exception as e:
        bad += 1
        print("ANCHOR/COMPILE PROBLEM", mid, type(e).__name__, str(e)[:160])
print(len(ids), "mutants,", bad, "problems")
