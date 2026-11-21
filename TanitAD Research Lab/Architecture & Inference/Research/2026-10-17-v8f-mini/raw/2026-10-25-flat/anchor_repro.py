"""Reproduce, in memory, the collection-time error that kills mutants #0-5,#7: apply each mutant's edit to the UNMUTATED model source (from T) and run the test file's own _anchors() on it."""
import ast, sys, tempfile, types
from pathlib import Path
T = Path(sys.argv[1])
runner = (T / "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-flat/mutate_flat.py").read_text(encoding="utf-8")
src = runner[: runner.index('BASE = Path("C:/Users/Admin/pt_flat_mut")')]
sys.argv = ["x", str(T), sys.executable]
g = {"__name__": "mut"}
exec(compile(src, "mutate_flat_head", "exec"), g)
MUT = g["MUT"]
tsrc = (T / "stack/tests/test_v8f_flat.py").read_text(encoding="utf-8")
tree = ast.parse(tsrc)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_anchors")
model = T / "stack/tanitad/models/v8f_mini.py"
orig = model.read_text(encoding="utf-8")
_TMP = tempfile.TemporaryDirectory()


def run(text):
    tmp = Path(_TMP.name) / "v8f_mini_mut.py"
    tmp.write_text(text, encoding="utf-8", newline="")
    ns = {"MODEL_P": tmp}
    exec(compile(ast.Module([fn], []), "anchors", "exec"), ns)
    try:
        ns["_anchors"]()
        return "anchors OK (collection would pass)"
    except AssertionError as e:
        return "AssertionError at collection: " + (str(e)[:110] or "(no message)")
print("unmutated:", run(orig))
for i in (0, 1, 2, 3, 4, 5, 6, 7, 8):
    path, name, edits, grp = MUT[i]
    assert path == model, path
    t = orig
    for a, b in edits:
        assert t.count(a) == 1
        t = t.replace(a, b, 1)
    print(i, name, "->", run(t))
