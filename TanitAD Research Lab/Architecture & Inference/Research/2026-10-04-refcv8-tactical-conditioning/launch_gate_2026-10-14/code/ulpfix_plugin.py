"""pytest plugin (-p ulpfix_plugin): re-compile AnchoredDiffusionDecoder._r8_split with the base/extras slices made
CONTIGUOUS, to test the fix before editing the repo. Prints a marker so the run proves the patch was applied."""
import inspect
import textwrap
from tanitad.refs import refc as RC

src = textwrap.dedent(inspect.getsource(RC.AnchoredDiffusionDecoder._r8_split))
old = "c, o = fn(kv, cond, x[:, sl], *rest)"
assert src.count(old) == 1, "anchor not found"
src = src.replace(old, "c, o = fn(kv, cond, x[:, sl].contiguous(), *rest)")
ns = {}
exec(compile(src, RC.__file__, "exec"), RC.__dict__, ns)
RC.AnchoredDiffusionDecoder._r8_split = ns["_r8_split"]
print("ZZULPFIX-APPLIEDZZ")
