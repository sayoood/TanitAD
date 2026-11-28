"""Build the two Fs overlays (MM 2026-11-23): 10848403's files + 95b3d174's field_reach fix.  Reads the mirror with git show; writes LF-only files to ship/fs/.  usage: build_overlays.py"""
import ast, re, subprocess, sys
M = 'C:/Users/Admin/tanitad-push'
PK = 'TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan'
OLD = '10848403249a0c3d9beabbba70ac56d86bac78e6'
FIX = '95b3d1746ff02011dd5c6cf887890d8fbb015af1'
OUT = 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/ship/fs/'


def show(c, f):
    r = subprocess.run(['git', '-C', M, 'show', f'{c}:{PK}/refe/{f}'], capture_output=True, check=True).stdout.decode('utf-8')
    assert '\r' not in r
    return r


def span(src, a, b_pat):          # text from "def a(" up to the first top-level non-blank line after judge_field_reach's body (a comment "# ----")
    i = src.index(f'\ndef {a}(') + 1
    j = src.index('\n# ---- ', src.index('\ndef judge_field_reach(', i))
    return i, j


go, gn = show(OLD, 'gate_checks.py'), show(FIX, 'gate_checks.py')
io, jo = span(go, 'field_reach', None)
inn, jn = span(gn, 'field_reach', None)
new = go[:io] + gn[inn:jn] + go[jo:]
ast.parse(new)
open(OUT + 'gate_checks.py', 'wb').write(new.encode('utf-8'))

to, ts = show(OLD, 'test_field_p7_profile.py'), show(FIX, 'test_scorerft_profile.py')
a = ts.index('def make_model(leaky=False):'); b = ts.index('def install_fake_rank_module')
mm = ts[a:b].rstrip() + '\n\n\n'
a = ts.index('def chk_field_reach_active_sample'); b = ts.index('\n\nCHECKS = {', a)
chk = ts[a:b].rstrip() + '\n\n\n'
assert to.count('import gate_token as GT  # noqa: E402\nimport launch_gate as L  # noqa: E402\n') == 1
t = to.replace('import gate_token as GT  # noqa: E402\nimport launch_gate as L  # noqa: E402\n', 'import gate_token as GT  # noqa: E402\nimport launch_gate as L  # noqa: E402\nimport gate_checks as GC  # noqa: E402\n')
t = t.replace('import tempfile\nfrom pathlib import Path\n', 'import tempfile\nimport types\nfrom pathlib import Path\nfrom types import SimpleNamespace\n', 1)
assert t.count('\nCHECKS = {') == 1
t = t.replace('\nCHECKS = {', '\n' + mm + chk + 'CHECKS = {', 1)
old_reg = '"no_scipy": chk_no_scipy, "live_wiring": chk_live_wiring}'
assert t.count(old_reg) == 1
t = t.replace(old_reg, '"no_scipy": chk_no_scipy, "live_wiring": chk_live_wiring, "field_reach_active": chk_field_reach_active_sample}')
mut = '    ("gate_checks.py", "    out[\\"active_index\\"] = idx[active[0]]\n    sl = slice(idx[active[0]], idx[active[0]] + 1)", "    out[\\"active_index\\"] = idx[0]\n    sl = slice(idx[0], idx[0] + 1)", "field_reach_active"),   # always probe idx[0] (the Fs defect)\n'
mut = mut.replace('\n    sl', '\\n    sl')
anchor = '    ("launch_gate.py", "        out.append(f\\"{name}: --out {ow} already holds ckpt_last.pt'
assert t.count(anchor) == 1
i = t.index(anchor); j = t.index('\n]\n', i)
t = t[:j + 1] + mut + t[j + 1:]
ast.parse(t)
open(OUT + 'test_field_p7_profile.py', 'wb').write(t.encode('utf-8'))
open('E:/tmp_old_gc.py', 'w') if False else None
print('ok', len(new), len(t))
