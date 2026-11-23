"""Deliberate-regression arms for the two amendments, on COPIES of the exported tree (E: only).  Per-case outcomes come from running the check with a non-asserting `ok`.
  L2 : judge_census_head returns [] unconditionally  -> every FAIL case (c)..(k) (and the extras) must go RED, the PASS cases stay green.
  1b : field_reach always probes idx[0]              -> the index-2 case must go RED.
usage: mutation_l2.py <exported refe dir> <scratch dir on E:>  > mutation.txt"""
import os, shutil, subprocess, sys
src, scratch = sys.argv[1], sys.argv[2]
assert scratch.startswith('E:/') and not os.path.exists(scratch + '/mut'), scratch
DRIVER = '''
import sys
sys.path.insert(0, ".")
import test_scorerft_profile as T
rec = []
def ok(cond, msg=""):
    key = msg[0] if isinstance(msg, tuple) and msg and isinstance(msg[0], str) else (msg if isinstance(msg, str) else "?")
    rec.append((key, bool(cond)))
T.ok = ok
try:
    getattr(T, sys.argv[1])()
except Exception as e:
    rec.append(("EXC " + type(e).__name__, False))
red = sorted({k for k, v in rec if not v})
print("N_CHECKS", len(rec), "RED_CASES", len(red))
for k in red: print("RED", k[:110])
'''
O1 = '    c = census or {}\n    if not c.get("ran"):'
O2 = '    out["active_index"] = idx[active[0]]\n    sl = slice(idx[active[0]], idx[active[0]] + 1)'
M2 = '    out["active_index"] = idx[0]\n    sl = slice(idx[0], idx[0] + 1)'


def run(tag, fn, old=None, new=None):
    d = scratch + '/mut/' + tag
    shutil.copytree(src, d)
    if old:
        p = d + '/gate_checks.py'
        t = open(p, 'rb').read().decode('utf-8')
        assert t.count(old) == 1, tag
        open(p, 'wb').write(t.replace(old, new, 1).encode('utf-8'))
    open(d + '/_driver.py', 'w').write(DRIVER)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=d + ';' + os.environ.get('L2_Q', ''), TMP=scratch + '/tmp', TEMP=scratch + '/tmp', CUDA_VISIBLE_DEVICES='-1')
    r = subprocess.run([sys.executable, '_driver.py', fn], cwd=d, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print('== %s (rc %d)' % (tag, r.returncode))
    print(r.stdout.strip())
    if r.returncode:
        print(r.stderr.strip()[-600:])


os.makedirs(scratch + '/tmp', exist_ok=True)
run('L2_unmutated', 'chk_census_l2')
run('L2_mutated_judge_census_head_returns_empty', 'chk_census_l2', O1, '    return []\n' + O1)
run('1b_unmutated', 'chk_field_reach_active_sample')
run('1b_mutated_always_idx0', 'chk_field_reach_active_sample', O2, M2)
