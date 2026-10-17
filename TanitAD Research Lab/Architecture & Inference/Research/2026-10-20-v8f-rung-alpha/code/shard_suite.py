"""Run the FULL stack suite on a snapshot as N parallel pytest shards (no xdist in the venv), then merge the junit files.

usage: shard_suite.py <stack dir> <out dir> <n shards> <timing junit from an earlier serial run>

Every tests/test_*.py file is run exactly once (shards are a partition of the files, LPT-balanced by the earlier serial times;
files with no timing get 5 s). Each shard is its own pytest process with OMP/MKL threads capped so 12 shards do not oversubscribe the box.
The merged <out dir>/merged.xml has every testcase of every shard; the done marker is the last line of <out dir>/DONE.
"""
import collections
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

STACK, OUT, N, TIMING = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
OUT.mkdir(parents=True, exist_ok=True)
PY = 'C:/Users/Admin/venvs/tanitad/Scripts/python.exe'

tm = collections.Counter()
for tc in ET.parse(TIMING).getroot().iter('testcase'):
    tm[tc.get('classname', '')] += float(tc.get('time') or 0)

files = sorted(p.relative_to(STACK).as_posix() for p in (STACK / 'tests').rglob('test_*.py'))


def cost(f):
    mod = f[:-3].replace('/', '.')
    return sum(v for k, v in tm.items() if k == mod or k.startswith(mod + '.')) or 5.0


shards = [[] for _ in range(N)]
load = [0.0] * N
for f in sorted(files, key=cost, reverse=True):
    i = load.index(min(load))
    shards[i].append(f)
    load[i] += cost(f)
print('files', len(files), 'shard loads (s)', [round(x) for x in load], flush=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUTF8='1')
procs = []
for i, fl in enumerate(shards):
    log = open(OUT / f'shard{i:02d}.log', 'w', encoding='utf-8')
    cmd = [PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--continue-on-collection-errors', '-rfE',
           f'--junitxml={(OUT / f"shard{i:02d}.xml").as_posix()}', *fl]
    procs.append(subprocess.Popen(cmd, cwd=STACK, env=env, stdout=log, stderr=subprocess.STDOUT))
rcs = [p.wait() for p in procs]
root = ET.Element('testsuites')
n = 0
for i in range(N):
    p = OUT / f'shard{i:02d}.xml'
    if not p.exists():
        print('MISSING junit for shard', i, flush=True)
        continue
    for tc in ET.parse(p).getroot().iter('testcase'):
        root.append(tc)
        n += 1
ET.ElementTree(root).write(OUT / 'merged.xml', encoding='utf-8')
(OUT / 'DONE').write_text(f'shard rcs {rcs}\nmerged testcases {n}\nZZSHARD-DONE\n', encoding='utf-8')
print('ZZSHARD-DONE', rcs, n, flush=True)
