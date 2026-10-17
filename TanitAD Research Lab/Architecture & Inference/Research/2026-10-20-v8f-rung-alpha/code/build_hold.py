"""Build the qland hold for v8F alpha-1 from the working tree. Does NOT run land_hold.sh, does not git add/commit/push.

stage/    NEW files (text -> CRLF, the repo's convention; binaries untouched)
mods-rw/  the 9 edited files, full contents (the working tree is the base commit's archive + the edits; verified against
          the CURRENT tip's blob below: only the hold's own hunks may differ)
msg.txt   last line = the Co-Authored-By trailer
usage: python build_hold.py <tree> <hold dir>
"""
import difflib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TREE = Path(sys.argv[1])
HOLD = Path(sys.argv[2])
GIT_DIR = 'C:/Users/Admin/tanitad-push/.git'
BR = 'refs/heads/agent/arch-inf-20260803'
PKG = 'TanitAD Research Lab/Architecture & Inference/Research/2026-10-20-v8f-rung-alpha'
env = dict(os.environ, GIT_DIR=GIT_DIR)


def git(*a, text=True):
    p = subprocess.run(['git', *a], env=env, capture_output=True)
    if p.returncode:
        raise SystemExit(f'git {a} failed: {p.stderr.decode("utf-8", "replace")[:300]}')
    return p.stdout.decode('utf-8', 'replace') if text else p.stdout


tip = git('rev-parse', BR).strip()
print('tip', tip)

NEW = [
    'stack/tanitad/models/p10_prior.py',
    'stack/tanitad/configs/v8f_p10/p10_prior.pt',
    'stack/tanitad/configs/v8f_p10/p10_fixture.npz',
    'stack/tanitad/configs/v8f_p10/p10_pose_prior.pt',
    'stack/tanitad/configs/v8f_p10/p10_pose_fixture.npz',
    'stack/tanitad/configs/v8f_p10/MANIFEST.json',
    'stack/tests/test_p10_prior.py',
    'stack/tests/test_p10_pose_prior.py',
    'stack/tests/test_launch_gate_v8f.py',
]
for root, _d, files in os.walk(TREE / PKG):
    for f in files:
        NEW.append(str(Path(root, f).relative_to(TREE)).replace('\\', '/'))
MODS = [
    'stack/tanitad/models/kinematic_prior.py', 'stack/tanitad/refs/refc.py', 'stack/scripts/refc_v3_train.py',
    'stack/tanitad/train/declared_vs_built.py', 'stack/scripts/launch_gate.py', 'stack/scripts/launch_gate_arms.py',
    'stack/tanitad/eval/refcv7_loader.py', 'stack/tests/test_residual_prior.py', 'stack/tests/test_declared_vs_built.py',
]
BINARY = ('.pt', '.npz', '.npy', '.png', '.pdf')

if HOLD.exists():
    # only a hold THIS script built (it leaves build_report.json) and that was never landed may be rebuilt
    if not (HOLD / 'build_report.json').is_file() or (HOLD.parent / ('landed_' + HOLD.name.removeprefix('hold_'))).exists():
        raise SystemExit(f'{HOLD} exists and is not a rebuildable hold -- refusing')
    for sub in ('stage', 'mods-rw'):
        shutil.rmtree(HOLD / sub, ignore_errors=True)
    for f in ('msg.txt', 'build_report.json'):
        (HOLD / f).unlink(missing_ok=True)
(HOLD / 'stage').mkdir(parents=True)
(HOLD / 'mods-rw').mkdir(parents=True)

# 1. stage: none of them may exist on the tip (the lander's ADD-OVERWRITE guard)
for rel in NEW:
    p = subprocess.run(['git', 'cat-file', '-e', f'{tip}:{rel}'], env=env, capture_output=True)
    if p.returncode == 0:
        raise SystemExit(f'ADD-EXISTS-ON-TIP {rel}')
    src, dst = TREE / rel, HOLD / 'stage' / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    b = src.read_bytes()
    if not rel.endswith(BINARY):
        b = b.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
    dst.write_bytes(b)
print('stage files', len(NEW))

# 2. mods-rw: full edited files; each must be tip-blob + only the hold's hunks
report = {}
for rel in MODS:
    tipb = git('cat-file', '-p', f'{tip}:{rel}', text=False)
    new = (TREE / rel).read_bytes()
    dst = HOLD / 'mods-rw' / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(new)
    a = tipb.decode('utf-8').splitlines()
    b = new.decode('utf-8').splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    add = rem = 0
    hunks = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != 'equal':
            hunks += 1
            rem += i2 - i1
            add += j2 - j1
    mixed = new.count(b'\r\n') != new.count(b'\n')
    report[rel] = {'tip_lines': len(a), 'new_lines': len(b), 'added': add, 'removed': rem, 'hunks': hunks, 'mixed_line_endings': mixed}
    print(f'{rel}: +{add} -{rem} in {hunks} hunks; mixed endings: {mixed}')
(HOLD / 'build_report.json').write_text(json.dumps({'tip': tip, 'mods_rw': report, 'stage': NEW}, indent=1), encoding='utf-8')
