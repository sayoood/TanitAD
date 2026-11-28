"""SFTT G-LIVE L2 amendment (MM 2026-11-23): export tip refe/ + stack/taniteval/tools to E: l2_scratch/x (autocrlf off), then overlay the amended gate_checks.py,
launch_gate.py, test_scorerft_profile.py from l2/ (the hold).  Nothing is written to C: or F:.  usage: prep_l2.py"""
import io, os, subprocess, sys, tarfile
SHIP = 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/ship'
X = SHIP + '/l2_scratch/x'
M = 'C:/Users/Admin/tanitad-push'
PK = 'TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan'
TIP = '429086acdf8d2754a8f4242f9a9887ca4322b427'
assert not os.path.exists(X + '/refe'), X + ' has a previous export'
os.makedirs(X, exist_ok=True)
tb = subprocess.run(['git', '-C', M, '-c', 'core.autocrlf=false', 'archive', '--format=tar', TIP, PK + '/refe'], capture_output=True, check=True).stdout
with tarfile.open(fileobj=io.BytesIO(tb)) as t:
    for m in t.getmembers():
        if not m.isfile():
            continue
        rel = m.name[len(PK) + 1:]
        assert rel.startswith('refe/') and '..' not in rel, m.name
        dst = X + '/' + rel
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, 'wb').write(t.extractfile(m).read())
tq = subprocess.run(['git', '-C', M, '-c', 'core.autocrlf=false', 'archive', '--format=tar', TIP, 'stack', 'taniteval', 'tools'], capture_output=True, check=True).stdout
nq = 0
with tarfile.open(fileobj=io.BytesIO(tq)) as t:
    for m in t.getmembers():
        if not m.isfile() or m.name.startswith('taniteval/results/'):
            continue
        assert '..' not in m.name, m.name
        dst = X + '/q/' + m.name
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, 'wb').write(t.extractfile(m).read())
        nq += 1
for f in ('gate_checks.py', 'launch_gate.py', 'test_scorerft_profile.py'):
    d = open(SHIP + '/l2/' + f, 'rb').read()
    assert b'\r\n' not in d, f
    open(X + '/refe/' + f, 'wb').write(d)
print('exported refe + q (%d files) + 3 overlays' % nq)
