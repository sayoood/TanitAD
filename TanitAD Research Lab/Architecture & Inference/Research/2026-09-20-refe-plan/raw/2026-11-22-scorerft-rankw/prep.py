"""MM 2026-11-21: register the weight rule's output as SCORERFT_RANK_W. usage: prep.py <W as the pod logged it (WR_W)> <weight_rule_rank.json pulled from the pod>
1. the tip's refe/ is exported to F:/mm_e07efd49/rankw/refe (autocrlf off) and launch_gate.py's ONE `SCORERFT_RANK_W = None ` line is set to repr(float(W));
2. the JSON must say status OK and rank_w == W, and W must be one of weight_rule_rank.CANDIDATES;
3. writes land_paths.json for the land (launch_gate.py modified on the tip blob + the JSON as evidence). Tests are run by tests.sh on the exported tree."""
import io, json, os, subprocess, sys, tarfile
S = 'C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/e07efd49-d158-4484-85f3-d8888196bc46/scratchpad'
O = S + '/rankw_land'; M = 'C:/Users/Admin/tanitad-push'; X = 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/rankw_x'   # 11-22: E:, not F: (F: below floor)
RAWD = 'raw/2026-11-22-scorerft-rankw/'
PK = 'TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan'
w_s, js = sys.argv[1], sys.argv[2]
W = float(w_s)
def git(*a, **k): return subprocess.run(['git', '-C', M, *a], capture_output=True, **k)
tip = git('ls-remote', 'origin', 'refs/heads/agent/arch-inf-20260803').stdout.decode().split()[0]
subprocess.run(['git', '-C', M, 'fetch', '-q', 'origin', 'agent/arch-inf-20260803'], check=True)
assert git('rev-parse', 'FETCH_HEAD').stdout.decode().strip() == tip
d = json.load(open(js, encoding='utf-8'))
# MM 2026-11-22: the tool writes the decision under d['rule'] (status SELECT | K2 | K2b | ERROR, rank_w); the 11-21 draft read d['status'] == 'OK' / d['rank_w'],
# keys the tool never writes (the same caller defect as after_f3; MEASURED against the real pod JSON a9f13c78). A synthetic-label result is refused.
r = d['rule']
assert r['status'] == 'SELECT' and float(r['rank_w']) == W, (r.get('status'), r.get('rank_w'), W)
assert d['NOT_A_DECISION'] is False and d['label_backend'] != 'synthetic', (d['NOT_A_DECISION'], d['label_backend'])
# 1. export the tip's refe/ (and, for tests.sh's PYTHONPATH, stack/ + taniteval/ without results/ + tools/ under X/q: no longer the C: sweep tree)
assert not os.path.exists(X + '/refe') and not os.path.exists(X + '/q'), X + ' has a previous export: delete it first'
os.makedirs(X, exist_ok=True)
tb = subprocess.run(['git', '-C', M, '-c', 'core.autocrlf=false', 'archive', '--format=tar', tip, PK + '/refe'], capture_output=True, check=True).stdout
with tarfile.open(fileobj=io.BytesIO(tb)) as t:
    for m in t.getmembers():
        if not m.isfile(): continue
        rel = m.name[len(PK) + 1:]
        assert rel.startswith('refe/') and '..' not in rel, m.name
        dst = X + '/' + rel; os.makedirs(os.path.dirname(dst), exist_ok=True); open(dst, 'wb').write(t.extractfile(m).read())
tq = subprocess.run(['git', '-C', M, '-c', 'core.autocrlf=false', 'archive', '--format=tar', tip, 'stack', 'taniteval', 'tools'], capture_output=True, check=True).stdout
nq = 0
with tarfile.open(fileobj=io.BytesIO(tq)) as t:
    for m in t.getmembers():
        if not m.isfile() or m.name.startswith('taniteval/results/'): continue
        assert '..' not in m.name, m.name
        dst = X + '/q/' + m.name; os.makedirs(os.path.dirname(dst), exist_ok=True); open(dst, 'wb').write(t.extractfile(m).read()); nq += 1
print('exported refe + q (%d files)' % nq)
src = open(X + '/refe/weight_rule_rank.py', encoding='utf-8').read()          # no import (torch); the grid line is checked as text
assert src.count('CANDIDATES = tuple(2.0 ** k for k in range(-6, 3))') == 1, 'the weight rule grid changed: re-derive the candidate check'
assert W in tuple(2.0 ** k for k in range(-6, 3)), W
lg = X + '/refe/launch_gate.py'; t = open(lg, 'rb').read().decode('utf-8')
old = 'SCORERFT_RANK_W = None '
assert t.count(old) == 1 and '\r\n' not in t
new = ('SCORERFT_RANK_W = %r' % W).ljust(len(old) - 1) + ' '
t2 = t.replace(old, new, 1)
assert t2.count('SCORERFT_RANK_W = %r' % W) == 1
open(lg, 'wb').write(t2.encode('utf-8'))
os.makedirs(O + '/files', exist_ok=True)
items = []
def stage(repo, data, base):
    assert len(repo) <= 200 and b'\r\n' not in data, repo
    dst = O + '/files/%03d_%s' % (len(items), os.path.basename(repo)); open(dst, 'wb').write(data); items.append([repo, dst, 'raw', base])
bt = git('rev-parse', tip + ':' + PK + '/refe/launch_gate.py').stdout.decode().strip()
stage(PK + '/refe/launch_gate.py', t2.encode('utf-8'), 'base=' + bt)
for repo in (PK + '/' + RAWD + 'weight_rule_rank.json', PK + '/' + RAWD + 'weight_rule_rank.log', PK + '/' + RAWD + 'DRY_tests_standin_w0.5.tsv'):
    assert git('rev-parse', '-q', '--verify', tip + ':' + repo).returncode != 0, 'exists at tip: ' + repo
stage(PK + '/' + RAWD + 'weight_rule_rank.json', open(js, 'rb').read().replace(b'\r\n', b'\n'), 'base=<absent>')
stage(PK + '/' + RAWD + 'weight_rule_rank.log', open(O + '/weight_rule_rank.log', 'rb').read().replace(b'\r\n', b'\n'), 'base=<absent>')
stage(PK + '/' + RAWD + 'DRY_tests_standin_w0.5.tsv', open(O + '/DRY_tests_standin_w0.5.tsv', 'rb').read().replace(b'\r\n', b'\n'), 'base=<absent>')
# tests.sh's real tests.tsv + prep.py / tests.sh are staged by the MM after ZZRANKW_TESTS_OK (add_evidence.py)
json.dump({'tip': tip, 'W': W, 'items': items}, open(O + '/prep.json', 'w'))
json.dump(items, open(O + '/land_paths.json', 'w'))
print('ZZRANKW_PREP tip', tip[:8], 'W', repr(W), 'launch_gate base', bt[:12])
