"""MM 2026-11-22: after ZZRANKW_TESTS_OK, add tests.tsv + prep.py + tests.sh + pull_framed.py to prep.py's land list, check the tip has not moved, write the message."""
import json, os, subprocess
S = 'C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/e07efd49-d158-4484-85f3-d8888196bc46/scratchpad'
O = S + '/rankw_land'; M = 'C:/Users/Admin/tanitad-push'
PK = 'TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan'
RAWD = PK + '/raw/2026-11-22-scorerft-rankw/'
prep = json.load(open(O + '/prep.json'))
tip = subprocess.run(['git', '-C', M, 'ls-remote', 'origin', 'refs/heads/agent/arch-inf-20260803'], capture_output=True).stdout.decode().split()[0]
assert tip == prep['tip'], ('the tip moved since prep.py', tip[:8], prep['tip'][:8])
tsv = open(O + '/tests.tsv', 'rb').read()
assert tsv.count(b'\t0\t1\t') == 13 and b'\t0\t0\t' not in tsv, 'tests.tsv is not 13 passes'
items = prep['items']
for name in ('tests.tsv', 'prep.py', 'tests.sh', 'pull_framed.py', 'add_evidence.py'):
    data = open(O + '/' + name, 'rb').read().replace(b'\r\n', b'\n')
    repo = RAWD + name
    assert len(repo) <= 200
    dst = O + '/files/%03d_%s' % (len(items), name)
    open(dst, 'wb').write(data)
    items.append([repo, dst, 'raw', 'base=<absent>'])
for it in items:
    assert b'\r\n' not in open(it[1], 'rb').read(), it[0]
json.dump(items, open(O + '/land_paths.json', 'w'))
open(O + '/msg.txt', 'w', newline='\n').write(
    'REFe SCORER-FT: register the weight rule output SCORERFT_RANK_W = 0.03125 (package B)\n\n'
    'MEASURED on the pod 2026-11-22 22:31Z (A3 weight rule, T argv, C1 init, pdm labels): RULE SELECT rank_w = 1/32,\n'
    'r(1/32) = 1.7496 >= 1 (r(1/64) = 0.8748), rho(1/32) = 0.2924 <= 1; replay diffs 0.0; pairs ep 25,025 / gate 105 (1 set: the gate term is thin).\n'
    'launch_gate.py: the one line SCORERFT_RANK_W = None -> 0.03125. Tests on the registered tree: 13/13 pass (tests.tsv).\n'
    'prep.py reads the rule JSON under d[rule] (the 11-21 draft read keys the tool never writes); tests.sh no longer uses the C: sweep tree.\n\n'
    'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n')
print('tip', tip[:8], 'items', len(items))
for it in items:
    print(' ', it[3][:18], it[0][len(PK) + 1:])
