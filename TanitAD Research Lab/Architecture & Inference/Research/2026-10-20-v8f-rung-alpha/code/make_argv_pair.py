"""v8F rung alpha: derive the alpha-0 / alpha-1 launch argv PAIR (and the optional alpha-1s) from the committed refcv8 launch argv.

alpha-0  = stack/ops/runs.d/refcv8-r101-s0.argv.json's argv (the refcv8 recipe, residual prior ha0_ext_pose), with a
           distinct --out;
alpha-1  = alpha-0 with the residual prior swapped to the frozen POSE-ONLY P10: --residual-prior p10_pose --p10-prior <file>
           --p10-prior-sha256 <sha256>, and its own --out. Both arms read past poses only (no steer at inference);
alpha-1s = OPTIONAL: alpha-0 with the STEER-reading P10 (--residual-prior p10). Pending a PI ruling on the recorded steer at
           inference (SPEC_REFCV7 A5); the `v8f` profile REFUSES it by name, `v8f_steer` carries it.

ONLY those (and --out) differ. Everything else -- the recipe, the warm start, the data, the seed, the geometry -- is the
refcv8 launch's, VERBATIM. The geometry (--steps 48852 --batch 16 ...) is a PLACEHOLDER until open item V8F-RUNG-GEOMETRY
closes (the tiny rung is batch 8, ~4.5 h/arm): both arms must carry the SAME values when it does.

usage: python make_argv_pair.py <repo root> <out dir>
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / 'stack' / 'scripts'))
sys.path.insert(0, str(ROOT / 'stack'))
import launch_gate as LG  # noqa: E402

BASE = ROOT / 'stack' / 'ops' / 'runs.d' / 'refcv8-r101-s0.argv.json'
# the artifact ships INSIDE the repo (stack/tanitad/configs/v8f_p10/); the launch tree is the Thor checkout of that repo, so the
# path is repo-relative under it. The sha256 is the identity (the gate checks it); the path is not.
P10_THOR = '/home/nvidia/v8f_run/launch_tree/stack/tanitad/configs/v8f_p10/p10_pose_prior.pt'
SHA = 'e70a9ae30b4418705442bffdeb62d9810a43b0a6ccce0c177796c72295fe693d'
P10S_THOR = '/home/nvidia/v8f_run/launch_tree/stack/tanitad/configs/v8f_p10/p10_prior.pt'
SHAS = 'af5a29ddadb4c7d98f1db0d8116bddd0634e83a7e687fc601989d3fda984d7b3'
assert LG.V8F_P10_SHA256 == SHA and LG.V8F_P10_STEER_SHA256 == SHAS


def pairs(a):
    out, i = [], 0
    while i < len(a):
        j = i + 1
        while j < len(a) and not a[j].startswith('--'):
            j += 1
        out.append((a[i], list(a[i + 1:j])))
        i = j
    return out


def set_flag(argv, flag, vals):
    ps, done = [], False
    for f, v in pairs(argv):
        if f == flag:
            if not done and vals is not None:
                ps.append((f, [str(x) for x in vals]))
            done = True
        else:
            ps.append((f, v))
    if not done and vals is not None:
        ps.append((flag, [str(x) for x in vals]))
    return [t for f, v in ps for t in [f, *v]]


base = json.loads(BASE.read_text(encoding='utf-8'))
a0 = set_flag(list(base['argv']), '--out', ['/home/nvidia/v8f_run/runs/v8f-a0-s0'])
def with_p10(out, mode, path, sha):
    return set_flag(set_flag(set_flag(set_flag(a0, '--out', [out]), '--residual-prior', [mode]), '--p10-prior', [path]),
                    '--p10-prior-sha256', [sha])


a1 = with_p10('/home/nvidia/v8f_run/runs/v8f-a1-s0', 'p10_pose', P10_THOR, SHA)
a1s = with_p10('/home/nvidia/v8f_run/runs/v8f-a1s-s0', 'p10', P10S_THOR, SHAS)
OUT.mkdir(parents=True, exist_ok=True)
for name, argv, prof, prior in (('v8f-a0-s0', a0, 'refcv8', 'ha0_ext_pose'), ('v8f-a1-s0', a1, 'v8f', 'p10_pose'),
                                ('v8f-a1s-s0', a1s, 'v8f_steer', 'p10')):
    rec = {'schema': 'tanitad.launch_argv/1', 'arm': name, 'stage': 'DRAFT -- not launched; geometry is the V8F-RUNG-GEOMETRY placeholder',
           'what': f'v8F rung alpha, arm {name.split("-")[1]}: the refcv8 launch argv with residual prior {prior}. '
                   f'Gate profile: {prof}.',
           'derived_from': {'file': 'stack/ops/runs.d/refcv8-r101-s0.argv.json', 'arm': base['arm'], 'n_tokens': len(base['argv'])},
           'argv': argv}
    (OUT / f'argv_{name}.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
p0 = {f: tuple(v) for f, v in pairs(a0)}


def diff_vs_a0(a):
    p1 = {f: tuple(v) for f, v in pairs(a)}
    return {'only_in_arm': {f: p1[f] for f in p1 if f not in p0},
            'changed': {f: [list(p0[f]), list(p1[f])] for f in p0 if f in p1 and p0[f] != p1[f]},
            'only_in_a0': {f: p0[f] for f in p0 if f not in p1}}


diff = {'a1': diff_vs_a0(a1), 'a1s': diff_vs_a0(a1s)}
print(json.dumps(diff, indent=1))
print('n tokens', len(a0), len(a1), len(a1s), '| a0 sha256', LG.argv_sha256(a0)[:16], '| a1 sha256', LG.argv_sha256(a1)[:16],
      '| a1s sha256', LG.argv_sha256(a1s)[:16])
for k, d in diff.items():
    assert set(d['changed']) == {'--residual-prior', '--out'} and set(d['only_in_arm']) == {'--p10-prior', '--p10-prior-sha256'} \
        and not d['only_in_a0'], k
# the argv rules, both profiles (the nav-tau record is a fixture here: that rule is not under test)
import tempfile
tau = Path(tempfile.mkdtemp()) / 'nav_tau.json'
tau.write_text(json.dumps({'status': 'OK', 'tau': float(LG.flag_values(a1, '--nav-compliance-tau-rad')[0]),
                           'split_v2_cache': LG.flag_values(a1, '--v2-cache')[0], 'labels': LG.flag_values(a1, '--v7-labels')[0]}),
               encoding='utf-8')
opts = {'nav_tau_record': str(tau)}
res = {}
for lab, argv in (('a0', a0), ('a1', a1), ('a1s', a1s)):
    for prof in ('refcv8', 'v8f', 'v8f_steer'):
        ref, pend = LG.profile_argv_rules(LG.PROFILES[prof], argv, opts)
        res[f'{lab} under {prof}'] = {'refusals': ref, 'pi_pending': pend}
        print(f'{lab} under {prof}: {len(ref)} refusals {[r[:130] for r in ref][:3]}')
res['open_items_v8f'] = LG.open_item_reasons(LG.PROFILES['v8f'])
res['open_items_v8f_steer'] = LG.open_item_reasons(LG.PROFILES['v8f_steer'])
res['open_items_refcv8'] = LG.open_item_reasons(LG.PROFILES['refcv8'])
(OUT / 'argv_pair_check.json').write_text(json.dumps({'diff': diff, 'argv_sha256': {'a0': LG.argv_sha256(a0), 'a1': LG.argv_sha256(a1), 'a1s': LG.argv_sha256(a1s)},
                                                      'rules': res}, indent=1), encoding='utf-8')
