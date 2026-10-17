"""v8F alpha-1: the DELIBERATE-REGRESSION MATRIX, run as a script so the evidence is a raw artifact and not 'pytest passed'.

Each row re-installs ONE defect (the gate's own `apply_p10_arm`, the same function the real gate CLI applies under
`launch_gate.py run --arm <name>`) and records, for the check that is graded on it:
  verdict   RED (the guard fired) | ESCAPED (it did not)
  matches   the arm's regex in launch_gate_arms.ARMS matches a reason the guard gave (else the CLI would grade MISCREDITED)
  first     the first reason
The CONTROL row (no arm) must read GREEN on every check, or no RED below means anything.
CPU only (CUDA_VISIBLE_DEVICES=-1), the tiny P10 arm of the P10 module tests on REAL eval141 fixture windows.

usage: run_mutation_matrix.py <tree> <out.json> pose|steer
  pose   arm alpha-1 AS LAUNCHED (`--residual-prior p10_pose`, profile `v8f`)  -- the pose-only P10
  steer  the optional arm alpha-1s (`--residual-prior p10`, profile `v8f_steer`)
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/Admin/v8fa1/tree')
OUT = sys.argv[2] if len(sys.argv) > 2 else None
VARIANT = sys.argv[3] if len(sys.argv) > 3 else 'pose'
STACK = ROOT / 'stack'
for p in (str(STACK), str(STACK / 'scripts'), str(STACK / 'tests'), str(ROOT / 'taniteval')):
    sys.path.insert(0, p)
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import launch_gate as LG                                         # noqa: E402
import launch_gate_arms as AR                                    # noqa: E402
import test_p10_prior as TP                                      # noqa: E402
import test_launch_gate_v8f as TV                                # noqa: E402
from tanitad.train import declared_vs_built as D                 # noqa: E402
from tanitad.models import p10_prior as P10                      # noqa: E402

spec = importlib.util.spec_from_file_location('refc_v3_train_matrix', STACK / 'scripts' / 'refc_v3_train.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
V = TV.VARS[VARIANT]
z = np.load(V.fixture, allow_pickle=False)
fx = {k: z[k] for k in z.files}
PROF = V.prof
OTHER = TV.VARS['steer' if VARIANT == 'pose' else 'pose']
rows = []
print(f'== variant {VARIANT}: mode {V.mode}, profile {PROF["name"]}, artifact {V.file.name} sha {V.sha[:12]}')


def ctx(arm):
    return TV._ctx(V, arm)


def row(arm, check, reasons, extra=None):
    graded = arm in AR.ARMS and check in AR.ARMS[arm]['checks']
    sig = AR.ARMS[arm]['sig'] if graded else None             # the regex is only a contract for the arm's GRADED check
    joined = ' || '.join(str(r) for r in reasons)
    rec = {'arm': arm or 'CONTROL', 'check': check, 'verdict': 'RED' if reasons else 'GREEN',
           'graded_check': bool(graded),
           'n_reasons': len(reasons), 'matches_arm_regex': (bool(re.search(sig, joined)) if (sig and reasons) else None),
           'first': (str(reasons[0])[:260] if reasons else None)}
    if extra:
        rec.update(extra)
    rows.append(rec)
    print(f"{rec['arm']:<26} {check:<8} {rec['verdict']:<5} regex={rec['matches_arm_regex']}  {rec['first'] or ''}"[:230])


def dvb(arm):
    pat = LG._Patcher()
    try:
        if arm:
            LG.apply_p10_arm(ctx(arm), pat, T)
        args, cfg, model = TV._arm_v(T, V)
        mm = D.check(model, args, T.build_parser(), forbid_kinds=())
        return [str(m) for m in mm if 'p10' in str(m).lower() or 'residual' in str(m).lower()]
    finally:
        pat.restore()


# ---- G-DVB -------------------------------------------------------------------------------------------------------
row(None, 'G-DVB', dvb(None))
for arm in ('p10_zero_weights', 'p10_unfrozen', 'p10_variant_flipped'):
    row(arm, 'G-DVB', dvb(arm))
for arm, kw in (('p10_wrong_sha', dict(sha='0' * 64)), ('p10_missing_file', dict(path=V.file.parent / '__nope__.pt')),
                ('wrong_variant_file (not an ARMS entry)', dict(path=OTHER.file, sha=OTHER.sha))):
    try:
        TV._arm_v(T, V, **kw)
        row(arm, 'G-DVB', [])
    except P10.P10Error as e:
        row(arm, 'G-DVB', [f'P10Error: {e}'])
# the profile's own argv rules on the wrong-sha argv (the profile half of the same arm)
a0, a1 = TV.alpha_pair(VARIANT)
import tempfile
_tau = Path(tempfile.mkdtemp()) / 'nav_tau.json'
_tau.write_text(json.dumps({'status': 'OK', 'tau': float(LG.flag_values(a1, '--nav-compliance-tau-rad')[0]),
                            'split_v2_cache': LG.flag_values(a1, '--v2-cache')[0],
                            'labels': LG.flag_values(a1, '--v7-labels')[0]}), encoding='utf-8')
opts = {'nav_tau_record': str(_tau)}
row(None, 'argv', LG.profile_argv_rules(PROF, a1, opts)[0],
    {'note': f'the alpha-1 launch argv under the {PROF["name"]} profile'})
for lab, bad in (('p10_wrong_sha', TV._with(a1, '--p10-prior-sha256', ['0' * 64])),
                 ('argv:ha0_ext_pose (alpha-0 prior)', TV._with(a1, '--residual-prior', ['ha0_ext_pose'])),
                 ('argv:ha0_ext (the steer echo)', TV._with(a1, '--residual-prior', ['ha0_ext'])),
                 (f'argv:{OTHER.mode} (the other P10 variant)', TV._with(a1, '--residual-prior', [OTHER.mode])),
                 ('argv:no --p10-prior', TV._with(a1, '--p10-prior', None))):
    row(lab, 'argv', LG.profile_argv_rules(PROF, bad, opts)[0])


# ---- G-LIVE ------------------------------------------------------------------------------------------------------
def live(arm):
    rec, reasons, det = TV._live(T, V, fx, arm)
    return reasons, {'rows_checked': rec['rows_checked'], 'max_abs_diff_a': round(rec['max_abs_diff_a'], 6),
                     'p10_calls': rec['p10_calls']}


reasons, ex = live(None)
row(None, 'G-LIVE', reasons, ex)
for arm in ('p10_fallback_ha0_ext', 'p10_clock_shift', 'p10_unfrozen', 'p10_zero_weights') + (
        ('p10_pose_a0_stride2',) if VARIANT == 'pose' else ()):
    reasons, ex = live(arm)
    row(arm, 'G-LIVE', reasons, ex)

# ---- G-CKPT ------------------------------------------------------------------------------------------------------
rec, _r, _d = TV._live(T, V, fx)
_a, _c, model = TV._arm_v(T, V)
ck = {'model': {}, 'p10_prior': T._p10_ckpt_extra(model)['p10_prior']}
row(None, 'G-CKPT', LG.judge_p10_ckpt(PROF, V.argv_obs(), ck, rec)[0])
pat = LG._Patcher()
try:
    LG.apply_p10_arm(ctx('p10_ckpt_stamp_dropped'), pat, T)
    dropped = {'model': {}, **T._p10_ckpt_extra(model)}
finally:
    pat.restore()
row('p10_ckpt_stamp_dropped', 'G-CKPT', LG.judge_p10_ckpt(PROF, V.argv_obs(), dropped, rec)[0])

# ---- G-EVAL ------------------------------------------------------------------------------------------------------
_a, _c, mt = TV._arm_v(T, V)
_a, _c, ml = TV._arm_v(T, V)
row(None, 'G-EVAL', LG.judge_p10_eval(PROF, V.argv_obs(), mt, ml)[0])
for arm in ('p10_loader_other_p10', 'p10_variant_flipped'):
    pat = LG._Patcher()
    try:
        LG.apply_p10_arm(ctx(arm), pat)
        _a, _c, other = TV._arm_v(T, V)
    finally:
        pat.restore()
    row(arm, 'G-EVAL', LG.judge_p10_eval(PROF, V.argv_obs(), mt, other)[0])

# ---- G-CLOCK -----------------------------------------------------------------------------------------------------
r0, d0 = LG.p10_clock_check(PROF, V.argv_obs(), [], ROOT)
row(None, 'G-CLOCK', r0, {'in_decoder_vs_banked_max_m': d0['in_decoder_vs_banked_max_m'],
                          'one_frame_early_frac_beyond_tol': d0['one_frame_early_control']['frac_beyond_tol'],
                          'ade6_P10u_vs_P10_m': d0['ade6_P10u_vs_P10_m']['mean']})
for arm in ('p10_clock_shift',) + (('p10_pose_a0_stride2',) if VARIANT == 'pose' else ()):
    pat = LG._Patcher()
    try:
        LG.apply_p10_arm(ctx(arm), pat)
        r1, d1 = LG.p10_clock_check(PROF, V.argv_obs(), [], ROOT)
    finally:
        pat.restore()
    row(arm, 'G-CLOCK', r1, {'in_decoder_vs_banked_max_m': d1['in_decoder_vs_banked_max_m']})

ctl = [r for r in rows if r['arm'] == 'CONTROL']
arms = [r for r in rows if r['arm'] != 'CONTROL']
summary = {'variant': VARIANT, 'profile': PROF['name'], 'mode': V.mode, 'artifact_sha256': V.sha,
           'n_controls': len(ctl), 'controls_all_green': all(r['verdict'] == 'GREEN' for r in ctl),
           'n_arm_rows': len(arms), 'arm_rows_red': sum(r['verdict'] == 'RED' for r in arms),
           'arm_rows_escaped': [f"{r['arm']}@{r['check']}" for r in arms if r['verdict'] != 'RED'],
           'arm_rows_regex_mismatch': [f"{r['arm']}@{r['check']}" for r in arms
                                       if r['verdict'] == 'RED' and r['matches_arm_regex'] is False]}
print(json.dumps(summary, indent=1))
if OUT:
    json.dump({'rows': rows, 'summary': summary}, open(OUT, 'w'), indent=1)
