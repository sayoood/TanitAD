"""Cross-check every load-bearing number quoted in RESULT.md against its artifact.

⭐ The expectations are written as LITERALS, never as expressions over the JSON,
so this cannot pass by re-deriving the producer's own arithmetic.
"""
import json, pathlib, sys, re

D = pathlib.Path('TanitAD Research Lab/Architecture & Inference/Research/'
                 '2026-09-18-a4-factorised-vocabulary')
fr = json.loads((D / 'raw' / 'factorisation_rank.json').read_text(encoding='utf-8'))
sp = json.loads((D / 'raw' / 'survey_record_probe.json').read_text(encoding='utf-8'))
md = (D / 'RESULT.md').read_text(encoding='utf-8')

pa = fr['product_arithmetic']
fails = []


def eq(name, got, want):
    ok = got == want
    print(('  OK  ' if ok else '  FAIL') + ' %-38s got=%r want=%r' % (name, got, want))
    if not ok:
        fails.append(name)


def close(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(('  OK  ' if ok else '  FAIL') + ' %-38s got=%.6f want=%.6f' % (name, got, want))
    if not ok:
        fails.append(name)


print('-- artifact vs RESULT.md literals')
eq('C2 recovered paths', pa['C2_recovered']['paths'], 8)
eq('C2 recovered speeds', pa['C2_recovered']['speeds'], 16)
eq('C2 PASS', pa['C2_recovered']['PASS'], True)
eq('C3 recovered', (pa['C3_recovered']['paths'], pa['C3_recovered']['speeds']), (1, 1))
eq('C4 recovered at op point', (pa['C4_recovered']['paths'], pa['C4_recovered']['speeds']),
   (91, 23))
eq('bank paths lower bound', pa['bank_paths_LOWER_BOUND'], 45)
eq('bank speeds lower bound', pa['bank_speeds_LOWER_BOUND'], 84)
eq('product candidates', pa['product_candidates'], 3780)
eq('logits joint/factorised', (pa['logits_joint'], pa['logits_factorised']), (128, 129))
eq('floats joint/factorised', (pa['floats_joint'], pa['floats_factorised']), (2048, 20562))
close('density multiplier', pa['density_multiplier'], 29.53125, 1e-9)
close('floats ratio', pa['floats_ratio'], 10.0400390625, 1e-9)

c2b = fr['controls']['C2b_instrument_floor']
close('C2b max within-group rms', c2b['max_within_group_rms_m'], 0.9959, 5e-4)
close('C2b mean within-group rms', c2b['mean_within_group_rms_m'], 0.1815, 5e-4)
eq('C2b n_pairs', c2b['n_pairs'], 448)

c1b = {r['ds_m']: r for r in fr['C1b_absolute_grid_roundtrip']}
close('C1b ds=1.0 max', c1b[1.0]['max_abs_err_m'], 0.0657, 5e-4)
close('C1b ds=1.0 mean', c1b[1.0]['mean_abs_err_m'], 0.0043, 5e-5)
close('C1b ds=0.25 max', c1b[0.25]['max_abs_err_m'], 0.0187, 5e-4)
close('C1b ds=0.25 mean', c1b[0.25]['mean_abs_err_m'], 0.0011, 5e-5)

sweep = {r['n_grid']: r for r in fr['C1_roundtrip_grid_sweep']}
close('C1 G=16 max', sweep[16]['max_abs_err_m'], 0.298, 1e-3)
close('C1 G=4096 max', sweep[4096]['max_abs_err_m'], 0.00092, 1e-5)

L = fr['bank']['length_m']
close('arc length min', L['min'], 1.2042, 5e-4)
close('arc length max', L['max'], 216.9543, 5e-4)
close('arc length mean', L['mean'], 74.8825, 5e-4)
sp5 = fr['axis_spread']['final_speed_ms_p5_p50_p95']
close('final speed p5', sp5[0], 1.6629, 5e-4)
close('final speed p50', sp5[1], 10.6260, 5e-4)
close('final speed p95', sp5[2], 28.9518, 5e-4)

eq('substrate sha256', fr['sha256'],
   '68f81acf83b6b21ce533df282f78fe269e9b3b462eb82c726004821c7c1a806b')

print('-- survey probe')
f1 = sp['files']['raw_report1_autonomous_driving.json']
f2 = sp['files']['raw_report2_robotics_wm_jepa.json']
eq('report1 n_wp/n_preview', (f1['n_workflowProgress'], f1['n_resultPreview']), (111, 106))
eq('report2 n_wp/n_preview', (f2['n_workflowProgress'], f2['n_resultPreview']), (114, 109))
eq('report1 preview lengths', f1['distinct_resultPreview_lengths'], [401])
eq('report2 preview lengths', f2['distinct_resultPreview_lengths'], [401])
eq('refuted[12] keys', sp['refuted_entry_12']['keys'], ['claim', 'vote', 'source'])
eq('refuted[12] vote', sp['refuted_entry_12']['vote'], '0-3')
eq('library 29163 count', sp['library_banking']['occurrences_of_29163'], 0)
for i, frag in ((83, 'FACTUAL CORE VERIFIED'),
                (84, 'INTERPRETIVE CORE IS REFUTED'),
                (85, 'DESCRIPTIVE HALF: VERIFIED')):
    k = 'workflowProgress[%d]' % i
    got = frag in sp['verifier_previews'][k]['resultPreview']
    eq('verifier %d quote present' % i, got, True)
    eq('verifier %d refuted:true' % i,
       '"refuted":true' in sp['verifier_previews'][k]['resultPreview'], True)

print('-- RESULT.md contains the quoted values')
for lit in ('68f81acf83b6b21ce533df282f78fe269e9b3b462eb82c726004821c7c1a806b',
            '3,780', '29.5', '0.0657', '0.0043', '0.9959', '0.1815',
            '106 + 109', '401 characters', '216.95', '0.3796', '0.6843',
            '14.3264', '1.0882', '20,562', '2,048'):
    ok = lit in md
    print(('  OK  ' if ok else '  FAIL') + ' RESULT.md contains %r' % lit)
    if not ok:
        fails.append('md:' + lit)

# Negative control: the C2-REFUTED first reading (80 paths x 128 speeds, "80x")
# must appear ONLY inside the narrative that says C2 killed it -- never as a
# result. Tolerant of markdown emphasis and line wrapping; the CONTEXT test is
# the load-bearing half.
PAT = r'80\W{0,4}distinct\s+paths\s*(?:\*\*)?\s*and\s*(?:\*\*)?\s*128\W{0,4}distinct\s+speeds'


def negative_control(text):
    m = re.search(PAT, text)
    if not m:
        return False, 'the refuted reading is not present at all'
    ctx = text[max(0, m.start() - 400):m.start()]
    if 'FAILING FIRST' in ctx or 'refuted' in ctx.lower():
        return True, 'present, and inside the C2-failure narrative'
    return False, 'PRESENT OUTSIDE the C2-failure narrative -- reads as a result'


print('-- negative control (the C2-refuted reading must not be quoted as a result)')
ok, why = negative_control(md)
print(('  OK  ' if ok else '  FAIL') + ' %s' % why)
if not ok:
    fails.append('negative-control')

# DELIBERATE-REGRESSION ARM: reintroduce the real historical defect -- the same
# sentence, lifted out of the C2 narrative into the headline -- and the control
# MUST go RED. Without this the control could be green because it is inert.
mutated = md.replace('⭐ **C2 IS THE CONTROL THAT CHANGED THE RESULT, AND IT DID SO '
                     'BY FAILING FIRST.** The first\nrepresentation normalised '
                     'each path by its own length. Under it', 'The bank read')
red_ok, red_why = negative_control(mutated)
print(('  OK  ' if not red_ok else '  FAIL') +
      ' deliberate-regression arm goes RED (%s)' % red_why)
if red_ok:
    fails.append('negative-control-is-inert')

print()
print('FAILURES:', fails if fails else 'none')
sys.exit(1 if fails else 0)
