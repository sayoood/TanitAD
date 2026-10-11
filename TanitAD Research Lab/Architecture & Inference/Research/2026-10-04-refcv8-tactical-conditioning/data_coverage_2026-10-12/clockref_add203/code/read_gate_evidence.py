import json, sys
for lab in sys.argv[1:]:
    e = json.load(open(f'raw/gate_out_{lab}/evidence/G-CLOCK.json', encoding='utf-8'))
    d = e['details']
    print('=====', lab, e['status'], '| profile', e['binding']['profile'])
    for r in e['reasons']: print('  reason:', r[:420])
    print('  refs:', [(x['path'].split('/')[-1], x['n_clips']) for x in d['reference']['files']], 'total', d['reference']['n_reference_clips'])
    r = d['train']
    print('  TRAIN', {k: r[k] for k in ('n_windows', 'n_clips', 'n_violations', 'worst_abs_s', 'n_unverified_clips', 'unverified_frac', 'cap', 'cap_ok', 'n_fallback_clock_clips', 'n_measured_but_unreferenced', 'legacy_control_violations', 'n_labelled_clips', 'n_episodes')})
    g = r['trainer_g3']; print('  trainer_g3 (train):', {k: v for k, v in g.items() if k not in ('unverified_sids',)})
    r = d['eval']
    print('  EVAL ', {k: r[k] for k in ('n_windows', 'n_clips', 'n_violations', 'n_unverified_clips', 'unverified_frac', 'n_fallback_clock_clips', 'n_measured_but_unreferenced', 'mode')})
