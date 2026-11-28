import json
s = json.load(open('spec_distrule.json'))
ids = ('w1_', 'w5_', 'w8_', 'w9_', 'q5_', 'q8_')
s['mutations'] = [dict(m, test_files=['src/test_dist_rule_gaps.py'], expect=['test_' + m['id'][:2]]) for m in s['mutations'] if m['id'].startswith(ids)]
s['test_files'] = ['src/test_dist_rule_gaps.py']
s['ignore_failed'] = []
json.dump(s, open('spec_gaps.json', 'w'), indent=1)
print(len(s['mutations']))
