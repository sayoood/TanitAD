import json
s = json.load(open('spec_leadsup.json'))
T = 'stack/tests/test_lead_sup_gaps.py'
s['mutations'] = [dict(m, test_files=[T], expect=['test_' + m['id'][:2]]) for m in s['mutations'] if m['id'].startswith(('m2_', 'm5_'))]
s['test_files'] = [T]
json.dump(s, open('spec_gaps.json', 'w'), indent=1)
print(len(s['mutations']))
