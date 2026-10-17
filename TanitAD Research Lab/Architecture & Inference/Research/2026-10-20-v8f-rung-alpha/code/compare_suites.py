"""Compare two pytest junit files (mine = the hold applied, base = the pristine base commit c415f1b2): NEW failures/errors.

usage: python compare_suites.py <mine.xml> <base.xml> [out.json]
A test counts as FAILED if it has a <failure> or <error> child. NEW = failed in mine, not failed (or absent) in base.
Tests present only in mine (the hold's new test files) are listed separately; a new test that fails is a new failure.
"""
import json
import sys
import xml.etree.ElementTree as ET


def load(p):
    res = {}
    for tc in ET.parse(p).getroot().iter('testcase'):
        tid = f"{tc.get('classname', '')}::{tc.get('name', '')}"
        st = 'passed'
        for ch in tc:
            if ch.tag in ('failure', 'error'):
                st = 'failed'
                break
            if ch.tag == 'skipped':
                st = 'skipped'
        res[tid] = st
    return res


mine, base = load(sys.argv[1]), load(sys.argv[2])
new_fail = sorted(t for t, s in mine.items() if s == 'failed' and base.get(t) != 'failed')
fixed = sorted(t for t, s in base.items() if s == 'failed' and mine.get(t) == 'passed')
both = sorted(t for t, s in mine.items() if s == 'failed' and base.get(t) == 'failed')
only_mine = sorted(t for t in mine if t not in base)
only_base = sorted(t for t in base if t not in mine)
out = {'mine': {'n': len(mine), 'failed': sum(s == 'failed' for s in mine.values()), 'skipped': sum(s == 'skipped' for s in mine.values())},
       'base': {'n': len(base), 'failed': sum(s == 'failed' for s in base.values()), 'skipped': sum(s == 'skipped' for s in base.values())},
       'NEW_failures': new_fail, 'failed_in_both': both, 'fixed_by_the_hold': fixed,
       'tests_only_in_mine': len(only_mine), 'new_tests_failing': [t for t in only_mine if mine[t] == 'failed'],
       'tests_only_in_base': only_base[:50]}
print(json.dumps({k: (v if not isinstance(v, list) else (v[:40] + [f'... +{len(v) - 40}'] if len(v) > 40 else v)) for k, v in out.items()}, indent=1))
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], 'w'), indent=1)
print('ZZVERDICT', 'NEW-FAILURES' if (new_fail or out['new_tests_failing']) else '0-NEW-FAILURES')
