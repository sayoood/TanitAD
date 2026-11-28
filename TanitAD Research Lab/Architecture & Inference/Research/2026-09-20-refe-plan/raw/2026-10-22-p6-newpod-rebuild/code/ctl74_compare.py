#!/usr/bin/env python3
"""74-twin cross-pod reproducibility control: compare a NEW run (held-arm shards) against the BANKED control run of 2026-10-17 (old pod).
usage: ctl74_compare.py <new_dir with shards/s*/{audit,rows}.jsonl> <banked_dir with held/shards/s*/{audit,rows}.jsonl> <banked merged targets_rank1.jsonl> [--json out]
The NEW run's rank-0 ref file carries the BANKED kept rows' traj, so a kept row's `aug.divergence_m` is the max displacement (m, 4 dp) between the new and the banked
4 s trajectory: 0.0 iff they are the same to 4 dp.  Everything else is compared exactly (python float ==, i.e. bit equality of the parsed doubles)."""
import glob, json, sys, os

def jl(p):
    out = []
    for ln in open(p, encoding='utf-8'):
        if ln.strip():
            try: out.append(json.loads(ln))
            except json.JSONDecodeError: pass
    return out

def audit_of(root):
    d = {}
    for p in sorted(glob.glob(os.path.join(root, 's*', 'audit.jsonl'))):
        for r in jl(p): d[(r['log_name'], r['token'], int(r['k']))] = r
    return d

def rows_of(root):
    d = {}
    for p in sorted(glob.glob(os.path.join(root, 's*', 'rows.jsonl'))):
        for r in jl(p): d[(r['log_name'], r['token'], int(r['aug']['value']))] = r
    return d

AUD = ('v0', 'f_at_s', 'f10_raw', 'd_twin_at_s', 'commit', 'drop_reason', 'hold_misses', 'hold_hits')

def main(argv):
    new_dir, old_dir, merged = argv[1], argv[2], argv[3]
    outp = argv[argv.index('--json') + 1] if '--json' in argv else None
    na, oa = audit_of(os.path.join(new_dir, 'shards')), audit_of(os.path.join(old_dir, 'held', 'shards'))
    nr = rows_of(os.path.join(new_dir, 'shards'))
    orow = {}
    for r in jl(merged):
        a = r.get('aug')
        if isinstance(a, dict) and a.get('reroll') == '3c': orow[(r['log_name'], r['token'], int(a['value']))] = r
    res = {'n_new_audit': len(na), 'n_old_audit': len(oa), 'keys_only_new': sorted(set(na) - set(oa))[:3], 'keys_only_old': sorted(set(oa) - set(na))[:3]}
    eq = 0; diffs = []; maxdf = 0.0; maxdv = 0.0
    for k in sorted(set(na) & set(oa)):
        bad = [f for f in AUD if na[k].get(f) != oa[k].get(f)]
        if bad: diffs.append((k[1], k[2], bad))
        else: eq += 1
        if na[k].get('f10_raw') is not None and oa[k].get('f10_raw') is not None: maxdf = max(maxdf, abs(na[k]['f10_raw'] - oa[k]['f10_raw']))
        if na[k].get('v0') is not None and oa[k].get('v0') is not None: maxdv = max(maxdv, abs(na[k]['v0'] - oa[k]['v0']))
    res.update(audit_bit_equal=eq, audit_compared=len(set(na) & set(oa)), audit_diff_examples=diffs[:5], max_abs_f10_raw=maxdf, max_abs_v0=maxdv)
    req = 0; rdiff = []; maxdiv = 0.0
    for k in sorted(set(orow) | set(nr)):
        a, b = nr.get(k), orow.get(k)
        if a is None or b is None: rdiff.append((k[1], k[2], 'missing_in_new' if a is None else 'missing_in_old')); continue
        a2 = json.loads(json.dumps(a)); b2 = json.loads(json.dumps(b))
        maxdiv = max(maxdiv, float(a2['aug'].pop('divergence_m'))); b2['aug'].pop('divergence_m')
        if a2 == b2: req += 1
        else: rdiff.append((k[1], k[2], [f for f in set(a2) | set(b2) if a2.get(f) != b2.get(f)]))
    res.update(rows_bit_equal=req, rows_old=len(orow), rows_new=len(nr), rows_diff_examples=rdiff[:5], max_divergence_vs_banked_m=maxdiv)
    ok = (len(na) == len(oa) == eq and len(oa) > 0 and req == len(orow) == len(nr) and len(orow) > 0 and maxdiv == 0.0 and not res['keys_only_new'] and not res['keys_only_old'])
    res['verdict'] = 'ZZCTL74_PASS' if ok else 'ZZCTL74_FAIL'
    print(f"{res['verdict']} audit {eq}/{len(oa)} bit-equal (max|d f10_raw| {maxdf:.3g}, max|d v0| {maxdv:.3g}); kept rows {req}/{len(orow)} equal, new rows {len(nr)}; max divergence vs banked {maxdiv} m")
    if outp: json.dump(res, open(outp, 'w'), indent=1, default=str)
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))
