import json, os, csv, glob, collections
SR='C:/Users/Admin/srttc_s0'
def s0():
    rows=list(csv.DictReader(open(SR+'/run/plan/ranked_fit_logs.tsv'),delimiter='\t'))
    logs=[r['log'] for r in rows if float(r['first_fraction'])<=0.10]
    mp=json.load(open(SR+'/run/plan/master_plan.json'))
    tok_log=mp['tokens']
    by=collections.defaultdict(list)
    for t,l in tok_log.items(): by[l].append(t)
    assert set(by)==set(logs) and len(logs)==117, (len(by),len(logs))
    return logs, tok_log, {l:sorted(v) for l,v in by.items()}
