import numpy as np, json, sys, os, re, glob
sys.path.insert(0, 'C:/Users/Admin/par18/s0probe')
from common_s0 import *
logs, tok_log, by = s0()
T = set(tok_log)
HEX = re.compile(r'^[0-9a-f]{16}$')
cands = {
    'P2 S0 bank fan.npz (all units)': sorted(glob.glob(SR + '/work/bank/*.fan.npz')),
    'P2 S0 bank sets.npz (all units)': sorted(glob.glob(SR + '/work/bank/*.sets.npz')),
    'srttc_s0 dgdump': [SR + '/work/dgdump/dgdump.npz', SR + '/work/dgdump/dgdump_feat.npz'],
    'p2_full_table': ['D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz'],
    'p2_heldout_table_full': glob.glob('D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/*.npz'),
    'p2_zero_tail_table': glob.glob('D:/Projects/TanitAD-artifacts/p2_zero_tail_table/*.npz'),
    'headrow_probe oof_proposals_f10': ['D:/Projects/TanitAD-artifacts/headrow_probe_2026-10-19/oof_proposals_f10.npz'],
    'sh5 inputs': ['D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/data/sh5_inputs.npz', 'D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/data/sh5_labels.npz'],
    'l2_offprop feats': ['D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz'],
}
res = {}
for name, files in cands.items():
    toks = set()
    keyinfo = {}
    nprop = 0
    for f in files:
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            a = z[k]
            if a.dtype.kind == 'U' and a.ndim >= 1 and a.size > 0 and a.size < 5_000_000:
                flat = a.reshape(-1)
                if HEX.match(str(flat[0])):
                    toks |= set(map(str, flat.tolist()))
                    keyinfo[k] = list(a.shape)
            if k in ('proposals', 'proposals20'):
                nprop += int(a.shape[0])
        z.close()
    res[name] = {'files': len(files), 'distinct_tokens': len(toks), 'in_S0': len(toks & T), 'token_arrays': keyinfo, 'proposal_rows': nprop}
    print(name, res[name])
json.dump(res, open('C:/Users/Admin/par18/s0probe/token_scan.json', 'w'), indent=1)
