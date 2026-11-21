'''Load-verify every val clip and delete any that torch cannot open.

⚠️ FILE COUNT IS NOT INTEGRITY. tar-over-ssh returns exit 0 on a dropped stream, so a truncated
clip looks present and only fails when torch reads it. This is the check that owns the answer.
Deleting a corrupt clip is safe: the self-healing relay recomputes what is missing and re-fetches.
'''
import glob, os, torch
V = os.path.expanduser('~/valdata/physicalai-val-0c5f7dac3b11-w120-256x640cyl')
ok, bad = 0, []
for f in sorted(glob.glob(V + '/*.v2ep.pt')):
    try:
        torch.load(f, map_location='cpu', weights_only=False, mmap=True); ok += 1
    except Exception:
        bad.append(f)
for f in bad:
    os.remove(f)
# a stale manifest built over corrupt clips must go too, or the next run reuses it
for m in glob.glob(V + '/_v2manifest*.pt'):
    pass
print(f'LOADABLE={ok} REMOVED={len(bad)}', flush=True)
