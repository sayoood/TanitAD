"""READ-ONLY probe: for each of the 117 S0 logs, where do DB + camera frames exist NOW, and what was the S0 history. Writes probe_s0_raw.json (small)."""
import json, os, sys, glob, zipfile, sqlite3, collections, time
import numpy as np
sys.path.insert(0, 'C:/Users/Admin/par18/s0probe')
from common_s0 import *
logs, tok_log, by = s0()
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
DATA = 'D:/Projects/TanitAD/data'
# ---------------- DB inventory (from the find lists)
dbpaths = collections.defaultdict(list)
for fn in ('dbfind_D.txt', 'dbfind_C.txt'):
    for l in open(fn, encoding='utf-8'):
        p = l.strip()
        if p:
            dbpaths[os.path.basename(p)[:-3]].append(p.replace('\\', '/'))
# ---------------- batch history
B = SR + '/run/batches'
hist = {l: {} for l in logs}
for u in sorted(os.listdir(B)):
    a = os.path.join(B, u, 'audit.json')
    if not os.path.exists(a) or u.startswith('_'):
        continue
    j = json.load(open(a))
    for lg, r in j['per_log'].items():
        h = hist[lg]
        fb = {}
        for k, v in (r.get('frames_by_root') or {}).items():
            kk = k.replace('\\', '/').rstrip('/')
            fb['STAGE(pod->D:)' if 'srttc_s0_stage' in kk else kk.split('/data/')[-1]] = v
        h.update(unit=u, status=r.get('status'), n_tokens=r.get('n_tokens'), n_tokens_with_frames=r.get('n_tokens_with_frames'),
                 frames_by_root=fb, db_at_s0=(r.get('db') or '').replace('\\', '/'))
    pf = os.path.join(B, u, 'copy_manifest_frames.tsv')
    pd = os.path.join(B, u, 'copy_manifest_dbs.tsv')
    if os.path.exists(pf):
        for l in open(pf, encoding='utf-8'):
            p = l.rstrip('\n').split('\t')
            if len(p) >= 2 and p[1].isdigit():
                lg = p[0].split('/')[0]
                hist[lg].setdefault('staged_frames', {})[p[0]] = int(p[1])
    if os.path.exists(pd):
        for l in open(pd, encoding='utf-8'):
            p = l.rstrip('\n').split('\t')
            if len(p) >= 2 and p[1].isdigit():
                hist[p[0][:-3]]['staged_db_bytes'] = int(p[1])
# ---------------- frame roots: per (root, log, cam) -> set of jpg basenames
ROOTS = {
    'refe_proxy/pixels': (DATA + '/refe_proxy/pixels', 'zip'),
    'refe_m5/pixels': (DATA + '/refe_m5/pixels', 'dir'),
    'heldout_frames': ('C:/Users/Admin/p2heldout_work/frames_heldout', 'dir'),
    'nuplan-camera/mini': (DATA + '/nuplan-camera/mini', 'zip'),
    'nuplan-camera/scenarios4': (DATA + '/nuplan-camera/scenarios4', 'zip'),
    'nuplan-camera/probe': (DATA + '/nuplan-camera/probe', 'zip'),
    'srttc_s0_stage/frames': ('D:/Projects/TanitAD-artifacts/srttc_s0_stage/frames', 'dir'),
    'sh5_ctx/frames_full': ('D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/frames_full', 'dir'),
    'astra/e024_anchors': ('D:/TanitAstraResearch/data/e024_anchors_20261010/sensors', 'dir'),
    'astra/e024_seq_stage_00': ('D:/TanitAstraResearch/data/e024_sequence_images_v1/stage_00/sensors', 'dir'),
    'astra/e024_seq_stage_01': ('D:/TanitAstraResearch/data/e024_sequence_images_v1/stage_01/sensors', 'dir'),
    'astra/e024_seq_stage_02': ('D:/TanitAstraResearch/data/e024_sequence_images_v1/stage_02/sensors', 'dir'),
    'astra/e024_seq_stage_03': ('D:/TanitAstraResearch/data/e024_sequence_images_v1/stage_03/sensors', 'dir'),
    'astra/e024_seq_stage_04': ('D:/TanitAstraResearch/data/e024_sequence_images_v1/stage_04/sensors', 'dir'),
    'astra/e024_seq_stage_05': ('D:/TanitAstraResearch/data/e024_sequence_images_v1/stage_05/sensors', 'dir'),
    'astra/e024_seq_stage_06': ('D:/TanitAstraResearch/data/e024_sequence_images_v1/stage_06/sensors', 'dir'),
    'astra/e024_report_images': ('D:/TanitAstraResearch/data/e024_report_images_v1/sensors', 'dir'),
    'astra/e023_sequences': ('D:/TanitAstraResearch/data/e023_sequences_20261010/sensors', 'dir'),
}
avail = {}
n_zip_read = 0
root_exists = {}
for rn, (rp, kind) in ROOTS.items():
    root_exists[rn] = os.path.isdir(rp)
    if not os.path.isdir(rp):
        continue
    ents = set(os.listdir(rp))
    if kind == 'zip':
        for lg in logs:
            for c in CAMS:
                zn = f'{lg}_{c}.zip'
                if zn in ents:
                    try:
                        avail[(rn, lg, c)] = {os.path.basename(m) for m in zipfile.ZipFile(os.path.join(rp, zn)).namelist() if m.lower().endswith('.jpg')}
                        n_zip_read += 1
                    except Exception:
                        avail[(rn, lg, c)] = set()
    else:
        for lg in logs:
            if lg in ents:
                for c in CAMS:
                    d = os.path.join(rp, lg, c)
                    if os.path.isdir(d):
                        avail[(rn, lg, c)] = {f for f in os.listdir(d) if f.lower().endswith('.jpg')}


def nearest(ts, arr):
    j = int(np.argmin(np.abs(arr - ts)))
    return j, abs(int(arr[j]) - ts) / 1000.0


out = {}
for lg in logs:
    rec = {'n_tokens': len(by[lg]), 'hist': {k: v for k, v in hist[lg].items() if k != 'staged_frames'}}
    sf = hist[lg].get('staged_frames', {})
    rec['hist']['staged_frames_n'] = len(sf)
    rec['hist']['staged_frames_bytes'] = sum(sf.values())
    dbs = dbpaths.get(lg, [])
    rec['db_paths'] = [{'path': p, 'bytes': os.path.getsize(p)} for p in dbs]
    pres = {}
    for rn in ROOTS:
        cs = {c: len(avail.get((rn, lg, c), ())) for c in CAMS}
        if any(cs.values()):
            pres[rn] = cs
    rec['roots_present'] = pres
    if dbs:
        db = dbs[0]
        try:
            con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
            ts = {t: int(x) for t, x in con.execute("SELECT lower(hex(token)), timestamp FROM lidar_pc")}
            want = [t for t in by[lg] if t in ts]
            per = {}
            for c in CAMS:
                row = con.execute("SELECT token FROM camera WHERE channel=?", (c,)).fetchone()
                if row:
                    rs = con.execute("SELECT timestamp, filename_jpg FROM image WHERE camera_token=? ORDER BY timestamp", (row[0],)).fetchall()
                    per[c] = (np.array([r[0] for r in rs], dtype=np.int64), [os.path.basename(r[1]) for r in rs])
            loc = None
            try:
                loc = con.execute("SELECT map_version FROM log LIMIT 1").fetchone()
            except Exception:
                pass
            con.close()
            rec['db_ok'] = True
            rec['db_tokens_found'] = len(want)
            rec['map_version'] = str(loc[0]) if loc else None
            n_all4 = 0
            by_root = collections.Counter()
            miss_dt = 0
            needed = set()
            missing_files = set()
            for t in want:
                got_roots = []
                ok = True
                for c in CAMS:
                    if c not in per:
                        ok = False
                        break
                    arr, names = per[c]
                    j, dt = nearest(ts[t], arr)
                    if dt > 120.0:
                        ok = False
                        miss_dt += 1
                        break
                    base = names[j]
                    needed.add((c, base))
                    r = [rn for rn in ROOTS if base in avail.get((rn, lg, c), ())]
                    if not r:
                        ok = False
                        missing_files.add((c, base))
                    else:
                        got_roots.append(r[0])
                if ok:
                    n_all4 += 1
                    for r_ in set(got_roots):
                        by_root[r_] += 1
            rec['tokens_all4cams_resolved'] = n_all4
            rec['tokens_via_root'] = dict(by_root)
            rec['n_needed_frames'] = len(needed)
            rec['n_missing_frames'] = len(missing_files)
            rec['miss_dt_tokens'] = miss_dt
            mfb = 0
            mk = 0
            for c, base in missing_files:
                k = f'{lg}/{c}/{base}'
                if k in sf:
                    mfb += sf[k]
                    mk += 1
            rec['missing_frames_bytes_known'] = mfb
            rec['missing_frames_with_known_size'] = mk
        except Exception as e:
            rec['db_ok'] = False
            rec['db_error'] = repr(e)[:200]
    else:
        rec['db_ok'] = False
        rec['db_error'] = 'no DB file found under D:/Projects, D:/tanitad, D:/hold_temporal, D:/TanitAstra*, D:/Archive, D:/thor_archive, ... or C:/Users/Admin'
    out[lg] = rec
json.dump({'n_zip_read': n_zip_read, 'root_exists': root_exists, 'logs': out}, open('probe_s0_raw.json', 'w'), indent=0)
print('done, zips read', n_zip_read, root_exists)
nodb = [l for l in logs if not out[l]['db_ok']]
print('logs without usable local DB:', len(nodb), '| with DB:', 117 - len(nodb))
full = [l for l in logs if out[l].get('tokens_all4cams_resolved') == out[l]['n_tokens']]
print('logs fully resolved (all tokens x 4 cams):', len(full), 'tokens', sum(out[l]['n_tokens'] for l in full))
print('tokens resolved total:', sum(out[l].get('tokens_all4cams_resolved', 0) for l in logs), 'of', sum(out[l]['n_tokens'] for l in logs))
