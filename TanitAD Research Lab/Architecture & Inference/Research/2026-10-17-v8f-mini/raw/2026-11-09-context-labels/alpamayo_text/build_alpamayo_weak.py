"""Alpamayo CoT text -> weak strategic-context labels. Built 2026-11-08.

Reads VOCAB.json (frozen before scoring), the v7.2 train+eval label releases (exported from git tip
6a3fa0d7 into scratch X), and the pilot frames. CPU only. Writes alpamayo_weak.parquet and scores.json.
"""
import glob, gzip, json, math, re, collections, hashlib
import pandas as pd

SL = 'E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/strat_labels'
OUT = SL + '/alpamayo_text'
X = 'F:/TanitAD-scratch/alpamayo_text/x/'
REC = 'E:/Projects/TanitAD-artifacts/_s2build-copy-20260919/alpamayo/records.parquet'
V = json.load(open(OUT + '/VOCAB.json', encoding='utf8'))
NEG = re.compile(V['negation']['regex'], re.I)
PAT = {f: {c: re.compile(p, re.I) for c, p in cs.items()} for f, cs in V['fields'].items()}
FIELDS = V['text_fields_scanned']


def wil(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1))


def sentences(t):
    return len([s for s in re.split(r'(?<=[.!?])\s+|\n+', t.strip()) if len(s.strip()) > 3]) if t else 0


def scan(text):
    res = {}
    for f, cs in PAT.items():
        for c, rx in cs.items():
            pos = neg = None
            for m in rx.finditer(text):
                span = text[max(0, m.start() - 20):m.end() + 20].replace('\n', ' ')
                if NEG.search(text[max(0, m.start() - 40):m.start()]):
                    neg = neg or span
                else:
                    pos = pos or span
            res[(f, c)] = (pos, neg)
    return res


recs = {}
for fn, split in (('train.jsonl.gz', 'train'), ('eval.jsonl.gz', 'eval')):
    for l in gzip.open(X + fn, 'rt', encoding='utf8'):
        r = json.loads(l)
        recs[r['clip_id']] = (split, r['cot_source'])
for l in open(X + 'a1.jsonl', encoding='utf8'):
    r = json.loads(l)
    if r['clip_id'] not in recs:
        recs[r['clip_id']] = ('a1_only', {'cot': r['cot']})
rows = []
for cid, (split, cs) in recs.items():
    per = {k: (cs.get(k) or '') for k in FIELDS}
    text = '\n'.join(per[k] for k in FIELDS)
    row = dict(clip_id=cid, split=split, n_sent=sum(sentences(per[k]) for k in FIELDS),
               n_sent_short=sentences(per['cot']) + sentences(per['chain_of_causation']),
               has_text=bool(text.strip()), has_long=bool(per['components_analysis'] or per['motion_analysis']))
    s = scan(text)
    sp = {k: scan(per[k]) for k in FIELDS if per[k]}
    for (f, c), (p, n) in s.items():
        row[f'{f}.{c}'] = p is not None
        row[f'{f}.{c}.neg'] = n is not None
        row[f'{f}.{c}.span'] = p
        row[f'{f}.{c}.in'] = ','.join(k for k in sp if sp[k][(f, c)][0])
    rows.append(row)
D = pd.DataFrame(rows)
classes = [(f, c) for f, cs in V['fields'].items() for c in cs]
out = {'n_clips': len(D), 'by_split': D.split.value_counts().to_dict(), 'has_text': int(D.has_text.sum()),
       'has_long': int(D.has_long.sum()), 'sent_all_fields': D.n_sent.describe().round(2).to_dict(),
       'sent_short_cot_coc': D.n_sent_short.value_counts().sort_index().to_dict()}
cov = {}
for f, c in classes:
    k = int(D[f'{f}.{c}'].sum())
    cov[f'{f}.{c}'] = dict(k=k, n=len(D), pct=round(100 * k / len(D), 2), ci=wil(k, len(D)),
                           negated=int(D[f'{f}.{c}.neg'].sum()),
                           train=int(D[D.split == 'train'][f'{f}.{c}'].sum()), eval=int(D[D.split == 'eval'][f'{f}.{c}'].sum()),
                           by_source={s: int(D[f'{f}.{c}.in'].str.contains(s).sum()) for s in FIELDS})
out['coverage'] = cov
W1 = ['weather.rain', 'weather.snow', 'weather.fog', 'weather.snow_bank', 'weather.clear_sky', 'surface.wet',
      'surface.snow_ice', 'surface.dry', 'visibility.reduced', 'visibility.good']
L1 = ['illumination.dark', 'illumination.night', 'illumination.twilight', 'illumination.day', 'illumination.street_light']
anyw, anyl = D[W1].any(axis=1), D[L1].any(axis=1)
for nm, s in (('any_weather_surface_vis', anyw), ('any_light', anyl), ('any_either', anyw | anyl)):
    out[nm] = dict(k=int(s.sum()), n=len(D), ci=wil(int(s.sum()), len(D)))

R = pd.read_parquet(REC)
Rv = R[R.task == 'vqa']
vq = []
for _, r in Rv.iterrows():
    j = json.loads(r.raw_json)
    a = ' '.join(map(str, j.get('answer') or []))
    s = scan(a)
    vq.append(dict(cat=r.vqa_category, hits=[f'{f}.{c}' for (f, c), (p, n) in s.items() if p]))
V2 = pd.DataFrame(vq)
out['vqa_secondary'] = dict(n_clips=len(V2), n_with_any_hit=int((V2.hits.str.len() > 0).sum()),
                            hit_counts=dict(collections.Counter(h for hs in V2.hits for h in hs)),
                            n_env_vis_questions=int(V2.cat.isin(['environment', 'visibility']).sum()))

fr = pd.concat([pd.read_parquet(f) for f in glob.glob('F:/strat_out/pilot/frames/*.parquet')])
vl = fr[fr.weather_src == 'vlm']
inwin = vl[(vl.t_s >= 5.1) & (vl.t_s <= 11.1)]


def maj(df, col):
    d = df[df[col] != 'unknown']
    return {c: g[col].value_counts().idxmax() for c, g in d.groupby('clip_id')}


Dp = D.set_index('clip_id')
pil = sorted(set(fr.clip_id))
out['pilot_clips'] = len(pil)
out['pilot_in_corpus'] = int(sum(c in Dp.index for c in pil))
TGT = {
    ('weather', 'rain'): ('weather', {'rain'}), ('weather', 'snow'): ('weather', {'snow'}), ('weather', 'fog'): ('weather', {'fog'}),
    ('weather', 'snow_bank'): ('weather', {'snow'}), ('weather', 'clear_sky'): ('weather', {'clear'}),
    ('surface', 'wet'): ('surface', {'wet'}), ('surface', 'snow_ice'): ('surface', {'snow_ice'}), ('surface', 'dry'): ('surface', {'dry'}),
    ('visibility', 'reduced'): ('visibility', {'reduced'}), ('visibility', 'good'): ('visibility', {'good'}),
    ('illumination', 'dark'): ('illum_img', {'dark', 'lit_night'}), ('illumination', 'night'): ('illum_img', {'dark', 'lit_night'}),
    ('illumination', 'twilight'): ('illum_img', {'twilight'}), ('illumination', 'day'): ('illum_img', {'day', 'glare'}),
    ('illumination', 'street_light'): ('illum_img', {'lit_night'}),
}
PIKEY = {'weather': 'weather', 'surface': 'surface', 'visibility': 'visibility', 'illum_img': 'illumination'}


def pr(rows):
    n = len(rows)
    tp = sum(1 for t, r in rows if t and r)
    tpos = sum(1 for t, r in rows if t)
    rpos = sum(1 for t, r in rows if r)
    return dict(n=n, text_pos=tpos, ref_pos=rpos, tp=tp,
                precision=None if tpos == 0 else round(100 * tp / tpos, 1), p_ci=wil(tp, tpos),
                recall=None if rpos == 0 else round(100 * tp / rpos, 1), r_ci=wil(tp, rpos))


sc = {}
for name, frames in (('all_vlm_frames', vl), ('window_5.1_11.1', inwin)):
    M = {col: maj(frames, col) for col in ('weather', 'surface', 'visibility', 'illum_img')}
    sc[name] = {}
    for (f, c), (col, tg) in TGT.items():
        rows = [(bool(Dp.loc[cid, f'{f}.{c}']), M[col][cid] in tg) for cid in pil if cid in Dp.index and cid in M[col]]
        sc[name][f'{f}.{c}'] = pr(rows)
    sc[name]['_ref_class_counts'] = {col: dict(collections.Counter(M[col].values())) for col in M}
out['qwen_clip_majority'] = sc
pod = fr.groupby('clip_id').period_of_day.agg(lambda s: s.value_counts().idxmax())
PM = {'night': {'night'}, 'day': {'morning', 'midday', 'afternoon'}, 'twilight': {'evening'}}


def tflag(cid, c):
    r = Dp.loc[cid]
    return bool(r['illumination.' + c] or (c == 'night' and r['illumination.dark']))


out['period_vs_pilot_col'] = {c: pr([(tflag(cid, c), pod[cid] in PM[c]) for cid in pil if cid in Dp.index]) for c in PM}
out['period_col_counts'] = pod.value_counts().to_dict()

A = json.load(open(SL + '/pilot_audit_answers_PI_60.json'))['frames']
sc2 = {}
for (f, c), (col, tg) in TGT.items():
    key = PIKEY[col]
    rows = []
    for a in A:
        ans = a.get(key)
        if ans in (None, 'cant_tell') or a['clip_id'] not in Dp.index:
            continue
        rows.append((bool(Dp.loc[a['clip_id'], f'{f}.{c}']), ans in tg))
    sc2[f'{f}.{c}'] = pr(rows)
out['pi_audit_clip_text_vs_frame_answer'] = sc2
out['pi_tod'] = {c: pr([(tflag(a['clip_id'], c), a['tod'] in t) for a in A
                        if a['tod'] not in (None, 'cant_tell') and a['clip_id'] in Dp.index])
                 for c, t in (('night', {'night'}), ('twilight', {'dawn_dusk'}), ('day', {'day'}))}
q = {}
for col, key in PIKEY.items():
    n = k = 0
    for a in A:
        if a.get(key) in (None, 'cant_tell'):
            continue
        row = fr[(fr.clip_id == a['clip_id']) & (fr.frame_idx == a['frame_idx'])]
        if len(row) == 0:
            continue
        v = row.iloc[0][col]
        n += 1
        k += int(v == a[key] or (col == 'illum_img' and {v, a[key]} <= {'day', 'glare'}))
    q[key] = dict(k=k, n=n, ci=wil(k, n))
out['pilot_frame_label_vs_PI'] = q
out['pi_audit_clips'] = len({a['clip_id'] for a in A})
out['pilot_text_pos_counts'] = {f'{f}.{c}': int(sum(Dp.loc[cid, f'{f}.{c}'] for cid in pil if cid in Dp.index)) for f, c in classes}

W = pd.DataFrame({'clip_id': D.clip_id, 'split': D.split})
for f, cs in V['fields'].items():
    cl = [c for c in cs if D[f'{f}.{c}'].sum() > 0]
    for c in cl:
        W[f'{f}.{c}'] = D[f'{f}.{c}']
    if cl:
        W[f'{f}_category'] = D.apply(lambda r: '|'.join(c for c in cl if r[f'{f}.{c}']) or None, axis=1)
        W[f'{f}_span'] = D.apply(lambda r: ' || '.join(f'{c}: {r[f"{f}.{c}.span"]}' for c in cl if r[f'{f}.{c}']) or None, axis=1)
W['period_of_day'] = D.apply(lambda r: '|'.join(x for x, on in (('night', r['illumination.night'] or r['illumination.dark']),
                                                                ('twilight', r['illumination.twilight']), ('day', r['illumination.day'])) if on) or None, axis=1)
W['provenance'] = 'vlm-cot'
W['vocab_sha256'] = hashlib.sha256(open(OUT + '/VOCAB.json', 'rb').read()).hexdigest()
W.to_parquet(OUT + '/alpamayo_weak.parquet', index=False)
out['weak_parquet_cols'] = list(W.columns)
json.dump(out, open(OUT + '/scores.json', 'w', encoding='utf8'), indent=1, default=str)
print('done', len(W))
