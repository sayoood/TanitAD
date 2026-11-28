"""LANDING_READY manifest cross-reference (read-only)."""
import os, re, json, glob, hashlib, subprocess, collections

OUT = 'C:/Users/Admin/estrand/out'
W = OUT + '/work'
E = 'E:/Projects/TanitAD'
TIP = '780a025558d29102bf5f15ba695e5b376c64e301'
MIRROR = 'C:/Users/Admin/tanitad-push'
BS = chr(92)


def lp(path):
    fp = E + '/' + path
    if len(fp) > 240:
        fp = BS + BS + '?' + BS + fp.replace('/', BS)
    return fp


R = json.load(open(W + '/classes_scanned.json'))['results'] if os.path.exists(W + '/classes_scanned.json') \
    else json.load(open(W + '/classes_raw.json'))['results']

tip = {}
for rec in open(W + '/tiptree.bin', 'rb').read().split(b'\0'):
    if not rec:
        continue
    meta, path = rec.split(b'\t', 1)
    tip[path.decode('utf-8', 'surrogateescape')] = meta.split(b' ')[2].decode()

mans = sorted(p for p in R if os.path.basename(p).startswith('LANDING_READY') and p.endswith('.txt'))
print('manifests', len(mans))

cat = subprocess.Popen(['git', '-C', MIRROR, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def blob(sha):
    cat.stdin.write((sha + '\n').encode())
    cat.stdin.flush()
    hdr = cat.stdout.readline().split()
    if len(hdr) < 3:
        return None
    n = int(hdr[2])
    data = cat.stdout.read(n)
    cat.stdout.read(1)
    return data


ROOTS = ('FlyWheels/', 'TanitAD Research Lab/', 'Media/', 'Project Steering/', 'stack/', 'taniteval/', 'tools/',
         'colab/', 'products/', 'Paper/', 'Benchmarks', 'Implementation/', 'docs/', 'tests/')
E_A = re.compile(r'^([0-9a-f]{32})\s+(?:(\d+)\s+)?(\S.*?)\s+--?\s')
E_A2 = re.compile(r'^([0-9a-f]{32})\s+(?:(\d+)\s+)?(\S[^\s].*?)\s*$')
SHA1C = re.compile(r'^#\s+([0-9a-f]{40})')
PREFIX = re.compile(r'(?i)prefix of every path below:?\s*$')


def parse(txt):
    lines = txt.replace(chr(13), '').split(chr(10))
    ents = []   # (kind, ref_hash, path)
    prefix = ''
    last_sha1 = None
    for i, ln in enumerate(lines):
        if PREFIX.search(ln) and i + 1 < len(lines):
            nxt = lines[i + 1].lstrip('# ').strip()
            if nxt:
                prefix = nxt
            continue
        m1 = SHA1C.match(ln)
        if m1:
            last_sha1 = m1.group(1)
            continue
        if ln.startswith('#') or not ln.strip():
            continue
        ma = E_A.match(ln) or E_A2.match(ln)
        if ma:
            ents.append(('md5', ma.group(1), prefix + ma.group(3).strip()))
            last_sha1 = None
            continue
        s2 = ln.strip()
        if s2.startswith(ROOTS):
            ents.append(('sha1' if last_sha1 else 'path', last_sha1, s2))
            last_sha1 = None
    return ents


rows = []
for m in mans:
    try:
        txt = open(lp(m), 'rb').read().decode('utf-8', 'replace')
    except Exception as ex:
        rows.append(dict(manifest=m, cls=R[m]['cls'], error=str(ex)[:60]))
        continue
    ents = parse(txt)
    exact = differs = absent = unk = 0
    absent_list = []
    differs_list = []
    absent_inE = 0
    for kind, ref, path in ents:
        if path in tip:
            if kind == 'sha1':
                if tip[path] == ref:
                    exact += 1
                else:
                    differs += 1
                    differs_list.append(path)
            elif kind == 'md5':
                data = blob(tip[path])
                a_ = hashlib.md5(data).hexdigest() if data is not None else None
                b_ = hashlib.md5(data.replace(bytes([13, 10]), bytes([10]))).hexdigest() if data is not None else None
                if ref in (a_, b_):
                    exact += 1
                else:
                    differs += 1
                    differs_list.append(path)
            else:
                unk += 1   # path only: on tip, content not checkable
        else:
            absent += 1
            absent_list.append(path)
            if os.path.exists(lp(path)):
                absent_inE += 1
    rows.append(dict(manifest=m, cls=R[m]['cls'], entries=len(ents), tip_exact=exact, tip_differs=differs,
                     on_tip_unchecked=unk, not_on_tip=absent, not_on_tip_present_in_E=absent_inE,
                     not_on_tip_paths=absent_list, differs_paths=differs_list))
cat.stdin.close()
json.dump(rows, open(W + '/landing_ready_xref.json', 'w'), indent=1)
for r in rows:
    if 'error' in r:
        print('ERR', r['manifest'][-80:], r['error'])
        continue
    print(r['cls'], r['entries'], 'exact', r['tip_exact'], 'differs', r['tip_differs'], 'unchk', r['on_tip_unchecked'], 'absent', r['not_on_tip'], 'absentInE', r['not_on_tip_present_in_E'], r['manifest'][-80:])
