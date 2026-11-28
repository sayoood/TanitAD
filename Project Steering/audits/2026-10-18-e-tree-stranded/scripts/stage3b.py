"""S2b CANDIDATE-EDIT diff summaries (read-only)."""
import os, json, subprocess, difflib, collections, time

OUT = 'C:/Users/Admin/estrand/out'
W = OUT + '/work'
E = 'E:/Projects/TanitAD'
TIP = '780a025558d29102bf5f15ba695e5b376c64e301'
EHEAD = '37645fcc61b158a327e6597e64e0abb64dcbc46d'
MIRROR = 'C:/Users/Admin/tanitad-push'
BS = chr(92)


def lp(path):
    fp = E + '/' + path
    if len(fp) > 240:
        fp = BS + BS + '?' + BS + fp.replace('/', BS)
    return fp


R = json.load(open(W + '/classes_raw.json'))['results']
s2b = sorted(p for p, r in R.items() if r['cls'] == 'S2b')
print('S2b', len(s2b))

cat = subprocess.Popen(['git', '-C', MIRROR, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def show(spec):
    cat.stdin.write((spec + '\n').encode('utf-8', 'surrogateescape'))
    cat.stdin.flush()
    hdr = cat.stdout.readline().split()
    if len(hdr) < 3 or hdr[1] != b'blob':
        return None
    n = int(hdr[2])
    data = cat.stdout.read(n)
    cat.stdout.read(1)
    return data


def lines(b):
    return b.decode('utf-8', 'replace').replace('\r', '').split('\n')


def stat(a, b):
    """lines added / removed going a -> b (CR-normalised)."""
    la, lb = lines(a), lines(b)
    if len(la) + len(lb) > 400000:
        sa, sb = collections.Counter(la), collections.Counter(lb)
        return sum((sb - sa).values()), sum((sa - sb).values())
    add = rem = 0
    sm = difflib.SequenceMatcher(None, la, lb, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('replace', 'delete'):
            rem += i2 - i1
        if tag in ('replace', 'insert'):
            add += j2 - j1
    return add, rem


def lastchange(path):
    r = subprocess.run(['git', '-C', MIRROR, 'log', '-1', '--format=%ct %h', TIP, '--', path], capture_output=True)
    t = r.stdout.decode().split()
    return (int(t[0]), t[1]) if t else (None, None)


rows = []
for pth in s2b:
    r = R[pth]
    eb = open(lp(pth), 'rb').read()
    tb = show(TIP + ':' + pth)
    bb = show(EHEAD + ':' + pth)
    row = dict(path=pth, origins=r['origins'], e_size=len(eb), tip_size=None if tb is None else len(tb),
               base_size=None if bb is None else len(bb), e_mtime=r['mtime'])
    isbin = b'\0' in eb[:8192]
    row['binary'] = isbin
    row['tip_eq_base'] = (tb is not None and bb is not None and tb == bb)
    ct, ch = lastchange(pth)
    row['tip_last_change_ts'] = ct
    row['tip_last_change'] = ch
    row['e_newer_than_tip_change'] = (ct is not None and r['mtime'] > ct)
    if not isbin and tb is not None and len(eb) < 8_000_000 and len(tb) < 8_000_000:
        a, d = stat(tb, eb)       # tip -> E
        row['vs_tip_add'], row['vs_tip_rem'] = a, d
        row['eol_only'] = (a == 0 and d == 0)
        if bb is not None:
            a2, d2 = stat(bb, eb)  # E-HEAD base -> E
            row['vs_base_add'], row['vs_base_rem'] = a2, d2
            a3, d3 = stat(bb, tb)  # E-HEAD base -> tip
            row['tip_vs_base_add'], row['tip_vs_base_rem'] = a3, d3
    # verdict
    if row.get('eol_only'):
        v = 'EOL-ONLY (identical modulo line endings) -> treat as S1'
    elif row.get('vs_tip_add') == 0 and row.get('vs_tip_rem', 0) > 0:
        v = 'LIKELY-STALE (E is a strict line-subset of the tip: tip only adds)'
    elif row['tip_eq_base'] and not row.get('eol_only') and bb is not None:
        v = 'REAL-LOCAL-EDIT (tip unchanged since E-HEAD base; E has uncommitted edits)'
    elif bb is not None and tb is not None and not row['tip_eq_base']:
        v = 'CONFLICT (tip and E both moved off the E-HEAD base)'
    elif tb is None:
        v = 'NO-TIP-BLOB?'
    else:
        v = 'NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks)'
    row['verdict'] = v
    rows.append(row)
cat.stdin.close()
json.dump(rows, open(W + '/s2b_diffs.json', 'w'), indent=1)
print(collections.Counter(r['verdict'] for r in rows))
for r in rows:
    print(r['verdict'][:16], r.get('vs_tip_add'), r.get('vs_tip_rem'), '|base', r.get('vs_base_add'), r.get('vs_base_rem'), '|tipvsbase', r.get('tip_vs_base_add'), r.get('tip_vs_base_rem'), r['path'][-70:])
