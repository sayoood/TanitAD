"""E: working-tree stranded-work audit -- classifier.

Read-only against E: (plumbing with --no-optional-locks, file reads only).
Classes: S1 identical to tip, S2a stale (E blob is a historical version of the path),
S2b CANDIDATE-EDIT, S3 absent from tip but path in history (S3a stale / S3b content not a
historical version of that path), S4 absent from tip AND from path history (S4m MOVED:
same content on tip under another path; S4h same content in history under another path;
S4 STRANDED candidate).
"""
import os, sys, json, hashlib, subprocess, collections, re, time

OUT = 'C:/Users/Admin/estrand/out'
W = OUT + '/work'
E = 'E:/Projects/TanitAD'
TIP = '780a025558d29102bf5f15ba695e5b376c64e301'
GE = ['git', '--no-optional-locks', '-c', 'safe.directory=*', '-c', 'core.autocrlf=false', '-C', E]
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
BULK = 20 * 1024 * 1024


def p(*a):
    print(*a, flush=True)


def zsplit(b):
    parts = b.split(b'\0')
    if parts and parts[-1] == b'':
        parts.pop()
    return parts


def dec(b):
    return b.decode('utf-8', 'surrogateescape')


# ---- tip tree
tip = {}                       # path -> (mode, sha)
tip_sha2paths = collections.defaultdict(list)
for rec in zsplit(open(W + '/tiptree.bin', 'rb').read()):
    meta, path = rec.split(b'\t', 1)
    mode, typ, sha = meta.split(b' ')
    path = dec(path)
    tip[path] = (dec(mode), dec(sha))
    tip_sha2paths[dec(sha)].append(path)
p('tip files', len(tip))

# ---- history
hist_path_shas = collections.defaultdict(set)   # path -> {blob shas that path ever had}
hist_sha2path = {}                              # blob sha -> a path where it appeared
hist_deleted_paths = set()
ZERO = '0' * 40
raw = open(W + '/histraw.bin', 'rb').read()
ncommit = 0
for chunk in raw.split(b'\x01'):
    if not chunk:
        continue
    ncommit += 1
    # header: <sha>\0 then raw records
    head, _, rest = chunk.partition(b'\0')
    toks = rest.split(b'\0')
    if toks and toks[0].startswith(b'\n'):
        toks[0] = toks[0][1:]
    i = 0
    while i < len(toks) - 1:
        meta = toks[i]
        if not meta.startswith(b':'):
            i += 1
            continue
        path = dec(toks[i + 1])
        f = dec(meta).split(' ')
        oldsha, newsha, st = f[2], f[3], f[4][0]
        if newsha != ZERO:
            hist_path_shas[path].add(newsha)
            hist_sha2path.setdefault(newsha, path)
        else:
            hist_deleted_paths.add(path)
        if oldsha != ZERO:
            hist_path_shas[path].add(oldsha)
            hist_sha2path.setdefault(oldsha, path)
        i += 2
p('hist commits parsed', ncommit, 'paths', len(hist_path_shas), 'blobs', len(hist_sha2path))

# ---- E: lists
def run(args):
    return subprocess.run(args, capture_output=True, env=ENV, check=True).stdout

staged = {}   # path -> status
d = zsplit(run(GE + ['diff', '--cached', '--name-status', '-z', '--no-renames']))
for i in range(0, len(d), 2):
    staged[dec(d[i + 1])] = dec(d[i])
idx = {}      # path -> index blob sha (stage 0)
for rec in zsplit(run(GE + ['ls-files', '-s', '-z'])):
    meta, path = rec.split(b'\t', 1)
    mode, sha, stg = dec(meta).split(' ')
    if stg == '0':
        idx[dec(path)] = (mode, sha)
unstaged_mod = set(dec(x) for x in zsplit(run(GE + ['ls-files', '-m', '-z'])))
unstaged_del = set(dec(x) for x in zsplit(run(GE + ['ls-files', '-d', '-z'])))
untracked = [dec(x) for x in zsplit(run(GE + ['ls-files', '-o', '--exclude-standard', '-z']))]
p('staged', len(staged), 'unstaged_mod', len(unstaged_mod), 'unstaged_del', len(unstaged_del), 'untracked', len(untracked))

items = {}   # path -> origin set
def add(path, origin):
    items.setdefault(path, set()).add(origin)
for pth, st in staged.items():
    add(pth, 'staged-' + st)
for pth in unstaged_mod:
    add(pth, 'unstaged-M')
for pth in unstaged_del:
    add(pth, 'unstaged-D')
for pth in untracked:
    add(pth, 'untracked')
OUTNAME = '/classes_raw.json'
if '--ignored' in sys.argv:
    items = {}
    for ln in open(W + '/ignored_files.txt', encoding='utf-8').read().splitlines():
        if ln:
            add(ln, 'ignored')
    OUTNAME = '/classes_ignored.json'
p('distinct paths', len(items))


def gsha(data_len, h):
    return h.hexdigest()


def blob_hashes(path):
    """return dict(size, raw, lf, crlf) -- blob ids of worktree bytes."""
    fp = E + '/' + path
    if len(fp) > 240:
        fp = '\\\\?\\' + fp.replace('/', '\\')
    try:
        st = os.stat(fp)
    except OSError as ex:
        return dict(missing=True, err=str(ex)[:80])
    if not os.path.isfile(fp):
        return dict(missing=True, err='notfile')
    size = st.st_size
    res = dict(size=size, mtime=int(st.st_mtime))
    try:
        if size > 64 * 1024 * 1024:
            h = hashlib.sha1(b'blob %d\0' % size)
            with open(fp, 'rb') as f:
                while True:
                    b = f.read(8 << 20)
                    if not b:
                        break
                    h.update(b)
            res['raw'] = h.hexdigest()
            return res
        data = open(fp, 'rb').read()
    except OSError as ex:
        return dict(missing=True, err='read:' + str(ex)[:80], size=size)
    res['raw'] = hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
    if b'\0' not in data[:8192]:
        if b'\r\n' in data:
            d2 = data.replace(b'\r\n', b'\n')
            res['lf'] = hashlib.sha1(b'blob %d\0' % len(d2) + d2).hexdigest()
        elif b'\n' in data and b'\r' not in data:
            d2 = data.replace(b'\n', b'\r\n')
            res['crlf'] = hashlib.sha1(b'blob %d\0' % len(d2) + d2).hexdigest()
    return res


def variants(h):
    return [(k, h[k]) for k in ('raw', 'lf', 'crlf') if k in h]


def classify(path):
    origins = sorted(items[path])
    h = blob_hashes(path)
    r = dict(origins=origins, **h)
    if 'staged-A' in items[path] or 'staged-M' in items[path]:
        r['idx_sha'] = idx.get(path, (None, None))[1]
    if h.get('missing'):
        r['cls'] = 'MISSING-WT'
        isha = r.get('idx_sha')
        if isha:
            r['idx_on_tip'] = isha in tip_sha2paths
            r['idx_in_hist'] = isha in hist_sha2path
        return r
    vs = variants(h)
    on_tip = path in tip
    r['on_tip'] = on_tip
    if on_tip:
        tsha = tip[path][1]
        m = [k for k, v in vs if v == tsha]
        if m:
            r['cls'] = 'S1'
            r['match'] = m[0]
        else:
            ph = hist_path_shas.get(path, set())
            m2 = [k for k, v in vs if v in ph]
            if m2:
                r['cls'] = 'S2a'
                r['match'] = m2[0]
            else:
                r['cls'] = 'S2b'
                el_tip = [tip_sha2paths[v][0] for k, v in vs if v in tip_sha2paths]
                el_hist = [hist_sha2path[v] for k, v in vs if v in hist_sha2path]
                if el_tip:
                    r['elsewhere_tip'] = el_tip[0]
                if el_hist:
                    r['elsewhere_hist'] = el_hist[0]
    else:
        inhist = path in hist_path_shas
        ph = hist_path_shas.get(path, set())
        m_tip = [(k, tip_sha2paths[v][0]) for k, v in vs if v in tip_sha2paths]
        m_hist = [(k, hist_sha2path[v]) for k, v in vs if v in hist_sha2path]
        if inhist:
            m2 = [k for k, v in vs if v in ph]
            r['cls'] = 'S3a' if m2 else 'S3b'
            if m2:
                r['match'] = m2[0]
            if m_tip:
                r['moved_to_tip'] = m_tip[0][1]
            elif m_hist:
                r['in_hist_as'] = m_hist[0][1]
        else:
            if m_tip:
                r['cls'] = 'S4m'
                r['moved_to_tip'] = m_tip[0][1]
            elif m_hist:
                r['cls'] = 'S4h'
                r['in_hist_as'] = m_hist[0][1]
            else:
                r['cls'] = 'S4'
    return r


t0 = time.time()
if len(sys.argv) > 1 and sys.argv[1] == '--redo-missing':
    results = json.load(open(W + '/classes_raw.json'))['results']
    for path in [q for q, r in results.items() if r['cls'] == 'MISSING-WT']:
        results[path] = classify(path)
else:
    results = {}
    for n, path in enumerate(sorted(items), 1):
        if n % 1000 == 0:
            p('hashed', n, 'of', len(items), round(time.time() - t0), 's')
        results[path] = classify(path)

p('hash done', round(time.time() - t0), 's')
json.dump(dict(tip=TIP, results=results), open(W + OUTNAME, 'w'))
c = collections.Counter(r['cls'] for r in results.values())
p(dict(c))
