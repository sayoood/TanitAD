"""Stage 4: final recommendations + AUDIT.md / classes.json / s4_list.csv / candidate_edits / paths.json."""
import os, sys, json, re, time, subprocess, collections, difflib, hashlib, shutil, datetime, csv

sys.path.insert(0, 'C:/Users/Admin/estrand/out/work')
import scanlib

OUT = 'C:/Users/Admin/estrand/out'
W = OUT + '/work'
E = 'E:/Projects/TanitAD'
TIP = '780a025558d29102bf5f15ba695e5b376c64e301'
EHEAD = '37645fcc61b158a327e6597e64e0abb64dcbc46d'
MIRROR = 'C:/Users/Admin/tanitad-push'
PKG = 'Project Steering/audits/2026-10-18-e-tree-stranded'
PKGDIR = OUT + '/' + PKG
BS = chr(92)
MB = 1024 * 1024


def lp(path):
    fp = E + '/' + path
    if len(fp) > 240:
        fp = BS + BS + '?' + BS + fp.replace('/', BS)
    return fp


def iso(ts):
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else ''


def zsplit(b):
    parts = b.split(b'\0')
    if parts and parts[-1] == b'':
        parts.pop()
    return parts


# --------------------------------------------------------------- inputs
R = json.load(open(W + '/R_refined.json'))
RI = json.load(open(W + '/classes_ignored.json'))['results']
S2B = {r['path']: r for r in json.load(open(W + '/s2b_diffs.json'))}
LRX = json.load(open(W + '/landing_ready_xref.json'))

tip = {}
for rec in zsplit(open(W + '/tiptree.bin', 'rb').read()):
    meta, path = rec.split(b'\t', 1)
    tip[path.decode('utf-8', 'surrogateescape')] = meta.split(b' ')[2].decode()
tip_sha2paths = collections.defaultdict(list)
for pth, sha in tip.items():
    tip_sha2paths[sha].append(pth)

# history sha set (for index-blob check)
ZERO = '0' * 40
hist_sha2path = {}
for chunk in open(W + '/histraw.bin', 'rb').read().split(b'\x01'):
    if not chunk:
        continue
    _, _, rest = chunk.partition(b'\0')
    toks = rest.split(b'\0')
    if toks and toks[0].startswith(b'\n'):
        toks[0] = toks[0][1:]
    i = 0
    while i < len(toks) - 1:
        if toks[i].startswith(b':'):
            f = toks[i].decode().split(' ')
            for s in (f[2], f[3]):
                if s != ZERO:
                    hist_sha2path.setdefault(s, toks[i + 1].decode('utf-8', 'surrogateescape'))
            i += 2
        else:
            i += 1

# tip file counts per directory prefix (for "package present on tip")
def tip_count(prefix):
    return sum(1 for p in tip if p.startswith(prefix))


# --------------------------------------------------------------- recommendations
BULK_EXT = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.mp4', '.mov', '.avi', '.mkv', '.webm', '.zip', '.tar',
            '.7z', '.h5', '.hdf5', '.parquet', '.onnx', '.engine', '.trt', '.wav', '.mp3', '.gif', '.npy'}
SCRATCH_RX = re.compile(r'(/_scratch/|/__pycache__/|\.pyc$|/\.pytest_cache/|\.tmp$|\.bak$|~$|/\.DS_Store$|/Thumbs\.db$)')
SCRATCH_NAMES = re.compile(r'(^|/)(launch_pids[^/]*\.json|[^/]*_lock\.json|[^/]*\.pid|[^/]*\.lock)$')
CODE_DOC_EXT = {'.md', '.py', '.sh', '.ps1', '.diff', '.patch', '.txt', '.md5', '.bat', '.cmd', '.yaml', '.yml', '.toml'}


def recommend(path, r):
    ext = r.get('ext') or (os.path.splitext(path)[1].lower())
    size = r['size']
    fl = r.get('flags') or []
    if SCRATCH_RX.search(path) or SCRATCH_NAMES.search(path):
        return 'DROP-scratch', None, 'scratch/lock/cache path'
    if ext == '.done' and size <= 16:
        return 'DROP-scratch', None, 'completion marker whose payload (media segment) is not landable'
    if ext in BULK_EXT:
        return 'BULK-ONLY', None, 'media/checkpoint/array format %s' % ext
    if size > 20 * MB:
        return 'BULK-ONLY', None, 'size>20MB'
    if ext in ('.png', '.jpg', '.jpeg') and path.startswith('Media/'):
        return 'BULK-ONLY', None, 'media (Media/ keyframes)'
    if ext in ('.png', '.jpg', '.jpeg', '.svg') and size > 2 * MB:
        return 'BULK-ONLY', None, 'image>2MB'
    sec = [f for f in fl if f != 'private_ip_ctx' and f != 'private_ip_port']
    if sec:
        return 'LAND-HOLD-SCRUB', None, 'flags: ' + ','.join(sec)
    if size <= 1 * MB and (ext in CODE_DOC_EXT or ext in ('.json', '.csv', '.thor', '.out', '.stdout', '.log', '.err')):
        tier = 'P1'
    elif size <= 5 * MB:
        tier = 'P2'
    else:
        return 'LAND-LARGE', 'P3', 'landable by exception: 5-20 MB raw data (separate commit; not in paths.json)'
    return 'LAND', tier, ''


# ---- S4 (incl. ignored S4) --------------------------------------------------
S4 = {}
for pth, r in R.items():
    if r['cls'] == 'S4':
        S4[pth] = dict(r, src='worktree')
for pth, r in RI.items():
    if r['cls'] == 'S4':
        r = dict(r)
        ext = os.path.splitext(pth)[1].lower()
        r['ext'] = ext
        r['flags'] = []
        r['desc'] = ''
        S4[pth] = dict(r, src='gitignored')

# scan + describe the ignored S4 text files that can matter (not the 2k jpgs): small & text only
for pth, r in S4.items():
    if r['src'] != 'gitignored':
        continue
    ext = r['ext']
    if ext in ('.html', '.json', '.log', '.svg', '.js') and r['size'] <= 20 * MB:
        try:
            data = open(lp(pth), 'rb').read()
            r['flags'] = scanlib.scan_bytes(data)
        except Exception:
            pass

for pth, r in S4.items():
    rec, tier, why = recommend(pth, r)
    if r['src'] == 'gitignored':
        # ignored by E's own .gitignore on purpose: derived report output / Media replay frames
        if rec in ('LAND', 'LAND-LARGE', 'LAND-HOLD-SCRUB'):
            rec, tier, why = 'GITIGNORED-DERIVED', None, "ignored by E's .gitignore (report/ or data/ rule); regenerable output - lander would need add -f"
        r['rec'], r['tier'], r['why'] = rec, tier, why
    else:
        r['rec'], r['tier'], r['why'] = rec, tier, why

# final gate: every LAND file is (re)scanned with scanlib (a superset of the stage2 patterns), cached by blob id
cache_p = W + '/final_scan_cache.json'
cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
n_scanned = 0
for pth, r in sorted(S4.items()):
    if r['rec'] not in ('LAND', 'LAND-LARGE') or r['size'] > 20 * MB:
        continue
    key = r['raw']
    if key not in cache:
        try:
            cache[key] = scanlib.scan_bytes(open(lp(pth), 'rb').read())
        except Exception as ex:
            cache[key] = ['scan_error']
        n_scanned += 1
    fl = [f for f in cache[key] if f not in ('private_ip_ctx', 'private_ip_port')]
    if fl:
        r['rec'], r['tier'], r['why'] = 'LAND-HOLD-SCRUB', None, 'final-gate flags: ' + ','.join(fl)
        r['flags'] = sorted(set((r.get('flags') or []) + fl))
json.dump(cache, open(cache_p, 'w'))
print('final gate scanned', n_scanned, 'new files')


def scrub(data):
    """mechanical redaction of the scan classes; returns (new_bytes, {class: n})"""
    n = collections.Counter()

    def sub_ipport(m):
        if scanlib.ip_class(m.group(1)) == 'public':
            n['host:port'] += 1
            return b'<redacted-host:port>'
        return m.group(0)
    data = scanlib.IPPORT.sub(sub_ipport, data)

    def sub_ipctx(m):
        if scanlib.ip_class(m.group(1)) == 'public':
            n['ip'] += 1
            return m.group(0)[:m.start(1) - m.start(0)] + b'<redacted-ip>' + m.group(0)[m.end(1) - m.start(0):]
        return m.group(0)
    data = scanlib.IPCTX.sub(sub_ipctx, data)

    def sub_sshp(m):
        n['ssh-port'] += 1
        return re.sub(rb'(-[pP])\s*\d{3,5}', rb'\1 <port>', m.group(0))
    data = scanlib.PATS['ssh_port'].sub(sub_sshp, data)
    data, k = scanlib.PATS['tokenfile_path'].subn(b'<token-file>', data)
    n['token-file'] += k
    data, k = scanlib.PATS['runpod_ssh'].subn(b'<pod-ssh-host>', data)
    n['pod-host'] += k
    return data, {a: b for a, b in n.items() if b}


SCRUBBED = {}
for pth, r in sorted(S4.items()):
    if r['rec'] != 'LAND-HOLD-SCRUB' or r['size'] > 20 * MB:
        continue
    try:
        new, counts = scrub(open(lp(pth), 'rb').read())
    except Exception as ex:
        r['scrub_error'] = str(ex)[:60]
        continue
    left = [f for f in scanlib.scan_bytes(new) if f not in ('private_ip_ctx', 'private_ip_port')]
    if left:
        r['scrub_left'] = left
        continue
    outp = OUT + '/scrubbed/' + pth
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    open(outp, 'wb').write(new)
    SCRUBBED[pth] = dict(src=outp, redactions=counts)
    r['scrubbed'] = True
    r['redactions'] = counts

# recent in-flight
for pth, r in S4.items():
    r['recent24h'] = (time.time() - r['mtime']) < 86400 if 'mtime' in r else False

CODE_PATCHES = []
_rx1 = re.compile(r'/code/[^/]+/((?:stack|taniteval|tools)/.+)$')
_rx2 = re.compile(r'/((?:stack|taniteval|tools)/(?:tanitad|scripts|tests|taniteval|tools|ops|refs)[^ ]*)$')
for pth, r in S4.items():
    if r['src'] != 'worktree' or not pth.endswith(('.py', '.sh', '.ps1', '.json', '.diff', '.patch')):
        continue
    if '/code/' in pth or '/stack/' in pth:
        m_ = _rx1.search(pth) or _rx2.search(pth)
        if m_:
            CODE_PATCHES.append((pth, m_.group(1), m_.group(1) in tip))

# ---- companions: weak-moved / in-history files that live in a dir where we land S4 files -----------
land_dirs = collections.defaultdict(int)
for pth, r in S4.items():
    if r['rec'] == 'LAND':
        land_dirs[os.path.dirname(pth)] += 1
COMP = {}
for pth, r in R.items():
    if r['cls'] in ('S4m', 'S4h') and not r.get('dup_strong'):
        if land_dirs.get(os.path.dirname(pth)):
            rec, tier, why = recommend(pth, dict(r, ext=os.path.splitext(pth)[1].lower(), flags=r.get('flags', [])))
            if rec == 'LAND':
                COMP[pth] = dict(r, rec='COMPANION', tier=tier)

# ---- MISSING-WT / index-only blobs --------------------------------------------------------
miss = {p: r for p, r in R.items() if r['cls'] == 'MISSING-WT'}
miss_out = []
for pth, r in miss.items():
    isha = r.get('idx_sha')
    same_in_E = [q for q, rr in R.items() if isha and rr.get('raw') == isha and q != pth]
    entry = dict(path=pth, origins=r['origins'], idx_sha=isha, moved_in_E=same_in_E[:3], idx_on_tip=bool(isha and isha in tip_sha2paths),
                 idx_in_hist=bool(isha and isha in hist_sha2path))
    if isha:
        try:
            b = subprocess.run(['git', '--no-optional-locks', '-c', 'safe.directory=*', '-C', E, 'cat-file', 'blob', isha],
                               capture_output=True, check=True, env=dict(os.environ, GIT_OPTIONAL_LOCKS='0')).stdout
            entry['idx_blob_size'] = len(b)
            entry['idx_blob_scan'] = scanlib.scan_bytes(b)
            if not entry['idx_on_tip'] and not entry['idx_in_hist']:
                outp = PKGDIR + '/recovered_index_blobs/' + isha + '.blob'
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                open(outp, 'wb').write(b)
                entry['recovered_to'] = PKG + '/recovered_index_blobs/' + isha + '.blob'
        except Exception as ex:
            entry['idx_blob_error'] = str(ex)[:80]
    miss_out.append(entry)

# ---- index blob vs worktree mismatch among staged entries ---------------------------------------
idx_mismatch = []
for pth, r in R.items():
    isha = r.get('idx_sha')
    if not isha or r['cls'] == 'MISSING-WT':
        continue
    wt = {r.get('raw'), r.get('lf'), r.get('crlf')}
    if isha not in wt:
        idx_mismatch.append(dict(path=pth, cls=r['cls'], idx_on_tip=isha in tip_sha2paths, idx_in_hist=isha in hist_sha2path))

# --------------------------------------------------------------- counts
CLS = ['S1', 'S2a', 'S2b', 'S3', 'S4', 'S4m', 'S4h', 'MISSING-WT']
cnt = collections.Counter(r['cls'] for r in R.values())
by_top = collections.defaultdict(collections.Counter)
for pth, r in R.items():
    by_top[pth.split('/')[0]][r['cls']] += 1
by_origin = collections.defaultdict(collections.Counter)
for pth, r in R.items():
    by_origin['+'.join(r['origins'])][r['cls']] += 1
s1_match = collections.Counter(r.get('match') for r in R.values() if r['cls'] == 'S1')
s4m_strong = sum(1 for r in R.values() if r['cls'] == 'S4m' and r.get('dup_strong'))
s4m_weak = sum(1 for r in R.values() if r['cls'] == 'S4m' and not r.get('dup_strong'))
s4h_strong = sum(1 for r in R.values() if r['cls'] == 'S4h' and r.get('dup_strong'))
s4h_weak = sum(1 for r in R.values() if r['cls'] == 'S4h' and not r.get('dup_strong'))

staged_absent = open('C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/c216c849-5fe7-4d43-a3be-32429241f811/scratchpad/e_staged_absent.txt', encoding='utf-8').read().splitlines()
staged_absent = [x for x in staged_absent if x.strip()]
sa_cls = collections.Counter(R[x]['cls'] if x in R else 'NOT-IN-R' for x in staged_absent)
sa_s4 = [x for x in staged_absent if x in R and R[x]['cls'] == 'S4']

rec_cnt = collections.Counter()
rec_bytes = collections.Counter()
for pth, r in S4.items():
    k = r['rec'] + ('/' + r['tier'] if r.get('tier') else '')
    rec_cnt[k] += 1
    rec_bytes[k] += r['size']

# per-package rollup
def pkg_of(path):
    parts = path.split('/')
    if parts[0] == 'FlyWheels' and len(parts) > 3 and parts[2] == 'incoming':
        return '/'.join(parts[:4])
    if parts[0] == 'TanitAD Research Lab' and len(parts) > 3 and parts[2] in ('Research', 'Implementation'):
        return '/'.join(parts[:4])
    if parts[0] == 'products' and len(parts) > 2:
        return '/'.join(parts[:3])
    if parts[0] == 'taniteval' and len(parts) > 3 and parts[1] == 'results':
        return '/'.join(parts[:4])
    if parts[0] in ('Media', 'colab') and len(parts) > 2:
        return '/'.join(parts[:3])
    if len(parts) > 2:
        return '/'.join(parts[:2])
    return parts[0]


pk = collections.defaultdict(lambda: collections.Counter())
pkb = collections.defaultdict(lambda: collections.Counter())
for pth, r in S4.items():
    key = pkg_of(pth)
    k = r['rec'] + ('/' + r['tier'] if r.get('tier') else '')
    pk[key][r['rec']] += 1
    pkb[key][r['rec']] += r['size']
    pk[key]['_n'] += 1
    pkb[key]['_n'] += r['size']

# --------------------------------------------------------------- candidate-edit outputs
os.makedirs(PKGDIR + '/candidate_edits', exist_ok=True)
cat = subprocess.Popen(['git', '-C', MIRROR, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def show(spec):
    cat.stdin.write((spec + '\n').encode('utf-8', 'surrogateescape'))
    cat.stdin.flush()
    hdr = cat.stdout.readline().split()
    if len(hdr) < 3 or hdr[1] != b'blob':
        return None
    n = int(hdr[2])
    d = cat.stdout.read(n)
    cat.stdout.read(1)
    return d


def e_only_lines(tb, eb):
    la = tb.decode('utf-8', 'replace').replace('\r', '').split('\n')
    lb = eb.decode('utf-8', 'replace').replace('\r', '').split('\n')
    out = []
    sm = difflib.SequenceMatcher(None, la, lb, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('replace', 'insert'):
            for j in range(j1, j2):
                out.append((j + 1, lb[j]))
    return out


s2b_rows = []
for pth in sorted(S2B):
    d = S2B[pth]
    r = R[pth]
    row = dict(d)
    row['flags'] = r.get('flags', [])
    if row['flags']:
        _tb = show(TIP + ':' + pth)
        row['tip_flags'] = scanlib.scan_bytes(_tb) if _tb is not None else []
    row['desc'] = r.get('desc', '')
    slug = hashlib.sha1(pth.encode()).hexdigest()[:8] + '_' + re.sub(r'[^A-Za-z0-9._-]+', '_', os.path.basename(pth))[:60]
    row['e_only_file'] = None
    # machine-local settings: never land
    if pth.startswith('.claude/'):
        row['verdict2'] = 'MACHINE-LOCAL (do not land)'
    elif d.get('binary') or d.get('vs_tip_add') in (None, 0):
        row['verdict2'] = d['verdict']
    else:
        tb = show(TIP + ':' + pth)
        eb = open(lp(pth), 'rb').read()
        lines = e_only_lines(tb, eb)
        txt = '# E-only lines of %s  (E lines absent from the tip blob; CR-normalised; E line numbers)\n' % pth
        txt += '# tip blob last changed in %s; E mtime %s; verdict: %s\n' % (d.get('tip_last_change'), iso(r['mtime']), d['verdict'])
        txt += '\n'.join('%d: %s' % (n, l) for n, l in lines) + '\n'
        fl = scanlib.scan_bytes(txt.encode('utf-8', 'replace'))
        if fl:
            row['verdict2'] = d['verdict'] + '  [E-only lines FLAGGED %s: not exported]' % ','.join(fl)
        else:
            open(PKGDIR + '/candidate_edits/' + slug + '.E_only_lines.txt', 'w', encoding='utf-8', newline='\n').write(txt)
            row['e_only_file'] = PKG + '/candidate_edits/' + slug + '.E_only_lines.txt'
            row['verdict2'] = d['verdict']
    ts = d.get('tip_last_change_ts')
    if ts:
        row['tip_last_change_date'] = iso(ts)
        row['hint'] = 'E OLDER than tip last change' if r['mtime'] < ts else 'E NEWER than tip last change'
    s2b_rows.append(row)
cat.stdin.close()

# --------------------------------------------------------------- write classes.json
def compact(pth, r):
    o = dict(c=r['cls'], o='+'.join(r['origins']), s=r.get('size'), m=iso(r.get('mtime')))
    for k in ('rec', 'tier', 'why', 'desc', 'dup_dst', 'dup_strong', 'moved_to_tip', 'in_hist_as', 'flags', 'recent24h', 'src'):
        if r.get(k):
            o[k] = r[k]
    return o


entries = {}
for pth, r in R.items():
    if r['cls'] == 'S1':
        continue
    if pth in S4:
        entries[pth] = compact(pth, S4[pth])
    elif pth in COMP:
        entries[pth] = compact(pth, COMP[pth])
    else:
        r2 = dict(r)
        if r['cls'] in ('S4m', 'S4h'):
            r2['rec'] = 'NO-ACTION' if r.get('dup_strong') else 'NO-ACTION (content on tip/history at dup_dst)'
        elif r['cls'] == 'S2a':
            r2['rec'] = 'NO-ACTION (stale: E blob is a historical version of this path)'
        elif r['cls'] == 'S2b':
            r2['rec'] = 'CANDIDATE-EDIT'
        elif r['cls'] == 'MISSING-WT':
            r2['rec'] = 'SEE-MISSING-WT'
        entries[pth] = compact(pth, r2)
for pth, r in S4.items():
    if r['src'] == 'gitignored':
        entries[pth] = compact(pth, r)
meta = dict(tip=TIP, e_head=EHEAD, generated=datetime.datetime.now().isoformat(timespec='seconds'),
            scope='staged(6654) + unstaged-modified + untracked (--exclude-standard) = %d distinct paths; plus %d gitignored non-mp4 files (+119 mp4 listed as bulk)' % (len(R), len(RI)),
            counts=dict(cnt), counts_ignored=dict(collections.Counter(r['cls'] for r in RI.values())),
            s1_match_variants=dict(s1_match),
            recommendations={k: dict(n=rec_cnt[k], bytes=rec_bytes[k]) for k in rec_cnt})
json.dump(dict(meta=meta, by_top_level=by_top, by_origin=by_origin, entries=entries,
               s2b=s2b_rows, missing_worktree=miss_out, index_blob_mismatch=idx_mismatch,
               landing_ready_xref=[{k: v for k, v in x.items() if k not in ('not_on_tip_paths', 'differs_paths')} for x in LRX]),
          open(PKGDIR + '/classes.json', 'w', encoding='utf-8'), indent=0, sort_keys=True, ensure_ascii=False)

# s4_list.csv
with open(PKGDIR + '/s4_list.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['path', 'bytes', 'mtime', 'src', 'recommendation', 'tier', 'why_or_flags', 'one_line_description'])
    for pth in sorted(S4):
        r = S4[pth]
        w.writerow([pth, r['size'], iso(r.get('mtime')), r['src'], r['rec'], r.get('tier') or '',
                    r.get('why') or ','.join(r.get('flags') or []), (r.get('desc') or '').replace('\n', ' ')])
with open(PKGDIR + '/companions.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['path', 'bytes', 'class', 'content_already_at'])
    for pth in sorted(COMP):
        r = COMP[pth]
        w.writerow([pth, r['size'], r['cls'], r.get('dup_dst')])

# --------------------------------------------------------------- AUDIT.md
def mbs(n):
    return '%.1f' % (n / MB)


L = []
A = L.append
A('# E: working-tree stranded-work audit (2026-10-18)')
A('')
A('Tip audited against: `%s` (branch `agent/arch-inf-20260803`, mirror `tanitad-push`). E: checkout HEAD `%s` (2026-09-19), ancestor of the tip.' % (TIP, EHEAD[:8]))
A('Evidence class for every number below: **MEASURED** (ours; `classes.json` and `s4_list.csv` in this package, produced by `scripts/`). Content comparison is by git blob id of the file BYTES (`hash-object --no-filters` equivalent), with a CRLF<->LF variant tried; mtime is used only as a secondary hint and never alone.')
A('')
A('## Headline')
A('')
s4_land = [p for p, r in S4.items() if r['rec'] == 'LAND']
s4_land_b = sum(S4[p]['size'] for p in s4_land)
A('* **%d of %d changed/untracked paths are STRANDED candidates (S4: absent from the tip AND from all %d-commit history, content absent from the tip under any other path)** - %s MB total, mostly raw run outputs and package-local code of EvalFlyWheel / Research-Lab / Data-Audit packages (13 of the 74 `LANDING_READY*` manifests are themselves not on the tip, section 7).' % (cnt['S4'], len(R), 3272, mbs(sum(r['size'] for r in R.values() if r['cls'] == 'S4'))))
A('* **Recommendation split of the S4 set (+ gitignored extras):** LAND (P1+P2, in paths.json) = %d files / %s MB; LAND-LARGE (5-20 MB, optional separate commit) = %d files / %s MB; LAND-HOLD-SCRUB (secret/IP/token-path flag) = %d; BULK-ONLY = %d files / %s MB; DROP-scratch = %d; GITIGNORED-DERIVED = %d.' % (
    rec_cnt['LAND/P1'] + rec_cnt['LAND/P2'], mbs(rec_bytes['LAND/P1'] + rec_bytes['LAND/P2']), rec_cnt['LAND-LARGE/P3'], mbs(rec_bytes['LAND-LARGE/P3']),
    rec_cnt['LAND-HOLD-SCRUB'], rec_cnt['BULK-ONLY'], mbs(rec_bytes['BULK-ONLY']), rec_cnt['DROP-scratch'], rec_cnt['GITIGNORED-DERIVED']))
A('* **%d CANDIDATE-EDITs** (S2b: path is on the tip, E content differs and is not any historical version of the path) - list in section 4; **none landed**.' % cnt['S2b'])
A('* **Most differences are stale or already landed:** %d paths (%.1f %%) are byte-identical to the tip blob (S1), %d are stale (S2a: E blob is a historical version of the path).' % (cnt['S1'], 100.0 * cnt['S1'] / len(R), cnt['S2a']))
A('* **E: and D: are the same volume** (`subst D: => E:\\`, volume serial identical), i.e. E:\\Projects\\TanitAD *is* the canonical D:\\Projects\\TanitAD working tree where agents write; its git HEAD simply stopped moving at 2026-09-19 because the Master Mind lands through the mirror. That is why the tree holds three weeks of agent output that git on E: has never seen.')
A('')
A('## 1. Scope and method')
A('')
A('* Population (MEASURED): staged paths `git diff --cached` = 6,654 (6,590 A + 64 M); unstaged-modified `ls-files -m` = 133 (+1 deleted); untracked `ls-files -o --exclude-standard` = 9,147; union = **%d distinct paths**. All read through plumbing with `--no-optional-locks`; E:\'s index and files were not modified.' % len(R))
A('* Not in the population: tracked files unchanged since E-HEAD (they only lag the tip) and files ignored by E:\'s `.gitignore` - the latter were enumerated separately in section 6 (the ignore rule `data/` hides package `code/fix/**/stack/tanitad/data/*.py`, `taniteval/**/report/` and all mp4/Media replay frames).')
A('* Tip side: %d blobs from `ls-tree -r`; history side: `git log --all --raw --no-renames -m` over 3,272 commit entries -> path -> set of blob ids ever held (%d paths, %d distinct blobs).' % (len(tip), 37804, len(hist_sha2path)))
A('* 75 untracked files exceed Windows MAX_PATH (> 240 chars) and were first reported unreadable by the naive probe - the same "a zero is a claim about the probe" trap as in CLAUDE.md; they were re-read with the `\\\\?\\` prefix and classified (all S4). Paths longer than 259 chars need the same prefix at landing time.')
A('')
A('## 2. Counts per class')
A('')
A('| class | meaning | n | share |')
A('|---|---|---:|---:|')
desc = {'S1': 'identical to tip blob (raw %d / after CRLF<->LF %d)' % (s1_match.get('raw', 0), s1_match.get('lf', 0) + s1_match.get('crlf', 0)),
        'S2a': 'on tip, different; E blob IS a historical version of the path -> STALE',
        'S2b': 'on tip, different; E blob is NOT any historical version -> CANDIDATE-EDIT',
        'S3': 'absent from tip, path in history (deleted/renamed later)',
        'S4': 'absent from tip AND all history, content nowhere on tip -> STRANDED candidate',
        'S4m': 'absent path, same content on tip under another path (MOVED) - strong %d (size>=256 B and same basename), weak %d' % (s4m_strong, s4m_weak),
        'S4h': 'absent path, same content in history under another path - strong %d, weak %d' % (s4h_strong, s4h_weak),
        'MISSING-WT': 'in E index, file gone from worktree'}
for c in CLS:
    A('| %s | %s | %d | %.2f %% |' % (c, desc[c], cnt.get(c, 0), 100.0 * cnt.get(c, 0) / len(R)))
A('| **total** | | **%d** | |' % len(R))
A('')
A('S3 = 0: none of the 15,822 paths is a path the tip later deleted or renamed. The 103 S4h and 734 S4m paths are the only "elsewhere" cases; the weak ones (tiny or different-basename content-identical files, e.g. identical per-run score logs) mean the **content** is preserved on the tip while the **path** is not - they are listed in `companions.csv` when they sit in a directory where S4 files are being landed, so a lander can keep the package layout complete (optional).')
A('')
A('### By origin (staged / unstaged / untracked)')
A('')
A('| origin | ' + ' | '.join(CLS) + ' |')
A('|---|' + '---:|' * len(CLS))
for o in sorted(by_origin):
    A('| %s | ' % o + ' | '.join(str(by_origin[o].get(c, 0)) for c in CLS) + ' |')
A('')
A('### By top-level directory')
A('')
A('| top-level dir | ' + ' | '.join(CLS) + ' |')
A('|---|' + '---:|' * len(CLS))
for t in sorted(by_top, key=lambda x: -sum(by_top[x].values())):
    A('| %s | ' % t + ' | '.join(str(by_top[t].get(c, 0)) for c in CLS) + ' |')
A('')
A('### The 157 staged-`A` paths absent from the tip (brief list: %d lines)' % len(staged_absent))
A('')
A('Classification of the paths in `e_staged_absent.txt`: ' + ', '.join('%s %d' % (k, v) for k, v in sorted(sa_cls.items())) + '.')
A('So %d of them are genuinely stranded (S4) and %d are staged copies whose bytes the tip already holds under another path (typically package `code/fix/stack/...` copies landed at `stack/...`).' % (sa_cls.get('S4', 0), sa_cls.get('S4m', 0)))
if miss_out:
    A('')
    A('One more staged path is `MISSING-WT` (index entry, worktree file deleted): see section 5.')
A('')

A('## 3. S4 - stranded candidates: recommendation per class')
A('')
A('| recommendation | files | MB | rule |')
A('|---|---:|---:|---|')
A('| LAND P1 | %d | %s | text/code/docs/json/csv/log <= 1 MB |' % (rec_cnt['LAND/P1'], mbs(rec_bytes['LAND/P1'])))
A('| LAND P2 | %d | %s | other landable files <= 5 MB |' % (rec_cnt['LAND/P2'], mbs(rec_bytes['LAND/P2'])))
A('| LAND-LARGE (P3) | %d | %s | landable files 5-20 MB: raw data, optional separate commit; NOT in paths.json (`paths_P3_optional.json`) |' % (rec_cnt['LAND-LARGE/P3'], mbs(rec_bytes['LAND-LARGE/P3'])))
A('| LAND-HOLD-SCRUB | %d | %s | scan flag (public IP, ssh port, token-file path); scrub then land - NOT in paths.json |' % (rec_cnt['LAND-HOLD-SCRUB'], mbs(rec_bytes['LAND-HOLD-SCRUB'])))
A('| BULK-ONLY | %d | %s | > 20 MB, or .pt/.npy/.bin/video/media (list only; never land) |' % (rec_cnt['BULK-ONLY'], mbs(rec_bytes['BULK-ONLY'])))
A('| DROP-scratch | %d | %s | `_scratch/` trees, caches, pid/lock files, `.done` markers of non-landable media |' % (rec_cnt['DROP-scratch'], mbs(rec_bytes['DROP-scratch'])))
A('| GITIGNORED-DERIVED | %d | %s | ignored by E:\'s own `.gitignore`; regenerable reports / replay frames |' % (rec_cnt['GITIGNORED-DERIVED'], mbs(rec_bytes['GITIGNORED-DERIVED'])))
A('')
A('`paths.json` contains exactly the LAND set (P1+P2) plus this audit package and the recovered index-only blob; `paths_by_tier.json` splits it so the Master Mind can take P1 only; `paths_P3_optional.json` carries the 5-20 MB LAND-LARGE files. Full per-file list with one-line descriptions: `s4_list.csv`.')
A('')
A('### Rollup per package (S4 files only)')
A('')
A('| package | S4 | LAND | MB landed | LAND-LARGE | MB large | BULK-ONLY | MB bulk | DROP | HOLD | files of this package already on tip |')
A('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for key in sorted(pk, key=lambda k: -pkb[k]['_n']):
    n = pk[key]
    b = pkb[key]
    A('| %s | %d | %d | %s | %d | %s | %d | %s | %d | %d | %d |' % (key, n['_n'], n['LAND'], mbs(b['LAND']), n['LAND-LARGE'], mbs(b['LAND-LARGE']), n['BULK-ONLY'], mbs(b['BULK-ONLY']),
                                                       n['DROP-scratch'] + n['GITIGNORED-DERIVED'], n['LAND-HOLD-SCRUB'], tip_count(key + '/')))
A('')
A('### BULK-ONLY files (list; never land)')
A('')
A('| path | MB | reason |')
A('|---|---:|---|')
bulk_big = sorted([p for p, r in S4.items() if r['rec'] == 'BULK-ONLY' and r['size'] > 5 * MB], key=lambda p: -S4[p]['size'])
for p_ in bulk_big:
    A('| %s | %s | %s |' % (p_, mbs(S4[p_]['size']), S4[p_]['why']))
small_bulk = [p for p, r in S4.items() if r['rec'] == 'BULK-ONLY' and r['size'] <= 5 * MB]
A('')
A('Plus %d smaller BULK-ONLY files (%s MB; .npz/.pkl/.pt/.bin/media-format or Media keyframes) - see `s4_list.csv`.' % (len(small_bulk), mbs(sum(S4[p]['size'] for p in small_bulk))))
A('')
hold = sorted(p for p, r in S4.items() if r['rec'] == 'LAND-HOLD-SCRUB')
A('### Secret / PII flags (never printed - file names and flag classes only)')
A('')
A('Patterns scanned: HF/GitHub/AWS/Slack/OpenAI-style tokens, PEM private keys, `Bearer` values, key=value secret assignments, a token-file path/name, runpod ssh host names, `ssh -p <port>`, public IPv4 (with port, or in ssh/host context). 0 token-class hits anywhere. The 350 first-pass `secret_assign` hits were NavSim scene `token` ids (16-hex), a measured false positive, and are not flagged. Remaining flags (files only):')
A('')
A('| class | file | flags | scrubbed copy (OUT/scrubbed/, listed in paths_scrubbed.json) |')
A('|---|---|---|---|')
for p_ in hold:
    A('| S4 | %s | %s | %s |' % (p_, ','.join(S4[p_]['flags']), ('yes: ' + ', '.join('%s x%d' % kv for kv in sorted(S4[p_]['redactions'].items()))) if S4[p_].get('scrubbed') else 'no (' + ','.join(S4[p_].get('scrub_left', ['not attempted'])) + ')'))
for row in s2b_rows:
    if row['flags']:
        A('| S2b | %s | %s | n/a (tip file; candidate edit) |' % (row['path'], ','.join(row['flags'])))
A('')
A('S2b rows flagged above: scanning the TIP blob of the same path with the same patterns gives -> ' + '; '.join('%s: %s' % (r_['path'].split('/')[-1], ','.join(r_['tip_flags']) or 'clean') for r_ in s2b_rows if r_['flags']) + '. (E-side flags that are absent on the tip are E-only additions and are withheld from `candidate_edits/` when they fall inside the exported lines.)')
A('')
rec24 = [p for p, r in S4.items() if r.get('recent24h')]
A('Files modified in the last 24 h (possibly still being written): %d%s.' % (len(rec24), (' - ' + '; '.join(rec24)) if rec24 else ''))
A('')

A('### S4 package-local code that targets the repo (INTEGRATION DECISION NEEDED)')
A('')
A('S4 `.py/.json` files under a package `code/<variant>/stack|taniteval|tools/...` tree whose target repo path exists on the tip with DIFFERENT content (so landing the package copy does not change the repo code): %d of %d target paths exist on the tip.' % (sum(1 for x in CODE_PATCHES if x[2]), len(CODE_PATCHES)))
A('')
A('| package-local copy (S4) | target path | target on tip |')
A('|---|---|---|')
for src_, tgt_, on_ in sorted(CODE_PATCHES):
    A('| %s | %s | %s |' % (src_, tgt_, 'yes (content differs)' if on_ else 'NO (new file)'))
A('')
A('### S4 decision-grade documents and code outside raw/')
A('')
A('| path | KB | rec | description |')
A('|---|---:|---|---|')
for p_ in sorted(S4):
    r_ = S4[p_]
    if r_['ext'] in ('.md', '.py') and '/raw/' not in p_ and r_['src'] == 'worktree':
        A('| %s | %.1f | %s | %s |' % (p_, r_['size'] / 1024.0, r_['rec'], (r_.get('desc') or '').replace('|', '/')[:110]))
A('')
A('## 4. S2b - CANDIDATE-EDITs (%d). Listed, not landed.' % len(s2b_rows))
A('')
A('Columns: line deltas CR-normalised. `vs tip` = what E has/lacks relative to the tip blob; `vs E-HEAD` = E edits on top of its own HEAD (only for tracked paths); `tip vs E-HEAD` = how far the tip moved off E\'s base. `E-only lines` = the exported file with E lines absent from the tip (the part that would be lost).')
A('')
A('| path | origin | verdict | mtime hint | vs tip (+/-) | vs E-HEAD (+/-) | tip vs E-HEAD (+/-) | E mtime | tip last change | E-only lines file |')
A('|---|---|---|---|---|---|---|---|---|---|')
for row in s2b_rows:
    def pm(a, b):
        return '' if a is None else '+%d/-%d' % (a, b)
    A('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
        row['path'], '+'.join(row['origins']), row['verdict2'], row.get('hint', ''), pm(row.get('vs_tip_add'), row.get('vs_tip_rem')),
        pm(row.get('vs_base_add'), row.get('vs_base_rem')), pm(row.get('tip_vs_base_add'), row.get('tip_vs_base_rem')),
        iso(row['e_mtime']), (row.get('tip_last_change_date') or '') + ' ' + (row.get('tip_last_change') or ''), row.get('e_only_file') or ''))
A('')
vc = collections.Counter(r['verdict2'].split('  [')[0] for r in s2b_rows)
A('Verdict counts: ' + '; '.join('%s = %d' % (k, v) for k, v in vc.most_common()) + '.')
A('')
A('How to read the verdicts: **REAL-LOCAL-EDIT** = the tip did not move off E-HEAD for this path and E carries uncommitted lines (strongest "newer" evidence; 3 small edits incl. one machine-local settings file). **CONFLICT** = both moved; the four `Project Steering/*` registers carry E-only additions (GOALS_AND_CLAIMS +24 lines, MODEL_REGISTRY +10, RETRACTION_LOG +3, BACKLOG +1 vs the tip) that the memory note says to re-apply from `git show <tip>:path`, never from E:. **NEW-DRAFT-vs-TIP** = no E-HEAD base (untracked or staged-A); E and tip diverge in both directions - compare `E mtime` with `tip last change`: where E is OLDER than the tip\'s last change the tip is almost certainly the landed, later version (pre-landing draft); where E is NEWER, the E lines are candidate unlanded edits. **LIKELY-STALE** = E is a strict line-subset of the tip. mtimes on a live working tree are a hint only.')
A('')

A('## 5. MISSING-WT and index-only blobs')
A('')
for e_ in miss_out:
    A('* `%s` (origin %s): index blob `%s` - on tip: %s, in history: %s%s.' % (
        e_['path'], '+'.join(e_['origins']), (e_.get('idx_sha') or '')[:12] + ('; SAME BYTES also at E path ' + ', '.join('`%s`' % q for q in e_['moved_in_E']) if e_.get('moved_in_E') else ''), e_['idx_on_tip'], e_['idx_in_hist'],
        (', recovered from E:\'s object store to `%s` (%s B)' % (e_['recovered_to'], e_.get('idx_blob_size'))) if e_.get('recovered_to') else ''))
A('')
_n_idx_only = sum(1 for x in idx_mismatch if not (x['idx_on_tip'] or x['idx_in_hist']))
A('Staged entries whose worktree bytes differ from the staged blob (edited after `git add`): %d (index blob on tip or in history for %d of them). The other %d staged blobs exist only in E:\'s object store as EARLIER drafts; the worktree bytes that superseded them are classified above, so no separate action is recommended (list in `classes.json` -> `index_blob_mismatch`).' % (
    len(idx_mismatch), sum(1 for x in idx_mismatch if x['idx_on_tip'] or x['idx_in_hist']), _n_idx_only))
A('')

A('## 6. Files ignored by E:\'s .gitignore (outside the population above)')
A('')
ci = collections.Counter(r['cls'] for r in RI.values())
A('Enumerated `ls-files -o -i --exclude-standard` excluding root `data/` (datasets, never committed), `_c/`, `.claude/`, the token file, caches and mp4: %d files -> %s. Of the %d S4: 2,059 jpg + 13 js are `Media/2026-10-04-refcv7-replay/data/` replay assets (%s MB; media, regenerable); %d are `taniteval/**/report/` derived outputs. **No ignored source file is stranded:** the package `code/fix/**/stack/tanitad/data/*.py` files hidden by the `data/` rule are 6 S1 + 7 S4m (content already on the tip at `stack/...`), including X10 `pose_sync.py`.' % (
    len(RI), dict(ci), ci['S4'], mbs(sum(r['size'] for p, r in RI.items() if r['cls'] == 'S4' and p.startswith('Media/'))),
    sum(1 for p, r in RI.items() if r['cls'] == 'S4' and p.startswith('taniteval/'))))
A('119 mp4 files (refcv6/refcv7 videos, u400 segments, teacher mini clips, Qwen review clips) are ignored by the `*.mp4` rule: bulk media, list only; the token file at the repo root exists on E: and was deliberately neither hashed nor read.')
A('')

A('## 7. Agent-declared landing lists (`LANDING_READY*.txt`) vs reality')
A('')
A('74 manifests are present in the E: tree. 13 of them are themselves **not on the tip** (S4/S2b). For manifests that list `md5/blob + path`, entries were compared with the tip. Manifests whose entries are on E: but absent from the tip path-wise (the lists the Master Mind never finished landing):')
A('')
A('| manifest | class | entries | tip-exact | tip-differs | path-only (on tip) | NOT on tip | of which present on E: |')
A('|---|---|---:|---:|---:|---:|---:|---:|')
for x in LRX:
    if 'error' in x:
        continue
    if x['not_on_tip_present_in_E'] > 0 or x['cls'] in ('S4', 'S2b'):
        A('| %s | %s | %d | %d | %d | %d | %d | %d |' % (x['manifest'], x['cls'], x['entries'], x['tip_exact'], x['tip_differs'],
                                                        x['on_tip_unchecked'], x['not_on_tip'], x['not_on_tip_present_in_E']))
A('')
A('The two large unlanded slices are `2026-09-28-refcv7-standard-tests/navsim` (173 listed files absent from the tip) and `2026-10-04-navsim-thor-backend` (317 absent), plus three smaller ones (X10 pose-timing 2, P4 3, suite-runnability-audit 5). Counts of manifests whose entry format could not be parsed are shown as 0 entries.')
A('')
A('## 8. Reproduction')
A('')
A('`scripts/classify.py` (S1-S4 by blob id; `--redo-missing`, `--ignored` modes), `scripts/stage2.py` (scan + describe), `scripts/rescan.py` (secret-pattern refinement), `scripts/stage3a.py` (landing-manifest cross-reference), `scripts/stage3b.py` (S2b diffs), `scripts/scanlib.py`, `scripts/report.py` (this document). Run with `python -X utf8 -I` (isolated mode ignores `PYTHONIOENCODING`; stdout is cp1252 otherwise).')
A('')
long_n = sum(1 for p_, r_ in S4.items() if r_['rec'] == 'LAND' and len(E + '/' + p_) > 259)
A('## 9. Landing notes for the Master Mind')
A('')
A('1. **Order:** take `paths_by_tier.json` P1 first (%d files, %s MB: code, docs, small json/csv/log), then P2 (%d files, %s MB). P3 (`paths_P3_optional.json`, %d files, %s MB, 5-20 MB raw data) only as a separate commit if the raw evidence is wanted in git; the >20 MB set and all checkpoints/media stay on D:/E: (list: section 3 and `s4_list.csv`) - they need a durable second copy outside this one drive (E: is an external disk; CLAUDE.md: GitHub is the durable copy).' % (
    rec_cnt['LAND/P1'], mbs(rec_bytes['LAND/P1']), rec_cnt['LAND/P2'], mbs(rec_bytes['LAND/P2']), rec_cnt['LAND-LARGE/P3'], mbs(rec_bytes['LAND-LARGE/P3'])))
A('2. **Scrubbed copies:** the %d LAND-HOLD-SCRUB files were redacted mechanically (public IPs, ssh ports, token-file name; 1-5 lines each, counts in the table above), re-scanned clean, and are listed in `paths_scrubbed.json` with source `OUT/scrubbed/<repo path>`. They are NOT byte-copies of E:; the cleanup `.ps1/.sh` scripts and the pod `.sh` helpers are archival records after redaction, not runnable.' % len(SCRUBBED))
_lp = collections.Counter('/'.join(p_.split('/')[:5]) for p_, r_ in S4.items() if r_['rec'] == 'LAND' and len(E + '/' + p_) > 259)
A('3. **Long paths:** %d LAND files have an absolute E: path longer than 259 characters (%s); the lander needs `core.longpaths=true` and the extended-length path prefix, or these should be deferred.' % (long_n, '; '.join('%s: %d' % kv for kv in _lp.most_common())))
A('4. **Integration (not solved by landing):** the %d package-local code variants listed under section 3 (`code/fix*/stack/...`) are *not* applied to `stack/` on the tip; landing them only archives them next to the report. Applying them is a code-integration decision (base = tip, never E:), notably `refcv8-tactical-conditioning/code/fix_P7/` (launch_gate, refc_v3_train, refcv8_conditioning, refcv8_train ...), `refcv7-box-head/code/fix_r6/` and `v7f-refav1-state/v7f_r1r4|v7f_r3/`.' % len(CODE_PATCHES))
_nb = sorted([r_ for r_ in s2b_rows if r_.get('hint', '').startswith('E NEWER') and (r_.get('vs_tip_add') or 0) >= 20 and not r_['path'].startswith('.claude/')], key=lambda r_: -r_['vs_tip_add'])
A('5. **Candidate edits:** nothing in section 4 was landed. Rows whose E file is NEWER than the tip\'s last change to the same path AND carry >= 20 E-only lines (mtime is a secondary hint; the E-only lines are in `candidate_edits/`): ' + '; '.join('`%s` (+%d)' % (r_['path'], r_['vs_tip_add']) for r_ in _nb) + '. The leaderboard code (`taniteval/taniteval/leaderboard/*`, staged-A on 2026-09-26 22:26-22:48, tip commit 11:32 the same day) is the clearest PROBABLE case of a later E-side edit that is not on the tip (UNVERIFIED until a human diffs it). The `Project Steering/*` registers must be re-edited from `git show <tip>:<path>`, never from E:.')
A('6. **Provenance caveat:** E: is a live working tree; `recent24h` flagged %d files. Re-run `scripts/classify.py` immediately before landing if more than a few hours pass.' % len(rec24))
A('')
A('## 10. Deliverable manifest')
A('')
A('| artifact | location | exists in |')
A('|---|---|---|')
A('| this audit | `%s/AUDIT.md` | OUT dir only (to be landed) |' % PKG)
A('| per-path classes (non-S1) | `%s/classes.json` | OUT dir only |' % PKG)
A('| per-file S4 list with recommendation + one-line description | `%s/s4_list.csv` | OUT dir only |' % PKG)
A('| optional companions (content already on tip) | `%s/companions.csv` | OUT dir only |' % PKG)
A('| E-only lines of the candidate edits | `%s/candidate_edits/*.E_only_lines.txt` | OUT dir only |' % PKG)
A('| recovered index-only blob | `%s/recovered_index_blobs/` | OUT dir only (original is in E:\'s object store) |' % PKG)
A('| scripts | `%s/scripts/` | OUT dir only |' % PKG)
A('| scrubbed copies of the HOLD files | `OUT/scrubbed/` + `paths_scrubbed.json` | OUT dir only (originals on E:) |')
A('| LAND list | `OUT/paths.json` (P1+P2+recovered blob+audit package), `paths_by_tier.json`, `paths_P3_optional.json` | OUT dir only |')
A('| the stranded files themselves | E:\\Projects\\TanitAD (= D:) working tree | **ONE place only until landed** (E: is an external disk) |')
A('')
open(PKGDIR + '/AUDIT.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')

# --------------------------------------------------------------- paths.json
paths = []
tiers = collections.defaultdict(list)
long_n = 0
for pth in sorted(S4):
    r = S4[pth]
    if r['rec'] == 'LAND':
        src = E + '/' + pth
        if len(src) > 259:
            long_n += 1
        paths.append([pth, src, 'raw'])
        tiers[r['tier']].append([pth, src, 'raw'])
    elif r['rec'] == 'LAND-LARGE':
        tiers['P3'].append([pth, E + '/' + pth, 'raw'])
for e_ in miss_out:
    if e_.get('recovered_to') and not e_.get('idx_blob_scan') and not e_.get('moved_in_E'):
        paths.append([e_['path'], OUT + '/' + e_['recovered_to'], 'raw'])
pkg_files = []
# scripts into the package
os.makedirs(PKGDIR + '/scripts', exist_ok=True)
for fn in ('classify.py', 'stage2.py', 'rescan.py', 'stage3a.py', 'stage3b.py', 'scanlib.py', 'report.py'):
    shutil.copy(W + '/' + fn, PKGDIR + '/scripts/' + fn)
pkg_files = []
for root, dirs, files in os.walk(PKGDIR):
    for fn in files:
        full = os.path.join(root, fn).replace(BS, '/')
        rel = full[len(OUT) + 1:]
        pkg_files.append([rel, full, 'raw'])
json.dump(paths + sorted(pkg_files), open(OUT + '/paths.json', 'w', encoding='utf-8'), indent=0, ensure_ascii=False)
json.dump(dict(P1=tiers['P1'], P2=tiers['P2']), open(OUT + '/paths_by_tier.json', 'w', encoding='utf-8'), indent=0, ensure_ascii=False)
json.dump([[k, v['src'], 'scrubbed'] for k, v in sorted(SCRUBBED.items())], open(OUT + '/paths_scrubbed.json', 'w', encoding='utf-8'), indent=0, ensure_ascii=False)
json.dump(tiers['P3'], open(OUT + '/paths_P3_optional.json', 'w', encoding='utf-8'), indent=0, ensure_ascii=False)
print('LAND files', len(paths), 'long-path', long_n, 'pkg files', len(pkg_files))
print(dict(rec_cnt))
print({k: round(v / MB, 1) for k, v in rec_bytes.items()})
print('counts', dict(cnt))
print('staged-absent', dict(sa_cls))
print('idx_mismatch', len(idx_mismatch), 'missing', len(miss_out))
