"""Stage 2: refine classes, scan S4/S2b for secrets, describe, recommend.  Read-only on E:."""
import os, sys, json, re, time, subprocess, collections, difflib, zipfile, hashlib

OUT = 'C:/Users/Admin/estrand/out'
W = OUT + '/work'
E = 'E:/Projects/TanitAD'
TIP = '780a025558d29102bf5f15ba695e5b376c64e301'
MIRROR = 'C:/Users/Admin/tanitad-push'
BS = chr(92)
GE = ['git', '--no-optional-locks', '-c', 'safe.directory=*', '-c', 'core.autocrlf=false', '-C', E]
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
NOW = time.time()


def p(*a):
    print(*a, flush=True)


def lp(path):
    fp = E + '/' + path
    if len(fp) > 240:
        fp = BS + BS + '?' + BS + fp.replace('/', BS)
    return fp


raw = json.load(open(W + '/classes_raw.json'))
R = raw['results']

# ------------------------------------------------------------------ S4m / S4h refinement
for pth, r in R.items():
    if r['cls'] in ('S4m', 'S4h'):
        dst = r.get('moved_to_tip') or r.get('in_hist_as')
        same_bn = os.path.basename(dst) == os.path.basename(pth)
        r['dup_dst'] = dst
        r['dup_strong'] = bool(r['size'] >= 256 and same_bn)
        r['dup_samebn'] = same_bn
# MOVED strong => genuinely moved/renamed; weak (tiny or different basename) => content-identical
# elsewhere but the path is a distinct deliverable location.

# ------------------------------------------------------------------ scanning
TEXT_EXT = {'.json', '.jsonl', '.log', '.txt', '.csv', '.py', '.sh', '.ps1', '.md', '.out', '.stdout',
            '.diff', '.md5', '.thor', '.done', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.html', '.tsv',
            '.xml', '.err', '.sha256', '.bat', '.cmd', '.js', '.ipynb', ''}
PATS = {
    'hf_token': re.compile(rb'hf_[A-Za-z0-9]{20,}'),
    'gh_token': re.compile(rb'(?:ghp_|gho_|ghs_|github_pat_)[A-Za-z0-9_]{25,}'),
    'private_key': re.compile(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    'sk_key': re.compile(rb'\bsk-[A-Za-z0-9]{32,}'),
    'aws_key': re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
    'slack_tok': re.compile(rb'\bxox[abprs]-[A-Za-z0-9-]{10,}'),
    'bearer': re.compile(rb'(?i)\bbearer\s+[A-Za-z0-9._\-]{24,}'),
    'secret_assign': re.compile(rb'(?i)(?:api[_-]?key|secret|passwd|password|token)["\']?\s*[:=]\s*["\'][A-Za-z0-9_\-./+]{16,}["\']'),
    'tokenfile_path': re.compile((b'Key' + rb's\.txt')),
    'runpod_ssh': re.compile(rb'(?i)@ssh\.runpod\.io|proxy\.runpod\.net|runpod\.io/'),
    'ssh_port': re.compile(rb'(?i)\bssh\b[^\n]{0,120}\s-p\s*\d{3,5}\b|\bscp\b[^\n]{0,120}\s-P\s*\d{3,5}\b'),
}
IPPORT = re.compile(rb'(?<![\d.])((?:\d{1,3}\.){3}\d{1,3}):(\d{2,5})(?![\d])')
IPCTX = re.compile(rb'(?i)(?:ssh|@|host(?:name)?|\bip\b|pod|runpod|server)[^\n]{0,30}?(?<![\d.])((?:\d{1,3}\.){3}\d{1,3})(?![\d.])')


def ip_class(a):
    try:
        o = [int(x) for x in a.decode().split('.')]
    except Exception:
        return None
    if len(o) != 4 or any(x > 255 for x in o):
        return None
    if o[0] in (0, 127, 255) or o[0] >= 224:
        return 'skip'
    if o[0] == 10 or (o[0] == 192 and o[1] == 168) or (o[0] == 172 and 16 <= o[1] <= 31) or (o[0] == 169 and o[1] == 254):
        return 'private'
    return 'public'


def scan_bytes(data):
    flags = set()
    for k, rx in PATS.items():
        if rx.search(data):
            flags.add(k)
    for m in IPPORT.finditer(data):
        c = ip_class(m.group(1))
        if c == 'public':
            flags.add('public_ip_port')
            break
        if c == 'private':
            flags.add('private_ip_port')
    for m in IPCTX.finditer(data):
        c = ip_class(m.group(1))
        if c == 'public':
            flags.add('public_ip_ctx')
            break
        if c == 'private':
            flags.add('private_ip_ctx')
    return sorted(flags)


NAME_FLAGS = re.compile(r'(?i)(^|/)((?:key|token|secret)s?\.(?:txt|json|ini)|\.env|id_rsa|id_ed25519|.*\.pem|.*\.key|.*credentials.*|.*token.*\.txt)$')


def is_text(path, ext, head):
    return (ext in TEXT_EXT) and (b'\0' not in head)


def describe(path, ext, size, data_head):
    """one-line description from the head bytes (no secret check here)."""
    try:
        t = data_head.decode('utf-8', 'replace')
    except Exception:
        return ''
    t = t.replace('\r', '')
    if ext == '.md':
        for ln in t.split('\n'):
            ln = ln.strip()
            if ln:
                return ln.lstrip('# ')[:140]
    if ext == '.py':
        m = re.search(r'^(?:#[^\n]*\n)*\s*(?:r|u)?("""|\'\'\')(.*?)(\1)', t, re.S)
        if m:
            for ln in m.group(2).split('\n'):
                if ln.strip():
                    return 'py: ' + ln.strip()[:130]
        for ln in t.split('\n'):
            if ln.startswith('#') and len(ln) > 3:
                return 'py: ' + ln.lstrip('# ')[:130]
        return 'py source'
    if ext in ('.sh', '.ps1', '.bat'):
        for ln in t.split('\n'):
            ln = ln.strip()
            if ln.startswith('#') and not ln.startswith('#!') and len(ln) > 3:
                return 'script: ' + ln.lstrip('# ')[:130]
        return 'script'
    if ext == '.json':
        t2 = t.strip()
        try:
            if size <= len(data_head):
                j = json.loads(t2)
                if isinstance(j, dict):
                    return 'json dict keys: ' + ','.join(list(j)[:8])[:130]
                return 'json list n=%d' % len(j)
        except Exception:
            pass
        m = re.findall(r'"([A-Za-z0-9_\-]{1,40})"\s*:', t2)
        seen = []
        for k in m:
            if k not in seen:
                seen.append(k)
            if len(seen) >= 8:
                break
        return 'json (head keys): ' + ','.join(seen)[:120]
    if ext == '.jsonl':
        ln = t.split('\n')[0]
        m = re.findall(r'"([A-Za-z0-9_\-]{1,40})"\s*:', ln)
        return 'jsonl rows; keys: ' + ','.join(m[:8])[:110]
    if ext in ('.csv', '.tsv'):
        return 'table; header: ' + t.split('\n')[0][:120]
    if ext == '.diff':
        return 'unified diff'
    if ext == '.done':
        return 'completion marker: ' + t.strip()[:30]
    for ln in t.split('\n'):
        ln = ln.strip()
        if ln:
            return ('text: ' + ln)[:140]
    return 'empty'


def describe_bin(path, ext, size):
    if ext == '.npz':
        try:
            with zipfile.ZipFile(lp(path)) as z:
                nm = [n[:-4] if n.endswith('.npy') else n for n in z.namelist()]
            return 'npz arrays: ' + ','.join(nm[:8]) + ('' if len(nm) <= 8 else ' ..(%d)' % len(nm))
        except Exception:
            return 'npz (unreadable header)'
    return {'.pt': 'torch tensor/checkpoint', '.pkl': 'pickle', '.png': 'png image', '.jpg': 'jpeg image',
            '.mp4': 'video', '.bin': 'raw binary', '.gz': 'gzip', '.xz': 'xz'}.get(ext, 'binary ' + ext)


# ------------------------------------------------------------------ pass over candidates
cands = [pth for pth, r in R.items() if r['cls'] in ('S4', 'S2b', 'S3b', 'MISSING-WT') or
         (r['cls'] in ('S4m', 'S4h') and not r.get('dup_strong'))]
p('scan candidates', len(cands))
t0 = time.time()
tot = 0
for i, pth in enumerate(sorted(cands)):
    r = R[pth]
    ext = os.path.splitext(pth)[1].lower()
    if pth.endswith('.jsonl.gz'):
        ext = '.gz'
    r['ext'] = ext
    r['name_flag'] = bool(NAME_FLAGS.search(pth))
    if r['cls'] == 'MISSING-WT':
        continue
    size = r['size']
    r['recent24h'] = (NOW - r['mtime']) < 86400
    flags = []
    desc = ''
    try:
        with open(lp(pth), 'rb') as f:
            head = f.read(65536)
        txt = is_text(pth, ext, head[:8192])
        if txt and size <= 20 * 1024 * 1024:
            data = head if size <= len(head) else open(lp(pth), 'rb').read()
            tot += len(data)
            flags = scan_bytes(data)
            desc = describe(pth, ext, size, head[:8192])
        elif txt:
            r['scan_skipped'] = 'size>20MB'
            desc = describe(pth, ext, size, head[:8192])
        else:
            desc = describe_bin(pth, ext, size)
            r['scan_skipped'] = 'binary'
    except Exception as ex:
        r['scan_error'] = str(ex)[:80]
    if r['name_flag']:
        flags.append('secret_filename')
    r['flags'] = flags
    # a description may itself echo a secret/IP; withhold if any scan pattern fires on it
    if desc:
        if scan_bytes(desc.encode('utf-8', 'replace')):
            desc = '(withheld: pattern hit in description)'
        elif flags:
            desc = desc  # kept, but already re-scanned above
    r['desc'] = desc
    if (i + 1) % 400 == 0:
        p('scanned', i + 1, 'of', len(cands), round(time.time() - t0), 's', round(tot / 1e9, 2), 'GB text')

p('scan done', round(time.time() - t0), 's')
json.dump(dict(tip=TIP, results=R), open(W + '/classes_scanned.json', 'w'))
