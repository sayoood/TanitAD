"""Shared secret/IP scan used for deliverable gating. Never prints matches."""
import re

PATS = {
    'hf_token': re.compile(rb'hf_[A-Za-z0-9]{20,}'),
    'gh_token': re.compile(rb'(?:ghp_|gho_|ghs_|github_pat_)[A-Za-z0-9_]{25,}'),
    'private_key': re.compile(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    'sk_key': re.compile(rb'\bsk-[A-Za-z0-9]{32,}'),
    'aws_key': re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
    'slack_tok': re.compile(rb'\bxox[abprs]-[A-Za-z0-9-]{10,}'),
    'bearer': re.compile(rb'(?i)\bbearer\s+[A-Za-z0-9._\-]{24,}'),
    'tokenfile_path': re.compile(rb'(?i)\b(?:key|token|secret|credential)s?\.txt\b'),
    'runpod_ssh': re.compile(rb'(?i)@ssh\.runpod\.io|proxy\.runpod\.net|runpod\.io/'),
    'ssh_port': re.compile(rb'(?i)\bssh\b[^\n]{0,120}\s-p\s*\d{3,5}\b|\bscp\b[^\n]{0,120}\s-P\s*\d{3,5}\b'),
}
KW = re.compile(rb'(?i)(api[_-]?key|apikey|secret|passwd|password|access[_-]?token|auth[_-]?token|hf[_-]?token|api[_-]?token)')
KW_TAIL = re.compile(rb'^[A-Za-z0-9_\-]{0,20}["\']?\s*[:=]\s*["\'][A-Za-z0-9_\-./+]{16,}["\']')
IPPORT = re.compile(rb'(?<![\d.])((?:\d{1,3}\.){3}\d{1,3}):(\d{2,5})(?![\d])')
IPCTX = re.compile(rb'(?i)(?:ssh|@|host(?:name)?|\bip\b|pod|runpod|server)[^\n]{0,30}?(?<![\d.])((?:\d{1,3}\.){3}\d{1,3})(?![\d.])')
WELLKNOWN = {b'8.8.8.8', b'8.8.4.4', b'1.1.1.1', b'9.9.9.9', b'1.0.0.1'}


def ip_class(a):
    if a in WELLKNOWN:
        return 'skip'
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
    for m in KW.finditer(data):
        if KW_TAIL.match(data[m.end():m.end() + 80]):
            flags.add('secret_assign')
            break
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
