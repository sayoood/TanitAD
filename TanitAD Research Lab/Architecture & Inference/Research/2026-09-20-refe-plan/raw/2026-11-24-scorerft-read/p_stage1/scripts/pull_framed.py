"""MM 2026-11-22: pull a pod text file line-framed (the pd.sh output can drop lines), rebuild it exactly, check its sha256 against the pod's.
usage: pull_framed.py <pod path> <local out>"""
import hashlib, re, shutil, subprocess, sys
pod, out = sys.argv[1], sys.argv[2]
cmd = ("awk '{print \"@\" sprintf(\"%06d\", NR) \"@\" $0 \"#\"}' " + pod
       + "; echo \"ZZSHA $(sha256sum " + pod + " | cut -c1-64)\""
       + "; echo \"ZZEOL $(tail -c 1 " + pod + " | wc -l)\"")
r = subprocess.run([shutil.which('bash'), 'C:/Users/Admin/par18/pd.sh', '90', cmd], capture_output=True)
text = r.stdout.decode('utf-8', errors='strict')
rows, sha, eol = [], None, None
for line in text.split('\n'):
    if line.startswith('ZZSHA '):
        sha = line.split()[1]
    elif line.startswith('ZZEOL'):
        parts = line.split()
        eol = parts[1] if len(parts) > 1 else ''
    elif line.startswith('@'):
        m = re.match(r'^@(\d{6})@(.*)#$', line)
        assert m, line[:60]
        rows.append((int(m.group(1)), m.group(2)))
assert sha and [n for n, _ in rows] == list(range(1, len(rows) + 1)), ('gap or no sha', len(rows), sha, r.stderr[-300:])
assert eol in ('0', '1'), ('no EOL probe', eol)
data = '\n'.join(s for _, s in rows) + ('\n' if eol == '1' else '')
b = data.encode('utf-8')
got = hashlib.sha256(b).hexdigest()
assert got == sha, ('sha mismatch', got[:16], sha[:16])
open(out, 'wb').write(b)
print('ZZPULL_OK', len(rows), 'lines', sha[:16], out)
