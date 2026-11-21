#!/usr/bin/env bash
# Copy the EVAL-split /3 files (refcv6's eval139 clips that PASSED the export) from Thor to the dev box, verified.
# Runs on the dev box (Git Bash). Reads the export's MANIFEST.jsonl (already pulled into the package) for the list and
# each file's sha256; copies with one tar stream over ssh (LAN); verifies EVERY file twice: sha256 against the manifest,
# md5 against Thor's own md5sum of the same file. Refuses to write into a non-empty destination.
set -euo pipefail
MAN="$1"                       # local path of the pulled MANIFEST.jsonl
DEST="${2:-D:/refcv6_eval_kit/data/sam3_gt_v3_eval}"
SRC=/home/nvidia/data/sam3_gt_v3
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then echo "[copy] $DEST not empty: refusing"; exit 2; fi
mkdir -p "$DEST"
LIST="$DEST/.eval_list.txt"
"$PY" -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if l.strip()]
ev=sorted(r['sha12'] for r in rows if r.get('split')=='eval' and r.get('status')=='PASS')
open(sys.argv[2],'w',newline='\n').write('\n'.join(s+'.sam3mapgt.npz' for s in ev)+'\n')  # LF: Windows text mode wrote CRLF and Thor's tar got 'x.npz\r' (2026-09-27 02:30 fix)
print('eval PASS files:',len(ev))
" "$MAN" "$LIST"
N=$(wc -l < "$LIST")
T0=$(date +%s)
# scp, 16 files per call (2026-09-27 fix). The first version piped `tar cf -` through ssh; that stream ran at
# ~0.1 MB/s and hit its 1800 s timeout at 52 of 137 files, while scp moved the same files ~5x faster.
mapfile -t FILES < "$LIST"
for ((i=0; i<${#FILES[@]}; i+=16)); do
  SRCS=()
  for f in "${FILES[@]:i:16}"; do SRCS+=("tanitad-thor-wifi:$SRC/$f"); done
  timeout 900 scp -q -o BatchMode=yes "${SRCS[@]}" "$DEST/"
done
T1=$(date +%s)
timeout 600 ssh -o BatchMode=yes tanitad-thor-wifi "cd $SRC && md5sum \$(cat)" < "$LIST" > "$DEST/.thor_md5.txt"
( cd "$DEST" && md5sum -c .thor_md5.txt > .md5_check.txt 2>&1 ) || true
BAD_MD5=$(grep -vc ': OK$' "$DEST/.md5_check.txt" || true)
"$PY" -c "
import json,hashlib,sys,os
rows={r['sha12']:r for r in (json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if l.strip())}
d=sys.argv[2]; bad=[]; n=0; tot=0
for f in sorted(x for x in os.listdir(d) if x.endswith('.sam3mapgt.npz')):
    h=hashlib.sha256(open(os.path.join(d,f),'rb').read()).hexdigest(); n+=1; tot+=os.path.getsize(os.path.join(d,f))
    if h!=rows[f[:12]]['sha256']: bad.append(f[:12])
rec={'n_files':n,'bytes':tot,'sha256_mismatch':bad,'md5_not_ok_lines':int(sys.argv[3]),'copy_s':int(sys.argv[4]),
     'verdict':'PASS' if (not bad and int(sys.argv[3])==0 and n==int(sys.argv[5])) else 'FAIL'}
json.dump(rec,open(os.path.join(d,'COPY_VERIFY.json'),'w'),indent=1); print(json.dumps(rec))
" "$MAN" "$DEST" "$BAD_MD5" "$((T1-T0))" "$N"
