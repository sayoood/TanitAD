#!/usr/bin/env bash
# S0P-STUB-TRANSPORT (marker the chain looks for: a file carrying it is never the real client).  Dry-run "ssh": a FAKE box 2 that lives in the folder $S0P_FAKE_BOX (D:\x -> $S0P_FAKE_BOX/D/x).  It reaches nothing real.
# It understands exactly the commands s0p_common.sh sends; anything else is logged to $BOX/unknown.log and fails.  Every command is logged to $BOX/ssh.log.
# Control files under $BOX/ctl: ntotal (not used), npos / nneg (the dry-run JSON), dry_bad, hang_SH<i>, fail_SH<i> (one failed job, then removed), free_bytes (D: free).
set -u
BOX="${S0P_FAKE_BOX:?}"; TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PYT="${S0P_TEST_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
shift 5; cmd="$*"; echo "$cmd" >> "$BOX/ssh.log"
topath() { printf '%s' "$BOX/$(printf '%s' "$1" | sed 's#\\#/#g; s#^\([A-Za-z]\):#\1#')"; }
rx() { [[ "$cmd" =~ $1 ]]; }
if rx 'New-Item -ItemType Directory -Force ([^ ]+) \| Out-Null'; then mkdir -p "$(topath "${BASH_REMATCH[1]}")"; exit 0; fi
if rx 'Test-Path ([^"]+)"'; then if [ -e "$(topath "${BASH_REMATCH[1]}")" ]; then echo True; else echo False; fi; exit 0; fi
if rx '^type ([^ ]+)$'; then cat "$(topath "${BASH_REMATCH[1]}")" 2> /dev/null; exit $?; fi
if rx '^certutil -hashfile ([^ ]+) SHA256$'; then p="$(topath "${BASH_REMATCH[1]}")"; [ -f "$p" ] || exit 1; echo "SHA256 hash of file:"; sha256sum "$p" | cut -d' ' -f1 | tr 'a-f' 'A-F'; echo "CertUtil: -hashfile command completed successfully."; exit 0; fi
if rx 'Get-PSDrive D\)\.Free'; then if [ -e "$BOX/ctl/free_bytes" ]; then cat "$BOX/ctl/free_bytes"; else echo 50000000000; fi; exit 0; fi
if rx 'Get-ChildItem ([^ ]+) -Recurse -File'; then p="$(topath "${BASH_REMATCH[1]}")"; if [ -d "$p" ]; then du -sb "$p" | cut -f1; else echo 0; fi; exit 0; fi
if rx 'Remove-Item -Recurse -Force ([^"]+)"'; then rm -rf "$(topath "${BASH_REMATCH[1]}")"; exit 0; fi
if rx 'Remove-Item -Force ([^"]+)"'; then rm -f "$(topath "${BASH_REMATCH[1]}")"; exit 0; fi
if rx '^rmdir ([^ ]+)$'; then rmdir "$(topath "${BASH_REMATCH[1]}")" 2> /dev/null; exit $?; fi
if rx 'launch\.ps1 -Job ([^ ]+) -Log ([^ ]+)'; then
  job="$(topath "${BASH_REMATCH[1]}")"; lg="$(topath "${BASH_REMATCH[2]}")"
  line="$(tr -d '\r' < "$job" | grep -m1 '^rem S0PJOB')"
  mode="$(sed 's/.*mode=\([a-z]*\).*/\1/' <<< "$line")"; i="$(sed 's/.* i=\([0-9]*\).*/\1/' <<< "$line")"; k="$(sed 's/.* k=\([0-9]*\).*/\1/' <<< "$line")"; gid="$(sed 's/.* gid=\([0-9A-Za-z_]*\).*/\1/' <<< "$line")"
  MD="$BOX/D/tanitad/stopperc/s0p_jobs"; mkdir -p "$MD"; echo "$mode $i $k $gid" >> "$BOX/launches.log"
  if [ "$mode" = dry ]; then tag=DRY; else tag="SH$i"; fi
  [ -e "$BOX/ctl/hang_$tag" ] && { echo "launched (hangs)" > "$lg"; exit 0; }
  if [ -e "$BOX/ctl/fail_$tag" ]; then rm -f "$BOX/ctl/fail_$tag"; echo "rc=1" > "$MD/${tag}_$gid.fail"; echo "job failed" > "$lg"; exit 0; fi
  if [ "$mode" = dry ]; then
    np="$(cat "$BOX/ctl/npos" 2> /dev/null || echo 4000)"; nn="$(cat "$BOX/ctl/nneg" 2> /dev/null || echo 12000)"
    { echo 'pool {"n_observed": 286035}'; echo '{'
      [ -e "$BOX/ctl/dry_bad" ] || echo ' "pool_equals_banked_prevalence": true,'
      echo ' "disjoint": {'; echo '  "overlap": 0'; echo ' },'; echo " \"n_neg_sample\": $nn,"; echo " \"n_pos_sample\": $np,"; echo " \"n_sample\": $((np + nn)),"
      echo ' "sample_sha256": "'"$(printf 'sample%.0s' 1 | sha256sum | cut -d' ' -f1)"'"'; echo '}'; } > "$lg"
    mkdir -p "$BOX/D/tanitad/stopperc/s0p_out"; echo '{}' > "$BOX/D/tanitad/stopperc/s0p_out/sample.json"
    echo done > "$MD/${tag}_$gid.done"; exit 0
  fi
  np="$(cat "$BOX/ctl/npos" 2> /dev/null || echo 4000)"; nn="$(cat "$BOX/ctl/nneg" 2> /dev/null || echo 12000)"
  "$PYT" -I "$TESTS/fake_shard.py" --out "$BOX/D/tanitad/stopperc/s0p_out/shard_$i" --i "$i" --k "$k" --n-total $((np + nn)) --n-pos "$np" --sample-sha "$(printf 'sample%.0s' 1 | sha256sum | cut -d' ' -f1)" > "$lg" 2>&1 || { echo "rc=1" > "$MD/${tag}_$gid.fail"; exit 0; }
  echo done > "$MD/${tag}_$gid.done"; exit 0
fi
echo "UNKNOWN: $cmd" >> "$BOX/unknown.log"; exit 1
