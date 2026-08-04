#!/usr/bin/env bash
# Wait for the parity-corpus pull to land all 2376 episodes, then run the STRICT
# verification automatically and leave the verdict on local disk.
#
# WHY THIS EXISTS: the transfer outlives the session that started it. Without this,
# "the corpus arrived" and "the corpus was verified" are separated by however long it
# takes someone to notice — and an unverified corpus is exactly the state that lets a
# truncated cache train silently (the failure `parity.py` was written for).
#
# ⚠️ Logs to /tmp — LOCAL disk, never a network mount. A MooseFS write failure raises
# OSError [Errno 5] INSIDE print() and kills the process with no diagnostic at all;
# a logger writing to the failing filesystem cannot report that it is failing.
#
# Install:
#   scp watch_and_verify.sh tanitad-thor:~/parity_verify/
#   ssh -n tanitad-thor 'chmod +x ~/parity_verify/watch_and_verify.sh && \
#       cd /tmp && nohup ~/parity_verify/watch_and_verify.sh > /tmp/watch_verify.log 2>&1 &'
set -u
CACHE="$HOME/epcache/epcache-256px-phase0/physicalai-train-e438721ae894"
VDIR="$HOME/parity_verify"
WANT=2376
DEADLINE=$(( $(date +%s) + 8*3600 ))     # give up after 8 h rather than spin forever

echo "[watch] $(date -u +%FT%TZ) waiting for $WANT episodes in $CACHE"
while :; do
  N=$(ls "$CACHE"/ep_*.pt 2>/dev/null | wc -l)
  if [ "$N" -ge "$WANT" ]; then
    echo "[watch] $(date -u +%FT%TZ) all $N episodes present — verifying STRICT"
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[watch] $(date -u +%FT%TZ) DEADLINE reached at $N/$WANT — giving up. The pull"
    echo "[watch] is /tmp/pull_parity.py; snapshot_download RESUMES, so re-launch it."
    exit 2
  fi
  echo "[watch] $(date -u +%FT%TZ) $N/$WANT"
  sleep 120
done

cd "$VDIR" || exit 1
PYTHONPATH="$HOME/TanitAD/stack" "$HOME/venvs/tanitad-train/bin/python" \
  verify_epcache_bytes.py --cache "$CACHE" --expected hf_expected_train.json \
  --mode strict --sha256 all --load 8 --out "$VDIR/verify_train_full.json"
RC=$?
echo "[watch] $(date -u +%FT%TZ) verifier exit=$RC"
# ⛔ Exit codes are not evidence — print the verdict itself.
"$HOME/venvs/tanitad-train/bin/python" - <<'PY'
import json, pathlib
p = pathlib.Path.home()/"parity_verify/verify_train_full.json"
if not p.exists():
    print("[watch] NO verdict file written — the verifier died before writing.")
    raise SystemExit(1)
d = json.loads(p.read_text())
for k in ("VERDICT", "parity_verdict", "n_episodes_on_disk", "bytes_on_disk",
          "size_checked", "sha256_checked", "sha256_seconds"):
    print(f"[watch] {k:20s} {d.get(k)}")
for k in ("size_mismatches", "sha256_mismatches", "load_failures"):
    v = d.get(k) or []
    print(f"[watch] {k:20s} {len(v)}" + (f"  FIRST: {v[0]}" if v else ""))
print("[watch] parity record:", json.dumps(d.get("parity"), default=str))
PY
echo "[watch] done $(date -u +%FT%TZ)"
