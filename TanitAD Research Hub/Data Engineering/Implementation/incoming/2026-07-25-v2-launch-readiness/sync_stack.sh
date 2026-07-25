set -u
cd /workspace/TanitAD
TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "=== HOST $(hostname)  UTC $TS ==="
echo "--- PRE md5 (drift check) ---"
for f in stack/scripts/train_flagship_v4.py stack/scripts/train_flagship4b.py stack/scripts/refb_labels.py stack/scripts/v2_compressed.py stack/tanitad/config.py; do
  [ -f "$f" ] && md5sum "$f" || echo "ABSENT $f"
done
echo "--- BACKUP ---"
tar -czf /workspace/tmp/stack_backup_$TS.tgz --exclude=__pycache__ --exclude='*.pyc' stack 2>/dev/null
ls -l /workspace/tmp/stack_backup_$TS.tgz
echo "--- EXTRACT ---"
tar -xzf /workspace/tmp/stack_sync.tgz -C /workspace/TanitAD && echo EXTRACT_OK
find /workspace/TanitAD/stack /workspace/TanitAD/taniteval -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "--- POST md5 ---"
for f in stack/scripts/train_flagship_v4.py stack/scripts/train_flagship4b.py stack/scripts/refb_labels.py stack/scripts/v2_compressed.py stack/tanitad/config.py; do
  [ -f "$f" ] && md5sum "$f" || echo "ABSENT $f"
done
