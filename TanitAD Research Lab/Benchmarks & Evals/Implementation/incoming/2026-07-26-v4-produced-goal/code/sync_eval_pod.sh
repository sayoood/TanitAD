#!/usr/bin/env bash
# Sync the eval pod's taniteval + v4eval/stack to the current repo state.
# Backup already taken at /root/_bak_20260726_produced_goal.
set -e
STAGE=/root/v4eval/_sync
find "$STAGE" -mindepth 1 -delete 2>/dev/null || true
mkdir -p "$STAGE"
tar xzf /root/sync_payload.tgz -C "$STAGE" --no-same-owner --no-same-permissions

cp -a "$STAGE"/taniteval/taniteval/. /root/taniteval/taniteval/
cp -a "$STAGE"/stack/tanitad/.       /root/v4eval/stack/tanitad/
cp -a "$STAGE"/stack/scripts/.       /root/v4eval/stack/scripts/

# drop every stale bytecode cache in the synced trees
find /root/taniteval/taniteval /root/v4eval/stack -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "SYNC_COPIED"

cd /root/v4eval && find tanitad scripts -name '*.py' | sort | xargs md5sum \
    | awk '{printf "%s  stack/%s\n", $1, $2}' > "$STAGE"/pod_after.txt
cd /root/taniteval && find taniteval -name '*.py' | sort | xargs md5sum \
    | awk '{printf "%s  %s\n", $1, $2}' >> "$STAGE"/pod_after.txt
sort -k2 "$STAGE"/pod_after.txt > "$STAGE"/pod_after_sorted.txt
wc -l "$STAGE"/pod_after_sorted.txt
