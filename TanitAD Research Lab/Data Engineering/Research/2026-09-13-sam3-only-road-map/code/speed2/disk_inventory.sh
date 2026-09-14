#!/bin/bash
# Thor disk inventory for the cleanup (PI 2026-09-14). Per directory: bytes, files, newest file mtime (days ago), and whether a
# crontab / systemd user unit / live process references it. Level 1 under /home/nvidia, level 2 under the big trees.
# Output: /home/nvidia/disk_inventory.tsv  (path  bytes  files  newest_days  referenced)
OUT=/home/nvidia/disk_inventory.tsv; : > $OUT.tmp
NOW=$(date +%s)
REFS=$( (crontab -l 2>/dev/null; ls ~/.config/systemd/user 2>/dev/null; ps -eo args) | tr ' ' '\n' | grep "^/home/nvidia" | sort -u)
row() {
  local p=$1
  local b=$(du -sb "$p" 2>/dev/null | cut -f1)
  local n=$(find "$p" -type f 2>/dev/null | wc -l)
  local m=$(find "$p" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
  local age=$([ -n "$m" ] && echo $(( (NOW - m) / 86400 )) || echo -1)
  local r=$(echo "$REFS" | grep -c "^$p")
  printf '%s\t%s\t%s\t%s\t%s\n' "$p" "${b:-0}" "$n" "$age" "$r" >> $OUT.tmp
}
for p in /home/nvidia/* /home/nvidia/.cache; do [ -d "$p" ] && row "$p"; done
for top in /home/nvidia/data /home/nvidia/sam3map /home/nvidia/qwendrive /home/nvidia/experiments /home/nvidia/models /home/nvidia/epcache_prefix \
           /home/nvidia/wpd_a3 /home/nvidia/tanit-astra-research /home/nvidia/percprobe /home/nvidia/v7tiny /home/nvidia/ckpt_snaps /home/nvidia/backup \
           /home/nvidia/valdata /home/nvidia/nurec_scenes /home/nvidia/.cache; do
  for p in "$top"/*; do [ -e "$p" ] && row "$p"; done
done
sort -t$'\t' -k2,2nr $OUT.tmp > $OUT && mv $OUT.tmp /home/nvidia/disk_inventory_unsorted_aside.tsv
echo "ZZINVENTORY-$(wc -l < $OUT)ZZ"
