#!/bin/bash
# Thor cleanup, DELETION part -- for the PI to run (Claude does not delete data; the archive part was a verified move to the dev box).
# Everything here is a duplicate or regenerable scratch; nothing is referenced by a cron job, unit or live process (checked
# 2026-09-14 21:20). Default is a dry run that lists sizes; pass --yes to delete.
#   /home/nvidia/tmp/refc-base-30k        1.2 GB  second copy of a refc-base checkpoint pulled 2026-08-29 (the models/refc-base
#                                                 original is archived on the dev box: D:/thor_archive/2026-09-14/models/refc-base)
#   /home/nvidia/tmp/refc_pull.log        tiny    its pull log
#   /home/nvidia/data/_measure_tmp        0.4 GB  12 v2ep.pt copies made for the epcache sizing measurement (2026-08-29)
#   /home/nvidia/.cache/uv                1.7 GB  uv package download cache (rebuilds itself on the next install)
#   /home/nvidia/.cache/pip               0.3 GB  pip package download cache
#   /dev/shm/mi_*                         0.2 GB  RAM scratch of the multi-instance SAM3 throughput probe
#   /home/nvidia/archive_manifest_*       tiny    md5 manifests of the archive (copies are in D:/thor_archive/2026-09-14/_manifests)
TARGETS=(/home/nvidia/tmp/refc-base-30k /home/nvidia/tmp/refc_pull.log /home/nvidia/data/_measure_tmp /home/nvidia/.cache/uv /home/nvidia/.cache/pip)
TARGETS+=(/dev/shm/mi_* /home/nvidia/archive_manifest_*)
if [ "$1" != "--yes" ]; then
  echo "DRY RUN (pass --yes to delete):"
  du -shc "${TARGETS[@]}" 2>/dev/null
  exit 0
fi
for t in "${TARGETS[@]}"; do
  [ -e "$t" ] && rm -rf -- "$t" && echo "deleted $t"
done
df -h / | tail -1
