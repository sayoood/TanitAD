#!/bin/sh
# Copy a file to the G: mount with retries and a CONTENT check.
# The mount flaps (Errno 22 / paged-out) for minutes at a time; a copy that
# "succeeded" is not evidence, so every file is md5-compared after the copy and
# the script exits non-zero if any pair disagrees.
set -u
copy_verify() {
  src="$1"; dst="$2"
  mkdir -p "$(dirname "$dst")" 2>/dev/null
  i=0
  while [ $i -lt 12 ]; do
    i=$((i + 1))
    cp "$src" "$dst" 2>/dev/null
    a=$(md5sum "$src" 2>/dev/null | cut -d' ' -f1)
    b=$(md5sum "$dst" 2>/dev/null | cut -d' ' -f1)
    if [ ${#a} -eq 32 ] && [ "$a" = "$b" ]; then
      echo "BANK OK   $a  $dst"; return 0
    fi
    sleep 5
  done
  echo "BANK FAIL $dst (src md5=${a:-none} dst md5=${b:-none})"; return 1
}
rc=0
while [ $# -gt 0 ]; do
  copy_verify "$1" "$2" || rc=1
  shift 2
done
exit $rc
