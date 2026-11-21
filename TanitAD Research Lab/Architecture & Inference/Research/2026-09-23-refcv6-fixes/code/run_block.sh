#!/usr/bin/env bash
# Runs the refcv6 supervisor's REAL error-count block, cut out of the script between its two
# anchors (`    n_err=0` .. the `echo "ZZ${ARM}...` line, minus its tee), against one stderr log,
# and prints the token it would emit. MODE=grep_exit2 replaces grep with a function that fails
# like an unreadable file (exit 2, nothing on stdout). A third argument `raw` prints the token's
# exact bytes; otherwise they are shown by od, so a newline inside the token is visible.
set -u
SCRIPT="$1"; ERRLOG="$2"; ARM=refcv6-r101-s0; cur=6571; STEPS=50400; launch=1
if [ "${MODE:-}" = grep_exit2 ]; then grep() { echo "grep: $ERRLOG: Input/output error" >&2; return 2; }; fi
blk="$(awk '/^    n_err=0$/{f=1} f{print} /^    echo "ZZ\$\{ARM\}/{exit}' "$SCRIPT" | sed 's/| tee -a "\$SUPLOG" > \/dev\/null//')"
[ -n "$blk" ] || { echo "BLOCK-NOT-FOUND"; exit 3; }
out="$(eval "$blk")"
if [ "${3:-}" = raw ]; then printf '%s' "$out"; else printf '%s' "$out" | od -An -c | tr -s ' ' | tr -d '\n'; echo; fi
