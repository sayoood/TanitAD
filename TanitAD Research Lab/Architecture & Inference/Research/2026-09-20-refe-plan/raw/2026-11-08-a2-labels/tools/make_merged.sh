#!/bin/bash
# merged = pinned tip refe/ (git archive of d5ec25ec) + the A2 overlay on top.  usage: make_merged.sh <tip_refe_dir> <out_dir>
set -e
rm -rf "${2:?}"; cp -r "${1:?}" "$2"; cp "$(dirname "$0")"/../overlay/refe/*.py "$2"/
echo merged "$2"
