#!/usr/bin/env bash
set -u
S="$1"; NAME="$2"
ssh -n -o BatchMode=yes tanitad-thor-wifi "md5sum /home/nvidia/ckpt_snaps/$NAME" > "$S/ck/$NAME.remote.md5"
scp -q -o BatchMode=yes "tanitad-thor-wifi:/home/nvidia/ckpt_snaps/$NAME" "$S/ck/$NAME"
md5sum "$S/ck/$NAME" > "$S/ck/$NAME.local.md5"
echo "DONE $NAME $(stat -c %s "$S/ck/$NAME")"
