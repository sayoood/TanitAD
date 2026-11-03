#!/usr/bin/env bash
# Stand-in for `scp [-r] tanitad-box2:<path> <dest>` in the pull test: maps D:/tanitad/runs/v8f_mini_full onto $RNEXT_FAKE_ROOT and copies locally (a trailing /* is a local glob). Reaches nothing.
rec=0; args=()
for a in "$@"; do case "$a" in -r) rec=1 ;; -q|-o) ;; BatchMode=*|ConnectTimeout=*) ;; *) args+=("$a") ;; esac; done
src=${args[0]}; dst=${args[1]}
src=${src#tanitad-box2:}; src=${src/D:\/tanitad\/runs\/v8f_mini_full/$RNEXT_FAKE_ROOT}
if [ "$rec" = 1 ]; then cp -r $src "$dst" 2>/dev/null; else cp "$src" "$dst"; fi
