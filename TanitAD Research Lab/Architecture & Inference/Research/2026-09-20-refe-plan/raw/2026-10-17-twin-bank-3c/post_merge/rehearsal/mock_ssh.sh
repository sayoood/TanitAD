#!/bin/bash
# REHEARSAL ONLY.  Stands in for `ssh <pod> "<cmd>"`: runs <cmd> inside WSL Ubuntu against the MOCK pod tree (/tmp/p6mock).  stdin is forwarded (the helper script is fed that way).
# The command travels in an environment variable (WSLENV), not on a command line, so no quoting layer can alter it.
export P6REMOTE_CMD="$1"
export WSLENV="P6REMOTE_CMD"
MSYS_NO_PATHCONV=1 exec wsl.exe -e bash -c 'export PATH=/tmp/p6mock/stubs/bin:$PATH; export P6MOCK=/tmp/p6mock; eval "$P6REMOTE_CMD"'
