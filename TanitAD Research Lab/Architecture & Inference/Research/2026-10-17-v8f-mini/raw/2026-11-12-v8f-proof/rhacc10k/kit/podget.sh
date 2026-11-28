#!/usr/bin/env bash
# podget.sh '<remote command>' > local_file   (binary-safe stdout; address read from pod_direct.txt at run time, never printed)
F=/c/Users/Admin/v8pod/pod_direct.txt
H=$(grep -oP '^host=\K.*' "$F" | tr -d '\r'); P=$(grep -oP '^port=\K.*' "$F" | tr -d '\r')
ssh -i ~/.ssh/tanitad_pod -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -p "$P" root@"$H" "$@" 2>/dev/null
