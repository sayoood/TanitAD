#!/bin/bash
# usage: podsetup.sh <lever> <pod tree folder name>   copies the lever tree twice (pristine, work) into the mutlists folder and pushes the spec
L=$1; T=$2
timeout 600 bash /c/Users/Admin/v8pod/pssh.sh "df -BG /workspace | tail -1 | cut -c1-80; mkdir -p /workspace/v8proof/mutlists/$L/tree_pristine /workspace/v8proof/mutlists/$L/tree_work; cd /workspace/v8proof/mutlists/$L; cp -r /workspace/v8proof/$T/stack /workspace/v8proof/$T/taniteval tree_pristine/; cp -r /workspace/v8proof/$T/stack /workspace/v8proof/$T/taniteval tree_work/; ls tree_work" 2>/dev/null | sed '/hf_/d'
timeout 60 bash /c/Users/Admin/v8pod/pssh.sh "cat > /workspace/v8proof/mutlists/$L/spec_$L.json" < $L/spec_$L.json 2>/dev/null | sed '/hf_/d'
sha256sum $L/spec_$L.json | cut -c1-16; timeout 60 bash /c/Users/Admin/v8pod/pssh.sh "sha256sum /workspace/v8proof/mutlists/$L/spec_$L.json" 2>/dev/null | sed '/hf_/d' | cut -c1-16
