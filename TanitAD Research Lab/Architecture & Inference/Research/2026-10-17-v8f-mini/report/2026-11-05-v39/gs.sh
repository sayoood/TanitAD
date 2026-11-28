#!/bin/bash
# gs.sh <relpath under R/raw> : print a tip blob
R="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
git -C C:/Users/Admin/tanitad-push -c safe.directory='*' show "574dea8:$R/$1"
