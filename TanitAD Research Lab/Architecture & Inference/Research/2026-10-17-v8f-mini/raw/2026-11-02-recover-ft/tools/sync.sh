#!/bin/sh
# copy the staged patch over the scratch tree (tip export + patch). usage: sync.sh [tree]
T="${1:-F:/TanitAD-scratch/recover_ft/tree}"
cp -r "E:/Projects/TanitAD-artifacts/dagland/recover_ft/stack_patch/stack/." "$T/stack/"
