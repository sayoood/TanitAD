#!/bin/bash
# driver: control first (decides the 99 CPU-h), then the token-free data staging
W=/workspace
bash $W/ops/bank3c/ctl74.sh
bash $W/ops/stage_data.sh
