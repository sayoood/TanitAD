#!/bin/bash
cd /root
export PYTHONPATH=/root/taniteval:/root/TanitAD/stack
echo "########## DIAGNOSTIC (40 eps, per-arm process) ##########"
for k in flagship-speed flagship-nospeed refa-dinov2 refa-ijepa refa-dynin; do
  echo "=== DIAG $k ==="
  timeout 1800 python3 -m taniteval.bench --model "$k" --episodes 40 2>&1 \
    | grep -E '\[diag\]|FAILED|skipped|Error' | head -6
done
echo "########## PLANNING (40 eps, per-arm process) ##########"
for k in flagship-speed flagship-nospeed refa-dinov2 refa-ijepa refb; do
  echo "=== PLAN $k ==="
  timeout 1800 python3 -m taniteval.planning --model "$k" --max-eps 40 2>&1 \
    | grep -E '\[plan\]|FAILED|skipped|Error' | head -6
done
echo "########## DASHBOARD ##########"
python3 -c "from taniteval import report; print(report.build())"
echo "########## DONE ##########"
