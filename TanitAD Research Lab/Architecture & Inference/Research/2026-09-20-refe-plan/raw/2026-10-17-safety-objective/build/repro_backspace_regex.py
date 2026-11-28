import re
a = "R1d"
fixed = rf"--arm-name {a}\b|--arm {a}\b|arms/{a}/(sft|preflight|monitor|launch_gate)"
broken = rf"--arm-name {a}" + chr(8) + rf"|--arm {a}" + chr(8) + rf"|arms/{a}/(sft|preflight|monitor|launch_gate)"
calls = "launch_gate.py run --profile refe_rls1d --arm-name R1d --out-dir /x\nrls1_checks.py monitor --arm R1d --out /x/arms/R1d"
print("control (a string that has no R1d in it):", bool(re.search(fixed, "launch_gate.py run --profile refe_rls1d --arm-name R0d")), bool(re.search(broken, "launch_gate.py run --profile refe_rls1d --arm-name R0d")))
print("calls naming R1d: fixed regex matches =", bool(re.search(fixed, calls)), "| round-2 regex (backspace) matches =", bool(re.search(broken, calls)))
print("the round-2 alternatives that can match at all: only the third (arms/R1d/...):", [bool(re.search(x, calls)) for x in broken.split("|")[:2]])
