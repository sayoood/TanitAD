import re
core = open("C:/Users/Admin/b8carve/full/lists.sh").read()


def get(n):
    return re.search(n + r'="([^"]*)"', core).group(1).split()


CORE, GATEX, CONS = get("CORE"), get("GATEX"), get("CONS")
LG = "refe/test_launch_gate.py::--static+--no-banked"
R = "C:/Users/Admin/b8carve/full"


def lane(tree, tag, tests):
    seen = []
    for t in tests:
        if t not in seen:
            seen.append(t)
    return f'C:/Python314/python.exe C:/Users/Admin/b8carve/lane_runner.py {R}/{tree} {R}/{tag}_logs {R}/{tag}_summary.json {" ".join(seen)}'


gate_main = [LG, "refe/test_rl_safeobj.py", "refe/test_carve_feed_gate.py"] + CORE + ["refe/test_carve_feed.py", "refe/test_p6_levers.py", "refe/test_p6_chain.py"]
lines = ["@echo off", "set PYTHONIOENCODING=utf-8", "set LANE_TIMEOUT=3000",
         lane("gate_full", "gate", gate_main), f"echo done > {R}/LANE_gate_DONE.txt",
         lane("mine_full", "mine", CORE), f"echo done > {R}/LANE_mine_DONE.txt",
         lane("base_full", "base", CORE), f"echo done > {R}/LANE_base_DONE.txt",
         f"echo done > {R}/MAIN_DONE.txt",
         lane("gate_full", "gate", CONS), f"echo done > {R}/EXTRA_gate_DONE.txt",
         lane("base_full", "base", CONS + ["refe/test_p6_levers.py", "refe/test_p6_chain.py"]), f"echo done > {R}/EXTRA_DONE.txt"]
open("C:/Users/Admin/b8carve/full/lanes_full.cmd", "w", newline="").write("\r\n".join(lines) + "\r\n")
print("\n".join(lines))
