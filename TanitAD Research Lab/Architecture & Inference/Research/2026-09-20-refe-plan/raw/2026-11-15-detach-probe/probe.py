"""MM 2026-11-15 probe: does a child console process die (0xC000013A) when its parent was started with DETACHED_PROCESS?
parent mode A = the p7_read_chain --detach flags (NEW_PROCESS_GROUP | NO_WINDOW | DETACHED_PROCESS);
parent mode B = the same without DETACHED_PROCESS (a hidden console the children inherit).
In each mode the parent starts a child exactly as p7_read_chain.step_families does (Popen, no creationflags), the child sleeps
20 s, and the parent records the child's return code. 0 = survived; 3221225786 = 0xC000013A (CTRL_C / console close)."""
import subprocess, sys, time, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
if len(sys.argv) > 1 and sys.argv[1] == "child":
    time.sleep(20)
    sys.exit(0)
if len(sys.argv) > 1 and sys.argv[1] == "parent":
    mode = sys.argv[2]
    p = subprocess.Popen([PY, os.path.abspath(__file__), "child"], stdin=subprocess.DEVNULL,
                         stdout=open(os.path.join(HERE, f"child_{mode}.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    rc = p.wait()
    with open(os.path.join(HERE, f"result_{mode}.json"), "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "child_rc": rc, "t": time.time()}, f)
    sys.exit(0)
# launcher: start both parents detached, then exit
F_A = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
F_B = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
for mode, fl in (("A_detached", F_A), ("B_nowindow", F_B)):
    subprocess.Popen([PY, os.path.abspath(__file__), "parent", mode], stdin=subprocess.DEVNULL,
                     stdout=open(os.path.join(HERE, f"parent_{mode}.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                     creationflags=fl)
print("launched")
