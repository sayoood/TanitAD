"""launch_detached_lh.py -- start code/chain_lh.sh fully detached (own process group, no console, breakaway when allowed). Copied from
refcv9_l2/selfsharp/launch_detached.py; only the command differs."""
import subprocess
DETACHED_PROCESS, NEW_GROUP, BREAKAWAY = 0x00000008, 0x00000200, 0x01000000
cmd = [r"C:\Program Files\Git\bin\bash.exe", "-c", "cd /c/Users/Admin/refcv8_pf/refcv9_lonhead && exec bash code/chain_lh.sh > logs/chain_lh.nohup 2>&1"]
for flags, tag in ((DETACHED_PROCESS | NEW_GROUP | BREAKAWAY, "detached+newgroup+breakaway"), (DETACHED_PROCESS | NEW_GROUP, "detached+newgroup (no breakaway permitted)")):
    try:
        p = subprocess.Popen(cmd, creationflags=flags, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
        print("LAUNCHED", tag, "WinPID", p.pid)
        break
    except OSError as e:
        print("launch variant failed:", tag, e)
