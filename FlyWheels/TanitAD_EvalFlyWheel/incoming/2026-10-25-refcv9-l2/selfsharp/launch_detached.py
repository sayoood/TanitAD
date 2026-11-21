"""launch_detached.py -- start chain_ss.sh fully detached from the calling tool session (own process group, no console, breakaway from the caller's job when allowed)."""
import subprocess, sys
DETACHED_PROCESS, NEW_GROUP, BREAKAWAY = 0x00000008, 0x00000200, 0x01000000
cmd = [r"C:\Program Files\Git\bin\bash.exe", "-c", "cd /c/Users/Admin/refcv8_pf/refcv9_l2/selfsharp && exec bash chain_ss.sh > logs/chain.nohup 2>&1"]
for flags, tag in ((DETACHED_PROCESS | NEW_GROUP | BREAKAWAY, "detached+newgroup+breakaway"), (DETACHED_PROCESS | NEW_GROUP, "detached+newgroup (no breakaway permitted)")):
    try:
        p = subprocess.Popen(cmd, creationflags=flags, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
        print("LAUNCHED", tag, "WinPID", p.pid)
        break
    except OSError as e:
        print("launch variant failed:", tag, e)
