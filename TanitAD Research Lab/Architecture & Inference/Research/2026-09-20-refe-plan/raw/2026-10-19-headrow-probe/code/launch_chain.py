import os, subprocess
RUN = "C:/Users/Admin/par18/probe/run/f10"
env = dict(os.environ, PATH="C:/Program Files/Git/usr/bin;C:/Program Files/Git/bin;" + os.environ.get("PATH", ""))
p = subprocess.Popen(["C:/Program Files/Git/bin/bash.exe", "C:/Users/Admin/par18/probe/code/chain.sh"], env=env, stdin=subprocess.DEVNULL,
                     stdout=open(RUN + "/chain.out", "ab"), stderr=subprocess.STDOUT, creationflags=0x00000008 | 0x00000200 | 0x08000000, close_fds=True)
print(p.pid)
