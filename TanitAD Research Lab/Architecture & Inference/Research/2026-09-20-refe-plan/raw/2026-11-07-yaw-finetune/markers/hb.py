import time, os
with open(r'E:\Projects\TanitAD-artifacts\yawrate_wp\YF2\markers\hb_py.txt','a') as f:
    for i in range(60):
        f.write(f'{time.strftime("%H:%M:%S")} pid {os.getpid()} ppid {os.getppid()}\n'); f.flush(); time.sleep(5)
