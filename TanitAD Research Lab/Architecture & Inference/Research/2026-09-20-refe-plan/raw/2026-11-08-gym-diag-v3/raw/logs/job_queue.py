import subprocess,time,sys,os
PY=r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
T=r"E:\Projects\TanitAD-artifacts\scorer_gym_20261107\pilot_v3_diag"
O=r"F:\tanitad_scratch\mgym3\diag"
jobs=[]
for s,i in (("stall",7),("stall",10),("stall",11),("stall",15)):
    jobs.append(([PY,"-u",T+r"\cf_ttc.py",s,str(i),"log",f"{O}\cf_{s}{i}.json"],f"{O}\cf_{s}{i}.log"))
for s,i in (("pilot",4),("pilot",5),("pilot",6),("stall",9)):
    jobs.append(([PY,"-u",T+r"\q3_const.py",s,str(i),f"{O}\q3_{s}{i}.json"],f"{O}\q3_{s}{i}.log"))
def running():
    out=subprocess.run(["powershell","-NoProfile","-Command","(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'pilot_v3_diag' -and $_.CommandLine -match 'venvs.driverl-eval' }).Count"],capture_output=True,text=True).stdout.strip()
    try: return max(int(out or 0)-1,0)
    except: return 99
env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",PYTHONDONTWRITEBYTECODE="1")
F=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW
open(O+r"\queue.status","w").write("started\n")
for cmd,log in jobs:
    while running()>=4: time.sleep(20)
    subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=open(log,"w"),stderr=subprocess.STDOUT,creationflags=F,env=env)
    open(O+r"\queue.status","a").write("launched "+log+"\n"); time.sleep(30)
while running()>0: time.sleep(20)
open(O+r"\queue.status","a").write("QUEUE_DONE\n")
