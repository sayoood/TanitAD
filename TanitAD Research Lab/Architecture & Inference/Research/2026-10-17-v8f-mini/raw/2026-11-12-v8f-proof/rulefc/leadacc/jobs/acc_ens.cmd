@echo off
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set TEMP=D:\tanitad\tmp
set TMP=D:\tanitad\tmp
start /b /belownormal /wait "" D:\tanitad\venv\Scripts\python.exe D:\tanitad\leadacc\ens_read_box2.py --out D:/tanitad/leadacc/ens --arm K25D --expect-acc D:/tanitad/leadacc/K25D/ACC.json --heads D:/tanitad/leadacc/K25D/head_30000.pt=40433c9b4416dfe30cd34f3957f7ef785146f798839861c873c2ee5472910270 D:/tanitad/leadacc/K25D_s149/head_30000.pt=f5dcc2f5af44d8382ac35b8336b78964f071bfd3e0115f596d34f1bd6012ab96 D:/tanitad/leadacc/K25DS/head_30000.pt=b762cde6e70239bb74b785de5ca302f382ef8e4fa4ba5016a68fabe1307cb93a
