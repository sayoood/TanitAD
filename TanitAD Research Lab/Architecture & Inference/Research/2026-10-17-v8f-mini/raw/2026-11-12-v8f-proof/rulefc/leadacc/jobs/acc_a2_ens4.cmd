@echo off
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set TEMP=D:\tanitad\tmp
set TMP=D:\tanitad\tmp
start /b /belownormal /wait "" D:\tanitad\venv\Scripts\python.exe D:\tanitad\leadacc\ens_read_n_box2.py --out D:/tanitad/leadacc/ens4 --arm K25D --expect-ens D:/tanitad/leadacc/ens/ENS.json --record ENS4 --compare ENS2 --sets "ENS2=0,1;ENS4=0,1,2,3;ENS5=0,1,2,3,4" --heads D:/tanitad/leadacc/K25D/head_30000.pt=40433c9b4416dfe30cd34f3957f7ef785146f798839861c873c2ee5472910270 D:/tanitad/leadacc/K25D_s149/head_30000.pt=f5dcc2f5af44d8382ac35b8336b78964f071bfd3e0115f596d34f1bd6012ab96 D:/tanitad/leadacc/K25DS/head_30000.pt=b762cde6e70239bb74b785de5ca302f382ef8e4fa4ba5016a68fabe1307cb93a D:/tanitad/leadacc/K25DS_s149/head_30000.pt=cd4ddcf1081484694ee7d9cc9df4ca378b35ff32b1c16c4e86e2f506ac82cf77 D:/tanitad/leadacc/K25D_s150/head_30000.pt=81e87c842d0845cd095644e33c0a607532f6384c0ef2df99114c66d18ccb09a5
