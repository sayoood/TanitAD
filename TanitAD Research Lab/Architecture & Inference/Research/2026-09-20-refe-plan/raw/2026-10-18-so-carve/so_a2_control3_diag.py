import os,sys,numpy as np
sys.path.insert(0,"C:/Users/Admin/socarve")
ARCH="C:/Users/Admin/AppData/Local/Temp/safehead/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0,ARCH+"/raw/2026-10-14-rls1b-attribution")
import attrib_rls1b as AT, so_carve_analysis as SC
DL,X="C:/Users/Admin/rls1b_dl","C:/Users/Admin/rls1b_dl/data/x"
kv,tgv,dv,ckv,_=AT.load_side(f"{X}/pdm_val/pdm_val",f"{X}/terms_val/terms_val"); y=AT.comps(tgv)["ttc"]; ar=np.arange(len(kv))
lgA=AT.load_scores_aligned(f"{DL}/r0/final/scores_update1000_A.npz",kv,ckv); lgB=AT.load_scores_aligned(f"{DL}/r0/final/scores_update1000_B.npz",kv,ckv)
pol=lambda lg,T: AT.softmax(np.log(AT.sel_v1_lane(AT.RV.sigmoid(lg))+AT.EPS)/T)
iB,iA=AT.picks(lgB),AT.picks(lgA); P1=-(100*(y[ar,iB]-y[ar,iA])).mean()
for T in (1e-4,1e-6,1e-8,1e-10):
    pB,pA=pol(lgB,T),pol(lgA,T)
    L=-(100*((pB*y).sum(-1)-(pA*y).sum(-1))).mean()
    print(T,"L",L,"P1",P1,"diff",L-P1,"max top-prob gap",(1-pB.max(-1)).max(),(1-pA.max(-1)).max(), "n argmax!=pick",(pB.argmax(-1)!=iB).sum()+(pA.argmax(-1)!=iA).sum())
