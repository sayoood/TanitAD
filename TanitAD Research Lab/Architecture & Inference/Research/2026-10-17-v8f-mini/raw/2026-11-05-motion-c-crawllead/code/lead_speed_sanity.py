# lead_speed_sanity: decomposition of the lead-speed ESTIMATE (v_lead_est = v_ego + (gap(k+2)-gap(k-2))/0.4) for CRAWL, MOVING_OK, all moving usable records, bank seed s0. Post-run check; enters no verdict.
import numpy as np, sys
sys.path.insert(0,'E:/Projects/TanitAD-artifacts/dagland/motion_c/crawllead/code')
import crawllead as C
lg=np.load(C.RAW_BANK/'logged_cache.npz'); names=lg['names'].astype(str)
lead=C.build_lead_arrays(names); L,G,LV,K=lead['L'],lead['G'],lead['V'],lead['K']
z=np.load(C.RAW_BANK/'bank_table_s0.npz'); cid=z['clip_g'].astype(np.int64); r=z['r'].astype(np.int64)
vg=lg['V'][lg['offs'][cid]+r]; vo=z['v_own']; us=z['reason']==0
crawl=us&(vo<0.5*vg)&(vg>=3); mok=us&(vg>=3)&(vg<=8)&(vo>=0.8*vg)
k2=2*r
def stat(m,name):
    m=m&(L[cid,k2]==1)&(L[cid,k2+2]==1)&(L[cid,k2-2]==1)
    gp,g0,gm=G[cid,k2+2],G[cid,k2],G[cid,k2-2]
    m=m&(abs(gp-g0)<=4)&(abs(g0-gm)<=4)
    ve=LV[cid,k2]; rel=(gp-gm)/0.4
    q=lambda a:np.round(np.percentile(a[m],[5,25,50,75,95]),3)
    print(name,m.sum(),'ve',q(ve),'rel',q(rel),'vlead',q(ve+rel),'vg',q(vg), 'corr(ve,vg)',np.corrcoef(ve[m],vg[m])[0,1])
stat(crawl,'CRAWL'); stat(mok,'MOK'); stat(us&(vg>=3),'ALLmoving')
# sanity of the estimator on a known identity: constant-gap lead -> v_lead=v_ego. fraction |rel|<0.5
