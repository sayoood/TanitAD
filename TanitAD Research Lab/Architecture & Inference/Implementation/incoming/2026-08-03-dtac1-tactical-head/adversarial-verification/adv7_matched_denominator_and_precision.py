import json, torch
D=torch.load('/home/nvidia/TanitAD/taniteval/results/dtac1_substrate_refc-base-30k.pt',map_location='cpu',weights_only=False)
log5,lat,lon,man5=D['log5'],D['lat'],D['lon'],D['man5']; n=len(lon)
LK,TL,TR,AC,BS=0,1,2,3,4
p=torch.log_softmax(log5.double(),-1).exp()
p_lat=torch.stack([p[:,LK]+p[:,AC]+p[:,BS],p[:,TL],p[:,TR]],1)
p_lon=torch.stack([p[:,BS]/p_lat[:,0].clamp_min(1e-30),p[:,LK]/p_lat[:,0].clamp_min(1e-30),
                   p[:,AC]/p_lat[:,0].clamp_min(1e-30)],1)
prior=torch.tensor([float((lon==i).sum())/n for i in range(3)],dtype=torch.double)
pred5=log5.argmax(1)
out={}
# MATCHED denominator: score the 5-WAY ARGMAX on the SAME 3-way lon label
lon_from5=torch.full((n,),1,dtype=torch.long)
lon_from5[pred5==AC]=2; lon_from5[pred5==BS]=0
def rep(pred,tag):
    M=torch.zeros(3,3,dtype=torch.long)
    for t,q in zip(lon.tolist(),pred.tolist()): M[t,q]+=1
    r=[float(M[i,i])/float(M[i].sum()) for i in range(3)]
    prec=[(float(M[i,i])/float(M[:,i].sum()) if M[:,i].sum()>0 else float('nan')) for i in range(3)]
    out[tag]={'recall':[round(x,4) for x in r],'precision':[round(x,4) for x in prec],
              'macroR':round(sum(r)/3,4),'acc':round(float(M.diag().sum())/n,4),
              'npred':M.sum(0).tolist(),'ntrue':M.sum(1).tolist()}
rep(lon_from5,'A_5way_argmax_scored_on_the_3way_lon_label  [MATCHED]')
rep(p_lon.argmax(1),'B_factored_raw_tau0                        [MATCHED]')
rep((p_lon.clamp_min(1e-30).log()-0.5*prior.log()[None,:]).argmax(1),'C_factored_adjusted_tau0.5          [MATCHED]')
# lon-active detection: predicted-active rate + precision
for tag,pr in [('5way',lon_from5),('tau0',p_lon.argmax(1)),
               ('tau0.5',(p_lon.clamp_min(1e-30).log()-0.5*prior.log()[None,:]).argmax(1))]:
    act=(pr!=1); tru=(lon!=1)
    out[f'lon_active_{tag}']={'pred_active_frac':round(float(act.float().mean()),4),
        'true_active_frac':round(float(tru.float().mean()),4),
        'precision':round(float((act&tru).sum())/max(1,int(act.sum())),4),
        'recall':round(float((act&tru).sum())/int(tru.sum()),4)}
print(json.dumps(out,indent=1))
