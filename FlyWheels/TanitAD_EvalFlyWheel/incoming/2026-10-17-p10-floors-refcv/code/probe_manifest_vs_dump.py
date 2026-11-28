import numpy as np, torch, math
from tanitad.models.kinematic import kappa_of_steer, STEER_WHEELBASE_M
from tanitad.refs.refa_v1_plan import unicycle_paths
from tanitad.models.metric_dynamics import gt_ego_waypoints
m=torch.load('eval139_v2manifest.pt', map_location='cpu', weights_only=False)
R='E:/refcv7_eval_kit/battery/step50400_a6text/'
for tag,H in (('dump_s0',[5,10,15,20]),('s6_s0',[10,20,30,40,50,60])):
  worst={'v0':0,'g':0,'ha0':0,'ha0_ext':0}
  for ei in (0,1,2,50,100,138):
    z=np.load(f'{R}{tag}/ep{ei:03d}.npz')
    P=m['poses'][int(z['clip_index'][0])]; A=m['actions'][int(z['clip_index'][0])]
    v=P[:,3]; ws=z['ws']
    for j,t0 in enumerate(ws):
        t0=int(t0)
        v0=float(v[t0]); worst['v0']=max(worst['v0'],abs(v0-float(z['v0'][j])))
        fut=P[t0+1:t0+max(H)+1]
        g=gt_ego_waypoints(P[t0][None],fut[None],H)[0].numpy()
        worst['g']=max(worst['g'],float(np.abs(g-z['g'][j]).max()))
        a0=float((v[t0]-v[t0-1])/0.1); st=float(A[t0,0])
        n=max(H)
        ext=torch.tensor([[a0,st]]).float().expand(n,2)
        path=unicycle_paths(ext[None],torch.tensor([v0]),0.1,action_units='steer')[0].numpy()
        sel=path[[h-1 for h in H]]
        worst['ha0_ext']=max(worst['ha0_ext'],float(np.abs(sel-z['ha0_ext'][j]).max()))
        p0=unicycle_paths(torch.zeros(1,n,2),torch.tensor([v0]),0.1,action_units='steer')[0].numpy()[[h-1 for h in H]]
        worst['ha0']=max(worst['ha0'],float(np.abs(p0-z['ha0'][j]).max()))
  print(tag, worst)
print(STEER_WHEELBASE_M, float(kappa_of_steer(torch.tensor([0.1]),STEER_WHEELBASE_M)), math.tan(0.1)/2.9)
