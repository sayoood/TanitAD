"""TIMING ONLY: rasterise one cached token's agent polygons into [8, 160, 128] future-occupancy (0.5 m cells, DAC-store frame). Not saved."""
import sys, time, lzma, pickle, numpy as np
V1="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
sys.path[:0]=[V1,"F:/Archive/devbox-C/navsim/nuplan-devkit"]
mc=pickle.loads(lzma.decompress(open(sys.argv[1],"rb").read()))
from PIL import Image, ImageDraw
ob=mc.observation; ego=mc.ego_state.rear_axle; c,s=np.cos(ego.heading),np.sin(ego.heading)
def run():
    out=np.zeros((8,160,128),np.uint8); nper=[]
    for k in range(8):                      # t = 0.5 .. 4.0 s  (10 Hz index 5*(k+1))
        occ=ob[5*(k+1)]; im=Image.new("L",(128,160),0); dr=ImageDraw.Draw(im); n=0
        for tok,g in zip(occ.tokens,occ._geometries):
            if "red_light" in tok: continue
            xy=np.asarray(g.exterior.coords)[:, :2]-[ego.x,ego.y]
            lx=c*xy[:,0]+s*xy[:,1]; ly=-s*xy[:,0]+c*xy[:,1]       # ego (rear-axle) frame
            ix=(ly+32)/0.5; iy=(lx+8)/0.5                           # y in [-32,32) -> col, x in [-8,72) -> row
            if ix.max()<0 or ix.min()>=128 or iy.max()<0 or iy.min()>=160: continue
            dr.polygon(list(zip(ix,iy)),fill=1); n+=1
        out[k]=np.asarray(im); nper.append(n)
    return out,nper
run(); t=time.time(); out,nper=run(); dt=time.time()-t
print("raster 8 steps s",round(dt,3),"polys in grid per step",nper,"occupied cells per step",[int(x.sum()) for x in out])
