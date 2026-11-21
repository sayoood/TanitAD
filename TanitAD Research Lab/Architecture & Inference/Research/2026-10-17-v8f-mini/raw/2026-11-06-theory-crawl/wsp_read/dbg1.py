import numpy as np, p1_return_map as P
D,n=P.load_arm("P10")
v=D["v"]; g=D["gp"]; m=(v>=3)&(v<8)
print(m.sum(), np.round(np.percentile(g[m]-v[m],[0,10,50,90,100]),2))
idx=np.where(m)[0]
o=idx[np.argsort(-(g[m]-v[m]))][:8]
for i in o: print(D["scene"][i],D["f0"][i],D["k"][i],"v",round(v[i],2),"gp",round(g[i],2),"ge",round(D["ge"][i],2),"hv",round(D["hv"][i],2),"hn",round(D["hn"][i],2))
print("v pct", np.round(np.percentile(v,[0,10,25,50,75,90,100]),2))
print("gp pct", np.round(np.percentile(g,[0,10,25,50,75,90,100]),2))
print("by tick count", np.bincount(D["k"]))
Dc,_=P.load_arm("C0C_s0")
print("C0C v pct", np.round(np.percentile(Dc["v"],[0,10,25,50,75,90,100]),2))
print("C0C by tick med v", [round(float(np.median(Dc["v"][Dc["k"]==k])),2) for k in range(3)], "P10", [round(float(np.median(v[D["k"]==k])),2) for k in range(3)], "human", [round(float(np.median(D["hv"][D["k"]==k])),2) for k in range(3)])
