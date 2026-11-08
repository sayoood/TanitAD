import pickle,glob,os,json,gzip
R="F:/Archive/devbox-C/navsim"
exp=json.load(gzip.open(R+"/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz","rt",encoding="utf-8"))["tokens"]
M=R+"/data/openscene/navsim_logs/test/"
seen=0;bad=[];nomatch=0
for f in glob.glob(M+"*.pkl"):
    m=pickle.load(open(f,"rb")); ts=[int(x["timestamp"]) for x in m]
    for i,x in enumerate(m):
        if x["token"] not in exp: continue
        seen+=1; nx=ts[i+1:i+9]
        d=[(y-ts[i])/1e6 for y in nx]
        if len(nx)<8 or any(abs(d[k]-(k+1)*0.5)>0.1 for k in range(8)): bad.append((x["token"],os.path.basename(f)))
print(len(exp),seen,len(bad),len({b[1] for b in bad}))
json.dump({"n_export_tokens":len(exp),"n_found_in_meta":seen,"n_gap_tokens":len(bad),"n_logs_with_gap":len({b[1] for b in bad})},open("E:/Projects/TanitAD-artifacts/refe_selection_confirm_20261107/fc_diag/fc_all_navtest_meta_count.json","w"))
