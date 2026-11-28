import json,collections,sys
for fp in sys.argv[1:]:
    R=[json.loads(l) for l in open(fp) if l.strip().endswith("}")]
    c=collections.Counter(r.get("skip") or ("arms" if "arms" in r else "err") for r in R)
    A=[r for r in R if "arms" in r]
    print(fp.split("/")[-1],len(R),dict(c))
    for arm in ("A1","L1","L1x"):
        print("  ",arm,sum(r["arms"][arm]["applied"] for r in A if arm in r["arms"]),"/",len(A),"viol",sum(r["arms"][arm].get("cert_violation",0) for r in A if arm in r["arms"]),collections.Counter(r["arms"][arm]["status"] for r in A if arm in r["arms"]))
    print("   A1 lat flips>=2:",sum(r["arms"]["A1"].get("lat_sign_flips",0)>=2 for r in A),"A1 iters>=3:",sum(len(r["arms"]["A1"]["extent_seq_m"])>=4 for r in A))
