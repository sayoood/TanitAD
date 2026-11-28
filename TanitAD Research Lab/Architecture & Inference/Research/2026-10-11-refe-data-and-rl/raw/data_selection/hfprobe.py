import sys, json, truststore
truststore.inject_into_ssl()
import requests
def tree(repo, path="", recursive=True):
    url=f"https://huggingface.co/api/datasets/{repo}/tree/main/{path}".rstrip("/")
    out=[]; params={"recursive":"true"} if recursive else {}
    while url:
        r=requests.get(url,params=params,timeout=60); r.raise_for_status()
        out+=r.json()
        url=r.links.get("next",{}).get("url"); params={}
    return out
if __name__=="__main__":
    repo=sys.argv[1]; path=sys.argv[2] if len(sys.argv)>2 else ""
    rec = (len(sys.argv)>3 and sys.argv[3]=="rec")
    t=tree(repo,path,rec)
    tot=0
    from collections import defaultdict
    g=defaultdict(lambda:[0,0])
    for e in t:
        if e["type"]=="file":
            size=e.get("lfs",{}).get("size",e.get("size",0)); tot+=size
            key="/".join(e["path"].split("/")[:3] if rec else e["path"].split("/")[:2])
            g[key][0]+=1; g[key][1]+=size
        else:
            if not rec: print("DIR",e["path"])
    for k,(n,s) in sorted(g.items()): print(f"{k:75s} files={n:4d} GB={s/1e9:9.2f}")
    print("TOTAL GB",tot/1e9)
