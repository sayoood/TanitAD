import json, sqlite3
D="C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/"
ROOT="D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval/"
inv=json.load(open(D+"inventory_db.json"))["neither_trainval_logs"]
want={"us-nv-las-vegas-strip":None,"us-pa-pittsburgh-hazelwood":None,"us-ma-boston":None,"sg-one-north":None}
for l in inv:
    if all(want.values()): break
    con=sqlite3.connect(f"file:{ROOT}{l}.db?mode=ro",uri=True)
    m=con.execute("SELECT map_version FROM log").fetchone()[0]; c=con.execute("SELECT count(*) FROM lidar_pc").fetchone()[0]; con.close()
    if m in want and want[m] is None and 2500<=c<=4000: want[m]=(l,c)
print(json.dumps(want)); open(D+"sample/sample_logs.txt","w").write(",".join(v[0] for v in want.values()))
