import sqlite3, sys, os, glob
D = "/workspace/data/navtrain_dbs"
for tag, lg in [(t.split(":")[0], t.split(":")[1]) for t in sys.argv[1:]]:
    fp = os.path.join(D, lg + ".db")
    if not os.path.exists(fp):
        c = glob.glob(os.path.join(D, "**", lg + ".db"), recursive=True); fp = c[0] if c else None
    if fp is None: print(tag, lg, "NO DB"); continue
    con = sqlite3.connect(f"file:{fp}?mode=ro", uri=True)
    n, nz = con.execute("select count(*), sum(case when qx != 0 or qy != 0 then 1 else 0 end) from ego_pose").fetchone()
    r = con.execute("select qw, qx, qy, qz from ego_pose limit 1").fetchone()
    print(tag, lg, "ego_pose rows", n, "rows with qx|qy != 0:", nz, "sample", r)
