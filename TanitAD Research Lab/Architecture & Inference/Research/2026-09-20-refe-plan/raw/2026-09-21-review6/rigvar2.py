"""FIXED: unpickle the camera translation instead of truncating the blob."""
import sqlite3,glob,os,collections,pickle,json
files=sorted(glob.glob(r"D:\Projects\TanitAD\data\nuplan\dblinks\driverl_val14\*.db"))
by_veh=collections.defaultdict(set); nlog=collections.Counter(); errs=0
for p in files:
    con=sqlite3.connect("file:%s?mode=ro"%p.replace("\\","/"),uri=True)
    try:
        vn=con.execute("SELECT vehicle_name FROM log").fetchone()[0]
        row=con.execute("SELECT translation,rotation FROM camera WHERE channel='CAM_F0'").fetchone()
        if row is None: continue
        t=pickle.loads(row[0]); r=pickle.loads(row[1])
        by_veh[vn].add(tuple(round(float(x),6) for x in list(t)))
        nlog[vn]+=1
    except Exception as e:
        errs+=1; print("ERR",os.path.basename(p),repr(e)[:90])
    finally: con.close()
print("logs probed=%d  errors=%d"%(len(files),errs))
allt=set()
for vn in sorted(by_veh):
    print("  %-8s n_logs=%2d  distinct CAM_F0 translations=%d  ->  %s"%(
        vn,nlog[vn],len(by_veh[vn]),sorted(by_veh[vn])[0]))
    allt|=by_veh[vn]
print()
print("TOTAL distinct CAM_F0 translation vectors across the probed logs:",len(allt))
for t in sorted(allt): print("    ",t)
