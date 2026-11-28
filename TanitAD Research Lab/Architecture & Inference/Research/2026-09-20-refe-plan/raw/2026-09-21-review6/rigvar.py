"""Does the DB record ANY per-vehicle physical variation (vs none at all)?
Probes camera extrinsics per log; groups by log.vehicle_name.
CONTROL: ego_pose columns printed in full -- must contain pose/vel/acc and NO dimension."""
import sqlite3,glob,os,collections,json,statistics
files=sorted(glob.glob(r"D:\Projects\TanitAD\data\nuplan\dblinks\driverl_val14\*.db"))
by_veh=collections.defaultdict(list)
egocols=None
for p in files:
    con=sqlite3.connect("file:%s?mode=ro"%p.replace("\\","/"),uri=True)
    try:
        if egocols is None:
            egocols=[r[1] for r in con.execute("PRAGMA table_info(ego_pose)")]
            camcols=[r[1] for r in con.execute("PRAGMA table_info(camera)")]
        vn=con.execute("SELECT vehicle_name FROM log").fetchone()[0]
        row=con.execute("SELECT channel,translation FROM camera WHERE channel='CAM_F0'").fetchone()
        if row: by_veh[vn].append((os.path.basename(p),str(row[1])[:80]))
    except Exception as e:
        by_veh['ERR'].append((os.path.basename(p),str(e)[:60]))
    finally: con.close()
print("ego_pose columns (%d):"%len(egocols),egocols)
print("  -> any dimension-named col in ego_pose?",[c for c in egocols if any(k in c.lower() for k in ('width','length','wheel','height','size'))])
print("camera columns (%d):"%len(camcols),camcols)
print()
print("CAM_F0 translation, one sample per vehicle_name (n_logs probed=%d):"%len(files))
for vn in sorted(by_veh):
    vals={t for _,t in by_veh[vn]}
    print("  %-8s n_logs=%2d distinct_translations=%d  e.g. %s"%(vn,len(by_veh[vn]),len(vals),by_veh[vn][0][1][:70]))
allvals={t for v in by_veh.values() for _,t in v}
print()
print("TOTAL distinct CAM_F0 translations across all logs:",len(allvals),"(control: >1 means the DB DOES record per-rig physical variation)")
