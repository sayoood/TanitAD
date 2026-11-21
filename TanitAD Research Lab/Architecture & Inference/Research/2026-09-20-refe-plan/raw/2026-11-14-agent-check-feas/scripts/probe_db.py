import sqlite3, time, json
p="file:E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval/2021.06.07.11.59.52_veh-35_00008_00083.db?mode=ro&immutable=1"
c=sqlite3.connect(p, uri=True)
for t in ("lidar_box","track","category","lidar_pc","ego_pose","lidar","scene","scenario_tag","traffic_light_status"):
    cols=[(r[1],r[2]) for r in c.execute(f"pragma table_info({t})")]
    n=c.execute(f"select count(*) from {t}").fetchone()[0]
    print(t,n,cols)
print(c.execute("select name,description from category").fetchall())
print(c.execute("select c.name,count(*) from lidar_box b join track t on b.track_token=t.token join category c on t.category_token=c.token group by c.name").fetchall())
r=c.execute("select * from lidar_box limit 1").fetchone(); print([ (x if not isinstance(x,bytes) else x.hex()[:8]) for x in r])
# time cost of one token's agent query: boxes in window t0..t0+8s
pc=c.execute("select token,timestamp from lidar_pc order by timestamp").fetchall(); print("lidar_pc",len(pc),"span s",(pc[-1][1]-pc[0][1])/1e6, "hz",len(pc)/((pc[-1][1]-pc[0][1])/1e6))
t0=time.time()
mid=pc[len(pc)//2][1]
rows=c.execute("select b.x,b.y,b.z,b.yaw,b.width,b.length,b.height,b.vx,b.vy,b.track_token,p.timestamp,b.confidence from lidar_box b join lidar_pc p on b.lidar_pc_token=p.token where p.timestamp between ? and ?",(mid,mid+8_000_000)).fetchall()
print("one-token window query: rows",len(rows),"s",round(time.time()-t0,4))
print(rows[0])
print("confidence uniq",set(r[-1] for r in rows[:2000]))
