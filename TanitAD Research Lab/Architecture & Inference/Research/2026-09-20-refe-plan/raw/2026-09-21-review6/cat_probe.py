import sqlite3,sys,glob,os
files=sorted(glob.glob(r"D:\Projects\TanitAD\data\nuplan\dblinks\driverl_val14\*.db"))[:5]
for p in files:
    con=sqlite3.connect("file:%s?mode=ro"%p.replace("\\","/"),uri=True)
    cats=[r[0] for r in con.execute("SELECT name FROM category").fetchall()]
    negro=[r[0] for r in con.execute("SELECT name FROM category WHERE lower(name) LIKE '%ego%'").fetchall()]
    print(os.path.basename(p)[:40], "| n_categories=",len(cats), "| ego-like:",negro)
    print("    categories:",cats)
    con.close()
    break
