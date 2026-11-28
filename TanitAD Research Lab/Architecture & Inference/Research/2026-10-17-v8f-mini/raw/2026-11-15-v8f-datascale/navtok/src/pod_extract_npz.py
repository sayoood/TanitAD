import pyarrow.parquet as pq, numpy as np, json
t=pq.read_table("nav_sub.parquet")
print(t.schema.metadata)
texts=json.load(open("texts.json")); ix={s:i for i,s in enumerate(texts)}
cid=t["clip_id"].to_pylist(); ts=t["timestamp"].to_numpy(); tx=np.array([ix[s] for s in t["nav_text"].to_pylist()],np.int32)
u=sorted(set(cid)); ci={c:i for i,c in enumerate(u)}
c=np.array([ci[x] for x in cid],np.int32)
o=np.lexsort((ts,c))
np.savez_compressed("nav_sub.npz",clips=np.array(u),clip_i=c[o],ts_us=ts[o],text_i=tx[o])
print(len(u),len(c),ts.min(),ts.max())
