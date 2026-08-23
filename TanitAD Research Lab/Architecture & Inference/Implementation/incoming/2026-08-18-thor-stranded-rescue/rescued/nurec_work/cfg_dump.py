import msgpack, json, numpy as np
obj = msgpack.unpackb(open("/home/nvidia/nurec_work/x/volume.msgpack","rb").read(), raw=False, strict_map_key=False)
cfg = obj["nre_data"]["config"]
for k in cfg:
    v = cfg[k]
    if k == "layers":
        print(f"--- {k}: keys={list(v.keys())}")
        continue
    print(f"--- {k} ---")
    print(json.dumps(v, indent=1, default=str)[:3000])
