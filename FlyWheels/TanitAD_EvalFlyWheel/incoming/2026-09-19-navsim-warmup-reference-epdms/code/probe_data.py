import pickle, pathlib, sys, collections, yaml
root = pathlib.Path("C:/Users/Admin/navsim/data/openscene/warmup_two_stage/synthetic_scene_pickles")
class U(pickle.Unpickler):
    def find_class(self, m, n):
        if m == "pathlib" and n == "PosixPath":
            return pathlib.PurePosixPath
        return super().find_class(m, n)
files = sorted(root.iterdir())
nframes = collections.Counter(); nh = collections.Counter(); nf = collections.Counter(); ext = collections.Counter()
toklen = collections.Counter()
for p in files:
    with open(p, "rb") as f:
        d = U(f).load()
    md = d["scene_metadata"]
    nframes[len(d["frames"])] += 1
    nh[md["num_history_frames"]] += 1; nf[md["num_future_frames"]] += 1
    edt = d.get("extended_detections_tracks"); etl = d.get("extended_traffic_light_data")
    ext[(None if edt is None else len(edt), None if etl is None else len(etl))] += 1
    toklen[len(md["initial_token"])] += 1
print("n_files", len(files))
print("frames per synthetic scene", dict(nframes))
print("num_history_frames", dict(nh), "num_future_frames", dict(nf))
print("(len ext_detections, len ext_tl)", dict(ext))
print("initial_token lengths", dict(toklen))
print("keys", sorted(d.keys())); print("metadata keys", sorted(md.keys()))
print("example md", {k: md[k] for k in md if k != "map_name"})
