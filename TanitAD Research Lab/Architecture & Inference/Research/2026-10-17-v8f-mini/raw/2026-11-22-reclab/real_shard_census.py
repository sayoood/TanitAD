"""Real-shard census + reachability proof for --dagger-label speed_v1 (CPU, one python process). Inputs pulled read-only from the pod into tmp/realshard/ (deleted afterwards)."""
import hashlib, json, sys, time
sys.path.insert(0, "F:/TanitAD-scratch/reclab/tree97/stack")
sys.path.insert(0, "F:/TanitAD-scratch/reclab/tree97/taniteval")
import numpy as np, torch
import tanitad
from tanitad.data import v8f_dagger as G
from tanitad.data import v8f_recovery as RV
assert tanitad.__file__.replace("\\", "/").startswith("F:/TanitAD-scratch/reclab/tree97/"), tanitad.__file__
R = "F:/TanitAD-scratch/reclab/tmp/realshard/"
t0 = time.time()
arrays, header = G.load_shard(R + "shard_000of004.npz")
tracks = torch.load(R + "native_tracks_union.pt", map_location="cpu", weights_only=True)["tracks"]
join_clip = np.load(R + "join_union.npz", allow_pickle=False)["clip"]
print("loaded", len(arrays["ci"]), "records;", len(tracks), "tracks;", len(join_clip), "join clips", round(time.time() - t0, 1), "s", flush=True)
bank = G.DaggerBank(arrays, header, clip_names=join_clip, sha256=None, path=R)
n_ok = int((bank.reason == G.OK).sum())
eligible_human = bank._ok_idx.copy()


def read_clip(name):
    tr = np.asarray(tracks[name], np.float64)
    return {"pose": tr[0::2]}


t1 = time.time()
res = RV.label_bank(bank.a, bank.ci, join_clip, read_clip)
wall = time.time() - t1
bank.attach_recovery(res["slot"], res["label"], res["status_counts"])
elig_equal = bool(np.array_equal(bank._ok_idx, eligible_human)) and bool(np.array_equal(bank.eligible_mask, bank.reason == G.OK))
# reachability on the real records: labels that differ from the human label (speed_v1) vs the human kind (exactly 0 by construction: no labels attached)
has = bank.rl["slot"] >= 0
differ = 0
sh_changed = 0
for i in np.flatnonzero(has):
    c, r = int(bank.ci[i]), int(bank.a["r"][i])
    pg = read_clip(str(join_clip[c]))["pose"]
    hl = RV.human_label(pg[:, :3], r)
    lab = bank.rl["label"][bank.rl["slot"][i]]
    if not np.array_equal(lab, hl):
        differ += 1
rec = bank.gather(np.flatnonzero(has)[:5])
hb = G.DaggerBank(arrays, header, clip_names=join_clip)
out = {
    "base": "97fe0a56 + speed_v1 build (tree97)", "shard": "bank/s0 shard_000of004",
    "input_sha256": {"shard_000of004.npz": "aefbb46cd6849afb4e8fc9319b980293741ee37bcb87e86e622b587199907598", "native_tracks_union.pt": "9022952497659c897ccdbf5565e50046597550ed17a996da2cf8a271330c0926", "join_union.npz": "7d2be0e6379de435fe93d23ca74a2b33fd038b86b4ed467834c4926d90e913c0"},
    "n_records": int(bank.n), "reason_counts": bank.reason_counts(), "n_ok": n_ok,
    "label_bank_wall_s": round(wall, 1), "label_bank_wall_per_1000_ok_s": round(1000 * wall / max(n_ok, 1), 3),
    "status_counts": res["status_counts"], "census": res["census"],
    "reachability": {"speed_v1_records_with_label": int(has.sum()), "speed_v1_labels_differing_from_human_label": differ,
                     "human_kind_labels_attached": 0 if hb.rl is None else -1, "human_kind_label_kind": hb.label_kind, "human_gather_has_rl_keys": "rl_has" in hb.gather([0]),
                     "speed_v1_gather_has_rl_keys": "rl_has" in rec, "eligible_index_equal_under_both": elig_equal, "n_eligible_human": int(len(eligible_human)), "n_eligible_speed_v1": int(len(bank._ok_idx))},
    "total_wall_s": round(time.time() - t0, 1),
}
json.dump(out, open("F:/TanitAD-scratch/reclab/REAL_SHARD_CENSUS.json", "w"), indent=1)
print(json.dumps(out, indent=1))
