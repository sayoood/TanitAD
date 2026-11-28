"""DECN full-n probe, step 1: every remaining trained clip (the pool order of select_extra.py, after its first 859). Writes clips_full.json + .sha256 BEFORE any pull."""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DECN = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, DECN)
import select_extra as SE


def main():
    ents = {}
    for ln in open(f"{SE.THEORY}/wmact/train_stream/box2_meta/manifest.jsonl", encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln); ents[d["clip"]] = d
    m141 = json.load(open(f"{SE.THEORY}/wmact/train_stream/clips141.json", encoding="utf-8"))
    ce = json.load(open(f"{DECN}/clips_extra.json", encoding="utf-8"))
    import numpy as np
    ev_files = {f[:-3] for f in os.listdir(f"{SE.SF}/cache/eval/clips") if f.endswith(".pt")}
    ev_join = {str(c) for c in np.load(f"{SE.RAW}/join_eval141.npz", allow_pickle=True)["clip"]}
    eval_clips = sorted(ev_files | ev_join)
    pool, n_trained, n_pool = SE.select(ents, m141["clips"], eval_clips, n=10 ** 9)
    assert pool[:SE.N_EXTRA] == ce["clips"], "the pool order differs from DECN's 859"
    rest = pool[SE.N_EXTRA:]
    assert len(rest) == n_pool - SE.N_EXTRA and not (set(rest) & set(eval_clips)) and not (set(rest) & set(m141["clips"]))
    out = {"salt": SE.SALT, "n_trained": n_trained, "n_pool": n_pool, "clips": rest, "md5": {c: ents[c]["md5"] for c in rest}, "n_frames": {c: ents[c]["n_frames"] for c in rest}, "bytes": {c: ents[c]["bytes"] for c in rest},
           "clips_sha256": hashlib.sha256("\n".join(rest).encode()).hexdigest(), "total_bytes": sum(ents[c]["bytes"] for c in rest)}
    p = os.path.join(HERE, "clips_full.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    open(os.path.join(HERE, "clips_full.sha256"), "w").write(f"{h}  clips_full.json\n")
    print(len(rest), n_trained, n_pool, out["total_bytes"], h)


if __name__ == "__main__":
    main()
