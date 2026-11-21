"""PREREG_V8F_DECN step 1: the 859 further train clips (IMPL_CHOICES_DECN.md section 2). Writes clips_extra.json + clips_extra.sha256 BEFORE any pull."""
import hashlib, json, os, sys
import numpy as np
THEORY = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory"
ART = "E:/Projects/TanitAD-artifacts"
SF = f"{ART}/sfresh_inputs"
RAW = f"{SF}/code/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
HERE = os.path.dirname(os.path.abspath(__file__))
SALT = "decn-v1"
N_EXTRA = 859


def select(manifest, clips141, eval_clips, salt=SALT, n=N_EXTRA):
    clips_all = list(manifest)
    sha12 = lambda c: hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]
    order = sorted(clips_all, key=sha12)
    excluded = set(order[540:600])
    trained = [c for c in clips_all if c not in excluded]
    assert not (set(eval_clips) & set(clips_all)), "an eval clip is in the train manifest"
    pool = [c for c in trained if c not in set(clips141) and c not in set(eval_clips)]
    pool.sort(key=lambda c: (hashlib.sha256((c + salt).encode("utf-8")).hexdigest(), c))
    return pool[:n], len(trained), len(pool)


def main():
    ents = {}
    for ln in open(f"{THEORY}/wmact/train_stream/box2_meta/manifest.jsonl", encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln); ents[d["clip"]] = d
    m141 = json.load(open(f"{THEORY}/wmact/train_stream/clips141.json", encoding="utf-8"))
    clips141 = m141["clips"]
    ev_files = {f[:-3] for f in os.listdir(f"{SF}/cache/eval/clips") if f.endswith(".pt")}
    j = np.load(f"{RAW}/join_eval141.npz", allow_pickle=True)
    ev_join = {str(c) for c in j["clip"]}
    eval_clips = sorted(ev_files | ev_join)
    pick, n_trained, n_pool = select(ents, clips141, eval_clips)
    assert len(pick) == N_EXTRA and len(set(pick)) == N_EXTRA
    assert not (set(pick) & set(clips141)) and not (set(pick) & set(eval_clips))
    out = {"salt": SALT, "n_eval_clips": len(eval_clips), "n_trained": n_trained, "n_pool": n_pool, "clips": pick,
           "md5": {c: ents[c]["md5"] for c in pick}, "n_frames": {c: ents[c]["n_frames"] for c in pick}, "bytes": {c: ents[c]["bytes"] for c in pick},
           "clips_sha256": hashlib.sha256("\n".join(pick).encode()).hexdigest(), "total_bytes": sum(ents[c]["bytes"] for c in pick)}
    p = os.path.join(HERE, "clips_extra.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    open(os.path.join(HERE, "clips_extra.sha256"), "w").write(f"{h}  clips_extra.json\n")
    print({k: v for k, v in out.items() if k not in ("clips", "md5", "n_frames", "bytes")}, "file sha256", h)


if __name__ == "__main__":
    main()
