"""Screen NuRec scenes in the seeded order of SCREEN_RULE.md. Reads only rig_trajectories.json and sequence_tracks.json per scene.

    python -I screen.py            (resumable: scenes already in raw/screen_cache.jsonl are not re-fetched)
Writes raw/SCREEN_TABLE.json, then raw/SCREEN.done (or raw/SCREEN.fail). Never prints or writes the token.
"""
import sys, os, json, hashlib, time, traceback
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_scenes/"
sys.path.insert(0, T + "code")
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hfc, t2lib

QUOTA_LAUNCH, QUOTA_GUARD, MAX_SCREENED, BATCH = 42, 20, 800, 24
RULE_SHA = open(T + "SCREEN_RULE_SHA.txt").read().split()[0]
CACHE = T + "raw/screen_cache.jsonl"


def screen_one(sc, dirent):
    rec = {"scene": sc, "clip_id": sc}
    nbytes = 0
    try:
        rig = t2lib.fetch_member(sc, "rig_trajectories.json", dirent["rig_trajectories.json"])
        trk = t2lib.fetch_member(sc, "sequence_tracks.json", dirent["sequence_tracks.json"])
        nbytes = len(rig) + len(trk)
        Tm, tus, stride, dt = t2lib.ego_from_rig(rig)
        tracks = t2lib.tracks_from_json(trk)
        rec.update(t2lib.analyse(Tm, tus, tracks))
        rec["stride"], rec["native_dt_us"] = stride, int(dt)
        rec["n_tracks"] = len(tracks[0])
        rec["labels_seen"] = sorted(set(tracks[1]))
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "unparseable"
        rec["exclusion_reason"] = (type(e).__name__ + ": " + str(e))[:200]
    rec["bytes_read"] = nbytes
    return rec


def main():
    uni = json.load(open(T + "raw/UNIVERSE.json"))
    dirs = json.load(open(T + "raw/USDZ_DIRECTORY_RENDERER_MEMBERS.json"))
    ids = sorted(dirs)
    assert len(ids) == uni["n_renderable_universe"]
    perm = np.random.default_rng(0).permutation(len(ids))
    order = [ids[i] for i in perm]
    order_sha = hashlib.sha256("\n".join(order).encode()).hexdigest()
    cache = {}
    if os.path.exists(CACHE):
        for ln in open(CACHE, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln); cache[r["scene"]] = r
    n_launch = n_guard = 0
    table, stop_reason, prefetched_unused = [], None, 0
    t0 = time.time()
    pool = ThreadPoolExecutor(8)
    pos = 0
    while pos < len(order) and stop_reason is None:
        chunk = order[pos:pos + BATCH]
        todo = [s for s in chunk if s not in cache]
        for r in pool.map(lambda s: screen_one(s, dirs[s]), todo):
            cache[r["scene"]] = r
            with open(CACHE, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(r) + "\n")
        for k, sc in enumerate(chunk):
            r = dict(cache[sc]); r["order_index"] = pos + k
            r["selected"], r["slot"] = False, None
            if r["status"] == "ok":
                if r["is_launch"] and n_launch < QUOTA_LAUNCH:
                    n_launch += 1; r["selected"], r["slot"] = True, f"launch_{n_launch}"
                elif r["is_guard"] and n_guard < QUOTA_GUARD:
                    n_guard += 1; r["selected"], r["slot"] = True, f"guard_{n_guard}"
            table.append(r)
            if n_launch >= QUOTA_LAUNCH and n_guard >= QUOTA_GUARD:
                stop_reason = "both quotas met"
            elif len(table) >= MAX_SCREENED:
                stop_reason = "800 screened"
            if stop_reason:
                prefetched_unused = len(chunk) - k - 1
                break
        pos += BATCH
        print(f"screened {len(table)} launch {n_launch} guard {n_guard} t={round(time.time() - t0)}s", flush=True)
        json.dump({"partial": True, "n_screened": len(table), "n_launch": n_launch, "n_guard": n_guard}, open(T + "raw/SCREEN_TABLE.partial.json", "w"))
    if stop_reason is None:
        stop_reason = "universe exhausted"
    ok = [r for r in table if r["status"] == "ok"]
    c = lambda f: sum(1 for r in ok if f(r))
    summary = {
        "n_screened": len(table), "stop_reason": stop_reason, "quota_launch": QUOTA_LAUNCH, "quota_guard": QUOTA_GUARD,
        "launch_scenes_found": c(lambda r: r["is_launch"]), "launch_selected": n_launch,
        "guard_scenes_found": c(lambda r: r["is_guard"]), "guard_selected": n_guard,
        "neither": c(lambda r: not r["is_launch"] and not r["is_guard"]), "unparseable": len(table) - len(ok),
        "launch_selected_lead_yes": sum(1 for r in table if r["selected"] and r["slot"].startswith("launch") and r["launch_events"][0]["lead"]),
        "launch_selected_lead_no": sum(1 for r in table if r["selected"] and r["slot"].startswith("launch") and not r["launch_events"][0]["lead"]),
        "launch_events_total_in_selected": sum(len(r["launch_events"]) for r in table if r["selected"] and r["slot"].startswith("launch")),
        "prefetched_unused_after_stop": prefetched_unused,
        "bytes_read_screened": sum(r.get("bytes_read", 0) for r in table),
        "max_tick_dt_dev_s_over_screened": max([r["max_tick_dt_dev_s"] for r in ok if r.get("max_tick_dt_dev_s") is not None] or [None]),
        "labels_seen_union": sorted({l for r in ok for l in r.get("labels_seen", [])}),
    }
    out = {"rule_sha256": RULE_SHA, "repo": hfc.REPO, "revision": t2lib.REV, "numpy": np.__version__,
           "order_sha256": order_sha, "order_first10": order[:10], "universe": {k: uni[k] for k in (
               "n_release", "n_train_excluded", "n_after_train", "n_directory_read_errors", "n_missing_members", "n_renderable_universe")},
           "summary": summary, "scenes": table}
    tmp = T + "raw/SCREEN_TABLE.json.tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, T + "raw/SCREEN_TABLE.json")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    try:
        main()
        open(T + "raw/SCREEN.done", "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    except BaseException:
        open(T + "raw/SCREEN.fail", "w").write(traceback.format_exc())
        raise
