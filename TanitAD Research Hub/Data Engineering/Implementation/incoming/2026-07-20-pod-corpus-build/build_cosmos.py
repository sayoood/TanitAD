import os, sys, json, hashlib, time
sys.path.insert(0, '/root/TanitAD/stack')
sys.path.insert(0, '/root/TanitAD/stack/scripts')
sys.path.insert(0, '/root/taniteval')
import truststore; truststore.inject_into_ssl()
from pathlib import Path

SHARD = '/root/cosmos_dl/cosmos_synthetic/single_view/generation.tar.gz.part-000'
OUT = Path('/root/cosmos_data/pairs')
N_PAIRS = int(os.environ.get('N_PAIRS', '24'))


def main():
    import cosmos_pairs as cp
    t0 = time.time()
    print('[cosmos] pass1 scan index (decompresses the 43GB shard)...', flush=True)
    groups = cp.scan_index(SHARD)
    wanted = cp.select_pairs(groups, N_PAIRS)
    if not wanted:
        raise SystemExit('no clear+degraded pairs in shard slice')
    cp.extract(SHARD, wanted, OUT)
    cp.fetch_poses(OUT)
    manifest = {}
    for name, (base, chunk) in wanted.items():
        manifest.setdefault('%s_%s' % (base, chunk), []).append(Path(name).name)
    (OUT / 'pairs_manifest.json').write_text(json.dumps(manifest, indent=1))
    print('[cosmos] pairs staged in %.0fs' % (time.time() - t0), flush=True)

    # --- build epcache from the extracted clips ---
    from tanitad.data.cosmos_drive import discover_clips, build_episode, verify_real_clip
    from tanitad.data.mixing import save_episode
    clips = discover_clips(str(OUT), camera_subdir='generation')
    print('[cosmos] discover_clips: %d clips' % len(clips), flush=True)
    if not clips:
        raise SystemExit('discover_clips found 0 — layout mismatch')
    v = verify_real_clip(clips[0])
    print('[cosmos] verify_real_clip:', json.dumps(v), flush=True)

    h = hashlib.sha1(('cosmos|part000|pairs|n%d' % N_PAIRS).encode()).hexdigest()[:12]
    epdir = Path('/root/valdata/cosmos-val-%s' % h)
    epdir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for i, clip in enumerate(clips):
        try:
            ep = build_episode(clip, size=256)
            save_episode(ep, str(epdir / ('ep_%05d.pt' % i)))
            ok += 1
        except Exception as e:
            fail += 1
            print('[cosmos] skip %s: %s: %s' % (clip.get('clip_id'), type(e).__name__, e), flush=True)
    print('COSMOS_BUILD_DONE ok=%d fail=%d epdir=%s %.0fs' % (ok, fail, epdir, time.time() - t0), flush=True)
    print('COSMOS_EPDIR=%s' % epdir, flush=True)


if __name__ == '__main__':
    main()
