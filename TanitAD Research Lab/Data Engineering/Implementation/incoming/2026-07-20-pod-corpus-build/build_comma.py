import os, sys, hashlib, time
sys.path.insert(0, '/root/TanitAD/stack')
sys.path.insert(0, '/root/taniteval')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tanitad.data.comma2k19 import (discover_segments,
                                    sample_segments_across_routes, route_of)

RAW = '/root/comma_raw'
N = int(os.environ.get('N_EPS', '64'))
OUT_HASH = hashlib.sha1(
    ('comma2k19|Chunk_1|ftheta_v2|n%d|s2|ms300|stack3' % N).encode()
).hexdigest()[:12]
OUT = Path('/root/valdata/comma2k19-val-%s' % OUT_HASH)
OUT.mkdir(parents=True, exist_ok=True)


def work(args):
    idx, seg = args
    from tanitad.data.comma2k19 import build_episode
    from tanitad.data.mixing import save_episode
    try:
        ep = build_episode(Path(seg), size=256, stride=2, max_steps=300,
                           n_stack=3)
        save_episode(ep, str(OUT / ('ep_%05d.pt' % idx)))
        return (idx, seg, int(ep.frames.shape[0]), 'ok', None)
    except Exception as e:
        return (idx, seg, 0, 'fail', '%s: %s' % (type(e).__name__, e))


def main():
    segs = discover_segments(RAW)
    print('discovered %d segments / %d routes' %
          (len(segs), len({route_of(s) for s in segs})), flush=True)
    sel = sample_segments_across_routes(segs, N, seed=0)
    print('selected %d segments / %d routes' %
          (len(sel), len({route_of(s) for s in sel})), flush=True)
    tasks = list(enumerate(str(s) for s in sel))
    ok = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for f in as_completed(futs):
            idx, seg, n, st, err = f.result()
            if st == 'ok':
                ok += 1
            else:
                fail += 1
                print('FAIL idx=%d %s' % (idx, err), flush=True)
            if (ok + fail) % 8 == 0:
                print('progress %d/%d ok=%d fail=%d %.0fs' %
                      (ok + fail, len(tasks), ok, fail, time.time() - t0),
                      flush=True)
    print('BUILD_DONE ok=%d fail=%d %.0fs' % (ok, fail, time.time() - t0),
          flush=True)
    print('OUTDIR=%s' % OUT, flush=True)


if __name__ == '__main__':
    main()
