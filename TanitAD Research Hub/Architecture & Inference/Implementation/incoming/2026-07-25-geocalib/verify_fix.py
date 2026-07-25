"""Verify the single-threaded-decode fix: mirror the parallel agent's per-clip
flow (decode_canonical_geocalib with estimator= -> estimate(CUDA)+decode, repeated)
across several clips. Global watchdog so a regression REPORTS instead of hanging."""
import signal, time
import geocalib_intrinsics as gi


def _watchdog(s, f):
    print("GLOBAL_WATCHDOG_FIRED — still hanging", flush=True)
    raise SystemExit(2)


signal.signal(signal.SIGALRM, _watchdog)
signal.alarm(240)


class Stub:
    def __init__(self):
        self.stats = {"frames": 0}
    def __call__(self, rgb):
        self.stats["frames"] += 1
        return rgb


est = gi.GeoCalibEstimator()
files = ["testvid_pai.mp4", "testvid_comma.hevc", "testvid_pai.mp4", "testvid_comma.hevc"]
for i, f in enumerate(files):
    t0 = time.time()
    fr, meta = gi.decode_canonical_geocalib(f, Stub(), estimator=est, max_frames=8)
    print(f"[{i}] {f:20s} shape {tuple(fr.shape)} f_eff {meta['achieved_f_eff']} "
          f"canon {meta['fully_canonical']} hfov {round(meta['hfov_used_deg'],1)} "
          f"fb {meta['geocalib_fallback_used']}  ({time.time()-t0:.1f}s)", flush=True)
signal.alarm(0)
print("MULTICLIP_OK — estimate(CUDA)+decode ran clean across all clips, no hang", flush=True)
