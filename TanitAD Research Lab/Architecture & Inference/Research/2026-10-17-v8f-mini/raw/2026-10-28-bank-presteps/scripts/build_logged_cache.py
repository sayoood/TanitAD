"""One torch load of native_tracks.pt -> raw/logged_cache.npz (numpy only afterwards): for the bank's 4,512 TRAIN clips
the logged 0.2 s grid (x, y, yaw), the logged speed (track v column, even raw frames) and the clip lengths. Requires bank tables to exist.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bankio as B

def main():
    print("free RAM:", round(B.require_ram(3.0), 2), "GB")
    names = None
    for seed in ("s0", "s1"):
        z = np.load(B.RAW / f"bank_table_{seed}.npz")
        n = z["clip_names"].astype(str)
        names = n if names is None else names
        assert (names == n).all(), "the two banks list different clips"
    tracks = B.load_tracks()
    offs, Ts, G, V = B.clip_offsets(names, tracks)
    np.savez(B.RAW / "logged_cache.npz", names=names, offs=offs, Ts=Ts, G=G, V=V, tracks_sha256=np.array(B.TRACKS_SHA))
    print("clips", len(names), "rows", len(G), "finite v", int(np.isfinite(V).sum()))

if __name__ == "__main__":
    main()
