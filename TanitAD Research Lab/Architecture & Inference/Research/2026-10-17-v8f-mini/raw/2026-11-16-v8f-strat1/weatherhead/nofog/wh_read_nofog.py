"""wh_read_nofog.py -- the weather head re-scored with fog DROPPED (PI 2026-11-17: "Drop fog for now"; AMEND_WH_NOFOG.md).
Zero training.  Same T0 dumps, windows, estimator, controls and outcome rules as wh_read.py (imported, not forked); two changes only:
  1. weather windows whose label is fog are unscored (label -> -1), for every arm, the shipped head and the controls;
  2. the weather prediction is the argmax over clear / rain / snow (the fog logit is masked to -inf before the argmax).
K0 (the reader reproduces the banked STRAT1 rows) is checked on the UNMASKED world, as in the main read; K-ID reads the raw dumps and is unaffected.
The probe beside-row (no bar) is left unmasked and is labelled so.
usage: python wh_read_nofog.py --t0 <run2/t0> --out <dir> --kid <run2/kid> --instrument <run2/instrument.json>"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "F:/TanitAD-scratch/strat1/whead/build")
import wh_read as RD  # noqa: E402

WEATHER_K = list(RD.VARS).index("weather")


def weather_cols(world):
    """(start, stop) of the weather block inside the 41 logits, found from split_logits on a probe array"""
    n = None
    probe = np.arange(41, dtype=np.float64)[None, :]
    parts = RD.R.split_logits(probe)
    w = parts[WEATHER_K][0]
    start = int(w[0])
    n = int(len(w))
    assert np.array_equal(w, np.arange(start, start + n)), "weather block is not contiguous"
    return start, n


def fog_id(world):
    """the fog class id in the label table's weather vocabulary"""
    tab = world.tab
    for key in ("weather_vocab", "vocab_weather"):
        if key in tab:
            vocab = [str(x) for x in tab[key]]
            return vocab.index("fog"), vocab
    raise SystemExit("REFUSED: the label table carries no weather vocabulary; fog's id cannot be read from source")


class NoFogWorld(RD.World):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fog, self.vocab = fog_id(self)
        self.w0, self.wn = weather_cols(self)
        yk = self.y[:, WEATHER_K]
        self.n_fog_windows = int((yk == self.fog).sum())
        self.n_fog_episodes = int(len(np.unique(self.eid[yk == self.fog])))
        yk[yk == self.fog] = -1

    def mask(self, logits):
        lg = np.array(logits, dtype=np.float64, copy=True)
        lg[:, self.w0 + self.fog] = -np.inf
        return lg

    def rows(self, logits):
        return super().rows(self.mask(logits))

    def correct(self, logits):
        return super().correct(self.mask(logits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kid", default="")
    ap.add_argument("--instrument", default="")
    ap.add_argument("--n-boot", type=int, default=RD.L.NB)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    full = RD.World(a.n_boot)
    w = NoFogWorld(a.n_boot)
    k0_orig = RD.k0
    RD.k0 = lambda world, *x, **y: k0_orig(full, *x, **y)          # K0 on the unmasked world (it reproduces the banked rows)
    ins = json.load(open(a.instrument, encoding="utf-8")) if a.instrument else {}
    rep = RD.read_all(w, a.t0, a.kid, ins, False)
    rep["amendment"] = {"name": "AMEND_WH_NOFOG", "fog_id": w.fog, "vocab": w.vocab, "weather_cols": [w.w0, w.wn],
                        "fog_windows_unscored": w.n_fog_windows, "fog_episodes": w.n_fog_episodes,
                        "probe_beside_row": "UNMASKED (beside, no bar): the probe may still predict fog"}
    json.dump(rep, open(f"{a.out}/WH_READ_NOFOG.json", "w", encoding="utf-8"), indent=1, default=float)
    print("WROTE", a.out, "| fog windows unscored", w.n_fog_windows, "|",
          json.dumps({k: (v.get("outcome") if isinstance(v, dict) and "outcome" in v else v) for k, v in rep["outcomes"].items()})[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
