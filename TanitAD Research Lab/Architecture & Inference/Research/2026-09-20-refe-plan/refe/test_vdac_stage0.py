#!/usr/bin/env python3
"""Tests of vdac_stage0.py on SYNTHETIC fixtures only (a synthetic label store, synthetic banks, synthetic head outputs; no real data, no model).  CUDA_VISIBLE_DEVICES=-1, plain unittest:
    python test_vdac_stage0.py
Needs: the frozen s1_read.py (S1_READ_V1 or the repo layout), s1_read_v2.py beside / under ../eval (or S1_READ_V2), and the REAL H48 list for the sha-asserted exclusion path
(VDAC_H48_LOGS, or ../splits/h48_logs.txt in the repo layout; only its 48 NAMES are used to name synthetic logs)."""
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vdac_stage0 as S  # noqa: E402

R = S.R
PER_CITY, REPRO = 8, 6
MAPS = {"LV": "us-nv-las-vegas-strip", "BOS": "us-ma-boston", "PIT": "us-pa-pittsburgh-hazelwood", "SG": "sg-one-north"}
H48_PATH = os.environ.get("VDAC_H48_LOGS") or os.path.join(HERE, "..", "splits", "h48_logs.txt")


def road_label(c):
    """int8 SDF [160,128]: a road along x, 8 m wide, centre y = c (multiple of 0.5 m); + inside, 0.1 m units (as the S1a reader's tests)."""
    yf = R.Y0 + R.FINE_RES * (np.arange(R.NYF) + 0.5)
    s = np.clip(np.rint((4.0 - np.abs(yf - c)) / 0.1), -127, 127).astype(np.int8)
    return np.broadcast_to(s[None, :], (R.NXF, R.NYF)).copy()


def perfect_logit(label):
    frac, _ = R.block_targets(label[None])
    return np.where(frac[0] >= 0.5, 3.0, -3.0).astype(np.float32)


def write_shard(path, log, labels):
    ents, blobs, off = [], [], 0
    for t, lab in labels.items():
        b = zlib.compress(np.ascontiguousarray(lab, np.int8).tobytes())
        ents.append({"t": t, "o": off, "l": len(b)})
        blobs.append(b)
        off += len(b)
    head = json.dumps({"log": log, "entries": ents}).encode()
    with open(path, "wb") as f:
        f.write(b"AUXSDF1\n" + struct.pack("<I", len(head)) + head + b"".join(blobs))


class Fx:
    """a synthetic store: per city 2 'H48' logs (REAL H48 names, 40 frames each: they dominate the low hashes on purpose), 1 'SFT' log (10 frames), 4 ordinary logs (6 frames)."""

    def __init__(self, td, centres=(-2.0, 2.0)):
        h48 = open(H48_PATH).read().split()
        self.root = os.path.join(td, "store")
        os.makedirs(os.path.join(self.root, "shards"))
        self.city, self.h48, self.sft, self.ordinary, self.labels = {}, [], [], [], {}
        files = {}
        k = 0
        for ci, c in enumerate(S.CITIES):
            for kind, n_logs, n_fr in (("h48", 2, 40), ("sft", 1, 10), ("ord", 4, 6)):
                for i in range(n_logs):
                    lg = h48[k] if kind == "h48" else f"2021.0{ci + 1}.0{1 if kind == 'sft' else 2}.00.00.00_veh-{ci}_{i:05d}_{i + 9:05d}"
                    k += kind == "h48"
                    self.city[lg] = c
                    {"h48": self.h48, "sft": self.sft, "ord": self.ordinary}[kind].append(lg)
                    lab = {f"t{ci}{kind}{i}_{j:03d}": road_label(centres[(i + j) % len(centres)]) for j in range(n_fr)}
                    for t, l in lab.items():
                        self.labels[(lg, t)] = l
                    rel = f"shards/{lg}.aux1"
                    write_shard(os.path.join(self.root, rel), lg, lab)
                    with open(os.path.join(self.root, rel), "rb") as f:
                        files[rel] = {"bytes": 1, "sha256": hashlib.sha256(f.read()).hexdigest(), "n": len(lab), "map": MAPS[c]}
        sha = hashlib.sha256("\n".join(f"{k_}:{v['sha256']}" for k_, v in sorted(files.items())).encode()).hexdigest()
        json.dump({"schema": "aux-drv-label-v1", "store_sha256": sha, "files": files}, open(os.path.join(self.root, "manifest.json"), "w"))
        self.prefix = sha[:8]
        self.h48_path = H48_PATH
        self.sft_path = os.path.join(td, "sft.txt")
        open(self.sft_path, "w").write("\n".join(self.sft) + "\n")
        self.store = S.Store(self.root, self.prefix)
        self.excl = sorted(set(self.h48) | set(self.sft))
        self.bank = []
        for (lg, t) in sorted(self.labels):
            self.bank.append((lg, t, 0, 0))
            self.bank.append((lg, t, 1, 0))          # decoy: step 1
            self.bank.append((lg, t, 0, 3))          # decoy: rank 3
        for j in range(30):
            self.bank.append((self.ordinary[0], f"unlabelled{j}", 0, 0))       # decoy: no label in the store
        self.bank_path = os.path.join(td, "bank_keys.tsv")
        self.write_bank(self.bank)

    def write_bank(self, rows):
        with open(self.bank_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("log\ttoken\tstep\trank\n" + "".join(f"{a}\t{b}\t{c}\t{d}\n" for a, b, c, d in rows))


def write_pred(path, keys, logits, sdf=None):
    np.savez(path, log=np.asarray([k[0] for k in keys]), token=np.asarray([k[1] for k in keys]), step=np.zeros(len(keys), np.int64), logit=np.asarray(logits, np.float32),
             sdf_m=np.zeros_like(logits, np.float32) if sdf is None else sdf)


class TDraw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.fx = Fx(cls.td.name)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    # ------------------------------------------------------------------ the draw rule
    def test_draw_is_the_lowest_hashes_per_city_among_eligible(self):
        fx = self.fx
        rows = S.draw_list(fx.bank, fx.store, fx.excl, PER_CITY)
        self.assertEqual(len(rows), 4 * PER_CITY)
        for c in S.CITIES:
            elig = sorted((hashlib.sha256(f"{lg}|{t}".encode()).hexdigest(), lg, t) for (lg, t), _ in fx.labels.items() if fx.city[lg] == c and lg not in fx.excl)
            want = [(lg, t) for _, lg, t in elig[:PER_CITY]]
            got = [(r[0], r[1]) for r in rows if r[2] == c]
            self.assertEqual(got, want)
        self.assertEqual([r[2] for r in rows], [c for c in S.CITIES for _ in range(PER_CITY)])

    def test_draw_is_a_pure_function_row_order_and_repeat_do_not_matter(self):
        fx = self.fx
        a = S.list_text(S.draw_list(fx.bank, fx.store, fx.excl, PER_CITY))
        rng = np.random.default_rng(0)
        shuf = [fx.bank[i] for i in rng.permutation(len(fx.bank))]
        b = S.list_text(S.draw_list(shuf, fx.store, fx.excl, PER_CITY))
        self.assertEqual(a, b)
        self.assertEqual(hashlib.sha256(a.encode()).hexdigest(), hashlib.sha256(b.encode()).hexdigest())

    def test_ineligible_rows_are_never_drawn_and_the_next_one_takes_the_place(self):
        fx = self.fx
        rows = S.draw_list(fx.bank, fx.store, fx.excl, PER_CITY)
        first = (rows[0][0], rows[0][1])
        for tweak in ("step", "rank", "gone"):
            if tweak == "step":
                bank = [(l, t, 1 if (l, t) == first and s == 0 and r == 0 else s, r) for l, t, s, r in fx.bank]
            elif tweak == "rank":
                bank = [(l, t, s, 2 if (l, t) == first and s == 0 and r == 0 else r) for l, t, s, r in fx.bank]
            else:
                bank = [x for x in fx.bank if not ((x[0], x[1]) == first and x[2] == 0 and x[3] == 0)]
            new = S.draw_list(bank, fx.store, fx.excl, PER_CITY)
            self.assertNotIn(first, [(r[0], r[1]) for r in new])
            self.assertEqual(len(new), 4 * PER_CITY)
        for r in rows:
            self.assertNotIn(r[0], fx.excl)
            self.assertFalse(r[1].startswith("unlabelled"))

    def test_duplicate_step0_rank0_rows_are_refused(self):
        fx = self.fx
        with self.assertRaises(SystemExit):
            S.draw_list(fx.bank + [next(r for r in fx.bank if r[0] in fx.ordinary and r[2] == 0 and r[3] == 0)], fx.store, fx.excl, PER_CITY)

    def test_a_city_with_too_few_eligible_frames_is_refused(self):
        fx = self.fx
        with self.assertRaises(SystemExit):
            S.draw_list(fx.bank, fx.store, fx.excl, 10 ** 4)

    def test_mutation_include_h48_is_caught_by_the_verifier(self):
        """regression arm: a draw that forgets the H48 / SFT exclusion must put H48 frames in the list, and verify_list must refuse it."""
        fx = self.fx
        bad = S.draw_list(fx.bank, fx.store, fx.excl, PER_CITY, mutate="include_h48")
        self.assertTrue({r[0] for r in bad} & set(fx.excl), "the fixture must make the mutant visible")
        with self.assertRaises(SystemExit):
            S.verify_list(bad, fx.store, fx.excl, PER_CITY)
        S.verify_list(S.draw_list(fx.bank, fx.store, fx.excl, PER_CITY), fx.store, fx.excl, PER_CITY)

    def test_verifier_refuses_each_defect(self):
        fx = self.fx
        good = S.draw_list(fx.bank, fx.store, fx.excl, PER_CITY)
        v = lambda rows: S.verify_list(rows, fx.store, fx.excl, PER_CITY)  # noqa: E731
        with self.assertRaises(SystemExit):
            v(good[:-1])                                                                         # quota
        with self.assertRaises(SystemExit):
            v(good + [good[0]])                                                                  # duplicate (also quota)
        with self.assertRaises(SystemExit):
            v([good[0]] * 1 + good[1:-1] + [good[0]])                                            # duplicate, same length
        r0 = good[0]
        with self.assertRaises(SystemExit):
            v([(r0[0], r0[1], r0[2], "0" * 64)] + good[1:])                                      # a wrong hash
        other = next(c for c in S.CITIES if c != r0[2])
        with self.assertRaises(SystemExit):
            v([(r0[0], r0[1], other, r0[3])] + good[1:])                                         # a wrong city (quota)
        sw = [(good[0][0], good[0][1], "BOS", good[0][3])] + good[1:PER_CITY] + [(good[PER_CITY][0], good[PER_CITY][1], "LV", good[PER_CITY][3])] + good[PER_CITY + 1:]
        with self.assertRaises(SystemExit):
            v(sw)                                                                                # quotas kept, cities swapped: the store's city disagrees
        h = fx.h48[0]
        t = next(t for (l, t) in fx.labels if l == h)
        c = fx.city[h]
        doctored = good[:]
        i = next(i for i, r in enumerate(doctored) if r[2] == c)
        doctored[i] = (h, t, c, hashlib.sha256(f"{h}|{t}".encode()).hexdigest())
        with self.assertRaises(SystemExit):
            v(doctored)                                                                          # an H48 frame smuggled in
        i = next(i for i, r in enumerate(good) if r[2] == "LV")
        un = (fx.ordinary[0], "unlabelled1", "LV", hashlib.sha256(f"{fx.ordinary[0]}|unlabelled1".encode()).hexdigest())
        with self.assertRaises(SystemExit):
            v(good[:i] + [un] + good[i + 1:])                                                    # no label in the store

    def test_exclusion_list_assertions(self):
        fx = self.fx
        excl, h48, sft = S.excluded_logs_from(fx.h48_path, fx.sft_path)
        self.assertEqual(len(h48), 48)
        self.assertEqual(sorted(set(h48) | set(sft)), excl)
        bad = os.path.join(self.td.name, "h48_bad.txt")
        open(bad, "w").write("\n".join(h48[:-1]) + "\n")
        with self.assertRaises(SystemExit):
            S.excluded_logs_from(bad, fx.sft_path)                                               # not the frozen sha
        with self.assertRaises(SystemExit):
            S.excluded_logs_from(os.path.join(self.td.name, "nope.txt"), fx.sft_path)            # missing input: no fallback

    def test_store_refusals(self):
        fx = self.fx
        with self.assertRaises(SystemExit):
            S.Store(fx.root, "deadbeef")                                                         # not the pinned prefix
        with tempfile.TemporaryDirectory() as td2:
            import shutil
            r2 = os.path.join(td2, "s")
            shutil.copytree(fx.root, r2)
            sh = os.path.join(r2, "shards", sorted(os.listdir(os.path.join(r2, "shards")))[0])
            b = bytearray(open(sh, "rb").read())
            b[-1] ^= 1
            open(sh, "wb").write(bytes(b))
            with self.assertRaises(SystemExit):
                S.Store(r2, fx.prefix)                                                           # a shard that does not hash to the manifest


class TRead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.fx = Fx(cls.td.name)
        fx = cls.fx
        cls.rows = S.draw_list(fx.bank, fx.store, fx.excl, PER_CITY)
        cls.keys = [(r[0], r[1]) for r in cls.rows]
        cls.label = fx.store.labels(cls.keys)
        cls.perfect = np.stack([perfect_logit(l) for l in cls.label])
        sk = sorted(k for k in fx.labels if k[0] in fx.sft)
        cls.rkeys = sorted(sk, key=lambda k: (S.key_hash(*k), k))[:REPRO]
        cls.rlabel = fx.store.labels(cls.rkeys)
        cls.rperfect = np.stack([perfect_logit(l) for l in cls.rlabel])

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def read(self, logit, repro=None, mutate=None, banked=None):
        fx = self.fx
        bk = (self.rkeys, self.rperfect if banked is None else banked, self.rperfect if repro is None else repro, self.rlabel)
        return S.stage0_read(self.rows, logit, np.zeros_like(logit), self.label, fx.store.city, bk, n_boot=200, mutate=mutate)

    # ------------------------------------------------------------------ the threshold, both sides, mechanically
    def test_decide_reads_both_sides_of_0p85(self):
        self.assertEqual(S.decide(0.85), "S0-FIT")
        self.assertEqual(S.decide(0.8500001), "S0-FIT")
        self.assertEqual(S.decide(0.8499999), "S0-WEAK")
        self.assertEqual(S.decide(1.0), "S0-FIT")
        self.assertEqual(S.decide(0.0), "S0-WEAK")
        with self.assertRaises(SystemExit):
            S.decide(float("nan"))
        self.assertEqual(S.decide(0.85, mutate="flip_threshold"), "S0-WEAK")                     # the regression arm: a '>' threshold misreads exactly 0.85 (the test above goes RED on it)

    def test_known_iou_above_and_below_the_bar(self):
        """per city 8 frames, the first k shifted one 2 m cell: pooled IoU = (80(8-k) + 60k) / (80(8-k) + 100k), identical in every city, so the city-reweighted value equals it."""
        for k, want_row in ((2, "S0-FIT"), (3, "S0-WEAK")):
            lg = self.perfect.copy()
            for c in range(4):
                for j in range(k):
                    lg[c * PER_CITY + j] = np.roll(self.perfect[c * PER_CITY + j], 1, axis=1)
            r = self.read(lg)
            want = (80.0 * (8 - k) + 60.0 * k) / (80.0 * (8 - k) + 100.0 * k)
            self.assertAlmostEqual(r["in_sample_iou"], want, places=12)
            self.assertEqual(r["row"], want_row)
        self.assertEqual(self.read(self.perfect)["row"], "S0-FIT")

    def test_reported_values_floor_bands_flip_gap(self):
        r = self.read(self.perfect)
        self.assertAlmostEqual(r["in_sample_iou"], 1.0, places=12)
        self.assertAlmostEqual(r["controls"]["C0-FLOOR"]["value"], 0.125, places=12)             # all-drivable: road is 4 of 32 columns
        self.assertLess(r["controls"]["C0-FLIP"]["flipped"], 0.5)
        self.assertTrue(r["controls"]["C0-FLIP"]["ok"] and r["controls"]["C0-ID"]["ok"])
        self.assertEqual(sorted(r["bands"]), ["0-20", "0-40", "20-40", "40-72"])
        self.assertEqual(r["frames_per_city"], {c: PER_CITY for c in S.CITIES})
        self.assertEqual(r["invalid_controls"], [])

    # ------------------------------------------------------------------ controls fail closed
    def test_flip_blind_instrument_is_invalid(self):
        fx = self.fx
        sym = Fx.__new__(Fx)                                                                       # a symmetric road: the y-flip changes nothing
        lab = np.stack([road_label(0.0)] * len(self.rows))
        lg = np.stack([perfect_logit(l) for l in lab])
        bk = (self.rkeys, self.rperfect, self.rperfect, self.rlabel)
        r = S.stage0_read(self.rows, lg, np.zeros_like(lg), lab, fx.store.city, bk, n_boot=200)
        self.assertAlmostEqual(r["in_sample_iou"], 1.0, places=12)
        self.assertFalse(r["controls"]["C0-FLIP"]["ok"])
        self.assertEqual(r["row"], "S0-INVALID")

    def test_c0_id_failures_are_invalid(self):
        bad = self.rperfect.copy()
        bad[:, :20, :] = -bad[:, :20, :]                                                           # half the cells' signs differ
        r = self.read(self.perfect, repro=bad)
        self.assertFalse(r["controls"]["C0-ID"]["ok"])
        self.assertEqual(r["row"], "S0-INVALID")
        self.assertLess(r["controls"]["C0-ID"]["sign_agreement"], 0.999)
        ok = self.read(self.perfect, repro=self.rperfect.copy())
        self.assertEqual(ok["controls"]["C0-ID"]["sign_agreement"], 1.0)
        self.assertEqual(ok["controls"]["C0-ID"]["iou_diff"], 0.0)
        # 1 cell of the 6 x 1280 flips: agreement 0.99987 >= 0.999 passes the sign bar; the IoU bar is then what is read
        one = self.rperfect.copy()
        one[0, 3, 5] = -one[0, 3, 5]
        r1 = self.read(self.perfect, repro=one)
        self.assertGreaterEqual(r1["controls"]["C0-ID"]["sign_agreement"], 0.999)

    def test_reweighting_known_value_and_its_mutation(self):
        lg = self.perfect.copy()
        for j in range(PER_CITY):
            lg[PER_CITY + j] = np.roll(self.perfect[PER_CITY + j], 1, axis=1)                       # BOS fully shifted: IoU 0.6, other cities 1.0
        known = 1.0 - 0.4 * 3731 / 12146
        r = self.read(lg)
        self.assertAlmostEqual(r["in_sample_iou"], known, places=12)
        self.assertAlmostEqual(r["per_city"]["BOS"], 0.6, places=12)
        m = self.read(lg, mutate="no_reweight")
        self.assertAlmostEqual(m["in_sample_iou"], (3 * 640.0 + 480.0) / (3 * 640.0 + 800.0), places=12)
        self.assertGreater(abs(m["in_sample_iou"] - known), 1e-3)                                  # the known-value assertion above goes RED on the mutant

    def test_row_is_independent_of_the_interval(self):
        r = self.read(self.perfect)
        r2 = S.stage0_read(self.rows, self.perfect, np.zeros_like(self.perfect), self.label, self.fx.store.city, (self.rkeys, self.rperfect, self.rperfect, self.rlabel), n_boot=50, seed=1)
        self.assertEqual(r["row"], r2["row"])
        self.assertEqual(r["in_sample_iou"], r2["in_sample_iou"])

    # ------------------------------------------------------------------ input refusals (no fallback)
    def test_load_pred_refusals(self):
        td = self.td.name
        keys = self.keys
        lg = self.perfect
        p = os.path.join(td, "ok.npz")
        write_pred(p, keys, lg)
        a, _ = S.load_pred(p, keys)
        self.assertTrue((a == lg).all())
        rev = os.path.join(td, "rev.npz")
        write_pred(rev, keys[::-1], lg[::-1])
        a2, _ = S.load_pred(rev, keys)
        self.assertTrue((a2 == lg).all())                                                          # order-free, key-aligned
        for name, kk, ll in (("extra", keys + [("zz", "zz")], np.concatenate([lg, lg[:1]])), ("missing", keys[:-1], lg[:-1]), ("dup", keys + keys[:1], np.concatenate([lg, lg[:1]]))):
            q = os.path.join(td, name + ".npz")
            write_pred(q, kk, ll)
            with self.assertRaises(SystemExit, msg=name):
                S.load_pred(q, keys)
        bad = os.path.join(td, "step.npz")
        np.savez(bad, log=np.asarray([k[0] for k in keys]), token=np.asarray([k[1] for k in keys]), step=np.ones(len(keys), np.int64), logit=lg, sdf_m=np.zeros_like(lg))
        with self.assertRaises(SystemExit):
            S.load_pred(bad, keys)
        nan = lg.copy()
        nan[0, 0, 0] = np.nan
        q = os.path.join(td, "nan.npz")
        write_pred(q, keys, nan)
        with self.assertRaises(SystemExit):
            S.load_pred(q, keys)
        q = os.path.join(td, "shape.npz")
        write_pred(q, keys, np.zeros((len(keys), 80, 64), np.float32))
        with self.assertRaises(SystemExit):
            S.load_pred(q, keys)
        with self.assertRaises(SystemExit):
            S.load_pred(os.path.join(td, "absent.npz"), keys)

    # ------------------------------------------------------------------ the CLI end to end (dry-run, then real), on the synthetic store
    def cli(self, args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                rc = S.main(args)
            except SystemExit as e:
                rc = e.code if e.code is not None else 0
        return rc, buf.getvalue()

    def test_cli_make_list_then_read(self):
        fx, td = self.fx, self.td.name
        out = os.path.join(td, "out_list")
        bank_pred = os.path.join(td, "banked.npz")
        write_pred(bank_pred, self.rkeys + [(fx.ordinary[0], "x")], np.concatenate([self.rperfect, self.rperfect[:1]]))
        base = ["make-list", "--out-dir", out, "--bank-keys", fx.bank_path, "--store", fx.root, "--h48-logs", fx.h48_path, "--sft-heldout", fx.sft_path, "--banked-pred", bank_pred,
                "--store-sha256-prefix", fx.prefix, "--per-city-for-tests", str(PER_CITY), "--repro-n-for-tests", str(REPRO)]
        rc, txt = self.cli(base + ["--dry-run"])
        self.assertEqual(rc, 0, txt)
        self.assertIn("DRY-RUN", txt)
        self.assertFalse(os.path.exists(out), "a dry run writes nothing")
        rc, txt = self.cli(base)
        self.assertEqual(rc, 0, txt)
        meta = json.load(open(os.path.join(out, "stage0_frames.json")))
        lst = os.path.join(out, "stage0_frames.tsv")
        self.assertEqual(meta["frame_list_sha256"], hashlib.sha256(open(lst, "rb").read()).hexdigest())
        self.assertEqual(open(lst, "rb").read().count(b"\n"), 1 + 4 * PER_CITY)
        self.assertNotIn(b"\r", open(lst, "rb").read())
        self.assertEqual(meta["repro_frames"], REPRO)
        rep = open(os.path.join(out, "stage0_repro256.tsv")).read().split("\n")[1:-1]
        self.assertEqual([tuple(x.split("\t")[:2]) for x in rep], self.rkeys)
        # forwards in (synthetic), then read
        rows = S.read_list(lst)
        keys = [(r[0], r[1]) for r in rows]
        p_in, p_rep = os.path.join(td, "in.npz"), os.path.join(td, "rep.npz")
        write_pred(p_in, keys, np.stack([perfect_logit(l) for l in fx.store.labels(keys)]))
        write_pred(p_rep, self.rkeys, self.rperfect)
        rd = ["read", "--out-dir", os.path.join(td, "out_read"), "--frame-list", lst, "--frame-list-sha256", meta["frame_list_sha256"], "--preds-insample", p_in, "--banked-pred", bank_pred,
              "--preds-repro", p_rep, "--repro-n", str(REPRO), "--store", fx.root, "--h48-logs", fx.h48_path, "--sft-heldout", fx.sft_path, "--n-boot", "100",
              "--store-sha256-prefix", fx.prefix, "--per-city-for-tests", str(PER_CITY)]
        rc, txt = self.cli(rd + ["--dry-run"])
        self.assertEqual(rc, 0, txt)
        self.assertFalse(os.path.exists(os.path.join(td, "out_read")))
        rc, txt = self.cli(rd)
        self.assertEqual(rc, 0, txt)
        res = json.load(open(os.path.join(td, "out_read", "stage0_read.json")))
        self.assertEqual(res["row"], "S0-FIT")
        ins = json.load(open(os.path.join(td, "out_read", "in_sample_iou.json")))
        self.assertEqual(ins["in_sample_iou"], res["in_sample_iou"])
        self.assertEqual(ins["frame_list_sha256"], meta["frame_list_sha256"])
        # a wrong declared list sha, a doctored list, a missing C0-ID input: all refused, nothing written
        rc, txt = self.cli([x if x != meta["frame_list_sha256"] else "0" * 64 for x in rd])
        self.assertNotEqual(rc, 0)
        lines = open(lst).read().split("\n")
        h = fx.h48[0]
        t = next(t for (l, t) in fx.labels if l == h)
        c = fx.city[h]
        j = next(i for i, ln in enumerate(lines) if i > 0 and ln.split("\t")[2:3] == [c])
        lines[j] = f"{h}\t{t}\t{c}\t{hashlib.sha256((h + '|' + t).encode()).hexdigest()}"
        dl = os.path.join(td, "doctored.tsv")
        open(dl, "w", newline="\n").write("\n".join(lines))
        argv = list(rd)
        argv[argv.index("--frame-list") + 1] = dl
        argv[argv.index("--frame-list-sha256") + 1] = hashlib.sha256(open(dl, "rb").read()).hexdigest()     # sha declared correctly: the CONTENT check must still refuse
        rc, txt = self.cli(argv)
        self.assertNotEqual(rc, 0)
        self.assertIn("H48", str(rc) + txt)
        argv = list(rd)
        argv[argv.index("--preds-repro") + 1] = os.path.join(td, "absent.npz")
        rc, _ = self.cli(argv)
        self.assertNotEqual(rc, 0)

    def test_read_requires_every_input(self):
        rc, _ = self.cli(["read", "--out-dir", "x"])
        self.assertNotEqual(rc, 0)                                                                   # argparse: required flags missing -> SystemExit(2)


class TS1F(unittest.TestCase):
    def test_repro_check(self):
        a = np.random.default_rng(0).normal(size=(10, 40, 32)).astype(np.float32)
        ok, ag = S.s1f_repro_check(a, a.copy())
        self.assertTrue(ok and ag == 1.0)
        b = a.copy()
        b[:, :1, :] = -b[:, :1, :]
        ok, ag = S.s1f_repro_check(a, b)
        self.assertFalse(ok)
        self.assertLess(ag, 0.999)
        c = a.copy()
        c[0, 0, 0] = -c[0, 0, 0]
        self.assertTrue(S.s1f_repro_check(a, c)[0])                                                  # one cell of 12,800: 0.99992 >= 0.999

    def test_build_v2_manifest_and_row_comparison(self):
        with tempfile.TemporaryDirectory() as td:
            ps = []
            for n in ("a", "b", "c"):
                p = os.path.join(td, n + ".json")
                open(p, "w").write(json.dumps({"in_sample_iou": 0.8}))
                ps.append(p)
            base = {"cards_dir": "c", "store": "s", "files": {"head_s0_a": {}, "head_shuf_a": {}}}
            m = S.build_v2_manifest(base, ps[0], ps[1], ps[2])
            self.assertEqual([a["name"] for a in m["arms"]], ["HA-s0", "HA-SHUF", "HA-s0-initial"])
            self.assertEqual(m["arms"][0]["init"], "HA-s0-initial")
            self.assertEqual(m["arms"][0]["in_sample_iou_file"], "in_sample")
            self.assertEqual(m["files"]["in_sample"]["sha256"], hashlib.sha256(open(ps[2], "rb").read()).hexdigest())
            self.assertNotIn("head_init0_a", base["files"])                                          # the base manifest is not mutated
            spec, _ = S.V2.arms_from_manifest(m)
            self.assertEqual([a["role"] for a in spec], ["trained", "shuf", "init"])
        self.assertTrue(S.compare_rows("S0-FIT", "S1-FIT")["agree"])
        self.assertTrue(S.compare_rows("S0-WEAK", "S1-WEAK")["agree"])
        self.assertFalse(S.compare_rows("S0-FIT", "S1-WEAK")["agree"])
        self.assertFalse(S.compare_rows("S0-FIT", "INCOMPLETE")["agree"])

    def test_the_tool_uses_the_frozen_functions(self):
        for n in ("frame_stats", "city_iou_read", "p1_block", "boot_weights", "verdict"):
            self.assertIs(getattr(S.R, n), getattr(S.V2.R, n))
        self.assertEqual(S.V2.V1_SHA256, "58aa3a77b976a6fb4b08b4bc5126d572e8be8ef1fc37926d9c9ce0983eed718e")
        self.assertEqual(S.FIT_BAR, R.FIT_IN_SAMPLE)
        self.assertEqual(S.PER_CITY * len(S.CITIES), 2048)


if __name__ == "__main__":
    unittest.main(verbosity=2)
