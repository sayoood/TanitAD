"""tok_probe_u_ck.py -- the saving tool of card PREREG_V8F_PROOF_ADD123 (DSCALE-LEAD-2X, control q1). Tier T0; no model is trained.

It imports the banked probe tool ``tok_probe_u.py`` UNCHANGED (found in the folder named by the env variable TP_DIR if it is set, else beside this file) and runs its ``main`` with one own option,
``--ck-steps 19164,28746`` (removed from the argv before the rest goes on). During ``train`` it writes, for every arm, ``<out>/<ARM>/ckpt_step<N>.pt`` right after that arm's N-th update,
through the banked ``save_probe`` (the same blob format, ``step`` = N), so that ``tok_probe_u.py extract --ckpt <that file>`` reads it.

How: attributes of the imported module are replaced at run time (``build_probe`` to learn which probe is which arm, ``step_probe`` to count the updates of each probe and to save, ``run_training``
to learn out_dir, argv_sha and grid and to write the record). Nothing here draws a random number or changes an update: the saving reads the weights after the step and writes them.
A step of --ck-steps that is not below the run's total is refused (exit 3). After the run ``<out>/CK_RECORD.json`` holds: ck_steps, files (arm -> file names), tok_probe_u_sha256, tool_sha256.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REFUSED = 3


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_banked(tp_dir=None):
    """Import tok_probe_u.py from ``tp_dir`` (env TP_DIR, else the folder of this file) as the module ``tok_probe_u``."""
    d = tp_dir or os.environ.get("TP_DIR") or HERE
    path = os.path.join(d, "tok_probe_u.py")
    if not os.path.isfile(path):
        raise SystemExit("REFUSED: %s not found (put this tool beside tok_probe_u.py or set TP_DIR)" % path)
    spec = importlib.util.spec_from_file_location("tok_probe_u", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tok_probe_u"] = m
    spec.loader.exec_module(m)
    return m


def parse_ck_steps(text):
    try:
        steps = [int(x) for x in str(text).split(",") if x.strip() != ""]
    except ValueError:
        _refuse("--ck-steps %r is not a list of integers" % (text,))
    if not steps or any(s <= 0 for s in steps) or len(set(steps)) != len(steps):
        _refuse("--ck-steps %r must be distinct positive integers" % (text,))
    return sorted(steps)


def _refuse(text):
    sys.stderr.write("REFUSED: " + text + "\n")
    sys.stderr.flush()
    raise SystemExit(REFUSED)


class Saver:
    """Holds the replaced attributes of the imported module; ``restore`` puts the originals back."""

    def __init__(self, tpu, ck_steps):
        self.tpu = tpu
        self.ck_steps = sorted(int(s) for s in ck_steps)
        self.orig = {k: getattr(tpu, k) for k in ("build_probe", "step_probe", "run_training")}
        self.arm_of = {}      # id(probe) -> (probe, arm)
        self.count = {}       # id(probe) -> updates seen
        self.files = {}       # arm -> [file names]
        self.ctx = {"out_dir": None, "argv_sha": "", "grid": tpu.GRID_DEFAULT, "total": None}
        self.record_path = None

    def install(self):
        tpu, orig, me = self.tpu, self.orig, self

        def build_probe(arm, *a, **k):
            p = orig["build_probe"](arm, *a, **k)
            me.arm_of[id(p)] = (p, arm)
            me.count[id(p)] = 0
            return p

        def step_probe(probe, *a, **k):
            r = orig["step_probe"](probe, *a, **k)
            key = id(probe)
            if key in me.arm_of:
                me.count[key] += 1
                n = me.count[key]
                if n in me.ck_steps:
                    me._save(probe, me.arm_of[key][1], n)
            return r

        def run_training(stream, arms, total, lr_at, **k):
            bad = [s for s in me.ck_steps if s >= int(total)]
            if bad:
                _refuse("--ck-steps %s is not below the run's total of %d steps" % (",".join(str(s) for s in bad), int(total)))
            me.arm_of.clear()
            me.count.clear()
            me.files = {}
            me.ctx = {"out_dir": k.get("out_dir"), "argv_sha": k.get("argv_sha", ""), "grid": k.get("grid", tpu.GRID_DEFAULT), "total": int(total)}
            res = orig["run_training"](stream, arms, total, lr_at, **k)
            me._write_record()
            return res

        tpu.build_probe, tpu.step_probe, tpu.run_training = build_probe, step_probe, run_training
        return self

    def restore(self):
        for k, v in self.orig.items():
            setattr(self.tpu, k, v)

    def _save(self, probe, arm, n):
        c = self.ctx
        name = "ckpt_step%d.pt" % n
        path = os.path.join(c["out_dir"], arm, name)
        self.tpu.save_probe(probe, arm, path, n, c["argv_sha"], grid=c["grid"], n_tok_seen=self.tpu.GRIDS[c["grid"]])
        self.files.setdefault(arm, []).append(name)

    def _write_record(self):
        c = self.ctx
        for arm, names in self.files.items():
            if sorted(names) != sorted("ckpt_step%d.pt" % s for s in self.ck_steps):
                raise SystemExit("REFUSED: arm %s wrote %s, expected one file per step of %s" % (arm, names, self.ck_steps))
        rec = {"ck_steps": [int(s) for s in self.ck_steps], "files": {a: sorted(v) for a, v in self.files.items()},
               "tok_probe_u_sha256": sha256_file(self.tpu.__file__), "tool_sha256": sha256_file(os.path.abspath(__file__))}
        self.record_path = os.path.join(c["out_dir"], "CK_RECORD.json")
        with open(self.record_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rec, fh, indent=1)


def split_argv(argv):
    """Return (rest, ck_text) with ``--ck-steps X`` or ``--ck-steps=X`` removed."""
    rest, ck, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--ck-steps":
            if i + 1 >= len(argv):
                _refuse("--ck-steps needs a value")
            ck = argv[i + 1]
            i += 2
            continue
        if a.startswith("--ck-steps="):
            ck = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1
    return rest, ck


def main(argv=None, tpu=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest, ck = split_argv(argv)
    tpu = tpu or load_banked()
    if ck is None or not rest or rest[0] != "train":
        return tpu.main(rest)                                                 # no saving requested, or not a training run: the banked tool as it is
    steps = parse_ck_steps(ck)
    if "--steps" in rest:                                                     # early refusal, before the data are built
        try:
            total = int(rest[rest.index("--steps") + 1])
        except (ValueError, IndexError):
            total = 0
        if total and any(s >= total for s in steps):
            _refuse("--ck-steps %s is not below --steps %d" % (ck, total))
    saver = Saver(tpu, steps).install()
    try:
        return tpu.main(rest)
    finally:
        saver.restore()


if __name__ == "__main__":
    sys.exit(main())
