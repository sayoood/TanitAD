#!/usr/bin/env python3
"""Regenerate overlay/refe/{navsim_pdm_targets,online_labeller,label_worker}.py from the pinned tip copies and patches/*.diff.
usage: make_overlay.py <dir with the tip's refe/*.py (git archive of d5ec25ec)>      (a2/ is this script's parent's parent)"""
import os
import shutil
import subprocess
import sys

TIP = sys.argv[1] if len(sys.argv) > 1 else "F:/a2_scratch/tip_refe"
A2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OV = os.path.join(A2, "overlay", "refe")


def rd(p):
    s = open(p, newline="", encoding="utf-8").read()
    return s.replace("\r\n", "\n"), "\r\n" in s


def wr(p, s, crlf):
    open(p, "w", newline="", encoding="utf-8").write(s.replace("\n", "\r\n") if crlf else s)


def sub(s, old, new, count=1):
    assert s.count(old) == count, (s.count(old), old[:80])
    return s.replace(old, new)


for f in ("navsim_pdm_targets.py", "online_labeller.py", "label_worker.py"):
    shutil.copyfile(os.path.join(TIP, f), os.path.join(OV, f))

# ---------------------------------------------------------------- navsim_pdm_targets.score_plans
p = os.path.join(OV, "navsim_pdm_targets.py")
s, crlf = rd(p)
s = sub(s, '''def score_plans(parts, plans8: np.ndarray) -> dict:
    """plans8 [M, 8, 3] NAVSIM-frame poses (ego frame, 2 Hz) -> per-plan PDM targets (arrays of length M)."""''', '''def score_plans(parts, plans8: np.ndarray, behaviour: bool = False) -> dict:
    """plans8 [M, 8, 3] NAVSIM-frame poses (ego frame, 2 Hz) -> per-plan PDM targets (arrays of length M).
    behaviour=True (A2, default OFF): ADDITIONALLY return the behaviour-term labels under the key "behaviour" (behaviour_terms.behaviour_terms on the SAME simulated
    states `sim` this function already holds).  The seven PDM keys are computed by the unchanged code above the `if behaviour` line, so with behaviour=False the result is
    bit-identical to the pre-A2 function (tests/test_a2_off_identity.py proves it)."""''')
s = sub(s, '''    return {"nc": sc._multi_metrics[MultiMetricIndex.NO_COLLISION][sl].copy(),
            "dac": sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA][sl].copy(),
            "ep": ep[sl], "ttc": np.asarray(ttc)[sl].copy(), "c": np.asarray(com)[sl].copy(),
            "ddc": np.asarray(sc._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION])[sl].copy(),
            "pdms": pdms[sl]}''', '''    out = {"nc": sc._multi_metrics[MultiMetricIndex.NO_COLLISION][sl].copy(),
           "dac": sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA][sl].copy(),
           "ep": ep[sl], "ttc": np.asarray(ttc)[sl].copy(), "c": np.asarray(com)[sl].copy(),
           "ddc": np.asarray(sc._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION])[sl].copy(),
           "pdms": pdms[sl]}
    if behaviour:                                      # A2: label-only extension; reads `sim` (already computed), never writes a PDM key
        import behaviour_terms as BT
        out["behaviour"] = BT.behaviour_terms(sim, parts)
    return out''')
wr(p, s, crlf)

# ---------------------------------------------------------------- online_labeller
p = os.path.join(OV, "online_labeller.py")
s, crlf = rd(p)
s = sub(s, "def _pdm_init(metric_cache: str, lru: int = 8):", "def _pdm_init(metric_cache: str, lru: int = 8, behaviour: bool = False):")
s = sub(s, '    _W.update(PT=PT, root=metric_cache, lru=OrderedDict(), lru_n=int(lru))',
        '    _W.update(PT=PT, root=metric_cache, lru=OrderedDict(), lru_n=int(lru), behaviour=bool(behaviour))')
s = sub(s, '''    PT = _W["PT"]
    path = parts_path(''', '''    PT = _W["PT"]
    if _W.get("behaviour"):                                     # A2 (default OFF): the navtest quarantine is checked BEFORE a behaviour label is computed
        import behaviour_gate as BG
        BG.assert_not_navtest(meta.get("log_name"), meta.get("token"))
    path = parts_path(''')
s = sub(s, '''    labels = labels_from_targets(PT.score_plans(parts, PT.executed_plans8(props, repair=True)))
    return {"labels": labels, "sha": sha, "reason": None, "cpu_s": time.process_time() - t0}''', '''    if _W.get("behaviour"):
        t = PT.score_plans(parts, PT.executed_plans8(props, repair=True), behaviour=True)
        import behaviour_terms as BT
        bq, bact = BT.pack(t["behaviour"])
        return {"labels": labels_from_targets(t), "behaviour": bq, "behaviour_act": bact, "sha": sha, "reason": None, "cpu_s": time.process_time() - t0}
    labels = labels_from_targets(PT.score_plans(parts, PT.executed_plans8(props, repair=True)))
    return {"labels": labels, "sha": sha, "reason": None, "cpu_s": time.process_time() - t0}''')
s = sub(s, "    def __init__(self, python: str, workers: int, metric_cache: str):", "    def __init__(self, python: str, workers: int, metric_cache: str, behaviour: bool = False):")
s = sub(s, '"label_worker.py"), metric_cache], stdin', '"label_worker.py"), metric_cache] + (["--behaviour"] if behaviour else []), stdin')
s = sub(s, '''            if res.get("labels") is not None:
                res["labels"] = np.frombuffer(res["labels"], dtype=np.float32).reshape(res["labels_shape"]).copy()
            return res''', '''            if res.get("labels") is not None:
                res["labels"] = np.frombuffer(res["labels"], dtype=np.float32).reshape(res["labels_shape"]).copy()
            if res.get("behaviour") is not None:
                res["behaviour"] = np.frombuffer(res["behaviour"], dtype=np.float32).reshape(res["behaviour_shape"]).copy()
                res["behaviour_act"] = np.frombuffer(res["behaviour_act"], dtype=np.uint8).reshape(res["behaviour_shape"]).astype(bool)
            return res''')
s = sub(s, "min_sets_for_fail_check: int = 200, chunks: int = 1):", "min_sets_for_fail_check: int = 200, chunks: int = 1,\n                 behaviour: bool = False):")
s = sub(s, '''        self.chunks = int(chunks)
        if backend == "pdm" and not metric_cache:''', '''        self.chunks = int(chunks)
        self.behaviour = bool(behaviour)                          # A2: also return the behaviour-term labels in info["behaviour"] / info["behaviour_act"]; default OFF
        if self.behaviour and backend != "pdm":
            raise ValueError("behaviour terms exist only on the pdm backend (the synthetic control has none)")
        if backend == "pdm" and not metric_cache:''')
s = sub(s, "self._ex = _SubprocPool(self.python, n, self.metric_cache)", "self._ex = _SubprocPool(self.python, n, self.metric_cache, self.behaviour)")
s = sub(s, "initializer=_pdm_init, initargs=(self.metric_cache,))", "initializer=_pdm_init, initargs=(self.metric_cache, 8, self.behaviour))")
s = sub(s, '''        mask = np.zeros(B, np.float32)
        reasons = []
        try:''', '''        mask = np.zeros(B, np.float32)
        reasons = []
        bq = np.ones((B, self.n_prop, 5), np.float32) if self.behaviour else None
        bact = np.zeros((B, self.n_prop, 5), bool) if self.behaviour else None
        try:''')
s = sub(s, '''                    labels[i, h["cuts"][c]] = r["labels"]
''', '''                    labels[i, h["cuts"][c]] = r["labels"]
                    if self.behaviour:
                        bq[i, h["cuts"][c]], bact[i, h["cuts"][c]] = r["behaviour"], r["behaviour_act"]
''')
s = sub(s, '''                if mask[i] == 0.0:
                    labels[i] = 0.0''', '''                if mask[i] == 0.0:
                    if self.behaviour:
                        bq[i], bact[i] = 1.0, False                      # masked sample: neutral q, nothing applicable
                    labels[i] = 0.0''')
s = sub(s, '''        return labels, mask, {"stamp": h["stamp"], "lag_steps": lag, "reasons": reasons, "latency_s": time.perf_counter() - h["t_submit"]}''', '''        info = {"stamp": h["stamp"], "lag_steps": lag, "reasons": reasons, "latency_s": time.perf_counter() - h["t_submit"]}
        if self.behaviour:
            info["behaviour"], info["behaviour_act"] = bq, bact      # [B, M, 5] q in behaviour_terms.ORDER, [B, M, 5] applicable; masked samples: q = 1, act = False
        return labels, mask, info''')
wr(p, s, crlf)

# ---------------------------------------------------------------- label_worker
p = os.path.join(OV, "label_worker.py")
s, crlf = rd(p)
s = sub(s, "    OL._pdm_init(sys.argv[1])", '    OL._pdm_init(sys.argv[1], behaviour=("--behaviour" in sys.argv[2:]))')
s = sub(s, '''                res["labels"] = np.ascontiguousarray(res["labels"], dtype=np.float32).tobytes()''', '''                res["labels"] = np.ascontiguousarray(res["labels"], dtype=np.float32).tobytes()
            if res.get("behaviour") is not None:                   # A2 (only with --behaviour): same raw-bytes rule
                res["behaviour_shape"] = tuple(res["behaviour"].shape)
                res["behaviour"] = np.ascontiguousarray(res["behaviour"], dtype=np.float32).tobytes()
                res["behaviour_act"] = np.ascontiguousarray(res["behaviour_act"], dtype=np.uint8).tobytes()''')
wr(p, s, crlf)

os.makedirs(os.path.join(A2, "patches"), exist_ok=True)
for f in ("navsim_pdm_targets", "online_labeller", "label_worker"):
    r = subprocess.run(["diff", "-u", "--strip-trailing-cr", os.path.join(TIP, f + ".py"), os.path.join(OV, f + ".py")], capture_output=True, text=True)
    d = r.stdout.replace(TIP, "a/refe").replace(OV, "b/refe").replace("\\", "/")
    open(os.path.join(A2, "patches", f + ".diff"), "w", newline="\n").write(d)
    print(f, "diff lines", d.count("\n"))
