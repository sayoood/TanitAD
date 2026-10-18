#!/usr/bin/env python3
"""Run ONLY test_rls1_queue.queue_scenarios(only={"s1"}) N times from <refe dir>, recording what each sandbox saw.  Usage: s1probe.py <refe_dir> <N> [scen]"""
import json, os, sys, tempfile, time
from pathlib import Path
HERE = Path(sys.argv[1]); N = int(sys.argv[2]); SC = sys.argv[3] if len(sys.argv) > 3 else "s1"
os.chdir(HERE); sys.path.insert(0, str(HERE))
import test_rls1_queue as T
DELAY = os.environ.get("S1PROBE_DELAY")          # simulate a slow fake-S1 start (a loaded box): sleep BEFORE it takes its lock
if DELAY:
    _ws = T.write_sh
    def write_sh(p, body):
        if Path(p).name.startswith("other_s1aux"):
            body = body.replace('exec 8>', f'sleep {DELAY}\nexec 8>', 1)
        return _ws(p, body)
    T.write_sh = write_sh
rec = []
orig = T.Queue.close
def close(self):
    s1 = self.data / "s1aux" / "chain.log"
    smp = self.samples.read_text().split() if self.samples.exists() else []
    rec.append({"procs": [(os.path.basename(str(p.args[-1])), p.poll()) for p in self.procs], "s1log": s1.read_text() if s1.exists() else None,
                "qlog": self.log()[-600:], "qrc": (self.R / "queue.rc").read_text().strip() if (self.R / "queue.rc").exists() else None,
                "first_samples": smp[:8], "n_samples": len(smp)})
    orig(self)
T.Queue.close = close
text = T.CHAIN.read_text(encoding="utf-8")
fails = 0
for i in range(N):
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        try:
            (T.kmon_scenarios(text, Path(tmp), only={SC[5:]}) if SC.startswith("kmon:") else T.queue_scenarios(text, Path(tmp), only={SC})); ok, err = True, None
        except AssertionError as e:
            ok, err = False, str(e)[:200]; fails += 1
        print(json.dumps({"i": i, "ok": ok, "secs": round(time.time() - t0, 1), "err": err, **(rec[-1] if rec and not SC.startswith("kmon:") else {})}), flush=True)
print(f"ZZS1PROBE tree={HERE.parents[4].name} scen={SC} n={N} fails={fails}", flush=True)
