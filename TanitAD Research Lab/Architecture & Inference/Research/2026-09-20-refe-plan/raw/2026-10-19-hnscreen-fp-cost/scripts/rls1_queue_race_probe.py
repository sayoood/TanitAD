import sys, time, tempfile, shutil
from pathlib import Path
B = r"C:\Users\Admin\par18\hnfp3\base\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\refe"
sys.path.insert(0, B); sys.path.insert(0, B + r"\..\eval")
sys.dont_write_bytecode = True
import test_rls1_queue as T
text = T.CHAIN.read_text(encoding="utf-8")
tmp = Path(tempfile.mkdtemp(prefix="dbgq_", dir=r"C:\Users\Admin\par18\hnfp3\tmp_fin"))
q = T.Queue(text, tmp, "q_s1")
try:
    o = q.other("s1aux", "mark ZZS1AUX_START; sleep 4; mark ZZS1AUX_DONE")
    time.sleep(float(sys.argv[1]))
    q.observe(14)
    time.sleep(1.0)
    t0 = time.time()
    q.start()
    r = q.rc()
    print("rc", r, "elapsed %.2f" % (time.time() - t0))
    print("other still alive:", o.poll() is None, "other rc", o.poll())
    print("samples head:", q.samples.read_text().split()[:12] if q.samples.exists() else None)
    print("chain.log:", q.log()[:500])
    print("s1 chain.log:", (q.data / "s1aux" / "chain.log").read_text() if (q.data / "s1aux" / "chain.log").exists() else None)
    print("other output:", o.stdout.read() if o.poll() is not None else "(alive)")
finally:
    q.close()
    shutil.rmtree(tmp, ignore_errors=True)
