# VM side, run with `colab exec -f` (short): one ZZPOLL line -- is the runner alive, which stages/steps finished with
# which rc and seconds (no log tails), and the runner log's last lines. ASCII only.
import json
import os

st = {}
try:
    st = json.load(open("/content/p0/out/status.json"))
except Exception as e:  # noqa: BLE001
    st = {"status_error": repr(e)[:200]}
pid = open("/content/p0/runner.pid").read().strip() if os.path.exists("/content/p0/runner.pid") else None
alive = bool(pid) and os.path.exists(f"/proc/{pid}") and "zombie" not in open(f"/proc/{pid}/status").read().lower()
tail = open("/content/p0/runner.log", errors="replace").read()[-600:] if os.path.exists("/content/p0/runner.log") else ""
out = {"alive": alive, "ended": st.get("ended_utc"), "stages": st.get("stages"), "setup": st.get("setup"),
       "steps": {k: [v.get("rc"), v.get("s")] for k, v in st.get("steps", {}).items()},
       "packed": st.get("packed"), "tail": tail}
print("ZZPOLL " + json.dumps(out))
