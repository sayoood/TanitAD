"""Peak host-RSS sampler for the S0 dump subprocess (psutil, 5 s).  Appends 'time,pid,rss_mb' to run/logs/rss_dump.csv and keeps the running maximum in run/logs/rss_dump_peak.txt.  Exits after 24 h."""
import os, time
import psutil
OUT = "C:/Users/Admin/srttc_s0/run/logs"
peak = 0
end = time.time() + 24 * 3600
while time.time() < end:
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
            if "srttc_s0_dump.py" in cl and p.info["name"] and p.info["name"].lower().startswith("python"):
                mb = p.memory_info().rss / 2**20
                with open(f"{OUT}/rss_dump.csv", "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')},{p.info['pid']},{mb:.0f}\n")
                if mb > peak:
                    peak = mb
                    open(f"{OUT}/rss_dump_peak.txt", "w").write(f"{peak:.0f} MiB (max host RSS of a srttc_s0_dump.py process seen, {time.strftime('%Y-%m-%dT%H:%M:%S')})\n")
        except (psutil.Error, OSError):
            pass
    time.sleep(5)
