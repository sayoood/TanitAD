"""render.py -- render the pitch film page with headless Chrome over the DevTools protocol.
  python render.py frames <html> <outdir> t1,t2,...          sample frames as PNG (for checking)
  python render.py video  <html> <out.mp4> [fps] [t0] [t1]   every frame piped into ffmpeg
The page exposes window.__seek(t) and window.__dur; every frame is a pure function of t. No GPU is used."""
import asyncio
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request

import websockets

CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"
FFMPEG = "C:/Users/Admin/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe"
PORT = int(os.environ.get("RPORT", "9377"))
FMT = os.environ.get("RFMT", "png")
HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")


def wrap(html_path):
    src = io.open(html_path, encoding="utf-8").read()
    lines = [ln for ln in src.split("\n") if "fonts.googleapis.com" not in ln]
    out = HERE + "/render_page_%d.html" % PORT
    head = "<!doctype html><html><head><meta charset=\"utf-8\"><style>html,body{margin:0;background:#05080f}img{max-width:100%}[hidden]{display:none!important}</style></head><body>"
    io.open(out, "w", encoding="utf-8", newline="\n").write(head + "\n".join(lines) + "</body></html>")
    return "file:///" + out + "?render=1"


class Cdp:
    def __init__(self, ws):
        self.ws = ws
        self.n = 0
        self.errors = []

    async def call(self, method, params=None):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(method + ": " + json.dumps(msg["error"]))
                return msg.get("result", {})
            m = msg.get("method", "")
            if m == "Runtime.exceptionThrown":
                self.errors.append(json.dumps(msg["params"]["exceptionDetails"])[:600])
            elif m == "Runtime.consoleAPICalled" and msg["params"].get("type") == "error":
                self.errors.append(json.dumps(msg["params"]["args"])[:400])


async def open_page(url):
    prof = HERE + "/chrome_profile_%d" % PORT
    os.makedirs(prof, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio", "--no-first-run", "--no-default-browser-check",
                             "--remote-allow-origins=*", "--user-data-dir=" + prof, "--remote-debugging-port=%d" % PORT, "--window-size=1920,1080",
                             "--force-device-scale-factor=1", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wsurl = None
    for _ in range(100):
        try:
            lst = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/json/list" % PORT, timeout=2).read().decode("utf-8"))
            pages = [p for p in lst if p.get("type") == "page"]
            if pages:
                wsurl = pages[0]["webSocketDebuggerUrl"]
                break
        except Exception:
            pass
        time.sleep(0.2)
    if not wsurl:
        proc.kill()
        raise RuntimeError("no DevTools page target")
    ws = await websockets.connect(wsurl, max_size=None, ping_interval=None)
    c = Cdp(ws)
    await c.call("Page.enable")
    await c.call("Runtime.enable")
    await c.call("Emulation.setDeviceMetricsOverride", {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False})
    await c.call("Page.navigate", {"url": url})
    for _ in range(100):
        r = await c.call("Runtime.evaluate", {"expression": "document.readyState === 'complete' && typeof window.__seek === 'function'", "returnByValue": True})
        if r["result"].get("value") is True:
            break
        await asyncio.sleep(0.2)
    r = await c.call("Runtime.evaluate", {"expression": "document.fonts.ready.then(function(){ return window.__dur; })", "awaitPromise": True, "returnByValue": True})
    dur = r["result"].get("value")
    fam = await c.call("Runtime.evaluate", {"expression": "document.fonts.check('600 60px Bahnschrift') + ' ' + getComputedStyle(document.querySelector('.h2')).fontFamily.slice(0,40)", "returnByValue": True})
    print("page ready; duration", dur, "; fonts:", fam["result"].get("value"), "; errors so far:", len(c.errors), flush=True)
    return proc, ws, c, dur


async def shot(c, t, fmt="png", quality=None):
    await c.call("Runtime.evaluate", {"expression": "window.__seek(%.5f)" % t})
    p = {"format": fmt, "optimizeForSpeed": True, "captureBeyondViewport": False}
    if quality:
        p["quality"] = quality
    r = await c.call("Page.captureScreenshot", p)
    return base64.b64decode(r["data"])


async def main():
    mode, html = sys.argv[1], sys.argv[2]
    url = wrap(html)
    proc, ws, c, dur = await open_page(url)
    try:
        if mode == "frames":
            outdir = sys.argv[3]
            os.makedirs(outdir, exist_ok=True)
            for ts in sys.argv[4].split(","):
                t = float(ts)
                data = await shot(c, t)
                io.open("%s/f_%07.2f.png" % (outdir, t), "wb").write(data)
            print("frames written:", len(sys.argv[4].split(",")), flush=True)
        else:
            out = sys.argv[3]
            fps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
            t0 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
            t1 = float(sys.argv[6]) if len(sys.argv) > 6 else float(dur)
            n = int(round((t1 - t0) * fps))
            ff = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", ("mjpeg" if FMT == "jpeg" else "png"), "-i", "-",
                                   "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                   "-metadata", "title=TanitAD pitch film", out], stdin=subprocess.PIPE)
            st = time.time()
            for i in range(n):
                data = await shot(c, t0 + i / fps, FMT, 96 if FMT == "jpeg" else None)
                ff.stdin.write(data)
                if i % 300 == 0:
                    el = time.time() - st
                    print("frame %d of %d  %.1f s  %.1f fps  errors %d" % (i, n, el, (i + 1) / max(el, 1e-6), len(c.errors)), flush=True)
            ff.stdin.close()
            rc = ff.wait()
            print("ZZ VIDEO DONE frames %d  ffmpeg rc %d  %.0f s  errors %d" % (n, rc, time.time() - st, len(c.errors)), flush=True)
        for e in c.errors[:8]:
            print("PAGE ERROR:", e, flush=True)
    finally:
        try:
            await c.call("Browser.close")
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


asyncio.run(main())
