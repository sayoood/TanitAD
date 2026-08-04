#!/usr/bin/env python3
"""Drive a RunPod **web terminal** (GoTTY) as a non-interactive shell.

WHY THIS EXISTS. A Claude Code cloud session cannot SSH to the pods at all:
all non-443 TCP is silently dropped (MEASURED 2026-08-04 -- github.com:22, a
host that certainly answers, returns nothing either). The web terminal is the
one pod surface on 443, so with `*.proxy.runpod.net` allowed in the
environment's Custom network policy it becomes a usable shell -- no SSH key,
no browser, no Playwright.

TWO THINGS THAT COST A DEBUG ROUND EACH, both MEASURED, both handled below:

1. **Cloudflare ALPN-negotiates h2, and a WebSocket Upgrade over HTTP/2 is
   rejected.** `HTTP/2 400 Bad Request`, while the byte-identical request with
   `--http1.1` returns `HTTP/1.1 101 Switching Protocols`. Pinning ALPN to
   http/1.1 is the fix. TLS verification stays ON against the proxy CA bundle;
   this is not a weakening. (websocket-client would not accept an SSLContext
   through `sslopt` here, which is why the handshake is hand-rolled.)

2. **GoTTY v1 protocol constants -- Input is '1', NOT '0'.** '0' is
   `UnknownInput`; the server closes the connection on it, which looks exactly
   like an auth failure and sends you hunting a token that is not the problem.
     client->server: Input='1'  Ping='2'  ResizeTerminal='3'
     server->client: Output='1' Pong='2'  SetWindowTitle='3'
   Output payloads are BASE64, and frames split mid-base64 -- an unpadded chunk
   raises and silently dumps raw base64 into the transcript, which reads as a
   corrupted terminal rather than as a decode bug.

⛔ CREDENTIAL HANDLING. The base URL CONTAINS a root credential for the pod.
It is read from a file (default: `gotty_url.txt` next to this script, which is
git-ignored), never passed on argv, never printed, and never committed. Treat a
URL that has been pasted anywhere as burned and regenerate the terminal.

⛔ TREAT AS READ-ONLY. Both pods are marked DO NOT TOUCH. This is a debugging
window, not a place to restart trainers. If you must act, remember: `pgrep -f`
/ `pkill -f` SELF-MATCH (kill by explicit PID), `PYTHONPATH=` is required, and
`df` lies on a pod (use a real `dd` write test).

Usage:  python3 gotty_shell.py "<command>" [collect_seconds]
        GOTTY_URL_FILE=/path/to/url.txt python3 gotty_shell.py "..."
"""
import base64, json, os, re, socket, ssl, struct, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
URLF = os.environ.get("GOTTY_URL_FILE") or os.path.join(HERE, "gotty_url.txt")
BASE = open(URLF).read().strip().rstrip("/")
host = BASE.split("/")[2]
path = "/" + "/".join(BASE.split("/")[3:]) + "/ws"

cmd = sys.argv[1] if len(sys.argv) > 1 else "echo PROBE_OK"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0

prox = os.environ.get("HTTPS_PROXY", "").replace("http://", "")
phost, _, pport = prox.partition(":")

# 1) CONNECT through the agent proxy
raw = socket.create_connection((phost, int(pport)), timeout=30)
raw.sendall(f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n".encode())
resp = b""
while b"\r\n\r\n" not in resp:
    resp += raw.recv(1)
if b" 200 " not in resp.split(b"\r\n")[0]:
    sys.exit(f"proxy refused: {resp.splitlines()[0]!r}")

# 2) TLS with ALPN pinned to http/1.1
ctx = ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt")
ctx.set_alpn_protocols(["http/1.1"])
s = ctx.wrap_socket(raw, server_hostname=host)

# 3) WebSocket handshake
key = base64.b64encode(os.urandom(16)).decode()
req = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
       f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
       f"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: tty\r\n"
       f"Origin: https://{host}\r\n\r\n")
s.sendall(req.encode())
resp = b""
while b"\r\n\r\n" not in resp:
    d = s.recv(1)
    if not d:
        sys.exit("closed during handshake")
    resp += d
if b"101" not in resp.split(b"\r\n")[0]:
    sys.exit(f"handshake failed: {resp.decode(errors='replace')[:300]}")

def send(payload: bytes, opcode=0x1):
    hdr = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        hdr.append(0x80 | n)
    elif n < 65536:
        hdr.append(0x80 | 126); hdr += struct.pack(">H", n)
    else:
        hdr.append(0x80 | 127); hdr += struct.pack(">Q", n)
    m = os.urandom(4)
    hdr += m
    s.sendall(bytes(hdr) + bytes(b ^ m[i % 4] for i, b in enumerate(payload)))

buf = b""
def recv_frame():
    global buf
    def need(n):
        global buf
        while len(buf) < n:
            d = s.recv(65536)
            if not d:
                raise ConnectionError("closed")
            buf += d
    need(2)
    b1, b2 = buf[0], buf[1]
    op = b1 & 0x0F
    ln = b2 & 0x7F
    off = 2
    if ln == 126:
        need(4); ln = struct.unpack(">H", buf[2:4])[0]; off = 4
    elif ln == 127:
        need(10); ln = struct.unpack(">Q", buf[2:10])[0]; off = 10
    need(off + ln)
    data = buf[off:off + ln]
    buf = buf[off + ln:]
    return op, data

# 4) GoTTY v1 protocol -- MEASURED from the live frames, not assumed:
#      client->server: Input='1'  Ping='2'  ResizeTerminal='3'
#      server->client: Output='1' Pong='2'  SetWindowTitle='3'
#    and Output payloads are BASE64. Sending '0' (UnknownInput) makes
#    the server close the connection, which looks like an auth failure.
send(json.dumps({"Arguments": "", "AuthToken": ""}).encode())
send(b"3" + json.dumps({"columns": 200, "rows": 50}).encode())
time.sleep(1.0)
send(b"1" + (cmd + "\n").encode())

out = []
end = time.time() + secs
s.settimeout(3.0)
while time.time() < end:
    try:
        op, data = recv_frame()
    except (socket.timeout, ssl.SSLWantReadError):
        continue
    except Exception:
        break
    if op == 0x8:
        break
    if not data:
        continue
    t, payload = data[:1], data[1:]
    if t == b"1":
        # ⚠️ pad to a multiple of 4 — gotty frames split mid-base64 and an
        # unpadded chunk raises, which silently dumps RAW BASE64 into the
        # transcript and looks like corrupted terminal output.
        try:
            out.append(base64.b64decode(
                payload + b"=" * (-len(payload) % 4)).decode("utf-8", "replace"))
        except Exception:
            out.append(payload.decode("utf-8", "replace"))
try:
    send(b"", 0x8); s.close()
except Exception:
    pass

txt = "".join(out)
txt = re.sub(r"\x1b\][^\x07\x1b]*(\x07|\x1b\\)", "", txt)
txt = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", txt)
print(txt.replace("\r\n", "\n").replace("\r", "\n"))
