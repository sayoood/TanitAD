"""Does the PI's Colab Secret HF_TOKEN resolve in THIS session? Prints `ZZSECRET {json}` of BOOLEANS ONLY -- never the
value. Run on a VM: & colab\\colab.ps1 exec -s <name> --timeout 90 -f colab\\secret_probe.py

Why bounded: google.colab.userdata.get() asks the notebook FRONTEND for the secret and waits forever when no
frontend answers -- the open question for a CLI-provisioned (headless) session (COLAB_CLI_MCP.md 6a). This asks the
same way ('GetSecret' over the kernel's message channel) with a 30 s limit, so "no frontend" reads as a result
rather than a hang. ASCII only (colab-cli reads this file with the dev box's cp1252 codec).
"""
import json
import time

out = {"t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
try:
    from google.colab import _message

    t0 = time.time()
    resp = _message.blocking_request("GetSecret", request={"key": "HF_TOKEN"}, timeout_sec=30)
    out["seconds"] = round(time.time() - t0, 1)
    if not resp:
        out["result"] = "no reply"
    else:
        pay = resp.get("payload") or ""
        out.update({"result": "reply", "exists": bool(resp.get("exists")), "access": bool(resp.get("access")),
                    "payload_nonempty": bool(pay), "payload_looks_like_hf_token": pay.startswith("hf_"),
                    "payload_len": len(pay)})
        del pay
except Exception as e:  # noqa: BLE001 -- a probe reports, it does not crash
    out["result"] = f"error {type(e).__name__}: {str(e)[:160]}"
print("ZZSECRET " + json.dumps(out))
