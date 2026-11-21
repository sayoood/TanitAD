"""READ-ONLY compute-unit readout for the Colab account behind the cached colab-cli token (0.6.0 has no `usage`).

    $env:PYTHONUTF8="1"; $env:PYTHONPATH="D:\\Projects\\TanitAD\\colab\\win_shims"
    C:\\Users\\Admin\\venvs\\colab\\Scripts\\python.exe D:\\Projects\\TanitAD\\colab\\ccu_info.py

Prints one `ZZCCU {json}` line: balance, the CURRENT hourly burn (non-zero while any VM is assigned -- the only
measured source of a GPU's price), live assignments (tokens dropped) and accelerator eligibility. Allocates
nothing. The endpoint is Colab's own `/tun/m/ccu-info`, the call the web UI makes, on the CLI's own session.
The same account's subscription tier (`colab.pa.googleapis.com/v1/user-info`) answers 403 to this OAuth client
(MEASURED 2026-09-27), so eligibility + balance is the admissible evidence of the plan.
Account: fambouzouraa@gmail.com holds the PI's Google AI plan (PI, 2026-09-27).
"""
import json
import sys
from urllib.parse import urljoin

from colab_cli.client import TUN_ENDPOINT
from colab_cli.common import state

c = state.client
out = {}
try:
    out["ccu"] = c._issue_request(urljoin(c.colab_domain, f"{TUN_ENDPOINT}/ccu-info"), schema=dict)
except Exception as e:  # noqa: BLE001 -- a readout reports, it does not crash
    out["ccu_error"] = f"{type(e).__name__}: {str(e)[:200]}"
try:
    out["assignments"] = [{"endpoint": a.endpoint, "accelerator": str(a.accelerator), "variant": str(a.variant),
                           "machine_shape": str(a.machine_shape)} for a in c.list_assignments()]
except Exception as e:  # noqa: BLE001
    out["assignments_error"] = f"{type(e).__name__}: {str(e)[:200]}"
print("ZZCCU " + json.dumps(out))
sys.exit(0 if "ccu" in out and "assignments" in out else 1)
