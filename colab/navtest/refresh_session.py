"""Refresh (or re-adopt) ONE named colab-cli session's runtime-proxy token from the server, in a given state file.

    python refresh_session.py --config <sessions_navtest.json> --name <session> [--endpoint <ep>]

Why (RUNNER.md 9d, MEASURED 2026-08-19): the proxy token in SessionState expires after 3600 s; the CLI then reports
"session lost (404/401)", PRUNES the local entry and kills its keep-alive -- while the VM is alive and billing. A P0
job can outlive one token. `list_assignments()` returns each live assignment's CURRENT runtime_proxy_info, so this
rewrites the token and url of the session whose endpoint matches, keeping every other field (keep-alive pid, kernel
and session ids). With --endpoint and no local entry it re-adopts (RUNNER.md 9b) -- only that endpoint, never
another agent's assignment. Prints ZZREFRESH_OK <name> <expires_s> / ZZREFRESH_FAIL <why>; never prints a token.
"""
import argparse
import sys

from colab_cli.common import state
from colab_cli.state import SessionState


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--endpoint", default=None)
    a = ap.parse_args()
    state.config_path = a.config
    s = state.store.get(a.name)
    ep = a.endpoint or (s.endpoint if s else None)
    if not ep:
        print(f"ZZREFRESH_FAIL no local session {a.name!r} and no --endpoint")
        return 1
    hit = [x for x in state.client.list_assignments() if x.endpoint == ep]
    if not hit:
        print(f"ZZREFRESH_FAIL endpoint {ep} is not assigned on the server (reclaimed or stopped)")
        return 2
    rp = hit[0].runtime_proxy_info
    if s is None:
        s = SessionState(name=a.name, token=rp.token, url=rp.url, endpoint=ep, variant=str(hit[0].variant),
                         accelerator=str(hit[0].accelerator))
    else:
        s = s.model_copy(update={"token": rp.token, "url": rp.url})
    state.store.add(s)
    print(f"ZZREFRESH_OK {a.name} {rp.token_expires_in_seconds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
