# Google Drive egress: the three domains, measured — 2026-08-08

**Question.** Can a Claude-Code-on-the-web / pod session reach Google Drive, and which hosts does
that actually require? The programme's raw material lives on Drive (`G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD`
on the Windows dev box), but a web session gets a fresh clone and an egress proxy — no mount.

**Answer.** Yes. All three hosts are already permitted by this session's egress policy, and the
download chain is intact. No network-policy change is needed. What was missing was a checked,
repeatable way to *use* it — now `tools/gdrive_fetch.py`.

---

## 1. The domain set

| host | role | why it cannot be dropped |
|---|---|---|
| `drive.google.com` | entry point | `/uc?export=download` serves **no bytes** — it answers **303** |
| `drive.usercontent.google.com` | **the bytes** | allow only the entry point and every download dies one hop short |
| `*.googleusercontent.com` | adjacent content hosts | `lh3.`, `drive-thirdparty.`, the older `doc-XX-XX-docs.` shards |

## 2. Measurement

**Evidence class: MEASURED (ours).** Three independent tools, same session, 2026-08-08 —
`curl`, `python3`+`requests`, and `python3`+`urllib` (the tool's own `--check`). Raw output in
`raw/`.

```
drive.google.com                          302   ssl_verify_result=0
drive.google.com/uc?export=download&id=…  303 → https://drive.usercontent.google.com/download?id=…
drive.usercontent.google.com              404   ssl_verify_result=0
drive.usercontent.google.com/download?…   404   ssl_verify_result=0
lh3.googleusercontent.com                 400   ssl_verify_result=0
drive-thirdparty.googleusercontent.com    404   ssl_verify_result=0
```

`tools/gdrive_fetch.py --check` → all three `REACHABLE`, exit 0.

**Reading the negatives correctly is the whole point.** The 404s and the 400 are *healthy* — the
host answered us. Only a **403/407** (a proxy policy denial, per `/root/.ccr/README.md`) or a dead
tunnel is a real block. Reading a 404 as "Drive is blocked" would be the `df` trap in a fourth
costume: a probe answering a different question than the one asked. The agent proxy reported
`enabled: true`, `selective: false`, `recentRelayFailures: []` at probe time
(`raw/agentproxy_status_2026-08-08.json`).

**TLS:** `ssl_verify_result=0` on every hop — the pre-configured CA bundle is being read; nothing
needed pointing at it.

## 3. ⚠️ The scope caveat — these three cover ANONYMOUS downloads only

**MEASURED:** an unauthenticated `https://drive.google.com/` answers **302 →
`accounts.google.com/ServiceLogin`** — a host that is *not* on the allowlist and is deliberately
not being added.

⇒ The three domains fetch files shared as **"Anyone with the link"**. They do **not** carry
authenticated Drive access, which would additionally need `accounts.google.com` and the Google API
hosts.

This matters because of how the failure presents: a sign-in redirect looks like *"our allowlist is
too narrow"* and invites someone to widen it, when the true cause is *"this file is not shared"*.
`gdrive_fetch` therefore names that case explicitly in the refusal message rather than emitting a
generic block (`_auth_hint`, and `test_a_sign_in_redirect_says_the_file_is_private_not_that_the_allowlist_is_wrong`).

**Not measured, and therefore not claimed:** whether an authenticated/OAuth Drive flow is permitted
by this egress policy. No probe was run against `accounts.google.com` or `www.googleapis.com`, so
this document says nothing about them either way. Separately, the session carries a **Google Drive
MCP connector**, which is a different access path from HTTPS egress and was not exercised here.

## 4. What the allowlist is FOR

It is enforced in code, not written in a comment. Every redirect hop is re-checked and a hop that
leaves the allowlist is **refused, not followed**. The input to this tool is a share URL, and share
URLs arrive pasted out of docs, chat, and other agents' reports — "follow whatever `Location` comes
back" would turn a pasted string into arbitrary outbound egress.

Matching detail that carries the property: the wildcard is a **dot-boundary** suffix match, so
`notgoogleusercontent.com` and `googleusercontent.com.attacker.test` are refused and the bare apex
is not in the set. A plain `endswith` is the classic form of that bug; it is under test.

## 5. Deliverable manifest

| artifact | where it lives |
|---|---|
| `tools/gdrive_fetch.py` | repo, staged |
| `tools/tests/test_gdrive_fetch.py` (57 tests, offline) | repo, staged |
| `tools/README.md` — `## gdrive_fetch` section | repo, staged |
| `CLAUDE.md` — traps-preflight entry | repo, staged |
| this report | repo, staged |
| `raw/curl_probe_2026-08-08.txt` | repo, staged |
| `raw/gdrive_domain_probe_2026-08-08.json` (`--check` output) | repo, staged |
| `raw/agentproxy_status_2026-08-08.json` | repo, staged |

Nothing is stranded on a pod, in a worktree, or in an agent's context.

## 6. Two defects the tests caught during implementation

Both were caught by a test written *before* the behaviour was confirmed, which is the reason to
write them that way:

1. **A 404 was classified `UNREACHABLE`.** The first `_classify` treated every non-403/407 error as
   a dead host — so the preflight would have reported all three domains blocked while they were
   answering normally. That is precisely the misread this instrument exists to prevent, and it was
   in the instrument.
2. **The bare-id guard was silently dodged** by a short test stub, so the fetch tests were not
   exercising the parse path they appeared to cover.
