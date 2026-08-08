# 2026-08-08 trajectory render — NOT RUN. Two independent blockers, both MEASURED.

*Session: the first created after the PI's environment network change, i.e. the first able to test
it. Continuation brief: `CONTINUATION_trajrecon_2026-08-08.md` (on
`claude/trajectory-reconstruction-integration-kxy9mf`).*

**Verdict: the recording was NOT fetched and NOT processed. No video, no `report.json`, no
`validate.py` numbers.** Nothing was tuned, nothing was estimated, and no metric is reported —
the inputs never arrived. The upstream hold-out figures (2.23 m / 1.27 m/s / 0.84°) remain
**INHERITED** and are still not ours.

## Blocker 1 — the Drive hosts are still policy-denied at the agent proxy (MEASURED)

The brief's Step-1 test, run in this session:

```
curl -sS -o /dev/null -w "%{http_code}\n" --max-time 20 https://drive.google.com
→ curl: (56) CONNECT tunnel failed, response 403      exit 56, code 000
```

That is exactly the brief's documented "still blocked" signature. Per `CLAUDE.md` rule 2
(*absence at one location is not absence*) the probe was widened to the hosts `gdown` actually
pulls bytes from, rather than stopping at the one hostname:

| host | code | reachable? | role |
|---|---|---|---|
| `drive.google.com` | 000 | ⛔ CONNECT 403 | gdown's confirm-token flow |
| `drive.usercontent.google.com` | 000 | ⛔ CONNECT 403 | **where the bytes actually come from** |
| `docs.google.com` | 000 | ⛔ CONNECT 403 | legacy download path |
| `www.googleapis.com` | 403 | ✅ host reachable | Drive REST API — *answered by Google* |
| `storage.googleapis.com` | 400 | ✅ | — |
| `github.com` / `api.github.com` | 400 / 200 | ✅ | — |
| `objects.githubusercontent.com` | 404 | ✅ | **release-asset CDN** |
| `raw.githubusercontent.com` | 301 | ✅ | — |
| `huggingface.co`, `cdn-lfs.huggingface.co` | 000 | ⛔ | — |

Corroborated by the proxy's own log (`$HTTPS_PROXY/__agentproxy/status`), which names the denial
rather than leaving it inferred:

```json
{"kind":"connect_rejected",
 "detail":"gateway answered 403 to CONNECT (policy denial or upstream failure)",
 "host":"drive.google.com:443", "ts":"2026-08-08T18:23:44.525Z"}
```
(identical entries for `drive.usercontent.google.com:443` and `docs.google.com:443`.)

⚠️ **`www.googleapis.com` is reachable and is a real Drive endpoint**, so the route is *not*
uniformly closed — but it is credential-gated and the sandbox has no Google credential:

```
GET https://www.googleapis.com/drive/v3/files/1Ditz…?alt=media
→ 403 {"message":"The request is missing a valid API key."}
```

An API key would not help anyway — see Blocker 2.

## Blocker 2 — the file was never link-shared (MEASURED, and NEW)

The brief assumed sharing was in place: *"the fetch is anonymous and the file must be shared
'anyone with the link'."* It is not. The Drive connector reports exactly one permission:

```json
{"permissions":[{"displayName":"fambouzouraa","emailAddress":"fambouzouraa@gmail.com",
                 "role":"owner","type":"user"}]}
```

**There is no `anyone`/`link` role.** ⇒ Fixing the network alone would NOT have unblocked this:
`gdown` would have fetched the *interstitial/permission-denied HTML* — the few-KB failure the brief
warns reads like a successful download. **Both fixes are required; neither is sufficient alone.**

The file itself is confirmed correct and intact, so nothing is lost:

| field | value |
|---|---|
| title | `2026-08-08_14-19-54-android.zip` |
| `fileSize` | **179,607,367 B** — byte-exact match to the brief's expected size |
| mimeType | `application/zip` |
| id / parent | `1DitzfASALJhJ5GqE-sBEqa3t_oyOipHN` / `10lR3QsRtG_LLigYtubXvBEHcngFOwXhg` |

## Why there is no workaround (do not re-derive this)

The Drive MCP connector's `download_file_content` returns the whole file **inline as base64** and
its schema exposes only `fileId` / `exportMimeType` — **no range, offset or chunk parameter**
(checked against the live schema, not assumed). 179,607,367 B → ~239 MB of base64 ≈ tens of
millions of tokens. No context window holds it, and it cannot be paged. This matches the brief's
standing instruction: **do not burn a session working around it.**

## What unblocks it — pick either

**Option A — re-host, needs NO environment change (fastest).** `objects.githubusercontent.com` and
`api.github.com` are already reachable. Attach the zip as a **GitHub release asset** on
`sayoood/TanitAD` (assets allow up to 2 GB; the 100 MB limit applies to files in git, not releases)
and the next session fetches it with a plain `curl`. Any host on the reachable list works.

**Option B — fix both Drive blockers.** ① Set the file to **"Anyone with the link"**. ② Add
`drive.google.com`, `drive.usercontent.google.com`, `*.googleusercontent.com` to the environment's
allowed hosts, keeping the default package-manager list. ⚠️ **An environment change applies only to
sessions created after it** — the change must be made *before* the next session starts, and that
session must re-run the Step-1 probe first.

## Then the run is one command

Nothing about the pipeline is known-broken; 84 tests pass and it is **not** to be re-audited.
Once the zip is at `~/trajdata/in/` and verified at **179,607,367 bytes**:

```bash
pip install numpy scipy pandas opencv-python-headless matplotlib gdown
apt-get update -qq && apt-get install -y ffmpeg     # ffprobe too — imageio-ffmpeg is NOT enough
cd /home/user/TanitAD/stack
python -m tanitad.data.trajrecon.pipeline \
    --input-dir ~/trajdata/in --output-dir ~/trajdata/out --lane-width 3.50
```

`--lane-width 3.50` because the drive is in **France**; leave `--cam-height` at 1.17 so
`plane_calib` **measures** it. A `REJECT` verdict is a valid result — report it, do not tune flags.

## Root-cause class

**An INHERITED premise was carried as fact.** The brief stated the environment access "has since
been changed" and that the file "must be shared" — both were treated as settled preconditions.
Measured: the first was not effective for the Drive hosts, and the second was never done at all.
Same family as the standing rule that a claim deciding real work must be **MEASURED**, not
INHERITED — and the reason the Step-1 probe exists as a gate rather than a formality.

## Addendum — the smaller recording does NOT unblock it either (MEASURED, same session)

The PI added `2026-08-08_14-05-42-android.zip` (`1AiH-m9E5tt73Q8kyvwFqVGpfjGVvDyzK`,
**18,701,510 B**, an earlier drive the same day) and asked to try it. It is **10x smaller and still
36x too large** for the only route that does not need the network:

| | |
|---|---|
| binary | 18,701,510 B (18.7 MB) |
| base64 | 24,935,348 chars (24.9 MB) |
| tokens @ ~3.5 chars/token | **~7,124,385** |
| context window | ~200,000 |
| **over by** | **~36x** |

⛔ **The download was NOT attempted.** A truncated base64 stream yields a corrupt archive that would
then be processed as if real — the exact fictional-input failure the pipeline's `REJECT` exists to
prevent — and an untruncated one ends the session. The arithmetic decides it; no probe needed.

⚠️ **This file is ALSO owner-only**: `{"role":"owner","type":"user"}`, no `anyone`/link permission.
So Blocker 2 is not specific to the 180 MB file — **nothing in the folder is link-shared.**

**Consequence for sizing:** the inline-base64 ceiling is roughly **300-600 KB of binary**, i.e. ~0.5 %
of even this "small" zip. **No Sensor Logger recording will ever fit.** Stop treating a smaller file
as the workaround — the route is closed by orders of magnitude, not by a margin.

### Third route, discovered here: API key over the ALREADY-REACHABLE host

`www.googleapis.com` is reachable (measured above). For a file shared "anyone with the link", Drive
v3 accepts a **plain API key** — no OAuth, and **no environment change**:

```bash
curl -L -o ~/trajdata/in/rec.zip \
  "https://www.googleapis.com/drive/v3/files/<id>?alt=media&key=$GOOGLE_API_KEY"
```

Requires (a) link-sharing and (b) a Drive-API-enabled key. Costs a secret in the transcript, so the
GitHub release asset remains preferred — at 18.7 MB that upload is seconds.

## ✅ RESOLUTION 2026-08-08 — Blocker 2 CLEARED. Only "start a new session" remains.

**Blocker 2 (link-sharing) is FIXED — MEASURED, both files:**

```json
{"permissions":[{"role":"reader","type":"anyone"}, {"role":"owner","type":"user", ...}]}
```

| file | id | size | sharing |
|---|---|---|---|
| `2026-08-08_14-05-42-android.zip` | `1AiH-m9E5tt73Q8kyvwFqVGpfjGVvDyzK` | 18,701,510 B | ✅ `anyone/reader` |
| `2026-08-08_14-19-54-android.zip` | `1DitzfASALJhJ5GqE-sBEqa3t_oyOipHN` | 179,607,367 B | ✅ `anyone/reader` |

**Blocker 1 config is also in place** — the PI's screenshot shows network access `Custom` with
`drive.google.com`, `drive.usercontent.google.com`, `*.googleusercontent.com` and the
package-manager default list retained.

⛔ **But it cannot help THIS session** — re-probed after the change: `drive.google.com`,
`drive.usercontent.google.com`, `docs.google.com` all still `000`/CONNECT 403. **An environment
change applies only to sessions created after it.** This is not a fault to debug; it is the
documented semantics, and it is why Step 1 is a gate.

⚠️ **Confirmed there is no keyless side door, so do not go looking for one again.** With the file
now genuinely public, every reachable Google host was probed: `www.googleapis.com` and
`content.googleapis.com` answer `403` *"The request is missing a valid API key"* (254 B),
`drive.googleapis.com` 404s, and every `googleusercontent.com` host is still CONNECT-denied.
Public sharing does **not** make the Drive REST API anonymous.

### ⇒ The next session runs it. Nothing else is outstanding.

Sharing is done and the allowlist is configured, so a **newly created session** needs only:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" --max-time 20 https://drive.google.com   # expect 200/3xx
pip install gdown numpy scipy pandas opencv-python-headless matplotlib
apt-get update -qq && apt-get install -y ffmpeg          # ffprobe too
mkdir -p ~/trajdata/in
gdown 1AiH-m9E5tt73Q8kyvwFqVGpfjGVvDyzK -O ~/trajdata/in/2026-08-08_14-05-42-android.zip
ls -l ~/trajdata/in/            # MUST be 18,701,510 B -- a few KB means the interstitial
cd /home/user/TanitAD/stack
python -m tanitad.data.trajrecon.pipeline \
    --input-dir ~/trajdata/in --output-dir ~/trajdata/out --lane-width 3.50
```

Start with the **18.7 MB** recording (faster round trip), then repeat with the 180 MB one. Report
all four metric families, quote raw `report.json`, and treat a `REJECT` as a valid result.
