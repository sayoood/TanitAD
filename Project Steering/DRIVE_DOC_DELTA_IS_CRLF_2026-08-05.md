# ⛔ RETRACTED — "~2.7 KB of unrecovered Google Drive doc edits" is a LINE-ENDING ARTIFACT

**Status:** CLOSED. **Evidence class: MEASURED** (2026-08-05, byte comparison against the raw
Drive bytes). **This was PI priority item 2 and it is not work — there is nothing to recover.**

---

## The claim

The session handoff carried, as priority item 2, *"recover ~2.7 KB of Google Drive doc edits"*:

| file | claimed delta |
|---|---|
| `PROJECT_STATE.md` | +395 B |
| `Paper/TANITAD_PAPER.md` | +1,865 B |
| `README.md` | +369 B |
| `.gitignore` | +51 B |

The reasoning was reasonable: Drive's copies are **larger** than the repo's and Drive's
`modifiedTime` is **later** than the last commit of each file. Both are true. Neither implies an edit.

## The measurement

**Every delta equals that file's LINE COUNT, exactly.**

| file | repo bytes (LF) | repo newlines | Drive bytes | delta | verdict |
|---|---|---|---|---|---|
| `PROJECT_STATE.md` | 101,754 | **395** | 102,149 | **+395** | ✅ **byte-identical after CRLF→LF** |
| `Paper/TANITAD_PAPER.md` | 157,043 | **1,865** | 158,908 | **+1,865** | ✅ **byte-identical after CRLF→LF** |
| `README.md` | 27,706 | **369** | 28,075 | +369 | arithmetic match (not byte-verified) |
| `.gitignore` | 910 @ `7f34086` | **51** | 961 | +51 | arithmetic match (not byte-verified) |

The two large files were **verified by byte comparison**: the raw Drive bytes were fetched, both
sides normalised `\r\n → \n`, and the results are **equal, byte for byte** — 101,754 == 101,754 and
157,043 == 157,043. The other two match by exact newline arithmetic and were not downloaded; that
distinction is stated rather than glossed.

⚠️ `.gitignore` needed the **historical** repo size: it is 1,016 B at HEAD because this session added
`**/gotty_url.txt`. At `7f34086` — the pre-session state the handoff measured — it was **910 B with
51 newlines**, and 910 + 51 = **961** = Drive.

⇒ **The Drive copies are the same documents with CRLF line endings.** The sync writes CRLF; the repo
holds LF. There are no unrecovered edits.

---

## ⚠️ The trap in the measurement itself — `read_file_content` is NOT the file

The Drive connector's `read_file_content` returns a *natural-language representation*, not the bytes.
On `PROJECT_STATE.md` it returned **106,585 B** for a **102,149 B** file, **escaped the markdown**
(`\# TanitAD` for `# TanitAD`) and appended trailing spaces — so a diff built on it showed **293 of
395 lines differing** and would have "recovered" hundreds of phantom changes into the repo.

⇒ **For any byte-level comparison use `download_file_content` (base64 of the real bytes), never
`read_file_content`.** The latter is for reading, and it says so; it is easy to reach for anyway.

---

## Root-cause class — this is the THIRD CRLF incident of the session

`RETRACTION_LOG.md` class: **a line-ending difference read as a content difference**.

1. The **v5 trainer pin** went red on a file nobody had edited — the CRLF hash reproduced the old
   pin and the git blob was identical on both refs. Fixed by hashing normalised bytes.
2. This one — a whole *work item on the PI's priority list*, sized in kilobytes, that was zero bytes
   of content.
3. And the general form already in `CLAUDE.md`: **a size or timestamp difference is not a content
   difference.** Compare content, normalised, before sizing the work.

⚠️ Note how plausible the wrong reading was: *bigger file* + *later mtime* + *four files at once*.
Every signal pointed at real edits. The cheap discriminating check — does the delta equal the line
count? — takes one `wc -l` per file and would have closed it immediately.

**Standing check for anything Drive-synced:** before diffing, normalise line endings; if the byte
delta equals the newline count, stop — it is the sync, not an edit.
