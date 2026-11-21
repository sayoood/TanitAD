#!/usr/bin/env python3
"""S3 XML LISTING (metadata only: keys + sizes; NO object body is read) of the public nuPlan bucket. Every request is logged to raw/requests_log.jsonl.
Usage: s3_list.py <prefix> <out.json> [delimiter]"""
import sys, json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET, os
BASE = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/"
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw", "requests_log.jsonl")
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

def log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

def list_prefix(prefix, delimiter="/"):
    keys, prefixes, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        url = BASE + "?" + urllib.parse.urlencode(q)
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read(); status = r.status
        except Exception as e:
            log({"url": url, "method": "GET(list)", "error": repr(e), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
            return {"error": repr(e), "keys": keys, "prefixes": prefixes}
        log({"url": url, "method": "GET(list)", "status": status, "bytes": len(body), "secs": round(time.time() - t0, 2), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        root = ET.fromstring(body)
        for c in root.findall(NS + "Contents"):
            keys.append({"key": c.find(NS + "Key").text, "size": int(c.find(NS + "Size").text), "lastmod": c.find(NS + "LastModified").text})
        for p in root.findall(NS + "CommonPrefixes"):
            prefixes.append(p.find(NS + "Prefix").text)
        if root.find(NS + "IsTruncated").text == "true":
            token = root.find(NS + "NextContinuationToken").text
        else:
            break
    return {"keys": keys, "prefixes": prefixes}

if __name__ == "__main__":
    prefix, out = sys.argv[1], sys.argv[2]
    delim = sys.argv[3] if len(sys.argv) > 3 else "/"
    r = list_prefix(prefix, delim)
    r["prefix"] = prefix
    json.dump(r, open(out, "w"), indent=1)
    print(prefix, "keys", len(r["keys"]), "prefixes", len(r["prefixes"]), r.get("error", ""))
