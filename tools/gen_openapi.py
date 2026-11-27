"""Build studio/static/openapi.json from the routes in studio/server.py (do_GET / do_POST).
The server routes by hand, so this reads the route literals out of the source: `python tools/gen_openapi.py` rewrites the
file, `--check` fails if it is stale (the test suite runs that). Local only: the spec holds paths, never data."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "studio" / "server.py"
OUT = ROOT / "studio" / "static" / "openapi.json"
NAMES = ["card", "run", "pack", "template", "version", "id", "name", "file"]
SKIP = {"/", "/idtest", "/openapi.json", "/api/docs"}


def split_top(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(": depth += 1
        if ch == ")": depth -= 1
        if ch == "|" and depth == 0: out.append(cur); cur = ""
        else: cur += ch
    return out + [cur]


def expand(rx):
    """A route regex to path templates: literal alternations fan out, other groups become a \\0regex\\1 marker (named later)."""
    m = re.search(r"\((?!\?)", rx)
    if not m: return [rx.replace("\\.", ".")]
    i, depth = m.start(), 0
    for j in range(i, len(rx)):
        depth += rx[j] == "("; depth -= rx[j] == ")"
        if depth == 0: break
    inner, head, tail = rx[i + 1:j], rx[:i], rx[j + 1:]
    alts = split_top(inner)
    mids = alts if len(alts) > 1 else None
    tails = expand(tail)
    if mids is None: return [head + "\0" + inner + "\1" + t for t in tails]
    return [head + a + t for mid in mids for a in expand(mid) for t in tails]


def name_params(path):
    """Marker groups to {name}: the segment before it for card/run/pack/template, else by the regex's shape."""
    out = []
    def one(m):
        rx, before = m.group(1), path[:m.start()].rstrip("/").rsplit("/", 1)[-1]
        nm = before if before in ("card", "run", "pack", "template") else "id" if "32" in rx else "version" if "\\d+" in rx else "col" if rx.startswith("c") else "file" if before == "download" else "name"
        while nm in out: nm += "_"
        out.append(nm); return "{" + nm + "}"
    return re.sub("\0(.*?)\1", one, path), out


def routes():
    src = SRC.read_text().splitlines()
    start = {m: next(i for i, l in enumerate(src) if f"def do_{m}(" in l) for m in ("GET", "POST")}
    end = {"GET": start["POST"], "POST": len(src)}
    found = {}
    for m in ("GET", "POST"):
        for l in src[start[m]:end[m]]:
            for lit in re.findall(r'(?:p|self\.path(?:\.split\("\?"\)\[0\])?) == "(/[^"]*)"', l): found.setdefault(lit, set()).add(m)
            for rx in re.findall(r'(?:fullmatch|match)\(r"(/[^"]+)"', l):
                if rx.startswith("/api/") or rx.startswith("/js"):
                    for t in expand(rx): found.setdefault(t, set()).add(m)
            if m == "GET":
                for pre in re.findall(r'p\.startswith\("(/api/[^"]*)"\)', l):
                    if not any(x.startswith(pre) for x in found): found.setdefault(pre + "\0[\\w-]+\1", set()).add(m)
    return found


def build():
    paths = {}
    for raw, methods in sorted(routes().items()):
        if raw in SKIP or not raw.startswith("/api/"): continue
        path, params = name_params(raw)
        item = paths.setdefault(path, {})
        for m in sorted(methods):
            ctype = "image/png" if path.endswith(".png") else "application/octet-stream" if re.search(r"/(download|reader\.tflite)", path) else "application/json"
            op = {"tags": [path.split("/")[2]], "summary": f"{m} {path}",
                  "parameters": [{"name": p, "in": "path", "required": True, "schema": {"type": "string"}} for p in params],
                  "responses": {"200": {"description": "OK", "content": {ctype: {"schema": {} if ctype == "application/json" else {"type": "string", "format": "binary"}}}},
                                "4XX": {"description": "Refused or invalid: {\"error\": message}"}}}
            if m == "POST": op["requestBody"] = {"content": {"application/json": {"schema": {"type": "object"}}}}
            item[m.lower()] = op
    return {"openapi": "3.0.3", "info": {"title": "Card Studio local API", "version": "1",
            "description": "Local only (127.0.0.1). Generated from studio/server.py by tools/gen_openapi.py; request and response bodies are not modelled. Cross-site requests are refused, so use this page from the studio's own origin."},
            "paths": paths}


if __name__ == "__main__":
    text = json.dumps(build(), indent=1) + "\n"
    if "--check" in sys.argv:
        sys.exit(0 if OUT.exists() and OUT.read_text() == text else "openapi.json is stale: run python tools/gen_openapi.py")
    OUT.write_text(text); print(f"{len(build()['paths'])} paths -> {OUT.relative_to(ROOT)}")
