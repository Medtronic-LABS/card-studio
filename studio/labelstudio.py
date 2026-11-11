"""Optional second way to label a card's cells: a Label Studio running on this machine (open source, Apache-2.0).

Studio's own Label tab needs nothing else. Label Studio is for when several people label, or a reviewer checks labels. Studio then
    push   makes one Label Studio project per card and one task per inked cell that has no label yet;
    pull   reads what was annotated there and records it through labeling.save, so every label passes the same checks.
Cell images are not copied or uploaded: Label Studio serves them from studio/cards/ through its local-files storage, which is why it has to
be started with that folder as its document root (`python labelstudio.py start` does that). Both programs listen on 127.0.0.1 only, and
this module refuses any Label Studio address that is not on this machine.

Settings are in studio/labelstudio.json (gitignored: it holds the access token): {"url": "http://127.0.0.1:8780", "token": "..."}.
`start` writes it the first time, with a generated local account, and runs the Label Studio installed in studio/.venv-labelstudio
(python -m venv .venv-labelstudio && .venv-labelstudio/bin/pip install label-studio).

A pull records the Label Studio answer for every annotated cell, replacing what Studio had for that cell. A cell marked blank or unreadable
there is recorded as that. An answer the field cannot hold is not recorded and is listed in the result.

CLI: python labelstudio.py start | status | push <card> | pull <card>
"""
import json, os, secrets, sys, urllib.error, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cards, labeling

CONFIG, VENV, DATA = HERE / "labelstudio.json", HERE / ".venv-labelstudio", HERE / "labelstudio-data"
URL = "http://127.0.0.1:8780"
LOCAL = {"127.0.0.1", "localhost", "::1"}
TITLE = "Card Studio: {}"
PAGE_SIZE = 500
LABEL_CONFIG = """<View>
  <Header value="$field ($kind) · page $page · visit $visit"/>
  <Image name="image" value="$image" zoom="true" maxWidth="700px"/>
  <TextArea name="label" toName="image" rows="1" maxSubmissions="1" editable="true" placeholder="what the cell says"/>
  <Choices name="note" toName="image" choice="single" showInLine="true">
    <Choice value="blank"/>
    <Choice value="unreadable"/>
  </Choices>
</View>"""


class Unavailable(Exception):
    """Label Studio is not set up, not running, or refused the token."""


def config():
    try: c = json.loads(CONFIG.read_text())
    except (OSError, ValueError): c = {}
    url = (c.get("url") or URL).rstrip("/"); host = urllib.parse.urlparse(url).hostname
    if urllib.parse.urlparse(url).scheme != "http" or host not in LOCAL:
        raise Unavailable(f"Label Studio address {url} is not on this machine: cell images are patient data and are only served locally")
    return {"url": url, "token": c.get("token") or ""}


def _call(method, path, body=None, cfg=None, timeout=60):
    cfg = cfg or config()
    if not cfg["token"]: raise Unavailable("no Label Studio token yet: run `python studio/labelstudio.py start`, or put {\"url\", \"token\"} in studio/labelstudio.json")
    req = urllib.request.Request(cfg["url"] + path, data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers={"Authorization": f"Token {cfg['token']}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(); return json.loads(data) if data else None
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "replace")
        if e.code in (401, 403): raise Unavailable(f"Label Studio refused the token ({e.code}): {detail}")
        raise ValueError(f"Label Studio answered {e.code} to {method} {path}: {detail}")
    except (urllib.error.URLError, OSError) as e:
        raise Unavailable(f"Label Studio is not reachable at {cfg['url']} ({getattr(e, 'reason', e)}): start it with `python studio/labelstudio.py start`")


def _project(card, create=False):
    title = TITLE.format(card); res = _call("GET", f"/api/projects?page_size={PAGE_SIZE}")
    found = [p for p in (res.get("results", res) if isinstance(res, dict) else res) if p.get("title") == title]
    if found: return found[0]
    if not create: return None
    p = _call("POST", "/api/projects", {"title": title, "description": f"Cells of the {card} card, cut by Card Studio. Labels are pulled back into Studio.", "label_config": LABEL_CONFIG})
    # the storage entry is what allows Label Studio to serve files from this card's folder; nothing is synced or copied
    _call("POST", "/api/storages/localfiles", {"project": p["id"], "title": f"{card} cells", "path": str((cards.CARDS / card).resolve()), "use_blob_urls": False})
    return p


def _tasks(pid):
    out, page = [], 1
    while True:
        try: res = _call("GET", f"/api/tasks?project={pid}&page={page}&page_size={PAGE_SIZE}&fields=all")
        except ValueError as e:
            if page > 1 and " 404 " in str(e): break       # past the last page
            raise
        chunk = res.get("tasks", res.get("results", [])) if isinstance(res, dict) else res
        out += chunk
        if len(chunk) < PAGE_SIZE: break
        page += 1
    return out


def _key(task):
    d = task.get("data") or {}; v = str(d.get("visit", ""))        # a task shows the visit as a number; Studio's key is cN
    return (str(d.get("page", "")), str(d.get("field", "")), v if v.startswith("c") else f"c{v}")


def image_url(card, field, visit, page):
    """What a task's image points at: the crop on disk, relative to Label Studio's document root (studio/cards)."""
    return "/data/local-files/?d=" + urllib.parse.quote(f"{card}/crops/{field}/{visit}/{page}.png")


def push(card, field=None):
    """Make a task for every inked cell that Studio has no record for and Label Studio has no task for. -> {project, url, added, already}."""
    todo = []; after = None
    while True:
        q = labeling.queue(card, field, "todo", 500, after)["cells"]
        todo += q
        if len(q) < 500: break
        after = "|".join((q[-1]["page"], q[-1]["field"], q[-1]["visit"]))
    kinds = {f["key"]: f["kind"] for f in cards.fields(card)}
    p = _project(card, create=True); have = {_key(t) for t in _tasks(p["id"])}
    new = [c for c in todo if (c["page"], c["field"], c["visit"]) not in have]
    for i in range(0, len(new), 200):
        _call("POST", f"/api/projects/{p['id']}/import", [{"data": {"image": image_url(card, c["field"], c["visit"], c["page"]), "card": card, "field": c["field"],
                                                                     "kind": kinds[c["field"]], "visit": c["visit"][1:], "page": c["page"]}} for c in new[i:i + 200]], timeout=300)
    return {"card": card, "project": p["id"], "url": f"{config()['url']}/projects/{p['id']}", "added": len(new), "already": len(todo) - len(new)}


def _answer(task):
    """(label, note) from a task's newest annotation that was not skipped, or None."""
    anns = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
    if not anns: return None
    a = max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or ""); label, note = "", ""
    for r in a.get("result") or []:
        v = r.get("value") or {}
        if r.get("type") == "textarea" and v.get("text"): label = str(v["text"][0])
        if r.get("type") == "choices" and v.get("choices"): note = str(v["choices"][0])
    return (label, note) if (label.strip() or note) else None


def pull(card):
    """Record every annotated cell in Studio. -> {annotated, new, changed, same, rejected: [{page, field, visit, label, reason}]}."""
    labeling._card(card); p = _project(card)
    if p is None: raise ValueError(f"no Label Studio project for {card} yet: send cells to it first")
    have = labeling._read(card); new = changed = same = 0; rejected = []
    for t in _tasks(p["id"]):
        ans = _answer(t)
        if ans is None: continue
        page, field, visit = _key(t); label, note = ans
        note = note if note in ("blank", "unreadable") else ""
        was = have.get((page, field, visit))
        try:
            r = labeling.save(card, page, field, visit, label, note, confirm=True)
        except (ValueError, KeyError) as e:
            rejected.append({"page": page, "field": field, "visit": visit, "label": label, "reason": str(e).strip("'\"")}); continue
        if was is None: new += 1
        elif (was["label"], was["note"]) != (r["label"], r["note"]): changed += 1
        else: same += 1
    return {"card": card, "project": p["id"], "annotated": new + changed + same + len(rejected), "new": new, "changed": changed, "same": same, "rejected": rejected}


def status(card=None):
    """What the Label tab shows: is Label Studio set up and running, and for a card, how far its project has got. Never raises."""
    out = {"installed": (VENV / "bin" / "label-studio").exists(), "configured": False, "running": False, "url": URL, "problem": "", "project": None}
    try:
        cfg = config(); out["url"] = cfg["url"]; out["configured"] = bool(cfg["token"])
        _call("GET", "/api/projects?page_size=1", cfg=cfg, timeout=5); out["running"] = True
        if card:
            labeling._card(card); p = _project(card)
            if p: out["project"] = {"id": p["id"], "url": f"{cfg['url']}/projects/{p['id']}", "tasks": p.get("task_number"), "annotated": p.get("num_tasks_with_annotations")}
    except (Unavailable, ValueError, KeyError) as e:
        out["problem"] = str(e).strip("'\"")
    return out


def start(port=8780):
    """Run the Label Studio in studio/.venv-labelstudio on 127.0.0.1 with studio/cards as its local-files root. The first time, a local
    account and access token are generated and written to labelstudio.json; they are never printed."""
    exe = VENV / "bin" / "label-studio"
    if not exe.exists(): sys.exit(f"Label Studio is not installed in {VENV}.\n  python -m venv {VENV} && {VENV}/bin/pip install label-studio")
    try: c = json.loads(CONFIG.read_text())
    except (OSError, ValueError): c = {}
    if not c.get("token"):
        c = {"url": f"http://127.0.0.1:{port}", "token": secrets.token_hex(20), "username": "studio@example.com", "password": secrets.token_urlsafe(18)}
        CONFIG.write_text(json.dumps(c, indent=1)); CONFIG.chmod(0o600)
    cards.CARDS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true", "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(cards.CARDS.resolve()),
           "LOCAL_FILES_SERVING_ENABLED": "true", "LOCAL_FILES_DOCUMENT_ROOT": str(cards.CARDS.resolve()),
           "LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN": "true", "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK": "true", "COLLECT_ANALYTICS": "false", "LABEL_STUDIO_COLLECT_ANALYTICS": "false"}
    args = [str(exe), "start", "--no-browser", "--internal-host", "127.0.0.1", "--host", f"http://127.0.0.1:{port}", "--port", str(port), "--data-dir", str(DATA)]
    if c.get("username"): args += ["--username", c["username"], "--password", c["password"], "--user-token", c["token"], "--enable-legacy-api-token"]
    print(f"Label Studio on http://127.0.0.1:{port} (this machine only); cell images are served from {cards.CARDS}", flush=True)
    os.execve(str(exe), args, env)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    try:
        if cmd == "start": start()
        elif cmd == "status": print(json.dumps(status(sys.argv[2] if len(sys.argv) > 2 else None), indent=1))
        elif cmd == "push" and len(sys.argv) == 3: print(json.dumps(push(sys.argv[2]), indent=1))
        elif cmd == "pull" and len(sys.argv) == 3: print(json.dumps(pull(sys.argv[2]), indent=1))
        else: sys.exit(__doc__.split("CLI:")[1].strip())
    except (Unavailable, ValueError, KeyError) as e:
        sys.exit(str(e).strip("'\""))
