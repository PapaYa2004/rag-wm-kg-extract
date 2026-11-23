"""A tiny OpenAI-compatible server that answers DSPy ChatAdapter prompts
deterministically. Used only by the offline end-to-end test (no GPU, no LLM).

Behaviour by passage content:
  "GARBAGE" -> unparseable entity answer (exercises entity parse-failure counting)
  "BADREL"  -> invalid primary relation answer (exercises KGGen's relation fallback)
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VOCAB = ["statin", "breast cancer", "mortality", "Finland", "cholesterol", "vitamin D",
         "Vitamin D", "heart disease", "omega-3", "fish oil", "inflammation", "BADREL"]


def _fields(system: str) -> list[str]:
    m = re.search(r"Your output fields are:\n(.*?)(?:\n(?:All interactions|Output format|In adhering)|\Z)", system, re.S)
    return re.findall(r"\d+\. `(\w+)`", m.group(1)) if m else []


def _value(raw: str):
    raw = re.split(r"\n\nRespond with ", raw)[0].strip()
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    items = re.findall(r"<item>\s*(.*?)\s*</item>", raw, re.S)
    return items if items or "<item>" in raw else raw


def _blocks(user: str) -> dict:
    """Inputs in either DSPy format: [[ ## f ## ]] markers or <f>...</f> tags."""
    out = {}
    parts = re.split(r"\[\[ ## (\w+) ## \]\]\n", user)
    for i in range(1, len(parts) - 1, 2):
        out[parts[i]] = _value(parts[i + 1])
    for name, body in re.findall(r"<(\w+)>\n(.*?)\n</\1>", user, re.S):
        out.setdefault(name, _value(body))
    return out


def _xml_item(x) -> str:
    if isinstance(x, dict):
        return "\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in x.items())
    return str(x)


def _render(vals: dict, mode: str) -> str:
    if mode == "json":      # DSPy JSONAdapter
        return json.dumps(vals)
    if mode == "markers":   # classic DSPy ChatAdapter
        return "".join(f"[[ ## {k} ## ]]\n{v if isinstance(v, str) else json.dumps(v)}\n\n"
                       for k, v in vals.items()) + "[[ ## completed ## ]]"
    out = []                # DSPy 3.5 ChatAdapter tag format
    for k, v in vals.items():
        body = "\n".join(f"<item>\n{_xml_item(x)}\n</item>" for x in v) if isinstance(v, list) else str(v)
        out.append(f"<{k}>\n{body}\n</{k}>")
    return "\n".join(out)


def answer(messages: list[dict]) -> str:
    system = messages[0]["content"]
    user = messages[-1]["content"]
    if "Respond with a JSON object" in user:
        mode = "json"
    elif "Output format (replace placeholders):" in system + user:
        mode = "xml"
    else:
        mode = "markers"
    outs = _fields(system)
    inp = _blocks(user)
    text = inp.get("source_text", "")
    vals = {}
    for f in outs:
        if f == "entities":
            if "GARBAGE" in text:
                return "I cannot answer in that format."
            vals[f] = [v for v in VOCAB if v.lower() in text.lower() and v != "BADREL"
                       and not (v == "Vitamin D" and "vitamin D" in text)]
        elif f in ("relations", "fixed_relations"):
            ents = inp.get("entities", [])
            fallback = "must be one of" in system
            if "BADREL" in text and not fallback and f == "relations":
                vals[f] = [{"subj": "nope"}]  # fails pydantic validation -> fallback path
                continue
            rels = [{"subject": a, "predicate": "is associated with", "object": b}
                    for a, b in zip(ents, ents[1:])]
            if "vitamin D" in ents and "breast cancer" in ents:
                rels.append({"subject": "vitamin D", "predicate": "reduces risk of", "object": "breast cancer"})
            vals[f] = rels
        elif f == "duplicates":
            item = inp.get("item", "")
            pool = inp.get("set", [])
            vals[f] = [x for x in pool if x != item and x.lower() == item.lower()]
        elif f == "alias":
            vals[f] = inp.get("item", "")
        elif f == "reasoning":
            vals[f] = "ok"
        else:
            vals[f] = ""
    return _render(vals, mode)


class Handler(BaseHTTPRequestHandler):
    model = "fake-model"

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [{"id": self.model, "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = answer(req["messages"])
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": self.model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})


def start(port: int = 0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
