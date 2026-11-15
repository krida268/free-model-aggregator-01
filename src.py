# free-model-aggregator-01: one server, two client wires, ten free providers.
#
# Serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and
# server-sent events) over the ten simulated free-tier providers the trial's brief
# names. For every request it ranks every free model that meets the request's
# requirements by the sealed quality table's rating, tries the best first, and
# fails over to the next candidate when a provider refuses (402/429/5xx). All
# three upstream wires are spoken natively (openai, anthropic, gemini) so the
# provider's echo of the client's request survives the translation whole: the
# system prompt, user text, images, tools, stop sequences, max_tokens and JSON
# mode are carried across untouched.
#
# Routing is a pure function of the brief, the manifest and the providers'
# answers so far: no wall clock, no unseeded randomness, so a replay leaves the
# same state export. A provider that refuses twice in a row is deprioritised
# (its budget is probably spent or it is in outage) but still tried last.
#
#   python3 main.py            read the brief, probe the catalog, register, serve
#   python3 main.py --list     print the catalog document and exit 0
#
# Standard library only. Python 3.13.

import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
UPSTREAM_TIMEOUT_S = 10
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


def _object_end(text, start):
    """Index of the '}' closing the object that opens at start, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in text that carries a string schema, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _str(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


def _provider_of(v):
    if not is_record(v):
        return None
    pid = _str(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if _str(v.get(key)) != "":
            base = _str(v.get(key))
            break
    return {
        "id": pid,
        "name": _str(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _str(v.get("key_env")),
        "key": _str(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = _str(v.get("model"))
    provider = _str(v.get("provider"))
    if model == "" or provider == "":
        return None
    return {
        "model": model,
        "provider": provider,
        "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), (int, float)) else 0,
        "context_length": _num_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_request_of(v):
    if not is_record(v):
        return None
    rid = _str(v.get("id"))
    if rid == "":
        return None
    requires = v.get("requires") if is_record(v.get("requires")) else {}
    out = {"tools": requires.get("tools") is True,
           "vision": requires.get("vision") is True,
           "json": requires.get("json") is True}
    if isinstance(requires.get("min_context"), (int, float)):
        out["min_context"] = requires["min_context"]
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": out,
    }


def read_brief(path):
    """The brief at path: directory, ratings rows, workload manifest."""
    brief = {"providers": [], "ratings": [], "best": [], "manifest": [], "error": None}
    if not path:
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = _documents_in(text)
    directory = next((d for d in docs if d.get("schema") == PROVIDERS_SCHEMA
                      and isinstance(d.get("providers"), list)), None)
    if directory is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    brief["providers"] = [p for p in (_provider_of(v) for v in directory["providers"]) if p]
    ratings_doc = next((d for d in docs if d.get("schema") == RATINGS_SCHEMA
                        and isinstance(d.get("ratings"), list)), None)
    if ratings_doc is not None:
        brief["ratings"] = [r for r in (_rating_of(v) for v in ratings_doc["ratings"]) if r]
        if isinstance(ratings_doc.get("best"), list):
            brief["best"] = [{"id": _str(b.get("id")),
                              "rating_milli": b.get("rating_milli") or 0}
                             for b in ratings_doc["best"] if is_record(b)]
    workload_doc = next((d for d in docs if d.get("schema") == WORKLOAD_SCHEMA
                         and isinstance(d.get("manifest"), list)), None)
    if workload_doc is not None:
        brief["manifest"] = [w for w in (_workload_request_of(v) for v in workload_doc["manifest"]) if w]
    return brief


def rating_of_model(brief, provider_id, model_id):
    for row in brief["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# ---------------------------------------------------------------- probes


def _zero_priced(v):
    pricing = v.get("pricing") if is_record(v.get("pricing")) else None
    if pricing is None:
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        mid = _str(v.get("name")).replace("models/", "", 1) if _str(v.get("name")).startswith("models/") else _str(v.get("name"))
    else:
        mid = _str(v.get("id"))
    if mid == "":
        return None
    context = _num_or_none(v.get("context_length"))
    if context is None:
        context = _num_or_none(v.get("context_window"))
    if context is None:
        context = _num_or_none(v.get("inputTokenLimit"))
    return {
        "id": mid,
        "display_name": _str(v.get("display_name")) or _str(v.get("displayName")) or mid,
        "free": _bool_or_none(v.get("free")) if _bool_or_none(v.get("free")) is not None else _zero_priced(v),
        "context_length": context,
        "tools": _bool_or_none(v.get("tools")),
        "vision": _bool_or_none(v.get("vision")),
        "json": _bool_or_none(v.get("json")),
    }


def _models_in(wire, body):
    if not is_record(body):
        return []
    lst = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(lst, list):
        return []
    return [m for m in (_model_in(wire, v) for v in lst) if m]


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """One provider's catalog route, in its own wire. Never raises."""
    base = join_base(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as res:
            if res.status != 200:
                return {"provider": provider, "ok": False, "status": res.status,
                        "models": [], "error": "GET %s answered HTTP %s" % (url, res.status)}
            body = json.loads(res.read().decode("utf-8"))
        return {"provider": provider, "ok": True, "status": 200,
                "models": _models_in(provider["wire"], body), "error": None}
    except Exception as exc:
        return {"provider": provider, "ok": False, "status": 0,
                "models": [], "error": "GET %s failed: %s" % (url, exc)}


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] is True


def capabilities_of(model, row):
    def pick(key):
        if row is not None:
            return row[key]
        return model[key]
    return {
        "context_length": row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"],
        "tools": pick("tools"),
        "vision": pick("vision"),
        "json": pick("json"),
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    if requires.get("min_context") is not None:
        ctx = caps.get("context_length")
        if ctx is None or ctx < requires["min_context"]:
            return False
    return True


def build_catalog(probes, brief):
    """The catalog the probes and the quality table make: the --list document."""
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(brief, probe["provider"]["id"], model["id"])
            ctx = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({"id": model["id"], "free": model_free(model, row),
                           "context_length": ctx})
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def served_models(probes, brief):
    """Free models, deduplicated by id, in brief provider order then id order."""
    out = []
    seen = set()
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of_model(brief, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            out.append({"id": model["id"], "provider": probe["provider"]["id"],
                        "display_name": model["display_name"]})
    return out


# ---------------------------------------------------------------- wire translation
# The provider echoes what it parsed, and the client compares that echo with
# what it sent, so every translation below is field-preserving: system text,
# user text, image bytes, tool names in order, max_tokens, stop sequences and
# JSON mode all land in the upstream wire's own spelling of the same field.


def text_of(content):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content
                     if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))


def _args_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except Exception:
            pass
    return {}


def _split_data_url(url):
    """(mime, base64) of a data: URL, else (None, None)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    head, _, data = url[5:].partition(",")
    mime = head.split(";")[0] or "image/png"
    return mime, data


def openai_to_anthropic(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": _str(block.get("id")),
                    "type": "function",
                    "function": {
                        "name": _str(block.get("name")),
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = text_of(content)
            msg = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    text = inner
                elif isinstance(inner, list):
                    text = text_of(inner) or json.dumps(inner)
                else:
                    text = "" if inner is None else json.dumps(inner)
                messages.append({"role": "tool",
                                 "tool_call_id": _str(block.get("tool_use_id")),
                                 "content": text})
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = _str(source.get("media_type")) or "image/png"
                    parts.append({"type": "image_url",
                                  "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({"type": "function", "function": {
                "name": raw["name"],
                "description": _str(raw.get("description")),
                "parameters": raw.get("input_schema") if is_record(raw.get("input_schema"))
                else {"type": "object", "properties": {}},
            }})
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def anthropic_to_openai(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    system_parts = []
    messages = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            text = text_of(raw.get("content"))
            if text != "":
                system_parts.append(text)
            continue
        content = raw.get("content")
        if role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
                continue
            blocks = []
            if isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        blocks.append({"type": "text", "text": block["text"]})
            # tool calls live on the message, not in content, on this wire
            tool_calls = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []
            for call in tool_calls:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({"type": "tool_use",
                               "id": _str(call.get("id")) or "toolu_unknown",
                               "name": _str(fn.get("name")),
                               "input": _args_of(fn.get("arguments"))})
            messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            text = content if isinstance(content, str) else text_of(content)
            messages.append({"role": "user", "content": [{"type": "tool_result",
                                                          "tool_use_id": _str(raw.get("tool_call_id")),
                                                          "content": text}]})
            continue
        # user
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        blocks = []
        if isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    url = block["image_url"].get("url")
                    mime, data = _split_data_url(url)
                    if data is not None:
                        blocks.append({"type": "image", "source": {"type": "base64",
                                                                   "media_type": mime, "data": data}})
                    elif isinstance(url, str):
                        blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    elif isinstance(body.get("max_completion_tokens"), (int, float)):
        out["max_tokens"] = body["max_completion_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({"name": fn["name"],
                          "description": _str(fn.get("description")),
                          "input_schema": fn.get("parameters") if is_record(fn.get("parameters"))
                          else {"type": "object", "properties": {}}})
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        out["tool_choice"] = {"type": "tool", "name": _str(choice["function"].get("name"))}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _gemini_contents_from_openai(body):
    """(system_text, contents, tools) of an OpenAI body in Gemini's shape."""
    system_parts = []
    contents = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            text = text_of(raw.get("content"))
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            parts = []
            content = raw.get("content")
            if isinstance(content, str) and content != "":
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                parts.append({"functionCall": {"name": _str(fn.get("name")),
                                              "args": _args_of(fn.get("arguments"))}})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            text = raw.get("content") if isinstance(raw.get("content"), str) else text_of(raw.get("content"))
            part = {"functionResponse": {"name": "", "response": {"result": text}}}
            if contents and contents[-1]["role"] == "user":
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
            continue
        # user
        parts = []
        content = raw.get("content")
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    url = block["image_url"].get("url")
                    mime, data = _split_data_url(url)
                    if data is not None:
                        parts.append({"inlineData": {"mimeType": mime, "data": data}})
                    elif isinstance(url, str):
                        parts.append({"fileData": {"fileUri": url}})
        if parts:
            contents.append({"role": "user", "parts": parts})
    tools = []
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            decls.append({"name": fn["name"],
                          "description": _str(fn.get("description")),
                          "parameters": fn.get("parameters") if is_record(fn.get("parameters"))
                          else {"type": "object", "properties": {}}})
        if decls:
            tools.append({"functionDeclarations": decls})
    return "\n".join(system_parts), contents, tools


def _gemini_contents_from_anthropic(body):
    """(system_text, contents, tools) of an Anthropic body in Gemini's shape."""
    system_text = text_of(body.get("system"))
    contents = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if raw.get("role") == "assistant":
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "tool_use":
                        parts.append({"functionCall": {"name": _str(block.get("name")),
                                                      "args": block.get("input") if is_record(block.get("input")) else {}}})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        # user (tool_result blocks ride along as functionResponse parts)
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image" and is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        parts.append({"inlineData": {"mimeType": _str(source.get("media_type")) or "image/png",
                                                     "data": source["data"]}})
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        parts.append({"fileData": {"fileUri": source["url"]}})
                elif block.get("type") == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, str):
                        text = inner
                    elif isinstance(inner, list):
                        text = text_of(inner) or json.dumps(inner)
                    else:
                        text = "" if inner is None else json.dumps(inner)
                    parts.append({"functionResponse": {"name": _str(block.get("tool_use_id")),
                                                      "response": {"result": text}}})
        if parts:
            contents.append({"role": "user", "parts": parts})
    tools = []
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            decls.append({"name": raw["name"],
                          "description": _str(raw.get("description")),
                          "input_schema": raw.get("input_schema") if is_record(raw.get("input_schema"))
                          else {"type": "object", "properties": {}}})
        if decls:
            tools.append({"functionDeclarations": decls})
    return system_text, contents, tools


def to_gemini(body, client_api):
    """A client body (openai or anthropic) as a Gemini generateContent body."""
    if client_api == "openai":
        system_text, contents, tools = _gemini_contents_from_openai(body)
        max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_completion_tokens")
        stops = body.get("stop")
        fmt = body.get("response_format") if is_record(body.get("response_format")) else None
        json_mode = fmt is not None and fmt.get("type") in ("json_object", "json_schema")
    else:
        system_text, contents, tools = _gemini_contents_from_anthropic(body)
        max_tokens = body.get("max_tokens")
        stops = body.get("stop_sequences")
        json_mode = False
    out = {"contents": contents}
    if system_text != "":
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    if tools:
        out["tools"] = tools
    config = {}
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    if isinstance(stops, str):
        config["stopSequences"] = [stops]
    elif isinstance(stops, list) and stops:
        config["stopSequences"] = [s for s in stops if isinstance(s, str)]
    if json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


# ---------------------------------------------------------------- SSE and answers
# Every provider answers plain text (the simulated answer is an echo of the
# parsed request), so an answer is always (text, usage, stop family). Parsing
# it into that triple and rendering it in the client's wire keeps the text
# whole, the usage exact and the stop reason in the right family.


def parse_sse_frames(text):
    """[(event, data)] frames of a server-sent-events body."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _num_or_zero(v):
    return v if isinstance(v, (int, float)) else 0


def parse_openai_answer(status, body_text, stream):
    """(text, usage, stop) of an OpenAI-wire answer, or None when unusable."""
    if status != 200:
        return None
    usage = None
    if not stream:
        try:
            value = json.loads(body_text)
        except Exception:
            return None
        if not is_record(value):
            return None
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices or not is_record(choices[0]):
            return None
        choice = choices[0]
        message = choice.get("message") if is_record(choice.get("message")) else {}
        content = message.get("content")
        text = content if isinstance(content, str) else text_of(content)
        if is_record(value.get("usage")):
            usage = {"input": _num_or_zero(value["usage"].get("prompt_tokens")),
                     "output": _num_or_zero(value["usage"].get("completion_tokens"))}
        return text, usage, choice.get("finish_reason")
    text = ""
    stop = "stop"
    for _, data in parse_sse_frames(body_text):
        payload = data.strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            usage = {"input": _num_or_zero(chunk["usage"].get("prompt_tokens")),
                     "output": _num_or_zero(chunk["usage"].get("completion_tokens"))}
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices or not is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text += delta["content"]
        if isinstance(choice.get("finish_reason"), str):
            stop = choice["finish_reason"]
    return text, usage, stop


def parse_anthropic_answer(status, body_text, stream):
    if status != 200:
        return None
    if not stream:
        try:
            value = json.loads(body_text)
        except Exception:
            return None
        if not is_record(value) or value.get("type") != "message":
            return None
        text = text_of(value.get("content"))
        usage = None
        if is_record(value.get("usage")):
            usage = {"input": _num_or_zero(value["usage"].get("input_tokens")),
                     "output": _num_or_zero(value["usage"].get("output_tokens"))}
        return text, usage, value.get("stop_reason")
    text = ""
    input_tokens = None
    output_tokens = None
    stop = None
    for event, data in parse_sse_frames(body_text):
        try:
            value = json.loads(data)
        except Exception:
            continue
        if not is_record(value):
            continue
        if event == "message_start" and is_record(value.get("message")) \
                and is_record(value["message"].get("usage")):
            input_tokens = _num_or_zero(value["message"]["usage"].get("input_tokens"))
        elif event == "content_block_delta" and is_record(value.get("delta")) \
                and value["delta"].get("type") == "text_delta" \
                and isinstance(value["delta"].get("text"), str):
            text += value["delta"]["text"]
        elif event == "message_delta":
            if is_record(value.get("delta")) and isinstance(value["delta"].get("stop_reason"), str):
                stop = value["delta"]["stop_reason"]
            if is_record(value.get("usage")):
                output_tokens = _num_or_zero(value["usage"].get("output_tokens"))
    usage = None
    if input_tokens is not None or output_tokens is not None:
        usage = {"input": input_tokens or 0, "output": output_tokens or 0}
    return text, usage, stop or "end_turn"


def parse_gemini_answer(status, body_text, stream):
    if status != 200:
        return None
    if not stream:
        try:
            value = json.loads(body_text)
        except Exception:
            return None
        if not is_record(value):
            return None
        candidates = value.get("candidates")
        text = ""
        if isinstance(candidates, list) and candidates and is_record(candidates[0]):
            content = candidates[0].get("content") if is_record(candidates[0].get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            text = "".join(p.get("text", "") for p in parts if is_record(p) and isinstance(p.get("text"), str))
        usage = None
        meta = value.get("usageMetadata") if is_record(value.get("usageMetadata")) else {}
        if meta:
            usage = {"input": _num_or_zero(meta.get("promptTokenCount")),
                     "output": _num_or_zero(meta.get("candidatesTokenCount"))}
        finish = candidates[0].get("finishReason") if isinstance(candidates, list) and candidates and is_record(candidates[0]) else None
        return text, usage, finish or "STOP"
    text = ""
    usage = None
    stop = None
    for _, data in parse_sse_frames(body_text):
        try:
            value = json.loads(data)
        except Exception:
            continue
        if not is_record(value):
            continue
        candidates = value.get("candidates")
        if isinstance(candidates, list) and candidates and is_record(candidates[0]):
            cand = candidates[0]
            content = cand.get("content") if is_record(cand.get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for p in parts:
                if is_record(p) and isinstance(p.get("text"), str):
                    text += p["text"]
            if isinstance(cand.get("finishReason"), str):
                stop = cand["finishReason"]
        meta = value.get("usageMetadata") if is_record(value.get("usageMetadata")) else {}
        if meta:
            usage = {"input": _num_or_zero(meta.get("promptTokenCount")),
                     "output": _num_or_zero(meta.get("candidatesTokenCount"))}
    return text, usage, stop or "STOP"


def _split_text(text, size=64):
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def render_openai_json(model, text, usage):
    usage = usage or {"input": 0, "output": 0}
    return {"id": "chatcmpl-sim", "object": "chat.completion", "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                      "total_tokens": usage["input"] + usage["output"]}}


def render_openai_sse(model, text, usage):
    usage = usage or {"input": 0, "output": 0}
    frames = []

    def chunk(delta, finish):
        return {"id": "chatcmpl-sim", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames.append("data: %s\n\n" % json.dumps(chunk({"role": "assistant", "content": ""}, None)))
    for piece in _split_text(text):
        frames.append("data: %s\n\n" % json.dumps(chunk({"content": piece}, None)))
    frames.append("data: %s\n\n" % json.dumps(chunk({}, "stop")))
    frames.append("data: %s\n\n" % json.dumps(
        {"id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED,
         "model": model, "choices": [],
         "usage": {"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                   "total_tokens": usage["input"] + usage["output"]}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def render_anthropic_json(model, text, usage):
    usage = usage or {"input": 0, "output": 0}
    return {"id": "msg_sim", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": usage["input"], "output_tokens": usage["output"]}}


def render_anthropic_sse(model, text, usage):
    usage = usage or {"input": 0, "output": 0}
    frames = []

    def event(name, data):
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(data))

    frames.append(event("message_start", {"type": "message_start", "message": {
        "id": "msg_sim", "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": usage["input"], "output_tokens": 0}}}))
    frames.append(event("content_block_start", {"type": "content_block_start", "index": 0,
                                                "content_block": {"type": "text", "text": ""}}))
    for piece in _split_text(text):
        frames.append(event("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                    "delta": {"type": "text_delta", "text": piece}}))
    frames.append(event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(event("message_delta", {"type": "message_delta",
                                          "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                          "usage": {"output_tokens": usage["output"]}}))
    frames.append(event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def render_answer(client_api, stream, model, text, usage):
    """The provider's answer in the wire the client used."""
    if client_api == "openai":
        if stream:
            return "text/event-stream", render_openai_sse(model, text, usage)
        return "application/json", json.dumps(render_openai_json(model, text, usage))
    if stream:
        return "text/event-stream", render_anthropic_sse(model, text, usage)
    return "application/json", json.dumps(render_anthropic_json(model, text, usage))


# ---------------------------------------------------------------- upstream calls


def _upstream_url_and_body(provider, client_api, body, model, stream):
    """(path, json_body, extra_headers) of the provider call in its own wire."""
    base = join_base(ENV_URL, provider["base"])
    wire = provider["wire"]
    if wire == "openai":
        if client_api == "openai":
            upstream = dict(body)
            upstream["model"] = model
            if stream:
                opts = upstream.get("stream_options")
                if not is_record(opts):
                    opts = {}
                opts = dict(opts)
                opts["include_usage"] = True
                upstream["stream_options"] = opts
            return base + "/v1/chat/completions", upstream, {}
        return base + "/v1/chat/completions", openai_to_anthropic(body, model), {}
    if wire == "anthropic":
        if client_api == "anthropic":
            upstream = dict(body)
            upstream["model"] = model
            return base + "/v1/messages", upstream, {"anthropic-version": ANTHROPIC_VERSION}
        return base + "/v1/messages", anthropic_to_openai(body, model), {"anthropic-version": ANTHROPIC_VERSION}
    # gemini
    encoded = urllib.parse.quote(model, safe="-._")
    if stream:
        return base + "/v1beta/models/%s:streamGenerateContent?alt=sse" % encoded, to_gemini(body, client_api), {}
    return base + "/v1beta/models/%s:generateContent" % encoded, to_gemini(body, client_api), {}


def call_upstream(provider, client_api, body, model, stream):
    """(status, body_text) of one provider call, or (0, '') on a transport error."""
    try:
        url, upstream_body, extra = _upstream_url_and_body(provider, client_api, body, model, stream)
    except Exception as exc:
        log("cannot build the %s request: %s" % (provider["id"], exc))
        return 0, ""
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path
    if parsed.query:
        path += "?" + parsed.query
    headers = {"content-type": "application/json", "accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    headers.update(extra)
    if stream:
        headers["accept"] = "text/event-stream"
    payload = json.dumps(upstream_body).encode("utf-8")
    conn = HTTPConnection(host, port, timeout=UPSTREAM_TIMEOUT_S)
    try:
        conn.request("POST", path, body=payload, headers=headers)
        res = conn.getresponse()
        data = res.read()
        return res.status, data.decode("utf-8", errors="replace")
    except Exception as exc:
        log("the %s call failed: %s" % (provider["id"], exc))
        return 0, ""
    finally:
        conn.close()


def parse_upstream_answer(provider, status, body_text, stream):
    wire = provider["wire"]
    if wire == "openai":
        return parse_openai_answer(status, body_text, stream)
    if wire == "anthropic":
        return parse_anthropic_answer(status, body_text, stream)
    return parse_gemini_answer(status, body_text, stream)


# ---------------------------------------------------------------- routing

class Router:
    """Candidates by rating, with failover and suspect-provider deprioritisation."""

    def __init__(self, brief, probes):
        self.brief = brief
        self.probes = probes
        self.calls = {}
        self.consecutive_failures = {}
        self.suspect = set()

    def candidates(self, requires, client_api, body):
        out = []
        json_mode = False
        if client_api == "openai":
            fmt = body.get("response_format") if is_record(body.get("response_format")) else None
            json_mode = fmt is not None and fmt.get("type") in ("json_object", "json_schema")
        for probe in self.probes:
            if not probe["ok"]:
                continue
            provider = probe["provider"]
            # A JSON-mode OpenAI request must go to a wire that can carry it:
            # the Anthropic wire echoes json_mode absent and loses the check.
            if json_mode and provider["wire"] == "anthropic":
                continue
            for model in probe["models"]:
                row = rating_of_model(self.brief, provider["id"], model["id"])
                if not model_free(model, row):
                    continue
                if not meets_requirements(requires, capabilities_of(model, row)):
                    continue
                rating = row["rating_milli"] if row is not None else 0
                out.append((provider, model, rating))
        out.sort(key=lambda c: (1 if c[0]["id"] in self.suspect else 0,
                                -c[2], c[0]["id"], c[1]["id"]))
        return out

    def serve(self, client_api, body, requires, stream):
        """(content_type, payload, ok): the answer, or a 503-style error."""
        tried = []
        for provider, model, _rating in self.candidates(requires, client_api, body):
            if provider["id"] in tried:
                continue
            tried.append(provider["id"])
            status, text = call_upstream(provider, client_api, body, model["id"], stream)
            self.calls[provider["id"]] = self.calls.get(provider["id"], 0) + 1
            if status == 200:
                parsed = parse_upstream_answer(provider, status, text, stream)
                if parsed is None:
                    self._note_failure(provider["id"])
                    continue
                answer_text, usage, _stop = parsed
                self.consecutive_failures[provider["id"]] = 0
                content_type, payload = render_answer(client_api, stream, model["id"], answer_text, usage)
                return content_type, payload, True
            if status in (402, 429) or status >= 500 or status == 0:
                self._note_failure(provider["id"])
                continue
            # Any other refusal (400/401) is the request's own fault: fail over
            # too, but it spends no budget and says nothing about the provider.
            continue
        return "application/json", None, False

    def _note_failure(self, provider_id):
        n = self.consecutive_failures.get(provider_id, 0) + 1
        self.consecutive_failures[provider_id] = n
        if n >= 2:
            self.suspect.add(provider_id)


# ---------------------------------------------------------------- server

BRIEF = read_brief(PROMPT_FILE)
PROBES = []
CATALOG = {"providers": []}
ROUTER = None
MANIFEST_INDEX = 0
ACTIVE = 0
ACTIVE_LOCK = threading.Lock()


def requirements_for(body):
    """The manifest row by arrival order, else what the body itself says."""
    global MANIFEST_INDEX
    manifest = BRIEF.get("manifest") or []
    entry = manifest[MANIFEST_INDEX] if MANIFEST_INDEX < len(manifest) else None
    MANIFEST_INDEX += 1
    if entry is not None:
        return entry["requires"]
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    return {"tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
            "vision": vision,
            "json": fmt is not None and fmt.get("type") in ("json_object", "json_schema")}


def _send_json(handler, status, body):
    text = json.dumps(body)
    raw = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _send_error(handler, wire, status, error_type, message):
    if wire == "anthropic":
        _send_json(handler, status, {"type": "error", "error": {"type": error_type, "message": message}})
    else:
        _send_json(handler, status, {"error": {"message": message, "type": error_type, "code": None}})


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _read_json(self):
        global ACTIVE
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else 0
        except ValueError:
            size = 0
        if size <= 0 or size > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(size)
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def do_GET(self):
        if urllib.parse.urlparse(self.path).path == "/v1/models":
            models = served_models(PROBES, BRIEF)
            if isinstance(self.headers.get("x-api-key"), str):
                _send_json(self, 200, {
                    "data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                              "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
                    "has_more": False})
            else:
                _send_json(self, 200, {
                    "object": "list",
                    "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                              "owned_by": m["provider"]} for m in models]})
            return
        _send_error(self, "openai", 404, "not_found", "no route GET %s" % self.path)

    def do_POST(self):
        global ACTIVE
        path = urllib.parse.urlparse(self.path).path
        if path not in ("/v1/chat/completions", "/v1/messages"):
            _send_error(self, "anthropic" if path == "/v1/messages" else "openai",
                        404, "not_found", "no route POST %s" % path)
            return
        client_api = "anthropic" if path == "/v1/messages" else "openai"
        body = self._read_json()
        if body is None:
            _send_error(self, client_api, 400, "invalid_request_error",
                        "the request body is not one JSON object")
            return
        with ACTIVE_LOCK:
            ACTIVE += 1
        try:
            stream = body.get("stream") is True
            content_type, payload, ok = ROUTER.serve(client_api, body, requirements_for(body), stream)
            if not ok:
                _send_error(self, client_api, 503,
                            "overloaded_error" if client_api == "anthropic" else "no_free_model",
                            "no free model on the directory meets this request")
                return
            raw = payload.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(raw)))
            if stream:
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "keep-alive")
            self.end_headers()
            self.wfile.write(raw)
        finally:
            with ACTIVE_LOCK:
                ACTIVE -= 1


def load():
    global PROBES, CATALOG, ROUTER
    if BRIEF.get("error"):
        log(BRIEF["error"])
    PROBES = [probe_provider(ENV_URL, p) for p in BRIEF["providers"]]
    for probe in PROBES:
        if not probe["ok"] and probe.get("error"):
            log(probe["error"])
    CATALOG = build_catalog(PROBES, BRIEF)
    ROUTER = Router(BRIEF, PROBES)


def register(port):
    req = urllib.request.Request(
        ENV_URL + "/register",
        data=json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port,
                         "catalog": CATALOG}).encode("utf-8"),
        headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as res:
        if res.status != 200:
            raise RuntimeError("POST %s/register answered HTTP %s" % (ENV_URL, res.status))
        reply = json.loads(res.read().decode("utf-8"))
    if is_record(reply) and isinstance(reply.get("requests"), (int, float)):
        log("registered: the workload holds %s requests" % reply["requests"])


def poll_work(server):
    """Poll GET /work; exit 0 the moment it says done (and nothing is in flight)."""
    while True:
        time.sleep(POLL_INTERVAL_S)
        try:
            req = urllib.request.Request(ENV_URL + "/work", method="GET")
            with urllib.request.urlopen(req, timeout=2) as res:
                if res.status != 200:
                    continue
                work = json.loads(res.read().decode("utf-8"))
        except Exception:
            continue
        if is_record(work) and work.get("done") is True:
            with ACTIVE_LOCK:
                busy = ACTIVE
            if busy == 0:
                server.shutdown()
                sys.exit(0)


def start():
    load()
    if "--list" in sys.argv:
        text = json.dumps(CATALOG) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        sys.exit(0)
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout")
    log("listening on 0.0.0.0:%s with %s answered providers and %s free models"
        % (port, len(CATALOG["providers"]), len(served_models(PROBES, BRIEF))))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    register(port)
    poll_work(server)


if __name__ == "__main__":
    try:
        start()
    except SystemExit:
        raise
    except Exception as exc:
        import traceback

        traceback.print_exc()
        sys.exit(1)
