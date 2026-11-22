# -*- coding: utf-8 -*-
"""Shared utilities: corpus loading, OpenRouter client with per-call JSONL logging.
The API key is read from a local file and is never logged or printed."""
import json, os, re, time, hashlib, threading, urllib.request, urllib.error, random, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
LOGDIR = os.path.join(ROOT, "logs")
os.makedirs(LOGDIR, exist_ok=True)
KEY_FILE = os.environ.get("OPENROUTER_KEY_FILE", r"C:\Users\leeji\Desktop\openrouter_apikey.txt")
_lock = threading.Lock()


def _key():
    t = open(KEY_FILE, encoding="utf-8-sig").read()
    m = re.findall(r"sk-or-v1-[A-Za-z0-9]+", t)
    if not m:
        raise RuntimeError("OpenRouter key not found")
    return m[0]


def load_corpus(name):
    rows = json.load(open(os.path.join(DATA, f"{name}.json"), encoding="utf-8"))
    return rows


def doc_text(r, max_words=120):
    w = r["abstract"].split()
    ab = " ".join(w[:max_words]) + (" ..." if len(w) > max_words else "")
    return f"{r['title']}. {ab}"


def short_text(r, max_words=50):
    w = r["abstract"].split()
    return f"{r['title']}. {' '.join(w[:max_words])}"


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def chat(messages, model, temperature=0.0, max_tokens=1500, seed=None, log_file="api_calls.jsonl", meta=None, retries=4):
    """One chat completion via OpenRouter. Every call (including failures) is appended to logs/<log_file> as one JSON line."""
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "usage": {"include": True}}
    if seed is not None:
        body["seed"] = seed
    data = json.dumps(body).encode()
    last_err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=data,
                                         headers={"Authorization": "Bearer " + _key(), "Content-Type": "application/json",
                                                  "HTTP-Referer": "https://github.com/leejinkyu0612-hue", "X-Title": "topic-stability-study"})
            resp = json.load(urllib.request.urlopen(req, timeout=180))
            lat = time.time() - t0
            if "choices" not in resp:
                raise RuntimeError(str(resp)[:300])
            content = resp["choices"][0]["message"].get("content") or ""
            usage = resp.get("usage", {}) or {}
            rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "model": model, "provider": resp.get("provider"),
                   "returned_model": resp.get("model"), "temperature": temperature, "seed": seed, "attempt": attempt,
                   "latency_s": round(lat, 3), "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                   "cost_usd": usage.get("cost"), "prompt_sha": sha(json.dumps(messages, ensure_ascii=False)),
                   "finish_reason": resp["choices"][0].get("finish_reason"), "ok": True, "response": content}
            if meta:
                rec.update(meta)
            with _lock:
                with open(os.path.join(LOGDIR, log_file), "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return content, rec
        except Exception as e:
            last_err = e
            msg = str(e)
            if isinstance(e, urllib.error.HTTPError):
                try:
                    msg = f"{e.code} {e.read()[:300]}"
                except Exception:
                    pass
            rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "model": model, "temperature": temperature,
                   "attempt": attempt, "ok": False, "error": msg[:400]}
            if meta:
                rec.update(meta)
            with _lock:
                with open(os.path.join(LOGDIR, log_file), "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"chat failed: {last_err}")


def extract_json(text):
    """Parse the first JSON object/array in a model response."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i = t.find(opener)
        j = t.rfind(closer)
        if i >= 0 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    return None
