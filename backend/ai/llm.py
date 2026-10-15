"""LLM access through OpenAI-compatible chat endpoints.

Gemini and Groq both expose /chat/completions with tool calling, so one client
covers both. Providers are tried in order; if every provider fails the caller
falls back to a deterministic answer built from the same tool results.
"""
import json
import time

import httpx

import config as C

PROVIDERS = [
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", lambda: C.GEMINI_API_KEY, lambda: C.GEMINI_MODEL),
    ("openai", "https://api.openai.com/v1/chat/completions", lambda: C.OPENAI_API_KEY, lambda: C.OPENAI_MODEL),
    ("groq", "https://api.groq.com/openai/v1/chat/completions", lambda: C.GROQ_API_KEY, lambda: C.GROQ_MODEL),
]


class LLMUnavailable(Exception):
    pass


def chat(messages, tools=None, temperature=0.3, max_tokens=4000, only=None):
    """Return (provider, assistant_message_dict). `only` pins one provider mid-conversation."""
    errors = []
    for name, url, key, model in PROVIDERS:
        if only and name != only:
            continue
        if not key():
            errors.append(f"{name}: no key")
            continue
        if name == "openai":  # reasoning models take max_completion_tokens and only the default temperature
            body = {"model": model(), "messages": messages, "max_completion_tokens": max_tokens}
        else:
            body = {"model": model(), "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            for attempt in range(2):  # one quick retry when the provider is overloaded
                r = httpx.post(url, json=body, headers={"Authorization": f"Bearer {key()}"}, timeout=40)
                if r.status_code not in (429, 503) or attempt or "insufficient_quota" in r.text:
                    break
                time.sleep(1.5)
            r.raise_for_status()
            return name, r.json()["choices"][0]["message"]
        except Exception as e:  # try the next provider
            detail = getattr(getattr(e, "response", None), "text", "")[:200]
            errors.append(f"{name}: {e} {detail}")
    raise LLMUnavailable("; ".join(errors))


def run_tools(messages, tools, handlers, max_rounds=5, provider=None):
    """Tool-calling loop. Returns (provider, final_text, tool_log)."""
    log = []
    for _ in range(max_rounds):
        provider, msg = chat(messages, tools, only=provider)
        calls = msg.get("tool_calls") or []
        if not calls:
            return provider, msg.get("content") or "", log
        # Send the assistant turn back unchanged: Gemini 3 requires its thought signatures.
        echo = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls", "extra_content")}
        echo["role"], echo["content"] = "assistant", msg.get("content") or ""
        messages.append(echo)
        for call in calls:
            fn = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = handlers[fn](**args)
            except Exception as e:
                result = {"error": str(e)}
            log.append({"tool": fn, "args": args})
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": fn,
                             "content": json.dumps(result, default=str)[:12000]})
    provider, msg = chat(messages, only=provider)
    return provider, msg.get("content") or "", log
