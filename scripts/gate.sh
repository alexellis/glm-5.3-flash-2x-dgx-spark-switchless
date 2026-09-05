#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:?usage: gate.sh http://HEAD:8000}"
MODEL=glm-5.3-flash

curl -fsS --max-time 15 "$BASE_URL/health" >/dev/null

python3 - "$BASE_URL" "$MODEL" <<'PY'
import base64
import json
import sys
import time
import urllib.request

base, model = sys.argv[1:]


def request(payload, timeout=900):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    return result, time.monotonic() - started


def deterministic(messages, **extra):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    payload.update(extra)
    return request(payload)


result, elapsed = deterministic(
    [{"role": "user", "content": "Reply with the single word: healthy"}]
)
content = result["choices"][0]["message"].get("content") or ""
assert "healthy" in content.lower(), content
print(f"PASS text ({elapsed:.2f}s)")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
result, elapsed = deterministic(
    [{"role": "user", "content": "Use the tool to get Bristol weather."}],
    tools=tools,
    tool_choice="auto",
    max_tokens=256,
)
calls = result["choices"][0]["message"].get("tool_calls") or []
assert calls and calls[0]["function"]["name"] == "get_weather", result
print(f"PASS tool call ({elapsed:.2f}s)")

needle = "INDIGO-OTTER-4417"
paragraph = "Routine status entry; all subsystems are nominal. " * 16
parts = [paragraph] * 500
parts[len(parts) // 2] += " The passphrase is " + needle + ". "
prompt = "".join(parts) + " Return only the passphrase."
result, elapsed = deterministic(
    [{"role": "user", "content": prompt}], max_tokens=64
)
content = result["choices"][0]["message"].get("content") or ""
assert needle in content, content
usage = result.get("usage", {})
print(f"PASS long context ({usage.get('prompt_tokens')} tokens, {elapsed:.2f}s)")

# Generate a dependency-free 32x32 red PNG; no private test image is shipped.
import struct
import zlib
raw = b"".join(b"\x00" + bytes([255, 0, 0]) * 32 for _ in range(32))
def chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(
        ">I", zlib.crc32(kind + body) & 0xffffffff
    )
png = b"\x89PNG\r\n\x1a\n"
png += chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
data_url = "data:image/png;base64," + base64.b64encode(png).decode()
message = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": "What is the dominant colour? Reply with one word."},
    ],
}]
result, elapsed = deterministic(message, max_tokens=32)
content = (result["choices"][0]["message"].get("content") or "").lower()
assert "red" in content, content
print(f"PASS image ({elapsed:.2f}s)")
PY

echo "GATE GREEN"
