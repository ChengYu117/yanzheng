import os
import json
import urllib.request
import urllib.error

token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
base_url = os.environ.get("ANTHROPIC_BASE_URL")

print(f"Token: {token[:15] if token else 'None'}...")
print(f"Base URL: {base_url}")

# Let's test making a direct OpenAI-compatible request to the base URL if compatible,
# or Anthropic-compatible request.
# Let's check if the base_url has /v1/messages or similar.
# Since it is ANTHROPIC_BASE_URL, it usually is the Anthropic API endpoint.
# Anthropic endpoint uses headers:
# x-api-key: token
# anthropic-version: 2023-06-01
# content-type: application/json

headers = {
    "x-api-key": token,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
    "user-agent": "claude-code/0.2.9"
}

# Try anthropic messages endpoint
url = f"{base_url.rstrip('/')}/v1/messages"
payload = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 100,
    "messages": [
        {"role": "user", "content": "Hello, are you there?"}
    ]
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers=headers, method="POST")

try:
    with urllib.request.urlopen(req) as resp:
        res_body = resp.read().decode("utf-8")
        print("Success:")
        print(res_body[:500])
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.reason}")
    print(e.read().decode("utf-8")[:500])
except Exception as e:
    print(f"Error: {e}")
