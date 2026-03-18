"""
llm.py — Ollama LLM wrapper with streaming and timeout handling

KEY FIXES over original:
  1. Uses stream=True — tokens come back one by one instead of
     waiting for the ENTIRE response. Prevents silent hangs.
  2. Hard timeout of 90s per call (not 120s)
  3. Limits response length with num_predict=600
  4. Better error messages (shows which model, what timed out)
  5. generate_json has a hard fallback so it NEVER raises — always
     returns a dict (possibly with an "error" key)
  6. Ollama's native format="json" mode for reliable JSON output
"""

import json
import re
import requests


class LocalLLM:
    """Wraps Ollama's REST API with streaming + timeout protection."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 90,
        max_tokens: int = 600,
    ):
        self.model      = model
        self.base_url   = base_url.rstrip("/")
        self.timeout    = timeout
        self.max_tokens = max_tokens

    def _post(self, messages: list[dict], is_json: bool = False) -> str:
        """
        POST to Ollama /api/chat with streaming.

        Streaming means we get tokens back one at a time instead of
        waiting for the full response — prevents the silent hang where
        Ollama is generating but we see nothing for minutes.
        """
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   True,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.3,
            },
        }
        if is_json:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            resp.raise_for_status()

            full_text = []
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    full_text.append(token)
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

            return "".join(full_text).strip()

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s for model {self.model}. "
                "Try: ollama pull llama3.2:1b"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate plain text. Returns a fallback string on failure."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            return self._post(messages)
        except Exception as e:
            print(f"[LLM] generate() failed: {e}")
            return "I encountered an issue processing that request."

    def generate_json(self, prompt: str, system: str = "") -> dict:
        """
        Generate a JSON response. NEVER raises — always returns a dict.
        Uses Ollama's native JSON mode for reliability, falls back to
        text parsing if needed.
        """
        json_instruction = (
            "Return ONLY a valid JSON object. "
            "No explanation, no markdown, no code fences. Raw JSON only."
        )
        sys_msg = f"{system}\n\n{json_instruction}" if system else json_instruction
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": prompt},
        ]

        # Try with Ollama's native JSON mode first
        try:
            raw = self._post(messages, is_json=True)
            if raw:
                result = self._parse_json(raw)
                if result and "error" not in result:
                    return result
        except Exception:
            pass

        # Try without JSON mode (plain text + parse)
        try:
            raw = self._post(messages, is_json=False)
            if raw:
                return self._parse_json(raw)
        except Exception as e:
            print(f"[LLM] generate_json() failed: {e}")

        return {"status": "fallback", "error": "llm_failed"}

    def _parse_json(self, text: str) -> dict:
        """Four strategies to extract JSON from LLM output."""
        if not text:
            return {}

        # Strategy 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip markdown fences
        cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 3: find first complete { } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 4: find last non-nested { } block
        matches = list(re.finditer(r"\{[^{}]*\}", text))
        for m in reversed(matches):
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue

        return {}
