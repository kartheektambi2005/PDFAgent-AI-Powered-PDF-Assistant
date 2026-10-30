"""Thin Gemini REST client (embeddings + JSON generation). The API key is sent as a header, never in a URL."""
import json
import time

import numpy as np
import requests

from . import config

BASE = "https://generativelanguage.googleapis.com/v1beta"
EMBED_BATCH = 100


class GeminiError(Exception):
    pass


class Gemini:
    def __init__(self, api_key: str = config.GEMINI_API_KEY, embed_model: str = config.EMBED_MODEL,
                 gen_model: str = config.GEN_MODEL, dim: int = config.EMBED_DIM, thinking_budget=None):
        if not api_key:
            raise GeminiError("GEMINI_API_KEY is not set. Put it in your environment or a .env file.")
        self.key, self.embed_model, self.gen_model, self.dim = api_key, embed_model, gen_model, dim
        self.thinking_budget = thinking_budget  # None = model default; 0 disables thinking where supported
        self.calls = {"embed": 0, "generate": 0}  # lets tests/scenarios count real API calls

    def _post(self, path: str, body: dict) -> dict:
        for attempt in range(3):
            try:
                r = requests.post(f"{BASE}/{path}", json=body, timeout=60,
                                  headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})
            except requests.RequestException as e:
                raise GeminiError(f"Network error talking to Gemini ({type(e).__name__}).") from e
            if r.status_code in (429, 500, 503) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                try:
                    msg = r.json()["error"]["message"]
                except Exception:
                    msg = r.text[:200]
                raise GeminiError(f"Gemini API error {r.status_code}: {msg}")
            return r.json()
        raise GeminiError("Gemini API unavailable after retries.")

    def list_models(self) -> list[dict]:
        r = requests.get(f"{BASE}/models?pageSize=200", headers={"x-goog-api-key": self.key}, timeout=30)
        if r.status_code != 200:
            raise GeminiError(f"ListModels failed: {r.status_code}")
        return r.json().get("models", [])

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        """task: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY. Batched; returns (n, dim) float32."""
        out = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            body = {"requests": [{
                "model": f"models/{self.embed_model}",
                "content": {"parts": [{"text": t}]},
                "taskType": task,
                "outputDimensionality": self.dim,
            } for t in batch]}
            data = self._post(f"models/{self.embed_model}:batchEmbedContents", body)
            self.calls["embed"] += 1
            out.extend(e["values"] for e in data["embeddings"])
        return np.asarray(out, dtype=np.float32)

    def generate(self, system: str, prompt: str, schema: dict, max_tokens: int) -> tuple[dict, dict]:
        """One JSON-constrained generation. Returns (parsed_json, {'input': n, 'output': n})."""
        cfg = {"temperature": 0, "maxOutputTokens": max_tokens,
               "responseMimeType": "application/json", "responseSchema": schema}
        if self.thinking_budget is not None:
            cfg["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": cfg}
        data = self._post(f"models/{self.gen_model}:generateContent", body)
        self.calls["generate"] += 1
        u = data.get("usageMetadata", {})
        # candidatesTokenCount excludes "thinking" tokens, which are billed as output.
        usage = {"input": u.get("promptTokenCount", 0),
                 "output": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)}
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text), usage
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            reason = (data.get("candidates") or [{}])[0].get("finishReason", "unknown")
            raise GeminiError(f"Model returned no usable answer (finishReason={reason}).") from e
