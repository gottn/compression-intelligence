"""Download all model snapshots (weights + tokenizer + config) into the HF cache, one at a time.

Writes one JSON line per model to logs/download.jsonl so failures (gated, missing) are recorded.
"""
import json
import sys
import time

from huggingface_hub import snapshot_download

MODELS = [
    "openai-community/gpt2",
    "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*"]


def download(repo_id: str) -> dict:
    t0 = time.time()
    try:
        path = snapshot_download(repo_id, allow_patterns=PATTERNS)
        return {"model": repo_id, "ok": True, "path": path, "s": round(time.time() - t0, 1)}
    except Exception as e:  # record and continue
        return {"model": repo_id, "ok": False, "error": f"{type(e).__name__}: {e}"[:500], "s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    models = sys.argv[1:] or MODELS
    with open("logs/download.jsonl", "a") as f:
        for m in models:
            rec = download(m)
            print(json.dumps(rec), flush=True)
            f.write(json.dumps(rec) + "\n")
            f.flush()
