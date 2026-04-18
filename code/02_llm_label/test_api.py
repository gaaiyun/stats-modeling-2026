"""测试硅基流动 API 连通性，并自动选出可用模型"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
from configs.config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL

CANDIDATE_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Pro/Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-V2.5",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]


def test_chat(model: str, prompt: str = "请用一句话介绍'新质生产力'。") -> tuple[bool, str]:
    url = f"{SILICONFLOW_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.2,
        "stream": False,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return True, content.strip()
        else:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"EXC: {e}"


def main():
    print(f"API Key: {SILICONFLOW_API_KEY[:12]}…")
    print(f"Base URL: {SILICONFLOW_BASE_URL}\n")

    available = []
    for m in CANDIDATE_MODELS:
        ok, msg = test_chat(m)
        flag = "OK " if ok else "FAIL"
        print(f"[{flag}] {m:50s} -> {msg[:120]}")
        if ok:
            available.append(m)
        time.sleep(0.5)

    print("\n=== 可用模型 ===")
    for m in available:
        print(" -", m)

    out = Path(__file__).resolve().parents[2] / "logs" / "available_models.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps({"available": available, "tested_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
