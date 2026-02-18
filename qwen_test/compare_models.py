#!/usr/bin/env python3
"""
Compare job evaluation quality across local Ollama models and Anthropic Claude.
Uses the same request payload, swaps the model, prints a side-by-side summary.
"""

import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path

import anthropic
import requests

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_NATIVE_URL = "http://localhost:11434/api/chat"
MODELS = ["qwen2.5:32b", "qwen3:14b", "qwen2.5:14b", "deepseek-r1:14b", "claude-haiku-4-5-20251001"]

# Models that don't support native tool calling — use prompt-based JSON extraction instead
NO_TOOL_MODELS = {"deepseek-r1:14b", "deepseek-r1:32b"}

# Models served via Anthropic API instead of Ollama
ANTHROPIC_MODELS = {"claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"}

# Models that need the native Ollama API (not OpenAI compat) due to options like num_ctx
# Maps model name -> extra options dict
NATIVE_API_MODELS = {"qwen2.5:32b": {"num_ctx": 2048}}

# Load .env for API key
def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"'))

_load_env()

with open("qwen_request.json") as f:
    BASE_REQUEST = json.load(f)

with open("input_payload.json") as f:
    ANTHROPIC_BASE = json.load(f)

PROMPT_FALLBACK_SYSTEM = """You are an expert Talent Evaluator.
Your goal is to identify growth opportunities.
1. Analyze the candidate's RESUME to establish a baseline.
2. Compare strictly against the JOB DESCRIPTION.

Output ONLY a valid JSON object — no prose, no markdown fences — with this exact structure:
{
  "verdict": "<one of: Step Up | Lateral | Title Regression | Pivot>",
  "one_line_summary": "<single sentence>",
  "match_scores": {
    "skills_match": <1-100>,
    "career_level_alignment": <1-100>,
    "experience_relevance": <1-100>,
    "culture_fit": <1-100>
  },
  "tech_stack_analysis": {
    "verdict": "<brief tech fit summary>",
    "matches": ["<skill>", ...],
    "gaps": ["<skill>", ...]
  }
}"""


def build_prompt_payload(model: str) -> dict:
    """Build a no-tools payload for models that don't support function calling."""
    # Pull resume text and user message from the base request
    system_msg = next(m for m in BASE_REQUEST["messages"] if m["role"] == "system")
    user_msg = next(m for m in BASE_REQUEST["messages"] if m["role"] == "user")
    resume_section = system_msg["content"].split("<candidate_resume>")[1].split("</candidate_resume>")[0].strip()

    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": PROMPT_FALLBACK_SYSTEM + f"\n\n<candidate_resume>\n{resume_section}\n</candidate_resume>",
            },
            user_msg,
        ],
    }


def call_anthropic(model: str) -> tuple[dict | None, float, str]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    payload = deepcopy(ANTHROPIC_BASE)

    start = time.time()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=payload["max_tokens"],
            temperature=payload["temperature"],
            system=payload["system"],
            tools=payload["tools"],
            tool_choice=payload["tool_choice"],
            messages=payload["messages"],
        )
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None, 0, str(e)

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s | tokens: input={resp.usage.input_tokens} output={resp.usage.output_tokens}")

    tool_block = next((b for b in resp.content if b.type == "tool_use"), None)
    if not tool_block:
        print(f"  [WARN] No tool_use block in response")
        return None, elapsed, "no_tool_call"

    return tool_block.input, elapsed, "ok"


def call_native_ollama(model: str, options: dict) -> tuple[dict | None, float, str]:
    """Call via native Ollama API for models needing options like num_ctx."""
    system_msg = next(m for m in BASE_REQUEST["messages"] if m["role"] == "system")
    user_msg = next(m for m in BASE_REQUEST["messages"] if m["role"] == "user")
    resume_section = system_msg["content"].split("<candidate_resume>")[1].split("</candidate_resume>")[0].strip()

    payload = {
        "model": model,
        "stream": False,
        "options": options,
        "messages": [
            {
                "role": "system",
                "content": PROMPT_FALLBACK_SYSTEM + f"\n\n<candidate_resume>\n{resume_section}\n</candidate_resume>",
            },
            user_msg,
        ],
    }
    start = time.time()
    try:
        resp = requests.post(OLLAMA_NATIVE_URL, json=payload, timeout=600)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print("  [ERROR] Request timed out after 600s")
        return None, 0, "timeout"
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] {e}")
        return None, 0, str(e)

    elapsed = time.time() - start
    raw = resp.json()
    content = raw["message"]["content"].strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
    try:
        args = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Failed to parse JSON: {e}\n  Raw: {content[:400]}")
        return None, elapsed, "parse_error"

    usage = raw.get("prompt_eval_count", "?"), raw.get("eval_count", "?")
    print(f"  Done in {elapsed:.1f}s | tokens: prompt={usage[0]} output={usage[1]}")
    return args, elapsed, "ok"


def call_model(model: str) -> tuple[dict | None, float, str]:
    if model in ANTHROPIC_MODELS:
        mode_label = "anthropic-tool-call"
        print(f"\n{'='*60}")
        print(f"  Running: {model}  [{mode_label}]")
        print(f"{'='*60}")
        return call_anthropic(model)

    if model in NATIVE_API_MODELS:
        mode_label = "native-prompt-JSON"
        print(f"\n{'='*60}")
        print(f"  Running: {model}  [{mode_label}]")
        print(f"{'='*60}")
        return call_native_ollama(model, NATIVE_API_MODELS[model])

    use_prompt_mode = model in NO_TOOL_MODELS
    payload = build_prompt_payload(model) if use_prompt_mode else deepcopy(BASE_REQUEST)
    if not use_prompt_mode:
        payload["model"] = model

    mode_label = "prompt-JSON" if use_prompt_mode else "tool-call"
    print(f"\n{'='*60}")
    print(f"  Running: {model}  [{mode_label}]")
    print(f"{'='*60}")

    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print("  [ERROR] Request timed out after 600s")
        return None, 0, "timeout"
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] {e}")
        return None, 0, str(e)

    elapsed = time.time() - start
    raw = resp.json()
    choice = raw["choices"][0]["message"]

    if use_prompt_mode:
        content = choice.get("content", "")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
        try:
            args = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  [ERROR] Failed to parse JSON from content: {e}")
            print(f"  Raw content: {content[:500]}")
            return None, elapsed, "parse_error"
    else:
        tool_calls = choice.get("tool_calls")
        if not tool_calls:
            content = choice.get("content", "")
            print(f"  [WARN] No tool_calls in response. Content: {content[:200]}")
            return None, elapsed, "no_tool_call"

        args_str = tool_calls[0]["function"]["arguments"]
        args_str = re.sub(r"<think>.*?</think>", "", args_str, flags=re.DOTALL).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError as e:
            print(f"  [ERROR] Failed to parse arguments JSON: {e}")
            print(f"  Raw args: {args_str[:500]}")
            return None, elapsed, "parse_error"

    tokens = raw.get("usage", {})
    print(f"  Done in {elapsed:.1f}s | tokens: {tokens}")
    return args, elapsed, "ok"


def print_result(model: str, args: dict, elapsed: float):
    scores = args.get("match_scores", {})
    avg = sum(scores.values()) / len(scores) if scores else 0
    tech = args.get("tech_stack_analysis", {})

    print(f"\n{'─'*60}")
    print(f"  MODEL   : {model}")
    print(f"  TIME    : {elapsed:.1f}s")
    print(f"  VERDICT : {args.get('verdict', 'N/A')}")
    print(f"  SUMMARY : {args.get('one_line_summary', 'N/A')}")
    print(f"\n  SCORES  (avg: {avg:.1f}/10):")
    for k, v in scores.items():
        print(f"    {k:<28} {v:>3}/10")
    print(f"\n  MATCHES ({len(tech.get('matches', []))}):")
    for m in tech.get("matches", []):
        print(f"    + {m}")
    print(f"\n  GAPS ({len(tech.get('gaps', []))}):")
    for g in tech.get("gaps", []):
        print(f"    - {g}")
    if tech.get("verdict"):
        print(f"\n  TECH VERDICT: {tech['verdict']}")


def unload_ollama_models():
    """Unload any currently loaded Ollama models to free RAM."""
    try:
        ps = requests.get("http://localhost:11434/api/ps", timeout=5).json()
        for m in ps.get("models", []):
            name = m["name"]
            print(f"  Unloading {name}...")
            requests.post(OLLAMA_NATIVE_URL, json={
                "model": name,
                "keep_alive": 0,
                "messages": [{"role": "user", "content": "x"}],
            }, timeout=15)
    except Exception as e:
        print(f"  [unload] {e}")
    time.sleep(3)


def main():
    results = {}
    unload_ollama_models()

    for model in MODELS:
        args, elapsed, status = call_model(model)
        results[model] = {"args": args, "elapsed": elapsed, "status": status}

        # Save raw result
        safe_name = model.replace(":", "_").replace(".", "_")
        out_path = f"result_{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump({"model": model, "elapsed": elapsed, "status": status, "result": args}, f, indent=2)
        print(f"  Saved → {out_path}")

    print(f"\n\n{'#'*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'#'*60}")

    for model, data in results.items():
        if data["status"] == "ok" and data["args"]:
            print_result(model, data["args"], data["elapsed"])
        else:
            print(f"\n  {model}: FAILED ({data['status']}) in {data['elapsed']:.1f}s")

    # Quick score diff table
    print(f"\n\n{'─'*60}")
    print(f"  SCORE DIFF TABLE")
    print(f"{'─'*60}")
    metrics = ["skills_match", "career_level_alignment", "experience_relevance", "culture_fit"]
    col = 16
    short_names = [m.split("/")[-1].split(":")[0][:col] for m in MODELS]
    header = f"  {'Metric':<30}" + "".join(f"{n:>{col}}" for n in short_names)
    print(header)
    print(f"  {'-'*28}" + "-" * (col * len(MODELS)))
    for metric in metrics:
        row = f"  {metric:<30}"
        for model in MODELS:
            data = results[model]
            if data["status"] == "ok" and data["args"]:
                val = data["args"].get("match_scores", {}).get(metric, "?")
                row += f"{str(val):>{col}}"
            else:
                row += f"{'ERR':>{col}}"
        print(row)

    avg_row = f"  {'avg':30}"
    for model in MODELS:
        data = results[model]
        if data["status"] == "ok" and data["args"]:
            scores = data["args"].get("match_scores", {})
            avg = sum(scores.values()) / len(scores) if scores else 0
            avg_row += f"{avg:>{col}.1f}"
        else:
            avg_row += f"{'ERR':>{col}}"
    print(f"  {'-'*28}" + "-" * (col * len(MODELS)))
    print(avg_row)


if __name__ == "__main__":
    main()
