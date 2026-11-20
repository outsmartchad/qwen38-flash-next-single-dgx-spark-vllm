#!/usr/bin/env python3
"""Measure localhost model streaming and fresh-prompt retrieval."""
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

BASE_URL = os.environ["SPARK_BENCHMARK_URL"].rstrip("/")
base_url = urlsplit(BASE_URL)
if (base_url.scheme != "http" or base_url.hostname not in ("127.0.0.1", "::1")
        or base_url.username or base_url.path or base_url.query or base_url.fragment):
    raise ValueError("SPARK_BENCHMARK_URL must point to the localhost model API.")
MODEL = os.environ["SPARK_BENCHMARK_MODEL"]
HASH_MAP_PROMPT = ("Write a detailed step-by-step explanation of how a hash map works, "
                   "including collision handling, resizing, and time complexity. Be thorough.")
CURL = ["curl", "--noproxy", "*", "-fsS", "-H", "Content-Type: application/json"]

def api(path, body=None, timeout=20):
    cmd = CURL + ["--max-time", str(timeout), BASE_URL + path]
    kwargs = {}
    if body is not None:
        cmd += ["--data-binary", "@-"]
        kwargs["input"] = json.dumps(body).encode()
    response = subprocess.run(cmd, check=True, capture_output=True, **kwargs)
    return json.loads(response.stdout)

def chat_body(prompt, thinking):
    template = {"enable_thinking": thinking}
    if MODEL == "Muse-Glimmer-30B":
        template = {"reasoning_strength": "high" if thinking else "low"}
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": template}

def stream(prompt, *, thinking, limit, temperature, force=False, barrier=None):
    body = chat_body(prompt, thinking)
    body.update({"max_tokens": limit, "temperature": temperature, "top_p": 1,
                 "stream": True, "stream_options": {"include_usage": True}})
    if temperature > 0:
        body["top_k"] = 20
    if force:
        body.update({"min_tokens": limit, "ignore_eos": True, "stop": []})
    cmd = CURL + ["-N", "--max-time", "420", "--data-binary", "@-",
                  BASE_URL + "/v1/chat/completions"]
    first = last = first_answer = None
    reasoning_events = 0
    pieces = []
    usage = {}
    saw_done = False
    finish_reason = None
    with tempfile.TemporaryFile() as payload:
        payload.write(json.dumps(body).encode())
        payload.seek(0)
        if barrier is not None:
            barrier.wait(timeout=20)
        start = time.monotonic()
        with subprocess.Popen(cmd, stdin=payload, stdout=subprocess.PIPE, text=True) as proc:
            for line in proc.stdout:
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    saw_done = True
                    continue
                chunk = json.loads(value)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"])[:300])
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    reason = delta.get("reasoning_content") or delta.get("reasoning")
                    answer = delta.get("content")
                    if reason or answer:
                        last = time.monotonic()
                        if first is None:
                            first = last
                        if reason:
                            reasoning_events += 1
                        if answer:
                            if first_answer is None:
                                first_answer = last
                            pieces.append(answer)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                usage.update(chunk.get("usage") or {})
            rc = proc.wait()
        finished = time.monotonic()
    if rc or not saw_done or first is None or last is None or not usage:
        raise RuntimeError(f"Incomplete stream: curl={rc}, done={saw_done}, usage={bool(usage)}")
    received = usage["completion_tokens"]
    if force and received != limit:
        raise RuntimeError(f"Received {received} output tokens, expected {limit}.")
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_tokens": usage["prompt_tokens"], "cached_prompt_tokens": details.get("cached_tokens"),
        "completion_tokens": received, "finish_reason": finish_reason,
        "reasoning_events": reasoning_events,
        "first_token_seconds": first - start,
        "first_answer_seconds": None if first_answer is None else first_answer - start,
        "elapsed_seconds": finished - start,
        "stream_decode_tokens_per_second": (received - 1) / (last - first) if last > first else None,
    }, (start, first, last, finished), "".join(pieces)

def decode_wave(streams, repetition):
    prompts = [HASH_MAP_PROMPT] if streams == 1 else [
        f"Request ID: decode-{index}\n{HASH_MAP_PROMPT}" for index in range(1, streams + 1)]
    start_together = threading.Barrier(streams)
    with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
        jobs = [pool.submit(stream, prompt, thinking=True, limit=400, temperature=0.7,
                            force=True, barrier=start_together) for prompt in prompts]
        outputs = [job.result() for job in jobs]
    marks = [output[1] for output in outputs]
    rows = [output[0] for output in outputs]
    tokens = sum(row["completion_tokens"] for row in rows)
    decode_seconds = max(mark[2] for mark in marks) - min(mark[1] for mark in marks)
    elapsed_seconds = max(mark[3] for mark in marks) - min(mark[0] for mark in marks)
    return {"streams": streams, "repetition": repetition, "requests": rows,
            "aggregate_decode_tokens_per_second": (tokens - streams) / decode_seconds,
            "aggregate_wall_tokens_per_second": tokens / elapsed_seconds}

def token_count(prompt):
    for path in ("/v1/tokenize", "/tokenize"):
        try:
            data = api(path, chat_body(prompt, False), timeout=50)
            break
        except subprocess.CalledProcessError:
            if path == "/tokenize":
                raise
    for key in ("count", "num_tokens", "token_count"):
        if type(data.get(key)) is int:
            return data[key]
    for key in ("tokens", "token_ids", "input_ids"):
        if isinstance(data.get(key), list):
            return len(data[key])
    raise RuntimeError(f"Unexpected tokenize fields: {sorted(data)}")

def retrieval_prompt(seed, count):
    rng = random.Random(seed)
    key = f"ICE-{rng.getrandbits(36):09X}"
    lines = [f"Batch {rng.getrandbits(64):016x}. This is a synthetic inventory.",
             "Find the audit key. Reply with the exact key only."]
    for i in range(count):
        if i == count // 2:
            lines.append(f"AUDIT KEY: {key}")
        lines.append(f"record {i:05d} | depot {rng.randrange(1, 99)} | units {rng.randrange(1, 1000)} | lot {rng.getrandbits(32):08x}")
    lines.append("What is the audit key? Output the exact key only.")
    return "\n".join(lines), key

def prefill(target, seed):
    trial, _ = retrieval_prompt(seed, 32)
    try:
        per_row = token_count(trial) / 32
        count = max(1, round(target/per_row))
        for _ in range(4):
            prompt, key = retrieval_prompt(seed, count)
            actual = token_count(prompt)
            if abs(actual - target) / target <= 0.005:
                break
            count = max(1, round(count * target / actual))
        else:
            raise RuntimeError(f"Tokenize could not size {target} request")
        sizing = "tokenize endpoint"
    except subprocess.CalledProcessError:
        calibration, _ = retrieval_prompt(seed ^ 0x2CFED51, 256)
        usage, _, _ = stream(calibration, thinking=False, limit=16, temperature=0)
        estimated = usage["prompt_tokens"] / 256
        prompt, key = retrieval_prompt(seed, max(1, round(target/estimated)))
        actual = None
        sizing = "independent generation-calibration prompt, actual usage reported below"
    result, _, answer = stream(prompt, thinking=False, limit=128 if MODEL == "Muse-Glimmer-30B" else 32, temperature=0)
    result["requested_prompt_tokens"] = target
    result["tokenize_prompt_tokens"] = actual
    result["prompt_sizing_method"] = sizing
    result["exact_key_found"] = answer.strip() == key
    cached = result["cached_prompt_tokens"]
    counted = result["prompt_tokens"] - (cached if type(cached) is int else 0)
    result["prompt_tokens_per_second_to_first_token"] = counted / result["first_token_seconds"]
    return result

def main():
    if MODEL not in [item["id"] for item in api("/v1/models")["data"]]:
        raise RuntimeError("Wrong model on localhost.")
    sanity, _, answer = stream("Reply with exactly the word READY.", thinking=False,
                               limit=128 if MODEL == "Muse-Glimmer-30B" else 24, temperature=0)
    if answer.strip().upper().rstrip(".") != "READY":
        raise RuntimeError("Basic chat smoke failed.")
    print(f"Chat smoke passed ({sanity['completion_tokens']} output tokens).", file=sys.stderr, flush=True)
    stream(HASH_MAP_PROMPT, thinking=True, limit=32, temperature=0.7)
    waves = []
    for streams in range(1, int(os.environ.get("SPARK_BENCHMARK_MAX_STREAMS", "5")) + 1):
        for repetition in range(1, 4):
            wave = decode_wave(streams, repetition)
            waves.append(wave)
            med = statistics.median(row["first_token_seconds"] for row in wave["requests"])
            print(f"C{streams} {repetition}/3: {wave['aggregate_decode_tokens_per_second']:.1f} tok/s, TTFT {med*1000:.0f} ms",
                  file=sys.stderr, flush=True)
    if any(not any(r["reasoning_events"] for r in w["requests"]) for w in waves):
        raise RuntimeError("A measured wave returned no reasoning events.")
    print(json.dumps({"model": MODEL, "workload": "SparkDash hash-map prose",
                      "temperature": 0.7, "top_p": 1, "top_k": 20, "thinking": True,
                      "output_tokens": 400, "waves": waves}, indent=2, allow_nan=False))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        target = int(sys.argv[1])
        print(json.dumps(prefill(target, int(sys.argv[2])), indent=2, allow_nan=False))
