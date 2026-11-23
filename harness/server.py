"""Start and stop the vLLM 0.32.0 server the way our runs did, with the preemption plugin loaded, or SGLang 0.5.21
with the same settings where it has them and plugin/sglang/sitecustomize.py for the retraction log.

The KV cache is fixed in tokens through --kv-cache-memory-bytes, so the capacity C is the same on any GPU with
enough memory. The engine reports the size it actually built in its log; run.py records it beside C.
"""
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent / "plugin"
KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2   # Qwen3-8B, BF16: K and V, 36 layers, 8 KV heads, head dim 128, 2 bytes
QUIET_ENV = {"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "VLLM_NO_USAGE_STATS": "1",
             "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0"}


def serve_command(model, revision, kv_tokens, port, prefix_caching=False, vllm="vllm"):
    cmd = [vllm, "serve", model, "--served-model-name", "qwen3-8b", "--host", "127.0.0.1", "--port", str(port),
           "--max-model-len", "40960", "--kv-cache-memory-bytes", str(kv_tokens * KV_BYTES_PER_TOKEN),
           "--max-num-seqs", "256", "--max-num-batched-tokens", "8192", "--block-size", "16",
           "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
           "--generation-config", "vllm", "--seed", "0", "--disable-uvicorn-access-log"]
    if revision:
        cmd[3:3] = ["--revision", revision]
    return cmd


def sglang_command(model, revision, kv_tokens, port, prefix_caching=False, python="python3"):
    """SGLang has no KV size in bytes; --max-total-tokens caps the pool in tokens. Page size stays at its default of 1.
    The cap binds only if the static memory fraction leaves room for it: L40S containers differ by ~3 GB of free memory,
    so the fraction is high and the prefill CUDA graphs (1.4 GB) are off; run.py stops if the pool comes out short."""
    cmd = [python, "-m", "sglang.launch_server", "--model-path", model, "--served-model-name", "qwen3-8b", "--host", "127.0.0.1",
           "--port", str(port), "--context-length", "40960", "--max-total-tokens", str(kv_tokens),
           "--max-running-requests", "256", "--chunked-prefill-size", "8192", "--random-seed", "0", "--enable-metrics",
           "--mem-fraction-static", "0.95", "--disable-prefill-cuda-graph"]
    if not prefix_caching:
        cmd.append("--disable-radix-cache")
    if revision:
        cmd[5:5] = ["--revision", revision]
    return cmd


def server_env(preempt_log, engine="vllm"):
    env = dict(os.environ, **QUIET_ENV)
    plugin = PLUGIN_DIR / "sglang" if engine == "sglang" else PLUGIN_DIR
    env["PYTHONPATH"] = os.pathsep.join([str(plugin)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["ACR_PREEMPT_LOG"] = str(preempt_log)
    env["VLLM_SERVER_DEV_MODE"] = "1"   # needed for POST /reset_prefix_cache between cells
    return env


def killpg(p):
    if p is None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=30)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def start(out, model, revision, kv_tokens, port, prefix_caching=False, vllm="vllm", wait_s=1200, engine="vllm"):
    """Start the engine and wait for /health. Returns (process, log path), or (None, log path) if it did not come up.
    For SGLang, `vllm` names the Python executable that has sglang installed."""
    out = Path(out)
    log_path = out / f"server_{kv_tokens}.log"
    if engine == "sglang":
        cmd = sglang_command(model, revision, kv_tokens, port, prefix_caching, vllm)
    else:
        cmd = serve_command(model, revision, kv_tokens, port, prefix_caching, vllm)
    log = open(log_path, "a")
    log.write(f"# {' '.join(cmd)}\n")
    log.flush()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=server_env(out / "preempt.jsonl", engine),
                         start_new_session=True)
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            if urllib.request.urlopen(url, timeout=5).status == 200:
                print(f"{engine} up after {time.time() - t0:.0f} s with KV {kv_tokens} tokens", flush=True)
                return p, log_path
        except Exception:
            pass
        if p.poll() is not None:
            break
        time.sleep(3)
    killpg(p)
    print(log_path.read_text()[-4000:], flush=True)
    return None, log_path


def engine_kv_tokens(log_path):
    txt = Path(log_path).read_text()
    m = (re.findall(r"GPU KV cache size: ([\d,]+) tokens", txt)
         or re.findall(r"KV Cache is allocated\. dtype: \S+, #tokens: (\d+)", txt))
    return int(m[-1].replace(",", "")) if m else None


def plugin_loaded(preempt_log, since):
    p = Path(preempt_log)
    if not p.exists():
        return False
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("ev") == "plugin_loaded" and e.get("t", 0.0) >= since:
            return True
    return False


def idle_and_reset(base_url, parse_metrics, timeout=120, reset_path="/reset_prefix_cache"):
    """Wait until the engine has no running or waiting request, then reset the prefix cache (SGLang: /flush_cache)."""
    t0 = time.time()
    m = {}
    while time.time() - t0 < timeout:
        try:
            m = parse_metrics(urllib.request.urlopen(base_url + "/metrics", timeout=5).read().decode())
            if m.get("running", 1) == 0 and m.get("waiting", 1) == 0:
                break
        except Exception:
            pass
        time.sleep(1)
    ok = False
    try:
        req = urllib.request.Request(base_url + reset_path, data=b"", method="POST")
        ok = urllib.request.urlopen(req, timeout=30).status == 200
    except Exception as exc:
        print("reset_prefix_cache failed:", exc, flush=True)
    return dict(idle_after_s=round(time.time() - t0, 1), last=m, reset=ok)
