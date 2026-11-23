"""SGLang path of the proxy against a fake SGLang server on 127.0.0.1. No GPU, no sglang install.

Checks that on SGLang every request carries the same max_tokens (so SGLang's admission learns nothing about L), that
the proxy closes the stream once L tokens have arrived and the server sees the disconnect, the metric names, the
server command line and the KV-size line of SGLang's log.

    python -I tests/test_sglang_replay.py
"""
import asyncio
import json
import socket
import sys
import tempfile
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import replay  # noqa: E402
import server  # noqa: E402

SEEN = dict(bodies=[], disconnected=[], sent=[])
FAKE = dict(token_s=0.002, slots=None)


async def completions(request):
    body = await request.json()
    SEEN["bodies"].append(body)
    if FAKE["slots"] is not None:
        async with FAKE["slots"]:
            return await stream_tokens(request, body)
    return await stream_tokens(request, body)


async def stream_tokens(request, body):
    resp = web.StreamResponse()
    n = 0
    try:
        await resp.prepare(request)
        while n < body["max_tokens"]:
            n += 1
            chunk = dict(choices=[dict(text="x", index=0)], usage=dict(prompt_tokens=len(body["prompt"]), completion_tokens=n))
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await asyncio.sleep(FAKE["token_s"])
    except (ConnectionResetError, asyncio.CancelledError, aiohttp.ClientError):
        SEEN["disconnected"].append(body["rid"])
        return resp
    finally:
        SEEN["sent"].append(n)
    await resp.write(b"data: [DONE]\n\n")
    return resp


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def stream_check():
    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    port = free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    replay.URL, replay.ENGINE = f"http://127.0.0.1:{port}", "sglang"
    try:
        async with aiohttp.ClientSession() as session:
            reqs = [replay.Req(f"T_stock_25-r{k:05d}", k, 50, L, 0.0) for k, L in enumerate((5, 40, 120))]
            reqs.append(replay.Req("T_stock_25-r00003", 3, 30000, 7, 0.0))
            await asyncio.gather(*(replay.stream_one(session, r, [1] * r.p, lambda r, now: None) for r in reqs))
        await asyncio.sleep(0.3)
    finally:
        replay.ENGINE = "vllm"
        await runner.cleanup()
    return reqs


def test_stream_stops_at_trace_length():
    reqs = asyncio.run(stream_check())
    assert all(r.state == "finished" and r.n == r.L for r in reqs), [(r.state, r.n, r.L) for r in reqs]
    assert sorted(b["max_tokens"] for b in SEEN["bodies"]) == [replay.CONTEXT_LEN - 30000] + [replay.SGLANG_MAX_NEW] * 3
    assert sorted(b["rid"] for b in SEEN["bodies"]) == sorted(r.rid + replay.RID_END for r in reqs)
    assert all("ignore_eos" not in b and b["stream_options"]["continuous_usage_stats"] for b in SEEN["bodies"])
    assert all(b["logit_bias"] == {"151643": -100.0, "151645": -100.0} for b in SEEN["bodies"])
    assert sorted(SEEN["disconnected"]) == sorted(r.rid + replay.RID_END for r in reqs), SEEN
    assert max(SEEN["sent"]) < 200, SEEN["sent"]


async def calibrate_check(tmp):
    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    port = free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    replay.URL, replay.ENGINE = f"http://127.0.0.1:{port}", "sglang"
    FAKE.update(token_s=0.005, slots=asyncio.Semaphore(21))
    try:
        return await replay.calibrate(tmp, 190000, evict_wall_s=2.0)
    finally:
        replay.ENGINE = "vllm"
        FAKE.update(token_s=0.002, slots=None)
        await runner.cleanup()


def test_calibrate_when_engine_holds_some_back():
    """SGLang admitted 21 of the 24 decode-calibration requests at once on the L40S; tau must come from those 21."""
    with tempfile.TemporaryDirectory() as tmp:
        cal = asyncio.run(calibrate_check(tmp))
    assert cal["tau_n_concurrent"] == 21 and cal["tau_kv_tokens_range"] == [21 * 6000, 21 * 7600], cal
    assert cal["tau_full"] is not None and 0.004 < cal["tau_full"] < 0.02, cal
    assert sum(b["max_tokens"] == 1600 for b in SEEN["bodies"] if "-tau" in b["rid"]) == 24
    assert all(b["max_tokens"] == replay.SGLANG_MAX_NEW for b in SEEN["bodies"] if "-ev" in b["rid"])


def test_metrics_names():
    txt = "\n".join(['sglang:token_usage{model_name="qwen3-8b"} 0.97', 'sglang:num_running_reqs{model_name="qwen3-8b"} 31.0',
                     'sglang:num_queue_reqs{model_name="qwen3-8b"} 4.0',
                     'sglang:num_retracted_requests_total{model_name="qwen3-8b"} 12.0',
                     'vllm:num_requests_running{model_name="qwen3-8b"} 99.0'])
    replay.ENGINE = "sglang"
    try:
        m = replay.parse_metrics(txt)
    finally:
        replay.ENGINE = "vllm"
    assert (m["kv"], m["running"], m["waiting"], m["preempt"]) == (0.97, 31.0, 4.0, 12.0), m


def test_command_and_kv_line():
    cmd = server.sglang_command("Qwen/Qwen3-8B", "abc123", 190000, 8000)
    s = " ".join(cmd)
    for part in ("--max-total-tokens 190000", "--max-running-requests 256", "--context-length 40960", "--disable-radix-cache",
                 "--enable-metrics", "--revision abc123", "--served-model-name qwen3-8b", "--chunked-prefill-size 8192"):
        assert part in s, part
    assert cmd[cmd.index("--model-path") + 1] == "Qwen/Qwen3-8B"
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.log"
        log.write_text("[2026-11-23] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 190000, "
                       "K size: 13.05 GB, V size: 13.05 GB\n")
        assert server.engine_kv_tokens(log) == 190000
    env = server.server_env(Path(d) / "p.jsonl", "sglang")
    assert env["PYTHONPATH"].split(":")[0].endswith("plugin/sglang") and env["ACR_PREEMPT_LOG"].endswith("p.jsonl")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
