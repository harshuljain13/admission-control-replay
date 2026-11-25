"""Plumbing test against a fake engine on 127.0.0.1: every arm, the decision log, the measures and analyze.py.
No GPU and no network beyond the loopback. Runs in about half a minute.

    python -I tests/test_replay.py        (or: pytest tests)
"""
import asyncio
import json
import socket
import sys
import tempfile
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import analyze  # noqa: E402
import measure  # noqa: E402
import replay  # noqa: E402

STATE = dict(kv=0.0, running=0, preempt=0)


async def completions(request):
    body = await request.json()
    rid = request.headers["X-Request-Id"]
    resp = web.StreamResponse()
    await resp.prepare(request)
    STATE["running"] += 1
    try:
        for n in range(1, body["max_tokens"] + 1):
            await asyncio.sleep(0.004)
            chunk = dict(id=f"cmpl-{rid}", usage=dict(completion_tokens=n))
            await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
    except ConnectionResetError:   # the proxy cancels open streams when a cell ends
        pass
    finally:
        STATE["running"] -= 1
    return resp


async def metrics(request):
    STATE["kv"] = min(0.99, 0.1 * STATE["running"])
    if STATE["running"] > 6:
        STATE["preempt"] += 1
    txt = (f'vllm:kv_cache_usage_perc{{engine="0"}} {STATE["kv"]}\nvllm:num_requests_running{{engine="0"}} {STATE["running"]}\n'
           f'vllm:num_requests_waiting{{engine="0"}} 0\nvllm:num_preemptions_total{{engine="0"}} {STATE["preempt"]}\n'
           f'vllm:num_preemptions_created{{engine="0"}} 1.7e9\n')
    return web.Response(text=txt)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _run(tmp):
    port = free_port()
    replay.URL = f"http://127.0.0.1:{port}"
    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/metrics", metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    reqs = [dict(i=i, p=100 + 37 * i, L=20 + (i * 13) % 200, h=list(range(i, i + (100 + 37 * i + 15) // 16))) for i in range(200)]
    C = 20000
    cells = {}
    try:
        for name, arm, door in (("T_stock_25", "stock", False), ("T_gate_25", "gate", False), ("T_known_25", "known", False),
                                ("T_wm10door_25", "wm10", True), ("T_wm20_surge", "wm20", False)):
            cfg = dict(cell=name, arm=arm, door=door, trace="thinking", load=1.0, rate=8.0, arrival_seed=1, wall_s=6.0, C=C,
                       PF=8000.0, TAU=0.004, alpha=0.01, K=4000.0, window_s=1.0, count_window_s=4.0,
                       segments=[[0, 4.0], [2.0, 16.0], [4.0, 4.0]] if name.endswith("surge") else None)
            res = await replay.run_cell(cfg, reqs, f"{tmp}/cells_{C}/{name}")
            m = measure.cell_measures(res, [], cfg["PF"], C)
            decisions = Path(f"{tmp}/cells_{C}/{name}/decisions.jsonl").read_text().splitlines()
            print(arm, m["arrivals"], m["states"], "good", m["good"], "decisions", len(decisions), "h_last", m["h_last"])
            assert m["arrivals"] > 20 and len(decisions) > 0
            assert all("verdict" in json.loads(d) for d in decisions)
            cells[name] = dict(done=True)
    finally:
        await runner.cleanup()
    state = dict(notes=[], regimes=[dict(C=C, calibration=dict(PF=8000.0, tau_full=0.004), rates={}, cells=cells)])
    Path(tmp, "state.json").write_text(json.dumps(state))
    analyze.main(tmp)
    summary = json.loads(Path(tmp, "summary.json").read_text())
    # the cells ran 6 s, far under the 1,240 s validity floor, so every gate that needs them is inconclusive
    assert summary["outcome"] == "INCONCLUSIVE", summary["outcome"]
    assert set(summary["regimes"][0]["cells"]) == set(cells)


def test_fake_engine_all_arms():
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(_run(tmp))


if __name__ == "__main__":
    test_fake_engine_all_arms()
    print("ok")
