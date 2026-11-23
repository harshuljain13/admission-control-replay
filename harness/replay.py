"""The proxy, the four arms and the calibration, run against a vLLM or SGLang server (by default on 127.0.0.1:8000).

The proxy sees what a gateway sees: each request's prompt length on arrival, the tokens it streams back, and the
server's Prometheus metrics. Only the known-length arm reads the trace's output length. On vLLM every arm sends
max_tokens = L with ignore_eos, and vLLM's admission does not read max_tokens. SGLang's admission does read it
(it reserves min(max_new_tokens, 4096) times an adapted ratio), so on SGLang every arm sends max_tokens =
SGLANG_MAX_NEW, the trace's output clip (less if the prompt leaves less room in the context), and the proxy
closes the stream once L tokens have arrived; SGLang then aborts the request and frees its memory. SGLang aborts
every request whose rid starts with the closed one's rid, so each rid sent to SGLang ends with RID_END.
ignore_eos would switch SGLang (radix cache off) to a separate admission path that reserves the full max_new_tokens,
so on SGLang the EOS tokens are banned with logit_bias instead.

Decisions go through AdmitPerf's Policy class (vendored at commit 5bcb81c, see vendor/), so every decision is logged
in its format.
"""
import asyncio
import json
import math
import random
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor"))
from admitperf import Policy  # noqa: E402

URL = "http://127.0.0.1:8000"
MODEL = "qwen3-8b"
ENGINE = "vllm"           # or "sglang"; set by run.py
SGLANG_MAX_NEW = 16000
CONTEXT_LEN = 40960
RID_END = "|"
EOS_TOKEN_IDS = (151643, 151645)   # Qwen3 <|endoftext|>, <|im_end|>
TTFT_LIMIT, TPOT_LIMIT = 30.0, 0.150
VOCAB_LO, VOCAB_HI = 1000, 150000
TRACE_SEED = {"thinking": 1, "chat": 2, "thinking_long": 1}   # thinking_long reuses the Thinking block IDs


# ------------------------------------------------------------------------------------------------ prompts

class Prompts:
    """Token IDs from the trace's 16-token block hashes: block h of trace T is 16 IDs drawn with seed (T, h), so shared
    blocks in the trace are shared prefixes here, and the last block is cut to the prompt length."""

    def __init__(self, trace):
        self.tid = TRACE_SEED[trace]
        self.cache = {}

    def block(self, h):
        b = self.cache.get(h)
        if b is None:
            b = np.random.default_rng([self.tid, int(h)]).integers(VOCAB_LO, VOCAB_HI, 16).tolist()
            self.cache[h] = b
        return b

    def ids(self, req):
        out = []
        for h in req["h"]:
            out += self.block(h)
        return out[: req["p"]]


def unique_prompt(n, seed):
    return np.random.default_rng([99, seed]).integers(VOCAB_LO, VOCAB_HI, n).tolist()


# ------------------------------------------------------------------------------------------------ metrics

KEYS = {"vllm:kv_cache_usage_perc": "kv", "vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
        "vllm:num_preemptions_total": "preempt", "vllm:prompt_tokens_total": "prompt_tok", "vllm:generation_tokens_total": "gen_tok"}
# SGLang 0.5.21 with --enable-metrics. num_queue_reqs includes retracted requests, as vLLM's waiting count does.
KEYS_SGLANG = {"sglang:token_usage": "kv", "sglang:num_running_reqs": "running", "sglang:num_queue_reqs": "waiting",
               "sglang:num_retracted_requests_total": "preempt", "sglang:prompt_tokens_total": "prompt_tok",
               "sglang:generation_tokens_total": "gen_tok"}


def parse_metrics(txt):
    out = {}
    for line in txt.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        k = (KEYS_SGLANG if ENGINE == "sglang" else KEYS).get(name)
        if k:
            out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


# ------------------------------------------------------------------------------------------------ policies

class Stock(Policy):
    name = "stock_passthrough"
    baseline = True

    def decide(self, m):
        return self.admit()


class Watermark(Policy):
    name = "watermark_10pct"
    signal_quantity = "kv_pressure"

    def decide(self, m):
        if m["acr:preempted_waiting"] > 0:
            return self.defer("kv_pressure", 100)
        if m["acr:used_est"] + m["acr:cand_prompt"] + self.frac * m["acr:C"] > m["acr:C"]:
            return self.defer("kv_pressure", 100)
        return self.admit()


class BudgetGate(Policy):
    name = "recompute_budget_gate"
    signal_quantity = "kv_pressure"
    signal_structure = "formal-bound"

    def decide(self, m):
        if m["acr:preempted_waiting"] > 0:
            return self.defer("kv_pressure", 100)
        if m["acr:used_est"] + m["acr:cand_prompt"] + m["acr:h"] > m["acr:C"]:
            return self.defer("capacity_reserved", 100)
        return self.admit()


class KnownLength(Policy):
    name = "known_length_peak"
    signal_quantity = "predicted_length"

    def decide(self, m):
        if m["acr:preempted_waiting"] > 0:
            return self.defer("kv_pressure", 100)
        if m["acr:used_est"] + m["acr:cand_prompt"] > m["acr:C"]:
            return self.defer("kv_pressure", 100)
        if m["acr:oracle_peak"] > m["acr:C"]:
            return self.defer("predicted_length", 100)
        return self.admit()


def blocks_tokens(n):
    return 16 * (-(-n // 16)) + 16


def oracle_peak(items):
    """items: (remaining, resident tokens). Peak of the summed KV if all decode together: just before the item with
    the i-th smallest remaining finishes, every item with at least that remaining holds its tokens plus that many."""
    items = sorted(items)
    n = len(items)
    if not n:
        return 0.0
    suffix_m = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_m[i] = suffix_m[i + 1] + items[i][1]
    peak = suffix_m[0]
    for i, (rem, _) in enumerate(items):
        peak = max(peak, suffix_m[i] + (n - i) * rem)
    return peak


# ------------------------------------------------------------------------------------------------ one request

class Req:
    __slots__ = ("rid", "i", "p", "L", "t_arr", "t_send", "t_first", "t_last", "n", "state", "marks", "max_gap",
                 "t_mark", "t_restart", "task", "chunks", "err", "max_tokens")

    def __init__(self, rid, i, p, L, t_arr):
        self.rid, self.i, self.p, self.L, self.t_arr = rid, i, p, L, t_arr
        self.t_send = self.t_first = self.t_last = None
        self.n, self.state, self.marks, self.max_gap = 0, "queued", 0, 0.0
        self.t_mark = None
        self.t_restart = None
        self.max_tokens = None   # set only by calibration; trace requests never reveal L to SGLang
        self.task = None
        self.chunks = None
        self.err = None

    def rec(self, t0):
        f = lambda x: None if x is None else round(x - t0, 4)
        return dict(rid=self.rid, i=self.i, p=self.p, L=self.L, t_arr=f(self.t_arr), t_send=f(self.t_send), t_first=f(self.t_first),
                    t_last=f(self.t_last), n=self.n, state=self.state, marks=self.marks, max_gap=round(self.max_gap, 4), err=self.err)


async def stream_one(session, r, prompt, on_token, keep_chunks=False):
    sglang = ENGINE == "sglang"
    body = dict(model=MODEL, prompt=prompt, max_tokens=(r.max_tokens or min(SGLANG_MAX_NEW, CONTEXT_LEN - r.p)) if sglang else r.L, temperature=0.0,
                stream=True, stream_options=dict(include_usage=True, continuous_usage_stats=True))
    if sglang:
        body["rid"] = r.rid + RID_END
        body["logit_bias"] = {str(t): -100.0 for t in EOS_TOKEN_IDS}
    else:
        body["ignore_eos"] = True
    r.t_send = time.time()
    r.state = "sent"
    if keep_chunks:
        r.chunks = []
    try:
        async with session.post(URL + "/v1/completions", json=body, headers={"X-Request-Id": r.rid}) as resp:
            if resp.status != 200:
                r.state, r.err = "error", f"http {resp.status}: {(await resp.text())[:200]}"
                return
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                obj = json.loads(data)
                u = obj.get("usage") or {}
                n = int(u.get("completion_tokens") or 0)
                if n > r.n:
                    now = time.time()
                    if r.t_first is None:
                        r.t_first = now
                    else:
                        r.max_gap = max(r.max_gap, now - r.t_last)
                    r.t_last = now
                    r.n = n
                    if keep_chunks:
                        r.chunks.append((round(now, 4), n))
                    on_token(r, now)
                    if sglang and r.n >= r.L:
                        resp.close()   # SGLang sees the disconnect and aborts the request
                        break
        r.state = "finished" if r.n >= r.L else "short"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        r.state, r.err = "error", f"{type(exc).__name__}: {str(exc)[:200]}"


# ------------------------------------------------------------------------------------------------ one cell

class Cell:
    def __init__(self, cfg, reqs, out_dir, session):
        self.cfg, self.out_dir, self.session = cfg, Path(out_dir), session
        self.C = float(cfg["C"])
        self.arm, self.door = cfg["arm"], cfg.get("door", False)
        self.prompts = Prompts(cfg["trace"])
        self.reqs_src = reqs
        self.queue = []          # proxy FIFO
        self.live = {}           # rid -> Req, sent and not ended
        self.all = []
        self.m = dict(kv=0.0, running=0.0, waiting=0.0, preempt=0.0)
        self.mseries, self.ctl = [], []
        self.h = 0.0
        self.W_est_win, self.W_est_total = 0.0, 0.0
        self.est_events = []
        self.recent_L = []
        self.rejected = 0
        self.wake = asyncio.Event()
        pol_cls = dict(stock=Stock, wm10=Watermark, gate=BudgetGate, known=KnownLength)[self.arm]
        params = dict(frac=0.10) if self.arm == "wm10" else {}
        if self.arm == "gate":
            params = dict(alpha=cfg["alpha"], K=cfg["K"], window_s=cfg["window_s"])
        self.policy = pol_cls(log=str(self.out_dir / "decisions.jsonl"), experiment=cfg.get("experiment", "exp02"), run=cfg["cell"],
                              notes=f"Qwen3-8B, {cfg.get('engine', 'vLLM 0.32.0')}, {cfg.get('gpu', 'GPU not named')}, C={int(self.C)} tokens, prefix caching {cfg.get('prefix_caching', 'off')}, trace {cfg['trace']}, load {cfg['load']}x",
                              provenance={"C": "pinned", "alpha": "pre-registered", "K": "pre-registered", "frac": "pre-registered"}, **params)

    # --- signals the proxy derives
    def used_est(self):
        pend = sum(r.p for r in self.live.values() if r.t_first is None)
        return self.m["kv"] * self.C + pend

    def preempted_waiting(self):
        pend = sum(1 for r in self.live.values() if r.t_first is None)
        return max(0.0, self.m["waiting"] - pend)

    def metrics_for(self, c):
        m = {"vllm:kv_cache_usage_perc": self.m["kv"], "vllm:num_requests_waiting": self.m["waiting"],
             "vllm:num_requests_running": self.m["running"], "acr:C": self.C, "acr:used_est": self.used_est(),
             "acr:cand_prompt": float(c.p), "acr:preempted_waiting": self.preempted_waiting(), "acr:h": self.h,
             "acr:proxy_queue": float(len(self.queue))}
        if self.arm == "known":
            items = [(max(r.L - r.n, 1), blocks_tokens(r.p + r.n)) for r in self.live.values()]
            items.append((c.L, blocks_tokens(c.p)))
            m["acr:oracle_peak"] = oracle_peak(items)
        return m

    # --- events
    def on_token(self, r, now):
        if r.t_mark is not None and now - r.t_mark > 0.25:
            r.t_mark = None
            r.t_restart = now

    def on_preemptions(self, d, now):
        """Attribute d new preemptions to the requests the engine evicts first: on vLLM the most recently (re)started
        resident requests, on SGLang those with the fewest tokens decoded (ties: the longest prompt)."""
        cands = [r for r in self.live.values() if r.t_mark is None]
        if ENGINE == "sglang":
            cands.sort(key=lambda r: (r.n, -r.p))
        else:
            cands.sort(key=lambda r: max(r.t_send or 0.0, r.t_restart or 0.0), reverse=True)
        for r in cands[:d]:
            r.t_mark = now
            r.marks += 1
            w = (r.p + r.n) / self.cfg["PF"]
            self.W_est_win += w
            self.W_est_total += w
            self.est_events.append((round(now - self.t0, 3), r.rid, r.p + r.n))

    async def poll(self):
        last_pre = None
        while True:
            try:
                async with self.session.get(URL + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    m = parse_metrics(await resp.text())
                now = time.time()
                self.m.update(m)
                pre = m.get("preempt", 0.0)
                if last_pre is not None and pre > last_pre:
                    self.on_preemptions(int(round(pre - last_pre)), now)
                last_pre = pre
                self.mseries.append((round(now - self.t0, 3), round(m.get("kv", -1), 5), m.get("running"), m.get("waiting"), pre,
                                     len(self.queue), len(self.live), round(self.h, 1)))
                self.wake.set()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.mseries.append((round(time.time() - self.t0, 3), "error", type(exc).__name__))
            await asyncio.sleep(0.5)

    async def controller(self):
        K, a, win = self.cfg["K"], self.cfg["alpha"], self.cfg["window_s"]
        while True:
            await asyncio.sleep(win)
            share = self.W_est_win / win
            self.h = max(0.0, self.h + K * (share - a))
            self.ctl.append((round(time.time() - self.t0, 3), round(share, 5), round(self.h, 1)))
            self.W_est_win = 0.0
            self.wake.set()

    def send(self, r, now):
        prompt = self.prompts.ids(self.reqs_src[r.i])
        self.live[r.rid] = r
        r.task = asyncio.ensure_future(self._run(r, prompt))

    async def _run(self, r, prompt):
        try:
            await stream_one(self.session, r, prompt, self.on_token)
        finally:
            self.live.pop(r.rid, None)
            if r.state == "finished":
                self.recent_L.append(r.n)
                if len(self.recent_L) > 400:
                    self.recent_L.pop(0)
            self.wake.set()

    def door_rejects(self):
        if len(self.recent_L) >= 20:
            arr = np.asarray(self.recent_L, float)
            m1, m2 = float(arr.mean()), float((arr ** 2).mean())
        else:
            m1, m2 = 1000.0, 2.0e6
        backlog = sum(r.p * m1 + m2 / 2 for r in self.queue)
        return backlog * self.cfg["TAU"] / self.C > 0.8 * TTFT_LIMIT

    async def gatekeeper(self):
        while True:
            now = time.time()
            # clients give up when no first token arrives within the TTFT limit
            keep = []
            for r in self.queue:
                if now - r.t_arr > TTFT_LIMIT:
                    r.state = "abandoned_in_proxy"
                else:
                    keep.append(r)
            self.queue = keep
            for r in list(self.live.values()):
                if r.t_first is None and now - r.t_arr > TTFT_LIMIT and r.task is not None and not r.task.done():
                    r.task.cancel()
                    r.state = "abandoned_in_engine"
            while self.queue:
                c = self.queue[0]
                d = self.policy(self.metrics_for(c), request_id=c.rid)
                if not d.admitted:
                    break
                self.queue.pop(0)
                self.send(c, now)
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), 0.1)
            except asyncio.TimeoutError:
                pass

    async def run(self, arrivals, wall_s):
        self.t0 = time.time()
        tasks = [asyncio.ensure_future(self.poll()), asyncio.ensure_future(self.gatekeeper())]
        if self.arm == "gate":
            tasks.append(asyncio.ensure_future(self.controller()))
        k = 0
        try:
            while True:
                now = time.time()
                el = now - self.t0
                if el >= wall_s:
                    break
                while k < len(arrivals) and arrivals[k][0] <= el:
                    ta, idx = arrivals[k]
                    src = self.reqs_src[idx]
                    r = Req(f"{self.cfg['cell']}-r{k:05d}", idx, src["p"], src["L"], self.t0 + ta)
                    self.all.append(r)
                    k += 1
                    if self.door and self.door_rejects():
                        r.state = "rejected_by_door"
                        self.rejected += 1
                        self.policy.log.write(dict(at=time.time(), request_id=r.rid, verdict="reject", reason="wait_estimate_exceeded",
                                                   policy=self.policy.name, experiment=self.cfg.get("experiment", "exp02"), run=self.cfg["cell"]))
                        continue
                    self.queue.append(r)
                    self.wake.set()
                nxt = arrivals[k][0] if k < len(arrivals) else wall_s
                await asyncio.sleep(max(0.0, min(0.05, nxt - (time.time() - self.t0))))
        finally:
            t_end = time.time()
            for t in tasks:
                t.cancel()
            for r in list(self.live.values()):
                if r.task is not None and not r.task.done():
                    r.task.cancel()
                    if r.state == "sent":
                        r.state = "censored"
            await asyncio.gather(*tasks, *[r.task for r in self.all if r.task is not None], return_exceptions=True)
            for r in self.queue:
                r.state = "censored_in_proxy"
            self.policy.close()
        return t_end

    def summary(self, t_end):
        return dict(cfg=self.cfg, t0=self.t0, t_end=t_end, requests=[r.rec(self.t0) for r in self.all], metrics=self.mseries,
                    controller=self.ctl, est_events=self.est_events, W_est_total_s=self.W_est_total, door_rejected=self.rejected,
                    decision_log_dropped=self.policy.log.dropped)


def poisson_arrivals(rate, wall_s, n_avail, seed):
    rng = random.Random(seed)
    t, out, k = 0.0, [], 0
    while True:
        t += rng.expovariate(rate)
        if t >= wall_s:
            break
        out.append((t, k % n_avail))
        k += 1
    return out


async def run_cell(cfg, reqs, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    arrivals = poisson_arrivals(cfg["rate"], cfg["wall_s"], len(reqs), cfg["arrival_seed"])
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None, sock_read=600)) as s:
        cell = Cell(cfg, reqs, out_dir, s)
        t_end = await cell.run(arrivals, cfg["wall_s"])
        res = cell.summary(t_end)
        res["n_arrivals"] = len(arrivals)
        res["wrapped"] = len(arrivals) > len(reqs)
    (Path(out_dir) / "cell.json").write_text(json.dumps(res))
    return res


# ------------------------------------------------------------------------------------------------ calibration

async def _one(session, rid, prompt, L, keep_chunks=False, exact_max=False):
    r = Req(rid, -1, len(prompt), L, time.time())
    r.max_tokens = L if exact_max else None
    await stream_one(session, r, prompt, lambda *a: None, keep_chunks=keep_chunks)
    return r


async def calibrate(out_dir, C, seed=0, evict_wall_s=420.0):
    out = dict(C=C)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None, sock_read=900)) as s:
        # warm-up
        await _one(s, f"cal{seed}-warm", unique_prompt(512, 10_000 + seed), 8)
        # 1. prefill, cold then with the same prompt cached
        pf = []
        for j, n in enumerate([2048, 8192, 16384, 8192, 16384]):
            pr = unique_prompt(n, 20_000 + 10 * seed + j)
            cold = await _one(s, f"cal{seed}-cold{j}", pr, 1)
            hot = await _one(s, f"cal{seed}-hot{j}", pr, 1)
            pf.append(dict(tokens=n, ttft_cold=cold.t_first - cold.t_send, ttft_hot=hot.t_first - hot.t_send))
        out["prefill"] = pf
        big = [x for x in pf if x["tokens"] >= 8192]
        out["PF"] = float(np.median([x["tokens"] / x["ttft_cold"] for x in big]))
        out["PF_hot_equiv"] = float(np.median([x["tokens"] / x["ttft_hot"] for x in big]))
        # 2. decode step time near a full cache: 24 requests of 6,000 prompt tokens and 1,600 output tokens (no preemption).
        # They send max_tokens = 1,600 so SGLang's reservation (prompt + 0.7 * min(max_tokens, 4096)) admits all 24 at once.
        n_req = 24
        reqs = [asyncio.ensure_future(_one(s, f"cal{seed}-tau{j}", unique_prompt(6000, 30_000 + 100 * seed + j), 1600,
                                           keep_chunks=True, exact_max=True))
                for j in range(n_req)]
        rs = await asyncio.gather(*reqs)
        out["tau_no_token"] = [r.rid for r in rs if r.t_first is None]
        rs = [r for r in rs if r.t_first is not None]
        if not rs:
            raise RuntimeError(f"no decode-calibration request produced a token: {out['tau_no_token']}")
        # SGLang may hold a few back until the first ones finish; time only the requests that decoded together
        rs = [r for r in rs if r.t_first < min(x.t_last for x in rs)]
        out["tau_n_concurrent"] = len(rs)
        t_all_first = max(r.t_first for r in rs)
        t_first_end = min(r.t_last for r in rs)
        per = []
        for r in rs:
            pts = [(t, n) for t, n in r.chunks if t_all_first + 2 <= t <= t_first_end - 2]
            if len(pts) >= 2:
                per.append((pts[-1][0] - pts[0][0]) / max(pts[-1][1] - pts[0][1], 1))
        out["tau_full"] = float(np.median(per)) if per else None
        out["tau_full_window_s"] = t_first_end - t_all_first
        out["tau_kv_tokens_range"] = [len(rs) * 6000, len(rs) * 7600]
        one = await _one(s, f"cal{seed}-tau1", unique_prompt(2000, 40_000 + seed), 300, keep_chunks=True)
        c = one.chunks
        out["tau_single"] = (c[-1][0] - c[5][0]) / max(c[-1][1] - c[5][1], 1)
        # 3. forced evictions: 30 requests of 5,000 prompt tokens and 3,000 output tokens need 240,000 tokens at the end
        t_ev0 = time.time()
        tasks = [asyncio.ensure_future(_one(s, f"cal{seed}-ev{j:02d}", unique_prompt(5000, 50_000 + 100 * seed + j), 3000, keep_chunks=True))
                 for j in range(30)]
        done, pending = await asyncio.wait(tasks, timeout=evict_wall_s)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        evs = []
        for t in done:
            r = t.result()
            evs.append(dict(rid=r.rid, n=r.n, t_send=r.t_send - t_ev0, t_first=None if r.t_first is None else r.t_first - t_ev0,
                            t_last=None if r.t_last is None else r.t_last - t_ev0, max_gap=r.max_gap,
                            gaps_over_1s=[(round(c2[0] - t_ev0, 3), round(c2[0] - c1[0], 3), c1[1]) for c1, c2 in zip(r.chunks, r.chunks[1:]) if c2[0] - c1[0] > 1.0]))
        out["evict"] = dict(t0=t_ev0, finished=len(done), unfinished=len(pending), requests=evs)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "calibration.json").write_text(json.dumps(out))
    return out


def load_reqs(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
