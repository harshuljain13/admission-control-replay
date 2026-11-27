"""Offline checks with no server at all: the data against its manifest, the vendored AdmitPerf files against their
recorded hashes, the arms' arithmetic, the rebuild accounting, the vLLM command line and the plugin's entry point.

    python -I tests/test_offline.py        (or: pytest tests)
"""
import hashlib
import json
import sys
import tempfile
from importlib.metadata import distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))
import measure  # noqa: E402
import replay  # noqa: E402
import run  # noqa: E402
import server  # noqa: E402

RULE = json.loads((ROOT / "config" / "exp02.json").read_text())


def test_data_matches_manifest():
    man = json.loads((ROOT / "data" / "manifest.json").read_text())
    t = man["traces"]["thinking_long"]
    raw = (ROOT / "data" / "thinking_long.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == t["out_sha256"]
    rows = replay.load_reqs(ROOT / "data" / "thinking_long.jsonl")
    assert len(rows) == t["written"] == 622
    for k, r in enumerate(rows):
        assert r["i"] == k
        assert set(r) == {"i", "i_exp01", "p", "L", "L_raw", "h"}   # lengths and block hashes only, no text
        assert r["L_raw"] >= 2000 and r["L"] == min(r["L_raw"], 16000) and r["p"] + r["L"] <= 40896
        assert len(r["h"]) == -(-r["p"] // 16)
    assert abs(sum(r["p"] for r in rows) / len(rows) - t["p_mean"]) < 1e-6
    mt = sum(r["p"] * r["L"] + r["L"] ** 2 / 2 for r in rows) / len(rows)
    assert abs(mt - t["memtime_tokens_steps_mean"]) < 1e-3
    assert man["source"]["licence"] == "Apache-2.0"


def test_vendored_admitperf_unchanged():
    rec = json.loads((ROOT / "vendor" / "ADMITPERF_COMMIT.json").read_text())
    for f, sha in rec["sha256"].items():
        assert hashlib.sha256((ROOT / "vendor" / "admitperf" / f).read_bytes()).hexdigest() == sha, f
    assert (ROOT / "vendor" / "ADMITPERF_LICENSE").read_text().startswith("MIT License")


def test_prompts_share_prefixes():
    pr = replay.Prompts("thinking_long")
    a = dict(p=40, h=[1, 2, 3])
    b = dict(p=48, h=[1, 2, 9])
    ia, ib = pr.ids(a), pr.ids(b)
    assert len(ia) == 40 and len(ib) == 48 and ia[:32] == ib[:32] and ia[32:] != ib[32:40]


def test_oracle_peak():
    assert replay.oracle_peak([]) == 0.0
    assert replay.oracle_peak([(10, 100), (5, 50)]) == max(150, 150 + 2 * 5, 100 + 10)
    assert replay.oracle_peak([(1, 0), (1, 0), (100, 0)]) == 100


def _policy(cls, tmp, **kw):
    return cls(log=str(Path(tmp) / "d.jsonl"), experiment="t", run="t", notes="", provenance={}, **kw)


def test_arms_decide():
    with tempfile.TemporaryDirectory() as tmp:
        base = {"acr:C": 1000.0, "acr:used_est": 500.0, "acr:cand_prompt": 100.0, "acr:preempted_waiting": 0.0, "acr:h": 0.0,
                "acr:oracle_peak": 900.0}
        stock = _policy(replay.Stock, tmp)
        gate = _policy(replay.BudgetGate, tmp, alpha=0.01, K=200.0, window_s=2.0)
        wm = _policy(replay.Watermark, tmp, frac=0.10)
        known = _policy(replay.KnownLength, tmp)
        assert stock(dict(base, **{"acr:used_est": 5000.0}), request_id="a").admitted
        assert gate(base, request_id="b").admitted
        assert not gate(dict(base, **{"acr:h": 401.0}), request_id="c").admitted
        assert not gate(dict(base, **{"acr:preempted_waiting": 1.0}), request_id="d").admitted
        assert wm(base, request_id="e").admitted
        assert not wm(dict(base, **{"acr:used_est": 801.0}), request_id="f").admitted
        assert known(base, request_id="g").admitted
        assert not known(dict(base, **{"acr:oracle_peak": 1001.0}), request_id="h").admitted
        for p in (stock, gate, wm, known):
            p.close()
        assert len((Path(tmp) / "d.jsonl").read_text().splitlines()) == 8


def test_rebuild_events():
    plog = [dict(ev="preempt", t=10.0, rid="cmpl-T_x-r00001-abc", tokens=5000, computed=4990),
            dict(ev="lookup", t=12.0, rid="cmpl-T_x-r00001-abc", hit=0, tokens=5000),
            dict(ev="preempt", t=11.0, rid="cmpl-T_y-r00001", tokens=99, computed=99),
            dict(ev="preempt", t=20.0, rid="cmpl-T_x-r00002", tokens=300, computed=300)]
    evs = measure.rebuild_events(plog, "T_x", 0.0, 100.0)
    by = {e["rid"]: e for e in evs}
    assert set(by) == {"T_x-r00001", "T_x-r00002"}
    assert by["T_x-r00001"]["rebuilt"] == 5000 and by["T_x-r00001"]["resumed"]
    assert by["T_x-r00002"]["rebuilt"] == 0 and not by["T_x-r00002"]["resumed"]


def test_server_command_matches_exp02():
    cmd = server.serve_command("Qwen/Qwen3-8B", RULE["model"]["revision"], RULE["engine"]["kv_tokens"], 8000)
    s = " ".join(cmd)
    assert "--kv-cache-memory-bytes 28016640000" in s   # 190,000 tokens * 147,456 bytes
    assert "--no-enable-prefix-caching" in s and "--max-model-len 40960" in s and "--block-size 16" in s
    assert "--revision " + RULE["model"]["revision"] in s
    env = server.server_env("/tmp/p.jsonl")
    assert env["ACR_PREEMPT_LOG"] == "/tmp/p.jsonl" and str(server.PLUGIN_DIR) in env["PYTHONPATH"]


def test_plugin_entry_point_is_discoverable():
    eps = [ep for d in distributions(path=[str(server.PLUGIN_DIR)]) for ep in d.entry_points if ep.group == "vllm.general_plugins"]
    assert [ep.value for ep in eps] == ["acr_preemptlog:register"]


def test_cell_configs_and_rates():
    cal = dict(PF=10000.0, tau_full=0.05)
    rates = run.rates_from(cal, 190000)
    r = rates["thinking_long"]
    assert abs(r["lam_kv"] - 190000 / (0.05 * 42859083.11334405)) < 1e-12 and r["lam_cap"] == min(r["lam_kv"], r["lam_pf"])
    cfgs = run.cell_cfgs(RULE, rates, 190000, cal)
    assert [c["cell"] for c in cfgs] == [c["name"] for c in RULE["cells"]]
    g = next(c for c in cfgs if c["cell"] == "T_gate_25")
    assert g["K"] == 0.2 * 190000 and g["rate"] == 2.5 * r["lam_cap"] and g["wall_s"] == 1260 and g["count_window_s"] == 720
    one = run.cell_cfgs(RULE, rates, 190000, cal, cells=["T_known_25", "T_stock_25"])
    assert [c["cell"] for c in one] == ["T_known_25", "T_stock_25"]


def test_chat_data_matches_manifest():
    t = json.loads((ROOT / "data" / "manifest.json").read_text())["traces"]["chat"]
    raw = (ROOT / "data" / "chat.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == t["out_sha256"] and len(replay.load_reqs(ROOT / "data" / "chat.jsonl")) == 1500


def test_watermark_arms():
    cls, params = replay.arm_policy("wm20")
    assert params == dict(frac=0.20) and cls.name == "watermark_20pct" and issubclass(cls, replay.Watermark)
    assert replay.arm_policy("wm10") == (replay.Watermark, dict(frac=0.10))
    with tempfile.TemporaryDirectory() as tmp:
        wm = _policy(cls, tmp, **params)
        base = {"acr:C": 1000.0, "acr:used_est": 650.0, "acr:cand_prompt": 100.0, "acr:preempted_waiting": 0.0}
        assert wm(base, request_id="a").admitted and not wm(dict(base, **{"acr:used_est": 701.0}), request_id="b").admitted
        wm.close()


def test_piecewise_arrivals():
    arr = replay.piecewise_arrivals([[0, 1.0], [300, 4.0], [600, 1.0]], 1260, 622, 3450)
    n = [sum(1 for t, _ in arr if a <= t < b) for a, b in ((0, 300), (300, 600), (600, 1260))]
    assert 240 < n[0] < 360 and 1080 < n[1] < 1320 and 560 < n[2] < 760
    assert all(a[0] < b[0] for a, b in zip(arr, arr[1:])) and [k for _, k in arr[:3]] == [0, 1, 2]
    assert replay.piecewise_arrivals([[0, 2.0]], 500, 9, 7) == replay.poisson_arrivals(2.0, 500, 9, 7)


def test_exp03_cells():
    rule = json.loads((ROOT / "config" / "exp03.json").read_text())
    cal = dict(PF=10000.0, tau_full=0.05)
    rates = run.rates_from(cal, 190000)
    cfgs = {c["cell"]: c for c in run.cell_cfgs(rule, rates, 190000, cal)}
    assert len(rule["steps"]["2"]) == 16 and len(rule["steps"]["3"]) == 4 and set(rule["core_cells"]) <= set(cfgs)
    old = {c["cell"]: c for c in run.cell_cfgs(RULE, rates, 190000, cal)}
    for n in ("T_stock_25", "T_gate_25", "T_known_25", "T_wm10_25"):
        a, b = cfgs[n], old[n]
        assert all(a[k] == b[k] for k in ("arm", "trace", "load", "rate", "arrival_seed", "K", "alpha", "window_s", "wall_s"))
    assert cfgs["T_gate_25"]["experiment"] == "exp03" and cfgs["T_gate_40"]["rate"] == 4.0 * rates["thinking_long"]["lam_cap"]
    assert cfgs["C_wm20_40"]["rate"] == 4.0 * rates["chat"]["lam_cap"] and cfgs["C_wm20_40"]["segments"] is None
    cap = rates["thinking_long"]["lam_cap"]
    assert cfgs["T_gate_surge"]["segments"] == [[0, cap], [300, 4.0 * cap], [600, cap]]
    assert cfgs["T_gate_surge_s2"]["arrival_seed"] == 4450 and all(cfgs[n]["arm"] in ("stock", "gate", "wm10", "wm20", "known")
                                                                     for n in cfgs)
    ccap = rates["chat"]["lam_cap"]
    assert cfgs["C_probe"]["segments"][0] == [0, 0.15 * ccap] and cfgs["C_probe"]["segments"][-1] == [1080, 0.5 * ccap]
    assert cfgs["T_stock_25_AA"]["rate"] == cfgs["T_stock_25"]["rate"] and cfgs["T_stock_25_AA"]["arrival_seed"] == 3425
    probe = dict(rule, cells=[dict(name="Cm_gate_25", arm="gate", trace="chat", load=2.5, arrival_seed=3525, cap_rps=0.7)])
    assert run.cell_cfgs(probe, rates, 190000, cal)[0]["rate"] == 2.5 * 0.7


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
