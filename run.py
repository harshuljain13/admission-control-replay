"""Command-line entry point. Runs on one Linux machine with one NVIDIA GPU and vLLM 0.32.0 installed from
requirements-vllm.lock.txt. Each command starts the vLLM server, does its work, and stops the server.

    python -I run.py smoke --out runs/smoke             # short calibration and four 2-minute cells, plumbing only
    python -I run.py calibrate --out runs/exp02         # calibration only
    python -I run.py cell T_stock_25 --out runs/exp02   # one cell (calibrates first if the folder has no calibration)
    python -I run.py all --out runs/exp02               # every cell of config/exp02.json, with the two in-run checks
    python -I harness/analyze.py runs/exp02             # gates and summary.json

Cells already done in the run folder are skipped by `all`, so an interrupted run can be resumed.

    python -I run.py all --engine sglang --vllm /usr/bin/python3 --out runs/exp02_sglang   # same cells on SGLang 0.5.21

--deadline (Unix time) is the budget stop for paid machines: no cell starts unless it can end, drain included, before it.
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "harness"))
sys.path.insert(0, str(HERE / "vendor"))
import measure  # noqa: E402
import replay  # noqa: E402
import server  # noqa: E402

DATA = HERE / "data"
ENGINES = {"vllm": "vLLM 0.32.0", "sglang": "SGLang 0.5.21"}
CELL_DRAIN_S = 300.0


# ------------------------------------------------------------------------------------------- config

def load_rule(path):
    return json.loads(Path(path).read_text())


def rates_from(calib, C, manifest_path=DATA / "manifest.json"):
    """Capacity in requests per second for each trace: the smaller of the KV limit and the prefill limit."""
    man = json.loads(Path(manifest_path).read_text())["traces"]
    out = {}
    for tr in man:
        lam_kv = C / (calib["tau_full"] * man[tr]["memtime_tokens_steps_mean"])
        lam_pf = calib["PF"] / man[tr]["p_mean"]
        out[tr] = dict(lam_kv=lam_kv, lam_pf=lam_pf, lam_cap=min(lam_kv, lam_pf))
    return out


def cell_cfgs(rule, rates, C, calib, cells=None, wall_s=None, count_window_s=None, gpu=None, experiment=None, engine="vllm"):
    experiment = experiment or rule.get("id", "exp02")
    out = []
    for c in rule["cells"]:
        if cells is not None and c["name"] not in cells:
            continue
        cap = c.get("cap_rps") or rates[c["trace"]]["lam_cap"]   # a measured capacity, where the config gives one
        segs = [[t, x * cap] for t, x in c["surge"]] if c.get("surge") else None
        out.append(dict(cell=c["name"], arm=c["arm"], door=c.get("door", False), trace=c["trace"], load=c["load"],
                        rate=c["load"] * cap, segments=segs, surge=c.get("surge"), arrival_seed=c["arrival_seed"],
                        wall_s=wall_s or rule["cell"]["wall_s"], C=C, PF=calib["PF"], TAU=calib["tau_full"],
                        alpha=rule["gate"]["alpha"], K=rule["gate"]["K_fraction_of_C"] * C, window_s=rule["gate"]["window_s"],
                        count_window_s=count_window_s or rule["cell"]["count_window_s"], experiment=experiment, gpu=gpu,
                        prefix_caching="off", engine=ENGINES[engine]))
    if cells is not None:
        unknown = set(cells) - {c["cell"] for c in out}
        if unknown:
            sys.exit(f"unknown cell(s): {sorted(unknown)}; the config lists {[c['name'] for c in rule['cells']]}")
        out.sort(key=lambda c: cells.index(c["cell"]))
    return out


def gpu_name():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


# ------------------------------------------------------------------------------------------- one run folder

class RunDir:
    def __init__(self, out, C, model, revision, engine="vllm"):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        p = self.out / "state.json"
        fresh = dict(engine=ENGINES[engine], model=model, revision=revision, regimes=[], notes=[])
        self.state = json.loads(p.read_text()) if p.exists() else fresh
        self.regime = next((r for r in self.state["regimes"] if r["C"] == C), None)
        if self.regime is None:
            self.regime = dict(C=C, engine_reported_kv_tokens=None, cells={})
            self.state["regimes"].append(self.regime)

    def save(self):
        tmp = self.out / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.out / "state.json")

    def note(self, msg):
        print(msg, flush=True)
        self.state["notes"].append(msg)
        self.save()


def reset_path(engine):
    return "/flush_cache" if engine == "sglang" else "/reset_prefix_cache"


def run_one(rd, cfg, reqs, base_url, engine="vllm"):
    C = rd.regime["C"]
    print(f"cell {cfg['cell']}: arm {cfg['arm']} door {cfg['door']} rate {cfg['rate']:.4f}/s for {cfg['wall_s']} s", flush=True)
    cdir = rd.out / f"cells_{C}" / cfg["cell"]
    try:
        res = asyncio.run(replay.run_cell(cfg, reqs, cdir))
    except Exception as exc:
        rd.note(f"{cfg['cell']} failed: {type(exc).__name__} {str(exc)[:300]}")
        rd.regime["cells"][cfg["cell"]] = dict(failed=True)
        rd.regime[f"reset_after_{cfg['cell']}"] = server.idle_and_reset(base_url, replay.parse_metrics,
                                                                        reset_path=reset_path(engine))
        rd.save()
        return None
    rd.regime[f"reset_after_{cfg['cell']}"] = server.idle_and_reset(base_url, replay.parse_metrics, reset_path=reset_path(engine))
    plog = measure.read_jsonl(rd.out / "preempt.jsonl")
    m = measure.cell_measures(res, plog, cfg["PF"], C)
    (cdir / "measures.json").write_text(json.dumps(m, indent=1))
    rd.regime["cells"][cfg["cell"]] = dict(done=True)
    rd.save()
    keys = ("arrivals", "counted", "finished", "good", "goodput_rps", "rebuild_share", "kv_ge_095_share", "preemptions_log", "censored_share")
    print(json.dumps({k: m[k] for k in keys}), flush=True)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description="Admission-control replay against vLLM 0.32.0 on one local GPU.")
    ap.add_argument("command", choices=["smoke", "calibrate", "cell", "all"])
    ap.add_argument("cells", nargs="*", help="cell names for the `cell` command, e.g. T_stock_25 T_gate_25")
    ap.add_argument("--out", required=True, help="run folder; reuse it across commands so cells share one calibration")
    ap.add_argument("--config", default=str(HERE / "config" / "exp02.json"))
    ap.add_argument("--model", default=None, help="Hugging Face repo or a local folder (default: the config's Qwen/Qwen3-8B)")
    ap.add_argument("--revision", default=None, help="model revision (default: the config's pinned commit; ignored for a local folder)")
    ap.add_argument("--kv-tokens", type=int, default=None, help="KV capacity C in tokens (default: the config's 190000)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--vllm", default="vllm", help="the vllm executable (with --engine sglang: the Python that has sglang)")
    ap.add_argument("--engine", choices=sorted(ENGINES), default="vllm")
    ap.add_argument("--deadline", type=float, default=None,
                    help="Unix time; no cell starts that could end after it (budget stop)")
    ap.add_argument("--recalibrate", action="store_true", help="calibrate again even if the run folder has a calibration")
    args = ap.parse_args(argv)
    if args.command == "cell" and not args.cells:
        ap.error("name at least one cell, e.g. `python -I run.py cell T_stock_25 --out runs/exp02`")
    if args.command != "cell" and args.cells:
        ap.error("cell names go with the `cell` command only")

    rule = load_rule(args.config)
    smoke = args.command == "smoke"
    C = args.kv_tokens or rule["engine"]["kv_tokens"]
    model = args.model or rule["model"]["repo"]
    revision = None if Path(model).exists() else (args.revision or rule["model"]["revision"])
    rd = RunDir(args.out, C, model, revision, args.engine)
    replay.ENGINE = args.engine
    gpu = gpu_name()
    rd.state.setdefault("gpus", [])
    if gpu and gpu not in rd.state["gpus"]:
        rd.state["gpus"].append(gpu)
    reqs = {tr: replay.load_reqs(DATA / f"{tr}.jsonl") for tr in ("thinking_long", "chat") if (DATA / f"{tr}.jsonl").exists()}
    replay.URL = base_url = f"http://127.0.0.1:{args.port}"

    t_start = time.time()
    srv, log_path = server.start(rd.out, model, revision, C, args.port, prefix_caching=False, vllm=args.vllm, engine=args.engine)
    if srv is None:
        rd.note(f"server did not start with KV {C} tokens; see {log_path}")
        sys.exit(1)
    try:
        rd.regime["engine_reported_kv_tokens"] = server.engine_kv_tokens(log_path)
        print("engine reports KV tokens:", rd.regime["engine_reported_kv_tokens"], flush=True)
        kv = rd.regime["engine_reported_kv_tokens"]
        if kv is None or kv < 0.995 * C:
            rd.note(f"{args.engine} allocated {kv} KV tokens, below C={C}; memory, not the configured size, set the pool, "
                    "so this run would not match the others")
            sys.exit(1)
        if not server.plugin_loaded(rd.out / "preempt.jsonl", t_start):
            rd.note("the preemption plugin did not load, so rebuild measures would be empty; check that vllm runs from the "
                    "environment where requirements-vllm.lock.txt was installed")
            sys.exit(1)
        rd.save()

        if args.recalibrate or "calibration" not in rd.regime or args.command == "calibrate" or smoke:
            evict = rule["smoke"]["evict_wall_s"] if smoke else 420.0
            cal = asyncio.run(replay.calibrate(rd.out / f"calib_{C}", C, seed=1, evict_wall_s=evict))
            rd.regime["calibration"] = {k: v for k, v in cal.items() if k != "evict"}
            rd.regime["rates"] = rates_from(cal, C)
            rd.regime["reset_after_calibration"] = server.idle_and_reset(base_url, replay.parse_metrics,
                                                                         reset_path=reset_path(args.engine))
            rd.save()
            print("calibration:", json.dumps(rd.regime["calibration"])[:1500], flush=True)
            print("rates:", json.dumps(rd.regime["rates"]), flush=True)
        if args.command == "calibrate":
            return
        cal, rates = rd.regime["calibration"], rd.regime["rates"]

        if smoke:
            cfgs = cell_cfgs(rule, rates, C, cal, cells=rule["smoke"]["cells"], wall_s=rule["smoke"]["wall_s"],
                             count_window_s=rule["smoke"]["count_window_s"], gpu=gpu, experiment="exp02-smoke",
                             engine=args.engine)
        elif args.command == "cell":
            cfgs = cell_cfgs(rule, rates, C, cal, cells=args.cells, gpu=gpu, engine=args.engine)
        else:
            cfgs = cell_cfgs(rule, rates, C, cal, gpu=gpu, engine=args.engine)

        in_run = args.command == "all"
        n_core = len(rule["core_cells"])
        g3 = rd.regime.get("g3_in_run") or {}
        if in_run and "retry_factor" in g3:   # resuming after the G3 check asked for its one gain change
            base = next(c for c in cfgs if c["cell"] == rule["in_run"]["g3_cell"])
            cfgs.insert(n_core, dict(base, cell=rule["in_run"]["g3_retry_cell"], K=base["K"] * g3["retry_factor"]))
        i = 0
        while i < len(cfgs):
            cfg = cfgs[i]
            i += 1
            if in_run and rd.regime["cells"].get(cfg["cell"], {}).get("done"):
                print(f"{cfg['cell']} already done in {rd.out}; skipped", flush=True)
                m_path = rd.out / f"cells_{C}" / cfg["cell"] / "measures.json"
                m = json.loads(m_path.read_text())
            else:
                if args.deadline is not None and time.time() + cfg["wall_s"] + CELL_DRAIN_S > args.deadline:
                    rd.note(f"budget stop: {cfg['cell']} could not finish before the deadline; remaining cells not run")
                    break
                m = run_one(rd, cfg, reqs[cfg["trace"]], base_url, args.engine)
                if m is None:
                    continue
            if not in_run:
                continue
            # in-run check 1: G1 on the first cell; if it fails the run stops there
            if cfg["cell"] == rule["in_run"]["g1_cell"]:
                g1 = m["kv_ge_095_share"] >= rule["gates"]["G1"]["kv_share_min"] and (m["preemptions_metric"] or 0) > 0
                rd.regime["g1_in_run"] = dict(pass_=g1, kv_ge_095_share=m["kv_ge_095_share"], preemptions_metric=m["preemptions_metric"])
                rd.save()
                if not g1:
                    rd.note("G1 failed in run; remaining cells not run, as the config says")
                    break
            # in-run check 2: change the gain once if the gate misses G3
            if cfg["cell"] == rule["in_run"]["g3_cell"] and "g3_in_run" not in rd.regime:
                sp = rd.out / f"cells_{C}" / rule["in_run"]["g1_cell"] / "measures.json"
                sm = json.loads(sp.read_text()) if sp.exists() else None
                share_ok = m["rebuild_share"] is not None and m["rebuild_share"] <= rule["gates"]["G3"]["share_max"]
                gp_ok = sm is not None and m["goodput_rps"] >= rule["gates"]["G3"]["goodput_vs_stock_min"] * sm["goodput_rps"]
                rd.regime["g3_in_run"] = dict(share_ok=share_ok, goodput_ok=gp_ok)
                if not (share_ok and gp_ok):
                    factor = rule["in_run"]["gain_up"] if not share_ok else rule["in_run"]["gain_down"]
                    cfgs.insert(n_core, dict(cfg, cell=rule["in_run"]["g3_retry_cell"], K=cfg["K"] * factor))
                    rd.regime["g3_in_run"]["retry_factor"] = factor
                rd.save()
    finally:
        server.killpg(srv)
        rd.save()


if __name__ == "__main__":
    main()
