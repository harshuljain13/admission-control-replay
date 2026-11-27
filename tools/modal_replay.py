"""Run the replay on one Modal L40S, on SGLang 0.5.21 or vLLM 0.32.0 (ACR_ENGINE=sglang|vllm), with a budget stop.

    python tools/modal_replay.py check                          # CPU only: image builds, Python and engine import
    python tools/modal_replay.py launch 0 --approved-usd 8.30   # SGLang: setup, smoke test and calibration
    python tools/modal_replay.py launch 1 --approved-usd 8.30   # SGLang: the six core cells of Exp02
    ACR_ENGINE=vllm python tools/modal_replay.py launch --config config/exp03.json --out exp03_vllm \
        --cells T_stock_25 T_gate_25 --approved-usd 50          # named cells (calibrates first if the folder has none)
    python tools/modal_replay.py wait | spend | fetch DIR... | stop

The budget stop has three layers. Before a launch, the larger of Modal's bill for both apps and the wall time of every
past launch at the full rate, plus the stage's worst case, must stay within --approved-usd and the $100 cap; the stage's worst case is its Modal timeout times the
L40S rate with CPU and memory. Modal kills the function at that timeout. Inside it, run.py gets a --deadline ten
minutes earlier and starts no cell that could not finish by then. `spend` quotes Modal's own billing report.
The stage runs as a deployed function, so it survives the local client's 3-hour token expiry.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal
import modal.exception

ENGINE = os.environ.get("ACR_ENGINE", "sglang")
APP = f"acr-{ENGINE}-topic34"
APPS = ("acr-sglang-topic34", "acr-vllm-topic34")
VOLUME = "acr-sglang-runs"
RATE_USD_H = 0.000542 * 3600 + 8 * 0.0000131 * 3600 + 32 * 0.00000222 * 3600   # L40S + 8 cores + 32 GiB = $2.59/h
CAP_USD = 100.0
MARGIN_S = 600
HERE = Path(__file__).resolve().parents[1]
CORE = ["T_stock_25", "T_stock_10", "T_gate_25", "T_known_25", "T_wm10_25", "T_stock_25_AA"]
STAGES = {"0": [["smoke", "--out", "/runs/smoke"], ["calibrate", "--out", "/runs/exp02_sglang"]],
          "1": [["cell", *CORE, "--out", "/runs/exp02_sglang"]]}
STAGE_HOURS = {"0": 0.8, "1": 2.4}
CELL_HOURS, SETUP_HOURS = 0.40, 0.35   # SGLang stage 1 took 2.14 h for calibration and six cells
TIMEOUT_S = int(os.environ.get("ACR_TIMEOUT_S", "3600"))

BASE = {"sglang": "lmsysorg/sglang:v0.5.21-runtime", "vllm": "vllm/vllm-openai:v0.32.0"}[ENGINE]
# Modal's own setup step calls `python`; the vLLM image only has python3.
LINK_PY = ['RUN command -v python >/dev/null || ln -s "$(command -v python3)" /usr/local/bin/python']
image = (modal.Image.from_registry(BASE, setup_dockerfile_commands=LINK_PY).entrypoint([])
         .env({"HF_HOME": "/runs/hf", "PYTHONUNBUFFERED": "1", "ACR_ENGINE": ENGINE})
         .add_local_dir(HERE, "/acr", ignore=[".git", "runs", "**/__pycache__"]))
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)


@app.function(cpu=2, memory=4096, timeout=900)
def check():
    out = subprocess.run(["python3", "-c", f"import sys, {ENGINE}, aiohttp, numpy, torch; print(sys.version.split()[0], "
                          f"{ENGINE}.__version__, torch.__version__, torch.version.cuda)"], capture_output=True, text=True)
    t = subprocess.run(["python3", "-I", "tests/test_offline.py"], cwd="/acr", capture_output=True, text=True)
    return out.stdout + out.stderr[-2000:] + "\ntest_offline:\n" + t.stdout[-1500:] + t.stderr[-1500:]


MIN_FREE_MIB = 44000


@app.function(gpu="L40S", cpu=8, memory=32768, timeout=TIMEOUT_S, volumes={"/runs": vol})
def stage(name, deadline: float):
    import threading
    stop = threading.Event()

    def commit_loop():
        while not stop.wait(120):
            try:
                vol.commit()
            except Exception as exc:
                print("volume commit failed:", exc, flush=True)
    threading.Thread(target=commit_loop, daemon=True).start()
    # Modal L40S containers differ by ~3 GB of free memory; on the low ones SGLang cannot fit the 190k-token pool.
    free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).split()[0])
    print("GPU MiB free:", free, flush=True)
    if ENGINE == "sglang" and free < MIN_FREE_MIB:
        stop.set()
        return [f"low-memory GPU: {free} MiB free, need {MIN_FREE_MIB}"]
    codes = []
    try:
        for args in (STAGES[name] if isinstance(name, str) else name):
            exe = "python3" if ENGINE == "sglang" else "vllm"
            cmd = ["python3", "-I", "run.py", *args, "--engine", ENGINE, "--vllm", exe, "--deadline", str(deadline)]
            print("+", " ".join(cmd), flush=True)
            codes.append(subprocess.run(cmd, cwd="/acr").returncode)
            if codes[-1] != 0 or time.time() > deadline:
                break
    finally:
        stop.set()
        vol.commit()
    return codes


def sh(cmd, **kw):
    env = dict(os.environ, MODAL_IMAGE_BUILDER_VERSION="2025.06", **kw.pop("env", {}))
    return subprocess.run(cmd, env=env, check=True, text=True, **kw)


def billed_usd(since="2026-11-22"):
    out = sh(["modal", "billing", "report", "--start", since, "-r", "d", "--json"], capture_output=True).stdout
    rows = json.loads(out[out.index("["):]) if "[" in out else []
    return sum(float(r["cost"]) for r in rows if r.get("description") in APPS)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["check", "launch", "wait", "spend", "fetch", "stop"])
    ap.add_argument("stage", nargs="?", help="a named SGLang stage (0, 1), or with fetch: the folders to fetch")
    ap.add_argument("more", nargs="*")
    ap.add_argument("--cells", nargs="+", default=None, help="launch these cells instead of a named stage")
    ap.add_argument("--config", default="config/exp03.json")
    ap.add_argument("--out", default=None, help="run folder on the volume, e.g. exp03_vllm")
    ap.add_argument("--approved-usd", type=float, default=None, help="spend Harshul approved for this app, all stages so far")
    ap.add_argument("--to", default=str(HERE / "runs"))
    a = ap.parse_args(argv)
    state_path = HERE / "runs" / "modal_state.json"
    state_path.parent.mkdir(exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(launches=[])

    if a.command == "check":
        with modal.enable_output(), app.run():
            print(check.remote())
    elif a.command == "launch":
        if a.approved_usd is None or (a.cells is None) == (a.stage is None):
            ap.error("launch needs --approved-usd and either a stage or --cells with --out")
        if a.cells is not None:
            if not a.out:
                ap.error("--cells needs --out")
            sys.path.insert(0, str(HERE))
            import run
            run.cell_cfgs(run.load_rule(HERE / a.config), {t: dict(lam_cap=1.0) for t in ("thinking_long", "chat")},
                          1, dict(PF=1.0, tau_full=1.0), cells=a.cells)   # exits on an unknown cell name
            job = [["cell", *a.cells, "--config", a.config, "--out", f"/runs/{a.out}"]]
            need, label = SETUP_HOURS + CELL_HOURS * len(a.cells), f"{ENGINE}:{a.out}:{','.join(a.cells)}"
        else:
            if ENGINE != "sglang" or a.stage not in STAGES:
                ap.error(f"named stages are SGLang only: {sorted(STAGES)}")
            job, need, label = a.stage, STAGE_HOURS[a.stage], a.stage
        limit = min(a.approved_usd, CAP_USD)
        # Billing posts hours late, so also count each past launch's wall time at the full rate.
        est = sum(min(x["timeout_s"], x.get("t_end", time.time()) - x["t"]) / 3600 * RATE_USD_H for x in state["launches"])
        spent = max(billed_usd(), est)
        pending = [x for x in state["launches"] if x.get("open")]
        if pending:
            sys.exit(f"stage {pending[-1]['stage']} is still open; run `wait` first")
        room = limit - spent
        hours = min(room / RATE_USD_H, need * 1.3 + 0.25)
        if hours < need:
            sys.exit(f"budget stop: spent ${spent:.2f}, limit ${limit:.2f}, room for {hours:.2f} h, stage needs {need:.2f} h")
        timeout_s = int(hours * 3600)
        sh(["modal", "deploy", __file__], env=dict(ACR_TIMEOUT_S=str(timeout_s), ACR_ENGINE=ENGINE))
        fn = modal.Function.from_name(APP, "stage")
        deadline = time.time() + timeout_s - MARGIN_S
        call = fn.spawn(job, deadline)
        state["launches"].append(dict(stage=label, app=APP, call_id=call.object_id, t=time.time(), timeout_s=timeout_s,
                                      worst_usd=round(hours * RATE_USD_H, 2), billed_before=round(spent, 2), open=True))
        state_path.write_text(json.dumps(state, indent=1))
        print(f"stage {label} launched: {call.object_id}, timeout {timeout_s / 3600:.2f} h, worst case "
              f"${hours * RATE_USD_H:.2f} on top of ${spent:.2f} billed")
    elif a.command == "wait":
        x = state["launches"][-1]
        try:
            print("exit codes:", modal.FunctionCall.from_id(x["call_id"]).get(timeout=270))
            x["open"], x["t_end"] = False, time.time()
        except modal.exception.FunctionTimeoutError:   # a TimeoutError subclass, so it comes first
            print("stage hit its Modal timeout (the budget stop)")
            x["open"], x["t_end"] = False, time.time()
        except TimeoutError:
            print("still running", round((time.time() - x["t"]) / 3600, 2), "h")
        state_path.write_text(json.dumps(state, indent=1))
    elif a.command == "spend":
        est = sum(min(x["timeout_s"], x.get("t_end", time.time()) - x["t"]) / 3600 * RATE_USD_H for x in state["launches"])
        print(f"billed to {', '.join(APPS)} since 2026-11-22: ${billed_usd():.2f} (the latest hour posts late); "
              f"wall-time estimate ${est:.2f}; cap ${CAP_USD:.0f}")
    elif a.command == "fetch":
        for d in ([a.stage, *a.more] if a.stage else ["smoke", "exp02_sglang"]):
            sh(["modal", "volume", "get", "--force", VOLUME, d, a.to])
    elif a.command == "stop":
        sh(["modal", "app", "stop", "-y", APP])
        for x in state["launches"]:
            if x["open"]:
                x["open"], x["t_end"] = False, time.time()
        state_path.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    print(dt.datetime.now().isoformat(timespec="seconds"), file=sys.stderr)
    main()
