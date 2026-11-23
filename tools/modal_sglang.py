"""Run the SGLang replay on one Modal L40S, with a budget stop.

    python tools/modal_sglang.py check                          # CPU only: image builds, Python and sglang import
    python tools/modal_sglang.py launch 0 --approved-usd 8.30   # setup, smoke test and calibration
    python tools/modal_sglang.py launch 1 --approved-usd 8.30   # the six core cells of Exp02
    python tools/modal_sglang.py wait | spend | fetch | stop

The budget stop has three layers. Before a launch, the spend Modal has billed to this app so far plus the stage's
worst case must stay within --approved-usd and the $20 cap; the stage's worst case is its Modal timeout times the
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

APP = "acr-sglang-topic34"
VOLUME = "acr-sglang-runs"
RATE_USD_H = 0.000542 * 3600 + 8 * 0.0000131 * 3600 + 32 * 0.00000222 * 3600   # L40S + 8 cores + 32 GiB = $2.59/h
CAP_USD = 20.0
MARGIN_S = 600
HERE = Path(__file__).resolve().parents[1]
CORE = ["T_stock_25", "T_stock_10", "T_gate_25", "T_known_25", "T_wm10_25", "T_stock_25_AA"]
STAGES = {"0": [["smoke", "--out", "/runs/smoke"], ["calibrate", "--out", "/runs/exp02_sglang"]],
          "1": [["cell", *CORE, "--out", "/runs/exp02_sglang"]]}
STAGE_HOURS = {"0": 0.8, "1": 2.4}
TIMEOUT_S = int(os.environ.get("ACR_TIMEOUT_S", "3600"))

image = (modal.Image.from_registry("lmsysorg/sglang:v0.5.21-runtime")
         .env({"HF_HOME": "/runs/hf", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(HERE, "/acr", ignore=[".git", "runs", "**/__pycache__"]))
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)


@app.function(cpu=2, memory=4096, timeout=900)
def check():
    out = subprocess.run(["python3", "-c", "import sys, sglang, aiohttp, numpy, torch; print(sys.version.split()[0], "
                          "sglang.__version__, torch.__version__, torch.version.cuda)"], capture_output=True, text=True)
    t = subprocess.run(["python3", "-I", "tests/test_offline.py"], cwd="/acr", capture_output=True, text=True)
    return out.stdout + out.stderr[-2000:] + "\ntest_offline:\n" + t.stdout[-1500:] + t.stderr[-1500:]


MIN_FREE_MIB = 44000


@app.function(gpu="L40S", cpu=8, memory=32768, timeout=TIMEOUT_S, volumes={"/runs": vol})
def stage(name: str, deadline: float):
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
    if free < MIN_FREE_MIB:
        stop.set()
        return [f"low-memory GPU: {free} MiB free, need {MIN_FREE_MIB}"]
    codes = []
    try:
        for args in STAGES[name]:
            cmd = ["python3", "-I", "run.py", *args, "--engine", "sglang", "--vllm", "python3", "--deadline", str(deadline)]
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
    return sum(float(r["cost"]) for r in rows if r.get("description") == APP)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["check", "launch", "wait", "spend", "fetch", "stop"])
    ap.add_argument("stage", nargs="?", choices=sorted(STAGES))
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
        if a.stage is None or a.approved_usd is None:
            ap.error("launch needs a stage and --approved-usd")
        limit = min(a.approved_usd, CAP_USD)
        # Billing posts hours late, so also count each past launch's wall time at the full rate.
        est = sum(min(x["timeout_s"], x.get("t_end", time.time()) - x["t"]) / 3600 * RATE_USD_H for x in state["launches"])
        spent = max(billed_usd(), est)
        pending = [x for x in state["launches"] if x.get("open")]
        if pending:
            sys.exit(f"stage {pending[-1]['stage']} is still open; run `wait` first")
        room = limit - spent
        hours = min(room / RATE_USD_H, STAGE_HOURS[a.stage] * 1.6 + 0.25)
        if hours < STAGE_HOURS[a.stage]:
            sys.exit(f"budget stop: billed ${spent:.2f}, limit ${limit:.2f}, room for {hours:.2f} h, stage needs "
                     f"{STAGE_HOURS[a.stage]} h")
        timeout_s = int(hours * 3600)
        sh(["modal", "deploy", __file__], env=dict(ACR_TIMEOUT_S=str(timeout_s)))
        fn = modal.Function.from_name(APP, "stage")
        deadline = time.time() + timeout_s - MARGIN_S
        call = fn.spawn(a.stage, deadline)
        state["launches"].append(dict(stage=a.stage, call_id=call.object_id, t=time.time(), timeout_s=timeout_s,
                                      worst_usd=round(hours * RATE_USD_H, 2), billed_before=round(spent, 2), open=True))
        state_path.write_text(json.dumps(state, indent=1))
        print(f"stage {a.stage} launched: {call.object_id}, timeout {timeout_s / 3600:.2f} h, worst case "
              f"${hours * RATE_USD_H:.2f} on top of ${spent:.2f} billed")
    elif a.command == "wait":
        x = state["launches"][-1]
        try:
            print("exit codes:", modal.FunctionCall.from_id(x["call_id"]).get(timeout=270))
            x["open"], x["t_end"] = False, time.time()
        except TimeoutError:
            print("still running", round((time.time() - x["t"]) / 3600, 2), "h")
        state_path.write_text(json.dumps(state, indent=1))
    elif a.command == "spend":
        print(f"billed to {APP} since 2026-11-22: ${billed_usd():.2f} (the latest hour posts late)")
    elif a.command == "fetch":
        sh(["modal", "volume", "get", "--force", VOLUME, "smoke", a.to])
        sh(["modal", "volume", "get", "--force", VOLUME, "exp02_sglang", a.to])
    elif a.command == "stop":
        sh(["modal", "app", "stop", "-y", APP])
        for x in state["launches"]:
            if x["open"]:
                x["open"], x["t_end"] = False, time.time()
        state_path.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    print(dt.datetime.now().isoformat(timespec="seconds"), file=sys.stderr)
    main()
