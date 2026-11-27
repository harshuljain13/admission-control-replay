"""Score the stepped-rate capacity probe C_probe (config/exp03.json amendment_2026_11_27).
python probe.py runs/exp03_vllm/cells_190000/C_probe [lam_formula]"""
import json, sys, statistics as st
d = json.load(open(sys.argv[1] + "/cell.json"))
cfg, reqs, ms = d["cfg"], d["requests"], [m for m in d["metrics"] if m[1] != "error"]
segs, wall = cfg["segments"], cfg["wall_s"]
bounds = [s[0] for s in segs[1:]] + [wall]
def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
cap, ok_so_far = None, True
print(f"{'t0':>5} {'rps':>6} {'n':>4} {'aband':>6} {'ttft50':>7} {'ttft90':>7} {'tpot50':>7} {'wait_mean':>9} {'kv_mean':>7}  pass")
for (t0, rate), t1 in zip(segs, bounds):
    t1j = min(t1, wall - 35)              # a request needs 30 s to be judged abandoned
    win = [r for r in reqs if t0 <= r["t_arr"] < t1j]
    ab = sum(r["state"].startswith("abandoned") for r in win) / max(1, len(win))
    ttft = [r["t_first"] - r["t_arr"] for r in win if r["t_first"] is not None]
    ttft += [30.0] * sum(r["state"].startswith("abandoned") for r in win)
    tpot = [(r["t_last"] - r["t_first"]) / (r["n"] - 1) for r in win if r["t_first"] is not None and r["n"] > 1 and r["t_last"]]
    mm = [m for m in ms if t0 <= m[0] < t1]
    wait = st.mean(m[3] for m in mm if m[3] is not None) if mm else float("nan")
    kv = st.mean(m[1] for m in mm) if mm else float("nan")
    good = ab <= 0.02 and q(ttft, 0.9) <= 5.0
    ok_so_far = ok_so_far and good
    if ok_so_far:
        cap = rate
    print(f"{t0:5.0f} {rate:6.3f} {len(win):4d} {ab:6.3f} {q(ttft, .5):7.2f} {q(ttft, .9):7.2f} {q(tpot, .5):7.3f} {wait:9.1f} {kv:7.2f}  {good}")
print("measured chat capacity (rps):", cap)
if len(sys.argv) > 2 and cap:
    print("as a fraction of the formula lam_cap:", round(cap / float(sys.argv[2]), 3))
