"""Apply the Exp03 pre-registered decision rule (config/exp03.json "decision") to rescore.py's holds.json.
python decide.py holds.json [suffix] [chat_prefix, e.g. Cm for the re-calibrated chat cells] [p99]"""
import json, sys
holds = {c["cell"]: c for c in json.load(open(sys.argv[1]))}
SUF = sys.argv[2] if len(sys.argv) > 2 else ""
CH = sys.argv[3] if len(sys.argv) > 3 else "C"
HOLD = "hold_full_p99" if len(sys.argv) > 4 and sys.argv[4] == "p99" else "hold_full_p90"
SETTINGS = {"T 2.5x": ("T", "25"), "T 4x": ("T", "40"), "C 2.5x": (CH, "25"), "C 4x": (CH, "40"), "T surge": ("T", "surge")}
rows, missing = {}, []
for s, (t, tag) in SETTINGS.items():
    for arm in ("stock", "gate", "wm10", "wm20", "known"):
        c = holds.get(f"{t}_{arm}_{tag}{SUF}")
        if c:
            rows[(s, arm)] = (c["good_full"], c[HOLD], c["hold_full_p99"], c["abandoned_rate"], c["evicted_share"])
        elif arm != "known" and not (t == CH and arm == "known"):
            missing.append(f"{t}_{arm}_{tag}{SUF}")
print(f"{'setting':8} {'arm':6} {'good':>5} {HOLD[-3:]:>7} {'p99':>7} {'aband':>6} {'evict':>6}")
for (s, arm), (g, p90, p99, ab, ev) in rows.items():
    print(f"{s:8} {arm:6} {g:5d} {p90:7.1f} {p99:7.1f} {ab:6.2f} {ev:6.2f}")
def close(w, g):   # w within 5% of g: goodput not more than 5% lower, p90 hold not more than 5% higher
    return w[0] >= 0.95 * g[0] and w[1] <= 1.05 * g[1]
done = [s for s in SETTINGS if (s, "gate") in rows and all((s, w) in rows for w in ("wm10", "wm20"))]
if missing:
    print("missing:", missing)
for w in ("wm10", "wm20"):
    print(w, "close to gate in", sum(close(rows[(s, w)], rows[(s, "gate")]) for s in done), "of", len(done), "settings")
kill = len(done) == 5 and any(all(close(rows[(s, w)], rows[(s, "gate")]) for s in done) for w in ("wm10", "wm20"))
def best_wm(s):
    return max(("wm10", "wm20"), key=lambda w: (rows[(s, w)][0], -rows[(s, w)][1]))
gate_ok = all(close(rows[(s, "gate")], rows[(s, best_wm(s))]) for s in done)
wm_wrong = all(any(not close(rows[(s, w)], rows[(s, "gate")]) for s in done) for w in ("wm10", "wm20"))
print("settings scored:", len(done), "of 5")
print("verdict:", "A FIXED WATERMARK IS ENOUGH" if kill else
      ("ONE GATE SETTING HOLDS WHERE NO ONE WATERMARK DOES" if len(done) == 5 and gate_ok and wm_wrong else
       ("NEITHER CLAIM" if len(done) == 5 else "INCOMPLETE")))
