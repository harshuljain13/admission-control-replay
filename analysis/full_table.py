"""Full cell table for Raj: good requests (5 s hold incl. engine queue), evicted share, give-up share, rebuild share,
hold p99. python full_table.py holds.json cells_dir [engine]"""
import json, sys, os
holds = json.load(open(sys.argv[1]))
eng = sys.argv[3] if len(sys.argv) > 3 else "vLLM 0.32.0"
print(f"| engine | cell | trace | load | arm | counted | good | evicted | gave up | rebuild share | hold p99 (s) | valid |")
print("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
for c in sorted(holds, key=lambda c: c["cell"]):
    m = json.load(open(os.path.join(sys.argv[2], c["cell"], "measures.json")))
    n = c["cell"]
    tr = "thinking" if n.startswith("T") else "chat"
    load = "surge" if "surge" in n else ("probe" if "probe" in n else f"{m['load']}x")
    valid = "no: chat capacity mis-set" if n.startswith("C_") else "yes"
    print(f"| {eng} | {n} | {tr} | {load} | {m['arm']} | {c['counted']} | {c['good_full']} | {c['evicted_share']:.1%} | "
          f"{c['abandoned_rate']:.0%} | {m['rebuild_share']:.2%} | {c['hold_full_p99']:.0f} | {valid} |")
