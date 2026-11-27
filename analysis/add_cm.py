"""Apply the pre-registered chat capacity rule to C_probe2 and add the 8 Cm_* cells to config/exp03.json.
python add_cm.py probe_dir config.json  -> prints the Cm cell names, or exits 2 if the rule gives no usable capacity"""
import json, subprocess, sys
probe_dir, cfg_path = sys.argv[1], sys.argv[2]
out = subprocess.run([sys.executable, __file__.replace("add_cm.py", "probe.py"), probe_dir], capture_output=True, text=True).stdout
sys.stderr.write(out)
cap = out.split("measured chat capacity (rps):")[1].split()[0]
segs = json.load(open(probe_dir + "/cell.json"))["cfg"]["segments"]
d = json.load(open(cfg_path))
am = d["amendment_2026_11_27"]
if cap == "None" or abs(float(cap) - segs[-1][1]) < 1e-9:
    am["probe_result_2"] = f"C_probe2 gave no interior capacity (measured {cap}); Cm cells not added"
    json.dump(d, open(cfg_path, "w"), indent=1, ensure_ascii=False); open(cfg_path, "a").write("\n")
    sys.exit(2)
cap = float(cap)
am["probe_result_2"] = f"C_probe2 measured chat capacity = {cap:.4f} rps (highest step passing the rule); Cm cells use cap_rps = {cap:.4f}"
names = []
for load, seed in ((2.5, 3525), (4.0, 3540)):
    for arm in ("stock", "gate", "wm10", "wm20"):
        n = f"Cm_{arm}_{int(load * 10)}"
        d["cells"] = [c for c in d["cells"] if c["name"] != n]
        d["cells"].append(dict(name=n, arm=arm, trace="chat", load=load, arrival_seed=seed, cap_rps=round(cap, 4)))
        names.append(n)
json.dump(d, open(cfg_path, "w"), indent=1, ensure_ascii=False); open(cfg_path, "a").write("\n")
print(" ".join(names))
