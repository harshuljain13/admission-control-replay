"""CPU-only check of plugin/sglang/sitecustomize.py against a stand-in for SGLang's ScheduleBatch.

The stand-in module has the attributes the wrappers read (rid, origin_input_ids, output_ids, retraction_count,
is_retracted, prefix_indices). The check runs in a child started with multiprocessing "spawn", as SGLang starts its
scheduler, and confirms that the wrapped methods log and return exactly what the originals return.

    python3 -I tests/test_sglang_plugin.py
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PLUGIN = HERE / "plugin" / "sglang"

FAKE = textwrap.dedent('''
    class Req:
        def __init__(self, rid, p, out):
            self.rid, self.origin_input_ids, self.output_ids = rid, [0] * p, [0] * out
            self.retraction_count, self.is_retracted, self.prefix_indices = 0, False, []

    class ScheduleBatch:
        def __init__(self, reqs):
            self.reqs = reqs

        def retract_decode(self):
            order = sorted(range(len(self.reqs)), key=lambda i: (len(self.reqs[i].output_ids),
                           -len(self.reqs[i].origin_input_ids)), reverse=True)
            gone = [self.reqs[i] for i in reversed(order[1:])]  # popped from the end
            for r in gone:
                r.retraction_count += 1
                r.is_retracted = True
            self.reqs = [self.reqs[order[0]]]
            return gone, 0.5, []

        def prepare_for_extend(self):
            for r in self.reqs:
                r.prefix_indices = [0] * 3
            return "extended"
''')

CHILD = textwrap.dedent('''
    import multiprocessing as mp, sys

    def work(q):
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        b = ScheduleBatch([Req("a", 100, 50), Req("b|", 200, 10), Req("c|", 100, 10)])
        gone, ratio, abort = b.retract_decode()
        back = ScheduleBatch(gone)
        q.put(([r.rid for r in gone], ratio, abort, back.prepare_for_extend(), getattr(ScheduleBatch, "_acr_patched", False)))

    if __name__ == "__main__":
        mp.set_start_method("spawn")
        q = mp.Queue()
        p = mp.Process(target=work, args=(q,))
        p.start()
        print(repr(q.get()))
        p.join()
        sys.exit(p.exitcode)
''')


def test_plugin_in_spawned_child():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pkg = d / "sglang" / "srt" / "managers"
        pkg.mkdir(parents=True)
        for q in (d / "sglang", d / "sglang" / "srt", pkg):
            (q / "__init__.py").write_text("")
        (pkg / "schedule_batch.py").write_text(FAKE)
        (d / "child.py").write_text(CHILD)
        log = d / "preempt.jsonl"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PLUGIN), str(d)]), ACR_PREEMPT_LOG=str(log))
        out = subprocess.run([sys.executable, str(d / "child.py")], env=env, capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "(['b|', 'c|'], 0.5, [], 'extended', True)", out.stdout
        recs = [json.loads(x) for x in log.read_text().splitlines()]
        pre = [r for r in recs if r["ev"] == "preempt"]
        look = [r for r in recs if r["ev"] == "lookup"]
        # fewest decoded first is retracted first; "a" (50 decoded) is kept
        assert [r["rid"] for r in pre] == ["b", "c"], pre
        assert pre[0]["tokens"] == 210 and pre[0]["n"] == 1 and pre[0]["out"] == 10
        assert sorted(r["rid"] for r in look) == ["b", "c"] and all(r["hit"] == 0 for r in look), look
        assert any(r["ev"] == "plugin_loaded" for r in recs)

        # without ACR_PREEMPT_LOG the plugin does nothing
        env.pop("ACR_PREEMPT_LOG")
        out = subprocess.run([sys.executable, str(d / "child.py")], env=env, capture_output=True, text=True, timeout=60)
        assert out.returncode == 0 and out.stdout.strip().endswith("'extended', False)"), (out.stdout, out.stderr)


if __name__ == "__main__":
    test_plugin_in_spawned_child()
    print("test_plugin_in_spawned_child ok")
