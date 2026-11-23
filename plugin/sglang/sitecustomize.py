"""Instrumentation only, for SGLang 0.5.21: one log line per retraction, and one when a retracted request is
prefilled again. The same `preempt`/`lookup` lines as plugin/acr_preemptlog.py writes for vLLM.

SGLang has no plugin hook and starts its scheduler with multiprocessing "spawn", so this file is put on the
server's PYTHONPATH: every interpreter imports sitecustomize at start, and a finder patches ScheduleBatch right
after sglang.srt.managers.schedule_batch is first imported. Each wrapper calls the original and returns its
result, so no decision changes. Does nothing unless ACR_PREEMPT_LOG names the log file.
"""
import importlib.abc
import importlib.util
import json
import os
import sys
import threading
import time

TARGET = "sglang.srt.managers.schedule_batch"
_lock = threading.Lock()
_fh = None


def _rid(req):
    return req.rid[:-1] if req.rid.endswith("|") else req.rid


def _write(rec):
    global _fh
    try:
        with _lock:
            if _fh is None:
                _fh = open(os.environ["ACR_PREEMPT_LOG"], "a", buffering=1)
            _fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _tokens(req):
    return len(req.origin_input_ids) + len(req.output_ids)


def patch(mod):
    SB = mod.ScheduleBatch
    if getattr(SB, "_acr_patched", False):
        return
    orig_retract, orig_extend = SB.retract_decode, SB.prepare_for_extend

    def retract_decode(self, *a, **kw):
        res = orig_retract(self, *a, **kw)
        try:
            now = time.time()
            for req in res[0]:
                _write(dict(ev="preempt", t=now, rid=_rid(req), computed=_tokens(req), prompt=len(req.origin_input_ids),
                            out=len(req.output_ids), tokens=_tokens(req), n=int(req.retraction_count)))
            for req in res[2]:
                _write(dict(ev="abort", t=now, rid=_rid(req), tokens=_tokens(req)))
        except Exception:
            pass
        return res

    def prepare_for_extend(self, *a, **kw):
        try:
            back = [(_rid(r), len(r.prefix_indices), _tokens(r), int(r.retraction_count)) for r in self.reqs if r.is_retracted]
        except Exception:
            back = []
        res = orig_extend(self, *a, **kw)
        now = time.time()
        for rid, hit, tok, n in back:
            _write(dict(ev="lookup", t=now, rid=rid, hit=hit, tokens=tok, n=n))
        return res

    SB.retract_decode, SB.prepare_for_extend, SB._acr_patched = retract_decode, prepare_for_extend, True
    _write(dict(ev="plugin_loaded", t=time.time(), pid=os.getpid()))


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


if os.environ.get("ACR_PREEMPT_LOG") and not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
