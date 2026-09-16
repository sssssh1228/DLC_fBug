# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Polyfill registry identity collision
# sibling    : torch.compiler.substitute_in_graph

import gc
import torch

# Sibling construct: torch.compiler.substitute_in_graph.
def make_original():
    def original(x, *, offset=1):
        return x + offset
    return original

original = make_original()
stale_id = id(original)

@torch.compiler.substitute_in_graph(original, skip_signature_check=True)
def replacement(x, *, offset=1):
    return x - offset - 100

del original
del replacement
gc.collect()

def make_candidate():
    def candidate(x, *, offset=1):
        return x * 2 + offset
    return candidate

candidate = None
for _ in range(4096):
    current = make_candidate()
    if id(current) == stale_id:
        candidate = current
        break
    del current
if candidate is None:
    candidate = make_candidate()

def entry(x):
    return candidate(x, offset=3)

def capture(fn):
    try:
        return ("ok", fn(torch.tensor(5)).item())
    except Exception as exc:
        return ("error", type(exc).__name__)

eager_result = capture(entry)
try:
    compiled_result = capture(torch.compile(entry, backend="eager", fullgraph=True))
except Exception as exc:
    compiled_result = ("error", type(exc).__name__)
assert compiled_result == eager_result, (eager_result, compiled_result, id(candidate) == stale_id)
