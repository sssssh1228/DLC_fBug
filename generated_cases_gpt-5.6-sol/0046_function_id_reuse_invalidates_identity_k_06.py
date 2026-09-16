# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Stale disallow-in-graph identity
# sibling    : torch._dynamo.disallow_in_graph

import gc
import torch

# Sibling construct: torch._dynamo.disallow_in_graph.
victim = lambda x: x + 100
torch._dynamo.allow_in_graph(victim)
torch._dynamo.disallow_in_graph(victim)
stale_id = id(victim)
del victim
gc.collect()

def make_candidate():
    return lambda x: x * 3 - 1

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
    return candidate(x)

def capture(fn):
    try:
        return ("ok", fn(torch.tensor(4)).item())
    except Exception as exc:
        return ("error", type(exc).__name__)

eager_result = capture(entry)
try:
    compiled_result = capture(torch.compile(entry, backend="eager", fullgraph=True))
except Exception as exc:
    compiled_result = ("error", type(exc).__name__)
assert compiled_result == eager_result, (eager_result, compiled_result, id(candidate) == stale_id)
