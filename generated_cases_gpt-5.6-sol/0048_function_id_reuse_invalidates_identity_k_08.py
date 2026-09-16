# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Forbidden callable instance identity reuse
# sibling    : torch._dynamo.forbid_in_graph

import gc
import torch

# Sibling construct: torch._dynamo.forbid_in_graph.
class Operation:
    def __init__(self, bias):
        self.bias = bias

    def __call__(self, x):
        return x.square() + self.bias

victim = Operation(1000)
torch._dynamo.forbid_in_graph(victim)
stale_id = id(victim)
del victim
gc.collect()

def make_candidate():
    return Operation(7)

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
