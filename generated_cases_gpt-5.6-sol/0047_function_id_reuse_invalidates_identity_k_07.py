# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Reused identity after disable wrapper collection
# sibling    : torch.compiler.disable

import gc
import torch

# Sibling construct: torch.compiler.disable.
def make_disabled():
    offset = 11
    def victim(x):
        return x + offset
    return torch.compiler.disable(victim)

victim = make_disabled()
stale_id = id(victim)
del victim
gc.collect()

def make_candidate():
    scale = 5
    def candidate(x):
        return x * scale
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
    return candidate(x) + 2

def capture(fn):
    try:
        return ("ok", fn(torch.tensor(3)).item())
    except Exception as exc:
        return ("error", type(exc).__name__)

eager_result = capture(entry)
try:
    compiled_result = capture(torch.compile(entry, backend="eager", fullgraph=True))
except Exception as exc:
    compiled_result = ("error", type(exc).__name__)
assert compiled_result == eager_result, (eager_result, compiled_result, id(candidate) == stale_id)
