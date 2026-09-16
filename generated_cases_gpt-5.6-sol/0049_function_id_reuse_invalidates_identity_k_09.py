# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Stale constant-result decoration
# sibling    : torch._dynamo.assume_constant_result

import gc
import torch

# Sibling construct: torch._dynamo.assume_constant_result.
def make_victim():
    def victim():
        return 999
    return victim

victim = make_victim()
torch._dynamo.assume_constant_result(victim)
stale_id = id(victim)
del victim
gc.collect()

state = [0]

def make_candidate():
    def candidate(*, step=1):
        state[0] += step
        return state[0]
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
    return x + candidate(step=1)

def capture(fn):
    state[0] = 0
    try:
        return ("ok", [fn(torch.tensor(10)).item(), fn(torch.tensor(10)).item()])
    except Exception as exc:
        return ("error", type(exc).__name__)

eager_result = capture(entry)
try:
    compiled = torch.compile(entry, backend="eager")
    compiled_result = capture(compiled)
except Exception as exc:
    compiled_result = ("error", type(exc).__name__)
assert compiled_result == eager_result, (eager_result, compiled_result, id(candidate) == stale_id)
