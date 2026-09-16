# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Vectorized list extension
# sibling    : list.extend inside a torch.func.vmap body

import torch

# Sibling construct: list.extend inside a torch.func.vmap subgraph.
def run_once(use_compile):
    events = []

    def mapped(x):
        events.extend(["mapped"])
        return x.square() + 1

    def fn(x):
        return torch.func.vmap(mapped)(x)

    runner = torch.compile(fn, backend="eager") if use_compile else fn
    result = runner(torch.arange(4.0))
    return result, list(events)


eager_result, eager_state = run_once(False)
torch._dynamo.reset()
compiled_result, compiled_state = run_once(True)
assert torch.equal(compiled_result, eager_result), (eager_result, compiled_result)
assert compiled_state == eager_state, (eager_state, compiled_state)
