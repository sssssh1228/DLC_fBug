# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Checkpoint property mutation
# sibling    : descriptor-backed property assignment inside checkpoint

import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: property assignment inside a checkpoint subgraph.
class Counter:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value


def run_once(use_compile):
    counter = Counter()

    def inner(x):
        counter.value = counter.value + 1
        return x.sin() * 2

    def fn(x):
        return checkpoint(inner, x, use_reentrant=False)

    runner = torch.compile(fn, backend="eager") if use_compile else fn
    result = runner(torch.tensor([0.25, 0.5], requires_grad=True))
    return result.detach(), counter.value


eager_result, eager_state = run_once(False)
torch._dynamo.reset()
compiled_result, compiled_state = run_once(True)
assert torch.equal(compiled_result, eager_result), (eager_result, compiled_result)
assert compiled_state == eager_state, (eager_state, compiled_state)
