# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Gradient transform nonlocal rebinding
# sibling    : nonlocal augmented assignment inside torch.func.grad

import torch

# Sibling construct: nonlocal rebinding inside a torch.func.grad subgraph.
def run_once(use_compile):
    call_count = 0

    def primal(x):
        nonlocal call_count
        call_count += 1
        return (x.sin() * x).sum()

    def fn(x):
        return torch.func.grad(primal)(x)

    runner = torch.compile(fn, backend="eager") if use_compile else fn
    result = runner(torch.tensor([0.2, 0.7, 1.1]))
    return result, call_count


eager_result, eager_state = run_once(False)
torch._dynamo.reset()
compiled_result, compiled_state = run_once(True)
assert torch.equal(compiled_result, eager_result), (eager_result, compiled_result)
assert compiled_state == eager_state, (eager_state, compiled_state)
