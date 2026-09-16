# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : JVP set mutation
# sibling    : set.add inside a torch.func.jvp primal function

import torch

# Sibling construct: set.add inside a torch.func.jvp subgraph.
def run_once(use_compile):
    visited = set()

    def primal(x):
        visited.add("primal")
        return x.cos() + x.square()

    def fn(x):
        return torch.func.jvp(primal, (x,), (torch.ones_like(x),))

    runner = torch.compile(fn, backend="eager") if use_compile else fn
    primal_out, tangent_out = runner(torch.tensor([0.3, 0.9]))
    return (primal_out, tangent_out), set(visited)


eager_result, eager_state = run_once(False)
torch._dynamo.reset()
compiled_result, compiled_state = run_once(True)
assert torch.equal(compiled_result[0], eager_result[0]), (eager_result, compiled_result)
assert torch.equal(compiled_result[1], eager_result[1]), (eager_result, compiled_result)
assert compiled_state == eager_state, (eager_state, compiled_state)
