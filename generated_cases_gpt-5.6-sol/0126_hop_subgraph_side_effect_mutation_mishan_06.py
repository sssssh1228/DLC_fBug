# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Conditional dictionary update
# sibling    : dict.update inside torch.cond branch speculation

import torch

# Sibling construct: dict.update inside a torch.cond subgraph.
def run_once(use_compile):
    audit = {}

    def true_branch(x):
        audit.update({"branch": "true"})
        return x + 1

    def false_branch(x):
        audit.update({"branch": "false"})
        return x - 1

    def fn(x, pred):
        return torch.cond(pred, true_branch, false_branch, (x,))

    runner = torch.compile(fn, backend="eager") if use_compile else fn
    result = runner(torch.tensor(3.0), torch.tensor(True))
    return result, dict(audit)


eager_result, eager_state = run_once(False)
torch._dynamo.reset()
compiled_result, compiled_state = run_once(True)
assert torch.equal(compiled_result, eager_result), (eager_result, compiled_result)
assert compiled_state == eager_state, (eager_state, compiled_state)
