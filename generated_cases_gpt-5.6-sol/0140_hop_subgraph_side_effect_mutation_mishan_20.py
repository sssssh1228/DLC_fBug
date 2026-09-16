# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Conditional iterator consumption
# sibling    : iterator state mutation through next

import torch

# Sibling construct under test: iterator state mutation through next

def execute(compiled):
    tokens = iter((10, 20, 30))

    def fn(pred, x):
        def true_branch(value):
            token = next(tokens)
            return value + token

        def false_branch(value):
            token = next(tokens)
            return value - token

        return torch.cond(pred, true_branch, false_branch, (x,))

    if compiled:
        fn = torch.compile(fn, backend="eager", fullgraph=True)

    true_result = fn(torch.tensor(True), torch.tensor(2.0))
    false_result = fn(torch.tensor(False), torch.tensor(2.0))
    remaining = tuple(tokens)
    return (true_result, false_result), remaining


eager_outputs, eager_remaining = execute(False)
compiled_outputs, compiled_remaining = execute(True)
for eager, actual in zip(eager_outputs, compiled_outputs):
    torch.testing.assert_close(actual, eager)
assert compiled_remaining == eager_remaining, (compiled_remaining, eager_remaining)
