# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Conditional set update
# sibling    : set.update

import torch

# Sibling construct under test: set.update

def execute(compiled):
    visited = set()

    def fn(pred, x):
        def true_branch(value):
            visited.update(("true", "shared"))
            return value.square()

        def false_branch(value):
            visited.update(("false", "shared"))
            return -value

        return torch.cond(pred, true_branch, false_branch, (x,))

    if compiled:
        fn = torch.compile(fn, backend="eager", fullgraph=True)

    true_result = fn(torch.tensor(True), torch.tensor(3.0))
    false_result = fn(torch.tensor(False), torch.tensor(3.0))
    return (true_result, false_result), frozenset(visited)


eager_outputs, eager_state = execute(False)
compiled_outputs, compiled_state = execute(True)
for eager, actual in zip(eager_outputs, compiled_outputs):
    torch.testing.assert_close(actual, eager)
assert compiled_state == eager_state, (compiled_state, eager_state)
