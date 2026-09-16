# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Conditional dictionary update
# sibling    : dict.update

import torch

# Sibling construct under test: dict.update

def execute(compiled):
    state = {"true": 0, "false": 0}

    def fn(pred, x):
        def true_branch(value):
            state.update({"true": state["true"] + 1})
            return value + 1

        def false_branch(value):
            state.update({"false": state["false"] + 1})
            return value - 1

        return torch.cond(pred, true_branch, false_branch, (x,))

    if compiled:
        fn = torch.compile(fn, backend="eager", fullgraph=True)

    true_result = fn(torch.tensor(True), torch.tensor(4.0))
    false_result = fn(torch.tensor(False), torch.tensor(4.0))
    return (true_result, false_result), dict(state)


eager_outputs, eager_state = execute(False)
compiled_outputs, compiled_state = execute(True)
for eager, actual in zip(eager_outputs, compiled_outputs):
    torch.testing.assert_close(actual, eager)
assert compiled_state == eager_state, (compiled_state, eager_state)
