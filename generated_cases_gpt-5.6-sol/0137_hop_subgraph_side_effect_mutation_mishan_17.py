# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Conditional list extension
# sibling    : list.extend

import torch

# Sibling construct under test: list.extend

def execute(compiled):
    events = []

    def fn(pred, x):
        def true_branch(value):
            events.extend(["entered", "true"])
            return value * 2

        def false_branch(value):
            events.extend(["entered", "false"])
            return value / 2

        return torch.cond(pred, true_branch, false_branch, (x,))

    if compiled:
        fn = torch.compile(fn, backend="eager", fullgraph=True)

    first = fn(torch.tensor(True), torch.tensor(6.0))
    second = fn(torch.tensor(False), torch.tensor(6.0))
    return (first, second), tuple(events)


eager_outputs, eager_events = execute(False)
compiled_outputs, compiled_events = execute(True)
for eager, actual in zip(eager_outputs, compiled_outputs):
    torch.testing.assert_close(actual, eager)
assert compiled_events == eager_events, (compiled_events, eager_events)
