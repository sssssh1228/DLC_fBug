# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Conditional user subscription assignment
# sibling    : user-defined __setitem__

import torch

# Sibling construct under test: user-defined __setitem__

class Ledger:
    def __init__(self):
        self.entries = {}

    def __setitem__(self, key, value):
        self.entries[key] = value


def execute(compiled):
    ledger = Ledger()

    def fn(pred, x):
        def true_branch(value):
            ledger["true"] = "selected"
            return value + 5

        def false_branch(value):
            ledger["false"] = "selected"
            return value - 5

        return torch.cond(pred, true_branch, false_branch, (x,))

    if compiled:
        fn = torch.compile(fn, backend="eager", fullgraph=True)

    true_result = fn(torch.tensor(True), torch.tensor(8.0))
    false_result = fn(torch.tensor(False), torch.tensor(8.0))
    return (true_result, false_result), dict(ledger.entries)


eager_outputs, eager_entries = execute(False)
compiled_outputs, compiled_entries = execute(True)
for eager, actual in zip(eager_outputs, compiled_outputs):
    torch.testing.assert_close(actual, eager)
assert compiled_entries == eager_entries, (compiled_entries, eager_entries)
