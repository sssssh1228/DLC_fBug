# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : wrap custom item assignment
# sibling    : user-defined __setitem__ mutation inside a wrap HOP subgraph

import torch
from torch._higher_order_ops.wrap import wrap


class Ledger:
    def __init__(self):
        self.records = {"calls": 0}

    def __setitem__(self, key, value):
        self.records[key] = value


# Sibling under test: user-defined __setitem__ inside a wrap HOP subgraph.
def execute(use_compile):
    ledger = Ledger()

    def fn(x):
        def body(value):
            ledger["calls"] = ledger.records["calls"] + 1
            return value.square()

        return wrap(body, x)

    target = torch.compile(fn, backend="eager", fullgraph=True) if use_compile else fn
    output = target(torch.tensor([2.0, -3.0]))
    return output, dict(ledger.records)


eager_output, eager_records = execute(False)
compiled_output, compiled_records = execute(True)
torch.testing.assert_close(compiled_output, eager_output)
assert compiled_records == eager_records, (compiled_records, eager_records)
