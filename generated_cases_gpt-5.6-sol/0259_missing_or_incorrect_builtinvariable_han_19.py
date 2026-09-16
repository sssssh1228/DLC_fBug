# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Custom length dispatch
# sibling    : len() dispatch to __len__

import torch


class SizedPayload:
    def __init__(self, payload, size):
        self.payload = payload
        self.size = size

    def __len__(self):
        return self.size


def fn(x, size):
    # Sibling under test: len() dispatch to __len__.
    payload = SizedPayload(x, size)
    length = len(payload)
    return payload.payload + length


x = torch.tensor([2.0, 4.0])
expected = fn(x.clone(), 3)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x.clone(), 3)
torch.testing.assert_close(actual, expected)
