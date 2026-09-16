# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Sorted key and reverse handling
# sibling    : sorted() with a key function and reverse flag

import torch


def fn(x, values):
    # Sibling under test: sorted() with a key function and reverse flag.
    ordered = sorted(values, key=lambda item: (abs(item), item), reverse=True)
    encoding = ordered[0] * 100 + ordered[1] * 10 + ordered[2]
    return x + encoding


x = torch.tensor([0, 1], dtype=torch.int64)
values = [-2, 1, 3]
expected = fn(x.clone(), values.copy())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x.clone(), values.copy())
torch.testing.assert_close(actual, expected)
