# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Ternary power semantics
# sibling    : three-argument pow() modular exponentiation

import torch


def fn(x, base, exponent, modulus):
    # Sibling under test: three-argument pow() modular exponentiation.
    modular_power = pow(base, exponent, modulus)
    return x * modular_power


x = torch.tensor([1, 2, 3], dtype=torch.int64)
args = (7, 5, 13)
expected = fn(x.clone(), *args)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x.clone(), *args)
torch.testing.assert_close(actual, expected)
assert actual.dtype == expected.dtype
