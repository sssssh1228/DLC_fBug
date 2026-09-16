# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Extended unpack of constant tuple
# sibling    : extended iterable unpacking assignment

import torch

CONSTANT_FIELDS = (3, 4, 5, 6)


def fn(x):
    # Sibling construct: extended iterable unpacking assignment.
    first, *middle, last = CONSTANT_FIELDS
    return x * first, tuple(middle), last


x = torch.randn(4)
eager = fn(x)
compiled = torch.compile(fn, backend="eager")(x)
torch.testing.assert_close(compiled[0], eager[0])
assert compiled[1] == eager[1]
assert compiled[2] == eager[2]
