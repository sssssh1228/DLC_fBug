# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Extended unpack assignment from mixed tuple
# sibling    : UNPACK_EX extended iterable assignment

import torch

CONSTANT_MIDDLE = (5.0, 7.0)

# Sibling construct: UNPACK_EX extended iterable assignment.
def fn(x):
    values = (x, *CONSTANT_MIDDLE, x * 2.0)
    first, *middle, last = values
    return first + middle[0] - middle[1] + last

inp = torch.tensor(3.0)
eager = fn(inp)
compiled = torch.compile(fn, backend="eager")(inp)
torch.testing.assert_close(compiled, eager)
