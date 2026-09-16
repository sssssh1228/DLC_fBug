# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Direct constant tuple destructuring
# sibling    : UNPACK_SEQUENCE fixed-length assignment

import torch

CONSTANT_PAIR = (8.0, 3.0)

# Sibling construct: UNPACK_SEQUENCE fixed-length assignment.
def fn(x):
    left, right = CONSTANT_PAIR
    return x + left * right

inp = torch.tensor(1.5)
eager = fn(inp)
compiled = torch.compile(fn, backend="eager")(inp)
torch.testing.assert_close(compiled, eager)
