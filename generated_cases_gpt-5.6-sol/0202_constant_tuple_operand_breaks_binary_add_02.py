# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Reverse tuple concatenation
# sibling    : Binary tuple concatenation with the constant tuple on the left

import torch

PREFIX = (2.0, 3.0)

# Sibling construct: reverse tuple concatenation with a constant prefix.
def fn(x):
    values = PREFIX + (x, x.square())
    return values[0] * values[2] + values[1] * values[3]


x = torch.tensor([2.0, 5.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager")
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
