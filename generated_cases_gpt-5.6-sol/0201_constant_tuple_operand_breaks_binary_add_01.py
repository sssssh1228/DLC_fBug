# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : List concatenation with constant suffix
# sibling    : Binary list concatenation with a constant list operand

import torch

SUFFIX = [2.0, 3.0]

# Sibling construct: binary list concatenation with a constant list.
def fn(x):
    values = [x, x + 1.0] + SUFFIX
    return values[0] + values[1] + values[2] + values[3]


x = torch.tensor([1.0, 4.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager")
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
