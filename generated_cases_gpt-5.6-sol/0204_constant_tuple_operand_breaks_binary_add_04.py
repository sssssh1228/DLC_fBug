# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : List extend from constant tuple
# sibling    : list.extend sequence unpacking of a constant tuple

import torch

EXTRA = (2.0, 4.0)

# Sibling method: list.extend unpacking a constant tuple.
def fn(x):
    values = [x, x.neg()]
    values.extend(EXTRA)
    return values[0] - values[1] + values[2] + values[3]


x = torch.tensor([1.5, -2.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager")
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
