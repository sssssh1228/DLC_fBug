# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Tuple repetition
# sibling    : Binary sequence repetition through tuple __mul__

import torch

REPEAT = 2

# Sibling method: tuple sequence repetition through binary multiplication.
def fn(x):
    values = (x, x + 1.0) * REPEAT
    return values[0] + values[1] + values[2] + values[3]


x = torch.tensor([-1.0, 3.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager")
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
