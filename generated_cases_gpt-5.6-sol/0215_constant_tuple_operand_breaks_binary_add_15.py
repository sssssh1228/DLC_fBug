# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Iteration over constant tuple
# sibling    : FOR_ITER over a constant tuple sequence

import torch

CONSTANT_FACTORS = (1.0, 2.0, 4.0)

# Sibling construct: FOR_ITER over a constant tuple sequence.
def fn(x):
    result = torch.zeros_like(x)
    for factor in CONSTANT_FACTORS:
        result = result + x * factor
    return result

inp = torch.tensor(2.5)
eager = fn(inp)
compiled = torch.compile(fn, backend="eager")(inp)
torch.testing.assert_close(compiled, eager)
