# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Starred tuple display with constant tuple
# sibling    : BUILD_TUPLE_UNPACK via starred tuple construction

import torch

CONSTANT_PART = (3.0, 4.0)

# Sibling construct: BUILD_TUPLE_UNPACK via starred tuple construction.
def fn(x):
    runtime_part = (x, x + 1.0)
    merged = (*runtime_part, *CONSTANT_PART)
    return merged[0] + merged[1] + merged[2] + merged[3]

inp = torch.tensor(2.0)
eager = fn(inp)
compiled = torch.compile(fn, backend="eager")(inp)
torch.testing.assert_close(compiled, eager)
