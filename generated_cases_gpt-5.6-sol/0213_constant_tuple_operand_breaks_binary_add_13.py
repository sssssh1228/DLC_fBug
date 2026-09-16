# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Constant tuple positional splat
# sibling    : CALL_FUNCTION_EX positional argument unpacking

import torch

CONSTANT_ARGS = (2.0, 6.0)

def combine(x, scale, bias):
    return x * scale + bias

# Sibling construct: CALL_FUNCTION_EX positional argument unpacking.
def fn(x):
    return combine(x, *CONSTANT_ARGS)

inp = torch.tensor(4.0)
eager = fn(inp)
compiled = torch.compile(fn, backend="eager")(inp)
torch.testing.assert_close(compiled, eager)
