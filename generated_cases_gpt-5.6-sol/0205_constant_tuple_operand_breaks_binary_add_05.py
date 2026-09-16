# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Starred call expansion
# sibling    : Starred positional argument unpacking of a constant tuple

import torch

CONSTANT_ARGS = (2.0, 3.0)


def combine(a, b, c):
    return a * b + c


# Sibling construct: starred call argument expansion of a constant tuple.
def fn(x):
    dynamic_args = (x,)
    return combine(*dynamic_args, *CONSTANT_ARGS)


x = torch.tensor([2.0, 6.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager")
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
