# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Constant tuple positional expansion
# sibling    : starred positional argument expansion

import torch

CONSTANT_ARGS = (2, 5)


def collect(*args):
    return args


def fn(x):
    # Sibling construct: starred positional argument expansion.
    dynamic_args = (x.sigmoid(),)
    return collect(*dynamic_args, *CONSTANT_ARGS)


x = torch.randn(6)
eager = fn(x)
compiled = torch.compile(fn, backend="eager")(x)
torch.testing.assert_close(compiled[0], eager[0])
assert compiled[1:] == eager[1:]
