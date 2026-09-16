# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Constant tuple starred into list
# sibling    : starred iterable expansion in a list display

import torch

CONSTANT_TAIL = (7, 11)


def fn(x):
    # Sibling construct: starred iterable expansion in a list display.
    dynamic = (x.square(),)
    return [*dynamic, *CONSTANT_TAIL]


x = torch.randn(3)
eager = fn(x)
compiled = torch.compile(fn, backend="eager")(x)
torch.testing.assert_close(compiled[0], eager[0])
assert compiled[1:] == eager[1:]
