# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Tuple repetition by constant
# sibling    : tuple.__mul__ sequence repetition

import torch


def fn(x):
    # Sibling construct: tuple.__mul__ sequence repetition.
    values = (x.sin(), x.cos())
    return values * 2


x = torch.randn(4)
eager = fn(x)
compiled = torch.compile(fn, backend="eager")(x)
assert len(eager) == len(compiled)
for expected, actual in zip(eager, compiled):
    torch.testing.assert_close(actual, expected)
