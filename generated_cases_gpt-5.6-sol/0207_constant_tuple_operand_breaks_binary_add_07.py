# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Reflected tuple repetition
# sibling    : tuple.__rmul__ reflected sequence repetition

import torch


def fn(x):
    # Sibling construct: tuple.__rmul__ reflected sequence repetition.
    values = (x.relu(),)
    return 3 * values


x = torch.randn(5)
eager = fn(x)
compiled = torch.compile(fn, backend="eager")(x)
assert len(eager) == len(compiled)
for expected, actual in zip(eager, compiled):
    torch.testing.assert_close(actual, expected)
