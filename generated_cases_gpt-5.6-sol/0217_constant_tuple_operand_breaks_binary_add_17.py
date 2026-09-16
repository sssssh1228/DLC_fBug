# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Constant tuple as left operand
# sibling    : reversed tuple concatenation operand order

import torch


PREFIX = (3, 4)


def fn(x):
    # Sibling construct: reversed tuple concatenation operand order.
    dynamic_tail = (x, x.square())
    return PREFIX + dynamic_tail


x = torch.tensor(5)
eager = fn(x)
compiled = torch.compile(fn, backend="eager")(x)
assert type(eager) is type(compiled)
assert eager[:2] == compiled[:2]
assert torch.equal(eager[2], compiled[2])
assert torch.equal(eager[3], compiled[3])
