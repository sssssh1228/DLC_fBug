# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Augmented tuple concatenation
# sibling    : tuple augmented assignment via __iadd__ fallback

import torch


def make_fn():
    suffix = (7, 9)

    def fn(x):
        # Sibling construct: tuple augmented assignment via __iadd__ fallback.
        values = (x, x - 1)
        values += suffix
        return values

    return fn


fn = make_fn()
x = torch.tensor(6)
eager = fn(x)
compiled = torch.compile(fn, backend="eager")(x)
assert type(eager) is type(compiled)
assert len(eager) == len(compiled)
assert torch.equal(eager[0], compiled[0])
assert torch.equal(eager[1], compiled[1])
assert eager[2:] == compiled[2:]
