# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Set Iterator Shared Iter Identity
# sibling    : set_iterator __iter__ identity and exhaustion

import torch

# Sibling under test: set_iterator __iter__ identity and exhaustion.
def fn(it, x):
    alias = iter(it)
    assert alias is it
    total = next(alias) + next(it) + next(alias)
    exhausted_value = next(it, 100)
    return x * total + exhausted_value


def make_iterator():
    return iter({2, 5, 9})


def main():
    x = torch.tensor(3)
    eager = fn(make_iterator(), x)
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    compiled = compiled_fn(make_iterator(), x)
    torch.testing.assert_close(compiled, eager)


if __name__ == "__main__":
    main()
