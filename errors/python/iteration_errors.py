"""
Python Iteration and Control Flow Errors (p_err_86 - p_err_100)
Covers: StopIteration, recursion limits, context managers, generator errors
"""

import traceback
import sys


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Iteration & Control Flow Errors (86 - 100)
# ============================================================

def p_err_86():
    """Generator exhaustion"""
    def gen():
        yield 1
        yield 2
    
    g = gen()
    next(g)
    next(g)
    next(g)


def p_err_87():
    """Loop unpacking error"""
    for x, y, z in [[1, 2]]:
        pass


def p_err_88():
    """Enumerate type error"""
    for idx in enumerate(123):
        pass


def p_err_89():
    """Iterator without __iter__"""
    class NotIterator:
        pass
    
    obj = NotIterator()
    for item in obj:
        pass


def p_err_90():
    """Iterator without __next__"""
    class BadIterator:
        def __iter__(self):
            return self
    
    it = BadIterator()
    next(it)


def p_err_91():
    """Recursion limit exceeded"""
    def recursive():
        return recursive()
    
    recursive()


def p_err_92():
    """Context manager missing __enter__"""
    class BadContext:
        pass
    
    with BadContext():
        pass


def p_err_93():
    """Context manager missing __exit__"""
    class BadContext:
        def __enter__(self):
            return self
    
    with BadContext():
        pass


def p_err_94():
    """For loop with non-iterable"""
    for item in 123:
        pass


def p_err_95():
    """Enumerate on non-iterable"""
    for idx, item in enumerate(None):
        pass


def p_err_96():
    """Zip on non-iterable"""
    for a, b in zip([1, 2], None):
        pass


def p_err_97():
    """Map on non-iterable"""
    result = list(map(str, None))


def p_err_98():
    """Filter on non-iterable"""
    result = list(filter(None, 123))


def p_err_99():
    """Reversed on non-iterable"""
    result = list(reversed(123))


def p_err_100():
    """Sorted on incompatible types"""
    result = sorted([1, "two", 3.0])


# Registry
__all__ = [f'p_err_{i:02d}' for i in range(86, 101)]
