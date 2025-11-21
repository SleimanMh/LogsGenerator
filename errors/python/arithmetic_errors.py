"""
Python Arithmetic and Math Errors (p_err_66 - p_err_85)
Covers: ZeroDivisionError, domain errors, overflow, bitwise operations
"""

import traceback
import math


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Arithmetic & Math Errors (66 - 85)
# ============================================================

def p_err_66():
    """Division by zero"""
    x = 10
    y = 0
    result = x / y


def p_err_67():
    """Modulo by zero"""
    x = 10
    y = 0
    result = x % y


def p_err_68():
    """Math domain error - sqrt negative"""
    result = math.sqrt(-1)


def p_err_69():
    """Math domain error - log negative"""
    result = math.log(-1)


def p_err_70():
    """Overflow in exponentiation"""
    result = 10 ** 10000


def p_err_71():
    """Integer overflow in bitwise"""
    x = 1 << 10000


def p_err_72():
    """Bitwise operation on float"""
    x = 5.5
    result = x & 3


def p_err_73():
    """Math domain error - asin out of range"""
    result = math.asin(2)


def p_err_74():
    """Math domain error - acos out of range"""
    result = math.acos(2)


def p_err_75():
    """Power with invalid exponent"""
    result = (-2) ** 0.5


def p_err_76():
    """Division with type mismatch"""
    x = "10"
    y = 2
    result = x / y


def p_err_77():
    """Factorial of negative number"""
    result = math.factorial(-1)


def p_err_78():
    """Combinations with invalid input"""
    result = math.comb(5, 10)


def p_err_79():
    """Perm with invalid input"""
    result = math.perm(5, 10)


def p_err_80():
    """Gcd with non-integer"""
    result = math.gcd(5.5, 10)


def p_err_81():
    """Base conversion with invalid base"""
    x = 255
    result = format(x, 'b100')  # Invalid format spec


def p_err_82():
    """Radix conversion error"""
    x = int("ZZZ", 2)  # Invalid for base 2


def p_err_83():
    """Float precision error"""
    x = float('inf')
    y = float('inf')
    result = x - y


def p_err_84():
    """Zero to negative power"""
    result = 0 ** (-1)


def p_err_85():
    """Complex number domain error"""
    import cmath
    result = cmath.sqrt(-1 + 0j)
    result = result ** (1/3)
    result = math.factorial(5)  # factorial on complex


# Registry
__all__ = [f'p_err_{i:02d}' for i in range(66, 86)]
