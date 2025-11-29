"""
Python Advanced Errors (p_err_101 - p_err_130)
Covers: advanced core Python errors across types, control flow, stdlib usage
"""

import traceback
import json
import math
import re
from datetime import datetime


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Advanced Python Errors (101 - 130)
# ============================================================

def p_err_101():
    """TypeError: unsupported operand types for +"""
    value_left = 10
    value_right = "ten"
    combo = value_left + value_right  # noqa: F841


def p_err_102():
    """AttributeError: calling list-only method on int"""
    number_object = 42
    number_object.append(1)  # noqa: F841


def p_err_103():
    """ValueError: invalid literal for int()"""
    numeric_string = "xyz123"
    result_int = int(numeric_string)  # noqa: F841


def p_err_104():
    """KeyError: dictionary missing key"""
    data_map = {"alpha": 1, "beta": 2}
    missing = data_map["gamma"]  # noqa: F841


def p_err_105():
    """IndexError: list index out of range"""
    collection = [0, 1, 2]
    out_of_bounds = collection[10]  # noqa: F841


def p_err_106():
    """ZeroDivisionError: division by zero"""
    numerator_value = 7
    denominator_value = 0
    quotient_value = numerator_value / denominator_value  # noqa: F841


def p_err_107():
    """FileNotFoundError: opening non-existing file"""
    with open("non_existing_advanced_107.txt", "r", encoding="utf8") as f:
        content_data = f.read()  # noqa: F841


def p_err_108():
    """JSONDecodeError: invalid JSON string"""
    broken_json = "{invalid json: true"
    parsed_result = json.loads(broken_json)  # noqa: F841


def p_err_109():
    """AssertionError: manual assertion failure"""
    item_count = 5
    assert item_count == 10, "Item count mismatch"


def p_err_110():
    """TypeError: wrong number of arguments to function"""

    def helper_function(a, b):
        return a + b

    computed = helper_function(1)  # noqa: F841


def p_err_111():
    """ImportError: importing non-existing module"""
    __import__("non_existing_advanced_module_111")


def p_err_112():
    """UnicodeDecodeError: decoding bytes with wrong codec"""
    raw_bytes = b"\xff\xfe\xfa"
    decoded_value = raw_bytes.decode("utf-8")  # noqa: F841


def p_err_113():
    """UnicodeEncodeError: encoding non-ASCII to ASCII"""
    unicode_text = "price: 10€"
    encoded_bytes = unicode_text.encode("ascii")  # noqa: F841


def p_err_114():
    """IsADirectoryError: trying to open directory as file"""
    handle = open(".", "r")  # noqa: F841


def p_err_115():
    """OSError: invalid file operation on path"""
    # Trying to remove a file that is very unlikely to exist
    import os
    os.remove("non_existing_advanced_115.tmp")


def p_err_116():
    """RecursionError: maximum recursion depth exceeded"""

    def recursive_call(level):
        return recursive_call(level + 1)

    recursive_call(0)


def p_err_117():
    """RuntimeError: explicit runtime error"""
    raise RuntimeError("Advanced runtime failure in p_err_117")


def p_err_118():
    """NameError: using undefined variable"""
    value_result = undefined_advanced_variable_118  # noqa: F821,F841


def p_err_119():
    """UnboundLocalError: local variable referenced before assignment"""
    counter_value = 5

    def inner():
        nonlocal counter_value  # Make it trickier but still wrong usage in some contexts
        counter_value = counter_value + 1  # noqa: F841

    inner()


def p_err_120():
    """TypeError: list indices must be integers, not string"""
    sequence_items = [10, 20, 30]
    value_item = sequence_items["index"]  # noqa: F841


def p_err_121():
    """StopIteration: next() on exhausted iterator"""
    iterator_obj = iter([])
    item_val = next(iterator_obj)  # noqa: F841


def p_err_122():
    """ValueError: not enough values to unpack"""
    pair_values = [1]
    first_val, second_val = pair_values  # noqa: F841


def p_err_123():
    """OverflowError: math range error"""
    huge_exp = math.exp(1000)  # noqa: F841


def p_err_124():
    """TypeError: calling non-callable object"""
    callable_like = 42
    result_call = callable_like()  # noqa: F841


def p_err_125():
    """KeyError: popping missing key from dictionary"""
    storage_map = {"one": 1}
    removed_value = storage_map.pop("missing-key")  # noqa: F841


def p_err_126():
    """IndexError: popping from empty list"""
    empty_list_obj = []
    removed_element = empty_list_obj.pop()  # noqa: F841


def p_err_127():
    """re.error: invalid regular expression"""
    pattern_text = r"[unclosed"
    compiled_pattern = re.compile(pattern_text)  # noqa: F841


def p_err_128():
    """TypeError: 'NoneType' object is not iterable"""
    none_sequence = None
    for _value in none_sequence:  # noqa: F841
        pass


def p_err_129():
    """ValueError: invalid date format for strptime"""
    timestamp_text = "2025/30/99 25:61"
    parsed_time = datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M")  # noqa: F841


def p_err_130():
    """ZeroDivisionError: integer floor division by zero"""
    dividend_val = 100
    divisor_val = 0
    floor_result = dividend_val // divisor_val  # noqa: F841


__all__ = [f"p_err_{i:03d}" for i in range(101, 131)]
