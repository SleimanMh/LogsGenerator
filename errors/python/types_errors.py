"""
Python Type and Data Structure Errors (p_err_01 - p_err_25)
Covers: TypeError, KeyError, IndexError, AttributeError, ValueError, unpacking errors
"""

from utils.logger import log
import traceback


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Type & Data Structure Errors (01 - 25)
# ============================================================

def p_err_01():
    """Type mismatch: string concatenation"""
    x = "hello"
    y = 42
    result = x + y


def p_err_02():
    """Type mismatch: list indexing with string"""
    arr = [1, 2, 3]
    value = arr["index"]


def p_err_03():
    """Dictionary key error - multi-layer"""
    def get_nested_value(data):
        return data["level1"]["level2"]["missing"]
    
    data = {"level1": {"level2": {"value": 42}}}
    get_nested_value(data)


def p_err_04():
    """List index out of bounds - multi-layer"""
    def access_deep_list(items, indices):
        result = items
        for idx in indices:
            result = result[idx]
        return result
    
    nested = [[1, 2], [3, 4]]
    access_deep_list(nested, [0, 5])


def p_err_05():
    """Attribute error on None"""
    obj = None
    obj.method()


def p_err_06():
    """Unpacking value error"""
    x, y, z = [1, 2]


def p_err_07():
    """Zip length mismatch"""
    a = [1, 2, 3, 4, 5]
    b = [10, 20]
    result = list(zip(a, b))
    if len(result) != len(a):
        _ = a[10]


def p_err_08():
    """Integer conversion error"""
    value = int("not_a_number")


def p_err_09():
    """Float conversion error"""
    value = float("not_a_float")


def p_err_10():
    """Dictionary missing required key"""
    def process_config(config):
        return config["database"]["host"]
    
    config = {"database": {"port": 5432}}
    process_config(config)


def p_err_11():
    """Set operation on non-set"""
    x = [1, 2, 3]
    y = x.union([4, 5])


def p_err_12():
    """String format with wrong args"""
    template = "{} {} {}"
    result = template.format("a", "b")


def p_err_13():
    """Slice with non-integer"""
    arr = [1, 2, 3]
    value = arr["1":"2"]


def p_err_14():
    """Dictionary pop with missing key"""
    d = {"a": 1}
    value = d.pop("missing")


def p_err_15():
    """Tuple unpacking with wrong count"""
    t = (1, 2)
    a, b, c = t


def p_err_16():
    """Attribute access on wrong type"""
    s = "string"
    value = s.append("x")


def p_err_17():
    """Method call on None"""
    x = None
    y = x.split(",")


def p_err_18():
    """List multiplication with string"""
    arr = [1, 2, 3]
    result = arr * "invalid"


def p_err_19():
    """Dictionary comprehension KeyError"""
    d = {str(i): i for i in range(3)}
    result = d[3]


def p_err_20():
    """Nested dictionary access"""
    def get_value(d, key):
        return d[key]["nested"]["deep"]
    
    data = {"a": {"nested": {}}}
    get_value(data, "a")


def p_err_21():
    """Boolean operation on incompatible types"""
    x = "5" and 5
    result = x + "string"


def p_err_22():
    """Comparison between incompatible types"""
    x = "5"
    y = 5
    result = x < y


def p_err_23():
    """Index error with negative out of bounds"""
    arr = [1, 2, 3]
    value = arr[-100]


def p_err_24():
    """Attribute error on list"""
    lst = [1, 2, 3]
    value = lst.keys()


def p_err_25():
    """Dict update with non-dict"""
    d = {}
    d.update([1, 2, 3])


# Registry
__all__ = [f'p_err_{i:02d}' for i in range(1, 26)]
