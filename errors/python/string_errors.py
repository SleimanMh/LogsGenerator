"""
Python String and Encoding Errors (p_err_26 - p_err_45)
Covers: UnicodeDecodeError, UnicodeEncodeError, regex errors, format errors
"""

import traceback


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# String & Encoding Errors (26 - 45)
# ============================================================

def p_err_26():
    """Unicode decode error"""
    data = b'\x80\x81\x82'
    text = data.decode('utf-8')


def p_err_27():
    """Unicode encode error"""
    text = "Hello 你好 🎉"
    encoded = text.encode('ascii')


def p_err_28():
    """Regex pattern error"""
    import re
    pattern = re.compile("[")


def p_err_29():
    """String format missing argument"""
    template = "Hello {name}, you are {age} years old"
    result = template.format(name="Alice")


def p_err_30():
    """String index error"""
    s = "abc"
    char = s[10]


def p_err_31():
    """Bytes concatenation with string"""
    b = b"hello"
    s = "world"
    result = b + s


def p_err_32():
    """Regex substitution with invalid group"""
    import re
    pattern = r'(\w+)'
    text = "hello world"
    result = re.sub(pattern, r'\2', text)


def p_err_33():
    """String strip with invalid arg"""
    s = "hello"
    result = s.strip(['h'])


def p_err_34():
    """String split with wrong type"""
    s = "a,b,c"
    parts = s.split(1)


def p_err_35():
    """Bytes decode invalid encoding"""
    data = b"hello"
    text = data.decode('nonexistent-encoding')


def p_err_36():
    """String replace with non-string"""
    s = "hello"
    result = s.replace(1, "x")


def p_err_37():
    """Format string with dict missing key"""
    template = "{name} is {age}"
    result = template.format(**{"name": "Alice"})


def p_err_38():
    """Regex match on non-string"""
    import re
    pattern = re.compile(r'\d+')
    match = pattern.match(123)


def p_err_39():
    """String encoding with invalid codec"""
    text = "hello"
    encoded = text.encode('invalid-codec')


def p_err_40():
    """Unicode error in file read"""
    with open("test.txt", "w") as f:
        f.write("test")
    
    with open("test.txt", "rb") as f:
        data = f.read()
        text = data.decode('ascii')


def p_err_41():
    """String method on non-string"""
    x = 123
    result = x.upper()


def p_err_42():
    """Format with invalid placeholder"""
    template = "{0} {1} {2}"
    result = template.format("a", "b")


def p_err_43():
    """String concatenation type mismatch"""
    s = "text" + None


def p_err_44():
    """Regex search on bytes without pattern"""
    import re
    pattern = "invalid[pattern"
    re.compile(pattern)


def p_err_45():
    """String join on non-iterable"""
    s = "-"
    result = s.join(123)


# Registry
__all__ = [f'p_err_{i:02d}' for i in range(26, 46)]
