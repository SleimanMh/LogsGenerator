"""
Python I/O and File Errors (p_err_46 - p_err_65)
Covers: FileNotFoundError, PermissionError, JSON/YAML parsing, file operations
"""

import traceback
import os


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# I/O & File Errors (46 - 65)
# ============================================================

def p_err_46():
    """File not found error"""
    with open("/nonexistent/path/file.txt", "r") as f:
        data = f.read()


def p_err_47():
    """File permission denied"""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        temp_file = f.name
    
    try:
        os.chmod(temp_file, 0o000)
        with open(temp_file, "r") as f:
            data = f.read()
    finally:
        os.chmod(temp_file, 0o644)
        os.remove(temp_file)


def p_err_48():
    """JSON decode error"""
    import json
    invalid_json = '{"key": value}'
    data = json.loads(invalid_json)


def p_err_49():
    """YAML parse error"""
    import yaml
    invalid_yaml = """
    key: value
  invalid: indentation
    """
    data = yaml.safe_load(invalid_yaml)


def p_err_50():
    """File already exists"""
    import os
    import tempfile
    
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, "existing_file.txt")
    
    with open(temp_file, "w") as f:
        f.write("test")
    
    # Try to create exclusive
    with open(temp_file, "x") as f:
        f.write("new")


def p_err_51():
    """Directory not found"""
    with open("/nonexistent/directory/file.txt", "w") as f:
        f.write("test")


def p_err_52():
    """File closed error"""
    f = open("test.txt", "w")
    f.close()
    f.write("test")


def p_err_53():
    """JSON with invalid encoding"""
    import json
    data = {'key': set([1, 2, 3])}
    json_str = json.dumps(data)


def p_err_54():
    """Read from write-only file"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        temp_file = f.name
    
    try:
        with open(temp_file, "w") as f:
            data = f.read()
    finally:
        os.remove(temp_file)


def p_err_55():
    """Directory as file"""
    import tempfile
    temp_dir = tempfile.gettempdir()
    with open(temp_dir, "r") as f:
        data = f.read()


def p_err_56():
    """JSON key error - missing"""
    import json
    json_str = '{"a": 1}'
    data = json.loads(json_str)
    value = data["missing"]


def p_err_57():
    """File encoding mismatch"""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        f.write("test 你好")
        temp_file = f.name
    
    try:
        with open(temp_file, "r", encoding='ascii') as f:
            data = f.read()
    finally:
        os.remove(temp_file)


def p_err_58():
    """YAML type error"""
    import yaml
    invalid_yaml = "key: !!unknown tag"
    data = yaml.safe_load(invalid_yaml)


def p_err_59():
    """CSV parsing error"""
    import csv
    import io
    
    csv_data = "a,b,c\n1,2,3"
    reader = csv.DictReader(io.StringIO(csv_data))
    rows = list(reader)
    value = rows[0]["missing"]


def p_err_60():
    """File mode error"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        temp_file = f.name
    
    try:
        with open(temp_file, "x") as f:  # x mode on existing file
            f.write("test")
    finally:
        os.remove(temp_file)


def p_err_61():
    """Path error - invalid path"""
    with open("\x00invalid\x00path.txt", "r") as f:
        data = f.read()


def p_err_62():
    """JSON parse with circular reference"""
    import json
    
    def create_circular():
        obj = {}
        obj['self'] = obj
        return obj
    
    data = create_circular()
    json.dumps(data)


def p_err_63():
    """YAML duplicate key"""
    import yaml
    duplicate_yaml = """
    key: value1
    key: value2
    """
    data = yaml.safe_load(duplicate_yaml)


def p_err_64():
    """File system error - too many open files"""
    files = []
    try:
        for i in range(10000):
            f = open(f"test_{i}.txt", "w")
            files.append(f)
    finally:
        for f in files:
            try:
                f.close()
            except:
                pass
        import glob
        for f in glob.glob("test_*.txt"):
            try:
                os.remove(f)
            except:
                pass


def p_err_65():
    """JSON default encoder"""
    import json
    from datetime import datetime
    
    data = {'timestamp': datetime.now()}
    json.dumps(data)


# Registry
__all__ = [f'p_err_{i:02d}' for i in range(46, 66)]
