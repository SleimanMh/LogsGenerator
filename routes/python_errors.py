"""
Python Error Endpoints - 100+ Basic Python Errors
Covers: Type errors, I/O, string operations, collections, file handling, 
system operations, and basic programming logic errors
"""

from fastapi import FastAPI
from utils.logger import log
import traceback
from error_propagation.error_scenarios import (
    ScenarioContext, DataProcessor, PipelineManager, ErrorPropagator
)


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


def register_python_errors(app: FastAPI):

    # ============================================================
    # PYTHON ERROR CLASS: Type & Data Structure Errors (p_err_01 - p_err_25)
    # ============================================================

    def p_err_01():
        """Type mismatch: string concatenation"""
        x = "hello"
        y = 42
        result = x + y  # TypeError

    def p_err_02():
        """Type mismatch: list indexing with string"""
        arr = [1, 2, 3]
        value = arr["index"]  # TypeError

    def p_err_03():
        """Dictionary key error - multi-layer"""
        def get_nested_value(data):
            return data["level1"]["level2"]["missing"]
        
        data = {"level1": {"level2": {"value": 42}}}
        get_nested_value(data)  # KeyError

    def p_err_04():
        """List index out of bounds - multi-layer"""
        def access_deep_list(items, indices):
            result = items
            for idx in indices:
                result = result[idx]
            return result
        
        nested = [[1, 2], [3, 4]]
        access_deep_list(nested, [0, 5])  # IndexError

    def p_err_05():
        """Attribute error on None"""
        obj = None
        obj.method()  # AttributeError

    def p_err_06():
        """Unpacking value error"""
        x, y, z = [1, 2]  # ValueError

    def p_err_07():
        """Zip length mismatch"""
        a = [1, 2, 3, 4, 5]
        b = [10, 20]
        result = list(zip(a, b))
        if len(result) != len(a):
            _ = a[10]  # IndexError

    def p_err_08():
        """Integer conversion error"""
        value = int("not_a_number")  # ValueError

    def p_err_09():
        """Float conversion error"""
        value = float("not_a_float")  # ValueError

    def p_err_10():
        """Dictionary missing required key"""
        def process_config(config):
            return config["database"]["host"]
        
        config = {"database": {"port": 5432}}
        process_config(config)  # KeyError

    def p_err_11():
        """Set operation on non-set"""
        x = [1, 2, 3]
        y = x.union([4, 5])  # AttributeError

    def p_err_12():
        """String format with wrong args"""
        template = "{} {} {}"
        result = template.format("a", "b")  # IndexError

    def p_err_13():
        """Slice with non-integer"""
        arr = [1, 2, 3]
        result = arr["1":2]  # TypeError

    def p_err_14():
        """Dict pop with invalid key"""
        d = {"a": 1, "b": 2}
        value = d.pop("missing")  # KeyError

    def p_err_15():
        """Function call with wrong arg types"""
        def add(x, y):
            return x + y
        
        result = add("string", [1, 2, 3])  # TypeError

    def p_err_16():
        """Recursive data structure access"""
        def traverse(data, keys):
            for key in keys:
                data = data[key]
            return data
        
        nested = {"a": {"b": {"c": 1}}}
        traverse(nested, ["a", "b", "x"])  # KeyError

    def p_err_17():
        """Type coercion failure"""
        x = "abc"
        y = int(x)  # ValueError

    def p_err_18():
        """Comparison of incompatible types"""
        x = "hello"
        if x > 5:  # TypeError in some contexts
            pass

    def p_err_19():
        """Boolean operation on non-bool"""
        x = "true"
        result = x and 42  # works but can cascade to error

    def p_err_20():
        """Negative indexing out of range"""
        arr = [1, 2, 3]
        value = arr[-10]  # IndexError

    def p_err_21():
        """Dictionary comprehension key error"""
        data = [{"id": 1}, {"name": "test"}]
        result = {item["id"]: item["value"] for item in data}  # KeyError

    def p_err_22():
        """Lambda with wrong arguments"""
        f = lambda x, y: x + y
        result = f(1)  # TypeError

    def p_err_23():
        """Nested unpacking error"""
        (a, (b, c)) = (1, 2)  # ValueError

    def p_err_24():
        """Dictionary update with incompatible type"""
        d = {"a": 1}
        d.update([("b", 2)])
        d.update(("invalid",))  # ValueError

    def p_err_25():
        """Class instantiation without required args"""
        class MyClass:
            def __init__(self, required_param):
                self.param = required_param
        
        obj = MyClass()  # TypeError

    # ============================================================
    # PYTHON ERROR CLASS: String & Encoding Errors (p_err_26 - p_err_45)
    # ============================================================

    def p_err_26():
        """Unicode decode error"""
        data = b"\xff\xfe"
        text = data.decode("ascii")  # UnicodeDecodeError

    def p_err_27():
        """Unicode encode error"""
        text = "hello 😊"
        encoded = text.encode("ascii")  # UnicodeEncodeError

    def p_err_28():
        """String method on non-string"""
        x = 42
        result = x.upper()  # AttributeError

    def p_err_29():
        """Join on non-iterable"""
        result = "-".join(42)  # TypeError

    def p_err_30():
        """Split with invalid separator"""
        s = "hello"
        parts = s.split("")  # ValueError

    def p_err_31():
        """String formatting with missing placeholder"""
        template = "{name} is {age}"
        result = template.format(age=30)  # KeyError

    def p_err_32():
        """Regex compilation error"""
        import re
        pattern = re.compile("[Unclosed(")  # RegexError

    def p_err_33():
        """Find in non-string"""
        x = [1, 2, 3]
        idx = x.find(1)  # AttributeError

    def p_err_34():
        """String index method error"""
        s = "hello"
        idx = s.index("x")  # ValueError

    def p_err_35():
        """Replace on non-string"""
        x = 123
        result = x.replace("1", "2")  # AttributeError

    def p_err_36():
        """Bytes concatenation with string"""
        b = b"hello"
        s = "world"
        result = b + s  # TypeError

    def p_err_37():
        """String multiplication with negative"""
        s = "test"
        result = s * -5  # valid but can cascade

    def p_err_38():
        """Format string with wrong types"""
        template = "{:d}".format("not_int")  # ValueError

    def p_err_39():
        """Case conversion on non-ASCII"""
        s = 123
        result = s.lower()  # AttributeError

    def p_err_40():
        """Strip with invalid chars"""
        s = "hello"
        result = s.strip(123)  # TypeError

    def p_err_41():
        """F-string with invalid expression"""
        x = 5
        result = f"{x / 0}"  # ZeroDivisionError

    def p_err_42():
        """String slicing type mismatch"""
        s = "hello"
        result = s[:"5"]  # TypeError

    def p_err_43():
        """Startswith/endswith wrong type"""
        s = "hello"
        result = s.startswith(42)  # TypeError

    def p_err_44():
        """Center/ljust/rjust with invalid width"""
        s = "hello"
        result = s.center(-5)  # ValueError

    def p_err_45():
        """String ispredicate on non-string"""
        x = 123
        result = x.isdigit()  # AttributeError

    # ============================================================
    # PYTHON ERROR CLASS: I/O & File Errors (p_err_46 - p_err_65)
    # ============================================================

    def p_err_46():
        """File not found"""
        with open("nonexistent_file_xyz.txt", "r") as f:
            data = f.read()

    def p_err_47():
        """Directory not found"""
        import os
        os.chdir("/nonexistent/directory/path")

    def p_err_48():
        """Permission denied on file"""
        try:
            with open("/root/secret.txt", "w") as f:
                f.write("data")
        except PermissionError:
            raise

    def p_err_49():
        """JSON parsing error"""
        import json
        data = json.loads('{"incomplete": ')

    def p_err_50():
        """CSV parsing error"""
        import csv
        import io
        data = 'a,b,c\n1,2'
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader)
        for row in rows:
            _ = row["missing_column"]

    def p_err_51():
        """YAML parsing error (if pyyaml installed)"""
        import yaml
        yaml.safe_load("{invalid: [yaml")

    def p_err_52():
        """Pickle protocol error"""
        import pickle
        data = pickle.loads(b"invalid_pickle_data")

    def p_err_53():
        """File already exists error"""
        import os
        try:
            os.open("/tmp/test", os.O_CREAT | os.O_EXCL)
            os.open("/tmp/test", os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            raise

    def p_err_54():
        """Write to closed file"""
        f = open(__file__, "r")
        f.close()
        f.write("data")

    def p_err_55():
        """Read from write-only file"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w") as f:
            data = f.read()

    def p_err_56():
        """Seek in non-seekable file"""
        import io
        f = io.StringIO("data")
        f.close()
        f.seek(0)

    def p_err_57():
        """Write bytes to text file"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write(b"bytes")

    def p_err_58():
        """Encoding error in file write"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii") as f:
            f.write("hello 😊")

    def p_err_59():
        """Directory operations on file"""
        import os
        with open(__file__, "r") as f:
            pass
        os.listdir(__file__)

    def p_err_60():
        """Remove nonexistent file"""
        import os
        os.remove("/nonexistent/file.txt")

    def p_err_61():
        """Rename to existing file"""
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            path1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            path2 = f2.name
        os.rename(path1, path2)

    def p_err_62():
        """Chmod invalid permission"""
        import os
        os.chmod(__file__, "invalid")

    def p_err_63():
        """Path normalization error"""
        import pathlib
        p = pathlib.Path("/invalid\0path")
        p.resolve()

    def p_err_64():
        """Open with invalid mode"""
        with open(__file__, "q") as f:
            pass

    def p_err_65():
        """Read from binary file as text"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="wb") as f:
            f.write(b"\xff\xfe")
            f.seek(0)
        with open(f.name, "r") as f:
            data = f.read()

    # ============================================================
    # PYTHON ERROR CLASS: Arithmetic & Math Errors (p_err_66 - p_err_85)
    # ============================================================

    def p_err_66():
        """Division by zero"""
        result = 1 / 0

    def p_err_67():
        """Modulo by zero"""
        result = 5 % 0

    def p_err_68():
        """Divmod with zero divisor"""
        result = divmod(5, 0)

    def p_err_69():
        """Power with invalid exponent"""
        result = (0) ** (-1)

    def p_err_70():
        """Bitwise operation on float"""
        result = 5.5 & 3

    def p_err_71():
        """Left shift with negative"""
        result = 5 << -1

    def p_err_72():
        """Right shift with non-integer"""
        result = 5 >> 2.5

    def p_err_73():
        """Negative square root (math domain)"""
        import math
        result = math.sqrt(-1)

    def p_err_74():
        """Log of zero (math domain)"""
        import math
        result = math.log(0)

    def p_err_75():
        """Asin out of domain"""
        import math
        result = math.asin(2.0)

    def p_err_76():
        """Factorial of negative"""
        import math
        result = math.factorial(-5)

    def p_err_77():
        """Factorial of non-integer"""
        import math
        result = math.factorial(5.5)

    def p_err_78():
        """Complex number operations error"""
        x = complex(1, 2)
        result = x < 5  # TypeError

    def p_err_79():
        """Overflow in exponentiation"""
        result = 10 ** 10000000

    def p_err_80():
        """Trigonometric function with wrong type"""
        import math
        result = math.sin("90")

    def p_err_81():
        """Decimal precision loss"""
        from decimal import Decimal, getcontext
        getcontext().prec = 1
        result = Decimal("1") / Decimal("3")
        if result == 0:
            _ = 1 / 0

    def p_err_82():
        """Float to fraction with non-numeric"""
        from fractions import Fraction
        f = Fraction("not_a_number")

    def p_err_83():
        """Gcd with non-integer"""
        import math
        result = math.gcd(5.5, 3)

    def p_err_84():
        """Accumulation error in calculations"""
        result = 0
        for i in range(1000000):
            result += 0.1
        if result != 100000:
            _ = 1 / (result - 100000)

    def p_err_85():
        """Integer overflow (Python handles, but can cause issues)"""
        x = 10 ** 1000
        if x > 0:
            y = x // 0

    # ============================================================
    # PYTHON ERROR CLASS: Iteration & Control Flow (p_err_86 - p_err_100)
    # ============================================================

    def p_err_86():
        """Break outside loop"""
        def func():
            break
        func()

    def p_err_87():
        """Continue outside loop"""
        def func():
            continue
        func()

    def p_err_88():
        """Iterator exhausted"""
        it = iter([1, 2, 3])
        for _ in range(5):
            value = next(it)

    def p_err_89():
        """Generator already executing"""
        def gen():
            yield 1
        g = gen()
        it = iter(g)
        next(it)
        g.send(None)
        g.send(None)

    def p_err_90():
        """StopIteration propagation"""
        def process_items():
            items = iter([1, 2, 3])
            while True:
                yield next(items)
        
        gen = process_items()
        for i in range(10):
            val = next(gen)

    def p_err_91():
        """Return in generator"""
        def gen():
            return 42
            yield 1
        
        g = gen()
        try:
            while True:
                next(g)
        except StopIteration as e:
            if e.value is None:
                raise

    def p_err_92():
        """Yield in finally propagation"""
        def gen():
            try:
                yield 1
            finally:
                yield 2
        
        g = gen()
        next(g)

    def p_err_93():
        """Exception in except block"""
        try:
            raise ValueError("original")
        except ValueError:
            raise RuntimeError("handler error") from None

    def p_err_94():
        """Re-raising wrong exception"""
        try:
            raise ValueError("first")
        except:
            raise TypeError("second")

    def p_err_95():
        """Finally block exception handling"""
        try:
            try:
                raise ValueError("inner")
            finally:
                raise RuntimeError("finally")
        except RuntimeError:
            pass

    def p_err_96():
        """Context manager error propagation"""
        class BadContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                raise RuntimeError("exit error")
        
        with BadContext() as bc:
            pass

    def p_err_97():
        """Multiple exception contexts"""
        try:
            try:
                raise ValueError("first")
            except ValueError:
                raise TypeError("second")
        except TypeError:
            raise RuntimeError("third")

    def p_err_98():
        """Infinite recursion"""
        def recursive():
            return recursive()
        recursive()

    def p_err_99():
        """Recursion with exception"""
        def func(n):
            if n == 0:
                raise ValueError("base case")
            return func(n - 1)
        func(100)

    def p_err_100():
        """Context manager cleanup failure"""
        class FailingContext:
            def __init__(self, fail_on="both"):
                self.fail_on = fail_on
            
            def __enter__(self):
                if self.fail_on in ("enter", "both"):
                    raise RuntimeError("enter failed")
                return self
            
            def __exit__(self, *args):
                if self.fail_on in ("exit", "both"):
                    raise RuntimeError("exit failed")
        
        with FailingContext(fail_on="both"):
            pass

    # ============================================================
    # REGISTRATION ROUTES
    # ============================================================

    @app.get("/python/run-all")
    def python_run_all():
        """Execute all Python error scenarios"""
        functions = [
            # Type & Data Structure Errors
            p_err_01, p_err_02, p_err_03, p_err_04, p_err_05,
            p_err_06, p_err_07, p_err_08, p_err_09, p_err_10,
            p_err_11, p_err_12, p_err_13, p_err_14, p_err_15,
            p_err_16, p_err_17, p_err_18, p_err_19, p_err_20,
            p_err_21, p_err_22, p_err_23, p_err_24, p_err_25,
            # String & Encoding Errors
            p_err_26, p_err_27, p_err_28, p_err_29, p_err_30,
            p_err_31, p_err_32, p_err_33, p_err_34, p_err_35,
            p_err_36, p_err_37, p_err_38, p_err_39, p_err_40,
            p_err_41, p_err_42, p_err_43, p_err_44, p_err_45,
            # I/O & File Errors
            p_err_46, p_err_47, p_err_48, p_err_49, p_err_50,
            p_err_51, p_err_52, p_err_53, p_err_54, p_err_55,
            p_err_56, p_err_57, p_err_58, p_err_59, p_err_60,
            p_err_61, p_err_62, p_err_63, p_err_64, p_err_65,
            # Arithmetic & Math Errors
            p_err_66, p_err_67, p_err_68, p_err_69, p_err_70,
            p_err_71, p_err_72, p_err_73, p_err_74, p_err_75,
            p_err_76, p_err_77, p_err_78, p_err_79, p_err_80,
            p_err_81, p_err_82, p_err_83, p_err_84, p_err_85,
            # Iteration & Control Flow
            p_err_86, p_err_87, p_err_88, p_err_89, p_err_90,
            p_err_91, p_err_92, p_err_93, p_err_94, p_err_95,
            p_err_96, p_err_97, p_err_98, p_err_99, p_err_100,
        ]

        for fn in functions:
            try:
                fn()
            except Exception:
                log(clean_traceback(), level="ERROR")

        return {"status": "done", "class": "python", "executed": len(functions)}
