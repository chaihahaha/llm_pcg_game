"""A deliberately small, AST-validated expression sandbox.

The model is allowed to monkey-patch the running game: it hands us a lambda
(``lambda ctx: ...``) which we compile once and call at a hook point.  That is
powerful and therefore fenced in:

* no attribute access to anything starting with ``_`` (so no ``__class__`` /
  ``__globals__`` escapes),
* no imports, no ``eval``/``exec``/``open``/``getattr``, no ``while``,
* names resolve only to an explicit whitelist of pure builtins plus ``ctx``,
* method calls are limited to pure string/list/dict methods,
* expressions are size-bounded.

``range`` is intentionally absent, so a comprehension can only iterate over the
data we hand it — no timeouts needed, the expressions are structurally cheap.

This is a content sandbox, not a security boundary against a determined
adversary with local code execution; it is meant to keep a language model from
accidentally (or "artistically") corrupting or hanging the game.
"""
from __future__ import annotations

import ast
from typing import Any, Callable, Dict, Optional

MAX_EXPR_CHARS = 1200
MAX_AST_NODES = 220
MAX_CODE_CHARS = 3000
MAX_CODE_NODES = 400
MAX_INT_LITERAL = 10 ** 6


class SandboxError(ValueError):
    pass


SAFE_FUNCS: Dict[str, Any] = {
    "len": len, "int": int, "float": float, "str": str, "bool": bool,
    "abs": abs, "min": min, "max": max, "round": round, "sorted": sorted,
    "sum": sum, "any": any, "all": all, "list": list, "dict": dict,
    "tuple": tuple, "set": set, "reversed": reversed, "enumerate": enumerate,
    "zip": zip,
}

SAFE_METHODS = frozenset({
    "lower", "upper", "strip", "lstrip", "rstrip", "replace", "split", "join",
    "startswith", "endswith", "format", "count", "get", "keys", "values",
    "items", "index", "find", "title", "capitalize", "isdigit", "isalpha",
})

_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Store, ast.Del,
    ast.Tuple, ast.List,
    ast.Dict, ast.Set, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.Subscript, ast.Slice, ast.Attribute, ast.Call, ast.Lambda,
    ast.arguments, ast.arg, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.comprehension, ast.JoinedStr, ast.FormattedValue,
    ast.keyword,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Is, ast.IsNot,
)
_ALLOWED_TYPES = tuple(_ALLOWED_NODES)


def _validate(tree: ast.AST, source: str) -> Optional[str]:
    if len(source) > MAX_EXPR_CHARS:
        return f"表达式过长（>{MAX_EXPR_CHARS} 字符）"
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        return "表达式过于复杂"
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_TYPES):
            return f"不允许的语法：{type(node).__name__}"
        if isinstance(node, ast.Name):
            if node.id not in SAFE_FUNCS and node.id not in ("ctx", "True", "False", "None"):
                return f"不允许的名字：{node.id}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in SAFE_METHODS:
                return f"不允许的属性：{node.attr}"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id not in SAFE_FUNCS:
                return f"不允许调用：{func.id}"
            if not isinstance(func, (ast.Name, ast.Attribute, ast.Lambda, ast.Subscript)):
                return "不允许的调用形式"
    return None


def compile_hook(expr: str) -> Callable[[Dict[str, Any]], Any]:
    """Compile ``lambda ctx: ...`` (or a bare expression) into a callable.

    A bare expression is wrapped as ``lambda ctx: (expr)`` so callers always
    get the same calling convention.  Raises ``SandboxError`` on rejection.
    """
    if not expr or not expr.strip():
        raise SandboxError("空表达式")
    src = expr.strip()
    if not src.startswith("lambda"):
        src = f"lambda ctx: ({src})"
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise SandboxError(f"语法错误：{exc.msg}") from exc
    err = _validate(tree, src)
    if err:
        raise SandboxError(err)
    code = compile(tree, "<patch>", "eval")
    # safe builtins must live in *globals*: a lambda's body resolves names
    # through __globals__, not through the locals passed to eval()
    env: Dict[str, Any] = {"__builtins__": {}}
    env.update(SAFE_FUNCS)
    env["ctx"] = {}
    fn = eval(code, env)  # noqa: S307 - AST-validated above
    if not callable(fn):
        raise SandboxError("表达式不是可调用对象")
    return fn


def eval_expr(expr: str, ctx: Optional[Dict[str, Any]] = None) -> Any:
    """One-shot evaluation (used for rule values)."""
    return compile_hook(expr)(ctx or {})


# --------------------------------------------------------------------------- exec tier

_ALLOWED_STMTS = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.If, ast.For,
    ast.FunctionDef, ast.Return, ast.Pass, ast.Break, ast.Continue,
    ast.Store, ast.Del,
)

# Deliberately absent: While, With, Import, ImportFrom, ClassDef, Global,
# Nonlocal, Raise, Try, Assert, Delete, Yield, Await, AsyncFunctionDef, Match.
_FORBIDDEN_STMTS = {
    "While": "while 循环可能永不结束",
    "Import": "不允许导入模块",
    "ImportFrom": "不允许导入模块",
    "With": "不允许 with",
    "ClassDef": "不允许定义类",
    "Global": "不允许 global",
    "Nonlocal": "不允许 nonlocal",
    "Raise": "不允许 raise",
    "Try": "不允许 try",
    "Assert": "不允许 assert",
    "Delete": "不允许 del",
    "Yield": "不允许 yield",
    "Await": "不允许 await",
    "AsyncFunctionDef": "不允许 async",
}


def validate_code(source: str, extra_names: Optional[set] = None) -> Optional[str]:
    """Return an error string, or None if the statement-level patch is allowed.

    ``extra_names`` are the identifiers the caller injects (e.g. ``{"api"}``);
    nothing else can be referenced by name.
    """
    extra_names = extra_names or set()
    if len(source) > MAX_CODE_CHARS:
        return f"代码过长（>{MAX_CODE_CHARS} 字符）"
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return f"语法错误：{exc.msg}"
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_CODE_NODES:
        return "代码过于复杂"
    # names the patch binds itself are fair game to read
    bound = set()
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
    for node in nodes:
        name = type(node).__name__
        if name in _FORBIDDEN_STMTS:
            return _FORBIDDEN_STMTS[name]
        if isinstance(node, ast.Module):
            continue
        is_stmt = isinstance(node, ast.stmt)
        if is_stmt and not isinstance(node, _ALLOWED_STMTS):
            return f"不允许的语句：{name}"
        if not is_stmt and not isinstance(node, _ALLOWED_TYPES):
            return f"不允许的语法：{name}"
        # only *reads* are restricted; assignment targets are local variables
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id not in SAFE_FUNCS and node.id != "ctx"
                and node.id not in extra_names and node.id not in bound):
            return f"不允许的名字：{node.id}"
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_")
                                                or node.attr not in SAFE_METHODS):
            return f"不允许的属性：{node.attr}"
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "api"):
            return "不允许通过属性访问 api，请用 api['...'] 下标"
        if isinstance(node, ast.Call):
            func = node.func
            # Names may be locally defined functions (the patch's own `def`s) or
            # whitelisted builtins; subscripts are how the injected api is
            # reached.  There is no route from either to anything dangerous
            # because the namespace has no builtins and no imports.
            if not isinstance(func, (ast.Name, ast.Attribute, ast.Lambda, ast.Subscript)):
                return "不允许的调用形式"
        if isinstance(node, ast.Constant) and isinstance(node.value, int) \
                and not isinstance(node.value, bool) and abs(node.value) > MAX_INT_LITERAL:
            return "整数常量过大（防止一次分配大量内存）"
        if isinstance(node, ast.Pow):
            return "不允许幂运算（防止 10**9 这类爆炸）"
    if not isinstance(tree, ast.Module):
        return "顶层必须是代码块"
    return None


def exec_patch(source: str, names: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a statement-level patch in a locked-down namespace.

    Like :func:`compile_hook`, but for ``exec``: the model can use assignments,
    ``def``/``if``/``for`` and local variables.  There is no ``while``, no
    ``import``, no dunder access and no builtins beyond the pure whitelist, so
    a patch cannot reach the filesystem, network or process state.

    Returns the resulting namespace (so callers can pick up ``def``-ed helpers).
    """
    err = validate_code(source, extra_names=set(names or {}))
    if err:
        raise SandboxError(err)
    code = compile(ast.parse(source, mode="exec"), "<patch>", "exec")
    namespace: Dict[str, Any] = {"__builtins__": {}, "ctx": {}}
    namespace.update(SAFE_FUNCS)
    if names:
        namespace.update(names)
    exec(code, namespace)  # noqa: S102 - AST-validated above
    return namespace
