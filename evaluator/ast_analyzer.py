"""
ast_analyzer.py
===============
Parses Python code with tree-sitter and extracts structural signals.

Output format matches the rubric's expected AST signals:
{
    "has_recursion": bool,
    "has_base_case": bool,
    "loop_depth_max": int,
    "data_structures_used": [str],
    "builtin_calls": [str],
    "pattern_signatures": [str],
    "absent_patterns": [str]
}

Dependencies:
    pip install tree-sitter tree-sitter-python
"""

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())


def _make_parser():
    return Parser(PY_LANGUAGE)


def _text(node):
    return node.text.decode("utf-8") if node and node.text else ""


def _walk(node, node_type):
    results = []
    if node.type == node_type:
        results.append(node)
    for child in node.children:
        results.extend(_walk(child, node_type))
    return results


def _walk_multi(node, types_set):
    results = []
    if node.type in types_set:
        results.append(node)
    for child in node.children:
        results.extend(_walk_multi(child, types_set))
    return results


# ── FUNCTION DETECTION ────────────────────────────────────────────────────────

def _find_functions(root):
    funcs = []
    for fn in _walk(root, "function_definition"):
        name_node = fn.child_by_field_name("name")
        funcs.append({"name": _text(name_node), "node": fn})
    return funcs


# ── RECURSION ─────────────────────────────────────────────────────────────────

def _detect_recursion(root, functions):
    recursive_fns = []
    has_base = False
    for func in functions:
        calls = _walk(func["node"], "call")
        is_rec = any(
            _text(c.child_by_field_name("function")) == func["name"]
            for c in calls if c.child_by_field_name("function")
        )
        if is_rec:
            recursive_fns.append(func["name"])
            for if_node in _walk(func["node"], "if_statement"):
                if _walk(if_node, "return_statement"):
                    has_base = True
                    break
    return {
        "has_recursion": len(recursive_fns) > 0,
        "has_base_case": has_base,
        "recursive_functions": recursive_fns,
    }


# ── LOOPS ─────────────────────────────────────────────────────────────────────

def _loop_depth(node):
    depth = 1
    p = node.parent
    while p:
        if p.type in ("for_statement", "while_statement"):
            depth += 1
        p = p.parent
    return depth


def _detect_loops(root):
    loops = _walk_multi(root, {"for_statement", "while_statement"})
    if not loops:
        return {"max_depth": 0, "count": 0, "types": []}
    types = sorted({("for" if l.type == "for_statement" else "while") for l in loops})
    return {
        "max_depth": max(_loop_depth(l) for l in loops),
        "count": len(loops),
        "types": types,
    }


# ── IMPORTS ───────────────────────────────────────────────────────────────────

def _detect_imports(root):
    imports = []
    for node in _walk(root, "import_statement"):
        for ch in node.children:
            if ch.type == "dotted_name":
                imports.append(_text(ch))
    for node in _walk(root, "import_from_statement"):
        mod = node.child_by_field_name("module_name")
        if mod:
            mod_name = _text(mod)
            found_names = False
            for ch in node.children:
                if ch.type in ("dotted_name", "imported_name") and ch != mod:
                    imports.append(f"{mod_name}.{_text(ch)}")
                    found_names = True
            if not found_names:
                imports.append(mod_name)
    return imports


# ── DATA STRUCTURES ───────────────────────────────────────────────────────────

def _detect_data_structures(root, imports):
    types = set()
    calls = _walk(root, "call")

    for call in calls:
        callee = call.child_by_field_name("function")
        if not callee:
            continue
        ct = _text(callee)
        if ct in ("list", "dict", "set", "tuple", "frozenset"):
            types.add(ct)
        if ct in ("defaultdict", "deque", "Counter", "OrderedDict"):
            types.add(ct)
        if ct in ("heappush", "heappop", "heapify", "heapreplace"):
            types.add("heapq")
        if ct == "float":
            args = call.child_by_field_name("arguments")
            if args and "inf" in _text(args).lower():
                types.add("float_inf")
        if callee.type == "attribute":
            ft = _text(callee)
            if "heapq." in ft:
                types.add("heapq")
            if "bisect." in ft:
                types.add("bisect")

    if _walk(root, "list"):
        types.add("list")
    if _walk(root, "dictionary"):
        types.add("dict")
    if _walk(root, "set"):
        types.add("set")

    imp_str = " ".join(imports).lower()
    for kw, ds in [("heapq","heapq"),("deque","deque"),("defaultdict","defaultdict"),
                    ("counter","Counter"),("bisect","bisect"),("lru_cache","lru_cache"),("cache","lru_cache")]:
        if kw in imp_str:
            types.add(ds)

    details = []
    for comp in _walk(root, "comparison_operator"):
        if " in " in _text(comp) or " not in " in _text(comp):
            details.append("membership_test_found")
            break

    return {"types": sorted(types), "details": details}


# ── BUILTIN CALLS ─────────────────────────────────────────────────────────────

def _detect_builtin_calls(root):
    found = set()
    builtins = {"sorted","len","range","enumerate","zip","map","filter",
                "min","max","sum","abs","reversed","any","all"}
    methods = {"append","pop","popleft","appendleft","extend","push","add",
               "remove","discard","sort","reverse","insert","get","setdefault",
               "update","items","keys","values","index","count","copy",
               "heappush","heappop","heapify","heapreplace","union",
               "intersection","difference"}
    for call in _walk(root, "call"):
        callee = call.child_by_field_name("function")
        if not callee:
            continue
        ct = _text(callee)
        if ct in builtins:
            found.add(ct)
        if callee.type == "attribute":
            attr = callee.child_by_field_name("attribute")
            if attr and _text(attr) in methods:
                found.add(_text(attr))
    for dec in _walk(root, "decorator"):
        if "lru_cache" in _text(dec) or "cache" in _text(dec):
            found.add("lru_cache_decorator")
    return sorted(found)


# ── EARLY RETURNS ─────────────────────────────────────────────────────────────

def _detect_early_returns(root, functions):
    early = []
    for func in functions:
        body = func["node"].child_by_field_name("body")
        if not body:
            continue
        for i, ch in enumerate(body.children):
            if i > 3:
                break
            if ch.type == "if_statement" and _walk(ch, "return_statement"):
                early.append(func["name"])
                break
    return early


# ── COMPREHENSIONS ────────────────────────────────────────────────────────────

def _detect_comprehensions(root):
    types = set()
    for t, label in [("list_comprehension","list_comprehension"),
                     ("dictionary_comprehension","dict_comprehension"),
                     ("set_comprehension","set_comprehension"),
                     ("generator_expression","generator_expression")]:
        if _walk(root, t):
            types.add(label)
    return sorted(types)


# ── MAIN ENTRY ────────────────────────────────────────────────────────────────

def extract_signals(source_code: str) -> dict:
    """Parse Python source and extract all AST signals for rubric evaluation."""
    parser = _make_parser()
    tree = parser.parse(bytes(source_code, "utf-8"))
    root = tree.root_node

    functions = _find_functions(root)
    rec = _detect_recursion(root, functions)
    loops = _detect_loops(root)
    imports = _detect_imports(root)
    ds = _detect_data_structures(root, imports)
    calls = _detect_builtin_calls(root)
    early = _detect_early_returns(root, functions)
    comps = _detect_comprehensions(root)

    return {
        "has_recursion":        rec["has_recursion"],
        "has_base_case":        rec["has_base_case"],
        "recursive_functions":  rec["recursive_functions"],
        "loop_depth_max":       loops["max_depth"],
        "loop_types":           loops["types"],
        "loop_count":           loops["count"],
        "data_structures_used": ds["types"],
        "ds_details":           ds["details"],
        "builtin_calls":        calls,
        "imports":              imports,
        "has_early_returns":    len(early) > 0,
        "early_return_count":   len(early),
        "has_comprehensions":   len(comps) > 0,
        "comprehension_types":  comps,
        "function_count":       len(functions),
        "pattern_signatures":   [],   # filled by pattern_detector
        "absent_patterns":      [],   # filled by pattern_detector
    }
