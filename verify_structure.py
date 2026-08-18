#!/usr/bin/env python3
"""Static structural checker for the noetic crate (no rustc required).

Substitutes for the compiler's cheapest checks:

  1. delimiter balance per file (comment / string / char-literal aware)
  2. `pub mod x;` declarations in the crate root resolve to x.rs, and every
     Rust file is declared
  3. `use crate::m::{a, b}` symbols exist as pub items in module m
  4. every `g.method(…)` call on a Graph exists in autograd.rs
  5. format-string placeholder count matches argument count for
     println!/print!/format!/panic!/writeln!/write!/eprintln!

Run from any directory:  python3 verify_structure.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE if os.path.isfile(os.path.join(HERE, "Cargo.toml")) else os.path.dirname(HERE)
STANDARD_SRC = os.path.join(REPO, "src")
SRC = STANDARD_SRC if os.path.isfile(os.path.join(STANDARD_SRC, "main.rs")) else REPO
# Sources live at the repo root; the crate root that declares the modules is
# lib.rs (main.rs is a three-line shim over the library).
ROOT_FILE = "lib.rs"
files = sorted(f for f in os.listdir(SRC) if f.endswith(".rs"))
problems = []


def strip_code(text):
    """Blank out comments, strings and char literals, preserving offsets."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
            continue
        if c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(" " * (j - i))
            i = j
            continue
        if c == "r" and nxt == '"':
            j = text.find('"', i + 2)
            j = n if j == -1 else j + 1
            out.append(" " * (j - i))
            i = j
            continue
        if c == "b" and nxt == '"':
            i += 1
            out.append(" ")
            continue
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            out.append(" " * (j - i))
            i = j
            continue
        if c == "'":
            m = re.match(r"'(\\.|[^\\'])'", text[i:])
            if m:
                out.append(" " * len(m.group(0)))
                i += len(m.group(0))
                continue
            out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


if ROOT_FILE not in files:
    print(f"{ROOT_FILE} not found under {SRC}")
    sys.exit(1)

code = {}
for f in files:
    raw = open(os.path.join(SRC, f), encoding="utf-8").read()
    code[f] = (raw, strip_code(raw))

# ---- 1. delimiter balance -------------------------------------------------
for f in files:
    raw, c = code[f]
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    line = 1
    for ch in c:
        if ch == "\n":
            line += 1
        elif ch in "([{":
            stack.append((ch, line))
        elif ch in ")]}":
            if not stack:
                problems.append(f"{f}:{line}: unmatched closing '{ch}'")
                break
            open_ch, open_line = stack.pop()
            if open_ch != pairs[ch]:
                problems.append(
                    f"{f}:{line}: '{ch}' closes '{open_ch}' opened at line {open_line}"
                )
                break
    if stack:
        problems.append(
            f"{f}: {len(stack)} unclosed delimiters, first at line {stack[0][1]} '{stack[0][0]}'"
        )

# ---- 2. mod declarations --------------------------------------------------
root_raw, root_code = code[ROOT_FILE]
mods = re.findall(r"^\s*(?:pub\s+)?mod\s+([a-z_0-9]+)\s*;", root_code, re.M)
for m in mods:
    if m + ".rs" not in files:
        problems.append(f"{ROOT_FILE}: mod {m}; has no {m}.rs")
for f in files:
    if f in (ROOT_FILE, "main.rs"):
        continue
    if f[:-3] not in mods:
        problems.append(f"{f} is never declared with `mod {f[:-3]};` in {ROOT_FILE}")

# ---- 3. pub item inventory + use checking --------------------------------
pub_items = {}
for f in files:
    raw, c = code[f]
    names = set()
    for pat in [
        r"pub\s+fn\s+([A-Za-z_][A-Za-z_0-9]*)",
        r"pub\s+struct\s+([A-Za-z_][A-Za-z_0-9]*)",
        r"pub\s+enum\s+([A-Za-z_][A-Za-z_0-9]*)",
        r"pub\s+type\s+([A-Za-z_][A-Za-z_0-9]*)",
        r"pub\s+const\s+([A-Za-z_][A-Za-z_0-9]*)",
        r"pub\s+trait\s+([A-Za-z_][A-Za-z_0-9]*)",
    ]:
        names.update(re.findall(pat, c))
    pub_items[f[:-3]] = names

use_re = re.compile(
    r"use\s+crate::([a-z_0-9]+)::(\{[^}]*\}|[A-Za-z_][A-Za-z_0-9]*)\s*;"
)
for f in files:
    raw, c = code[f]
    for mod_name, what in use_re.findall(c):
        if mod_name not in pub_items:
            problems.append(f"{f}: use crate::{mod_name}::... but no such module")
            continue
        if what.startswith("{"):
            items = [x.strip() for x in what[1:-1].split(",") if x.strip()]
        else:
            items = [what]
        for it in items:
            base = it.split(" as ")[0].strip()
            if base in ("self", "*"):
                continue
            if base not in pub_items[mod_name]:
                problems.append(f"{f}: `{base}` is not a pub item of {mod_name}.rs")

# ---- 4. Graph method calls ------------------------------------------------
graph_methods = set(re.findall(r"pub\s+fn\s+([a-z_0-9]+)", code["autograd.rs"][1]))


def struct_fields(source, name):
    """Field names of `struct name { ... }`, read from the source rather than
    restated here, so renaming a field cannot leave this checker stale."""
    match = re.search(r"(?:pub\s+)?struct\s+" + name + r"\s*\{", source)
    if not match:
        return set()
    depth = 0
    end = len(source)
    for index in range(match.end() - 1, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    body = source[match.end() : end]
    return set(re.findall(r"^\s*(?:pub\s+)?([a-z_0-9]+)\s*:", body, re.M))


graph_fields = struct_fields(code["autograd.rs"][1], "Graph")
if not graph_fields:
    problems.append("autograd.rs: could not find the Graph struct to read its fields")
skip_calls = {
    "clone", "len", "iter", "push", "to_vec", "as_str", "sort_by", "is_empty",
    "cmp", "unwrap_or",
}
for f in files:
    raw, c = code[f]
    for m in re.finditer(r"\bg\.([a-z_0-9]+)\s*\(", c):
        name = m.group(1)
        if name in graph_methods or name in skip_calls:
            continue
        line = c[: m.start()].count("\n") + 1
        problems.append(f"{f}:{line}: g.{name}(...) is not a Graph method")
    for m in re.finditer(r"\bg\.([a-z_0-9]+)\b(?!\s*\()", c):
        name = m.group(1)
        if name in graph_fields or name in graph_methods:
            continue
        line = c[: m.start()].count("\n") + 1
        problems.append(f"{f}:{line}: g.{name} is not a Graph field")

# ---- 5. format placeholder arity ----------------------------------------
MACROS = (
    "println", "print", "format", "panic", "writeln", "write", "eprintln",
    "assert_eq", "assert",
)


def split_args(s):
    """Split a macro argument list on top-level commas (string / paren aware)."""
    args = []
    depth = 0
    cur = ""
    i = 0
    in_str = False
    while i < len(s):
        ch = s[i]
        if in_str:
            if ch == "\\":
                cur += s[i : i + 2]
                i += 2
                continue
            if ch == '"':
                in_str = False
            cur += ch
            i += 1
            continue
        if ch == '"':
            in_str = True
            cur += ch
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
            i += 1
            continue
        cur += ch
        i += 1
    if cur.strip():
        args.append(cur.strip())
    return args


for f in files:
    raw, c = code[f]
    for m in re.finditer(r"\b(" + "|".join(MACROS) + r")!\s*\(", raw):
        name = m.group(1)
        start = m.end()
        depth = 1
        i = start
        in_str = False
        while i < len(raw) and depth > 0:
            ch = raw[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            i += 1
        body = raw[start : i - 1]
        args = split_args(body)
        if not args:
            continue
        fmt_idx = 1 if name in ("writeln", "write") else 0
        if len(args) <= fmt_idx:
            continue
        fmt = args[fmt_idx]
        if not fmt.startswith('"'):
            continue
        inner = fmt[1:-1] if fmt.endswith('"') else fmt[1:]
        cleaned = inner.replace("{{", "").replace("}}", "")
        holes = re.findall(r"\{([^}]*)\}", cleaned)
        positional = [h for h in holes if not re.match(r"^[A-Za-z_]", h)]
        supplied = len(args) - fmt_idx - 1
        if name == "assert":
            continue
        if len(positional) != supplied:
            line = raw[: m.start()].count("\n") + 1
            problems.append(
                f"{f}:{line}: {name}! has {len(positional)} placeholder(s) but {supplied} argument(s)"
            )

# ---- report --------------------------------------------------------------
print("source directory:", SRC)
print("files checked:", len(files), "->", ", ".join(files))
print("total lines:", sum(code[f][0].count("\n") + 1 for f in files))
print("modules:", ", ".join(sorted(pub_items)))
print("pub items:", sum(len(v) for v in pub_items.values()))
print()
if problems:
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("no structural problems found")
