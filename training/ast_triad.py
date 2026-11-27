"""Seeded Python expressions, control flow and their executable teaching examples."""
import ast
import json
import random
import sys
import tempfile
from pathlib import Path


KINDS = ("AST deduction", "AST abduction", "AST induction", "AST next line", "AST changed variables")
SAMPLES = 9
TRACE_LIMIT = 40


def _expr(rng, depth):
    if not depth or rng.randrange(4) == 0:
        return ast.parse(rng.choice(("x", "i", str(rng.randint(1, 5)))), mode="eval").body
    op = rng.choice((ast.Add, ast.Sub, ast.BitXor, ast.Mod))
    rhs = ast.Constant(rng.randint(2, 7)) if op is ast.Mod else _expr(rng, depth - 1)
    return ast.BinOp(_expr(rng, depth - 1), op(), rhs)


def _stage(rng, name, level):
    """Construct a bounded loop. Its state, expression trees and branch bodies vary independently."""
    state = "int" if level == 1 else rng.choice(("list", "dict", "set"))
    initial = {"int": str(rng.randint(0, 5)), "list": "[]", "dict": "{}", "set": "set()"}[state]

    def update():
        value = ast.unparse(_expr(rng, rng.randint(1, 2)))
        if state == "list":
            return ast.parse(f"s.append({value})").body[0]
        if state == "set":
            return ast.parse(f"s.add({value})").body[0]
        if state == "dict":
            return ast.parse(f"s[i % {rng.randint(2, 4)}] = {value}").body[0]
        return ast.parse(f"s {rng.choice(('+=', '-=', '^='))} {value}").body[0]

    test = ast.parse(f"({ast.unparse(_expr(rng, 1))}) % {rng.randint(2, 5)}", mode="eval").body
    branch = ast.If(test=test, body=[update()], orelse=[update()] if rng.randrange(2) else [])
    stop = rng.randint(2, 5)
    if rng.randrange(2):
        loop = ast.parse(f"for i in range({stop}): pass").body
        loop[0].body = [branch]
    else:
        loop = ast.parse(f"i = 0\nwhile i < {stop}:\n    i += 1").body
        loop[1].body.insert(0, branch)
    reduction = {"int": "s", "list": "sum(s)", "dict": "sum(s.values())", "set": "sum(s)"}[state]
    fn = ast.parse(f"def {name}(x):\n    s = {initial}\n    return x + {reduction}").body[0]
    fn.body[1:1] = loop
    return fn


def generate(seed, level):
    """Generate source without executing it. The only loop bounds come from the grammar."""
    rng = random.Random(seed)
    if level not in (1, 2, 3):
        raise ValueError("expected a structural level from 1 through 3")
    if level == 3:
        outer, inner = rng.sample(("f", "g"), 2)
        stage = _stage(rng, outer, rng.choice((1, 2)))
        a, b = rng.randint(2, 4), rng.randint(1, 5)
        short = rng.choice((f"{a} * x + {b}", f"(x - {b}) // {a}", f"abs(x + {a}) ^ {b}"))
        tree = [stage, *ast.parse(f"def {inner}(x): return {short}\ndef solve(x): return f(g(x))").body]
    else:
        tree = [_stage(rng, "solve", level)]
    # ast.unparse separates definitions with blank lines. Line numbers in prompts and traces use this exact source.
    source = "\n".join(line for line in ast.unparse(ast.Module(tree, type_ignores=[])).splitlines() if line)
    mutant = source.replace("f(g(x))", "g(f(x))") if level == 3 else source.replace("return x + ", "return x - ")
    inputs = rng.sample(range(-19, 20), SAMPLES)
    return source, mutant, inputs


def _observe(solve, x):
    """State at each line event (before execution), relative to the previous event in that invocation."""
    trace, previous = [], {}

    def observe(frame, event, arg):
        if frame.f_code.co_filename != "<lesson>":
            return None
        if event == "return":
            previous.pop(frame, None)
        if event == "line" and len(trace) < TRACE_LIMIT:
            now = {k: repr(v) for k, v in sorted(frame.f_locals.items()) if isinstance(v, (int, list, dict, set))}
            old = previous.get(frame, {})
            diff = ",".join(f"{k}={v}" for k, v in now.items() if v != old.get(k))
            trace.append((frame.f_lineno, diff or "-"))
            previous[frame] = now
        return observe

    saved = sys.gettrace()
    sys.settrace(observe)
    try:
        value = solve(x)
    finally:
        sys.settrace(saved)
    return value, trace


def _functions(source):
    scope = {}
    exec(compile(source, "<lesson>", "exec"), scope)
    return scope


def _checked(seed, level):
    """Sandbox worker for one closed-grammar candidate. Hidden inputs are drawn separately from the five shown."""
    source, mutant, inputs = generate(seed, level)
    functions = _functions(source)
    solve = functions["solve"]
    outputs = [solve(x) for x in inputs]
    other = _functions(mutant)["solve"]
    if len(set(outputs)) < 2 or outputs == inputs:
        return None
    # One wrong execution must actually distinguish the mutation on shown AND unseen inputs.
    if not any(other(x) != y for x, y in zip(inputs[5:], outputs[5:])):
        return None
    witness = next((i for i in range(5) if other(inputs[i]) != outputs[i]), None)
    if witness is None:
        return None
    if level == 3 and any(all(fn(x) == y for x, y in zip(inputs[5:], outputs[5:])) for fn in (functions["f"], functions["g"])):
        return None
    x, y = inputs[witness], outputs[witness]
    wrong_input = next((test for test, answer in zip(inputs[:5], outputs[:5]) if answer != y), None)
    if wrong_input is None:
        return None
    visible = dict(zip(inputs[:5], outputs[:5]))
    # This memorizer passes every shown example. Only label it wrong after actually calling it on unseen inputs.
    lookup = f"def solve(x):\n    return {visible!r}.get(x, x)"
    memorize = _functions(lookup)["solve"]
    if not any(memorize(i) != o for i, o in zip(inputs[5:], outputs[5:])):
        return None
    result, trace = _observe(solve, x)
    if repr(result) != repr(y):
        return None
    return {"seed": seed, "level": level, "source": source, "inputs": inputs, "outputs": outputs,
            "x": x, "y": y, "wrong_y": other(x), "wrong_x": wrong_input, "lookup": lookup, "trace": trace}


def _batch(seed, count):
    out, seen = [], set()
    for i in range(count):
        for attempt in range(32):
            item = _checked(f"ast:{seed}:{i}:{attempt}", i % 3 + 1)
            if item is not None and item["source"] not in seen:
                seen.add(item["source"])
                item["attempt"] = attempt + 1
                out.append(item)
                break
        else:
            raise ValueError(f"no discriminating program for seed {seed}, index {i}")
    return out


def verified(seed, count):
    """Two fresh interpreter processes verify every result and trace, using the same runtime limits as member code."""
    from team.tools.sandbox import run_python
    worker = Path(__file__).read_text()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.json"
        first = None
        for run in range(2):
            result = run_python(worker, tmp, name="ast_worker.py", args=(str(seed), str(count)), timeout=30)
            if result["returncode"]:
                raise RuntimeError(f"AST verification run {run + 1}: {result['stderr']}")
            data = path.read_bytes()
            path.unlink()
            if first is not None and data != first:
                raise ValueError("AST verification runs disagreed")
            first = data
    return json.loads(first)


def rows(item):
    """Five questions, each with a first-program SFT row and a checked preference pair."""
    p, x, y = item["source"], item["x"], item["y"]
    visible = list(zip(item["inputs"][:5], item["outputs"][:5]))
    lines = item["trace"]
    # Keep the complete trace in the verifier. One short predecessor is enough to locate a probe
    # (including on a loop back edge) without charging the long state twice in a DPO row.
    choices = [i for i, (line, diff) in enumerate(lines[1:], 1) if line != lines[i - 1][0]
               and diff != "-" and not diff.startswith("x=") and len(diff) <= 30 and len(lines[i - 1][1]) <= 16]
    k = random.Random(item["seed"] + ":probe").choice(choices)
    line, diff = lines[k]
    prev_line, prev_diff = lines[k - 1]
    evidence = f"solve({x}), before-line event {k} (line | changed locals): {prev_line} | {prev_diff}"

    def prints(value):
        return f"print({value!r})"

    questions = (
        (f"{p}\nPrint the value of solve({x}).", prints(y), prints(item["wrong_y"])),
        (f"{p}\nPrint one integer x in [-19,19] such that solve(x) == {y}.", prints(x), prints(item["wrong_x"])),
        (f"Infer an integer function solve(x) from these (x, result) examples. It must generalize.\n{visible}",
         p, item["lookup"]),
        (f"{p}\n{evidence}\nPrint the next line number.", prints(line), prints(prev_line)),
        (f"{p}\n{evidence}\nPrint the next changed locals (line {line}).",
         prints(diff), prints("-")),
    )
    for kind, (task, chosen, rejected) in zip(KINDS, questions):
        head = [{"role": "system", "content": "Reply with one complete Python program in a ```python block."},
                {"role": "user", "content": "TASK: " + task}]
        row = {"task": task, "kind": kind, "seed": item["seed"], "who": "lesson", "member": 0, "lesson": "ast_triad",
               "messages": head, "output": f"```python\n{chosen}\n```"}
        yield row | {"type": "first"}
        yield row | {"type": "pair", "rejected": f"```python\n{rejected}\n```", "rejected_who": "ast_triad"}


if __name__ == "__main__":  # trusted sandbox worker; accepts just the seed and number of draws
    Path("results.json").write_text(json.dumps(_batch(sys.argv[1], int(sys.argv[2]))))
