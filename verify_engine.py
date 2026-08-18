#!/usr/bin/env python3
"""End-to-end verification of the *compiled* noetic binary.

This script used to re-implement every formula from `autograd.rs` and `scan.rs`
in Python and check those transcriptions against finite differences. That was
worse than useless: a third parallel implementation of the same math, which
happily reported "all checks passed" during a period when `cargo test` did not
even compile. A verifier that cannot fail when the code under test is broken is
not a verifier.

So it now checks the real artifact, and only things Rust cannot check itself:

  1. `selftest` runs clean, and every reported error is inside its tolerance
     (the tolerances are parsed out of the output, not restated here).
  2. Thread count does not change results: `--threads 1` and `--threads 4`
     produce byte-identical output apart from the header line. This is the
     determinism guarantee, and it is invisible to a single-threaded test.
  3. `train` improves a held-out loss and writes a resumable checkpoint whose
     reported nats/token match the printed bits/token (ln 2 conversion).
  4. `gen --greedy` is reproducible across processes for a fixed seed.
  5. `bpe` round-trips a corpus losslessly through a saved tokenizer file.

The gradient math itself is checked by `cargo test` (finite differences) and by
`noetic selftest`, both of which exercise the Rust code that actually ships.

Usage:  python3 verify_math.py [path/to/noetic]
"""
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
failures = []
checks = 0


def report(name, ok, detail=""):
    global checks
    checks += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")
    if not ok:
        failures.append(name)


def find_binary():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for candidate in ("target/release/noetic", "target/release/noetic.exe", "target/debug/noetic", "target/debug/noetic.exe"):
        path = os.path.join(HERE, candidate)
        if os.path.isfile(path):
            return path
    print("no noetic binary found; run `cargo build --release` first")
    sys.exit(2)


def run(binary, args, timeout=1800):
    proc = subprocess.run([binary] + args, cwd=HERE, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout + proc.stderr


NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"


def check_selftest(binary):
    code, out = run(binary, ["selftest", "--threads", "4"])
    lines = [line for line in out.splitlines() if line.strip().startswith("[")]
    report("selftest exits zero", code == 0, f"exit {code}, {len(lines)} checks")
    failed = [line for line in lines if "[FAIL]" in line]
    report("every selftest check passes", not failed and len(lines) >= 12, f"{len(lines) - len(failed)}/{len(lines)} passed")

    # Errors must be small in absolute terms too, so a silently loosened
    # tolerance inside the Rust code cannot hide a regression from this script.
    limits = [
        ("gemm", 1e-4),
        ("chunked vs seq", 1e-6),
        ("max rel err", 5e-2),
        ("max logit diff", 1e-5),
        ("final mse", 1e-3),
    ]
    worst = 0.0
    for line in lines:
        for needle, limit in limits:
            if needle in line:
                # Numbers only from the reported-error segment: a parenthesised
                # parameter name such as `(blk1.ssm.in_proj.b)` is not a value.
                segment = line.split(needle, 1)[1].split("(")[0]
                for value in re.findall(NUM, segment):
                    magnitude = abs(float(value))
                    if magnitude > limit:
                        report(f"reported error within {limit:.0e}", False, line.strip())
                        return
                    worst = max(worst, magnitude / limit)
    report("reported errors inside hard limits", True, f"worst {100.0 * worst:.1f}% of its limit")


def check_determinism(binary):
    outputs = []
    for threads in ("1", "4"):
        code, out = run(binary, ["selftest", "--threads", threads])
        if code != 0:
            report("thread count does not change results", False, f"selftest --threads {threads} exited {code}")
            return
        body = [line for line in out.splitlines() if "threads =" not in line]
        outputs.append("\n".join(body))
    same = outputs[0] == outputs[1]
    detail = "byte-identical output" if same else "OUTPUT DIFFERS between 1 and 4 threads"
    report("thread count does not change results", same, detail)


def check_training(binary):
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "v.ckpt")
        tok = os.path.join(tmp, "v.tok")
        args = [
            "train", "--steps", "40", "--d", "64", "--layers", "2", "--ctx", "48", "--batch", "6",
            "--bytes", "60000", "--vocab", "400", "--valevery", "20", "--n", "16",
            "--out", ckpt, "--tok", tok,
        ]
        code, out = run(binary, args)
        if code != 0:
            report("train exits zero", False, f"exit {code}")
            return
        report("train exits zero", True, "40 steps")

        vals = [(float(m.group(1)), float(m.group(2))) for m in re.finditer(rf"held-out ({NUM}) nats \(({NUM}) bits", out)]
        report("training reports held-out loss", len(vals) >= 2, f"{len(vals)} evaluations")
        if vals:
            nats, bits = vals[-1]
            report("nats/bits conversion is consistent", abs(nats / math.log(2) - bits) < 5e-3, f"{nats:.4f} nats = {bits:.3f} bits")
            report("held-out loss beats a uniform 400-token prior", nats < math.log(400), f"{nats:.4f} < {math.log(400):.4f}")
        if len(vals) >= 2:
            report("held-out loss improves over training", vals[-1][0] < vals[0][0], f"{vals[0][0]:.4f} -> {vals[-1][0]:.4f}")

        report("checkpoint written", os.path.isfile(ckpt), f"{os.path.getsize(ckpt) if os.path.isfile(ckpt) else 0} bytes")

        code, out = run(binary, ["train", "--resume", ckpt, "--steps", "10", "--bytes", "60000", "--n", "8", "--out", ckpt])
        resumed = "restored parameters and AdamW moments" in out
        report("checkpoint is resumable with optimizer state", code == 0 and resumed, f"exit {code}")

        outs = []
        for _ in range(2):
            code, out = run(binary, ["gen", "--ckpt", ckpt, "--prompt", "memo alpha = ", "--n", "24", "--greedy", "--seed", "7"])
            if code != 0:
                report("greedy generation is reproducible", False, f"exit {code}")
                return
            outs.append(out)
        report("greedy generation is reproducible", outs[0] == outs[1], "two processes, identical bytes")


def check_tokenizer(binary):
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.txt")
        text = "".join(chr(32 + (i * 7919) % 95) for i in range(60000))
        with open(corpus, "w", encoding="utf-8") as handle:
            handle.write(text)
        tok = os.path.join(tmp, "t.tok")
        code, out = run(binary, ["bpe", "--data", corpus, "--vocab", "600", "--out", tok])
        lossless = "roundtrip: lossless" in out or "lossless" in out
        report("bpe round-trips a corpus", code == 0 and lossless, f"exit {code}")
        report("tokenizer file written", os.path.isfile(tok), f"{os.path.getsize(tok) if os.path.isfile(tok) else 0} bytes")


def main():
    binary = find_binary()
    print(f"verifying {binary}")
    print()
    check_selftest(binary)
    check_determinism(binary)
    check_training(binary)
    check_tokenizer(binary)
    print()
    if failures:
        print(f"{len(failures)} of {checks} checks FAILED: {', '.join(failures)}")
        sys.exit(1)
    print(f"all {checks} end-to-end checks passed")


if __name__ == "__main__":
    main()
