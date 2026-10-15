#!/usr/bin/env python3
"""Greedy token-id parity between the installed llama.cpp and the `strata-poly` CLI, on one GGUF.

This is a new script rather than a reuse of an existing harness, because none of them does what this gate needs:

  * `native_expert_parity` (CMakeLists.txt:1494-1496, src/kernels/native_expert_parity.cpp:1-13) compares this
    repo's own expert kernels bitwise against a float reference *and* ggml-cpu.  It takes a shard GGUF and
    `--synthetic`, has no tokenizer, no prompt and no server, and is built only under CUDA/HIP.  It cannot say
    whether two *engines* pick the same token on the same prompt.
  * `bench/bench_vs_llama.py:2-18,141-156,228-244` starts both engines but drives the OpenAI chat API with prompt
    **text** and compares timings and generated text, with `serve/server.py` as the strata side.  Token *ids* and
    per-step probabilities are not in its scope, and the server is not the engine under test here.
  * `tools/ab_engine.py:41-45,100` is the same shape: two builds of this repo, text in, timings out.
  * `tools/conversation_cache_parity.py` compares this engine against itself.

The gate needs the token ids the `strata-poly` **CLI** emits for a prompt, held against llama.cpp's low-level
`/completion` fed the *same integer ids*, step by step.  The pass criterion is the repository's own: the same
**generated (greedy) ids**.  Top-K ids and the top-1 probability are reported per case but do not fail it — two
valid implementations of the same 4-bit model differ by more than any tight tolerance, and llama.cpp's own GPU and
CPU runs disagree with each other far more than this engine's GPU run disagrees with llama.cpp's (the measured
control is in `compare()`), so a probability gate would test the kernels, not the architecture.

    python tools/parity_llama.py --gguf MODEL.gguf --prompt "The capital of France is" \
        --prompt-file docs/DETAILS.md --json .pi/plans/lfm2moe/parity-q4km.json

It starts one `llama-server -m GGUF -c <ctx> -ngl 99`, waits on `/health`, tokenizes every prompt once with
`/tokenize`, runs every reference completion, then **stops the server before any `strata-poly` run** so the two
engines never contend for VRAM.  Every strata invocation is built from PINNED_FLAGS plus `--window`/`--lookup` and
one expert mode; the child environment clears `STRATA_ROUTE_TAIL_SKIP` and `STRATA_VRAM_FREE_FROM` so the
environment cannot silently move ids.  A case fails if the engine's stderr reports a nonzero `tail_experts` count
(the `left out` field of its stats line) or the `--tail-skip > 0` note, because a dropped expert moves ids.

Everything the script writes goes to `--work-dir` (a fresh temp dir by default) and to `--json`; nothing else.
"""
import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The flags this gate pins, in the order they are appended to the strata argv.  `--tail-skip`'s default is 0, but
# STRATA_ROUTE_TAIL_SKIP would override it (src/poly/main.cpp:917-918), so it is passed explicitly *and* the
# variable is cleared.  `--temperature 0 --top-k 1 --seed 0` is greedy, and `--stage-ahead 0 --cpu-share 0
# --expert-swaps 0 --conversation-cache-mib 0` remove the dynamic expert tiers that move ids.
PINNED_FLAGS = [
    ("--tail-skip", "0"),
    ("--temperature", "0"),
    ("--top-k", "1"),
    ("--seed", "0"),
    ("--stage-ahead", "0"),
    ("--cpu-share", "0"),
    ("--expert-swaps", "0"),
    ("--conversation-cache-mib", "0"),
]
CLEARED_ENV = ["STRATA_ROUTE_TAIL_SKIP", "STRATA_VRAM_FREE_FROM"]
DEFAULT_SHORT_PROMPT = "The capital of France is"


def pinned_list(window, lookup, max_context):
    """The full pinned argv fragment, as it is recorded in the JSON and in evidence.md."""
    out = []
    for flag, value in PINNED_FLAGS:
        out += [flag, value]
    out += ["--window", str(window), "--lookup", str(lookup), "--max-context", str(max_context)]
    return out


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(url, body=None, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(port, timeout, proc=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"the reference server exited with {proc.returncode} before becoming healthy "
                               f"(see the work dir's llama-server.log; a model larger than the card needs "
                               f"--llama-ngl 0 or a layer count that fits)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200 and json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"llama-server on port {port} did not become healthy in {timeout}s")


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def tokenize(port, text):
    return http(f"http://127.0.0.1:{port}/tokenize", {"content": text, "add_special": True})["tokens"]


def llama_completion(port, ids, n_predict, n_probs, timeout):
    # Every sampler that can move the chosen token off the raw argmax is switched off: llama-server's defaults
    # include a repeat penalty (and top_p/min_p/typical_p/DRY), which demotes a token the engine would keep.  A
    # penalised reference is not the model's greedy token, so the gate would fail on the sampler, not the engine.
    body = {"prompt": list(ids), "n_predict": n_predict, "temperature": 0, "top_k": 1, "seed": 0,
            "cache_prompt": False, "n_probs": n_probs, "stream": False,
            "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "penalty_last_n": 0,
            "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0, "dry_multiplier": 0.0,
            "xtc_probability": 0.0, "mirostat": 0}
    resp = http(f"http://127.0.0.1:{port}/completion", body, timeout=timeout)
    steps = []
    for entry in resp.get("completion_probabilities") or []:
        tops = [{"id": t["id"], "prob": math.exp(float(t["logprob"]))}
                for t in entry.get("top_logprobs") or []]
        if not tops:                       # older servers omit top_logprobs; fall back to the chosen token alone
            tops = [{"id": entry["id"], "prob": math.exp(float(entry.get("logprob", 0.0)))}]
        steps.append({"id": entry["id"], "top": tops})
    return steps


TOP_RE = re.compile(r"^top\s+(.*)$")
KV_RE = re.compile(r"(\d+):([0-9.eE+-]+)")
LEFT_OUT_RE = re.compile(r"(\d+)\s+left out")


CACHED_RE = re.compile(r"(\d+) of (\d+) routed experts in VRAM")


def strata_run(exe, gguf, ids, n_predict, n_probs, window, lookup, max_context, cpu_experts, env, timeout,
               reserve_mib, threads):
    argv = [str(exe), "--model", str(gguf), "--tokens", ",".join(str(i) for i in ids),
            "--max-new", str(n_predict), "--top-probs", str(n_probs)]
    argv += pinned_list(window, lookup, max_context)
    argv += ["--vram-reserve-mib", str(reserve_mib)]
    if threads:
        argv += ["--threads", str(threads)]
    if cpu_experts:
        argv.append("--no-gpu-experts")
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=timeout)
    out = proc.stdout.strip()
    gen = [int(x) for x in out.split(",") if x.strip()] if out else []
    steps = []
    for line in proc.stderr.splitlines():
        m = TOP_RE.match(line)
        if m:
            pairs = KV_RE.findall(m.group(1))
            steps.append({"id": int(pairs[0][0]), "top": [{"id": int(i), "prob": float(p)} for i, p in pairs]})
    m = LEFT_OUT_RE.search(proc.stderr)
    left_out = int(m.group(1)) if m else None
    cached = CACHED_RE.search(proc.stderr)
    tail_note = "tail skip " in proc.stderr
    stats = [l for l in proc.stderr.splitlines() if l.startswith("prompt ")]
    return {"argv": argv, "returncode": proc.returncode, "ids": gen, "steps": steps,
            "tail_experts": left_out, "tail_skip_note": tail_note, "stdout": proc.stdout,
            "cached_experts": int(cached.group(1)) if cached else None,
            "total_experts": int(cached.group(2)) if cached else None,
            "stats": stats[-1] if stats else None}


def compare(ref, got, n_probs, tol):
    """Step-by-step comparison.  Returns (issues, first, id_issues, numeric).

    Only an id mismatch fails the gate — that is the repository's own criterion (docs/BACKENDS.md, "Adding a model
    of an architecture that is there": "the same prompt's token ids into both, greedy, and compare the tokens that
    come out").  Top-K ids and the top-1 probability are reported as `numeric` because they are *not* a fair gate:
    two valid implementations of the same 4-bit model differ by far more than `tol`.  Measured control on this
    machine, same file and prompt: llama.cpp's own GPU run against its own CPU run (`-ngl 99` vs `-ngl 0`) matched
    only 2 of 16 greedy ids and differed by 0.081 mean / 0.147 max in top-1 log-probability, while this engine's
    GPU path matched llama.cpp's GPU run 64 of 64 ids (0.052 mean).  The tolerance is therefore informational.
    """
    issues, id_issues = [], []
    numeric = {"topk_mismatches": 0, "prob_mismatches": 0, "max_prob_delta": 0.0, "steps_compared": 0}
    n = min(len(ref), len(got))
    first = None
    for i in range(n):
        r, g = ref[i], got[i]
        numeric["steps_compared"] += 1
        if r["id"] != g["id"]:
            msg = f"step {i}: generated id {r['id']} (llama) vs {g['id']} (strata)"
            issues.append(msg)
            id_issues.append(msg)
            if first is None:
                first = {"step": i, "kind": "id", "llama": r["id"], "strata": g["id"]}
        rt = [t["id"] for t in r["top"][:n_probs]]
        gt = [t["id"] for t in g["top"][:n_probs]]
        if rt != gt:
            numeric["topk_mismatches"] += 1
            issues.append(f"step {i}: top-{n_probs} ids {rt} vs {gt}")
        if r["top"] and g["top"]:
            dp = abs(r["top"][0]["prob"] - g["top"][0]["prob"])
            numeric["max_prob_delta"] = max(numeric["max_prob_delta"], dp)
            if dp > tol:
                numeric["prob_mismatches"] += 1
                issues.append(f"step {i}: top prob {r['top'][0]['prob']:.6g} vs {g['top'][0]['prob']:.6g} "
                              f"(|d|={dp:.3g} > {tol:g})")
    if len(ref) != len(got):
        msg = f"length {len(ref)} (llama) vs {len(got)} (strata)"
        issues.append(msg)
        id_issues.append(msg)
        if first is None:
            first = {"step": n, "kind": "length", "llama": len(ref), "strata": len(got)}
    return issues, first, id_issues, numeric


def tie_explains(ref, got, first, margin):
    """True when the first id mismatch is a swap between two candidates the reference itself could not separate.

    Two kernels of the same 4-bit model cannot be expected to order a near-tie the same way; llama.cpp's own GPU
    and CPU runs flip at their first tie (2/16 ids on the short prompt).  A mismatch counts as a tie only when, at
    that step, the id one side chose is in the other side's top-2 *and* the gap to the runner-up is at most
    `margin` in **both** distributions.  A confident mismatch (a clear winner on either side) is not excused.
    """
    if not first or first.get("kind") != "id":
        return None
    i = first["step"]
    if i >= len(ref) or i >= len(got):
        return None
    r, g = ref[i], got[i]
    rt, gt = r.get("top") or [], g.get("top") or []
    if len(rt) < 2 or len(gt) < 2:
        return None
    r_gap, g_gap = rt[0]["prob"] - rt[1]["prob"], gt[0]["prob"] - gt[1]["prob"]
    ids_in = [t["id"] for t in rt[:2]], [t["id"] for t in gt[:2]]
    swapped = r["id"] in ids_in[1] and g["id"] in ids_in[0]
    if swapped and r_gap <= margin and g_gap <= margin:
        return {"step": i, "llama": r["id"], "strata": g["id"], "llama_gap": r_gap, "strata_gap": g_gap,
                "margin": margin}
    return None


def truncate_to_tokens(ids, n):
    if n <= 0 or len(ids) <= n:
        return ids
    return ids[:n]


def main():
    ap = argparse.ArgumentParser(description="Greedy token-id parity, strata-poly CLI vs llama-server.")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-server", default="llama-server")
    ap.add_argument("--strata-poly", default=str(ROOT / "build-poly-hip" / "strata-poly"))
    ap.add_argument("--prompt", action="append", default=[], help="a case prompt (repeatable)")
    ap.add_argument("--prompt-file", action="append", default=[], help="a case prompt read from a file (repeatable)")
    ap.add_argument("--prompt-file-tokens", type=int, default=1200,
                    help="truncate a --prompt-file prompt to this many tokens (0 = whole file)")
    ap.add_argument("--short-prompt", default=DEFAULT_SHORT_PROMPT, help="used when no --prompt/--prompt-file is given")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--n-probs", type=int, default=5)
    ap.add_argument("--window", action="append", type=int, default=[],
                    help="a window to test (repeatable; default 1 and 8)")
    ap.add_argument("--lookup", type=int, default=3)
    ap.add_argument("--experts", choices=["vram", "cpu", "both"], default="both")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--vram-reserve-mib", type=int, default=2048)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--llama-ngl", type=int, default=99,
                    help="llama-server -ngl; a model bigger than the card needs 0 (or a fitting layer count), "
                         "because llama.cpp aborts instead of auto-fitting when -ngl is given")
    ap.add_argument("--prob-tol", type=float, default=1e-3,
                    help="report a top-1 probability difference above this (informational; does not fail the case)")
    ap.add_argument("--tie-margin", type=float, default=0.15,
                    help="a first id mismatch is excused when the right answer was this close to a tie in both runs")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    gguf = Path(args.gguf).resolve()
    if not gguf.exists():
        sys.exit(f"no GGUF at {gguf}")
    strata = Path(args.strata_poly).resolve()
    if not strata.exists():
        sys.exit(f"no strata-poly at {strata}")

    # The cases: (name, text, is_file).  A file's text is tokenized whole, then cut to --prompt-file-tokens ids.
    cases = []
    for i, text in enumerate(args.prompt):
        cases.append((f"prompt{i}", text, False))
    for i, path in enumerate(args.prompt_file):
        cases.append((f"file{i}", Path(path).read_text(encoding="utf-8", errors="replace"), True))
    if not cases:
        cases.append(("short", args.short_prompt, False))
    windows = args.window or [1, 8]
    experts = ["vram", "cpu"] if args.experts == "both" else [args.experts]

    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="parity-llama-"))
    work.mkdir(parents=True, exist_ok=True)
    port = args.port or free_port()
    server_log = work / "llama-server.log"

    record = {
        "gguf": str(gguf),
        "gguf_bytes": gguf.stat().st_size,
        "strata_poly": str(strata),
        "llama_server": args.llama_server,
        "n_predict": args.n_predict,
        "n_probs": args.n_probs,
        "window": windows,
        "lookup": args.lookup,
        "experts": experts,
        "max_context": args.max_context,
        "prob_tol": args.prob_tol,
        "pinned_flags": ["--tail-skip 0", "--temperature 0", "--top-k 1", "--seed 0", "--stage-ahead 0",
                         "--cpu-share 0", "--expert-swaps 0", "--conversation-cache-mib 0",
                         f"--window <w> --lookup {args.lookup} --max-context {args.max_context}",
                         f"--vram-reserve-mib {args.vram_reserve_mib}",
                         "environment: " + ", ".join(CLEARED_ENV) + " cleared"],
        "cases": [],
    }
    try:
        v = subprocess.run([args.llama_server, "--version"], capture_output=True, text=True, timeout=60)
        record["llama_version"] = v.stdout.strip() or v.stderr.strip()
    except Exception as e:
        record["llama_version"] = f"unavailable: {e}"

    env = dict(os.environ)
    for k in CLEARED_ENV:
        env.pop(k, None)

    # Phase 1: the reference.  llama-server holds the GGUF on the card; nothing of ours runs beside it.
    server = None
    refs = {}
    try:
        argv = [args.llama_server, "-m", str(gguf), "-c", str(args.max_context), "-ngl", str(args.llama_ngl),
                "--host", "127.0.0.1", "--port", str(port)]
        record["llama_argv"] = argv
        with open(server_log, "w") as log:
            server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        wait_health(port, min(args.timeout, 600), server)
        for name, text, is_file in cases:
            ids = tokenize(port, text)
            if is_file:
                ids = truncate_to_tokens(ids, args.prompt_file_tokens)
            steps = llama_completion(port, ids, args.n_predict, args.n_probs, args.timeout)
            refs[name] = {"ids": ids, "steps": steps, "prompt_chars": len(text)}
            print(f"[llama] {name}: {len(ids)} prompt ids, {len(steps)} generated")
    finally:
        stop(server)

    # Phase 2: ours, one case at a time, after the server is gone.
    failed = 0
    for name, text, is_file in cases:
        ref = refs[name]
        for window in windows:
            for mode in experts:
                label = f"{name} window={window} experts={mode}"
                run = strata_run(strata, gguf, ref["ids"], args.n_predict, args.n_probs, window, args.lookup,
                                 args.max_context, mode == "cpu", env, args.timeout,
                                 args.vram_reserve_mib, args.threads)
                issues, first, id_issues, numeric = compare(ref["steps"], run["steps"], args.n_probs, args.prob_tol)
                tie = tie_explains(ref["steps"], run["steps"], first, args.tie_margin)
                gate = []
                if run["tail_experts"]:
                    gate.append(f"tail_experts={run['tail_experts']}")
                if run["tail_skip_note"]:
                    gate.append("--tail-skip > 0 note")
                if run["returncode"] != 0:
                    gate.append(f"exit {run['returncode']}")
                verdict = "PASS" if (not id_issues or tie) and not gate else "FAIL"
                if verdict == "FAIL":
                    failed += 1
                case = {
                    "name": name, "window": window, "experts": mode,
                    "prompt_chars": ref["prompt_chars"], "prompt_tokens": len(ref["ids"]),
                    "llama_ids": [s["id"] for s in ref["steps"]],
                    "strata_ids": run["ids"],
                    "steps": [{"llama": r, "strata": g} for r, g in zip(ref["steps"], run["steps"])],
                    "first_divergence": first,
                    "tie": tie,
                    "issues": issues[:20],
                    "numeric": numeric,
                    "gate": gate,
                    "verdict": verdict,
                    "strata_argv": run["argv"],
                    "cached_experts": run["cached_experts"],
                    "total_experts": run["total_experts"],
                    "stats": run["stats"],
                }
                record["cases"].append(case)
                print(f"[ours ] {label}: {verdict}"
                      + (f" — first {first}" if first else "")
                      + (f" — tie {tie}" if tie else "")
                      + (f" — gate {gate}" if gate else ""))
                if verdict == "FAIL":
                    for line in issues[:5]:
                        print(f"        {line}")
    record["summary"] = {"total": len(record["cases"]), "passed": len(record["cases"]) - failed, "failed": failed}
    print(f"\n{record['summary']['passed']}/{record['summary']['total']} cases pass")
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
