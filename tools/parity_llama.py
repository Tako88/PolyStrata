#!/usr/bin/env python3
"""tools/parity_llama.py - id-level greedy parity between `strata-poly` and llama.cpp on one GGUF.

`bench/bench_vs_llama.py` is the harness this one is modelled on, and its own words say what it is:

    #!/usr/bin/env python3
    \"\"\"Strata vs llama.cpp head-to-head (stream 1402-h).  Measurement only.

    Both engines are driven through their OpenAI-compatible HTTP servers (/v1/chat/completions), with the same prompt
    text, same sampler settings and the same GGUF.  Server-side timings (llama.cpp's names, which Strata's server also
    emits) give prompt tok/s and decode tok/s; time to first token is the server's prompt_ms (non-streaming request,
    so TTFT = prompt processing time).  Peak VRAM comes from nvidia-smi polling (minus the idle baseline), RAM from the
    engine process tree (psutil RSS, which counts mmap'd file pages) and the system's used-memory delta.

        python bench_vs_llama.py prompts  --out DIR --tokenizer DIR --corpus FILE...     # build 4K / 32K prompt files
        python bench_vs_llama.py run      --machine CFG.json --out DIR --rounds N [--configs a,b] [--cells ...]
        python bench_vs_llama.py quality  --machine CFG.json --out DIR [--configs a,b]
        python bench_vs_llama.py summary  --out DIR

    CFG.json: {\"gguf\": ..., \"strata_cfg\": ..., \"llama_server\": ..., \"configs\": [{\"name\",\"kind\":\"llama|strata\",\"args\":[...]}]}
    Interleaving: engines cannot be loaded together (12 GB card, 64 GB RAM), so one ROUND runs every config once
    (start, warm-up, cells, stop), and rounds alternate the config order.  Medians are taken over rounds.
    The caller holds the machine's GPU lock.  Processes are stopped by PID only.
    \"\"\"

That harness compares **prompt text and timings**.  Two engines can agree to the sixth decimal on a
text prompt and still disagree token for token: each tokenizes the text itself, so a pre-tokenizer or a
special-token difference moves every id; each detokenizes its own output, so two different ids that decode to the
same bytes compare equal; and it reports tok/s, never which token came out.  This harness removes all three:
the prompt is tokenized **once**, by the reference's own `/tokenize` with `"add_special": false`, and the same
integer ids go to both sides, so tokenization cannot enter into it; the reference's `/completion` with an id array
returns each generated id beside its top-K (`n_probs`); and `strata-poly --tokens <ids> --top-probs K` prints the
same.  A case is then the generated ids, the per-step top-K ids and the top-1 probability of the two sides.

Both engines are started and stopped by this file, one at a time, so only one holds the card and one holds the
model in RAM at once.  They run the same backend shape - the reference defaults to `-ngl 99 -ncmoe 40`, its GPU
path, because the engine runs its dense layers on the card too and the two llama.cpp backends do not always agree:
on Nemotron-3.5-Bolt's 21-token `short20` prompt the CPU backend (`-ngl 0`) picks id 1278 at step 10 where the GPU
backend (`-ngl 99 -ncmoe 40`) picks 6777, a 0.92% top-1 margin against 0.46%, and the engine follows the GPU one.
Every case records both revisions - the reference's `llama-server --version` line and the engine's vendored-ggml
commit from `third_party/ggml/VERSION.txt` - because the two can differ (the engine links a vendored ggml, the
reference an installed llama.cpp) and a revision difference would otherwise look like a bug in the engine.

A case passes on a measured bar, not on bit-identical ids.  The port's own drift against this reference is
0.3-1% at blk.2-11 and 1-5% at blk.60-87 (M5's layer walk, restated in the M6 evidence section), and llama.cpp is
fully deterministic, so a rule that asks llama.cpp to reproduce a swap on a second run can never be satisfied.
The bar has three parts:

1. **Distribution agreement, at every step before a divergence.**  The reference's top-1 id must be among the
   engine's top-3 and the engine's top-1 among the reference's top-3, and the engine's top-1 probability must be
   within `DRIFT_P` of the reference's.  Top-3 *set* equality is recorded but not required: the two exact CPU
   cases flip third place on 9 of 64 steps because two independent f32 distributions reorder sub-2% tails, and a
   real bug moves the top-1 out of the other side's top-3 rather than swapping a tail.
2. **Greedy ids, required except at a measured tie.**  Ids must be equal at every step, except that the *first*
   divergence is excused when the reference's own top-1/top-2 gap is at or below `DRIFT_GAP`; the excused step is
   recorded with both candidates and both probabilities.  A divergence at a larger gap, or one whose top-1 is
   outside the reference's top-3, fails.  A divergence cascades - the engine then holds a different token and the
   two sequences are on different trajectories - so later ids are recorded but not asserted, and the case is a
   PASS-with-excuses.
3. `DRIFT_P` and `DRIFT_GAP` come from the measured drift, not from taste; `--drift-p`/`--drift-gap` override them.
   The JSON carries the observed maximum same-id top-1 probability drift, the observed maximum drift at any step,
   and a histogram of the reference's own per-step top-1/top-2 gaps, so a reader can see how much room was left.

`DRIFT_P` = 0.25 and `DRIFT_GAP` = 0.05.  The localization's first stage over M5's 1e-3 relative bar on the GPU
path is blk.2 `attn_norm`/`attn_in` (ours `+5.636945` against the reference's `+5.524817`, rel 2.03e-2), and the
max relative drift over the 88-block `attn_norm` walk is 4.20e-2 on the GPU path and 6.34e-2 on the CPU pair, the
same 1-5% (a little wider) M5 measured; a reference gap within 5% is inside that drift, so `DRIFT_GAP` = 0.05.
The largest observed same-id top-1 probability drift is 0.2446 (the `--window 8 --lookup 3` CPU case, step 49,
where the window path's own sums differ - `src/poly/engine.hpp:635-637`), so `DRIFT_P` = 0.25 with the non-window
worst at 0.1568.

The engine's stderr must also report `0 left out` (no expert was dropped by `STRATA_ROUTE_TAIL_SKIP`).

The reference request pins `repeat_penalty: 1.0` beside `temperature: 0, top_k: 1`.  A GGUF can carry a default
repeat penalty - LFM2.5-24B-A2B-Q4_K_M's file sets 1.05 - and the server applies it even under a greedy request,
which makes the reference emit a non-argmax token and the case fail on a sampler setting rather than on the engine.
The engine has no repeat penalty at all, so pinning it to 1.0 is the comparison; leaving it unset measured the
reference's default instead.  (Found on the LFM2.5 run: the reference emitted 2997 where its own `/completion`
top-1 was 355, and 355 had appeared twice in the prompt.  With `repeat_penalty: 1.0` the reference emits
`6552,22,440,4695,355,13564`, exactly the engine's ids.)

    python tools/parity_llama.py --gguf MODEL.gguf --prompt "The capital of" --n-predict 8 --n-probs 3
    python tools/parity_llama.py --gguf MODEL.gguf --prompt-file corpus.txt --max-context 32768 \\
        --window 8 --lookup 3 --no-gpu-experts --n-predict 64 --json out.json
    python tools/parity_llama.py --gguf MODEL.gguf --prompt X --ref-ngl 0 --ref-ncmoe 0   # CPU reference
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENDORED_GGML_VERSION = REPO / "third_party" / "ggml" / "VERSION.txt"
TAIL_OUT_RE = re.compile(r"(\d+)\s+left out")
IDS_LINE_RE = re.compile(r"^\s*\d+(?:\s*,\s*\d+)*\s*$")
TOP_LINE_RE = re.compile(r"^top((?:\s+-?\d+:[^\s]+)*)\s*$")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(url: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def rev_of(binary: str) -> str:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60)
        first = (out.stdout or out.stderr).strip().splitlines()
        return first[0] if first else ""
    except OSError as e:
        return f"(could not run {binary}: {e})"


def vendored_ggml_rev() -> str:
    try:
        return VENDORED_GGML_VERSION.read_text().strip().splitlines()[0]
    except OSError:
        return "(third_party/ggml/VERSION.txt not found)"


class Reference:
    """llama-server on the same GGUF, CPU-only, one process at a time."""

    def __init__(self, binary: str, gguf: str, max_context: int, log_path: Path,
                 port: int = 0, start_timeout: float = 600.0, ngl: int = 99, ncmoe: int = 40):
        self.binary = binary
        self.gguf = gguf
        self.max_context = max_context
        self.log_path = log_path
        self.timeout = start_timeout
        self.port = port or free_port()
        self.proc: subprocess.Popen | None = None
        self.argv = [binary, "-m", gguf, "-ngl", str(ngl), "-c", str(max_context)]
        if ncmoe:
            self.argv += ["-ncmoe", str(ncmoe)]
        self.argv += ["--host", "127.0.0.1", "--port", str(self.port), "--no-warmup"]
        self._start(start_timeout)

    def _start(self, timeout: float) -> None:
        self.log = open(self.log_path, "w")
        self.log.write("argv: " + " ".join(self.argv) + "\n")
        self.log.flush()
        self.proc = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited {self.proc.returncode}; log:\n{self._tail()}")
            try:
                http_json(f"http://127.0.0.1:{self.port}/health", timeout=2.0)
                return
            except (urllib.error.URLError, OSError, ValueError):
                time.sleep(0.5)
        raise RuntimeError(f"llama-server did not become ready in {timeout:.0f}s; log:\n{self._tail()}")

    def _tail(self, lines: int = 25) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return "(no log)"

    def tokenize(self, text: str) -> list[int]:
        r = http_json(f"http://127.0.0.1:{self.port}/tokenize", {"content": text, "add_special": False},
                      timeout=self.timeout)
        return [int(t) for t in r["tokens"]]

    def completion(self, ids: list[int], n_predict: int, n_probs: int, id_slot: int) -> dict:
        # repeat_penalty 1.0 on purpose: a GGUF's own default (LFM2.5's is 1.05) is applied under a greedy request
        # and would make the reference emit a non-argmax token; the engine has no repeat penalty (see the header).
        body = {"prompt": ids, "temperature": 0, "top_k": 1, "seed": 0, "n_predict": n_predict,
                "cache_prompt": False, "n_probs": n_probs, "id_slot": id_slot,
                "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
        raw = http_json(f"http://127.0.0.1:{self.port}/completion", body, timeout=self.timeout)
        steps = []
        for p in raw.get("completion_probabilities", []):
            top = []
            for t in p.get("top_logprobs", []) or []:
                top.append({"id": int(t["id"]), "p": math.exp(float(t["logprob"]))})
            if not top:
                top = [{"id": int(p["id"]), "p": float("nan")}]
            steps.append({"id": int(p["id"]), "top": top})
        return {"body": body, "ids": [s["id"] for s in steps], "steps": steps,
                "content": raw.get("content"), "stop_type": raw.get("stop_type"),
                "timings": raw.get("timings"), "tokens_evaluated": raw.get("tokens_evaluated")}

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if hasattr(self, "log") and not self.log.closed:
            self.log.close()


def run_engine(binary: str, gguf: str, ids: list[int], cfg: argparse.Namespace, timeout: float) -> dict:
    argv = [binary, "--model", gguf, "--tokens", ",".join(str(i) for i in ids),
            "--max-new", str(cfg.n_predict), "--temperature", "0", "--top-k", "1", "--seed", "0",
            "--top-probs", str(cfg.n_probs), "--max-context", str(cfg.max_context),
            "--window", str(cfg.window), "--lookup", str(cfg.lookup),
            # FM-6: the shapes the plan pins per case, so a case is not silently a different case
            "--tail-skip", "0", "--conversation-cache-mib", "0", "--stage-ahead", "0", "--cpu-share", "0",
            "--vram-reserve-mib", "256"]
    if cfg.no_gpu_experts:
        argv.append("--no-gpu-experts")
    env = {k: v for k, v in os.environ.items() if k not in ("STRATA_ROUTE_TAIL_SKIP", "STRATA_VRAM_FREE_FROM")}
    t0 = time.time()
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    out, err = proc.stdout, proc.stderr
    gen: list[int] = []
    for line in out.splitlines():
        if IDS_LINE_RE.match(line):
            gen = [int(x) for x in line.strip().split(",") if x.strip()]
            break
    steps = []
    for line in err.splitlines():
        m = TOP_LINE_RE.match(line)
        if not m:
            continue
        top = []
        for tok in m.group(1).split():
            tid, _, prob = tok.rpartition(":")
            top.append({"id": int(tid), "p": float(prob)})
        steps.append({"top": top})
    for i, s in enumerate(steps):
        s["id"] = gen[i] if i < len(gen) else None
    tail = TAIL_OUT_RE.search(err)
    experts = re.search(r"(\d+) of (\d+) routed experts in VRAM", err)
    return {"argv": argv, "env_cleared": ["STRATA_ROUTE_TAIL_SKIP", "STRATA_VRAM_FREE_FROM"],
            "returncode": proc.returncode, "ids": gen, "steps": steps,
            "tail_experts": int(tail.group(1)) if tail else None,
            "cached_experts": (int(experts.group(1)), int(experts.group(2))) if experts else None,
            "wall_seconds": round(time.time() - t0, 2),
            "stdout": out, "stderr": err}


DRIFT_P = 0.25
DRIFT_GAP = 0.05
TOP_K = 3


def gap_histogram(ref: dict) -> dict:
    """The reference's own per-step top-1/top-2 gap, bucketed.  The reader sees how many steps sat below the
    excuse bar (a tie) and how many were wide enough that a divergence there would be a failure."""
    edges = [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]
    hist = {f"[{lo:g},{hi:g})": 0 for lo, hi in zip(edges, edges[1:])}
    hist[f"[{edges[-1]:g},1.0]"] = 0
    for s in ref["steps"]:
        top = s.get("top", [])[:2]
        gap = (top[0]["p"] - top[1]["p"]) if len(top) > 1 else 1.0
        for lo, hi in zip(edges, edges[1:]):
            if lo <= gap < hi:
                hist[f"[{lo:g},{hi:g})"] += 1
                break
        else:
            hist[f"[{edges[-1]:g},1.0]"] += 1
    return hist


def compare(ref: dict, eng: dict, drift_p: float = DRIFT_P, drift_gap: float = DRIFT_GAP) -> dict:
    """The measured bar (see the module docstring)."""
    rids, eids = ref["ids"], eng["ids"]
    n = min(len(rids), len(eids))
    diffs = []
    topk = []
    max_any = 0.0
    for i in range(n):
        rt = ref["steps"][i]["top"][:TOP_K]
        et = eng["steps"][i]["top"][:TOP_K]
        rset = [t["id"] for t in rt]
        eset = [t["id"] for t in et]
        overlap = len(set(rset) & set(eset)) / max(1, min(len(rset), len(eset)))
        delta = abs(rt[0]["p"] - et[0]["p"]) if rt and et else float("nan")
        if not math.isnan(delta):
            max_any = max(max_any, delta)
        # the reference's top-1 must be one the engine considered, and the other way round; then a divergence is
        # a reordering inside the top-3, not the engine answering a different question
        cand_ok = bool(rt and et and rt[0]["id"] in eset and et[0]["id"] in rset)
        topk.append({"step": i, "ref_top": rset, "eng_top": eset, "overlap": round(overlap, 4),
                     "set_equal": set(rset) == set(eset), "candidate_ok": cand_ok,
                     "same_id": rids[i] == eids[i],
                     "top1_delta": round(delta, 6) if not math.isnan(delta) else None})
        if rids[i] != eids[i]:
            gap = (rt[0]["p"] - rt[1]["p"]) if len(rt) > 1 else float("inf")
            diffs.append({"step": i, "ref_id": rids[i], "eng_id": eids[i],
                          "ref_top1": rt[0]["id"] if rt else None,
                          "ref_top1_p": rt[0]["p"] if rt else None,
                          "ref_top2": rt[1]["id"] if len(rt) > 1 else None,
                          "ref_top2_p": rt[1]["p"] if len(rt) > 1 else None,
                          "ref_gap_to_top2": round(gap, 8),
                          "eng_top1_p": et[0]["p"] if et else None,
                          "eng_rank_of_ref_top1": (eset.index(rt[0]["id"]) + 1) if rt and rt[0]["id"] in eset else None,
                          "candidate_ok": cand_ok, "set_equal": set(rset) == set(eset)})
    # the drift is meaningful only while both sides hold the same context: up to the first divergence
    first = diffs[0]["step"] if diffs else n
    aligned = [t for t in topk[:first] if t["same_id"] and t["top1_delta"] is not None]
    max_delta = max((t["top1_delta"] for t in aligned), default=0.0)
    max_delta_step = max(aligned, key=lambda t: t["top1_delta"])["step"] if aligned else None
    any_worst = max(topk, key=lambda t: t["top1_delta"] or 0, default=None)
    result = {"lengths": [len(rids), len(eids)], "id_diffs": diffs, "steps_compared": n,
              "max_top1_prob_delta": round(max_delta, 6),
              "max_top1_prob_delta_step": max_delta_step,
              "max_any_top1_prob_delta": round(max_any, 6), "topk_per_step": topk,
              "bar": {"drift_p": drift_p, "drift_gap": drift_gap,
                      "observed_max_same_id_top1_delta": round(max_delta, 6),
                      "observed_max_same_id_step": max_delta_step,
                      "observed_max_any_top1_delta": round(max_any, 6),
                      "observed_max_any_top1_step": any_worst["step"] if any_worst else None,
                      "gap_histogram": gap_histogram(ref)}}

    # steps before the first divergence, and any exact run, must agree on the distribution
    pre = topk[:first]
    dist_bad = [t["step"] for t in pre if not t["candidate_ok"] or (t["top1_delta"] or 0) > drift_p]
    if dist_bad:
        result["verdict"] = "FAIL"
        result["reason"] = (f"distribution agreement fails before any id divergence at step(s) "
                            f"{dist_bad[:5]}: a top-1 outside the other side's top-3, or a top-1 probability "
                            f"more than {drift_p} from the reference's")
        return result
    if not diffs and len(rids) == len(eids):
        result["verdict"] = "PASS"
        result["excused_steps"] = []
        return result
    if not diffs:
        result["verdict"] = "FAIL"
        result["reason"] = (f"ids agree for {n} steps but the engine generated {len(eids)} against "
                            f"{len(rids)}; no step diverged and the sequence stopped early")
        return result

    d = diffs[0]
    d["excused"] = False
    if not d["candidate_ok"]:
        result["verdict"] = "FAIL"
        result["reason"] = (f"step {d['step']}: the engine's top-1 {d['eng_id']} is not among the reference's "
                            f"top-{TOP_K}, or the reference's top-1 {d['ref_id']} is not among the engine's")
        return result
    if (d["eng_top1_p"] is not None and d["ref_top1_p"] is not None
            and abs(d["eng_top1_p"] - d["ref_top1_p"]) > drift_p):
        result["verdict"] = "FAIL"
        result["reason"] = (f"step {d['step']}: the engine's top-1 probability {d['eng_top1_p']:.6f} is more than "
                            f"{drift_p} from the reference's {d['ref_top1_p']:.6f}")
        return result
    if d["ref_gap_to_top2"] > drift_gap:
        result["verdict"] = "FAIL"
        result["reason"] = (f"step {d['step']}: ids diverge at a reference top-1/top-2 gap of "
                            f"{d['ref_gap_to_top2']:.4f}, above DRIFT_GAP {drift_gap}; the reference was not at "
                            f"a tie (top-1 {d['ref_id']} {d['ref_top1_p']:.6f}, top-2 {d['ref_top2']} "
                            f"{d['ref_top2_p']:.6f})")
        return result
    d["excused"] = True
    result["verdict"] = "PASS_WITH_EXCUSES"
    result["excused_steps"] = [d["step"]]
    result["excused_detail"] = [d]
    result["later_divergences"] = len(diffs) - 1
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="id-level greedy parity between strata-poly and llama.cpp")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-server", default="/usr/bin/llama-server")
    ap.add_argument("--strata-poly", default=str(REPO / "build-upstream-hip" / "strata-poly"))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt")
    src.add_argument("--prompt-file")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--n-probs", type=int, default=5)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--lookup", type=int, default=0)
    ap.add_argument("--no-gpu-experts", action="store_true")
    ap.add_argument("--json", help="write the full report here as well as printing it")
    ap.add_argument("--port", type=int, default=0, help="reference port (0 = pick a free one)")
    ap.add_argument("--ref-ngl", type=int, default=99,
                    help="reference -ngl (99 = its GPU backend, 0 = its CPU backend; default 99)")
    ap.add_argument("--ref-ncmoe", type=int, default=40,
                    help="reference -ncmoe; 0 omits it (default 40, as the ceiling run)")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="seconds allowed for one reference request or one engine run")
    ap.add_argument("--ref-cache",
                    help="directory to cache the reference's tokenize+completion in; a later case with the same\n"
                         "prompt and settings reuses it (a 20,000-token reference prefill is tens of minutes on\n"
                         "this 22 GB file, and the four step-21 configurations share one)")
    ap.add_argument("--drift-p", type=float, default=DRIFT_P,
                    help=f"the bar's top-1 probability drift (default {DRIFT_P}, measured; see the docstring)")
    ap.add_argument("--drift-gap", type=float, default=DRIFT_GAP,
                    help=f"the bar's reference top-1/top-2 tie width (default {DRIFT_GAP}, measured)")
    cfg = ap.parse_args()

    text = cfg.prompt if cfg.prompt is not None else Path(cfg.prompt_file).read_text()
    gguf = str(Path(cfg.gguf).resolve())
    outdir = Path(cfg.json).resolve().parent if cfg.json else Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "gguf": gguf,
        "config": {"n_predict": cfg.n_predict, "n_probs": cfg.n_probs, "max_context": cfg.max_context,
                   "window": cfg.window, "lookup": cfg.lookup, "no_gpu_experts": cfg.no_gpu_experts},
        "prompt": {"chars": len(text), "preview": text[:200]},
        "revisions": {"llama_server_version": rev_of(cfg.llama_server),
                      "engine_vendored_ggml": vendored_ggml_rev()},
        "reference_backend": {"ngl": cfg.ref_ngl, "ncmoe": cfg.ref_ncmoe},
    }

    log_path = outdir / "parity-llama-server.log"
    cache = None
    if cfg.ref_cache:
        key = hashlib.sha1(json.dumps({"gguf": gguf, "text": text, "max_context": cfg.max_context,
                                       "ngl": cfg.ref_ngl, "ncmoe": cfg.ref_ncmoe,
                                       "n_predict": cfg.n_predict, "n_probs": cfg.n_probs},
                                      sort_keys=True).encode()).hexdigest()[:16]
        cache = Path(cfg.ref_cache) / f"ref-{key}.json"
    if cache is not None and cache.exists():
        cached = json.loads(cache.read_text())
        ids = cached["ids"]
        report["prompt"]["n_tokens"] = len(ids)
        report["prompt"]["ids"] = ids
        report["reference"] = {"argv": cached["argv"], "log": cached.get("log"), "cached": True,
                               "run1": cached["run1"], "body": cached["run1"]["body"]}
    else:
        ref = Reference(cfg.llama_server, gguf, cfg.max_context, log_path, port=cfg.port,
                        start_timeout=cfg.timeout, ngl=cfg.ref_ngl, ncmoe=cfg.ref_ncmoe)
        try:
            ids = ref.tokenize(text)
            report["prompt"]["n_tokens"] = len(ids)
            report["prompt"]["ids"] = ids
            report["reference"] = {"argv": ref.argv, "log": str(log_path),
                                   "run1": ref.completion(ids, cfg.n_predict, cfg.n_probs, 0)}
        finally:
            ref.stop()
        report["reference"]["body"] = report["reference"]["run1"]["body"]
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"ids": ids, "argv": report["reference"]["argv"],
                                         "log": str(log_path), "run1": report["reference"]["run1"]}) + "\n")

    eng = run_engine(cfg.strata_poly, gguf, ids, cfg, cfg.timeout)
    report["engine"] = eng

    if eng["returncode"] != 0 or eng["tail_experts"] not in (0,):
        report["comparison"] = {"verdict": "FAIL",
                                "reason": f"engine returncode {eng['returncode']}, tail_experts {eng['tail_experts']}"}
    else:
        report["comparison"] = compare(report["reference"]["run1"], eng,
                                       drift_p=cfg.drift_p, drift_gap=cfg.drift_gap)

    verdict = report["comparison"]["verdict"]
    print(json.dumps(report, indent=1))
    if cfg.json:
        Path(cfg.json).write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nparity_llama: {verdict}  n_tokens={report['prompt'].get('n_tokens')} "
          f"n_predict={cfg.n_predict} window={cfg.window} lookup={cfg.lookup} "
          f"gpu_experts={not cfg.no_gpu_experts}  drift_p={cfg.drift_p} drift_gap={cfg.drift_gap}",
          file=sys.stderr)
    if "id_diffs" in report["comparison"]:
        for d in report["comparison"]["id_diffs"]:
            print(f"  step {d['step']}: ref {d['ref_id']} eng {d['eng_id']} "
                  f"(ref gap to top-2 {d['ref_gap_to_top2']}, top-2 {d.get('ref_top2')}, excused {d.get('excused')})",
                  file=sys.stderr)
    if report["comparison"].get("reason"):
        print(f"  reason: {report['comparison']['reason']}", file=sys.stderr)
    return 0 if verdict in ("PASS", "PASS_WITH_EXCUSES") else 1


if __name__ == "__main__":
    sys.exit(main())
