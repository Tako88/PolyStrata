# Qwen3.6-35B-A3B UD-Q4_K_M — PolyStrata vs llama.cpp on one RX 9060 XT (ROCm)

Measured 2026-10-13 on the machine in `machine.json`. Everything here comes from one run of
`bench/bench_vs_llama.py` (rounds 3, alternating order) with the servers stopped between configs:

    .venv/bin/python bench/bench_vs_llama.py run     --machine machine.json --out . --rounds 3
    .venv/bin/python bench/bench_vs_llama.py summary --out .

## What it was measured on

| | |
|---|---|
| GPU | AMD Radeon RX 9060 XT, gfx1200 (RDNA4), 15.92 GiB VRAM, ROCm 7.2.4, hipcc 7.2 |
| CPU / RAM | 12 hardware threads / 31.3 GiB + zram  |
| OS | CachyOS (Linux), kernel release of 2026-10-13 |
| Model | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, 22,663,387,424 B, sha256 `0b21525e…` |
| PolyStrata | build-poly-hip/strata-poly, this tree, `--exe`/`--args` in `strata-poly-hip.json` |
| llama.cpp | `/usr/bin/llama-server` version 0.6.0-dev (build 11548), **a HIP build**: its own log reports `ROCm0: AMD Radeon RX 9060 XT (16304 MiB, 16190 MiB free)` |

Both engines were driven through the OpenAI API — PolyStrata through `serve/server.py` with the
engine in `--serve` mode, llama.cpp through its own server. Prompts: `prompt-4k.txt` (4,052
tokens) and `prompt-32k.txt` (`--max-context 36864` for both).

## Medians over 3 rounds (n = 3 per cell at 32k, 6 at 4k, 12 for the 4k PolyStrata cell)

| config | prompt | temp | n | prompt tok/s | decode tok/s | TTFT s | draft acceptance | RSS GiB |
|---|---|---|---|---|---|---|---|---|
| **PolyStrata** | 4k | 0.0 | 12 | **1053.5** | **57.10** | **3.89** | 0.75 | 22.5 |
| **PolyStrata** | 4k | 0.7 | 12 | **929.1** | **58.05** | **4.55** | 0.64 | 22.5 |
| **PolyStrata** | 32k | 0.0 | 3 | **885.4** | **31.80** | **36.20** | 0.73 | 22.9 |
| **PolyStrata** | 32k | 0.7 | 3 | **885.4** | **37.30** | **36.20** | 0.74 | 22.9 |
| llama.cpp | 4k | 0.0 | 6 | 680.0 | 25.35 | 5.96 | – | 21.4 |
| llama.cpp | 4k | 0.7 | 6 | 685.2 | 25.57 | 5.91 | – | 21.4 |
| llama.cpp | 32k | 0.0 | 3 | 624.9 | 19.27 | 51.29 | – | 21.4 |
| llama.cpp | 32k | 0.7 | 3 | 609.2 | 19.09 | 52.61 | – | 21.4 |
| llama.cpp, `--spec-type ngram-simple` | 4k | 0.0 | 6 | 639.7 | 24.15 | 6.33 | – | 21.4 |
| llama.cpp, `--spec-type ngram-simple` | 4k | 0.7 | 6 | 685.5 | 25.85 | 5.91 | – | 21.4 |
| llama.cpp, `--spec-type ngram-simple` | 32k | 0.0 | 3 | 648.0 | 20.59 | 49.46 | – | 21.4 |
| llama.cpp, `--spec-type ngram-simple` | 32k | 0.7 | 3 | 649.0 | 19.46 | 49.38 | – | 21.4 |

Same model, same card, both fitted, one round of each per turn: PolyStrata reads the 4k prompt 1.55x
faster and answers 2.25x faster, and the 32k prompt 1.42x faster with 1.65x the answer speed. Its
draft guessing accepted 0.64-0.75 of its own guesses; llama.cpp's ngram speculation changed nothing
measurable on these prompts.

**Peak VRAM: 16.75 GB** (16,751,202,304 B), from a 2 Hz `rocm-smi --showmeminfo vram` sampler
running for the whole session (`vram-samples.txt`). The harness's own `VRAM peak MiB` column is
**unavailable on this machine**: `bench_vs_llama.py`'s `vram_used_mib()` shells out to `nvidia-smi`,
so every row reads `0 (+1)`. Do not read that column as a measurement.

## How each engine was made to fit 15.92 GiB with a 22.7 GB model

Neither engine runs the model as the plan's `machine.json` first wrote it — both needed a fitting
argument, and each is the *minimum change that fits*, not a tuned setting:

- **llama.cpp cannot load this model at `-ngl 99`.** It asked for 20,583.34 MiB on device 0 and
  failed: `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 20583.34 MiB on device 0:
  cudaMalloc failed: out of memory` (`server-llama-1791870786.log`). The runs above use
  `-ncmoe 16` (the MoE experts of the first 16 layers in RAM; the whole model is 41 layers). llama.cpp
  warns while doing it: `tensor overrides to CPU are used with mmap enabled - consider using
  --load-mode none for better performance`, so its CPU-side experts come from the same page-cached
  file PolyStrata reads, without PolyStrata's router-aware cache. `-ncmoe 16` loaded with the KV
  cache in 15.41 GB of the card's 15.92 GiB, so this is near the largest GPU share llama.cpp takes here.
- **PolyStrata needs `--vram-reserve-mib 2048` for a 32k prompt.** With the default reserve (509 MiB,
  a thirty-second of the card) the expert cache takes the free VRAM and the attention cache can no
  longer grow: the first run of this benchmark answered every 4k request and then failed all three
  rounds' 32k request with `no VRAM for the attention cache (4 MiB more)`
  (`server-polystrata-1791870750.log`, `…1791870814.log`, `…1791870848.log`, in
  `raw/errors.log`). With 2048 the second run had **no errors at all** and all cells produced data.
  The reserve is not free: it costs the expert cache 5831 experts (10.4 GiB) against 6674 (11.9 GiB)
  with the default, and the 4k prompt speed falls from 1207.6 to 1053.5 tok/s (run 1 against run 2,
  same config otherwise). Finding the reserve that keeps a 128k context usable is setup's job (M5);
  `1024` is what the 128k runs in `../../.pi/plans/poly-hip/m4-summary.md` used.

## Deviations from the plan's `machine.json`

- The **draft-MTP arm was dropped**: `--spec-type draft-mtp` needs a draft model (`-md`), and this box
  has no MTP GGUF for this model. `--spec-type ngram-simple` was added instead: it is llama.cpp's
  model-free guessing and the closest thing it has to PolyStrata's own lookup guessing.
- `--fit on` was not used: this build's `--fit` adjusts unset arguments to fit, which would have
  hidden which setting made the model fit.
- `strata_cfg` and `strata_exe` point at this tree's HIP build, and the config carries
  `--vram-reserve-mib 2048` and the model pack's expert profile.

## What was not tested

- **No 128k cell**: these prompts are 4k and 32k. The 128k context was measured on the command line
  (see `../../.pi/plans/poly-hip/m4-summary.md`), not through the server.
- **No long conversation**: one prompt, one answer per cell, no multi-turn growth of the attention cache.
- **No CUDA comparison**: this machine has no NVIDIA card, so the numbers above are this card's, not a
  statement about the CUDA path.
- **llama.cpp was not tuned**: `-ncmoe`, the KV types (`q8_0` for both) and `-fa on` are reasonable
  defaults, not the best settings anyone could find; a better `-ncmoe` or `--load-mode none` may move
  its numbers.
- The other AMD cards, other quantizations, and other ROCm versions are untested here.
