"""Tests for setup.py's Nemotron-3.5-Bolt-74B-A9B (docs/BACKENDS.md): its one pinned file, the estimate for an 8 GB
card, the install by GLM's steps with no draft model and no image encoder, the run config the setup writes, and the
AMD path on Linux (the HIP engine gets --kv-whole and needs no confirmation).  Mocked like tools/test_setup_glm.py,
whose harness the first classes use.

    python -m unittest tools.test_setup_nemotron
"""
from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402   (the module: its harness, not its tests)

FAM = setup.FAMILIES["nemotron"]
REV = "f56d370e3d3314a5458750f33129bee3627b5eba"
SIZE = "IQ2_XXS"
FILE = "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf"
N = 22018326528
SHA = "91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75"
# measured: the 22,018,326,528-byte file (22.018 GB), 17.51 GB of routed experts, and the 9.7 GiB of experts the
# engine held at 32K on the 16,304 MiB card (evidence.md M6 step 21b), so 16.30 - 9.7 = 6.6 GiB are not expert cache;
# "ram_gb" is poly_ram_need() for an 8 GB card
ARENA, DOWNLOAD, RAM8, BESIDE = 17.51, 22.0, 26, 6.2


class Pins(unittest.TestCase):
    def test_the_file(self):
        self.assertEqual(FAM["engine"], "poly")
        self.assertIs(FAM["vision"], False)
        self.assertTrue(FAM["experimental"])
        self.assertEqual(FAM["row_bytes"], 4096)
        self.assertEqual(FAM["hf"],
                         f"https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF/resolve/{REV}/")
        self.assertEqual(setup.HF_REVISIONS["unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF"], REV)
        self.assertEqual(list(setup.family_models("nemotron")), [SIZE])
        self.assertEqual((setup.model_file(FAM, SIZE, 1), setup.model_shards(FAM, SIZE)), (FILE, 1))
        self.assertEqual(FAM["sha256"][FILE], (N, SHA))
        self.assertAlmostEqual(N / 1e9, DOWNLOAD, delta=0.05)
        self.assertEqual((FAM["tag"], FAM["name"]), ("nemotron-", "nemotron-3.5-bolt-74b-a9b"))

    def test_nothing_is_fetched_beside_the_file(self):
        # no image encoder, no expert profile, no draft model or block, no Windows install and no AMD arch list:
        # the engine starts the profile and the GGUF is read in place
        for key in ("mmproj", "mmproj_hf", "profile", "draft", "draft_gb", "vision_gb", "windows", "hip_archs"):
            self.assertNotIn(key, FAM, key)
        for size, d in setup.family_models("nemotron").items():
            for key in ("draft_model", "draft", "profile", "hip_archs", "windows"):
                self.assertNotIn(key, d, f"{size} {key}")

    def test_the_other_engine_does_not_offer_it(self):
        self.assertNotIn(SIZE, setup.MODELS)                 # read in place by strata-poly, not a pack of the first


class Estimates(unittest.TestCase):
    def test_the_numbers_and_the_formulas(self):
        d = FAM["models"][SIZE]
        self.assertEqual((d["download_gb"], d["arena_gb"], d["vram_beside_gb"]), (DOWNLOAD, ARENA, BESIDE))
        self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)))
        self.assertAlmostEqual(setup.poly_ram_need(d, 8.0), RAM8, delta=0.5)
        self.assertLess(ARENA, d["download_gb"])             # the experts are part of the file

    def test_the_32k_beside_is_the_measured_engine_line(self):
        # at --max-context 32768 the engine held 8573 of 13824 experts, 9.7 GiB, on the 16,304 MiB card
        # (evidence.md M6 step 21b): the card less that is what setup reserves before any expert
        self.assertAlmostEqual(setup.poly_held_gb(FAM["models"][SIZE], 16304 / 1024) / 1.073741824, 9.7, delta=0.1)

    def test_the_about_line_says_the_first_size(self):
        n = setup.poly_ram_need(FAM["models"][SIZE], 8.0)
        self.assertIn(f"~{n:.0f} GB of RAM beside an 8 GB card", FAM["about"])


class Run(unittest.TestCase):
    """setup.main() for --family nemotron on an NVIDIA card, every outside effect mocked (GLM's harness)."""

    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_the_run_config(self):
        code, out, cfg = self.main(family="nemotron", ram=32.0, vram=15.92)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [FAM["hf"] + FILE])
        self.assertEqual(self.verified, [(FILE, N, SHA)])
        # the tokenizer alone: no pack, no draft block, no shipped profile
        self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
        pack = self.t / "data" / "packs" / "nemotron-iq2_xxs"
        self.assertFalse((pack / "expert-profile.txt").exists())
        args = cfg["args"]
        self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / "nemotron-IQ2_XXS" / FILE)
        self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
        self.assertEqual(args[args.index("--max-context") + 1], "32768")
        self.assertNotIn("--draft-model", args)              # its file has no draft block
        self.assertNotIn("--tail-skip", args)                # its answers do not need it
        # the family declares no batch policy, so the engine's own 1024 stands (the M5 measurement found the call
        # width flat from 1024 up, docs/BACKENDS.md)
        self.assertNotIn("--prompt-batch", args)
        self.assertNotIn("--kv-whole", args)                 # an NVIDIA build: the AMD path adds it
        self.assertEqual(cfg["model_name"], "nemotron-3.5-bolt-74b-a9b-iq2_xxs")
        self.assertEqual(cfg["display_name"], "Nemotron-3.5-Bolt-74B-A9B IQ2_XXS")
        self.assertEqual(Path(cfg["exe"]), self.t / setup.ENGINE_POLY_DIR / setup.POLY_EXE)
        self.assertNotIn("vision", cfg)

    def test_a_card_that_holds_every_expert_still_reads_1024(self):
        code, out, cfg = self.main(family="nemotron", ram=64.0, vram=31.84)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--prompt-batch", cfg["args"])


class AmdRun(unittest.TestCase):
    """setup.main() for --family nemotron over --backend hip on Linux: the HL engine gets --kv-whole and no question
    is asked.  GLM's harness, with the AMD card the KFD topology would report and the HIP build mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data").mkdir()
        self.downloads, self.runs, self.verified, self.asked, self.built_hip = [], [], [], [], None

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, amd):
        def fake_input(prompt=""):
            raise AssertionError(f"asked {prompt!r}")        # the AMD path on Linux asks nothing at these choices

        found = [{"index": 0, "name": "NVIDIA GeForce RTX 5090", "vram_gb": 31.84, "arch": "120", "driver": "580.97"}]

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_hip(gpu, llama):
            self.built_hip = sorted(set(gpu.get("archs") or [gpu["arch"]]))
            eng = self.t / setup.ENGINE_POLY_DIR
            eng.mkdir(exist_ok=True)
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "backend": "hip",
                                                        "cuda_dirs": ["/rocm/lib"]}))
            return eng

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "WIN", False),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "amd_gpus", lambda *a: list(amd)),
            mock.patch.object(setup, "amd_pci_devices", lambda *a: []),
            mock.patch.object(setup, "amd_device_access_problem", lambda *a: None),
            mock.patch.object(setup, "ram_gb", lambda: 32.0),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "cpu_cores", lambda: None),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: 500.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "build_engine_poly", mock.Mock(side_effect=AssertionError("the CUDA engine"))),
            mock.patch.object(setup, "build_engine_poly_hip", fake_hip),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "whole_shard", lambda s: False),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", lambda cmd, *a, **k: self.runs.append([str(x) for x in cmd])),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"start-{tag}.sh"),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.object(sys, "argv", ["setup.py", "--family", "nemotron", "--backend", "hip", "--yes",
                                            "--no-start", "--models-dir", str(self.t / "models")]),
            mock.patch("builtins.input", fake_input),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = glm_tests.quiet(setup.main)
        path = self.t / f"strata-{FAM['tag']}{SIZE.lower()}.json"
        return code, out, (json.loads(path.read_text()) if path.is_file() else None)

    def test_the_hip_engine_gets_kv_whole_without_a_question(self):
        card = {"index": 0, "name": "AMD Radeon RX 9060 XT", "vram_gb": 15.92, "arch": "gfx1200",
                "driver": "amdgpu", "vendor": "amd", "uma": False}
        code, out, cfg = self.main([card])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.built_hip, ["gfx1200"])        # poly_setup accepted this card for the family
        self.assertNotIn("has been run on Linux only", out)
        self.assertNotIn("has not been tried on Windows", out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "32768")
        self.assertEqual(args[-1], "--kv-whole")             # the AMD path appends it
        self.assertEqual(args.count("--kv-whole"), 1)


if __name__ == "__main__":
    unittest.main()
