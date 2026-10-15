"""Tests for setup.py's LFM2.5-24B-A2B choice (docs/BACKENDS.md): the pinned file with its size and SHA-256, the two
sizes and their estimates, the one AMD card this family was measured on, and the run config its setup writes (a
GGUF read in place by strata-poly: no expert profile shipped, no draft model, no --tail-skip).  Mocked - no GPU, no
compile, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_lfm2
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402   (the module: its harness, not its tests)

REV = "0912d33141f77e10705345a6732bfd83f0252fdc"
# size, file, bytes, arena_gb (its routed experts), download_gb (the whole file), ram_gb (poly_ram_need on an 8 GB
# card), vram_beside_gb (measured at 32K on the RX 9060 XT)
SIZES = [("Q4_K_M", "LFM2.5-24B-A2B-Q4_K_M.gguf", 14526258464, 13.816, 14.5, 19, 3.9),
         ("Q6_K", "LFM2.5-24B-A2B-Q6_K.gguf", 19689405728, 18.648, 19.7, 24, 4.1)]
SHA = {"LFM2.5-24B-A2B-Q4_K_M.gguf":
       "d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d",
       "LFM2.5-24B-A2B-Q6_K.gguf":
       "4c5f34cb94d4a29f0360d04eaa274bb1361ceae163f40f6361cb75d1bee139c2"}


class Pins(unittest.TestCase):
    def test_the_files(self):
        fam = setup.FAMILIES["lfm2"]
        self.assertEqual(fam["engine"], "poly")
        self.assertIs(fam["vision"], False)
        self.assertEqual(fam["row_bytes"], 20480)          # 10 attention layers x 2 x 8 kv x 64 head_dim x 2
        self.assertEqual(fam["hf"], f"https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/resolve/{REV}/")
        self.assertEqual(list(setup.family_models("lfm2")), [size for size, *_ in SIZES])
        self.assertEqual(list(fam["sha256"]), [file for _, file, *_ in SIZES])
        for size, file, n, arena, download, ram, beside in SIZES:
            with self.subTest(size=size):
                self.assertEqual((setup.model_file(fam, size, 1), setup.model_shards(fam, size)), (file, 1))
                self.assertEqual(fam["sha256"][file], (n, SHA[file]))
                self.assertAlmostEqual(n / 1e9, download, delta=0.05)
                d = fam["models"][size]
                self.assertEqual(d["arena_gb"], arena)
                self.assertEqual(d["download_gb"], download)
                self.assertEqual(d["vram_beside_gb"], beside)
                self.assertEqual(d["ram_gb"], ram)
                self.assertLess(arena, download)               # the experts are part of the file
                self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)))
                self.assertEqual(d["hip_archs"], setup.POLY_HIP_ARCHS)
                self.assertEqual((fam["tag"], fam["name"], fam["experimental"]),
                                 ("lfm2-", "lfm2.5-24b-a2b", True))

    def test_nothing_is_fetched_beside_the_file(self):
        fam = setup.FAMILIES["lfm2"]
        # no image encoder, no expert profile shipped, no draft model or block: the engine starts the profile
        for key in ("mmproj", "mmproj_hf", "profile", "draft", "draft_gb", "vision_gb"):
            self.assertNotIn(key, fam, key)
        for size, d in setup.family_models("lfm2").items():
            self.assertNotIn("draft_model", d, size)
            self.assertNotIn("draft", d, size)
            self.assertNotIn("profile", d, size)


class Estimates(unittest.TestCase):
    def test_the_32_gb_estimate(self):
        # both sizes fit an RTX 5090 (31.84 GiB) whole: OQ-5's branch is not taken
        for size, d in setup.family_models("lfm2").items():
            with self.subTest(size=size):
                self.assertEqual(setup.poly_held_gb(d, 31.84), d["arena_gb"])
                self.assertEqual(setup.poly_ram_need(d, 31.84), setup.POLY_RAM_BESIDE_GB)

    def test_the_8_gb_estimate(self):
        first = next(iter(setup.family_models("lfm2").values()))
        need = setup.poly_ram_need(first, 8.0)
        self.assertAlmostEqual(need, 19, delta=0.5)
        self.assertIn(f"~{need:.0f} GB of RAM beside an 8 GB card", setup.FAMILIES["lfm2"]["about"])

    def test_the_measured_amd_card(self):
        card, q6 = "gfx1200", setup.family_models("lfm2")["Q6_K"]
        for size, d in setup.family_models("lfm2").items():
            with self.subTest(size=size):
                self.assertIn("measured on gfx1200 with ROCm 7.2.4",
                              setup.poly_verdict(d, 64, 24.0, True, arch=card))
                self.assertTrue(setup.poly_fits(d, 64, 24.0, True, arch=card))
                for other in ("gfx1100", "gfx1101", None):     # an unmeasured AMD card: refused
                    self.assertIn("does not run on AMD cards", setup.poly_verdict(d, 64, 24.0, True, arch=other))
                    self.assertFalse(setup.poly_fits(d, 64, 24.0, True, arch=other))
        # the 32K VRAM-beside figures were measured on the RX 9060 XT with --vram-reserve-mib 2048
        self.assertGreater(q6["vram_beside_gb"], setup.family_models("lfm2")["Q4_K_M"]["vram_beside_gb"])
        self.assertEqual(setup.POLY_HIP_RESERVE_MIB, 2048)


class Run(unittest.TestCase):
    """setup.main() for --family lfm2, every outside effect mocked (GLM's harness)."""

    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_the_run_config(self):
        code, out, cfg = self.main(family="lfm2", ram=32.0, vram=15.92)
        self.assertEqual(code, 0, out)
        fam = setup.FAMILIES["lfm2"]
        size, file, _n, _arena, _download, _ram, _beside = SIZES[0]
        self.assertEqual(self.downloads, [fam["hf"] + file])
        self.assertEqual(self.verified, [(file, *fam["sha256"][file])])
        self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
        pack = self.t / "data" / "packs" / (fam["tag"] + size).lower()
        args = cfg["args"]
        self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / (fam["tag"] + size) / file)
        self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
        self.assertFalse((pack / "expert-profile.txt").exists())   # none is shipped: the engine starts it
        self.assertNotIn("--draft-model", args)                    # its file has no draft block
        self.assertNotIn("--tail-skip", args)                      # its answers do not need it
        self.assertEqual(cfg["model_name"], "lfm2.5-24b-a2b-q4_k_m")
        self.assertEqual(cfg["display_name"], "LFM2.5-24B-A2B Q4_K_M")
        self.assertEqual(Path(cfg["exe"]), self.t / setup.ENGINE_POLY_DIR / setup.POLY_EXE)
        self.assertNotIn("vision", cfg)

    def test_the_size_asked_for(self):
        code, out, cfg = self.main(family="lfm2", model="Q6_K", ram=48.0, vram=15.92)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [setup.FAMILIES["lfm2"]["hf"] + SIZES[1][1]])
        self.assertEqual(cfg["model_name"], "lfm2.5-24b-a2b-q6_k")
        self.assertEqual(Path(cfg["args"][cfg["args"].index("--model") + 1]).name, SIZES[1][1])

    def test_the_measured_amd_card_is_installed_over_hip(self):
        card = {"index": 0, "name": "AMD Radeon RX 9060 XT", "vram_gb": 15.92, "arch": "gfx1200", "driver": "amdgpu",
                "vendor": "amd"}
        code, out, cfg = self.main(["--backend", "hip"], family="lfm2", ram=31.3, vram=15.92, amd=[card])
        self.assertEqual(code, 0, out)
        self.assertEqual((self.built, self.built_hip), ([["gfx1200"]], True))
        args = cfg["args"]
        self.assertEqual(args[args.index("--vram-reserve-mib") + 1], str(setup.POLY_HIP_RESERVE_MIB))
        self.assertNotIn("--conversation-cache-mib", args)
        self.assertIn("conversation cache: off - the parked-conversation cache has not been measured on an AMD card",
                      out)
        # the same family on an AMD card it was not measured on: refused
        code, out, cfg = self.main(["--backend", "hip"], family="lfm2", ram=31.3, vram=23.9,
                                   amd=[{**card, "name": "AMD Radeon RX 7900 XTX", "arch": "gfx1100"}])
        self.assertNotEqual(code, 0)
        self.assertIn("has not been measured on gfx1100", out)


if __name__ == "__main__":
    unittest.main()
