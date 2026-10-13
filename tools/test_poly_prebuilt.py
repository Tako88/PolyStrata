"""Tests for setup.py's ready-made strata-poly (get_poly_prebuilt) from a local release folder: the asset of this
source is installed, one for another source or another card is not, and an engine already there is kept.  Also the
AMD (HIP) build, and update_poly_engine() on a HIP install (docs/AMD_HIP.md).  No GPU, no network, no pip.

    python -m unittest tools.test_poly_prebuilt
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

SRC = "0123456789abcdef"
GPU = {"arch": 89, "driver": "581.57"}


def release(folder: Path, src=SRC, archs=(75, 80, 86, 89, 120)) -> None:
    with zipfile.ZipFile(folder / setup.POLY_ASSET.format(src=src), "w") as f:
        f.writestr(setup.POLY_EXE, "new engine")
        f.writestr(setup.VEXE, "new encoder")
        f.writestr("BUILD.json", json.dumps({"source": "release", "archs": list(archs), "cuda": "13.0", "src": src,
                                             "vision_src": src}))


class PolyPrebuilt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.rel = Path(self.tmp.name) / "release"
        self.root.mkdir()
        self.rel.mkdir()
        self.eng = self.root / setup.ENGINE_POLY_DIR
        for p in (mock.patch.object(setup, "ROOT", self.root), mock.patch.object(setup, "WIN", True),
                  mock.patch.object(setup, "POLY_PREBUILT_URL", str(self.rel)),
                  mock.patch.object(setup, "source_hash", lambda parts: SRC),
                  mock.patch.object(setup, "cpu_info", lambda: ("an AVX2 CPU", True, False)),
                  mock.patch.object(setup, "pip_cuda_libs", lambda toolkit=13: None)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_installs_the_asset_of_this_source(self):
        release(self.rel)
        self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "new engine")
        self.assertEqual((self.eng / setup.VEXE).read_text(), "new encoder")
        self.assertEqual(json.loads((self.eng / "BUILD.json").read_text())["source"], "release")
        self.assertEqual(sorted(p.name for p in self.eng.iterdir()), sorted(["BUILD.json", setup.POLY_EXE, setup.VEXE]))

    def test_none_for_another_source(self):
        release(self.rel, src="fedcba9876543210")
        self.assertFalse(setup.get_poly_prebuilt(GPU))
        self.assertFalse((self.eng / setup.POLY_EXE).exists())

    def test_none_for_a_card_it_has_no_code_for(self):
        release(self.rel)
        self.assertFalse(setup.get_poly_prebuilt({"arch": 90, "driver": "581.57"}))
        self.assertFalse((self.eng / setup.POLY_EXE).exists())

    def test_none_with_a_driver_older_than_cuda_13(self):
        release(self.rel)
        self.assertFalse(setup.get_poly_prebuilt({"arch": 89, "driver": "576.88"}))

    def test_none_for_a_cpu_without_avx2(self):
        release(self.rel)
        with mock.patch.object(setup, "cpu_info", lambda: ("an older CPU", False, False)):
            self.assertFalse(setup.get_poly_prebuilt(GPU))

    def test_a_failed_pip_leaves_the_engine_there(self):
        self.eng.mkdir()
        (self.eng / setup.POLY_EXE).write_text("compiled here")
        release(self.rel)
        with mock.patch.object(setup, "pip_cuda_libs", side_effect=SystemExit(1)):
            with self.assertRaises(SystemExit):
                setup.get_poly_prebuilt(GPU)
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "compiled here")

    def test_an_update_downloads_in_place_of_a_ready_made_engine_alone(self):
        release(self.rel)
        self.eng.mkdir()
        cfg = {"exe": str(self.eng / setup.POLY_EXE), "args": []}
        for source, want in (("local", "compiled here"), ("release", "new engine")):
            (self.eng / setup.POLY_EXE).write_text("compiled here")
            (self.eng / "BUILD.json").write_text(json.dumps({"source": source, "archs": [89], "src": "older"}))
            with mock.patch.object(setup, "gpu_info", lambda i=None: GPU), \
                    mock.patch.object(setup, "build_engine_poly", lambda *a: self.eng), \
                    mock.patch.object(setup, "get_llama_cpp", lambda: None):
                setup.update_poly_engine(dict(cfg))
            self.assertEqual((self.eng / setup.POLY_EXE).read_text(), want, source)

    def test_an_update_without_a_release_of_its_source_keeps_the_ready_made_engine(self):
        release(self.rel, src="fedcba9876543210")
        self.eng.mkdir()
        (self.eng / setup.POLY_EXE).write_text("ready-made")
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "release", "archs": [89], "src": "older"}))
        build = mock.Mock()
        with mock.patch.object(setup, "gpu_info", lambda i=None: GPU), \
                mock.patch.object(setup, "build_engine_poly", build):
            setup.update_poly_engine({"exe": str(self.eng / setup.POLY_EXE), "args": []})
        build.assert_not_called()
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "ready-made")

    def test_a_ready_made_engine_with_an_older_encoder_is_replaced(self):
        release(self.rel)
        self.eng.mkdir()
        (self.eng / setup.POLY_EXE).write_text("ready-made")
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "release", "archs": [89], "src": SRC,
                                                         "vision_src": "older"}))
        self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertEqual((self.eng / setup.VEXE).read_text(), "new encoder")

    def test_an_engine_of_this_source_is_kept(self):
        self.eng.mkdir()
        (self.eng / setup.POLY_EXE).write_text("compiled here")
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "local", "archs": [89], "src": SRC}))
        release(self.rel)
        self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "compiled here")

    def test_a_compile_after_it_is_for_this_card_alone(self):
        release(self.rel)
        setup.get_poly_prebuilt(GPU)
        meta = json.loads((self.eng / "BUILD.json").read_text())
        meta["src"] = "a source it was not built from"
        (self.eng / "BUILD.json").write_text(json.dumps(meta))
        with mock.patch.object(setup, "install_build_tools", side_effect=RuntimeError("compiles")) as tools:
            with self.assertRaises(RuntimeError):
                setup.build_engine_poly(GPU, True, None)
        self.assertEqual(tools.call_args[0][0]["archs"], [89])

    def compile_here(self, gpu, toolkit, vision="none"):
        """build_engine_poly() with a compile that writes what it was asked: (the CMake definitions, the toolkit)."""
        made = {}

        def cmake(src, bdir, target, defs, vcvars, script):
            made.setdefault(target, defs)
            out = bdir / "bin" if target == "strata-vision" else bdir
            out.mkdir(parents=True, exist_ok=True)
            (out / (setup.VEXE if target == "strata-vision" else setup.POLY_EXE)).write_text(f"{target} {toolkit}")
        with mock.patch.object(setup, "install_build_tools", lambda g, yes: (str(self.root / "nvcc"), None)), \
                mock.patch.object(setup, "cmake_build", cmake), mock.patch.object(setup, "cuda_lib_dirs", lambda t: []), \
                mock.patch.object(setup, "source_version", lambda: "0.1.42"):
            setup.build_engine_poly(gpu, True, None, vision, toolkit)
        return made

    def test_cuda_12_compiles_in_place_of_a_ready_made_engine(self):
        release(self.rel)
        setup.get_poly_prebuilt(GPU)
        made = self.compile_here(GPU, 12, "gpu")
        self.assertEqual(sorted(made), ["strata-poly", "strata-vision"])
        self.assertNotIn("-DSTRATA_EXPERIMENTAL_SM60=ON", made["strata-poly"])
        meta = json.loads((self.eng / "BUILD.json").read_text())
        self.assertEqual((meta["toolkit"], meta["archs"], meta["source"]), (12, [89], "local"))
        # the ready-made one is not taken back while the model asks for CUDA 12, and is when it no longer does
        self.assertEqual(self.compile_here(GPU, 12, "gpu"), {})
        self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "new engine")

    def test_cuda_12_for_a_pascal_card_is_the_experimental_build(self):
        made = self.compile_here({"arch": 61, "driver": "550.54"}, 12)
        self.assertIn("-DSTRATA_EXPERIMENTAL_SM60=ON", made["strata-poly"])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=61", made["strata-poly"])

    def compile_hip(self, gpu, vision="none"):
        """build_engine_poly(hip=True) with a compile that writes what it was asked: (the build folder, the CMake
        definitions).  No ROCm and no hipcc: rocm_root() is mocked (docs/AMD_HIP.md)."""
        made = {}

        def cmake(src, bdir, target, defs, vcvars, script):
            made.setdefault(target, (bdir, defs))
            out = bdir / "bin" if target == "strata-vision" else bdir
            out.mkdir(parents=True, exist_ok=True)
            (out / (setup.VEXE if target == "strata-vision" else setup.POLY_EXE)).write_text(target)

        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup, "cmake_build", cmake):
            with mock.patch.object(setup, "rocm_root", lambda archs: (Path("/opt/rocm"), ["/opt/rocm/lib"])), \
                    mock.patch.object(setup, "source_version", lambda: "0.1.42"):
                setup.build_engine_poly(gpu, True, None, vision, hip=True)
        return made

    def test_the_amd_build_is_this_source_built_as_hip(self):
        made = self.compile_hip({"arch": "gfx1200"})
        bdir, defs = made["strata-poly"]
        self.assertEqual(Path(bdir).name, "build-poly-hip")
        for want in ("-DSTRATA_POLY_HIP=ON", "-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF",
                     "-DSTRATA_PREFILL_MMQ=ON", "-DCMAKE_HIP_ARCHITECTURES=gfx1200",
                     "-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++",
                     "-DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/amdgcn/bitcode"):
            self.assertIn(want, defs)
        meta = json.loads((self.eng / "BUILD.json").read_text())
        self.assertEqual((meta["backend"], meta["archs"], meta["lib_dirs"], meta["source"]),
                         ("hip", ["gfx1200"], ["/opt/rocm/lib"], "local-hip"))
        # the encoder is the CPU one (there is no HIP encoder build), and it is not compiled when none is wanted
        self.assertEqual(sorted(made), ["strata-poly"])
        made = self.compile_hip({"arch": "gfx1200"}, "cpu")
        self.assertEqual(sorted(made), ["strata-poly", "strata-vision"])
        self.assertIn("-DSTRATA_VISION_CUDA=OFF", made["strata-vision"][1])

    def test_the_amd_build_is_kept_for_the_card_it_was_made_for(self):
        self.compile_hip({"arch": "gfx1200"})
        self.assertEqual(self.compile_hip({"arch": "gfx1200"}), {})       # nothing to do
        self.assertEqual(sorted(self.compile_hip({"arch": "gfx1201"})), ["strata-poly"])   # another card: built again



NVIDIA = {"index": 0, "name": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0, "arch": 89, "driver": "581.57"}


class UpdateHip(unittest.TestCase):
    """update_poly_engine() on an AMD install ("backend": "hip" in engine-poly/BUILD.json).  The card comes from
    HIP's own numbering (amd_gpus), never from nvidia-smi, and the engine is this source compiled here again through
    the HIP path: on the AMD-only PC the NVIDIA warning is gone, and on a mixed AMD + NVIDIA PC no CUDA engine is
    fetched or compiled over the HIP one.  No AMD card: a truthful warning, and the installed engine starts."""

    HIP = {"index": 0, "name": "AMD Radeon RX 9060 XT", "vram_gb": 15.9, "arch": "gfx1200", "driver": "amdgpu",
           "vendor": "amd", "uma": False}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir()
        self.eng = self.root / setup.ENGINE_POLY_DIR
        self.eng.mkdir()
        (self.eng / setup.POLY_EXE).write_text("the installed engine")
        self.cfg_path = Path(self.tmp.name) / "strata-glm-3.0bit.json"
        for p in (mock.patch.object(setup, "ROOT", self.root),
                  mock.patch.object(setup, "source_hash", lambda parts: SRC),
                  mock.patch.object(setup, "get_llama_cpp", lambda: None)):
            p.start()
            self.addCleanup(p.stop)

    def update(self, meta, amd=(), nvidia=NVIDIA, cfg=None, backend="hip"):
        """update_poly_engine() against this installed engine (meta), on a PC whose AMD cards are `amd` and whose
        nvidia-smi answer is `nvidia`.  Returns (the build mock, the gpu_info mock, the config, the output)."""
        (self.eng / "BUILD.json").write_text(json.dumps(meta), encoding="utf-8")
        cfg = {"exe": str(self.eng / setup.POLY_EXE), "args": [], **({"backend": backend} if backend else {}),
               **(cfg or {})}
        self.cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        info = mock.Mock(return_value=nvidia)
        out = io.StringIO()
        with (mock.patch.object(setup, "amd_gpus", lambda: list(amd)),
              mock.patch.object(setup, "gpu_info", info),
              mock.patch.object(setup, "build_engine_poly") as build,
              mock.patch.object(setup, "get_poly_prebuilt") as self.prebuilt,
              contextlib.redirect_stdout(out)):
            setup.update_poly_engine(cfg, False, self.cfg_path)
        return build, info, cfg, " ".join(out.getvalue().split())

    def test_a_stale_hip_engine_is_compiled_again_for_the_amd_card(self):
        build, info, cfg, out = self.update({"backend": "hip", "source": "local-hip", "src": "an older source",
                                             "archs": ["gfx1200"], "lib_dirs": ["/opt/rocm/lib"]},
                                            [self.HIP], cfg={"gpu": 0})
        build.assert_called_once()
        args, kwargs = build.call_args
        self.assertEqual(args[0]["archs"], ["gfx1200"])        # the AMD card, in HIP's own numbering
        self.assertEqual(args[1:], (False, None, "none"))      # yes, llama, vision
        self.assertEqual(kwargs, {"hip": True})                # the HIP path, not a CUDA compile
        info.assert_not_called()                               # nvidia-smi was never asked
        self.prebuilt.assert_not_called()                      # and no CUDA engine fetched over it
        self.assertNotIn("NVIDIA", out)
        self.assertNotIn("could not update", out)
        self.assertEqual(cfg["lib_dirs"], ["/opt/rocm/lib"])
        self.assertEqual(json.loads(self.cfg_path.read_text(encoding="utf-8"))["exe"],
                         str(self.eng / setup.POLY_EXE))

    def test_the_amd_build_keeps_the_cpu_encoder_the_install_has(self):
        (self.eng / setup.VEXE).write_text("the cpu encoder")
        build, *_ = self.update({"backend": "hip", "src": "an older source", "archs": ["gfx1200"], "vision": "cpu",
                                 "vision_src": SRC, "lib_dirs": ["/opt/rocm/lib"]},
                                [self.HIP], cfg={"gpu": 0, "vision": {"exe": str(self.eng / setup.VEXE)}})
        self.assertEqual(build.call_args[0][3], "cpu")          # images stay on the CPU, never a CUDA encoder

    def test_an_up_to_date_hip_engine_is_left_alone(self):
        build, info, _cfg, out = self.update({"backend": "hip", "src": SRC, "archs": ["gfx1200"],
                                              "lib_dirs": ["/opt/rocm/lib"]}, [self.HIP], cfg={"gpu": 0})
        build.assert_not_called()
        info.assert_not_called()
        self.assertEqual(out, "")                               # nothing said: no warning, no NVIDIA

    def test_no_amd_card_starts_the_installed_engine_with_a_truthful_warning(self):
        build, info, cfg, out = self.update({"backend": "hip", "src": "an older source", "archs": ["gfx1200"]},
                                            [], cfg={"gpu": 0})
        build.assert_not_called()
        info.assert_not_called()                                # the NVIDIA card beside it is not this install's
        self.assertIn("could not update strata-poly (no AMD GPU found): starting the installed one", out)
        self.assertNotIn("NVIDIA", out)
        self.assertEqual(cfg["exe"], str(self.eng / setup.POLY_EXE))

    def test_a_card_the_config_no_longer_finds_gets_the_same_warning(self):
        # the config's "gpu" 3 is not there any more (a card taken out): not "no NVIDIA GPU found"
        build, _info, _cfg, out = self.update({"backend": "hip", "src": SRC, "archs": ["gfx1200"]},
                                              [self.HIP], cfg={"gpu": 3})
        build.assert_not_called()
        self.assertIn("no AMD GPU found", out)

    def test_a_cuda_install_still_uses_the_nvidia_card_beside_an_amd_one(self):
        # the CUDA path is untouched by this fix: its own card resolver, its own build call, no "hip"
        build, info, cfg, out = self.update({"source": "local", "src": "an older source", "archs": [89],
                                             "cuda_dirs": ["/cuda/bin"]}, [self.HIP], cfg={"gpu": 0}, backend=None)
        build.assert_called_once()
        args, kwargs = build.call_args
        self.assertEqual(args[0]["archs"], [89])
        self.assertEqual(args[1:], (False, None, "none", 13))   # yes, llama, vision, toolkit
        self.assertEqual(kwargs, {})                            # no hip=True: a CUDA compile
        info.assert_called()                                    # nvidia-smi, as before
        self.assertEqual(cfg["lib_dirs"], ["/cuda/bin"])
        self.assertNotIn("AMD GPU found", out)

    def test_a_cuda_install_without_an_nvidia_card_still_says_nvidia(self):
        build, _info, _cfg, out = self.update({"source": "local", "src": "an older source", "archs": [89]},
                                              [self.HIP], nvidia=None, cfg={"gpu": 0}, backend=None)
        build.assert_not_called()
        self.assertIn("could not update strata-poly (no NVIDIA GPU found)", out)
        self.assertNotIn("AMD GPU found", out)


if __name__ == "__main__":
    unittest.main()
