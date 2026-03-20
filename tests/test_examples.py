"""Tests for the OpenVINO export script and notebook.

These tests verify the export/inference pipeline works end-to-end
*without* needing real model weights or heavy dependencies.  The tests
mock the HuggingFace download, the GreenFormer model, and OpenVINO
conversion so they run fast in CI.

What is tested:
  - The export script's helper functions (preprocess, postprocess, masks)
  - The export-to-OpenVINO pipeline (with a tiny dummy model)
  - The video processing pipeline
  - The notebook can be executed via nbconvert (integration, slow)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = EXAMPLES_DIR / "export_and_run_openvino.py"


def _import_script():
    """Import export_and_run_openvino.py as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_script", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Prevent argparse from consuming pytest's argv
    with patch("sys.argv", ["export_and_run_openvino.py"]):
        spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script():
    """Import the export script once for the whole test module."""
    return _import_script()


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocess:
    """Verify preprocessing produces correct shape and dtype."""

    def test_output_shape(self, script):
        """preprocess must return [1, 4, img_size, img_size] float32."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, :] = (0, 200, 0)  # green
        inp, h, w = script.preprocess(img, img_size=64)
        assert inp.shape == (1, 4, 64, 64)
        assert inp.dtype == np.float32
        assert h == 100
        assert w == 200

    def test_mask_channel_range(self, script):
        """The 4th channel (mask) should be in [0, 1]."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :] = (0, 200, 0)
        inp, _, _ = script.preprocess(img, img_size=32)
        mask_channel = inp[0, 3, :, :]
        assert mask_channel.min() >= 0.0
        assert mask_channel.max() <= 1.0


# ---------------------------------------------------------------------------
# Green mask tests
# ---------------------------------------------------------------------------


class TestSimpleGreenMask:
    """Verify the naive green-screen mask heuristic."""

    def test_pure_green_gives_zero_mask(self, script):
        """Pure green pixels should be masked out (0 = background)."""
        green = np.full((32, 32, 3), [0.0, 1.0, 0.0], dtype=np.float32)
        mask = script._simple_green_mask(green)
        assert mask.shape == (32, 32, 1)
        assert mask.max() <= 0.1  # mostly background

    def test_non_green_gives_high_mask(self, script):
        """Non-green pixels should be foreground (close to 1)."""
        red = np.full((32, 32, 3), [1.0, 0.0, 0.0], dtype=np.float32)
        mask = script._simple_green_mask(red)
        assert mask.min() >= 0.9  # mostly foreground


# ---------------------------------------------------------------------------
# Postprocessing tests
# ---------------------------------------------------------------------------


class TestPostprocess:
    """Verify postprocessing resizes and clips correctly."""

    def test_output_shapes(self, script):
        """postprocess must return correctly sized uint8 arrays."""
        alpha = np.random.rand(1, 1, 64, 64).astype(np.float32)
        fg = np.random.rand(1, 3, 64, 64).astype(np.float32)
        alpha_u8, fg_u8, comp_u8 = script.postprocess(alpha, fg, 100, 200)
        assert alpha_u8.shape == (100, 200)
        assert alpha_u8.dtype == np.uint8
        assert fg_u8.shape == (100, 200, 3)
        assert comp_u8.shape == (100, 200, 3)

    def test_alpha_range(self, script):
        """Alpha output should be in [0, 255]."""
        alpha = np.full((1, 1, 32, 32), 0.5, dtype=np.float32)
        fg = np.full((1, 3, 32, 32), 0.5, dtype=np.float32)
        alpha_u8, _, _ = script.postprocess(alpha, fg, 48, 48)
        assert alpha_u8.min() >= 0
        assert alpha_u8.max() <= 255


# ---------------------------------------------------------------------------
# Synthetic image / video generation
# ---------------------------------------------------------------------------


class TestSyntheticGeneration:
    """Verify synthetic test data generators."""

    def test_synthetic_image_shape(self, script):
        img = script.make_synthetic_greenscreen(h=120, w=160)
        assert img.shape == (120, 160, 3)
        assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# Export + inference pipeline (with tiny model)
# ---------------------------------------------------------------------------


class TestExportPipeline:
    """Test the export and inference pipeline with a tiny dummy model."""

    @staticmethod
    def _make_tiny_model():
        """Create a minimal model that mimics GreenFormer's input/output contract."""

        class TinyModel(torch.nn.Module):
            def forward(self, x):
                b, c, h, w = x.shape
                alpha = torch.sigmoid(torch.zeros(b, 1, h, w))
                fg = torch.sigmoid(torch.zeros(b, 3, h, w))
                return {"alpha": alpha, "fg": fg}

        return TinyModel().eval()

    def test_export_creates_ir_files(self, script, tmp_path):
        """export_to_openvino must create .xml and .bin files."""
        model = self._make_tiny_model()
        ir_path = script.export_to_openvino(model, img_size=32, output_dir=tmp_path)
        assert ir_path.exists()
        assert (tmp_path / "corridorkey.bin").exists()

    def test_openvino_inference_matches_pytorch(self, script, tmp_path):
        """OpenVINO output should be numerically close to PyTorch."""
        import openvino as ov

        model = self._make_tiny_model()
        ir_path = script.export_to_openvino(model, img_size=32, output_dir=tmp_path)

        # OpenVINO inference
        core = ov.Core()
        compiled = core.compile_model(str(ir_path), "CPU")
        dummy_inp = np.zeros((1, 4, 32, 32), dtype=np.float32)
        ov_results = compiled(dummy_inp)
        ov_alpha = ov_results[0]
        ov_fg = ov_results[1]

        # PyTorch inference
        with torch.inference_mode():
            pt_out = model(torch.from_numpy(dummy_inp))
        pt_alpha = pt_out["alpha"].numpy()
        pt_fg = pt_out["fg"].numpy()

        np.testing.assert_allclose(ov_alpha, pt_alpha, atol=1e-5)
        np.testing.assert_allclose(ov_fg, pt_fg, atol=1e-5)


# ---------------------------------------------------------------------------
# Video processing pipeline
# ---------------------------------------------------------------------------


class TestVideoProcessing:
    """Test the video processing function with a mock compiled model."""

    def test_process_video_creates_output_files(self, script, tmp_path):
        """process_video must create input.mp4, alpha.mp4, foreground.mp4, composite.mp4."""

        # Mock compiled model: returns sigmoid(zeros) for any input
        def mock_infer(inp):
            b, c, h, w = inp.shape
            alpha = 1.0 / (1.0 + np.exp(-np.zeros((b, 1, h, w), dtype=np.float32)))
            fg = 1.0 / (1.0 + np.exp(-np.zeros((b, 3, h, w), dtype=np.float32)))
            return [alpha, fg]

        mock_compiled = MagicMock(side_effect=mock_infer)

        # Write a small synthetic video to disk
        vid_path = str(tmp_path / "test_input.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vid_path, fourcc, 30, (96, 64))
        for _ in range(3):
            writer.write(script.make_synthetic_greenscreen(h=64, w=96))
        writer.release()

        out_dir = tmp_path / "out"
        script.process_video(
            video_path=vid_path,
            compiled_model=mock_compiled,
            img_size=32,
            output_dir=out_dir,
        )

        assert (out_dir / "input.mp4").exists()
        assert (out_dir / "alpha.mp4").exists()
        assert (out_dir / "foreground.mp4").exists()
        assert (out_dir / "composite.mp4").exists()

    def test_process_video_calls_model_per_frame(self, script, tmp_path):
        """Model should be called once per frame."""
        n_frames = 5

        def mock_infer(inp):
            b, c, h, w = inp.shape
            alpha = np.full((b, 1, h, w), 0.5, dtype=np.float32)
            fg = np.full((b, 3, h, w), 0.5, dtype=np.float32)
            return [alpha, fg]

        mock_compiled = MagicMock(side_effect=mock_infer)

        # Write a temporary video from synthetic frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_path = str(tmp_path / "test_input.mp4")
        writer = cv2.VideoWriter(vid_path, fourcc, 30, (48, 32))
        for _ in range(n_frames):
            writer.write(script.make_synthetic_greenscreen(h=32, w=48))
        writer.release()

        out_dir = tmp_path / "out"
        script.process_video(
            video_path=vid_path,
            compiled_model=mock_compiled,
            img_size=32,
            output_dir=out_dir,
        )
        assert mock_compiled.call_count == n_frames


# ---------------------------------------------------------------------------
# Notebook execution (integration test — slow, requires jupyter)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNotebookExecution:
    """Execute the notebook end-to-end via nbconvert.

    Marked as 'slow' — skipped in default test runs.
    Run with: pytest -m slow tests/test_examples.py
    """

    def test_notebook_executes_without_error(self, tmp_path):
        """The examples notebook must execute without raising exceptions."""
        pytest.importorskip("nbconvert")
        nbformat = pytest.importorskip("nbformat")

        nb_path = EXAMPLES_DIR / "corridorkey_openvino.ipynb"
        assert nb_path.exists(), f"Notebook not found: {nb_path}"

        from nbconvert.preprocessors import ExecutePreprocessor

        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        # Run from the examples/ directory so relative paths work
        ep.preprocess(nb, {"metadata": {"path": str(EXAMPLES_DIR)}})
