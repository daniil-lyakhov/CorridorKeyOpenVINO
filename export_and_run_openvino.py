#!/usr/bin/env python3
"""
CorridorKey — Download, Export to OpenVINO, and Run Inference
=============================================================

This standalone script:
  1. Downloads the CorridorKey checkpoint from HuggingFace.
  2. Loads the PyTorch GreenFormer model.
  3. Exports it to OpenVINO IR via ``openvino.convert_model``.
  4. Runs inference with the OpenVINO model on a sample green-screen image.

Setup::

    # Clone this repository (--recursive pulls the CorridorKey submodule)
    git clone --recursive https://github.com/daniil-lyakhov/CorridorKeyOpenVINO.git
    cd CorridorKeyOpenVINO

    # Create a virtual environment and install dependencies
    python -m venv .venv
    source .venv/bin/activate          # Linux / macOS
    # .venv\\Scripts\\activate           # Windows
    pip install -r requirements.txt

Usage::

    # Single image
    python export_and_run_openvino.py --image path/to/greenscreen.png [--img-size 1024]

    # Video
    python export_and_run_openvino.py --video path/to/greenscreen.mp4 [--img-size 1024]

If neither ``--image`` nor ``--video`` is given a synthetic green-screen
image is generated and processed as a demo.
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Make CorridorKeyModule importable from the git submodule
# ---------------------------------------------------------------------------
_SUBMODULE_ROOT = Path(__file__).resolve().parent / "CorridorKey"
if str(_SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO_ID = "nikopueringer/CorridorKey_v1.0"
HF_CHECKPOINT_FILENAME = "CorridorKey_v1.0.pth"  # Actual filename on HuggingFace
DEFAULT_IMG_SIZE = 1024  # Use a smaller default than the repo's 2048 for quick demo

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# ---------------------------------------------------------------------------
# 1. Download checkpoint
# ---------------------------------------------------------------------------
def download_checkpoint(dest_dir: str | Path = "checkpoints") -> Path:
    """Download CorridorKey .pth from HuggingFace Hub (cached)."""
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / HF_CHECKPOINT_FILENAME

    if dest_file.exists():
        logger.info(f"Checkpoint already present: {dest_file}")
        return dest_file

    logger.info(f"Downloading checkpoint from huggingface.co/{HF_REPO_ID} …")
    cached = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CHECKPOINT_FILENAME)
    shutil.copy2(cached, dest_file)
    logger.info(f"Saved checkpoint → {dest_file}")
    return dest_file


# ---------------------------------------------------------------------------
# 2. Build the PyTorch model & load weights
# ---------------------------------------------------------------------------
def load_torch_model(checkpoint_path: Path, img_size: int, device: str = "cpu") -> torch.nn.Module:
    """Instantiate GreenFormer, load weights, return in eval mode."""
    # Import from the repo
    from CorridorKeyModule.core.model_transformer import GreenFormer

    model = GreenFormer(
        encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
        img_size=img_size,
        use_refiner=True,
    )
    model = model.to(device)
    model.eval()

    # Load state-dict
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip ``_orig_mod.`` prefix left by ``torch.compile`` and resize pos-embeds
    model_state = model.state_dict()
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if "pos_embed" in k and k in model_state and v.shape != model_state[k].shape:
            N_src, C = v.shape[1], v.shape[2]
            N_dst = model_state[k].shape[1]
            g_src = int(math.sqrt(N_src))
            g_dst = int(math.sqrt(N_dst))
            v = (
                F.interpolate(
                    v.permute(0, 2, 1).view(1, C, g_src, g_src),
                    size=(g_dst, g_dst),
                    mode="bicubic",
                    align_corners=False,
                )
                .flatten(2)
                .transpose(1, 2)
            )
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    logger.info(f"PyTorch model loaded  (img_size={img_size})")
    return model


# ---------------------------------------------------------------------------
# 3. Export to OpenVINO IR
# ---------------------------------------------------------------------------
def export_to_openvino(
    model: torch.nn.Module,
    img_size: int,
    output_dir: str | Path = "ir",
) -> Path:
    """Export the PyTorch model to OpenVINO IR (FP32) via openvino.convert_model."""
    import openvino as ov

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ir_path = output_dir / "corridorkey.xml"

    if ir_path.exists():
        logger.info(f"OpenVINO IR already exists: {ir_path}")
        return ir_path

    logger.info("Exporting to OpenVINO IR …")

    # Create example input: [B, 4, H, W]
    dummy = torch.zeros(1, 4, img_size, img_size, dtype=torch.float32)

    # Use openvino's torch→IR conversion (traces the model internally)
    ov_model = ov.convert_model(model, example_input=dummy)

    ov.save_model(ov_model, str(ir_path))
    logger.info(f"OpenVINO IR saved → {ir_path}")
    return ir_path


# ---------------------------------------------------------------------------
# 4. Pre-processing helpers (mirrors inference_engine.py)
# ---------------------------------------------------------------------------
def preprocess(image_bgr: np.ndarray, img_size: int) -> tuple[np.ndarray, int, int]:
    """
    Prepare a BGR uint8 image for model inference.

    Returns:
        inp: numpy array  [1, 4, H, W]  float32
        orig_h, orig_w: original frame dimensions (for later resize-back)
    """
    h, w = image_bgr.shape[:2]

    # Convert BGR → RGB float32 [0,1]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # --- Build a simple green-screen mask (chroma-key heuristic) ---
    # The model expects an auxiliary mask channel.  In the real pipeline
    # this comes from GVM / VideoMaMa / BiRefNet. Here we use a naive green
    # channel heuristic so the script is fully self-contained.
    mask = _simple_green_mask(image_rgb)  # [H, W, 1] float32 0-1

    # Resize to model resolution
    img_resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    if mask_resized.ndim == 2:
        mask_resized = mask_resized[:, :, np.newaxis]

    # ImageNet normalisation on the RGB channels
    img_norm = (img_resized - IMAGENET_MEAN) / IMAGENET_STD

    # Concatenate → [H, W, 4] then transpose → [1, 4, H, W]
    inp = np.concatenate([img_norm, mask_resized], axis=-1)
    inp = inp.transpose((2, 0, 1))[np.newaxis].astype(np.float32)
    return inp, h, w


def _simple_green_mask(rgb: np.ndarray) -> np.ndarray:
    """Very simple green-screen mask: green channel dominance → soft mask."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    green_dominance = g - np.maximum(r, b)
    mask = np.clip(green_dominance * 4.0, 0.0, 1.0)  # aggressive threshold
    # Invert: 1 = foreground, 0 = green background
    mask = 1.0 - mask
    return mask[:, :, np.newaxis].astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Post-processing
# ---------------------------------------------------------------------------
def postprocess(
    alpha: np.ndarray,
    fg: np.ndarray,
    orig_h: int,
    orig_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resize model outputs back to original resolution and build a composite.

    Returns:
        alpha_u8:  [H, W] uint8  (0-255)
        fg_u8:     [H, W, 3] uint8 BGR
        comp_u8:   [H, W, 3] uint8 BGR (FG over grey checkerboard)
    """
    # alpha: [1,1,H,W] → [H,W]
    alpha_2d = alpha[0, 0]
    alpha_2d = cv2.resize(alpha_2d, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    alpha_2d = np.clip(alpha_2d, 0.0, 1.0)

    # fg: [1,3,H,W] → [H,W,3] (RGB)
    fg_hwc = fg[0].transpose(1, 2, 0)
    fg_hwc = cv2.resize(fg_hwc, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    fg_hwc = np.clip(fg_hwc, 0.0, 1.0)

    # Simple composite: FG over mid-grey
    bg = np.full_like(fg_hwc, 0.35)
    alpha_3 = alpha_2d[:, :, np.newaxis]
    comp = fg_hwc * alpha_3 + bg * (1.0 - alpha_3)

    # Convert to uint8 BGR for saving/display
    alpha_u8 = (alpha_2d * 255).astype(np.uint8)
    fg_bgr_u8 = (cv2.cvtColor(fg_hwc, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
    comp_bgr_u8 = (cv2.cvtColor(comp, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)

    return alpha_u8, fg_bgr_u8, comp_bgr_u8


# ---------------------------------------------------------------------------
# 6. Generate a synthetic test image (if no real image is supplied)
# ---------------------------------------------------------------------------
def make_synthetic_greenscreen(h: int = 720, w: int = 1280) -> np.ndarray:
    """Create a synthetic green-screen image with a coloured circle as 'foreground'."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (0, 200, 0)  # BGR green background

    # Draw a "person" (circle + rectangle)
    cx, cy = w // 2, h // 2
    cv2.circle(img, (cx, cy - 80), 80, (180, 120, 70), -1)  # head
    cv2.rectangle(img, (cx - 60, cy), (cx + 60, cy + 200), (200, 80, 60), -1)  # body
    return img


# ---------------------------------------------------------------------------
# 7. Video helpers
# ---------------------------------------------------------------------------
def _iter_video_frames(video_path: str):
    """Open a video file and yield frames one at a time.

    Returns:
        (n_frames, height, width, fps, frame_iterator)
        *n_frames* is an estimate from the container metadata (may be 0 if unknown).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def _gen():
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    return n_frames, h, w, vid_fps, _gen()


def process_video(
    video_path: str,
    compiled_model,
    img_size: int,
    output_dir: Path,
    fps: float | None = None,
) -> None:
    """Stream a video, running per-frame inference on the fly.

    Frames are read one at a time and never stored in a list,
    keeping memory usage constant regardless of video length.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames, orig_h, orig_w, vid_fps, frame_iter = _iter_video_frames(video_path)
    logger.info(f"Opened {video_path}  ({orig_w}×{orig_h}, ~{n_frames} frames, {vid_fps:.1f} fps)")

    if fps is not None:
        vid_fps = fps

    # --- Prepare video writers ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    input_writer = cv2.VideoWriter(str(out_dir / "input.mp4"), fourcc, vid_fps, (orig_w, orig_h))
    alpha_writer = cv2.VideoWriter(str(out_dir / "alpha.mp4"), fourcc, vid_fps, (orig_w, orig_h), isColor=False)
    fg_writer = cv2.VideoWriter(str(out_dir / "foreground.mp4"), fourcc, vid_fps, (orig_w, orig_h))
    comp_writer = cv2.VideoWriter(str(out_dir / "composite.mp4"), fourcc, vid_fps, (orig_w, orig_h))

    # --- Single read-process-write loop ---
    total_ms = 0.0
    n_processed = 0
    for idx, frame_bgr in enumerate(frame_iter):
        # Save original frame
        input_writer.write(frame_bgr)

        # Preprocess → infer → postprocess
        inp, h, w = preprocess(frame_bgr, img_size)

        t0 = time.perf_counter()
        results = compiled_model(inp)
        dt = time.perf_counter() - t0
        total_ms += dt * 1000

        alpha_out = results[0]
        fg_out = results[1]

        alpha_u8, fg_u8, comp_u8 = postprocess(alpha_out, fg_out, h, w)

        alpha_writer.write(alpha_u8)
        fg_writer.write(fg_u8)
        comp_writer.write(comp_u8)

        n_processed = idx + 1
        total_str = f"/{n_frames}" if n_frames else ""
        if n_processed % 10 == 0 or idx == 0:
            logger.info(f"  frame {n_processed}{total_str}  ({dt * 1000:.1f} ms)")

    input_writer.release()
    alpha_writer.release()
    fg_writer.release()
    comp_writer.release()

    if n_processed == 0:
        sys.exit("Video has no frames")

    avg_ms = total_ms / n_processed
    logger.info(
        f"Video done: {n_processed} frames, avg {avg_ms:.1f} ms/frame ({1000.0 / avg_ms:.1f} fps).  Saved to {out_dir}/"
    )


# ---------------------------------------------------------------------------
# 8. Single-image inference
# ---------------------------------------------------------------------------
def process_image(
    image_path: str,
    compiled_model,
    img_size: int,
    output_dir: Path,
) -> None:
    """Run inference on a single green-screen image and save results."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        sys.exit(f"Cannot read image: {image_path}")
    logger.info(f"Loaded image {image_path}  ({image_bgr.shape[1]}×{image_bgr.shape[0]})")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the input
    cv2.imwrite(str(out_dir / "input.png"), image_bgr)
    logger.info(f"Saved input image → {out_dir}/input.png")

    inp, orig_h, orig_w = preprocess(image_bgr, img_size)
    logger.info(f"Preprocessed input shape: {inp.shape}")

    t0 = time.perf_counter()
    results = compiled_model(inp)
    dt = time.perf_counter() - t0
    logger.info(f"OpenVINO inference: {dt * 1000:.1f} ms")

    alpha_out = results[0]  # [1, 1, img_size, img_size]  float32 [0,1]
    fg_out = results[1]  # [1, 3, img_size, img_size]  float32 [0,1]

    alpha_u8, fg_u8, comp_u8 = postprocess(alpha_out, fg_out, orig_h, orig_w)

    cv2.imwrite(str(out_dir / "alpha.png"), alpha_u8)
    cv2.imwrite(str(out_dir / "foreground.png"), fg_u8)
    cv2.imwrite(str(out_dir / "composite.png"), comp_u8)
    logger.info(f"Results saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="CorridorKey → OpenVINO export & inference")
    parser.add_argument("--image", type=str, default=None, help="Path to a green-screen image (BGR)")
    parser.add_argument("--video", type=str, default=None, help="Path to a green-screen video")
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help="Model resolution (default 1024)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for result images/videos",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Where to store the .pth file",
    )
    parser.add_argument("--ir-dir", type=str, default="ir", help="Where to store the OpenVINO IR model")
    parser.add_argument("--device", type=str, default="CPU", help="OpenVINO device (CPU, GPU, NPU)")
    args = parser.parse_args()

    import openvino as ov

    ir_path = Path(args.ir_dir) / "corridorkey.xml"

    # ---- Step 1–3: Reuse existing IR or export from scratch ----
    if ir_path.exists():
        logger.info(f"OpenVINO IR already exists: {ir_path} — skipping download & export")
        torch_model = None
    else:
        # Download checkpoint
        ckpt_path = download_checkpoint(args.checkpoint_dir)
        # Load PyTorch model
        torch_model = load_torch_model(ckpt_path, img_size=args.img_size, device="cpu")
        # Export to OpenVINO
        ir_path = export_to_openvino(torch_model, img_size=args.img_size, output_dir=args.ir_dir)

    # ---- Step 4: Load OpenVINO model ----
    core = ov.Core()
    logger.info(f"Loading OpenVINO model on device={args.device} …")
    compiled = core.compile_model(str(ir_path), args.device)

    # ---- Step 5: Run on image or video ----
    out_dir = Path(args.output_dir)
    if args.video is None and args.image is None:
        # No input supplied — generate a synthetic green-screen image as a demo
        logger.info("No --image/--video supplied → generating synthetic green-screen image")
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(out_dir / "synthetic_input.png")
        cv2.imwrite(image_path, make_synthetic_greenscreen())
        logger.info(f"Saved synthetic image → {image_path}")
        args.image = image_path

    if args.video:
        process_video(
            video_path=args.video,
            compiled_model=compiled,
            img_size=args.img_size,
            output_dir=out_dir,
        )
    elif args.image:
        process_image(
            image_path=args.image,
            compiled_model=compiled,
            img_size=args.img_size,
            output_dir=out_dir,
        )

    logger.info("Done ✓")


if __name__ == "__main__":
    main()
