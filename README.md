# CorridorKey → OpenVINO Export & Inference

This example demonstrates how to export the **CorridorKey** green-screen keying
model to [OpenVINO](https://docs.openvino.ai/) IR format and run optimised
inference on **Intel CPUs, GPUs, and NPUs**.

## Contents

| File | Description |
|---|---|
| `corridorkey_openvino.ipynb` | Interactive notebook — full pipeline with inline visualisation |
| `export_and_run_openvino.py` | Standalone CLI script — export + image/video inference |
| `ir/` | Pre-exported OpenVINO IR model (512×512, FP32) |
| `requirements.txt` | Python dependencies |
| `CorridorKey/` | Git submodule — [CorridorKey](https://github.com/nikopueringer/CorridorKey) @ [`92dda57`](https://github.com/nikopueringer/CorridorKey/commit/92dda57dd41a2996be390002e4c0a8f9f67d8bd8) |

## Requirements

- **Python 3.10** (tested with 3.10.0)

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.10.0 | PyTorch model loading & reference inference |
| `timm` | 1.0.25 | Hiera (ViT) backbone used by GreenFormer |
| `opencv-python` | latest | Image / video I/O and preprocessing |
| `huggingface-hub` | 1.7.1 | Automatic checkpoint download |
| `openvino` | 2026.0.0 | Model export & optimised inference on Intel hardware |
| `matplotlib` | 3.10.8 | Visualisation (notebook only) |

## Quick Start

```bash
# Clone the repository (--recursive pulls the CorridorKey submodule)
git clone --recursive https://github.com/daniil-lyakhov/CorridorKeyOpenVINO.git
cd CorridorKeyOpenVINO

# If you already cloned without --recursive, init the submodule:
git submodule update --init

# Create a virtual environment (Python 3.10) and install dependencies
python3.10 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt

# Run the CLI script (generates a synthetic green-screen video)
python export_and_run_openvino.py --img-size 512

# Or open the notebook
jupyter notebook corridorkey_openvino.ipynb
```

## CLI Usage

```bash
# Single image
python export_and_run_openvino.py --image path/to/greenscreen.png --img-size 1024

# Video
python export_and_run_openvino.py --video path/to/clip.mp4 --img-size 512

# Use a different OpenVINO device
python export_and_run_openvino.py --device GPU
```

If neither `--image` nor `--video` is provided, a synthetic green-screen video
(60 frames) is generated and processed as a demo.

## Pipeline

1. **Download** the CorridorKey checkpoint from
   [HuggingFace](https://huggingface.co/nikopueringer/CorridorKey_v1.0)
   (automatic, ~400 MB)
2. **Load** the PyTorch `GreenFormer` model
3. **Export** to OpenVINO IR via `openvino.convert_model()`
4. **Run inference** on Intel hardware (CPU / GPU / NPU)
5. **Compare** OpenVINO vs PyTorch outputs for numerical correctness

## Pre-exported Model

The `ir/corridorkey.xml` / `ir/corridorkey.bin` files are a pre-exported IR at
512×512 resolution so you can skip steps 1–3 and go straight to inference.
To re-export at a different resolution, delete the `ir/` directory and run the
script or notebook — it will download the checkpoint and export automatically.
