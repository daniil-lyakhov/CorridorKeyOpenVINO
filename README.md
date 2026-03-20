# CorridorKey: Fast Inference on Intel Hardware with OpenVINO

Export the [CorridorKey](https://github.com/nikopueringer/CorridorKey) green-screen keying model to [OpenVINO](https://docs.openvino.ai/) and run fast inference on **Intel CPUs, GPUs, and NPUs**.

## Requirements

- **Python 3.10** (tested with 3.10.0)

Everything is listed in `requirements.txt`, so a single `pip install` covers it all.

## Quick start

```bash
# 1. Clone (--recursive grabs the CorridorKey submodule automatically)
git clone --recursive https://github.com/daniil-lyakhov/CorridorKeyOpenVINO.git
cd CorridorKeyOpenVINO

# Already cloned without --recursive? Just run:
#   git submodule update --init

# 2. Create a venv and install dependencies
python3.10 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt

# 3. Run the demo (processes a synthetic green-screen video)
python export_and_run_openvino.py --img-size 512

# …or open the notebook for a step-by-step walkthrough
jupyter notebook corridorkey_openvino.ipynb
```

## CLI usage

```bash
# Process a single image
python export_and_run_openvino.py --image path/to/greenscreen.png --img-size 1024

# Process a video
python export_and_run_openvino.py --video path/to/clip.mp4 --img-size 512

# Target a different Intel device
python export_and_run_openvino.py --device GPU
```

If you don't pass `--image` or `--video`, the script generates a synthetic green-screen video (60 frames) and processes it as a quick demo.

## How it works

1. **Download** the CorridorKey checkpoint from [HuggingFace](https://huggingface.co/nikopueringer/CorridorKey_v1.0) (~400 MB, cached after the first run)
2. **Load** the PyTorch `GreenFormer` model
3. **Export** to OpenVINO IR with `openvino.convert_model()`
4. **Run inference** on Intel hardware (CPU / GPU / NPU)
5. **Compare** OpenVINO vs. PyTorch outputs to verify numerical correctness

## Using the pre-exported model

The repo ships a pre-exported IR at 512 × 512 resolution (`ir/corridorkey.xml` + `ir/corridorkey.bin`), so you can skip steps 1–3 and jump straight to inference.

If you need a different resolution, just delete the `ir/` directory and re-run the script or notebook — the checkpoint will be downloaded and a fresh IR will be exported automatically.

## Model Compression

Comming soon!
