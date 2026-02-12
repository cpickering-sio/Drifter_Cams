# Depth Anything 3 — Underwater Organism Depth Estimation
### Step-by-step setup guide for Apple Silicon Mac (M-series)

This tutorial documents the full process of installing and running [Depth Anything 3 (DA3)](https://github.com/ByteDance-Seed/Depth-Anything-3) to generate depth maps and point clouds from underwater video frames, for use in relative length estimation of marine organisms (e.g. salps).

---

## System Requirements

- **OS:** macOS (Apple Silicon M-series — tested on M3 Ultra)
- **Python:** 3.10 or higher
- **Storage:** ~3–4 GB free (model weights + dependencies)
- **Input:** PNG frames extracted from underwater video (stereo or monocular)

---

## Step 1 — Create a Virtual Environment

Always use a virtual environment to avoid conflicts with system Python.

```bash
python3 -m venv da3_env
source da3_env/bin/activate
```

You should see `(da3_env)` at the start of your terminal prompt. **All subsequent commands should be run inside this environment.**

---

## Step 2 — Clone the Repository

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3
```

---

## Step 3 — Install PyTorch

Install PyTorch from the official index (required — the default PyPI version is too old):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify it installed correctly:

```bash
python3 -c "import torch; print(torch.__version__)"
```

You should see `2.8.0` or higher.

---

## Step 4 — Install DA3 (without auto-resolving dependencies)

`xformers` is listed as a dependency but does not support Mac — use `--no-deps` to skip it:

```bash
pip install -e . --no-deps
```

---

## Step 5 — Install Required Dependencies Manually

Run this single command to install everything needed for inference:

```bash
pip install "numpy<2" omegaconf hydra-core einops timm huggingface_hub \
    safetensors opencv-python scipy matplotlib imageio plyfile trimesh \
    open3d pillow-heif requests e3nn "moviepy==1.0.3"
```

> **Note:** `numpy<2` is required — numpy 2.x breaks compatibility with DA3.
> `moviepy==1.0.3` must be exactly this version — newer versions have a different API.

### Packages you do NOT need to install

These are listed as DA3 dependencies but are not required for basic inference and point cloud export:

| Package | Reason to skip |
|---|---|
| `xformers` | No Mac support |
| `pycolmap` | Only needed for COLMAP export |
| `evo` | Only needed for pose evaluation |
| `gsplat` | Only needed for 3D Gaussian Splatting rendering |
| `fastapi` / `uvicorn` | Only needed for web server/app |
| `pre-commit` | Dev tooling only |

---

## Step 6 — Patch Two Files

Two imports in DA3 will cause crashes on Mac. Comment them out with these commands:

**Patch 1 — remove pycolmap import:**
```bash
sed -i '' 's/from .colmap import export_to_colmap/# from .colmap import export_to_colmap/' \
  src/depth_anything_3/utils/export/__init__.py
```

**Patch 2 — remove evo import:**
```bash
sed -i '' 's/from evo.core.trajectory import PosePath3D/# from evo.core.trajectory import PosePath3D/' \
  src/depth_anything_3/utils/pose_align.py
```

---

## Step 7 — Verify the Installation

```bash
python3 -c "from depth_anything_3.api import DepthAnything3; print('DA3 import OK')"
```

Expected output:
```
[WARN ] Dependency `gsplat` is required for rendering 3DGS. ...
DA3 import OK
```

### Warnings that are safe to ignore

| Warning | Why it's safe to ignore |
|---|---|
| `Dependency gsplat is required for rendering 3DGS` | gsplat is only needed for Gaussian Splatting export, not point clouds |
| `Matplotlib is building the font cache` | One-time font cache build, harmless |

---

## Step 8 — Organise Your Frames

Place your PNG frames in a folder inside the repo. If you have stereo frames, keep left and right separate:

```
Depth-Anything-3/
└── DA3/
    ├── left_frames/
    │   ├── frame_000000.png
    │   ├── frame_000001.png
    │   └── ...
    ├── right_frames/
    │   ├── frame_000000.png
    │   └── ...
    └── salp_frames/        ← copy only frames containing your organism here
```

> **Tip:** For best results, manually select only frames that clearly show the organism of interest and copy them into a dedicated subfolder (e.g. `salp_frames`). DA3 works best when all input frames show the same subject.

---

## Step 9 — Run Inference

Save the following script as `run_da3.py` inside your `Depth-Anything-3` folder and update the paths:

```python
"""
DA3 inference script for underwater organism frames
Outputs: GLB point cloud + NPZ depth arrays + depth visualisation images
"""

import glob
import os
import torch
from depth_anything_3.api import DepthAnything3

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ── Paths — update these ──────────────────────────────────────────────────────
LEFT_FRAMES = "/path/to/your/salp_frames"
OUTPUT_DIR  = "/path/to/your/da3_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load frames ───────────────────────────────────────────────────────────────
images = sorted(glob.glob(os.path.join(LEFT_FRAMES, "*.png")))
print(f"Found {len(images)} frames")

if len(images) == 0:
    raise FileNotFoundError(f"No PNG files found in {LEFT_FRAMES}")

# ── Load model ────────────────────────────────────────────────────────────────
# First run downloads ~1.64 GB of model weights automatically
print("Loading model (will download ~1.64GB on first run)...")
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
model = model.to(device=device)
print("Model loaded.")

# ── Run inference ─────────────────────────────────────────────────────────────
print("Running inference...")
try:
    prediction = model.inference(
        images,
        export_dir=OUTPUT_DIR,
        export_format="glb-npz-depth_vis",  # point cloud + raw arrays + depth images
    )
    print("\n✓ Inference complete!")
    print(f"  Depth maps shape : {prediction.depth.shape}  (frames x H x W)")
    print(f"  Confidence shape : {prediction.conf.shape}")
    print(f"  Output saved to  : {OUTPUT_DIR}/")

except RuntimeError as e:
    if "mps" in str(e).lower() or "MPS" in str(e):
        print("\nMPS error — retrying on CPU...")
        device = torch.device("cpu")
        model = model.to(device=device)
        prediction = model.inference(
            images,
            export_dir=OUTPUT_DIR,
            export_format="glb-npz-depth_vis",
        )
        print("\n✓ Inference complete on CPU!")
    else:
        raise
```

Run it:

```bash
python3 run_da3.py
```

Expected output:
```
Using MPS (Apple GPU)
Found N frames
Loading model (will download ~1.64GB on first run)...
[INFO ] using MLP layer as FFN
Model loaded.
Running inference...
[INFO ] Processed Images Done taking X seconds.
[INFO ] Selecting reference view using strategy: saddle_balanced
[INFO ] Model Forward Pass Done. Time: X seconds
[INFO ] Conversion to Prediction Done. Time: X seconds
[INFO ] Exporting to GLB with num_max_points: 1000000
[INFO ] Export Results Done. Time: X seconds

✓ Inference complete!
```

---

## Step 10 — View the Point Cloud

### Option A — Browser (no install required)
Drag the `.glb` file from your output folder into **[gltf.report](https://gltf.report)** in your browser.

> Note: [3dviewer.net](https://3dviewer.net) may show "no meshes found" — use gltf.report instead for point clouds.

### Option B — CloudCompare (recommended for measurements)
1. Download [CloudCompare](https://www.cloudcompare.org) (free)
2. File → Open → select your `.glb` file
3. Use the **Point Picking** tool to click two endpoints on an organism
4. Read the distance value (units are relative, not metric — see note below)

---

## Output Files

| File | Contents | Use for |
|---|---|---|
| `.glb` | 3D point cloud | Visual inspection in browser or CloudCompare |
| `.npz` | Raw depth + confidence arrays | Programmatic measurement in Python |
| `depth_vis_*.png` | Colourised depth maps per frame | Sanity checking DA3 is seeing the organism |

---

## Important Caveats for Underwater Use

- **Depth values are relative, not metric.** DA3 was trained on terrestrial data. Without stereo calibration, you cannot get true mm/cm measurements directly from the output.
- **Water refraction** causes apparent depth to be ~33% shallower than true depth. If you later calibrate and get metric depth, divide by 1.33 to correct.
- **Backscatter and turbidity** reduce confidence in open-water regions. Filter by the `conf` array in the `.npz` if doing programmatic analysis.
- **Stereo without calibration** — if you have stereo frames (left + right cameras) but no calibration data, it is still worth calibrating your rig with a checkerboard pattern. This unlocks true metric depth and is the standard approach in marine biology stereo-video measurement systems.

---

## Valid Export Formats

The `export_format` argument accepts these values (combine with `-`):

| Format | Output |
|---|---|
| `glb` | 3D point cloud (GLB file) |
| `npz` | Raw numpy depth + confidence arrays |
| `depth_vis` | Colourised depth map images |
| `mini_npz` | Smaller/compressed numpy arrays |
| `gs_ply` | 3D Gaussian Splatting PLY (requires gsplat) |
| `gs_video` | Gaussian Splatting video (requires gsplat) |

Example combining formats: `export_format="glb-npz-depth_vis"`

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'depth_anything_3'` | Run `pip install -e .` from inside the inner `Depth-Anything-3` folder where `pyproject.toml` lives |
| `ERROR: Invalid requirement: '#'` | Don't copy inline comments (`# like this`) into terminal commands |
| `No matching distribution found for torch>=2.10` | Install torch from the PyTorch index: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `ModuleNotFoundError: No module named 'omegaconf'` | Run the full dependency install command in Step 5 |
| `ModuleNotFoundError: No module named 'pycolmap'` | Apply the patch in Step 6 |
| `ValueError: Unsupported export format: ply` | Use `glb` not `ply` — see valid formats table above |
| GLB viewer shows "no meshes" | Use gltf.report instead of 3dviewer.net |
| numpy compatibility errors | Ensure `numpy<2` is installed: `pip install "numpy<2"` |
