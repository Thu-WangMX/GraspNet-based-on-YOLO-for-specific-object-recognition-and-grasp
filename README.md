# 🤖 GraspNet-based-on-YOLO-for-specific-object-recognition-and-grasp

An experimental real-robot grasping pipeline that combines:

- 🧠 **YOLOv8-based stain / target region detection** for **specific-object recognition and workspace masking**.
- ✋ **GraspNet baseline** (point-cloud grasp detection with PointNet++ backbone) for **6-DoF grasp synthesis**.
- 🤝 **Flexiv robot** execution (via Flexiv RDK) for **end-to-end autonomous grasping** driven by RGB-D perception.

The system is designed for **tabletop scenes** where a specific object or stained region (e.g., a dirty patch) is detected by YOLO and then handed over to GraspNet to compute feasible grasps for a physical Flexiv arm.

> ⚠️ This repository is **research / experimental code**. It is not a general-purpose SDK and assumes familiarity with Linux, Conda, CUDA, RealSense, and Flexiv RDK.

---

## ✨ 1. Features

- 🎯 **Target-aware perception**
  - Uses a YOLOv8 model (e.g. `yolo8l_batch8_run1.pt`) to detect a **specific semantic class** (e.g. `"solid"` stain) in the RGB image.
  - Generates a **binary workspace mask** (`generated_masks/mask_*.png`) highlighting the target region.

- 🧩 **Grasp synthesis (GraspNet baseline)**
  - Converts depth image to a point cloud using known **camera intrinsics**.
  - Applies the workspace mask to filter the point cloud.
  - Runs a GraspNet baseline network to predict grasp candidates and selects the best grasp.

- 📷 **RealSense camera integration**
  - Uses Intel RealSense RGB-D camera(s) via `pyrealsense2`.
  - Initializes streams, applies depth filtering, and exposes `get_frames()` and `get_intrinsics()` via `RealsenseAPI`.

- 🤖 **Flexiv robot integration**
  - Uses a `FlexivRobot` wrapper around Flexiv RDK to:
    - 🔓 Open / 🔒 close gripper.
    - 🧭 Move along **Cartesian trajectories** via `trajectory_planner.py`.
    - 🔁 Execute a **complete grasp cycle**: move to capture pose → capture RGB-D → detect stain/target → compute grasp → execute grasp → lift and place.

---

## 📂 2. Repository Structure (simplified)

Some important files and directories:

```text
GraspNet-based-on-YOLO-for-specific-object-recognition-and-grasp/
├── grasp_out_api.py                # Main entry for the full pipeline (perception + GraspNet + robot)
├── FlexivRobot.py                  # Flexiv RDK wrapper (motion, gripper, safety helpers)
├── trajectory_planner.py           # Cartesian motion planning wrappers for Flexiv
├── pointnet2/                      # PointNet++ CUDA extension (for GraspNet)
│   ├── setup.py                    # Build script for C++/CUDA ops
│   └── _ext_src/                   # C++/CUDA sources
├── robot-stain-perception/
│   ├── tools/
│   │   ├── grasp_mask_api.py       # YOLO + depth → workspace mask generation
│   │   ├── perception_api_detect.py# StainPerceptionAPI (YOLOv8 detector)
│   │   └── realsense_api.py        # RealsenseAPI (RGB-D stream + intrinsics)
│   └── ...                         # Additional perception utilities
├── graspnet-baseline/              # GraspNet baseline code (as a submodule/ported copy)
│   └── ...                         # GraspNet dataset, model, utils
├── captured_img/                   # Saved RGB/D images for debugging
├── generated_masks/                # Saved binary masks from YOLO detection
├── checkpoints/                    # GraspNet model checkpoints
└── yolo8l_batch8_run1.pt           # YOLOv8 weights (not included; place manually)
```

> ℹ️ Paths / filenames above are illustrative and may need to be adjusted based on your local layout.

---

## ⚙️ 3. Setup

### 3.1. 🐍 Environment (example)

You can start from a clean Conda environment (Python 3.10 recommended):

```bash
conda create -n flexiv python=3.10 -y
conda activate flexiv
```

Install PyTorch + CUDA (adjust to your CUDA version as needed):

```bash
# Example used during development (CUDA 11.7)
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
```

Install basic dependencies (you may adjust / extend this list):

```bash
pip install opencv-python
pip install pyrealsense2
pip install matplotlib
pip install open3d
pip install pyyaml
pip install tqdm
pip install ultralytics   # for YOLOv8, if not already installed
```

You also need:

- 🦾 **Flexiv RDK** installed and configured on this machine.
- 🖧 The Flexiv controller & robot properly connected and reachable.
- 📸 Intel **RealSense SDK** (librealsense / `pyrealsense2`) installed and tested.

### 3.2. 🚀 Build PointNet++ CUDA extension (for GraspNet)

From the `pointnet2` directory:

```bash
cd pointnet2
pip install ninja  # optional but strongly recommended
python setup.py install
cd ..
```

If the build succeeds, you should be able to import:

```bash
python -c "import pointnet2; print('pointnet2 imported OK')"
```

---

## 📦 4. Preparing Models

### 4.1. 🧠 YOLOv8 weights

Place your YOLOv8 model checkpoint (e.g. trained on stain / specific-object dataset) into the repo root, for example:

```text
yolo8l_batch8_run1.pt
```

Ensure `grasp_out_api.py` and `grasp_mask_api.py` use the correct path, for example:

```python
YOLO_MODEL_PATH = "/absolute/path/to/yolo8l_batch8_run1.pt"
```

### 4.2. ✋ GraspNet checkpoints

Download the GraspNet baseline checkpoints (e.g. from the original GraspNet repository) and place them under:

```text
checkpoints/
    checkpoint-rs.tar
    ...
```

Update the corresponding path in `grasp_out_api.py` (or wherever the model is loaded), e.g.:

```python
CHECKPOINT_PATH = "checkpoints/checkpoint-rs.tar"
```

---

## ▶️ 5. Running the System

> ⚠️ Make sure the Flexiv robot is in a safe state, the workspace is clear, and you understand the motion before running any code on hardware.

### 5.1. 🧪 RealSense & YOLO mask-only test

You can first test the perception & mask generation (without robot motion) by using the utility in `robot-stain-perception/tools/grasp_mask_api.py` directly, or by running the part of `grasp_out_api.py` that only:

1. Initializes `RealsenseAPI`.
2. Captures a frame.
3. Runs `StainPerceptionAPI.detect_stains`.
4. Produces and saves `generated_masks/mask_*.png`.

Check that:

- ✅ The RGB image in `captured_img/` looks correct.
- ✅ The mask in `generated_masks/` is not all black and highlights the target region.

### 5.2. 🤖 Full pipeline (grasp_out_api.py)

Once perception is working and the environment is safe, run:

```bash
conda activate flexiv
python grasp_out_api.py
```

Typical flow:

1. The script connects to the RealSense camera and warms it up.
2. YOLOv8 runs on the captured RGB frame and detects stain / specific objects.
3. A binary workspace mask is generated and saved.
4. The script then:
   - Captures a new RGB-D image for GraspNet.
   - Converts depth to a point cloud using camera intrinsics.
   - Applies the workspace mask to filter points.
   - Runs GraspNet to detect a set of grasp candidates.
   - Chooses the best grasp (based on score, width, etc.).
5. You will be asked to confirm execution:
   - The Flexiv robot opens the gripper, follows a Cartesian trajectory to the grasp pose, closes the gripper, lifts, and places the object at a bin pose.

If anything goes wrong (e.g. empty point cloud, no detections), the script prints an error message and stops the robot.

---

## 🛠️ 6. Troubleshooting

- ❗ **`AttributeError: 'RealsenseAPI' object has no attribute 'get_intrinsics'`**

  Ensure that `RealsenseAPI` implements a `get_intrinsics()` method returning a dict with at least:

  ```python
  {
      "fx": ...,
      "fy": ...,
      "ppx": ...,
      "ppy": ...,
      "width": ...,
      "height": ...,
      "depth_scale": ...
  }
  ```

  and that it uses the same intrinsics as the depth/RGB used by GraspNet.

- ❗ **`KeyError: 'reachedTarget'` in `FlexivRobot.MoveL_multi_points`**

  Newer Flexiv RDK versions may not include a `reachedTarget` field in `primitive_states()`.  
  Guard against missing keys, for example:

  ```python
  states = self.robot.primitive_states()
  reached = states.get("reachedTarget", True)  # default True if key is missing
  ```

- ❗ **`ValueError: a must be greater than 0` in `np.random.choice`**

  This means the masked point cloud (`cloud_masked`) is empty. Common reasons:

  - The workspace mask is all black or misaligned.
  - Depth in the mask region is zero (no valid depth).
  - The wrong mask path is used.

  Add checks in `process_data()` to handle empty `cloud_masked` and verify your mask visually.

---

## 🙏 7. Acknowledgements

This project builds upon and/or is inspired by:

- 🤝 **[GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)** – point-cloud grasp detection and dataset utilities.
- 🧠 **YOLOv8 (Ultralytics)** – real-time object detection backbone used here for stain / object detection.
- 🦾 **Flexiv RDK** – robotic control stack used for Cartesian motion and gripper control.
- 📷 **Intel RealSense** – RGB-D sensing for both YOLO and GraspNet.

Please also cite the corresponding original papers / repositories if you use this project in academic work.

