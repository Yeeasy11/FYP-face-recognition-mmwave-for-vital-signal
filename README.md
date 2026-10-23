# FYP — Deployment

## Quick start (after `git clone`)

The repository **includes trained weights and face profiles**. Only the `datasets/` folder is excluded (for re-training). After clone you can run the demo without downloading `.pkl` / `.pth` files.

1. Install dependencies on the **host PC** and (if used) the **Raspberry Pi** — see sections below.
2. Set **[Network & IP addresses](#network--ip-addresses)** for your LAN (Pi ↔ host).
3. Start **host**: `python fyp_launcher.py` (or `python face_app.py`).
4. Start **Pi** (if used): `python3 shumeipai.py`.
5. Open the Web UI: `http://127.0.0.1:5000` (or `http://<host-lan-ip>:5000` from another device).

**First run:** `facenet-pytorch` may download InceptionResnetV1 weights once (internet required). Everything else for inference is already in the repo.

> **Clone size:** PhysFormer checkpoints, age model, and UBFC epoch files are committed (~hundreds of MB). Use a stable connection or shallow clone if needed.

---

## What is included in Git

| Asset | Path | Notes |
|-------|------|--------|
| PhysFormer (runtime default) | `weights/Physformer_UBFC_best.pkl` | Loaded first by `physformer_engine.py` |
| PhysFormer (VIPL pretrain) | `weights/Physformer_VIPL_fold1.pkl` | Fallback |
| UBFC training checkpoints | `physformer/PhysFormer/Physformer_UBFC_finetune/Physformer_UBFC_1_*.pkl` | All epochs kept for reproducibility |
| Age-group model | `age_model_effnet_b0.pth` | Repo root |
| Face profiles | `face_profiles_v2.npz`, `face_profile_meta.json` | Re-enroll in UI if you prefer a clean slate |
| YOLO weights | `yolov8n.pt` | Present; current pipeline does not require it |

Override HR weights with `PHYSFORMER_WEIGHTS=<full-path>` or disable with `PHYSFORMER_ENABLED=0`.

**Not in Git (only for training):** `datasets/` — see [Training datasets](#training-datasets-not-in-this-repository).

---

## Network & IP addresses

Use two LAN IPs in a typical setup:

| Role | Example placeholder | Used for |
|------|---------------------|----------|
| **Host PC** | `<host-ip>` e.g. `192.168.1.100` | Web UI `:5000`, launcher `:8787` |
| **Raspberry Pi** | `<pi-ip>` e.g. `192.168.1.50` | Radar `/api/radar`, camera `/camera/rgb` on `:5000` |

Ports are fixed unless you override env vars: **5000** = `face_app` / `shumeipai`, **8787** = `fyp_launcher`.

### Recommended: environment variables (no code edit)

Set these **every time** you run, or add them to your shell profile / systemd unit.

**On the host PC** (PowerShell):

```powershell
$env:RADAR_PI_BASE = "http://<pi-ip>:5000"    # Pi service (radar + optional camera)
$env:USE_PI_CAMERA = "1"                         # 1 = pull MJPEG from Pi; 0 = local RealSense on host
python fyp_launcher.py
```

**On the Raspberry Pi** (bash):

```bash
export FYP_PC_HOST=<host-ip>
export FYP_UI_URL=http://<host-ip>:5000
export FYP_LAUNCHER_URL=http://<host-ip>:8787
# export FYP_LAUNCHER_TOKEN=<same-as-host-if-set>
python3 shumeipai.py
```

| Variable | Machine | Meaning |
|----------|---------|---------|
| `RADAR_PI_BASE` | Host | Pi base URL; `face_app` polls `…/api/radar` and (if `USE_PI_CAMERA=1`) `…/camera/rgb` |
| `USE_PI_CAMERA` | Host | `1` = Pi camera stream; `0` = host RealSense |
| `FYP_PC_HOST` | Pi | Host IP; used to build default UI/launcher URLs if those are unset |
| `FYP_UI_URL` | Pi | Full URL of host Web UI (browser / kiosk) |
| `FYP_LAUNCHER_URL` | Pi | Host launcher base (`POST …/launch` to start `face_app`) |
| `FYP_LAUNCHER_TOKEN` | Both | Optional shared secret; must match on Pi and host if set |
| `PHYSFORMER_WEIGHTS` | Host | Optional explicit path to a `.pkl` file |
| `PHYSFORMER_ENABLED` | Host | Set `0` to disable HR model |

### Optional: edit built-in defaults in source (team LAN baked in)

If you prefer not to export variables each session, change the default constants below to your lab IPs (then commit only if the whole team shares the same subnet).

| File | Line (approx.) | Constant | Default | Points to |
|------|----------------|----------|---------|-----------|
| `config.py` | `_DEFAULT_RADAR_PI_BASE` | Pi URL for host | `http://10.200.179.43:5000` | **Pi** |
| `fyp_launcher.py` | `_BUILTIN_DEFAULT_RADAR` | Injected `RADAR_PI_BASE` if unset | `http://10.200.179.43:5000` | **Pi** |
| `shumeipai.py` | `_DEFAULT_PC_HOST` | Host IP when env empty | `10.200.179.140` | **Host** |

`shumeipai.py` also auto-sets (when `FYP_UI_URL` / `FYP_LAUNCHER_URL` are not set):

- `FYP_UI_URL` → `http://<FYP_PC_HOST>:5000`
- `FYP_LAUNCHER_URL` → `http://<FYP_PC_HOST>:8787`

**Keep Pi and host defaults in sync:** `config.py` / `fyp_launcher.py` must use the **Pi** IP; `shumeipai.py` must use the **host** IP.

### Where each service connects (data flow)

```text
Browser  →  http://<host-ip>:5000     (face_app Web UI)
face_app →  RADAR_PI_BASE/api/radar   (Pi)
face_app →  RADAR_PI_BASE/camera/rgb  (Pi, if USE_PI_CAMERA=1)
shumeipai → FYP_LAUNCHER_URL/launch   (host :8787, wake face_app)
shumeipai → FYP_UI_URL                (host :5000, optional kiosk browser)
```

### Local-only URLs (usually do not change)

| URL | File | Note |
|-----|------|------|
| `http://127.0.0.1:5000` | `face_app.py`, `fyp_launcher.py` | Browser on the **same machine** as the host |
| `http://127.0.0.1:5000/api/status` | `fyp_launcher.py` → `FYP_FACEAPP_CHECK` | Launcher health check |

---

## Third-party code

This repository includes the official PhysFormer implementation under `physformer/PhysFormer/`, obtained from [ZitongYu/PhysFormer](https://github.com/ZitongYu/PhysFormer). Our application loads that code for training and inference via `physformer_engine.py`.

## Training datasets (not in this repository)

Only `datasets/` is gitignored. Obtain these locally if you **re-train** models:

| Dataset | Used for | Typical local path |
|---------|----------|----------------------|
| UBFC-rPPG | PhysFormer fine-tuning | `datasets/UBFC-rPPG/subject*/` (`vid.avi`, `ground_truth.txt`) |
| VIPL-HR | PhysFormer pretrain (optional; fold1 `.pkl` already in `weights/`) | VIPL release / upstream README |
| UTKFace / FairFace | Age-group classifier | `datasets/UTKFace/` or FairFace root + CSV |

## System layout

| Component | Machine | Role |
|-----------|---------|------|
| `shumeipai.py` | **Raspberry Pi** | 24 GHz radar serial I/O, optional RealSense/UVC camera MJPEG (`/camera/rgb`, `/camera/depth`), radar JSON (`/api/radar`) |
| `fyp_launcher.py` | Host PC (Windows/Linux) | Starts `face_app.py`, exposes `POST /launch` for the Pi to wake the host UI |
| `face_app.py` | Host PC | Face enrollment/recognition, age estimation, PhysFormer HR, visual respiration, radar fusion; Web UI on port **5000** |

The browser always points at the **host** (`http://<host-ip>:5000`). The Pi does not run face or PhysFormer models.

## Deploy on Raspberry Pi (`shumeipai.py`)

**Run only on the Raspberry Pi** (radar + camera edge node):

```bash
cd /path/to/FYP
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python-headless Flask numpy scipy pyserial

# Intel RealSense D435 on the Pi (optional): build librealsense + pyrealsense2 per Intel ARM docs
# https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_raspbian.md

export FYP_SERIAL=/dev/ttyUSB0
export FYP_PC_HOST=<host-ip>
export FYP_UI_URL=http://<host-ip>:5000
export FYP_LAUNCHER_URL=http://<host-ip>:8787
# export FYP_LAUNCHER_TOKEN=<shared-secret>

python3 shumeipai.py
```

Service listens on `0.0.0.0:5000` by default (`FYP_PORT`). Useful endpoints: `/api/radar`, `/camera/rgb`, `/camera/depth`. Set `FYP_CAMERA_ENABLE=0` for radar-only.

Camera modes (`FYP_CAMERA_MODE`): `auto` (RealSense then UVC), `realsense`, or `uvc`. Use `FYP_CAMERA_DEVICE=/dev/videoN` when V4L2 index probing is wrong.

## Deploy on host PC (`fyp_launcher.py` + `face_app.py`)

Recommended: use one Python environment with CUDA if available (PhysFormer).

```powershell
cd C:\path\to\FYP
conda create -n fyp python=3.10 -y
conda activate fyp
pip install torch torchvision opencv-python pyrealsense2 ultralytics facenet-pytorch flask numpy scipy
```

**Option A — Pi camera + Pi radar (default in `config.py`):**

```powershell
$env:RADAR_PI_BASE = "http://<pi-ip>:5000"
$env:USE_PI_CAMERA = "1"
python fyp_launcher.py
```

`fyp_launcher` starts `face_app` and listens on **8787** (`FYP_LAUNCHER_PORT`). Open `http://127.0.0.1:5000` in a browser on the host.

**Option B — Local Intel RealSense on the host:**

```powershell
$env:USE_PI_CAMERA = "0"
$env:RADAR_PI_BASE = "http://<pi-ip>:5000"   # still required if radar runs on the Pi
python face_app.py
```

## Citations

If you use PhysFormer or VIPL-HR, please cite:

```bibtex
@inproceedings{yu2021physformer,
   title={PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer},
   author={Yu, Zitong and Shen, Yuming and Shi, Jingang and Zhao, Hengshuang and Torr, Philip and Zhao, Guoying},
   booktitle={CVPR},
   year={2022}
}

@article{yu2023physformer++,
  title={PhysFormer++: Facial Video-based Physiological Measurement with SlowFast Temporal Difference Transformer},
  author={Yu, Zitong and Shen, Yuming and Shi, Jingang and Zhao, Hengshuang and Cui, Yawen and Zhang, Jiehua and Torr, Philip and Zhao, Guoying},
  journal={International Journal of Computer Vision (IJCV)},
  pages={1--24},
  year={2023}
}

@article{niu2019rhythmnet,
  title={Rhythmnet: End-to-end heart rate estimation from face via spatial-temporal representation},
  author={Niu, Xuesong and Shan, Shiguang and Han, Hu and Chen, Xilin},
  journal={IEEE Transactions on Image Processing},
  year={2019}
}
```
