import os

WIDTH = 640
HEIGHT = 480
FPS = 30

VIDEO_MINIMAL_OVERLAY = os.environ.get("VIDEO_MINIMAL_OVERLAY", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
VIDEO_JPEG_QUALITY = max(40, min(95, int(os.environ.get("VIDEO_JPEG_QUALITY", "78"))))
VIDEO_JPEG_DEPTH_QUALITY = max(40, min(95, int(os.environ.get("VIDEO_JPEG_DEPTH_QUALITY", "72"))))
VIDEO_ENCODE_DEPTH_STREAM = os.environ.get("VIDEO_ENCODE_DEPTH_STREAM", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

_DEFAULT_RADAR_PI_BASE = "http://10.200.179.43:5000"
RADAR_PI_BASE = (os.environ.get("RADAR_PI_BASE") or _DEFAULT_RADAR_PI_BASE).strip().rstrip(
    "/"
)

_USE_PI_RAW = (os.environ.get("USE_PI_CAMERA") or "1").strip().lower()
USE_PI_CAMERA = _USE_PI_RAW not in ("0", "false", "no")

WINDOW_NAME = "D435 Person Detection"
DEPTH_WINDOW_NAME = "D435 Depth"

YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.35
YOLO_IMGSZ = 640
YOLO_DEVICE = "cpu"

MIN_BBOX_WIDTH = 40
MIN_BBOX_HEIGHT = 80

TRACK_IOU_THRESHOLD = 0.3
TRACK_MIN_HITS = 1
TRACK_MAX_MISSES = 8
TRACK_DRAW_MAX_MISSES = 3

DEPTH_VALID_MIN_M = 0.15
DEPTH_VALID_MAX_M = 10.0
DEPTH_MIN_VALID_PIXELS = 5

DEPTH_VITAL_OPTIMAL_MIN_M = 0.45
DEPTH_VITAL_OPTIMAL_MAX_M = 1.9

FUSION_DEPTH_SCENE_MIN_M = float(os.environ.get("FUSION_DEPTH_SCENE_MIN_M", "0.35"))
FUSION_DEPTH_SCENE_MAX_M = float(os.environ.get("FUSION_DEPTH_SCENE_MAX_M", "2.3"))
FUSION_USE_DEPTH_BIN_MATCH = os.environ.get(
    "FUSION_USE_DEPTH_BIN_MATCH", "1"
).strip() not in ("0", "false", "False")

DEPTH_VIS_ALPHA = 0.03

FACE_DEPTH_QUALITY_GATE = os.environ.get("FACE_DEPTH_QUALITY_GATE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

RECOGNITION_LOCK_WINDOW_SEC = float(os.environ.get("RECOGNITION_LOCK_WINDOW_SEC", "5"))
RECOGNITION_LOCK_MAJORITY = float(os.environ.get("RECOGNITION_LOCK_MAJORITY", "0.51"))
RECOGNITION_LOCK_MIN_VOTES = max(3, int(os.environ.get("RECOGNITION_LOCK_MIN_VOTES", "10")))

AGE_LOCK_ON_CORRECT = os.environ.get("AGE_LOCK_ON_CORRECT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
AGE_BAND_SEARCH_ON_ERROR = os.environ.get("AGE_BAND_SEARCH_ON_ERROR", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

FACE_PROFILE_FILE = "face_profiles_v2.npz"
FACE_PROFILE_META_FILE = "face_profile_meta.json"
FACE_PROFILE_VERSION = 2
FACE_MAX_PROFILES = 10
DISPLAY_NAME_MAX_LEN = 40

FACE_MIRROR_PREVIEW = os.environ.get("FACE_MIRROR_PREVIEW", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

ENROLL_SAMPLES_TARGET = 20
ENROLL_SAMPLE_INTERVAL_SEC = 0.25

FACE_RECOGNITION_THRESHOLD = 0.42

FACE_DUPLICATE_THRESHOLD = 0.72

FACE_MIN_MTCNN_PROB = 0.85

FACE_MIN_BLUR_VARIANCE = 40.0

FACE_MTCNN_MIN_FACE_SIZE = 60

FACE_PRETRAINED = "vggface2"

AGE_MODEL_FILE = "age_model_effnet_b0.pth"
AGE_INPUT_SIZE = 224
AGE_CALIBRATION_SHIFT = 1
AGE_SMOOTHING_WINDOW = 7
AGE_TRAIN_UTKFACE_ROOT = "UTKFace"
AGE_TRAIN_FAIRFACE_ROOT = None
AGE_TRAIN_FAIRFACE_CSV = None
PHYSFORMER_ENABLED = os.environ.get("PHYSFORMER_ENABLED", "1").strip() not in (
    "0",
    "false",
    "False",
)
PHYSFORMER_WEIGHTS = os.environ.get("PHYSFORMER_WEIGHTS", "").strip()

AGE_CLASSES = [
    {"id": "child", "label": "儿童", "label_en": "Child", "range": "0-12"},
    {"id": "teen", "label": "青少年", "label_en": "Teen", "range": "13-19"},
    {"id": "young_adult", "label": "青年", "label_en": "Young Adult", "range": "20-35"},
    {"id": "middle_aged", "label": "中年", "label_en": "Middle Age", "range": "36-55"},
    {"id": "senior", "label": "老年", "label_en": "Senior", "range": "56+"},
]
