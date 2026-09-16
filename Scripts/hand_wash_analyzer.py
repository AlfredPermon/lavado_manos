import os
import logging
import warnings
import contextlib
import sys
import importlib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")

warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy.*",
)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@contextlib.contextmanager
def _suppress_startup_noise():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stderr(devnull), contextlib.redirect_stdout(devnull):
            yield

import cv2
import numpy as np
from collections import deque, Counter, defaultdict
with _suppress_startup_noise():
    import tensorflow as tf
    import keras
    from keras.layers import Layer
    import mediapipe as mp
    import cvzone.HandTrackingModule as _cvhm
    from cvzone.HandTrackingModule import HandDetector
from ultralytics import YOLO
import torch

tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except ImportError:
    pass

try:
    if "keras.src.models.functional" not in sys.modules:
        sys.modules["keras.src.models.functional"] = importlib.import_module(
            "keras.src.engine.functional"
        )
except ImportError:
    pass

_original_inputlayer_from_config = tf.keras.layers.InputLayer.from_config


@classmethod
def _compat_inputlayer_from_config(cls, config):
    config = dict(config)
    if "batch_shape" in config and "batch_input_shape" not in config:
        config["batch_input_shape"] = config.pop("batch_shape")
    return _original_inputlayer_from_config.__func__(cls, config)


tf.keras.layers.InputLayer.from_config = _compat_inputlayer_from_config
keras.layers.InputLayer.from_config = _compat_inputlayer_from_config

_original_layernorm_from_config = tf.keras.layers.LayerNormalization.from_config


@classmethod
def _compat_layernorm_from_config(cls, config):
    config = dict(config)
    config.pop("rms_scaling", None)
    return _original_layernorm_from_config.__func__(cls, config)


tf.keras.layers.LayerNormalization.from_config = _compat_layernorm_from_config
keras.layers.LayerNormalization.from_config = _compat_layernorm_from_config

_original_mha_from_config = tf.keras.layers.MultiHeadAttention.from_config


@classmethod
def _compat_mha_from_config(cls, config):
    config = dict(config)
    if "query_shape" not in config:
        shapes_dict = config.pop("shapes_dict", None)
        build_config = config.pop("build_config", None)
        if shapes_dict is None and isinstance(build_config, dict):
            shapes_dict = build_config.get("shapes_dict")
        if isinstance(shapes_dict, dict):
            if "query_shape" in shapes_dict:
                config["query_shape"] = shapes_dict["query_shape"]
            if "value_shape" in shapes_dict:
                config["value_shape"] = shapes_dict["value_shape"]
            if "key_shape" in shapes_dict:
                config["key_shape"] = shapes_dict["key_shape"]
    if "query_shape" not in config:
        allowed_keys = {
            "num_heads", "key_dim", "value_dim", "dropout", "use_bias",
            "output_shape", "attention_axes", "kernel_initializer",
            "bias_initializer", "kernel_regularizer", "bias_regularizer",
            "activity_regularizer", "kernel_constraint", "bias_constraint",
            "name", "dtype", "trainable",
        }
        kwargs = {k: v for k, v in config.items() if k in allowed_keys}
        return cls(**kwargs)
    return _original_mha_from_config.__func__(cls, config)


tf.keras.layers.MultiHeadAttention.from_config = _compat_mha_from_config
keras.layers.MultiHeadAttention.from_config = _compat_mha_from_config

try:
    from keras.src.mixed_precision import policy as _keras_policy

    _original_get_policy = _keras_policy.get_policy

    def _compat_get_policy(identifier):
        if isinstance(identifier, dict):
            name = identifier.get("config", {}).get("name") or identifier.get("name")
            if name:
                return _keras_policy.Policy(name)

        result = _original_get_policy(identifier)
        if isinstance(result, str):
            return _keras_policy.Policy(result)
        return result

    _keras_policy.get_policy = _compat_get_policy
except Exception:
    pass


class ProjectHandDetector(HandDetector):
    def __init__(self, model_path, staticMode=False, maxHands=2,
                 modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.tipIds = [4, 8, 12, 16, 20]

        with open(model_path, "rb") as model_file:
            model_bytes = model_file.read()

        base_options = _cvhm.python.BaseOptions(model_asset_buffer=model_bytes)
        options = _cvhm.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.maxHands,
            running_mode=_cvhm.vision.RunningMode.VIDEO
        )

        self.detector = _cvhm.vision.HandLandmarker.create_from_options(options)
        self.timestamp = 0

# ============================================================
# CUSTOM LAYERS
# ============================================================


def _normalize_custom_layer_kwargs(kwargs):
    normalized = dict(kwargs)
    if "dtype" in normalized:
        normalized.pop("dtype", None)
    return normalized

class SliceLast(Layer):
    def __init__(self, start: int, end: int, **kwargs):
        super().__init__(**_normalize_custom_layer_kwargs(kwargs))
        self.start, self.end = start, end

    def call(self, x):
        return x[..., self.start:self.end]

    def compute_output_shape(self, input_shape):
        *rest, last = input_shape
        out_last = None if last is None else (self.end - self.start)
        return (*rest, out_last)

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "end": self.end})
        return config


class TimeValidMask(Layer):
    def __init__(self, **kwargs):
        super().__init__(**_normalize_custom_layer_kwargs(kwargs))

    def call(self, x):
        return tf.math.logical_not(
            tf.reduce_all(tf.equal(x, 0.0), axis=-1)
        )

    def compute_output_shape(self, input_shape):
        b, t, _ = input_shape
        return (b, t)


class ExpandDim(Layer):
    def __init__(self, axis: int, **kwargs):
        super().__init__(**_normalize_custom_layer_kwargs(kwargs))
        self.axis = axis

    def call(self, x):
        return tf.expand_dims(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if self.axis < 0:
            axis = len(input_shape) + 1 + self.axis
        else:
            axis = self.axis
        return input_shape[:axis] + (1,) + input_shape[axis:]

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

# ============================================================
# HELPER FUNCTIONS (Feature Extraction)
# ============================================================

def extract_hand_landmarks_mp(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

def extract_frame_landmarks_main(frame, hands_mp):
    results = hands_mp.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_data = []
    presence_flags = [0, 0]
    
    # Metadata for logic (not for model)
    hands_info = {'Left': None, 'Right': None}

    if results.multi_hand_landmarks:
        hands_landmarks = results.multi_hand_landmarks
        multi_handedness = results.multi_handedness
        num_hands = len(hands_landmarks)

        # Standard Feature Extraction (Maintain original order/logic for Model compatibility)
        # The model was likely trained on whatever MP outputted (usually detection order).
        # We MUST NOT change 'frame_data' structure.
        
        frame_data.extend(extract_hand_landmarks_mp(hands_landmarks[0]))
        presence_flags[0] = 1
        
        # Try to identify handedness for logic
        if multi_handedness:
            try:
                label = multi_handedness[0].classification[0].label
                hands_info[label] = hands_landmarks[0]
            except: pass

        if num_hands > 1:
            frame_data.extend(extract_hand_landmarks_mp(hands_landmarks[1]))
            presence_flags[1] = 1
            
            if multi_handedness and len(multi_handedness) > 1:
                try:
                    label = multi_handedness[1].classification[0].label
                    hands_info[label] = hands_landmarks[1]
                except: pass
        else:
            frame_data.extend([0.0] * 63)
    else:
        frame_data.extend([0.0] * 63)
        frame_data.extend([0.0] * 63)

    frame_data.extend(presence_flags)
    return frame_data, hands_info

def normalize_coords(lm_list, img_w, img_h):
    coords = []
    for x, y, z in lm_list:
        coords.append(x / img_w)
        coords.append(y / img_h)
        coords.append(z)
    return coords

def distance_norm(p1, p2, img_w, img_h):
    dx = (p1[0] - p2[0]) / img_w
    dy = (p1[1] - p2[1]) / img_h
    return float((dx**2 + dy**2) ** 0.5)

def bbox_iou_norm(bbox1, bbox2, img_w, img_h):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    inter_x_min = max(x1, x2)
    inter_y_min = max(y1, y2)
    inter_x_max = min(x1 + w1, x2 + w2)
    inter_y_max = min(y1 + h1, y2 + h2)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area + 1e-6
    return float(inter_area / union_area)

def extract_frame_features_cvzone(frame, detector):
    img_h, img_w, _ = frame.shape
    hands, _ = detector.findHands(frame, draw=False, flipType=False)

    frame_coords = []
    presence_flags = [0, 0]
    extra_feats = []

    if not hands:
        frame_coords.extend([0.0] * 63)
        frame_coords.extend([0.0] * 63)
        extra_feats.extend([0.0] * 15)
        extra_feats.extend([0.0] * 15)
        extra_feats.extend([0.0] * 3)
        return frame_coords + presence_flags + extra_feats

    hands_sorted = sorted(hands, key=lambda h: h["center"][0])
    num_hands = len(hands_sorted)

    hand_centers = []
    hand_bboxes = []
    wrist_points = []

    for i in range(2):
        if i < num_hands:
            hand = hands_sorted[i]
            lm_list = hand["lmList"]
            bbox = hand["bbox"]
            cx, cy = hand["center"]
            hand_type = hand["type"]

            presence_flags[i] = 1

            coords = normalize_coords(lm_list, img_w, img_h)
            frame_coords.extend(coords)

            hand_centers.append((cx, cy))
            hand_bboxes.append(bbox)
            wrist_points.append(lm_list[0])

            x_bbox, y_bbox, w_bbox, h_bbox = bbox
            x_bn, y_bn, w_bn, h_bn = x_bbox/img_w, y_bbox/img_h, w_bbox/img_w, h_bbox/img_h
            cx_n, cy_n = cx/img_w, cy/img_h

            type_onehot = [1.0, 0.0] if hand_type == "Left" else [0.0, 1.0]
            fingers = detector.fingersUp(hand)

            wrist = lm_list[0]
            mcp_mid = lm_list[9]
            idx_tip = lm_list[8]
            pinky_tip = lm_list[20]

            palm_size = distance_norm(wrist, mcp_mid, img_w, img_h)
            finger_spread = distance_norm(idx_tip, pinky_tip, img_w, img_h)

            feats_hand = [
                x_bn, y_bn, w_bn, h_bn, cx_n, cy_n,
                *type_onehot, *[float(f) for f in fingers],
                float(palm_size), float(finger_spread)
            ]
            extra_feats.extend(feats_hand)
        else:
            frame_coords.extend([0.0] * 63)
            extra_feats.extend([0.0] * 15)

    if num_hands >= 2:
        (cx1, cy1), (cx2, cy2) = hand_centers[0], hand_centers[1]
        dist_centers = distance_norm([cx1, cy1, 0], [cx2, cy2, 0], img_w, img_h)
        wrist1, wrist2 = wrist_points[0], wrist_points[1]
        dist_wrists = distance_norm(wrist1, wrist2, img_w, img_h)
        bbox1, bbox2 = hand_bboxes[0], hand_bboxes[1]
        iou = bbox_iou_norm(bbox1, bbox2, img_w, img_h)
    else:
        dist_centers, dist_wrists, iou = 0.0, 0.0, 0.0

    extra_feats.extend([float(dist_centers), float(dist_wrists), float(iou)])
    return frame_coords + presence_flags + extra_feats

# ============================================================
# ANALYZER CLASS
# ============================================================

class HandWashAnalyzer:
    def __init__(self, model_config="v2_35"):
        self.model_config = model_config
        self.models_main = []
        self.models_cvz = []
        self.yolo_model = None
        
        # Default configs
        self.n_timesteps_main = 35
        self.n_timesteps_cvz = 35
        self.stride = 1
        self.confidence_threshold = 0.4
        
        # Paths
        self.model_paths_main = []
        self.model_paths_cvz = []
        self.yolo_path = r"Modelos\YOLOV-ED-74.pt"
        
        self._setup_config()
        
        # State
        self.reset_state()

        self.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.scripts_dir = os.path.abspath(os.path.dirname(__file__))
        
        # Tools
        with _suppress_startup_noise():
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        # Ensure cvzone's HandDetector uses the local `hand_landmarker.task` file
        # This prevents mediapipe from trying to open a malformed path that mixes
        # the venv site-packages path and a relative project path (observed on Windows).
        self.hand_landmarker_task_path = self._resolve_hand_landmarker_task_path()

        with _suppress_startup_noise():
            self.detector = ProjectHandDetector(
                self.hand_landmarker_task_path,
                maxHands=2,
                detectionCon=0.6,
                minTrackCon=0.5
            )
        self.yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validation thresholds
        self.seq_len_req = 30
        self.yolo_peaks_req = 5

    def reset_state(self):
        self.sequence_main = deque(maxlen=self.n_timesteps_main)
        self.sequence_cvz = deque(maxlen=self.n_timesteps_cvz)
        
        self.pred_buf_main = [deque(maxlen=5) for _ in self.models_main]
        self.pred_buf_cvz = [deque(maxlen=5) for _ in self.models_cvz]
        
        self.current_step_idx = 0
        self.frames_validated = 0
        self.wash_complete = False
        
        # YOLO State
        self.yolo_peak_counter = 0
        self.yolo_in_peak = False
        self.yolo_peak_timer = 0
        
        self.yolo_armed = False
        self.p0_done_once = False
        self.p1_done_after_p0 = False
        self.aux_p0_frames = 0
        self.aux_p1_frames = 0
        self.p0_event_cooldown = 0
        self.p1_event_cooldown = 0
        
        # Bilateral Tracking (Side A / Side B)
        self.bilateral_state = {
            'active': False,
            'side_a_frames': 0,
            'side_b_frames': 0,
            'current_side': None, # 'A' or 'B'
            'last_switch_time': 0,
            'overtime_frames': 0
        }

    def _resolve_hand_landmarker_task_path(self):
        candidates = [
            os.path.join(self.scripts_dir, "hand_landmarker.task"),
            os.path.join(self.project_dir, "Scripts", "hand_landmarker.task"),
            os.path.join(os.getcwd(), "hand_landmarker.task"),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        raise FileNotFoundError(
            "No se encontró 'hand_landmarker.task' en las rutas esperadas."
        )

    def _setup_config(self):
        if self.model_config == "v2_35":
            self.n_timesteps_main = 35
            self.n_timesteps_cvz = 35
            self.model_paths_main = [
                r"New Models\Mediapipe Models\LSTM_Original_Model_35_Timesteps.keras",
                r"New Models\Mediapipe Models\LSTM_CNN_Model_35_Timesteps.keras",
            ]
            self.model_paths_cvz = [
                r"New Models\CVzone Models\LSTM-CVzone-35-timesteps.keras",
                r"New Models\CVzone Models\LSTM-CNN-CVzone-35-timesteps.keras",
            ]
            self.model_names_main = ["MP LSTM", "MP LSTM-CNN"]
            self.model_names_cvz = ["CVZ LSTM", "CVZ LSTM-CNN"]
            
        elif self.model_config == "v2_70":
            self.n_timesteps_main = 70
            self.n_timesteps_cvz = 70
            self.model_paths_main = [
                r"New Models\Mediapipe Models\LSTM_Original_Model_70_Timesteps.keras",
                r"New Models\Mediapipe Models\LSTM_CNN_Model_70_Timesteps.keras",
            ]
            self.model_paths_cvz = [
                r"New Models\CVzone Models\LSTM-CVzone-70-timesteps-MasDropout.keras",
                r"New Models\CVzone Models\LSTM-CNN-CVzone-70-timesteps.keras",
            ]
            self.model_names_main = ["MP LSTM", "MP LSTM-CNN"]
            self.model_names_cvz = ["CVZ LSTM", "CVZ LSTM-CNN"]
            
        elif self.model_config == "v1":
            # Legacy config from Version 1
            self.n_timesteps_main = 70
            self.n_timesteps_cvz = 70
            self.model_paths_main = [
                r"Modelos\LSTM_CNN_Mediapipe.keras",
                r"Modelos\LSTM-ED-80.keras",
                r"Modelos\LSTM_CNN_ConRuido.keras",
            ]
            self.model_paths_cvz = [r"Modelos\LSTM_CNN_CVzone.keras"]
            self.model_names_main = ["LSTM+CNN", "LSTM ED-80", "LSTM CNN Ruido"]
            self.model_names_cvz = ["CVZ LSTM"]

    def load_models(self):
        # GPU Setup
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except: pass

        # Helper to fix paths if running from Scripts/ or root
        def _fix_path(p):
            if os.path.exists(p): return p
            # Try looking in parent directory
            parent = os.path.join("..", p)
            if os.path.exists(parent): return parent
            return p

        # Fix paths before loading
        self.model_paths_main = [_fix_path(p) for p in self.model_paths_main]
        self.model_paths_cvz = [_fix_path(p) for p in self.model_paths_cvz]
        self.yolo_path = _fix_path(self.yolo_path)

        def _load_for_inference(path):
            return tf.keras.models.load_model(
                path,
                compile=False,
                safe_mode=False,
            )

        with _suppress_startup_noise():
            with tf.keras.utils.custom_object_scope({
                "SliceLast": SliceLast,
                "TimeValidMask": TimeValidMask,
                "ExpandDim": ExpandDim,
            }):
                self.models_main = [_load_for_inference(p) for p in self.model_paths_main]
                self.models_cvz = [_load_for_inference(p) for p in self.model_paths_cvz]
        
        # Optimize models with dedicated tf.function to prevent retracing
        for model in self.models_main + self.models_cvz:
            # Use default argument to capture the current model instance correctly
            model.infer = tf.function(lambda x, m=model: m(x, training=False))

        with _suppress_startup_noise():
            self.yolo_model = YOLO(self.yolo_path)

    # @tf.function removed to prevent retracing issues with varying model arguments
    # We now use model.infer() attached during load_models
    # def _infer_step(self, model, x):
    #    return model(x, training=False)

    def process_frame(self, frame, frame_idx, effective_fps=15.0):
        # Feature Extraction
        lm_main, hands_info = extract_frame_landmarks_main(frame, self.mp_hands)
        self.sequence_main.append(lm_main)
        
        feats_cvz = extract_frame_features_cvzone(frame, self.detector)
        self.sequence_cvz.append(feats_cvz)
        
        # Outputs
        result = {
            "current_step": self.current_step_idx,
            "wash_complete": self.wash_complete,
            "model_used": "Iniciando...",
            "confidence": 0.0,
            "predicted_step": -1,
            "yolo_picos": self.yolo_peak_counter,
            "progress": 0.0,
            "status_color": (100, 100, 100), # Gray
            "debug_info": {},
            "missing_side": None # 'A', 'B' or None
        }

        # --- LSTM Inference ---
        cls_main, conf_main = [-1]*len(self.models_main), [0.0]*len(self.models_main)
        cls_cvz, conf_cvz = [-1]*len(self.models_cvz), [0.0]*len(self.models_cvz)
        
        # Main Models
        if len(self.sequence_main) == self.n_timesteps_main and (frame_idx % self.stride == 0):
            seq_np = np.expand_dims(np.array(self.sequence_main, dtype=np.float32), axis=0)
            for i, model in enumerate(self.models_main):
                logits = model.infer(seq_np)[0].numpy()
                best = int(np.argmax(logits))
                self.pred_buf_main[i].append(best)
                mc = Counter(self.pred_buf_main[i]).most_common(1)[0][0]
                cls_main[i] = mc
                conf_main[i] = float(logits[mc])
                
        # CVZ Models
        if len(self.sequence_cvz) == self.n_timesteps_cvz and (frame_idx % self.stride == 0):
            seq_np = np.expand_dims(np.array(self.sequence_cvz, dtype=np.float32), axis=0)
            for i, model in enumerate(self.models_cvz):
                logits = model.infer(seq_np)[0].numpy()
                best = int(np.argmax(logits))
                self.pred_buf_cvz[i].append(best)
                mc = Counter(self.pred_buf_cvz[i]).most_common(1)[0][0]
                cls_cvz[i] = mc
                conf_cvz[i] = float(logits[mc])

        # Global Consensus
        score_by_class = defaultdict(float)
        for c, cf in zip(cls_main, conf_main):
            if c >= 0 and cf > self.confidence_threshold: score_by_class[c] += cf
        for c, cf in zip(cls_cvz, conf_cvz):
            if c >= 0 and cf > self.confidence_threshold: score_by_class[c] += cf
            
        global_lstm_cls = -1
        global_lstm_conf = 0.0
        if score_by_class:
            global_lstm_cls = max(score_by_class, key=score_by_class.get)
            global_lstm_conf = score_by_class[global_lstm_cls] / (sum(score_by_class.values()) + 1e-8)

        # Mapping LSTM class to Step
        lstm_to_step = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5}
        aux_pred_step_idx = lstm_to_step.get(global_lstm_cls, -1) if global_lstm_cls >= 0 else -1

        # --- YOLO Arming Logic ---
        ARM_SEQ_LEN = 5
        ARM_COOLDOWN = int(effective_fps * 0.6)
        
        if self.p0_event_cooldown > 0: self.p0_event_cooldown -= 1
        if self.p1_event_cooldown > 0: self.p1_event_cooldown -= 1
        
        if aux_pred_step_idx == 0:
            self.aux_p0_frames += 1
            self.aux_p1_frames = max(0, self.aux_p1_frames - 1)
        elif aux_pred_step_idx == 1:
            self.aux_p1_frames += 1
            self.aux_p0_frames = max(0, self.aux_p0_frames - 1)
        else:
            self.aux_p0_frames = max(0, self.aux_p0_frames - 1)
            self.aux_p1_frames = max(0, self.aux_p1_frames - 1)
            
        if self.aux_p0_frames >= ARM_SEQ_LEN and self.p0_event_cooldown == 0:
            self.p0_event_cooldown = ARM_COOLDOWN
            self.aux_p0_frames = 0
            if not self.p0_done_once: self.p0_done_once = True
            elif self.p1_done_after_p0 and not self.yolo_armed: self.yolo_armed = True
            
        if self.aux_p1_frames >= ARM_SEQ_LEN and self.p1_event_cooldown == 0:
            self.p1_event_cooldown = ARM_COOLDOWN
            self.aux_p1_frames = 0
            if self.p0_done_once: self.p1_done_after_p0 = True

        # --- Selection & Logic ---
        pred_step_idx = -1
        confidence = 0.0
        model_used = ""
        
        # Helper for Bilateral Check
        def check_bilateral_side(step_idx, h_info):
            if not h_info['Left'] or not h_info['Right']:
                return None
            
            l_wrist = h_info['Left'].landmark[0]
            r_wrist = h_info['Right'].landmark[0]
            
            # Step 4 (Palm on Dorsum) & Step 6 (Knuckles)
            # Heuristic: Hand Crossing (X-axis)
            if step_idx == 1 or step_idx == 3: 
                # Normal: Left.x < Right.x (assuming mirrored image? or standard)
                # If Right.x < Left.x -> Right is crossed over to Left side (Side A)
                # If Left.x > Right.x -> Left is crossed over to Right side (Side B)
                # Note: MP coords: x increases left->right (0 to 1)
                
                # Check for significant crossing
                if r_wrist.x < l_wrist.x:
                    return 'A' # Right over Left
                else:
                    return 'B' # Left over Right (Normal or Crossed? Actually uncrossed is L < R)
                    # Wait, 'Palma con dorso' requires crossing?
                    # "Right palm on Left dorsum" -> Right hand moves Left.
                    # "Left palm on Right dorsum" -> Left hand moves Right.
                    # Normal: L(0.3) ... R(0.7)
                    # Right on Left: L(0.3), R(0.3) -> R < L + epsilon?
                    # Let's assume ANY significant overlap or swap is the trigger.
                    # Side A: Right hand is on left side (x < 0.5?) or relative?
                    
            # Step 7 (Thumbs)
            # Distance: Right Palm (9) to Left Thumb (4) vs Left Palm (9) to Right Thumb (4)
            if step_idx == 4:
                l_thumb = h_info['Left'].landmark[4]
                r_thumb = h_info['Right'].landmark[4]
                l_palm = h_info['Left'].landmark[9]
                r_palm = h_info['Right'].landmark[9]
                
                # Simple Euclidean distance (ignoring Z for robustness? or include?)
                # Just 2D for simplicity
                d_r_l = ((r_palm.x - l_thumb.x)**2 + (r_palm.y - l_thumb.y)**2)**0.5
                d_l_r = ((l_palm.x - r_thumb.x)**2 + (l_palm.y - r_thumb.y)**2)**0.5
                
                if d_r_l < d_l_r: return 'A' # Right cleaning Left Thumb
                else: return 'B' # Left cleaning Right Thumb
                
            # Step 8 (Tips)
            # Right Tips (12) to Left Palm (9)
            if step_idx == 5:
                l_palm = h_info['Left'].landmark[9]
                r_palm = h_info['Right'].landmark[9]
                l_tips = h_info['Left'].landmark[12] # Middle finger tip
                r_tips = h_info['Right'].landmark[12]
                
                d_r_l = ((r_tips.x - l_palm.x)**2 + (r_tips.y - l_palm.y)**2)**0.5
                d_l_r = ((l_tips.x - r_palm.x)**2 + (l_tips.y - r_palm.y)**2)**0.5
                
                if d_r_l < d_l_r: return 'A'
                else: return 'B'
                
            return None

        if not self.wash_complete:
            if self.current_step_idx == 2: # Paso 3: Intercaladas (YOLO)
                if not self.yolo_armed:
                    model_used = "YOLO (Esperando Activación)"
                    result["debug_info"]["yolo_status"] = "Waiting P0->P1->P0"
                else:
                    model_used = "YOLOv8"
                    # Run YOLO
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    proc = cv2.GaussianBlur(rgb, (5, 5), 0)
                    res = self.yolo_model(proc, verbose=False, conf=0.0, iou=0.0, device=self.yolo_device)
                    
                    best_conf = 0.0
                    for r in res:
                        for box in r.boxes:
                            if float(box.conf[0]) > best_conf: best_conf = float(box.conf[0])
                    
                    confidence = best_conf
                    if best_conf >= 0.70: # Min Conf
                        pred_step_idx = 2
                    
                    # Peak detection
                    if best_conf >= 0.80 and not self.yolo_in_peak:
                        self.yolo_peak_counter += 1
                        self.yolo_in_peak = True
                        self.yolo_peak_timer = 3
                    
                    if self.yolo_in_peak:
                        self.yolo_peak_timer -= 1
                        if self.yolo_peak_timer <= 0: self.yolo_in_peak = False
            else:
                model_used = "LSTM Consensus"
                pred_step_idx = aux_pred_step_idx
                confidence = global_lstm_conf

        result["model_used"] = model_used
        result["confidence"] = confidence
        result["predicted_step"] = pred_step_idx
        result["yolo_picos"] = self.yolo_peak_counter

        # --- State Machine Update ---
        SEQ_LEN_REQ = self.seq_len_req
        YOLO_PEAKS_REQ = self.yolo_peaks_req
        
        status_color = (0, 0, 255) # Red (BGR)
        
        if not self.wash_complete:
            if self.current_step_idx == 2:
                # Progress based on peaks
                prog = min(self.yolo_peak_counter / YOLO_PEAKS_REQ, 1.0) if self.yolo_armed else 0.0
                result["progress"] = prog
                
                if self.yolo_armed and self.yolo_peak_counter >= YOLO_PEAKS_REQ:
                    self.current_step_idx += 1
                    self.yolo_peak_counter = 0
                    self.yolo_in_peak = False
                    self.frames_validated = 0
                    status_color = (0, 255, 0) # Green
                elif self.yolo_peak_counter > 0:
                     status_color = (0, 255, 0)
            else:
                # Progress based on frames
                if pred_step_idx == self.current_step_idx:
                    self.frames_validated += 1
                    status_color = (0, 255, 0)
                    
                    # Bilateral Logic Update
                    # Only for Steps 4, 6, 7, 8 (Indices 1, 3, 4, 5)
                    if self.current_step_idx in [1, 3, 4, 5]:
                        side = check_bilateral_side(self.current_step_idx, hands_info)
                        if side:
                            if side == 'A': self.bilateral_state['side_a_frames'] += 1
                            elif side == 'B': self.bilateral_state['side_b_frames'] += 1
                            self.bilateral_state['current_side'] = side
                    
                else:
                    if self.frames_validated > 0: self.frames_validated -= 1
                
                # Progress calculation with Bilateral Constraint
                # If bilateral, we require BOTH sides to have some minimum frames (e.g. 30% of req each)
                # OR we just display the warning and let time handle it?
                # User wants "measure and evaluate".
                
                prog = min(self.frames_validated / SEQ_LEN_REQ, 1.0)
                result["progress"] = prog
                
                # Check for Bilateral Missing Side
                if self.current_step_idx in [1, 3, 4, 5]:
                    min_side_frames = int(SEQ_LEN_REQ * 0.3) # 30% of total req per side
                    
                    # If total progress is high but one side is low, flag it
                    if prog > 0.5:
                        if self.bilateral_state['side_a_frames'] < min_side_frames and self.bilateral_state['side_b_frames'] > min_side_frames:
                             result["missing_side"] = 'A'
                        elif self.bilateral_state['side_b_frames'] < min_side_frames and self.bilateral_state['side_a_frames'] > min_side_frames:
                             result["missing_side"] = 'B'
                        else:
                             result["missing_side"] = None
                    
                    # Prevent advancing if sides are not balanced?
                    # "Ensure... measure and evaluate".
                    # If we are at 100% progress but missing a side, we HOLD?
                    # Let's hold up to 150% frames, then force.
                    if self.frames_validated >= SEQ_LEN_REQ:
                        # Check if we are missing a side
                        missing = False
                        if (self.bilateral_state['side_a_frames'] < min_side_frames or \
                            self.bilateral_state['side_b_frames'] < min_side_frames):
                                missing = True
                        
                        # Logic to break the loop:
                        # If missing side, we increment overtime.
                        # If overtime < LIMIT, we HOLD (keep frames_validated just below threshold)
                        # If overtime >= LIMIT, we let it pass (force advance)
                        
                        OVERTIME_LIMIT = SEQ_LEN_REQ * 1.5 # Allow 1.5x more time to fix it
                        
                        if missing and self.bilateral_state['overtime_frames'] < OVERTIME_LIMIT:
                            # HOLD
                            self.frames_validated = SEQ_LEN_REQ - 1 # Keep at 99%
                            self.bilateral_state['overtime_frames'] += 1
                            self.status_color = (255, 165, 0) # Orange
                            
                            # Update result progress to reflect we are "waiting" (maybe > 100% visually?)
                            # Or just keep at 99%
                            result["progress"] = 0.99
                        else:
                            # Either not missing, or Overtime exceeded -> Advance
                            pass # Fall through to advancement logic below
                
                if self.frames_validated >= SEQ_LEN_REQ:
                    self.current_step_idx += 1
                    self.frames_validated = 0
                    self.status_color = (0, 255, 0)
                    # Reset Bilateral
                    self.bilateral_state = {
                        'active': False, 
                        'side_a_frames': 0, 
                        'side_b_frames': 0, 
                        'current_side': None, 
                        'last_switch_time': 0,
                        'overtime_frames': 0
                    }
            
            if self.current_step_idx > 5:
                self.wash_complete = True
                self.current_step_idx = 5
        
        result["current_step"] = self.current_step_idx
        result["wash_complete"] = self.wash_complete
        result["status_color"] = status_color
        
        return result
