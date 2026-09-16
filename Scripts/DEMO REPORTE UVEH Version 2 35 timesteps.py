import os
import cv2
import numpy as np
from collections import deque, Counter, defaultdict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer

import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

from ultralytics import YOLO
import torch
import pandas as pd


# ============================================================
# 0. MODO DE EJECUCIÓN: DEMO vs ANÁLISIS
# ============================================================

ANALYSIS_MODE   = False   # True = batch UVEH (sin UI)
SAVE_METADATA   = True
METADATA_CSV    = "metadata_inferencia_multi_personas.csv"

PERSON_ID = "Persona_01"


# ============================================================
# 1. CAPAS PERSONALIZADAS COMPARTIDAS
# ============================================================

class SliceLast(Layer):
    """Devuelve x[..., start:end]"""
    def __init__(self, start: int, end: int, **kwargs):
        super().__init__(**kwargs)
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
    """True si el timestep NO es todo cero en coords."""
    def call(self, x):
        return tf.math.logical_not(
            tf.reduce_all(tf.equal(x, 0.0), axis=-1)
        )

    def compute_output_shape(self, input_shape):
        b, t, _ = input_shape
        return (b, t)

    def get_config(self):
        return super().get_config()


class ExpandDim(Layer):
    """tf.expand_dims(x, axis) como capa serializable."""
    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
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
# 2. CONFIGURACIÓN GENERAL
# ============================================================

# ----------- 2 MAIN (MediaPipe) -----------
model_paths_main = [
    r"New Models\Mediapipe Models\LSTM_Original_Model_35_Timesteps.keras",
    r"New Models\Mediapipe Models\LSTM_CNN_Model_35_Timesteps.keras",
]
model_names_main = [
    "MAIN MP Arch-LSTM",
    "MAIN MP Arch-LSTM-CNN",
]

# -----------------------------
# Modelos CVZONE -> 2 modelos
# -----------------------------
model_paths_cvz = [
    r"New Models\CVzone Models\LSTM-CVzone-35-timesteps.keras",
    r"New Models\CVzone Models\LSTM-CNN-CVzone-35-timesteps.keras",
]
model_names_cvz = [
    "CVZ Arch-LSTM",
    "CVZ Arch-LSTM-CNN",
]

# -----------------------------
# YOLO (Paso 3)
# -----------------------------
YOLO_MODEL_PATH = r"Modelos\YOLOV-ED-74.pt"

# Fuente default (demo)
VIDEO_SOURCE = r"Videos\vlc-record-2025-11-26-13h38m38s-rtsp___169.254.11.181_554_live1s1.sdp- copy.mp4"
# VIDEO_SOURCE = 0

# Ventanas temporales
n_timesteps_main = 35
n_timesteps_cvz  = 35

# ----------- Hiperparámetros LSTM -----------
CONFIDENCE_THRESHOLD_LSTM = 0.4
STRIDE = 1
SEQ_LEN_REQUIRED = 30

# ----------- Optimizaciones de procesamiento -----------
FRAME_SKIP = 2
YOLO_INFERENCE_EVERY_N = 2

# ----------- Map pasos OMS -----------
label_map_steps = {
    0: 'Paso 1: Palma con Palma',
    1: 'Paso 2: Palma con Dorso',
    2: 'Paso 3: Palmas Intercaladas',
    3: 'Paso 4: Dorso con Dorso',
    4: 'Paso 5: Barrido con Pulgar',
    5: 'Paso 6: Puntas con Palma'
}

# ----------- Map clases LSTM -> pasos OMS -----------
lstm_to_step = {
    0: 0,
    1: 1,
    2: 3,
    3: 4,
    4: 5
}

label_map_lstm = {
    0: "palma con palma",
    1: "palma con dorso",
    2: "dorsos con dorso",
    3: "barrido con pulgar",
    4: "puntas"
}

# ----------- Parámetros YOLO (Paso 3) -----------
YOLO_CONF_MIN        = 0.70
YOLO_PEAK_THR        = 0.80
YOLO_PEAK_COOLDOWN   = 3
YOLO_PEAKS_REQUIRED  = 5


# ============================================================
# 3. MEDIAPIPE (MAIN) EXTRACCIÓN LANDMARKS
# ============================================================

mp_hands = mp.solutions.hands
hands_mp = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_hand_landmarks_mp(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

def extract_frame_landmarks_main(frame):
    """
    Vector 128:
    - Mano1 63
    - Mano2 63
    - Flags 2
    """
    results = hands_mp.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    frame_data = []
    presence_flags = [0, 0]

    if results.multi_hand_landmarks:
        hands_landmarks = results.multi_hand_landmarks
        num_hands = len(hands_landmarks)

        frame_data.extend(extract_hand_landmarks_mp(hands_landmarks[0]))
        presence_flags[0] = 1

        if num_hands > 1:
            frame_data.extend(extract_hand_landmarks_mp(hands_landmarks[1]))
            presence_flags[1] = 1
        else:
            frame_data.extend([0.0] * 63)
    else:
        frame_data.extend([0.0] * 63)
        frame_data.extend([0.0] * 63)

    frame_data.extend(presence_flags)
    return frame_data


# ============================================================
# 4. CVZONE EXTRACCIÓN FEATURES (161)
# ============================================================

detector = HandDetector(maxHands=2, detectionCon=0.6, minTrackCon=0.5)

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

    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)
    inter_x_max = min(x1_max, x2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area + 1e-6
    return float(inter_area / union_area)

def extract_frame_features_cvzone(frame):
    """
    161 floats:
    - 63 mano1 coords norm
    - 63 mano2 coords norm
    - 2 flags
    - 15 feats mano1
    - 15 feats mano2
    - 3 inter-mano
    """
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
            x_bn = x_bbox / img_w
            y_bn = y_bbox / img_h
            w_bn = w_bbox / img_w
            h_bn = h_bbox / img_h

            cx_n = cx / img_w
            cy_n = cy / img_h

            type_onehot = [1.0, 0.0] if hand_type == "Left" else [0.0, 1.0]

            fingers = detector.fingersUp(hand)

            wrist = lm_list[0]
            mcp_mid = lm_list[9]
            idx_tip = lm_list[8]
            pinky_tip = lm_list[20]

            palm_size = distance_norm(wrist, mcp_mid, img_w, img_h)
            finger_spread = distance_norm(idx_tip, pinky_tip, img_w, img_h)

            feats_hand = [
                x_bn, y_bn, w_bn, h_bn,
                cx_n, cy_n,
                *type_onehot,
                *[float(f) for f in fingers],
                float(palm_size),
                float(finger_spread)
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
        dist_centers = 0.0
        dist_wrists = 0.0
        iou = 0.0

    extra_feats.extend([float(dist_centers), float(dist_wrists), float(iou)])
    return frame_coords + presence_flags + extra_feats


# ============================================================
# 5. CARGA DE MODELOS
# ============================================================

# TF GPU memory growth
tf_gpus = tf.config.list_physical_devices("GPU")
for gpu in tf_gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# YOLO device
yolo_device = "cuda" if torch.cuda.is_available() else "cpu"

with keras.utils.custom_object_scope({
    "SliceLast": SliceLast,
    "TimeValidMask": TimeValidMask,
    "ExpandDim": ExpandDim,
}):
    models_main = [keras.models.load_model(p) for p in model_paths_main]
    models_cvz  = [keras.models.load_model(p) for p in model_paths_cvz]

print("[OK] MAIN cargados:", len(models_main))
print("[OK] CVZ cargados:", len(models_cvz))

print("[OK] Cargando YOLO...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("[OK] YOLO cargado:", YOLO_MODEL_PATH)

@tf.function
def infer_step(model, x):
    return model(x, training=False)


# ============================================================
# 6. PROCESAR VIDEO (DEMO o ANALISIS)
# ============================================================

def procesar_video(video_source, person_id):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ADVERTENCIA] No se pudo abrir: {video_source}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    effective_fps = max(1.0, fps / max(1, FRAME_SKIP))

    video_name = os.path.basename(str(video_source))

    # ------------------ estado máquina de estados (para los 6 pasos) ------------------
    current_step_idx = 0
    frames_validated = 0
    wash_complete = False

    # secuencias
    sequence_main = deque(maxlen=n_timesteps_main)
    sequence_cvz  = deque(maxlen=n_timesteps_cvz)

    # buffers por modelo
    pred_buf_main = [deque(maxlen=5) for _ in models_main]
    cls_main = [-1] * len(models_main)
    conf_main = [0.0] * len(models_main)
    lbl_main = ["Iniciando..."] * len(models_main)

    pred_buf_cvz = [deque(maxlen=5) for _ in models_cvz]
    cls_cvz = [-1] * len(models_cvz)
    conf_cvz = [0.0] * len(models_cvz)
    lbl_cvz = ["Iniciando..."] * len(models_cvz)

    # consenso global
    global_lstm_cls = -1
    global_lstm_label = "Incierto..."
    global_lstm_conf = 0.0

    # ------------------ estado YOLO picos ------------------
    yolo_peak_counter = 0
    yolo_in_peak = False
    yolo_peak_timer = 0

    # ------------------ ARMADO YOLO (P0 -> P1 -> P0) ------------------
    yolo_armed = False
    p0_done_once = False
    p1_done_after_p0 = False

    ARM_SEQ_LEN_REQUIRED = 5   # puedes ajustar aparte si quieres
    aux_p0_frames = 0
    aux_p1_frames = 0

    ARM_EVENT_COOLDOWN_FRAMES = int(effective_fps * 0.6)  # ~0.6s
    p0_event_cooldown = 0
    p1_event_cooldown = 0

    # metadata
    metadata_records = []

    frame_idx = 0
    processed_frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP != 0):
            continue

        processed_frame_idx += 1
        timestamp_s = processed_frame_idx / effective_fps

        frame = cv2.resize(frame, (640, 480))
        img_display = frame.copy()

        # features
        lm_main = extract_frame_landmarks_main(frame)
        sequence_main.append(lm_main)

        feats_cvz = extract_frame_features_cvzone(frame)
        sequence_cvz.append(feats_cvz)

        # ------------------ infer MAIN (2) ------------------
        if len(sequence_main) == n_timesteps_main and (frame_idx % STRIDE == 0):
            seq_np = np.array(sequence_main, dtype=np.float32)
            x = np.expand_dims(seq_np, axis=0)

            for i, model in enumerate(models_main):
                logits = infer_step(model, x)[0].numpy()
                best = int(np.argmax(logits))
                pred_buf_main[i].append(best)

                mc = Counter(pred_buf_main[i]).most_common(1)[0][0]
                conf = float(logits[mc])

                cls_main[i] = mc
                conf_main[i] = conf
                lbl_main[i] = label_map_lstm.get(mc, f"clase_{mc}") if conf > CONFIDENCE_THRESHOLD_LSTM else "Incierto..."

        # ------------------ infer CVZ (2) ------------------
        if len(sequence_cvz) == n_timesteps_cvz and (frame_idx % STRIDE == 0):
            seq_cvz_np = np.array(sequence_cvz, dtype=np.float32)
            xcvz = np.expand_dims(seq_cvz_np, axis=0)

            for i, model in enumerate(models_cvz):
                logits = infer_step(model, xcvz)[0].numpy()
                best = int(np.argmax(logits))
                pred_buf_cvz[i].append(best)

                mc = Counter(pred_buf_cvz[i]).most_common(1)[0][0]
                conf = float(logits[mc])

                cls_cvz[i] = mc
                conf_cvz[i] = conf
                lbl_cvz[i] = label_map_lstm.get(mc, f"clase_{mc}") if conf > CONFIDENCE_THRESHOLD_LSTM else "Incierto..."

        # ------------------ CONSENSO GLOBAL (4 modelos) ------------------
        score_by_class = defaultdict(float)

        for c, cf in zip(cls_main, conf_main):
            if c >= 0 and cf > CONFIDENCE_THRESHOLD_LSTM:
                score_by_class[c] += cf

        for c, cf in zip(cls_cvz, conf_cvz):
            if c >= 0 and cf > CONFIDENCE_THRESHOLD_LSTM:
                score_by_class[c] += cf

        if score_by_class:
            global_lstm_cls = max(score_by_class, key=score_by_class.get)
            total = sum(score_by_class.values())
            global_lstm_conf = score_by_class[global_lstm_cls] / (total + 1e-8)
            global_lstm_label = label_map_lstm.get(global_lstm_cls, f"clase_{global_lstm_cls}")
        else:
            global_lstm_cls = -1
            global_lstm_conf = 0.0
            global_lstm_label = "Incierto..."

        # ------------------ AUX PRED (para armado YOLO) ------------------
        aux_pred_step_idx = -1
        if global_lstm_cls >= 0 and global_lstm_conf > CONFIDENCE_THRESHOLD_LSTM:
            aux_pred_step_idx = lstm_to_step.get(global_lstm_cls, -1)

        # -------------------------------
        # ARMADO YOLO: eventos P0 -> P1 -> P0 (sin bug de repetir paso 0)
        # -------------------------------
        if p0_event_cooldown > 0:
            p0_event_cooldown -= 1
        if p1_event_cooldown > 0:
            p1_event_cooldown -= 1

        if aux_pred_step_idx == 0:
            aux_p0_frames += 1
            aux_p1_frames = max(0, aux_p1_frames - 1)
        elif aux_pred_step_idx == 1:
            aux_p1_frames += 1
            aux_p0_frames = max(0, aux_p0_frames - 1)
        else:
            aux_p0_frames = max(0, aux_p0_frames - 1)
            aux_p1_frames = max(0, aux_p1_frames - 1)

        p0_event = (aux_p0_frames >= ARM_SEQ_LEN_REQUIRED) and (p0_event_cooldown == 0)
        if p0_event:
            p0_event_cooldown = ARM_EVENT_COOLDOWN_FRAMES
            aux_p0_frames = 0

            if not p0_done_once:
                p0_done_once = True
            else:
                if p1_done_after_p0 and not yolo_armed:
                    yolo_armed = True
                    print("[ARM] YOLO armado: secuencia P0 -> P1 -> P0 completada.")

        p1_event = (aux_p1_frames >= ARM_SEQ_LEN_REQUIRED) and (p1_event_cooldown == 0)
        if p1_event:
            p1_event_cooldown = ARM_EVENT_COOLDOWN_FRAMES
            aux_p1_frames = 0

            if p0_done_once and not p1_done_after_p0:
                p1_done_after_p0 = True

        # ------------------ SELECCIÓN MOTOR (por paso requerido real) ------------------
        pred_step_idx = -1
        confidence = 0.0
        model_used = "Esperando..."

        yolo_last_conf_for_log = 0.0
        yolo_detected_this_frame = False

        if not wash_complete:
            # Paso 3
            if current_step_idx == 2:
                # bloquea hasta armar YOLO
                if not yolo_armed:
                    model_used = "YOLO (bloqueado esperando P0->P1->P0)"
                    pred_step_idx = -1
                    confidence = 0.0
                else:
                    model_used = "YOLOv8"

                    run_yolo = (YOLO_INFERENCE_EVERY_N <= 1) or (processed_frame_idx % YOLO_INFERENCE_EVERY_N == 0)

                    found_target = False
                    best_conf = 0.0

                    if run_yolo:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        proc = cv2.GaussianBlur(rgb, (5, 5), 0)

                        results_y = yolo_model(
                            proc,
                            verbose=False,
                            conf=0.0,
                            iou=0.0,
                            device=yolo_device
                        )

                        for r in results_y:
                            for box in r.boxes:
                                conf_box = float(box.conf[0])
                                if conf_box > best_conf:
                                    best_conf = conf_box
                                    found_target = True
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    if not ANALYSIS_MODE:
                                        cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 0, 255), 2)

                    yolo_last_conf_for_log = best_conf

                    if found_target and best_conf >= YOLO_CONF_MIN:
                        pred_step_idx = 2
                        confidence = best_conf
                        yolo_detected_this_frame = True
                    else:
                        pred_step_idx = -1
                        confidence = 0.0

                    # pico
                    if found_target and best_conf >= YOLO_PEAK_THR and not yolo_in_peak:
                        yolo_peak_counter += 1
                        yolo_in_peak = True
                        yolo_peak_timer = YOLO_PEAK_COOLDOWN
                        print(f"[YOLO] Pico! Conf={best_conf:.2f} | Picos={yolo_peak_counter}/{YOLO_PEAKS_REQUIRED}")

                    if yolo_in_peak:
                        yolo_peak_timer -= 1
                        if yolo_peak_timer <= 0:
                            yolo_in_peak = False

            else:
                # otros pasos: consenso LSTM
                model_used = "Consenso (2 MAIN + 2 CVZ)"
                if global_lstm_cls >= 0:
                    pred_step_idx = lstm_to_step.get(global_lstm_cls, -1)
                    confidence = global_lstm_conf
                else:
                    pred_step_idx = -1
                    confidence = 0.0

        # ------------------ MÁQUINA DE ESTADOS (progreso real) ------------------
        if not wash_complete:
            if current_step_idx == 2:
                if yolo_armed and (yolo_peak_counter >= YOLO_PEAKS_REQUIRED):
                    print(f"[OK] PASO 3 completado por YOLO (picos).")

                    yolo_peak_counter = 0
                    yolo_in_peak = False
                    yolo_peak_timer = 0

                    frames_validated = 0
                    current_step_idx += 1

                    if current_step_idx > 5:
                        wash_complete = True
                        current_step_idx = 5

                status_color = (0, 255, 0) if yolo_peak_counter > 0 else (0, 0, 255)

            else:
                if pred_step_idx == current_step_idx:
                    frames_validated += 1
                    status_color = (0, 255, 0)
                else:
                    if frames_validated > 0:
                        frames_validated -= 1
                    status_color = (0, 0, 255)

                if frames_validated >= SEQ_LEN_REQUIRED:
                    print(f"[OK] PASO {current_step_idx} completado: {label_map_steps.get(current_step_idx)}")
                    frames_validated = 0
                    current_step_idx += 1

                    if current_step_idx > 5:
                        wash_complete = True
                        current_step_idx = 5

        # ------------------ METADATA (UVEH) ------------------
        if ANALYSIS_MODE:
            if current_step_idx == 2:
                mode_str = "yolo"
                label_str = "intercalado" if yolo_detected_this_frame else "sin_deteccion"
                raw_label = label_str
            else:
                mode_str = "lstm"
                label_str = global_lstm_label
                raw_label = global_lstm_label

            record = {
                "person_id": person_id,
                "video_name": video_name,
                "frame": processed_frame_idx,
                "timestamp_s": timestamp_s,

                "step_required_idx": current_step_idx,
                "step_required": label_map_steps.get(current_step_idx, "NA"),

                "mode": mode_str,
                "label": label_str,
                "raw_label": raw_label,

                "pred_step_idx": pred_step_idx,
                "pred_step_label": label_map_steps.get(pred_step_idx, "NA") if pred_step_idx >= 0 else "NA",

                "global_lstm_cls": global_lstm_cls,
                "global_lstm_label": global_lstm_label,
                "global_lstm_conf": global_lstm_conf,

                "yolo_armed": yolo_armed,
                "p0_done_once": p0_done_once,
                "p1_done_after_p0": p1_done_after_p0,
                "aux_pred_step_idx": aux_pred_step_idx,
                "aux_p0_frames": aux_p0_frames,
                "aux_p1_frames": aux_p1_frames,

                "yolo_conf": yolo_last_conf_for_log,
                "yolo_peak_counter": yolo_peak_counter,

                "frames_validated": frames_validated,
                "wash_complete": wash_complete,
            }
            metadata_records.append(record)

        # ------------------ UI (solo demo) ------------------
        if not ANALYSIS_MODE:
            cv2.rectangle(img_display, (0, 0), (640, 150), (30, 30, 30), -1)

            if wash_complete:
                cv2.putText(img_display, "LAVADO COMPLETADO", (80, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                req_text = f"REQUERIDO: {label_map_steps.get(current_step_idx, '---')}"
                cv2.putText(img_display, req_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if current_step_idx == 2:
                    pred_text = f"Motor: {model_used} | Armed={yolo_armed} | Conf={confidence:.2f} | Picos={yolo_peak_counter}/{YOLO_PEAKS_REQUIRED}"
                else:
                    pred_text = f"Motor: {model_used} | LSTM={global_lstm_cls}({global_lstm_label}) | Conf={global_lstm_conf:.2f}"

                cv2.putText(img_display, pred_text, (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # barra
                bar_x, bar_y, bar_w, bar_h = 10, 75, 620, 20
                cv2.rectangle(img_display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

                if current_step_idx == 2:
                    prog_ratio = (yolo_peak_counter / YOLO_PEAKS_REQUIRED) if yolo_armed else 0.0
                    prog_ratio = min(max(prog_ratio, 0.0), 1.0)
                else:
                    prog_ratio = min(frames_validated / SEQ_LEN_REQUIRED, 1.0)

                fill_w = int(bar_w * prog_ratio)
                cv2.rectangle(img_display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), status_color, -1)

                y0, dy = 105, 18

                # MAIN
                for i, (nm, lb, cf) in enumerate(zip(model_names_main, lbl_main, conf_main)):
                    cv2.putText(img_display, f"{nm}: {lb} ({cf:.2f})", (10, y0 + i*dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

                # CVZ
                offset = len(model_names_main)
                for j, (nm, lb, cf) in enumerate(zip(model_names_cvz, lbl_cvz, conf_cvz)):
                    cv2.putText(img_display, f"{nm}: {lb} ({cf:.2f})", (10, y0 + (offset+j)*dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            cv2.imshow("Sistema Hibrido (2 MAIN + 2 CVZ + YOLO)", img_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # sin UI
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return metadata_records


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    # === CONFIG ===
    ANALYSIS_MODE = False      # True para UVEH
    SAVE_METADATA = True
    METADATA_CSV  = "metadata_inferencia_multi_personas.csv"
    
    VIDEOS = [
        ("Persona_01", r"Videos\vlc-record-2025-11-28-13h05m04s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"),
        ("Persona_02", r"Videos\vlc-record-2025-11-26-13h38m38s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"),
        ("Persona_03", r"Videos\vlc-record-2025-12-05-12h03m08s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"),
        ("Persona_04", r"Videos\vlc-record-2025-12-05-12h24m18s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"),
        ("Persona_05", r"Videos\vlc-record-2025-12-05-13h16m20s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"),
        ("Persona_06", r"Videos\vlc-record-2025-12-05-13h35m25s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4")
    ]
    

    all_metadata = []

    if VIDEOS:
        for pid, vpath in VIDEOS:
            print(f"\nProcesando {pid}: {vpath}")
            regs = procesar_video(vpath, pid)
            all_metadata.extend(regs)
    else:
        regs = procesar_video(VIDEO_SOURCE, PERSON_ID)
        all_metadata.extend(regs)

    # ========================================================
    # 8. GUARDAR + ANÁLISIS
    # ========================================================
    if ANALYSIS_MODE and SAVE_METADATA and all_metadata:
        df = pd.DataFrame(all_metadata)
        df.to_csv(METADATA_CSV, index=False, encoding="utf-8-sig")
        print(f"\nMetadata guardada en: {METADATA_CSV}")

        pasos_validos = [
            "barrido con pulgar",
            "dorsos con dorso",
            "intercalado",
            "palma con palma",
            "palma con dorso",
            "puntas",
        ]

        conteo_pasos_global = (
            df["label"]
            .value_counts()
            .reindex(pasos_validos + ["sin_deteccion"], fill_value=0)
        )

        print("\n=== Conteo global por paso (incluye sin_deteccion) ===")
        print(conteo_pasos_global)

        cobertura_global = {p: (conteo_pasos_global[p] > 0) for p in pasos_validos}
        print("\n¿Se detectó cada uno de los 6 pasos (GLOBAL)?")
        for paso, presente in cobertura_global.items():
            print(f"  {paso}: {'Sí' if presente else 'No'}")

        df["deteccion_6_pasos"] = df["label"].isin(pasos_validos)

        print("\n\n========================")
        print("ANÁLISIS POR PERSONA")
        print("========================")

        for pid in sorted(df["person_id"].unique()):
            print(f"\n--- Persona: {pid} ---")
            df_p = df[df["person_id"] == pid].copy()

            conteo_pasos_p = (
                df_p["label"]
                .value_counts()
                .reindex(pasos_validos + ["sin_deteccion"], fill_value=0)
            )

            print("Conteo por paso (incluye sin_deteccion):")
            print(conteo_pasos_p)

            cobertura_p = {p: (conteo_pasos_p[p] > 0) for p in pasos_validos}
            print("\n¿Se detectó cada uno de los 6 pasos para esta persona?")
            for paso, presente in cobertura_p.items():
                print(f"  {paso}: {'Sí' if presente else 'No'}")

            if all(cobertura_p.values()):
                print("Cobertura completa: los 6 pasos fueron detectados al menos una vez.")
            else:
                faltantes = [p for p, ok in cobertura_p.items() if not ok]
                print(f"Faltan detecciones para: {faltantes}")
