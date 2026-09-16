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

ANALYSIS_MODE   = False   # Se vuelve True en el main para correr batch
SAVE_METADATA   = True    # Guardar CSV al final si ANALYSIS_MODE es True
METADATA_CSV    = "metadata_inferencia_multi_personas.csv"

# ID genérico; se sobreescribe si usas VIDEOS
PERSON_ID = "Persona_01"     # ID o nombre de la persona evaluada


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

# Rutas de modelos LSTM 
model_paths_main = [
    r"Modelos\LSTM_CNN_Mediapipe.keras",
    r"Modelos\LSTM-ED-80.keras",
    r"Modelos\LSTM_CNN_ConRuido.keras",
]

model_names_main = [
    "LSTM + CNN",
    "LSTM ED-80",
    "LSTM CNN con ruido",
]

# Modelo cvzone (4º LSTM)
MODEL_PATH_CVZ = r"Modelos\LSTM_CNN_CVzone.keras"
model_name_cvz = "Modelo cvzone"

# Modelo YOLO (Paso 3)
YOLO_MODEL_PATH = r"Modelos\YOLOV-ED-74.pt"

# Fuente por defecto si NO usas VIDEOS (modo demo)
VIDEO_SOURCE = r"Videos\vlc-record-2025-11-26-13h38m38s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"
# VIDEO_SOURCE = 0  # Cámara web

# Ventanas temporales
n_timesteps_main = 70   # para los 3 modelos main
n_timesteps_cvz  = 70   # para modelo cvzone

# Features de modelos main: 126 coords + 2 flags = 128
N_FEATURES_MAIN = 128

# Hiperparámetros LSTM
CONFIDENCE_THRESHOLD_LSTM = 0.4
STRIDE = 1              # cada cuántos frames hacer inferencia
SEQ_LEN_REQUIRED = 30   # frames válidos para pasos 1,2,4,5,6

# Optimizaciones de procesamiento
FRAME_SKIP = 2                # procesa 1 de cada 2 frames
YOLO_INFERENCE_EVERY_N = 2     # corre YOLO cada 2 frames procesados

# Map paso global OMS (6 pasos)
label_map_steps = {
    0: 'Paso 1: Palma con Palma',
    1: 'Paso 2: Palma con Dorso',      
    2: 'Paso 3: Palmas Intercaladas',
    3: 'Paso 4: Dorso con Dorso',
    4: 'Paso 5: Barrido con Pulgar',
    5: 'Paso 6: Puntas con Palma'
}

# Map de clases LSTM (0–4) a pasos OMS (0–5)
lstm_to_step = {
    0: 0,  # LSTM clase 0 -> Paso 1
    1: 1,  # LSTM clase 1 -> Paso 2
    2: 3,  # LSTM clase 2 -> Paso 4
    3: 4,  # LSTM clase 3 -> Paso 5
    4: 5   # LSTM clase 4 -> Paso 6
}
step_to_lstm = {v: k for k, v in lstm_to_step.items()}

# label_map textual de LSTM (5 clases)
label_map_lstm = {
    0: "palma con palma",
    1: "palma con dorso",     
    2: "dorsos con dorso",
    3: "barrido con pulgar",
    4: "puntas"
}

# -----------------------------
# Parámetros YOLO (Paso 3)
# -----------------------------
YOLO_CONF_MIN        = 0.70   # mínimo para decir "hay detección"
YOLO_PEAK_THR        = 0.80   # umbral de "pico" fuerte
YOLO_PEAK_COOLDOWN   = 3      # frames de cooldown entre picos
YOLO_PEAKS_REQUIRED  = 5     # número de picos necesarios para completar el paso 3


# ============================================================
# 3. MEDIAPIPE Y EXTRACCIÓN DE LANDMARKS (MODELOS MAIN)
# ============================================================

mp_hands = mp.solutions.hands
hands_mp = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_hand_landmarks_mp(hand_landmarks):
    return [coord
            for lm in hand_landmarks.landmark
            for coord in (lm.x, lm.y, lm.z)]

def extract_frame_landmarks_main(frame):
    """
    Devuelve vector 1D:
    - Mano 1: 63
    - Mano 2: 63
    - Flags de presencia: 2
    Total = 128
    """
    results = hands_mp.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    frame_data = []
    presence_flags = [0, 0]

    if results.multi_hand_landmarks:
        hands_landmarks = results.multi_hand_landmarks
        num_hands = len(hands_landmarks)

        # Mano 1
        frame_data.extend(extract_hand_landmarks_mp(hands_landmarks[0]))
        presence_flags[0] = 1

        # Mano 2 (si existe)
        if num_hands > 1:
            frame_data.extend(extract_hand_landmarks_mp(hands_landmarks[1]))
            presence_flags[1] = 1
        else:
            frame_data.extend([0.0] * 63)
    else:
        frame_data.extend([0.0] * 63)
        frame_data.extend([0.0] * 63)

    frame_data.extend(presence_flags)
    return frame_data  # len = 128


# ============================================================
# 4. CVZONE: DETECTOR Y FEATURES PARA MODELO 4
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
    iou = inter_area / union_area
    return float(iou)

def extract_frame_features_cvzone(frame):
    """
    Devuelve un vector 1D con:
    - 63 coords mano 1 (normalizadas)
    - 63 coords mano 2 (normalizadas)
    - 2 flags de presencia
    - 15 features mano 1
    - 15 features mano 2
    - 3 features inter-mano
    Total: 126 + 2 + 33 = 161 floats
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

            if hand_type == "Left":
                type_onehot = [1.0, 0.0]
            else:
                type_onehot = [0.0, 1.0]

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
# 5. CARGA DE MODELOS (4 LSTM + YOLO)
# ============================================================

tf_gpus = tf.config.list_physical_devices("GPU")
for gpu in tf_gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# Configuracion dispositivo YOLO
yolo_device = "cuda" if torch.cuda.is_available() else "cpu"

models_main = []

with keras.utils.custom_object_scope({
    "SliceLast": SliceLast,
    "TimeValidMask": TimeValidMask,
    "ExpandDim": ExpandDim,
}):
    for path in model_paths_main:
        m = keras.models.load_model(path)
        models_main.append(m)

    model_cvz = keras.models.load_model(MODEL_PATH_CVZ)

print("Modelos main cargados:", len(models_main))
print("Modelo cvzone cargado:", MODEL_PATH_CVZ)

print("Cargando YOLO...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("Modelo YOLO cargado:", YOLO_MODEL_PATH)

print("GPUs disponibles (TF):", tf.config.list_physical_devices('GPU'))


@tf.function
def infer_step(model, x):
    return model(x, training=False)


# ============================================================
# 6. FUNCIÓN PARA PROCESAR UN VIDEO (UI + / - METADATA)
# ============================================================

def procesar_video(video_source, person_id):
    """
    Procesa un solo video/fuente y devuelve metadata_records (si ANALYSIS_MODE).
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ADVERTENCIA] No se pudo abrir la fuente de video: {video_source}")
        return []

    print("Fuente de video:", video_source)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # fps efectivo cuando hay frame skipping
    effective_fps = max(1.0, fps / max(1, FRAME_SKIP))

    video_name = os.path.basename(str(video_source))

    # ---------- Estado global del lavado (por video) ----------
    current_step_idx = 0       # 0–5
    frames_validated = 0
    wash_complete = False

    # Secuencias
    sequence_main = deque(maxlen=n_timesteps_main)
    sequence_cvz  = deque(maxlen=n_timesteps_cvz)

    # Buffers LSTM individuales
    predictions_buffers_main = [deque(maxlen=5) for _ in models_main]
    current_labels_main = ["Iniciando..." for _ in models_main]
    current_confs_main  = [0.0 for _ in models_main]
    current_cls_main    = [-1   for _ in models_main]

    # Buffer modelo cvzone
    predictions_buffer_cvz = deque(maxlen=5)
    current_label_cvz = "Iniciando..."
    current_conf_cvz  = 0.0
    current_cls_cvz   = -1

    # Consenso global (sobre espacio LSTM 0–4)
    global_lstm_cls   = -1
    global_lstm_label = "Incierto..."
    global_lstm_conf  = 0.0

    frame_idx = 0
    processed_frame_idx = 0

    # Estado específico para YOLO (Paso 3)
    yolo_peak_counter = 0
    yolo_in_peak      = False
    yolo_peak_timer   = 0

    # Para análisis UVEH
    metadata_records = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # saltar frames para acelerar
        if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP != 0):
            continue

        processed_frame_idx += 1
        timestamp_s = processed_frame_idx / effective_fps

        frame = cv2.resize(frame, (640, 480))
        img_display = frame.copy()

        # --------------------------------------------------------
        # 6.1 FEATURES PARA MODELOS MAIN (mediapipe)
        # --------------------------------------------------------
        lm_main = extract_frame_landmarks_main(frame)
        sequence_main.append(lm_main)

        # --------------------------------------------------------
        # 6.2 FEATURES PARA MODELO CVZONE
        # --------------------------------------------------------
        feats_cvz = extract_frame_features_cvzone(frame)
        sequence_cvz.append(feats_cvz)

        # --------------------------------------------------------
        # 6.3 INFERENCIA MODELOS MAIN (3 modelos LSTM)
        # --------------------------------------------------------
        if len(sequence_main) == n_timesteps_main and (frame_idx % STRIDE == 0):
            seq_np = np.array(sequence_main, dtype=np.float32)
            input_seq = np.expand_dims(seq_np, axis=0)

            for i, model in enumerate(models_main):
                logits = infer_step(model, input_seq)
                res = logits[0].numpy()

                best_class = int(np.argmax(res))
                predictions_buffers_main[i].append(best_class)

                # voto más frecuente en el buffer
                most_common_class = Counter(predictions_buffers_main[i]).most_common(1)[0][0]
                confidence_i = float(res[most_common_class])

                current_cls_main[i] = most_common_class
                label_i = label_map_lstm.get(most_common_class, f"clase_{most_common_class}")

                if confidence_i > CONFIDENCE_THRESHOLD_LSTM:
                    current_labels_main[i] = label_i
                    current_confs_main[i]  = confidence_i
                else:
                    current_labels_main[i] = "Incierto..."
                    current_confs_main[i]  = confidence_i

        # --------------------------------------------------------
        # 6.4 INFERENCIA MODELO CVZONE (4º LSTM)
        # --------------------------------------------------------
        if len(sequence_cvz) == n_timesteps_cvz and (frame_idx % STRIDE == 0):
            seq_cvz_np = np.array(sequence_cvz, dtype=np.float32)
            input_seq_cvz = np.expand_dims(seq_cvz_np, axis=0)

            logits_cvz = infer_step(model_cvz, input_seq_cvz)
            res_cvz = logits_cvz[0].numpy()

            best_class_cvz = int(np.argmax(res_cvz))
            predictions_buffer_cvz.append(best_class_cvz)

            most_common_cvz = Counter(predictions_buffer_cvz).most_common(1)[0][0]
            confidence_cvz = float(res_cvz[most_common_cvz])

            current_cls_cvz = most_common_cvz
            label_cvz = label_map_lstm.get(most_common_cvz, f"clase_{most_common_cvz}")

            if confidence_cvz > CONFIDENCE_THRESHOLD_LSTM:
                current_label_cvz = label_cvz
                current_conf_cvz  = confidence_cvz
            else:
                current_label_cvz = "Incierto..."
                current_conf_cvz  = confidence_cvz

        # --------------------------------------------------------
        # 6.5 CONSENSO GLOBAL ENTRE LOS 4 LSTM (0–4)
        # --------------------------------------------------------
        score_by_class = defaultdict(float)

        for cls, conf in zip(current_cls_main, current_confs_main):
            if cls >= 0 and conf > CONFIDENCE_THRESHOLD_LSTM:
                score_by_class[cls] += conf

        if current_cls_cvz >= 0 and current_conf_cvz > CONFIDENCE_THRESHOLD_LSTM:
            score_by_class[current_cls_cvz] += current_conf_cvz

        if score_by_class:
            global_lstm_cls = max(score_by_class, key=score_by_class.get)
            total_score = sum(score_by_class.values())
            global_lstm_conf = score_by_class[global_lstm_cls] / (total_score + 1e-8)
            global_lstm_label = label_map_lstm.get(global_lstm_cls, f"clase_{global_lstm_cls}")
        else:
            global_lstm_cls = -1
            global_lstm_conf = 0.0
            global_lstm_label = "Incierto..."

        # --------------------------------------------------------
        # 6.6 SELECCIÓN MOTOR (YOLO o LSTM) SEGÚN PASO REQUERIDO
        # --------------------------------------------------------
        pred_step_idx = -1
        confidence = 0.0
        model_used = "Esperando..."

        # variables para log de YOLO en metadata
        yolo_last_conf_for_log = 0.0
        yolo_detected_this_frame = False

        if not wash_complete:
            # Paso 3 YOLO exclusivo con conteo de picos
            if current_step_idx == 2:
                model_used = "YOLOv8"

                # ejecutar YOLO solo cada N frames procesados
                run_yolo_this_frame = (
                    YOLO_INFERENCE_EVERY_N <= 1
                    or (processed_frame_idx % YOLO_INFERENCE_EVERY_N == 0)
                )

                found_target = False
                best_conf = 0.0

                if run_yolo_this_frame:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame_y = cv2.GaussianBlur(rgb_frame, (5, 5), 0)

                    results_y = yolo_model(
                        processed_frame_y,
                        verbose=False,
                        conf=0.0,   
                        iou=0.0,
                        device=yolo_device
                    )

                    for r in results_y:
                        for box in r.boxes:
                            # if int(box.cls[0]) != ID_INTERCALADO: continue
                            conf_box = float(box.conf[0])
                            if conf_box > best_conf:
                                best_conf = conf_box
                                found_target = True
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                if not ANALYSIS_MODE:
                                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 0, 255), 2)

                yolo_last_conf_for_log = best_conf

                # 1) detección mínima
                if found_target and best_conf >= YOLO_CONF_MIN:
                    pred_step_idx = 2
                    confidence = best_conf
                    yolo_detected_this_frame = True
                else:
                    pred_step_idx = -1
                    confidence = 0.0

                # 2) pico fuerte (confianza muy alta) -> cuenta como repetición
                if found_target and best_conf >= YOLO_PEAK_THR and not yolo_in_peak:
                    yolo_peak_counter += 1
                    yolo_in_peak = True
                    yolo_peak_timer = YOLO_PEAK_COOLDOWN
                    print(f"[YOLO] Pico detectado! Conf={best_conf:.2f} | Picos acumulados: {yolo_peak_counter}")

                # 3) cooldown para no contar el mismo pico muchas veces
                if yolo_in_peak:
                    yolo_peak_timer -= 1
                    if yolo_peak_timer <= 0:
                        yolo_in_peak = False

            else:
                # Cualquier otro paso -> consenso de los 4 LSTM
                model_used = "4x LSTM (consenso)"
                if global_lstm_cls >= 0:
                    pred_step_idx = lstm_to_step.get(global_lstm_cls, -1)
                    confidence = global_lstm_conf
                else:
                    pred_step_idx = -1
                    confidence = 0.0

        # --------------------------------------------------------
        # 6.7 MÁQUINA DE ESTADOS DE PASOS + BARRA / PICOS
        # --------------------------------------------------------
        if not wash_complete:
            if current_step_idx == 2:
                # Paso 3: usar picos YOLO
                if yolo_peak_counter >= YOLO_PEAKS_REQUIRED:
                    print(f"PASO {current_step_idx} ({label_map_steps.get(current_step_idx)}) COMPLETADO por YOLO (picos).")
                    # reset estado local de YOLO para futuros lavados
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
                # Pasos 1,2,4,5,6 -> lógica LSTM clásica
                if pred_step_idx == current_step_idx:
                    frames_validated += 1
                    status_color = (0, 255, 0)  # verde
                else:
                    if frames_validated > 0:
                        frames_validated -= 1
                    status_color = (0, 0, 255)  # rojo

                if frames_validated >= SEQ_LEN_REQUIRED:
                    print(f"PASO {current_step_idx} ({label_map_steps.get(current_step_idx)}) COMPLETADO.")
                    frames_validated = 0
                    current_step_idx += 1

                    if current_step_idx > 5:
                        wash_complete = True
                        current_step_idx = 5

        # --------------------------------------------------------
        # 6.8 LOG DE METADATA PARA ANÁLISIS UVEH
        # --------------------------------------------------------
        if ANALYSIS_MODE:
            if current_step_idx == 2:
                mode_str   = "yolo"
                label_str  = "intercalado" if yolo_detected_this_frame else "sin_deteccion"
                raw_label  = label_str
            else:
                mode_str   = "lstm"
                label_str  = global_lstm_label
                raw_label  = global_lstm_label

            record = {
                "person_id":        person_id,
                "video_name":       video_name,
                "frame":            processed_frame_idx,
                "timestamp_s":      timestamp_s,
                "step_required_idx":current_step_idx,
                "step_required":    label_map_steps.get(current_step_idx, "NA"),
                "mode":             mode_str,             
                "pred_step_idx":    pred_step_idx,
                "pred_step_label":  label_map_steps.get(pred_step_idx, "NA") if pred_step_idx >= 0 else "NA",
                "global_lstm_cls":  global_lstm_cls,
                "global_lstm_label":global_lstm_label,
                "global_lstm_conf":global_lstm_conf,
                "yolo_conf":        yolo_last_conf_for_log,
                "yolo_peak_counter":yolo_peak_counter,
                "frames_validated": frames_validated,
                "wash_complete":    wash_complete,
                "label":            label_str,
                "raw_label":        raw_label,
            }
            metadata_records.append(record)

        # --------------------------------------------------------
        # 6.9 UI: PANEL SUPERIOR + BARRA DE PROGRESO + MODELOS
        # --------------------------------------------------------
        if not ANALYSIS_MODE:
            # Fondo superior
            cv2.rectangle(img_display, (0, 0), (640, 150), (30, 30, 30), -1)

            if wash_complete:
                cv2.putText(img_display, "LAVADO COMPLETADO", (80, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                # Texto: Paso requerido
                req_text = f"REQUERIDO: {label_map_steps.get(current_step_idx, '---')}"
                cv2.putText(img_display, req_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Info motor + predicción
                if current_step_idx == 2:
                    # Motor YOLO: Se mantiene la visualización original simple
                    pred_text = f"Motor: {model_used} | Conf YOLO ultima: {confidence:.2f} | Picos: {yolo_peak_counter}/{YOLO_PEAKS_REQUIRED}"
                    cv2.putText(img_display, pred_text, (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    # Motor LSTM: IMPLEMENTACIÓN DEL RESALTADO POR SEGMENTOS
                    
                    # --- CONFIGURACIÓN DE COLORES Y FUENTES ---
                    COLOR_BASE_L = (200, 200, 200)       # Gris claro para texto fijo
                    COLOR_HIGHLIGHT_L = (0, 255, 255)    # Amarillo para resaltar
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    thickness_base = 1
                    thickness_highlight = 2 
                    x_start, y_start = 10, 55 

                    text_base = f"Motor: {model_used} | LSTM clase: "

                    text_class = f"{global_lstm_cls} ({global_lstm_label})"

                    text_connector = " | Conf: "

                    text_conf = f"{global_lstm_conf:.2f}"
                    
                    x_current = x_start
                    
                    cv2.putText(img_display, text_base, (x_current, y_start),
                                fontFace, fontScale, COLOR_BASE_L, thickness_base)
                    
                    (w1, h1), baseline1 = cv2.getTextSize(text_base, fontFace, fontScale, thickness_base)
                    x_current += w1 
                    
                    cv2.putText(img_display, text_class, (x_current, y_start),
                                fontFace, fontScale, COLOR_HIGHLIGHT_L, thickness_highlight)
                    
                    (w2, h2), baseline2 = cv2.getTextSize(text_class, fontFace, fontScale, thickness_highlight)
                    x_current += w2 
                    
                    cv2.putText(img_display, text_connector, (x_current, y_start),
                                fontFace, fontScale, COLOR_BASE_L, thickness_base)
                    
                    (w3, h3), baseline3 = cv2.getTextSize(text_connector, fontFace, fontScale, thickness_base)
                    x_current += w3 
                    
                    cv2.putText(img_display, text_conf, (x_current, y_start),
                                fontFace, fontScale, COLOR_HIGHLIGHT_L, thickness_highlight)

                # Barra de Progreso 
                bar_x, bar_y, bar_w, bar_h = 10, 75, 620, 20
                cv2.rectangle(img_display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                              (50, 50, 50), -1)

                if current_step_idx == 2:
                    prog_ratio = min(yolo_peak_counter / YOLO_PEAKS_REQUIRED, 1.0)
                else:
                    prog_ratio = min(frames_validated / SEQ_LEN_REQUIRED, 1.0)

                fill_w = int(bar_w * prog_ratio)
                cv2.rectangle(img_display, (bar_x, bar_y),
                              (bar_x + fill_w, bar_y + bar_h), status_color, -1)

                # Info detallada de cada LSTM 
                y0 = 105
                dy = 18
                for i, (name, lbl, conf_i) in enumerate(zip(model_names_main, current_labels_main, current_confs_main)):
                    txt = f"{name}: {lbl} ({conf_i:.2f})"
                    cv2.putText(img_display, txt, (10, y0 + i * dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

                txt_cvz = f"{model_name_cvz}: {current_label_cvz} ({current_conf_cvz:.2f})"
                cv2.putText(img_display, txt_cvz, (10, y0 + len(model_names_main) * dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            cv2.imshow("Sistema Hibrido 4xLSTM + YOLO (6 pasos OMS)", img_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return metadata_records


# ============================================================
# 7. MAIN: PROCESAR UNO O VARIOS VIDEOS Y GUARDAR METADATA
# ============================================================

if __name__ == "__main__":
    # Para análisis batch pongan esto en True
    ANALYSIS_MODE = False
    SAVE_METADATA = True
    METADATA_CSV = "metadata_inferencia_multi_personas.csv"

    # Llenen esta lista con todas las personas y videos que quieran procesar (se puede automatizar)
    VIDEOS = [
        ("Persona_01", r"Videos\vlc-record-2025-11-28-13h05m04s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"),
        ("Persona_02", r"Videos\vlc-record-2025-11-26-13h38m38s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"),
        ("Persona_03", r"Videos\vlc-record-2025-12-05-12h03m08s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"),
        ("Persona_04", r"Videos\vlc-record-2025-12-05-12h11m10s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"),
        ("Persona_05", r"Videos\vlc-record-2025-12-05-12h24m18s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"),
        ("Persona_06", r"Videos\vlc-record-2025-12-05-13h16m20s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4"),
        ("Persona_07", r"Videos\vlc-record-2025-12-05-13h35m25s-rtsp___169.254.11.181_554_live1s1.sdp-.mp4")
    ]
    

    all_metadata = []

    if VIDEOS:  # modo multiples videos
        for person_id, video_path in VIDEOS:
            print(f"\nProcesando video de {person_id}: {video_path}")
            registros = procesar_video(video_path, person_id)
            all_metadata.extend(registros)
    else:       # modo solo un video (VIDEO_SOURCE + PERSON_ID)
        print(f"\nProcesando fuente única: {VIDEO_SOURCE}")
        registros = procesar_video(VIDEO_SOURCE, PERSON_ID)
        all_metadata.extend(registros)

    # ========================================================
    # 8. GUARDAR METADATA (TODAS LAS PERSONAS + VIDEOS)
    #    Y HACER EL ANÁLISIS GLOBAL / POR PERSONA
    # ========================================================
    if ANALYSIS_MODE and SAVE_METADATA and all_metadata:
        df = pd.DataFrame(all_metadata)
        df.to_csv(METADATA_CSV, index=False, encoding="utf-8-sig")
        print(f"\nMetadata guardada en: {METADATA_CSV}")

        # --------------------------------------------------------------------
        # 1) CONTEO GLOBAL
        # --------------------------------------------------------------------
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

        cobertura_global = {paso: (conteo_pasos_global[paso] > 0) for paso in pasos_validos}
        print("\n¿Se detectó cada uno de los 6 pasos (GLOBAL)?")
        for paso, presente in cobertura_global.items():
            print(f"  {paso}: {'Sí' if presente else 'No'}")

    
        df["deteccion_6_pasos"] = df["label"].isin(pasos_validos)

        # --------------------------------------------------------------------
        # 2) ANÁLISIS POR PERSONA
        # --------------------------------------------------------------------
        print("\n\n========================")
        print("ANÁLISIS POR PERSONA")
        print("========================")

        for person_id in sorted(df["person_id"].unique()):
            print(f"\n--- Persona: {person_id} ---")
            df_p = df[df["person_id"] == person_id].copy()

            conteo_pasos_p = (
                df_p["label"]
                .value_counts()
                .reindex(pasos_validos + ["sin_deteccion"], fill_value=0)
            )

            print("Conteo por paso (incluye sin_deteccion):")
            print(conteo_pasos_p)

            cobertura_p = {paso: (conteo_pasos_p[paso] > 0) for paso in pasos_validos}
            print("\n¿Se detectó cada uno de los 6 pasos para esta persona?")
            for paso, presente in cobertura_p.items():
                print(f"  {paso}: {'Sí' if presente else 'No'}")

            if all(cobertura_p.values()):
                print("Cobertura completa: los 6 pasos fueron detectados al menos una vez.")
            else:
                faltantes = [p for p, ok in cobertura_p.items() if not ok]
                print(f"Faltan detecciones para: {faltantes}")
