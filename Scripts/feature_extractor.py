import cv2
import numpy as np
import mediapipe as mp

class EnhancedFeatureExtractor:
    """
    Extractor de características optimizado para higiene de manos.
    Garantiza:
    1. Ordenamiento estricto por lateralidad (Mano Izquierda siempre slots 0..62, Derecha slots 63..125).
    2. Normalización centrada en la muñeca (Wrist-Centric) e invariante a escala.
    3. Cálculo de derivadas de velocidad temporal (v_x, v_y, v_z).
    4. Métricas geométricas inter-mano (distancias yemas-palma, pulgar-palma).
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.prev_raw_coords = None

    def normalize_hand_landmarks(self, landmarks_3d):
        """
        Centra los landmarks en la muñeca (Nodo 0) y los escala según la longitud de la palma
        (distancia entre la muñeca 0 y el nudillo medio 9).
        """
        if not landmarks_3d or len(landmarks_3d) != 21:
            return [0.0] * 63

        wrist = landmarks_3d[0]
        middle_mcp = landmarks_3d[9]

        scale = np.sqrt(
            (middle_mcp[0] - wrist[0])**2 +
            (middle_mcp[1] - wrist[1])**2 +
            (middle_mcp[2] - wrist[2])**2
        )
        if scale < 1e-6:
            scale = 1.0

        normalized = []
        for lm in landmarks_3d:
            nx = (lm[0] - wrist[0]) / scale
            ny = (lm[1] - wrist[1]) / scale
            nz = (lm[2] - wrist[2]) / scale
            normalized.extend([nx, ny, nz])

        return normalized

    def extract_structured_frame(self, frame, wrist_centric=True):
        """
        Extrae datos ordenados estrictamente por lateralidad:
        - Mano Izquierda: 63 floats
        - Mano Derecha: 63 floats
        - Flags de presencia: [has_left, has_right] (2 floats)
        Total base: 128 floats
        """
        results = self.mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        left_lms_raw = None
        right_lms_raw = None
        has_left = 0.0
        has_right = 0.0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                lms_3d = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]

                if label == 'Left' and left_lms_raw is None:
                    left_lms_raw = lms_3d
                    has_left = 1.0
                elif label == 'Right' and right_lms_raw is None:
                    right_lms_raw = lms_3d
                    has_right = 1.0
                elif left_lms_raw is None:
                    left_lms_raw = lms_3d
                    has_left = 1.0
                elif right_lms_raw is None:
                    right_lms_raw = lms_3d
                    has_right = 1.0

        if wrist_centric:
            left_vec = self.normalize_hand_landmarks(left_lms_raw)
            right_vec = self.normalize_hand_landmarks(right_lms_raw)
        else:
            left_vec = [c for lm in left_lms_raw for c in lm] if left_lms_raw else [0.0] * 63
            right_vec = [c for lm in right_lms_raw for c in lm] if right_lms_raw else [0.0] * 63

        raw_coords = left_vec + right_vec  # 126 floats
        presence_flags = [has_left, has_right]

        # Velocidades (derivadas temporales)
        if self.prev_raw_coords is not None:
            velocities = [c - p for c, p in zip(raw_coords, self.prev_raw_coords)]
        else:
            velocities = [0.0] * 126
        self.prev_raw_coords = raw_coords

        # Caracteristicas Geometricas Inter-Mano
        geom_features = self.compute_interhand_geometry(left_lms_raw, right_lms_raw, has_left, has_right)

        full_features = raw_coords + presence_flags + velocities + geom_features
        legacy_features = raw_coords + presence_flags  # 128 floats compatibles con modelos existentes

        return {
            "legacy_128": legacy_features,
            "full_features": full_features,
            "left_raw": left_lms_raw,
            "right_raw": right_lms_raw,
            "has_left": has_left,
            "has_right": has_right
        }

    def compute_interhand_geometry(self, left_lms, right_lms, has_left, has_right):
        """
        Calcula 16 métricas geométricas clave entre ambas manos:
        - Distancias de yemas (8, 12, 16, 20) a la palma opuesta (9)
        - Distancia del pulgar (4) a la palma opuesta (9)
        - Distancia entre muñecas (0)
        - Extensión de dedos
        """
        if not (has_left and has_right and left_lms and right_lms):
            return [0.0] * 16

        def dist3d(p1, p2):
            return float(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2))

        # Yemas Izquierda -> Palma Derecha (9)
        d_L8_R9 = dist3d(left_lms[8], right_lms[9])
        d_L12_R9 = dist3d(left_lms[12], right_lms[9])
        d_L16_R9 = dist3d(left_lms[16], right_lms[9])
        d_L20_R9 = dist3d(left_lms[20], right_lms[9])
        d_L4_R9 = dist3d(left_lms[4], right_lms[9])

        # Yemas Derecha -> Palma Izquierda (9)
        d_R8_L9 = dist3d(right_lms[8], left_lms[9])
        d_R12_L9 = dist3d(right_lms[12], left_lms[9])
        d_R16_L9 = dist3d(right_lms[16], left_lms[9])
        d_R20_L9 = dist3d(right_lms[20], left_lms[9])
        d_R4_L9 = dist3d(right_lms[4], left_lms[9])

        # Muñecas
        d_wrist = dist3d(left_lms[0], right_lms[0])

        # Pulgar a Pulgar
        d_thumb_thumb = dist3d(left_lms[4], right_lms[4])

        # Nudillo medio a Nudillo medio
        d_mid_mid = dist3d(left_lms[9], right_lms[9])

        # Apertura de dedos (Thumb 4 - Pinky 20)
        spread_L = dist3d(left_lms[4], left_lms[20])
        spread_R = dist3d(right_lms[4], right_lms[20])

        # Overlap estimado X
        overlap_x = max(0.0, 1.0 - abs(left_lms[0][0] - right_lms[0][0]))

        return [
            d_L8_R9, d_L12_R9, d_L16_R9, d_L20_R9, d_L4_R9,
            d_R8_L9, d_R12_L9, d_R16_L9, d_R20_L9, d_R4_L9,
            d_wrist, d_thumb_thumb, d_mid_mid, spread_L, spread_R, overlap_x
        ]
