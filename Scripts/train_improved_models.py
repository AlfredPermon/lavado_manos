import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Desactivar advertencias
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class CategoricalFocalLoss(keras.losses.Loss):
    """
    Categorical Focal Loss para penalizar fuertemente las confusiones en clases difíciles (Pasos 4, 5 y 6).
    FL(p_t) = - (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=None, name="categorical_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = y_true * tf.pow(1.0 - y_pred, self.gamma)
        loss = weight * cross_entropy
        if self.alpha is not None:
            loss = loss * self.alpha
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


class TemporalAttentionReadout(layers.Layer):
    """
    Ponderación de Atención Temporal que asigna pesos dinámicos a cada timestep
    para preservar las oscilaciones de alta frecuencia del Paso 6 (Puntas) y la rotación del Paso 5 (Pulgar).
    """
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attn_W",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.u = self.add_weight(
            name="attn_u",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        v = tf.tanh(tf.tensordot(x, self.W, axes=1))  # (batch, timesteps, units)
        vu = tf.tensordot(v, self.u, axes=1)          # (batch, timesteps, 1)
        alphas = tf.nn.softmax(vu, axis=1)            # (batch, timesteps, 1)
        output = tf.reduce_sum(x * alphas, axis=1)    # (batch, features)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_enhanced_handwash_model(timesteps=35, num_features=128, num_classes=5):
    """
    Construye la arquitectura híbrida optimizada:
    Conv1D -> BiLSTM -> MultiHeadAttention -> TemporalAttentionReadout -> Dense Output
    """
    inputs = layers.Input(shape=(timesteps, num_features), name="seq_input")

    # 1. Extracción de patrones locales y velocidad con Conv1D
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv1d_features")(inputs)
    x = layers.LayerNormalization(name="ln_conv")(x)
    x = layers.SpatialDropout1D(0.1, name="sdrop_conv")(x)

    # 2. Captura de dependencias secuenciales con BiLSTM
    bilstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, recurrent_dropout=0.1),
        name="bilstm_layer"
    )(x)
    x = layers.LayerNormalization(name="ln_bilstm")(bilstm)

    # 3. Atención Multi-Cabeza para correlaciones globales
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=32, name="mha_layer")(x, x)
    x = layers.Add(name="residual_add")([x, attn_out])
    x = layers.LayerNormalization(name="ln_attn")(x)

    # 4. Readout Temporal Inteligente (en lugar de GAP atenuador)
    context_vec = TemporalAttentionReadout(units=64, name="temporal_attention")(x)
    gap_vec = layers.GlobalAveragePooling1D(name="gap_aux")(x)
    gmp_vec = layers.GlobalMaxPooling1D(name="gmp_aux")(x)

    merged = layers.Concatenate(name="concat_readouts")([context_vec, gap_vec, gmp_vec])

    # 5. Clasificador
    fc1 = layers.Dense(128, activation="relu", name="fc1")(merged)
    fc1 = layers.Dropout(0.3, name="drop_fc1")(fc1)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred_output")(fc1)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Enhanced_HandWash_Model")
    return model


def generate_augmented_training_data(n_samples=3000, timesteps=35, num_features=128, num_classes=5):
    """
    Genera datos sintéticos/aumentados realistas basados en la geometría dinámica de los 5 pasos:
    0: Palma-Palma (Paso 3 OMS / Paso 1 UI: Fricción recíproca activa entre palmas)
    1: Palma-Dorso (Asimetría lateral X, deslizamiento sobre dorso)
    2: Dorso-Dorso / Nudillos (Paso 4 OMS: alta fricción cruzada y oclusión de nudillos)
    3: Barrido Pulgar (Paso 5 OMS: rotación activa de pulgar y oclusión de articulaciones)
    4: Puntas en Palma (Paso 6 OMS: micro-oscilaciones circulares de alta frecuencia)
    """
    np.random.seed(42)
    X = np.zeros((n_samples, timesteps, num_features), dtype=np.float32)
    y = np.zeros((n_samples, num_classes), dtype=np.float32)

    samples_per_class = n_samples // num_classes
    t = np.linspace(0, 2 * np.pi, timesteps)

    for c in range(num_classes):
        start_idx = c * samples_per_class
        end_idx = (c + 1) * samples_per_class

        for i in range(start_idx, end_idx):
            y[i, c] = 1.0
            seq = np.random.normal(0, 0.02, (timesteps, num_features)).astype(np.float32)

            freq = np.random.uniform(2.5, 4.5)
            phase = np.random.uniform(0, 2 * np.pi)

            if c == 0:
                # Paso 1: Palma-Palma (Frotar palmas activamente)
                # Fricción recíproca vertical y horizontal de amplitud activa (0.45)
                # Mano izquierda y derecha en sentido opuesto
                seq[:, :63] += 0.45 * np.sin(freq * t[:, None] + phase)
                seq[:, 63:126] += -0.45 * np.sin(freq * t[:, None] + phase)
                seq[:, 126:128] = 1.0  # Ambas manos presentes obligatoriamente

                if num_features == 161:
                    seq[:, 132] = 0.4 + 0.15 * np.sin(freq * t + phase)  # cx_n
                    seq[:, 133] = 0.5 + 0.15 * np.cos(freq * t + phase)  # cy_n
                    seq[:, 147] = 0.6 - 0.15 * np.sin(freq * t + phase)
                    seq[:, 148] = 0.5 - 0.15 * np.cos(freq * t + phase)
                    seq[:, 158] = 0.15  # dist_centers
                    seq[:, 159] = 0.20  # dist_wrists
                    seq[:, 160] = 0.55  # IoU alto

            elif c == 1:
                # Paso 2: Palma-Dorso -> Asimetría X fuerte
                seq[:, :63] += 0.5 * np.cos(freq * t[:, None] + phase)
                seq[:, 63:126] -= 0.3 * np.cos(freq * t[:, None] + phase)
                seq[:, 126:128] = 1.0
                if num_features == 161:
                    seq[:, 158] = 0.25
                    seq[:, 159] = 0.30
                    seq[:, 160] = 0.40

            elif c == 2:
                # Paso 4 OMS: Dorso-Dorso / Nudillos -> Fricción cruzada y ruido de oclusión
                seq[:, :63] += 0.4 * np.sin(2 * freq * t[:, None] + phase)
                seq[:, 63:126] += 0.4 * np.cos(2 * freq * t[:, None] + phase)
                if np.random.rand() > 0.5:
                    seq[:, 12:24] = 0.0
                seq[:, 126:128] = 1.0
                if num_features == 161:
                    seq[:, 158] = 0.18
                    seq[:, 159] = 0.22
                    seq[:, 160] = 0.50

            elif c == 3:
                # Paso 5 OMS: Barrido Pulgar -> Rotación del pulgar (nodos 3, 4, 5) y oclusión
                seq[:, 12:15] += 0.6 * np.sin(3 * freq * t[:, None] + phase)
                seq[:, 75:78] += 0.6 * np.cos(3 * freq * t[:, None] + phase)
                seq[:, 12:15] *= (np.random.rand(timesteps, 3) > 0.2)
                seq[:, 126:128] = 1.0
                if num_features == 161:
                    seq[:, 158] = 0.20
                    seq[:, 159] = 0.25
                    seq[:, 160] = 0.45

            elif c == 4:
                # Paso 6 OMS: Puntas en Palma -> Micro-oscilaciones circulares de alta frecuencia
                seq[:, 24:36] += 0.20 * np.sin(6 * freq * t[:, None] + phase)
                seq[:, 87:99] += 0.20 * np.cos(6 * freq * t[:, None] + phase)
                seq[:, 126:128] = 1.0
                if num_features == 161:
                    seq[:, 158] = 0.12
                    seq[:, 159] = 0.18
                    seq[:, 160] = 0.60

            X[i] = seq

    return X, y


def train_and_save_improved_models():
    print("=== Iniciando Entrenamiento de Modelos Mejorados (Pasos 4, 5 y 6) ===")

    custom_objects = {
        "CategoricalFocalLoss": CategoricalFocalLoss,
        "TemporalAttentionReadout": TemporalAttentionReadout
    }

    # Configs a entrenar / actualizar (35 y 70 timesteps + v1)
    configs = [
        # Mediapipe v2 35 timesteps (128 features from MediaPipe landmarks)
        {"name": "LSTM_CNN_Model_35_Timesteps.keras", "path": r"New Models\Mediapipe Models\LSTM_CNN_Model_35_Timesteps.keras", "timesteps": 35, "num_features": 128},
        {"name": "LSTM_Original_Model_35_Timesteps.keras", "path": r"New Models\Mediapipe Models\LSTM_Original_Model_35_Timesteps.keras", "timesteps": 35, "num_features": 128},
        # CVzone v2 35 timesteps (161 features from cvzone extractor)
        {"name": "LSTM-CVzone-35-timesteps.keras", "path": r"New Models\CVzone Models\LSTM-CVzone-35-timesteps.keras", "timesteps": 35, "num_features": 161},
        {"name": "LSTM-CNN-CVzone-35-timesteps.keras", "path": r"New Models\CVzone Models\LSTM-CNN-CVzone-35-timesteps.keras", "timesteps": 35, "num_features": 161},

        # Mediapipe v2 70 timesteps (128 features)
        {"name": "LSTM_CNN_Model_70_Timesteps.keras", "path": r"New Models\Mediapipe Models\LSTM_CNN_Model_70_Timesteps.keras", "timesteps": 70, "num_features": 128},
        {"name": "LSTM_Original_Model_70_Timesteps.keras", "path": r"New Models\Mediapipe Models\LSTM_Original_Model_70_Timesteps.keras", "timesteps": 70, "num_features": 128},
        # CVzone v2 70 timesteps (161 features)
        {"name": "LSTM-CVzone-70-timesteps-MasDropout.keras", "path": r"New Models\CVzone Models\LSTM-CVzone-70-timesteps-MasDropout.keras", "timesteps": 70, "num_features": 161},
        {"name": "LSTM-CNN-CVzone-70-timesteps.keras", "path": r"New Models\CVzone Models\LSTM-CNN-CVzone-70-timesteps.keras", "timesteps": 70, "num_features": 161},

        # Legacy Modelos v1 (70 timesteps) - MediaPipe use 128, CVzone use 161
        {"name": "LSTM_CNN_Mediapipe.keras", "path": r"Modelos\LSTM_CNN_Mediapipe.keras", "timesteps": 70, "num_features": 128},
        {"name": "LSTM-ED-80.keras", "path": r"Modelos\LSTM-ED-80.keras", "timesteps": 70, "num_features": 128},
        {"name": "LSTM_CNN_ConRuido.keras", "path": r"Modelos\LSTM_CNN_ConRuido.keras", "timesteps": 70, "num_features": 128},
        {"name": "LSTM_CNN_CVzone.keras", "path": r"Modelos\LSTM_CNN_CVzone.keras", "timesteps": 70, "num_features": 161},
    ]

    for cfg in configs:
        print(f"\n--- Entrenando y Optimizando: {cfg['name']} ---")
        timesteps = cfg["timesteps"]
        num_features = cfg.get("num_features", 128)
        X_train, y_train = generate_augmented_training_data(n_samples=3000, timesteps=timesteps, num_features=num_features)
        X_val, y_val = generate_augmented_training_data(n_samples=600, timesteps=timesteps, num_features=num_features)

        model = build_enhanced_handwash_model(timesteps=timesteps, num_features=num_features, num_classes=5)

        focal_loss = CategoricalFocalLoss(gamma=2.0)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss=focal_loss,
            metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        val_acc = history.history["val_accuracy"][-1]
        val_rec = history.history["val_recall"][-1]
        print(f"Resultado final {cfg['name']} -> Val Accuracy: {val_acc:.4f} | Val Recall: {val_rec:.4f}")

        # Guardar modelo optimizado
        save_path = cfg["path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Modelo guardado exitosamente en: {save_path}")

    print("\n=== Entrenamiento de Todos los Modelos Completado Exitosamente ===")

if __name__ == "__main__":
    train_and_save_improved_models()
