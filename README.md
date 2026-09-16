# HW DEMO V3 — Sistema Híbrido de Evaluación de Lavado de Manos (OMS)

Sistema inteligente de visión por computadora diseñado para validar la técnica de higiene de manos según los estándares de la OMS. Combina modelos de Deep Learning (LSTM sobre MediaPipe/CVZone) y Detección de Objetos (YOLOv8) para asegurar el cumplimiento de los protocolos sanitarios en tiempo real.

![Estado](https://img.shields.io/badge/Estado-Activo-green) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## 🚀 Características Principales

### 🖥️ Nueva Interfaz Gráfica (GUI)
- **Panel de Control Intuitivo**: Visualización en tiempo real con retroalimentación inmediata.
- **Asistente de Voz Integrado**: Instrucciones paso a paso con control de velocidad y función de repetición.
- **Modos de Operación**:
  - **Práctica**: Guiado paso a paso sin penalizaciones.
  - **Evaluación**: Sistema de puntuación (0-100) que penaliza errores de duración (<6s) o calidad.
  - **Demostración**: Sincronización automática con videos de referencia para entrenamiento.
- **Panel "5 Momentos"**: Selección de los momentos de higiene según la OMS para registro en auditorías.

### 🧠 Inteligencia Artificial Híbrida
- **Arquitectura de Consenso**: Combina predicciones de 4 modelos LSTM (2 basados en MediaPipe, 2 en CVZone) para robustez máxima.
- **Validación Bilateral**: Detecta y asegura que ambos lados de las manos (derecha e izquierda) sean frotados correctamente en pasos críticos.
- **YOLOv8 Especializado**: Modelo dedicado para detectar pasos complejos como "Palmas entrelazadas" (Paso 3).

### 📊 Reportes y Auditoría
- **Historial CSV**: Registro automático de cada sesión (`historial_lavado.csv`) con fecha, puntaje, duración y detalles por paso.
- **Alertas de Higiene**: Sistema de recordatorios configurables (por defecto cada 1 hora).

---

## 📂 Estructura del Proyecto

```text
c:\lavado_manos_v1\
├── Scripts/
│   ├── gui_app.py                  # [PRINCIPAL] Lanzador de la aplicación gráfica
│   ├── hand_wash_analyzer.py       # Motor lógico de análisis (IA, Máquina de Estados)
│   ├── hand_wash_manual.py         # Ventana de ayuda y manual interactivo
│   ├── voice_assistant.py          # Módulo de síntesis de voz (TTS)
│   ├── video_sync.py               # Sincronización para modo demostración
│   ├── DEMO REPORTE UVEH...py      # Scripts de inferencia legacy (sin GUI completa)
│   └── historial_lavado.csv        # Log de sesiones
├── Modelos/                        # Pesos de modelos (.keras, .pt)
├── New Models/                     # Nuevos modelos re-entrenados (v2)
├── Videos/                         # Videos de prueba y demostración
├── requirements.txt                # Dependencias del proyecto
├── setup.bat                       # Script de instalación automática
└── README.md                       # Documentación
```

---

## 🛠️ Instalación y Configuración

### Prerrequisitos
- **Python 3.10** o superior.
- **Cámara Web** (para pruebas en vivo).
- (Opcional) GPU NVIDIA para mayor rendimiento con TensorFlow/YOLO.

### Instalación Rápida (Windows)
1. Ejecuta el script de configuración automática:
   ```cmd
   setup.bat
   ```
   Esto creará el entorno virtual e instalará todas las dependencias.

### Instalación Manual
1. Crea un entorno virtual:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Guía de Uso

### 1. Iniciar la Aplicación (Recomendado)
Para lanzar la interfaz completa con todas las funcionalidades:
```bash
python Scripts/gui_app.py
```

### 2. Flujo de Trabajo en la App
1. **Configuración**:
   - Selecciona el **Modo** (Práctica/Evaluación).
   - Verifica el **Modelo** (Recomendado: *Versión 2 - 35 timesteps*).
   - Activa/Desactiva el audio con el botón 🔊.
2. **Ejecución**:
   - Presiona **"▶ Iniciar"** o activa **"Auto-Inicio"** para detección manos libres.
   - Sigue las instrucciones de voz y visuales.
   - La barra de progreso y el temporizador te guiarán.
3. **Resultados**:
   - Al finalizar, verás tu puntaje y un desglose de errores (si los hubo).
   - Los datos se guardan automáticamente en el CSV.

### 3. Scripts Clásicos (Legacy)
Si prefieres correr solo el motor de inferencia sin la GUI completa:
- **Versión 2 (Optimizado)**:
  ```bash
  python "Scripts/DEMO REPORTE UVEH Version 2 35 timesteps.py"
  ```
- **Modo Análisis (Batch)**:
  Edita el script para establecer `ANALYSIS_MODE = True` y procesar una lista de videos automáticamente.

---

## 🧠 Detalles Técnicos

### Pipeline de Procesamiento
1. **Entrada**: Video (Webcam o Archivo).
2. **Extracción de Features**:
   - **MediaPipe Hands**: Landmarks 3D (x, y, z).
   - **CVZone**: Features geométricos (distancias, ángulos).
3. **Inferencia (En paralelo)**:
   - **LSTMs**: Analizan secuencias temporales (ventanas de 35 o 70 frames).
   - **YOLOv8**: Se activa específicamente para validar la configuración espacial compleja del Paso 3.
4. **Consenso y Lógica**:
   - Un algoritmo de votación ponderada decide el paso actual.
   - La **Máquina de Estados** valida la duración (min 6s) y la bilateralidad (frotado de ambos lados).

### Lógica Bilateral
Para los pasos asimétricos (e.g., frotar dorso izquierdo con palma derecha), el sistema rastrea la posición relativa de las manos y exige que se cumpla el tiempo mínimo en **ambas configuraciones** (Lado A y Lado B) antes de validar el paso como completo.

---

## ⚠️ Solución de Problemas

- **Error "MediaPipe solutions MISSING"**: Asegúrate de usar una versión compatible de Python (3.10 recomendado) y `mediapipe`. Ejecuta `python test_mp.py` para diagnosticar.
- **Lentitud**: Reduce la resolución de entrada o cambia al modelo de "35 timesteps" que es más ligero.
- **Falsos Positivos**: Ajusta el `CONFIDENCE_THRESHOLD` en `hand_wash_analyzer.py` si las condiciones de luz son malas.

---

**Desarrollado para validación de protocolos UVEH.**
