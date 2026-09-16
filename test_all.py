import mediapipe as mp
import tensorflow as tf
print(f"MediaPipe version: {mp.__version__}")
print(f"TensorFlow version: {tf.__version__}")
try:
    print(f"Solutions: {mp.solutions}")
    print("MediaPipe solutions OK")
except AttributeError:
    print("MediaPipe solutions MISSING")

print("All imports successful")
