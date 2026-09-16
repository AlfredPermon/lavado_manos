import mediapipe as mp
import sys
print(f"MediaPipe file: {mp.__file__}")
try:
    import mediapipe.python.solutions
    print("Direct import of mediapipe.python.solutions worked")
except ImportError as e:
    print(f"Direct import failed: {e}")

try:
    print(f"Solutions: {mp.solutions}")
except AttributeError:
    print("mp.solutions missing")
