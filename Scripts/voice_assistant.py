import threading
import queue
import subprocess
import time
import re
import unicodedata

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

class VoiceAssistant:
    def __init__(self):
        self.queue = queue.Queue()
        self.is_running = True
        self.current_process = None
        self.muted = False
        self.speed = 0  # Default speed (-10 to 10)
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        self.last_text = ""
        
        # Initialize Recognizer if available
        self.recognizer = sr.Recognizer() if HAS_SR else None
        self.mic_index = None
        self.mic_name = "Micrófono predeterminado"
        if self.recognizer:
            self.mic_index, self.mic_name = self._select_microphone()
            # Optimized settings for better sensitivity
            self.recognizer.energy_threshold = 120
            self.recognizer.dynamic_energy_threshold = False # Disable auto-adjustment by default to prevent desensitization
            self.recognizer.pause_threshold = 1.0 # Seconds of silence to consider phrase end
            self.recognizer.phrase_threshold = 0.3 # Minimum seconds of speaking
            self.recognizer.non_speaking_duration = 0.35 # Keep slightly more responsive

    def calibrate(self, duration=1.0):
        """Calibrates microphone for ambient noise."""
        if not HAS_SR or not self.recognizer:
            return False
            
        try:
            with self._open_microphone() as source:
                print(f"Calibrating background noise with: {self.mic_name}")
                # Temporarily enable dynamic threshold for calibration
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                
                # Lock the threshold to prevent it from skyrocketing due to transient noise
                self.recognizer.dynamic_energy_threshold = False
                
                # Ensure threshold isn't too extreme
                current_threshold = self.recognizer.energy_threshold
                print(f"Calibration raw threshold: {current_threshold}")
                
                # Lower the threshold slightly so normal speech is accepted more easily.
                target_threshold = current_threshold * 0.75
                if target_threshold < 25:
                    self.recognizer.energy_threshold = 25
                elif target_threshold > 2000:
                    self.recognizer.energy_threshold = 2000
                else:
                    self.recognizer.energy_threshold = target_threshold
                    
                print(f"Calibration complete. Locked Threshold: {self.recognizer.energy_threshold}")
            return True
        except Exception as e:
            print(f"Calibration error: {e}")
            return False

    def listen(self, timeout=5, phrase_time_limit=3, calibrate=False):
        """
        Listens to the microphone and returns recognized text.
        Args:
            calibrate (bool): Whether to adjust for ambient noise before listening.
        Returns:
            str: Recognized text (lowercase), or Error code (ERROR_*, NO_SPEECH, UNKNOWN).
        """
        if not HAS_SR or not self.recognizer:
            return "ERROR_IMPORT"
            
        try:
            # Use default microphone
            with self._open_microphone() as source:
                if calibrate:
                    print("Adjusting for ambient noise (Calibration requested)...")
                    self.recognizer.dynamic_energy_threshold = True
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    self.recognizer.dynamic_energy_threshold = False
                
                print(f"Listening with {self.mic_name}... (Threshold: {self.recognizer.energy_threshold})")
                
                try:
                    # Listen uses the recognizer settings
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                except sr.WaitTimeoutError:
                    print("Timeout waiting for speech.")
                    return "NO_SPEECH"
                
            try:
                # Recognize using Google Web Speech API
                print("Recognizing...")
                text = self.recognizer.recognize_google(audio, language="es-ES")
                print(f"Recognized: {text}")
                return text.lower()
            except sr.UnknownValueError:
                print("Speech not understood.")
                return "UNKNOWN"
            except sr.RequestError as e:
                print(f"Service Error: {e}")
                return "ERROR_SERVICE"
        except Exception as e:
            print(f"Listen Error: {e}")
            return "ERROR_MIC"

    def _normalize_text(self, text):
        normalized = unicodedata.normalize("NFKD", text)
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        return re.sub(r"[^a-z0-9\s]", " ", normalized.lower()).strip()

    def _select_microphone(self):
        if not HAS_SR:
            return None, "Micrófono no disponible"

        try:
            names = sr.Microphone.list_microphone_names()
        except Exception:
            return None, "Micrófono predeterminado"

        best_index = None
        best_score = -10**9

        for index, raw_name in enumerate(names):
            name = self._normalize_text(raw_name)
            score = 0

            if "mapper" in name or "asignador" in name:
                score -= 2
            if any(term in name for term in ["output", "altavoces", "speaker", "speakers", "display audio", "stereo", "mezcla", "lg fhd"]):
                score -= 10
            if any(term in name for term in ["microfono", "microfonos", "mic input", "microphone"]):
                score += 8
            if any(term in name for term in ["intel smart sound", "realtek", "digital microphone"]):
                score += 4
            if any(term in name for term in ["primario de captura", "input"]):
                score += 1

            if score > best_score:
                best_score = score
                best_index = index

        selected_name = names[best_index] if best_index is not None and best_index < len(names) else "Micrófono predeterminado"
        print(f"Selected microphone: index={best_index}, name={selected_name}")
        return best_index, selected_name

    def _open_microphone(self):
        if self.mic_index is None:
            return sr.Microphone()
        return sr.Microphone(device_index=self.mic_index)

    def list_microphones(self):
        if not HAS_SR:
            return []
        try:
            return list(enumerate(sr.Microphone.list_microphone_names()))
        except Exception:
            return []

    def speak(self, text, force=False):
        if self.muted:
            return
            
        if force:
            self.stop_current()
            with self.queue.mutex:
                self.queue.queue.clear()
        
        # Avoid adding duplicate consecutive messages if not forced
        if text == self.last_text and not force:
             if self.queue.empty() and self.current_process is None:
                 pass # Allow if it's been a while? For now, let's allow it but maybe limit frequency elsewhere
             else:
                 return # Don't stack same message

        self.last_text = text
        self.queue.put(text)

    def _process_queue(self):
        while self.is_running:
            try:
                text = self.queue.get(timeout=0.1)
                if text:
                    self._run_tts(text)
                    self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Voice Assistant Error: {e}")

    def _run_tts(self, text):
        if self.muted:
            return

        try:
            # Escape quotes for PowerShell
            safe_text = text.replace('"', '\\"').replace("'", "''")
            # PowerShell command using System.Speech
            ps_script = f"""
            Add-Type -AssemblyName System.Speech; 
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; 
            $synth.Rate = {self.speed}; 
            $synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female);
            $synth.Speak('{safe_text}');
            """
            
            # Run PowerShell
            self.current_process = subprocess.Popen(
                ["powershell", "-Command", ps_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.current_process.wait()
            self.current_process = None
            
        except Exception as e:
            print(f"TTS Error: {e}")

    def stop_current(self):
        if self.current_process:
            try:
                self.current_process.terminate()
            except:
                pass
            self.current_process = None

    def set_mute(self, mute):
        self.muted = mute
        if mute:
            self.stop_current()
            with self.queue.mutex:
                self.queue.queue.clear()

    def set_speed(self, speed):
        # Speed between -10 and 10
        self.speed = max(-10, min(10, speed))

    def repeat_last(self):
        if self.last_text:
            self.speak(self.last_text, force=True)
            
    def is_speaking(self):
        return self.current_process is not None or not self.queue.empty()

    def close(self):
        self.is_running = False
        self.stop_current()
