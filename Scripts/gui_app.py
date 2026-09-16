import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
import sys
import os
import datetime
import csv
import json
import re
import unicodedata

# Add current directory to path to allow imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hand_wash_analyzer import HandWashAnalyzer
    from hand_wash_manual import HandWashHelp
    from voice_assistant import VoiceAssistant
    from video_sync import VideoSynchronizer
except ImportError:
    # If running from root, try adding Scripts to path
    sys.path.append(os.path.join(os.getcwd(), 'Scripts'))
    from hand_wash_analyzer import HandWashAnalyzer
    from hand_wash_manual import HandWashHelp
    from voice_assistant import VoiceAssistant
    from video_sync import VideoSynchronizer

class HandWashApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Evaluación de Lavado de Manos - OMS")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Configuration
        self.steps_info = {
            0: ("Paso 3", "Frotar palmas (Palma con Palma)"),
            1: ("Paso 4", "Palma con dorso (Derecha sobre Izquierda y viceversa)"),
            2: ("Paso 5", "Palmas entrelazadas (Intercaladas)"),
            3: ("Paso 6", "Dorso de dedos (Nudillos)"),
            4: ("Paso 7", "Pulgar (Movimiento rotatorio)"),
            5: ("Paso 8", "Puntas de dedos (Pulpejos)")
        }
        
        self.analyzer = None
        self.cap = None
        self.is_running = False
        self.thread = None
        self.video_source = 0 # Default to webcam
        self.current_frame = None
        self.help_window = None
        
        # Initialize Voice Assistant
        self.voice = VoiceAssistant()
        
        # Voice Texts (Evaluation Mode)
        self.intro_text = (
            "Técnica de lavado de manos (con agua y jabón): "
            "1. Mojar: Abrir el grifo, humedecer las manos con agua. "
            "2. Aplicar jabón: Depositar suficiente jabón para cubrir toda la superficie"
        )
        self.start_process_text = "Iniciando proceso de lavado de manos"
        self.final_text = (
            "Enjuagar: Enjuagar bien las manos y muñecas, manteniéndolas hacia abajo. "
            "Secar: Secar con toalla de papel desechable. "
            "Cerrar grifo: Usar la misma toalla para cerrar la llave."
        )
        
        self.full_steps_desc = {
            0: "Paso 3: Frote las palmas de las manos entre sí.",
            1: "Paso 4: Frote la palma de la mano derecha contra el dorso de la mano izquierda entrelazando los dedos y viceversa.",
            2: "Paso 5: Frote las palmas de las manos entre sí, con los dedos entrelazados.",
            3: "Paso 6: Frote el dorso de los dedos de una mano con la palma de la mano opuesta, agarrándose los dedos.",
            4: "Paso 7: Frote con un movimiento de rotación el pulgar izquierdo, atrapándolo con la palma de la mano derecha y viceversa.",
            5: "Paso 8: Frote la punta de los dedos de la mano derecha contra la palma de la mano izquierda, haciendo un movimiento de rotación y viceversa."
        }
        
        # New State Variables for Requirements
        self.mode = "Práctica" # practice, evaluation, demo
        self.step_timer_start = 0
        self.step_duration_target = 10 # Default target
        self.min_step_duration = 6.0   # Minimum duration for Evaluation
        self.step_durations = {}       # Track time per step
        self.step_status = {}          # Track valid/invalid per step
        
        self.total_score = 100
        self.errors_log = []
        self.last_wash_time = datetime.datetime.now()
        self.reminder_interval = 60 * 60 # 1 hour default
        self.reminder_active = True
        self.waiting_for_hands = False
        
        # Voice Recognition State
        self.moments_voice_prompt_done = False
        self.voice_prompt_active = False
        self.voice_attempts = 0
        self.voice_max_attempts = 2
        
        self._setup_ui()
        
        # Start Reminder Check Loop
        self.check_reminders()
        
        # Initialize default analyzer
        self.load_analyzer("v2_35")
        
        self.synchronizer = VideoSynchronizer(self.analyzer)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.voice.close()
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def toggle_mute(self):
        current_text = self.btn_mute.cget("text")
        if current_text == "🔊":
            self.voice.set_mute(True)
            self.btn_mute.config(text="🔇")
        else:
            self.voice.set_mute(False)
            self.btn_mute.config(text="🔊")

    def change_speed(self, event=None):
        speed_map = {"Lento": -3, "Normal": 0, "Rápido": 3}
        val = self.speed_var.get()
        self.voice.set_speed(speed_map.get(val, 0))

    def announce_step(self, step_idx):
        if step_idx not in self.steps_info:
            return
            
        step_num, step_desc = self.steps_info[step_idx]
        
        if self.mode == "Práctica":
            text = f"{step_num}: {step_desc}"
        elif self.mode == "Evaluación":
            # Full instruction for Evaluation as requested
            text = self.full_steps_desc.get(step_idx, f"{step_num}: {step_desc}")
        elif self.mode == "Demostración":
            # Expanded instruction
            # We can hardcode detailed instructions or map them
            details = {
                0: "Frote las palmas de las manos entre sí vigorosamente.",
                1: "Coloque la palma derecha sobre el dorso izquierdo y entrelace los dedos.",
                2: "Frote palma contra palma con los dedos entrelazados.",
                3: "Apoye el dorso de los dedos contra la palma opuesta agarrándose los dedos.",
                4: "Rodee el pulgar izquierdo con la palma derecha y frote con movimiento de rotación.",
                5: "Frote la punta de los dedos contra la palma opuesta haciendo un movimiento de rotación."
            }
            detail = details.get(step_idx, "")
            text = f"{step_num}: {step_desc}. {detail}"
        else:
            text = f"{step_num}: {step_desc}"
            
        self.voice.speak(text)

    def _setup_ui(self):
        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 10))
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Step.TLabel", font=("Helvetica", 11), padding=5)
        style.configure("ActiveStep.TLabel", font=("Helvetica", 11, "bold"), background="#e6f3ff", foreground="#0066cc")
        style.configure("SuccessStep.TLabel", font=("Helvetica", 11), background="#e6ffe6", foreground="#006600")

        # Main Layout
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, text="Sistema de Evaluación de Lavado de Manos (OMS)", style="Header.TLabel").pack(side=tk.LEFT)
        
        # Controls Frame (Top Right)
        controls_frame = ttk.Frame(header_frame)
        controls_frame.pack(side=tk.RIGHT)
        
        # Voice Controls
        voice_frame = ttk.LabelFrame(controls_frame, text="Voz")
        voice_frame.pack(side=tk.LEFT, padx=5)
        
        self.btn_mute = ttk.Button(voice_frame, text="🔊", width=3, command=self.toggle_mute)
        self.btn_mute.pack(side=tk.LEFT, padx=2)
        
        self.btn_repeat = ttk.Button(voice_frame, text="↺", width=3, command=lambda: self.voice.repeat_last())
        self.btn_repeat.pack(side=tk.LEFT, padx=2)
        
        self.speed_var = tk.StringVar(value="Normal")
        speed_combo = ttk.Combobox(voice_frame, textvariable=self.speed_var, values=["Lento", "Normal", "Rápido"], state="readonly", width=6)
        speed_combo.pack(side=tk.LEFT, padx=2)
        speed_combo.bind("<<ComboboxSelected>>", self.change_speed)
        
        # Mode Selection
        ttk.Label(controls_frame, text="Modo:").pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="Práctica")
        mode_combo = ttk.Combobox(controls_frame, textvariable=self.mode_var, state="readonly", width=15)
        mode_combo['values'] = ["Práctica", "Evaluación", "Demostración"]
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)

        ttk.Label(controls_frame, text="Modelo:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="Versión 2 (35 timesteps) - Recomendado")
        model_combo = ttk.Combobox(controls_frame, textvariable=self.model_var, state="readonly", width=30)
        model_combo['values'] = [
            "Versión 2 (35 timesteps) - Recomendado",
            "Versión 2 (70 timesteps) - Secuencias largas",
            "Versión 1 (Legacy)"
        ]
        model_combo.pack(side=tk.LEFT, padx=5)
        model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        
        btn_source = ttk.Button(controls_frame, text="📷 Fuente", width=8, command=self.change_video_source)
        btn_source.pack(side=tk.LEFT, padx=2)
        
        self.btn_help = ttk.Button(controls_frame, text="📖 Ayuda", width=8, command=self.toggle_help)
        self.btn_help.pack(side=tk.LEFT, padx=2)
        
        self.btn_start = ttk.Button(controls_frame, text="▶ Iniciar", width=8, command=self.toggle_process)
        self.btn_start.pack(side=tk.LEFT, padx=2)
        
        self.btn_restart = ttk.Button(controls_frame, text="↻ Reiniciar", width=10, command=self.restart_process)
        self.btn_restart.pack(side=tk.LEFT, padx=2)
        
        self.auto_start_var = tk.BooleanVar(value=False)
        chk_auto = ttk.Checkbutton(controls_frame, text="Auto-Inicio", variable=self.auto_start_var)
        chk_auto.pack(side=tk.LEFT, padx=5)

        # Content Area
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: Video
        video_frame = ttk.LabelFrame(content_frame, text="Visualización en Tiempo Real")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Reduced size by ~30% (640*0.7=448, 480*0.7=336)
        self.canvas_video = tk.Canvas(video_frame, bg="black", width=448, height=336)
        self.canvas_video.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._show_canvas_message("Seleccione una fuente para previsualizar.")
        
        # Right: Steps & Status
        info_panel = ttk.Frame(content_frame, width=350)
        info_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status Box
        status_frame = ttk.LabelFrame(info_panel, text="Panel de Control")
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Feedback / Status
        self.lbl_status = ttk.Label(status_frame, text="Esperando inicio...", font=("Helvetica", 12, "bold"), wraplength=320)
        self.lbl_status.pack(fill=tk.X, padx=10, pady=10)
        
        # Timer
        timer_frame = ttk.Frame(status_frame)
        timer_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Total Time Display (New)
        ttk.Label(timer_frame, text="Tiempo Total:", font=("Helvetica", 10)).pack(anchor=tk.W)
        self.lbl_total_time = ttk.Label(timer_frame, text="00:00:00", font=("Helvetica", 14, "bold"))
        self.lbl_total_time.pack(anchor=tk.W, pady=(0, 5))

        ttk.Label(timer_frame, text="Tiempo Paso Actual:").pack(anchor=tk.W)
        self.lbl_timer = ttk.Label(timer_frame, text="00:00", font=("Helvetica", 14, "bold"), foreground="blue")
        self.lbl_timer.pack(anchor=tk.W)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=1.0)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Quality Indicators
        indicators_frame = ttk.Frame(status_frame)
        indicators_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.lbl_confidence = ttk.Label(indicators_frame, text="Calidad/Confianza: 0%")
        self.lbl_confidence.pack(anchor=tk.W)
        
        self.lbl_score = ttk.Label(indicators_frame, text="Puntaje: 100", font=("Helvetica", 10, "bold"))
        self.lbl_score.pack(anchor=tk.W, pady=(5, 0))

        self.lbl_model_used = ttk.Label(status_frame, text="Motor: -", font=("Helvetica", 8))
        self.lbl_model_used.pack(anchor=tk.W, padx=10, pady=(0, 10))

        # Five Moments Selection (New)
        moments_frame = ttk.LabelFrame(info_panel, text="5 Momentos (Higiene de Manos)")
        moments_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.moments_vars = []
        moments_data = [
            ("1. Antes del contacto con paciente", "Proteger al paciente de gérmenes en sus manos"),
            ("2. Antes de tarea aséptica", "Proteger al paciente de gérmenes dañinos"),
            ("3. Después de exposición a fluidos", "Protegerse a sí mismo y al entorno"),
            ("4. Después del contacto con paciente", "Protegerse y proteger el entorno"),
            ("5. Después del contacto con entorno", "Protegerse de gérmenes del paciente")
        ]
        
        for i, (text, tooltip) in enumerate(moments_data):
            var = tk.BooleanVar()
            # Trace changes to trigger video start
            var.trace_add("write", self.on_moment_checkbox_click)
            self.moments_vars.append(var)
            chk = ttk.Checkbutton(moments_frame, text=text, variable=var)
            chk.pack(anchor=tk.W, padx=5, pady=2)
            # Simple tooltip via bind (optional, or just rely on clear text)
            # Let's keep it clean.
            
        # Steps List
        steps_frame = ttk.LabelFrame(info_panel, text="Progreso del Protocolo")
        steps_frame.pack(fill=tk.BOTH, expand=True)
        
        self.step_labels = {}
        for i in range(6):
            step_num, step_desc = self.steps_info[i]
            frame = ttk.Frame(steps_frame)
            frame.pack(fill=tk.X, pady=2)
            
            lbl = ttk.Label(frame, text=f"{step_num}: {step_desc}", style="Step.TLabel", wraplength=300)
            lbl.pack(fill=tk.X, padx=5, pady=5)
            self.step_labels[i] = lbl

    def toggle_help(self):
        if self.help_window is None or not self.help_window.winfo_exists():
            self.help_window = HandWashHelp(self.root)
        else:
            self.help_window.deiconify()
            self.help_window.lift()

    def load_analyzer(self, config_key):
        self.lbl_status.config(text="Cargando modelos... Por favor espere.")
        self.root.update()
        
        try:
            if self.analyzer:
                del self.analyzer
            
            # Map friendly names to internal keys
            if "35 timesteps" in config_key: key = "v2_35"
            elif "70 timesteps" in config_key: key = "v2_70"
            elif "Legacy" in config_key: key = "v1"
            else: key = "v2_35"
            
            self.analyzer = HandWashAnalyzer(model_config=key)
            self.analyzer.load_models()
            
            if hasattr(self, 'synchronizer'):
                self.synchronizer.analyzer = self.analyzer
            
            self.lbl_status.config(text="Modelos cargados. Listo para iniciar.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelos: {str(e)}")
            self.lbl_status.config(text="Error de carga.")

    def on_model_change(self, event):
        selection = self.model_var.get()
        # Pause if running
        was_running = self.is_running
        if was_running:
            self.toggle_process()
            
        self.load_analyzer(selection)
        
        if was_running:
            self.toggle_process()

    def on_mode_change(self, event):
        self.mode = self.mode_var.get()
        self.lbl_status.config(text=f"Modo seleccionado: {self.mode}. Configure fuente y presione Iniciar.", foreground="black")
        
        # In Demo mode, we might suggest loading a video, but NOT start it.
        if self.mode == "Demostración":
            # Optional: Prompt for video file immediately if user wants, but don't start
            pass
        elif self.mode == "Evaluación":
             self.total_score = 100
             self.lbl_score.config(text=f"Puntaje: {self.total_score}")
             # Intro text can be spoken when "Start" is pressed, not here to avoid noise while configuring.

    def check_reminders(self):
        if self.reminder_active:
            now = datetime.datetime.now()
            diff = (now - self.last_wash_time).total_seconds()
            
            if diff > self.reminder_interval:
                # Alert!
                self.show_reminder_alert()
                self.last_wash_time = now # Reset to avoid spam, or snooze
        
        self.root.after(60000, self.check_reminders) # Check every minute

    def show_reminder_alert(self):
        # Create a non-blocking top level window or just a message
        # Since we want it to be "non-invasive", a subtle popup or sound is better.
        # But for now, a Toplevel that doesn't steal focus too aggressively.
        alert = tk.Toplevel(self.root)
        alert.title("Recordatorio de Higiene")
        alert.geometry("400x150")
        ttk.Label(alert, text="🔔 Recordatorio de Lavado de Manos", font=("Helvetica", 12, "bold"), foreground="red").pack(pady=20)
        ttk.Label(alert, text="Ha pasado 1 hora desde su último lavado.\nPor favor, proceda a la estación de lavado.").pack()
        ttk.Button(alert, text="Entendido", command=alert.destroy).pack(pady=10)
        
        # Optional: Play sound
        try:
            self.root.bell()
        except: pass

    def change_video_source(self):
        use_camera = messagebox.askyesno("Fuente de Video", "¿Desea usar la cámara web?\n\nSí: Cámara Web\nNo: Seleccionar Archivo de Video")
        
        new_source = None
        if use_camera:
            new_source = 0
        else:
            filename = filedialog.askopenfilename(title="Seleccionar Video", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
            if filename:
                new_source = filename
        
        if new_source is not None:
            self.video_source = new_source
            
            # If currently running, stop it to apply new source cleanly on next start
            if self.is_running:
                self.toggle_process() # Stops it

            self.preview_selected_source()
            self.lbl_status.config(text=f"Fuente configurada: {'Cámara' if new_source == 0 else 'Video'}. Presione Iniciar.", foreground="black")

    def preview_selected_source(self):
        if self.video_source == 0:
            self._show_canvas_message("Camara seleccionada.\nPresione Iniciar para activar la vista en tiempo real.")
            return

        cap_preview = cv2.VideoCapture(self.video_source)
        if not cap_preview.isOpened():
            self._show_canvas_message("No se pudo abrir el video seleccionado.")
            return

        ret, frame = cap_preview.read()
        cap_preview.release()

        if not ret or frame is None:
            self._show_canvas_message("No se pudo leer el primer fotograma del video.")
            return

        self._display_frame_on_canvas(frame)

    def _display_frame_on_canvas(self, frame):
        canvas_width = self.canvas_video.winfo_width()
        canvas_height = self.canvas_video.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 448
            canvas_height = 336

        frame_resized = cv2.resize(frame, (canvas_width, canvas_height))
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.canvas_video.delete("all")
        self.canvas_video.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.current_image_ref = img_tk

    def _show_canvas_message(self, text):
        self.canvas_video.delete("all")
        self.canvas_video.create_text(
            224,
            168,
            text=text,
            fill="white",
            font=("Helvetica", 12, "bold"),
            justify="center",
            width=380,
        )

    def restart_process(self):
        # If running, just reset state without stopping video (smoother)
        if self.is_running:
            if self.analyzer:
                self.analyzer.reset_state()
            self._reset_steps_ui()
            self.lbl_status.config(text="Reiniciado")
        else:
            # If stopped, trigger start logic (which now includes Moment Selection)
            self.toggle_process()
            
        # Reset Session Variables
        self.step_timer_start = time.time()
        self.session_start_time = time.time()
        self.total_score = 100
        self.errors_log = []
        self.lbl_score.config(text=f"Puntaje: {self.total_score}")
        self.lbl_timer.config(text="00:00")
        self.lbl_total_time.config(text="00:00:00")
        self.last_step_idx = -1
        
        # Reset Voice State
        self.moments_voice_prompt_done = False
        self.voice_prompt_active = False
        self.voice_attempts = 0
        
        # Clear Moments Selection (Requirement: Reset on new evaluation)
        for var in self.moments_vars:
            var.set(False)
        
        # Reset tracking
        self.step_durations = {}
        self.step_status = {}

    def toggle_process(self):
        if self.is_running:
            self.is_running = False
            self.btn_start.config(text="▶ Iniciar")
            self.lbl_status.config(text="Pausado")
            self.waiting_for_hands = False
            
            # Stop voice on pause
            self.voice.stop_current()
        else:
            if self.analyzer is None:
                messagebox.showwarning("Advertencia", "Los modelos no están cargados.")
                return
            
            # Ensure a source is selected (default 0 is always there, but check for files)
            if isinstance(self.video_source, str) and not os.path.exists(self.video_source):
                messagebox.showerror("Error", "El archivo de video seleccionado no existe.")
                return

            # Special checks for modes before starting
            if self.mode == "Evaluación":
                # Maybe play intro here now
                pass
            
            # Requirement: Do NOT start video automatically.
            # Start Moment Selection Flow instead.
            self.initiate_moment_selection()

    def initiate_moment_selection(self):
        # Check if already selected (unlikely due to reset, but safe check)
        # But per requirements we should force re-selection or at least prompt.
        
        # Clear previous selection to ensure explicit interaction
        for var in self.moments_vars:
            var.set(False)
            
        self.lbl_status.config(text="Selección de Momento Requerida", foreground="blue")
        self.start_voice_moment_sequence()

    def on_moment_checkbox_click(self, *args):
        # Callback for manual interaction
        # If any checkbox is set to True AND video is not running, start video.
        if not self.is_running:
             any_selected = any(var.get() for var in self.moments_vars)
             if any_selected:
                 # Check if we are in a valid state to start (e.g., after voice prompt or manual override)
                 # Requirement: "El inicio... se detona exclusivamente cuando el usuario selecciona..."
                 self.voice.stop_current() # Stop any ongoing voice prompt
                 self.voice_prompt_active = False # Stop listening thread if active
                 self.start_video_playback()
                 self.lbl_status.config(text="Momento seleccionado. Iniciando...", foreground="green")

    def start_calibration_async(self):
        self.lbl_status.config(text="Calibrando sincronización de video (0%)...", foreground="blue")
        self.btn_start.config(state="disabled") # Disable start button during calibration
        
        # Threaded calibration
        def run_calibration():
            try:
                success = self.synchronizer.analyze_video(
                    self.video_source, 
                    progress_callback=self.update_calibration_progress
                )
                self.root.after(0, lambda: self.on_calibration_complete(success))
            except Exception as e:
                print(f"Calibration Error: {e}")
                self.root.after(0, lambda: self.on_calibration_complete(False))
                
        threading.Thread(target=run_calibration, daemon=True).start()

    def update_calibration_progress(self, progress):
        # Called from thread, schedule UI update
        self.root.after(0, lambda: self.lbl_status.config(text=f"Calibrando sincronización de video ({int(progress*100)}%)..."))

    def on_calibration_complete(self, success):
        self.btn_start.config(state="normal")
        if success:
            self.lbl_status.config(text="Calibración completada.")
            self.start_video_playback()
        else:
            self.lbl_status.config(text="Fallo en calibración (usando por defecto).")
            # Should we start anyway? Yes, let's try.
            self.start_video_playback()

    def start_video_playback(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la fuente de video.")
            return
            
        self.is_running = True
        self.btn_start.config(text="⏹ Detener")
        self.analyzer.reset_state()
        self._reset_steps_ui()
        
        # Auto-Start Logic
        if self.auto_start_var.get():
            self.waiting_for_hands = True
            self.lbl_status.config(text="Esperando manos... (Modo Sin Contacto)", foreground="blue")
        else:
            self.waiting_for_hands = False
            # Manual Start Trigger for Evaluation
            if self.mode == "Evaluación":
                self.voice.speak(self.start_process_text, force=True)
        
        # Start loop
        self.update_video_loop()

    def _reset_steps_ui(self):
        for i in range(6):
            self.step_labels[i].configure(style="Step.TLabel")

    def update_video_loop(self):
        if not self.is_running:
            if self.cap:
                self.cap.release()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.is_running = False
            self.btn_start.config(text="Iniciar (Reiniciar)")
            self.lbl_status.config(text="Video finalizado")
            if self.cap: self.cap.release()
            return

        # Synchronization Logic (Demo Mode)
        loop_delay = 10
        if self.mode == "Demostración" and isinstance(self.video_source, str):
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            # Estimate progress based on frames_validated
            req = getattr(self.analyzer, 'seq_len_req', 30.0)
            proto_prog = min(1.0, self.analyzer.frames_validated / float(req))
            
            if self.analyzer.current_step_idx == 2: # YOLO step
                 req_yolo = getattr(self.analyzer, 'yolo_peaks_req', 5.0)
                 proto_prog = min(1.0, self.analyzer.yolo_peak_counter / float(req_yolo))
                 
            # Force progress bar to match video if synchronized
            target_step = self.synchronizer.get_current_step_from_time(timestamp)
            if target_step == self.analyzer.current_step_idx and target_step != -1:
                 # Override visual progress with video progress for smoothness
                 start_ms, end_ms = self.synchronizer.segments[target_step]
                 video_prog = (timestamp - start_ms) / (end_ms - start_ms + 1e-6)
                 video_prog = max(0.0, min(1.0, video_prog))
                 # Blend protocol progress with video progress
                 # Actually, just show video progress as it is the "Demo"
                 # But we need to update analyzer state so it doesn't look weird?
                 # No, just let the UI use 'proto_prog' but we can influence 'proto_prog' logic?
                 # Let's trust sync_step to force 'force_next_step'
                 pass

            action, stride_adj = self.synchronizer.sync_step(
                timestamp, 
                self.analyzer.current_step_idx, 
                proto_prog
            )
            
            self.analyzer.stride = int(stride_adj)
            
            if action == 'pause':
                # Video is ahead, slow down significantly to wait for protocol
                loop_delay = 200 
            elif action == 'force_next_step':
                # Critical: Video has moved on, force protocol to follow
                # We artificially complete the current step
                self.analyzer.current_step_idx += 1
                self.analyzer.frames_validated = 0
                self.analyzer.yolo_peak_counter = 0
                self.analyzer.yolo_in_peak = False
                loop_delay = 1 # Keep going fast
            elif action == 'slow_video':
                loop_delay = 50
            elif action == 'fast_video':
                # Video is behind, skip frames to catch up
                for _ in range(2): 
                    self.cap.read()
                loop_delay = 1
            elif action == 'fast_forward':
                # Significant lag, skip more
                for _ in range(5):
                    self.cap.read()
                loop_delay = 1

        # Resize for display efficiency (keep aspect ratio if possible, but fit canvas)
        # Dynamic resizing to fill container (eliminating dead space)
        canvas_width = self.canvas_video.winfo_width()
        canvas_height = self.canvas_video.winfo_height()
        
        # Ensure valid dimensions (during startup or minimization)
        if canvas_width < 10 or canvas_height < 10:
             canvas_width = 448
             canvas_height = 336
             
        frame_resized = cv2.resize(frame, (canvas_width, canvas_height))
        
        # Process Frame
        # We need an index counter for the analyzer logic (skipping frames, etc)
        if not hasattr(self, 'frame_cnt'): self.frame_cnt = 0
        self.frame_cnt += 1
        
        # Effective FPS estimation (could be better)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        try:
            results = self.analyzer.process_frame(frame_resized, self.frame_cnt, effective_fps=fps)
            
            # FORCE SYNC IN DEMO MODE: Override progress with Video Time
            if self.mode == "Demostración" and isinstance(self.video_source, str):
                 timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                 target_step = self.synchronizer.get_current_step_from_time(timestamp)
                 
                 # Only override if we are in the correct step
                 if target_step == self.analyzer.current_step_idx and target_step != -1:
                     if target_step in self.synchronizer.segments:
                         start_ms, end_ms = self.synchronizer.segments[target_step]
                         video_prog = (timestamp - start_ms) / (end_ms - start_ms + 1e-6)
                         video_prog = max(0.0, min(1.0, video_prog))
                         
                         results["progress"] = video_prog
                         
                         # Sync internal counters to match visual progress
                         # This prevents "jumps" if logic switches back to counters
                         req = getattr(self.analyzer, 'seq_len_req', 30.0)
                         self.analyzer.frames_validated = int(video_prog * req)
                         
                         if target_step == 2: # YOLO
                             req_yolo = getattr(self.analyzer, 'yolo_peaks_req', 5.0)
                             self.analyzer.yolo_peak_counter = int(video_prog * req_yolo)

            if self.waiting_for_hands:
                # Check for presence (confidence > 0.5 is a proxy)
                # Or check if any hand is detected. 
                # Analyzer returns confidence of the *step*, but we want presence.
                # However, if confidence is high, it means hands are doing something valid.
                if results["confidence"] > 0.5:
                    self.waiting_for_hands = False
                    
                    # Auto Start Trigger for Evaluation
                    if self.mode == "Evaluación":
                        self.voice.speak(self.start_process_text, force=True)
                        
                    self.analyzer.reset_state()
                    self.restart_process() # This resets timers and scores
                    # We continue to process this frame or next? Next is fine.
                else:
                    self.lbl_status.config(text="Esperando manos para iniciar...", foreground="blue")
            else:
                self.update_ui_with_results(results)
                # Draw overlay on frame
                self.draw_overlay(frame_resized, results)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        # Convert to TK format
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.canvas_video.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.current_image_ref = img_tk # Keep reference
        
        self.root.after(loop_delay, self.update_video_loop)

    def update_ui_with_results(self, results):
        current_step = results["current_step"]
        wash_complete = results["wash_complete"]
        progress = results["progress"]
        confidence = results["confidence"]
        model_used = results["model_used"]
        missing_side = results.get("missing_side", None)
        status_color = results.get("status_color", (0, 0, 0)) # BGR
        
        self.progress_var.set(progress)
        self.lbl_confidence.config(text=f"Calidad/Confianza: {confidence:.2f}")
        if confidence > 0.7:
            self.lbl_confidence.config(foreground="green")
        elif confidence > 0.4:
            self.lbl_confidence.config(foreground="orange")
        else:
            self.lbl_confidence.config(foreground="red")

        self.lbl_model_used.config(text=f"Motor: {model_used}")
        
        # Timer Logic
        if not hasattr(self, 'last_step_idx'): self.last_step_idx = -1
        if not hasattr(self, 'session_saved'): self.session_saved = False
        
        # Calculate elapsed for current step so far
        elapsed = 0
        if self.step_timer_start > 0:
            elapsed = time.time() - self.step_timer_start

        if current_step != self.last_step_idx:
            # Step Transition occurred
            if self.last_step_idx != -1 and self.step_timer_start > 0:
                 # Record duration of the finished step
                 final_duration = elapsed
                 self.step_durations[self.last_step_idx] = final_duration
                 
                 # Validate duration in Evaluation Mode
                 if self.mode == "Evaluación":
                     if final_duration < self.min_step_duration:
                         self.step_status[self.last_step_idx] = "Insuficiente"
                         self.total_score = max(0, self.total_score - 10)
                         self.lbl_score.config(text=f"Puntaje: {int(self.total_score)}")
                         self.voice.speak("Muy rápido. Debe frotar por 6 segundos.", force=True)
                         
                         # Visual mark of failure (optional, maybe in the list)
                         if self.last_step_idx in self.step_labels:
                             self.step_labels[self.last_step_idx].configure(style="Step.TLabel", foreground="red")
                     else:
                         self.step_status[self.last_step_idx] = "Correcto"
            
            # Start new step timer
            self.step_timer_start = time.time()
            elapsed = 0 # Reset for new step
            self.last_step_idx = current_step
            # Reset feedback color if step changes
            self.lbl_timer.config(foreground="blue")
            
            # Announce step
            self.announce_step(current_step)
            
        # Update Timer Display
        mins, secs = divmod(int(elapsed), 60)
        
        # Update Total Time
        total_elapsed = 0
        if hasattr(self, 'session_start_time') and self.session_start_time > 0:
             total_elapsed = time.time() - self.session_start_time
        
        t_hours, remainder = divmod(int(total_elapsed), 3600)
        t_mins, t_secs = divmod(remainder, 60)
        self.lbl_total_time.config(text=f"{t_hours:02d}:{t_mins:02d}:{t_secs:02d}")
        
        if self.mode == "Evaluación":
            # Countdown or Progress for 6s
            remaining = max(0, self.min_step_duration - elapsed)
            if remaining > 0:
                 self.lbl_timer.config(text=f"Mínimo: {int(remaining)}s", foreground="orange")
            else:
                 self.lbl_timer.config(text=f"¡Cumplido! ({int(elapsed)}s)", foreground="green")
        else:
            self.lbl_timer.config(text=f"{mins:02d}:{secs:02d}")
        
        # Scoring Logic (Evaluation Mode)
        if self.mode == "Evaluación" and not wash_complete:
            if confidence < 0.4:
                self.total_score = max(0, self.total_score - 0.1) # Decrease slowly
            self.lbl_score.config(text=f"Puntaje: {int(self.total_score)}")

        if wash_complete:
            if not self.session_saved:
                # RECORD FINAL STEP DURATION (Step 8 / Index 5)
                # Since we don't transition OUT of the last step, we must capture it here.
                # 'current_step' should be 5 here.
                if current_step == 5 and self.step_timer_start > 0:
                     elapsed = time.time() - self.step_timer_start
                     self.step_durations[5] = elapsed
                     
                     if self.mode == "Evaluación":
                         if elapsed < self.min_step_duration:
                             self.step_status[5] = "Insuficiente"
                             self.total_score = max(0, self.total_score - 10)
                             self.lbl_score.config(text=f"Puntaje: {int(self.total_score)}")
                             # No voice needed as we are done, but status is recorded
                         else:
                             self.step_status[5] = "Correcto"
                
                self.lbl_status.config(text="¡LAVADO COMPLETADO!", foreground="green")
                
                if self.mode == "Evaluación":
                    self.voice.speak(self.final_text, force=True)
                else:
                    self.voice.speak("Lavado completado. Excelente trabajo.")
                    
                self._reset_steps_ui()
                for i in range(6):
                     self.step_labels[i].configure(style="SuccessStep.TLabel")
                
                self.save_wash_session()
                self.session_saved = True
                self.last_wash_time = datetime.datetime.now() # Update for reminder
        else:
            self.session_saved = False # Reset flag if we restart or are not complete
            
            step_name = self.steps_info[current_step][0]
            step_desc = self.steps_info[current_step][1]
            
            # Detailed Feedback
            if confidence < 0.4:
                feedback_text = f"⚠ Mejorar postura en {step_name}"
                color = "red"
            elif missing_side:
                # Feedback for bilateral check
                if missing_side == 'A':
                    feedback_text = f"⚠ Falta frotar lado A (Derecha/Izq)"
                else:
                    feedback_text = f"⚠ Falta frotar lado B (Cambie de mano)"
                color = "orange"
                
                # Voice prompt if stuck
                if self.mode == "Evaluación" and time.time() % 5 < 0.5: # Periodic check to avoid spam? 
                    # Actually, better to do it once. But for now visual is key.
                    # We can use a simple throttle
                    pass
            else:
                feedback_text = f"En progreso: {step_name}"
                color = "black"
                
            self.lbl_status.config(text=f"{feedback_text}\n{step_desc}", foreground=color)
            
            # Update list styles
            for i in range(6):
                if i < current_step:
                    self.step_labels[i].configure(style="SuccessStep.TLabel")
                elif i == current_step:
                    self.step_labels[i].configure(style="ActiveStep.TLabel")
                else:
                    self.step_labels[i].configure(style="Step.TLabel")

    def save_wash_session(self):
        # Save results to CSV
        log_file = "historial_lavado.csv"
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Fecha", "Hora", "Modo", "Puntaje", "Duracion_Total", "Detalle_Pasos", "Momentos_Seleccionados"])
            
            now = datetime.datetime.now()
            duration = int(time.time() - getattr(self, 'session_start_time', time.time())) 
            
            # Format detailed step info
            details = []
            if self.mode == "Evaluación":
                for i in range(6):
                    dur = self.step_durations.get(i, 0)
                    stat = self.step_status.get(i, "N/A")
                    details.append(f"P{i+3}:{int(dur)}s({stat})")
            
            details_str = " | ".join(details)
            
            # Get Selected Moments
            selected_moments = []
            for i, var in enumerate(self.moments_vars):
                if var.get():
                    selected_moments.append(str(i+1))
            moments_str = ",".join(selected_moments) if selected_moments else "Ninguno"
            
            writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), self.mode, int(self.total_score), duration, details_str, moments_str])
            
        if self.mode == "Evaluación":
            # Detailed report in message box
            report = f"Puntaje Final: {int(self.total_score)}/100\n\n"
            
            # Add Moments to Report
            if selected_moments:
                report += "Momentos Seleccionados:\n"
                # Use the original text from moments_data (reconstructed or stored)
                # Since we stored indices in CSV, we can map back or just check vars
                # Let's map back for cleaner display
                moments_titles = [
                    "1. Antes del contacto con paciente",
                    "2. Antes de tarea aséptica",
                    "3. Después de exposición a fluidos",
                    "4. Después del contacto con paciente",
                    "5. Después del contacto con entorno"
                ]
                for idx_str in selected_moments:
                    idx = int(idx_str) - 1
                    if 0 <= idx < len(moments_titles):
                        report += f"- {moments_titles[idx]}\n"
                report += "\n"
            else:
                 report += "Momentos: Ninguno seleccionado\n\n"

            report += "Detalle por Paso (Min 6s):\n"
            for i in range(6):
                step_name = self.steps_info[i][0]
                dur = self.step_durations.get(i, 0)
                stat = self.step_status.get(i, "N/A")
                icon = "✅" if stat == "Correcto" else "❌"
                report += f"{step_name}: {int(dur)}s - {stat} {icon}\n"
                
            messagebox.showinfo("Resultados de Evaluación", report)
        else:
            # Just a small notification or nothing
            pass

    def draw_overlay(self, img, results):
        # Draw some simple debug info on the video itself if needed
        # Or just rely on the side panel
        # Let's draw the hand landmarks if we want, but the analyzer does not return them explicitly
        # We can draw the bounding box for YOLO if it detected something
        pass

    def start_voice_moment_sequence(self):
        self.voice_prompt_active = True
        self.lbl_status.config(text="Iniciando reconocimiento de voz...", foreground="blue")
        
        # Start sequence thread
        threading.Thread(target=self._run_voice_sequence, daemon=True).start()

    def _run_voice_sequence(self):
        # 1. Calibration
        self.root.after(0, lambda: self.lbl_status.config(text="Calibrando ruido ambiente (Silencio)...", foreground="orange"))
        self.voice.calibrate(duration=1.0)
        
        # 2. Instruction
        instruction = "Menciona el momento de lavado de manos del 1 al 5"
        self.voice.speak(instruction, force=True)
        
        # Wait for speech to finish
        time.sleep(0.5) 
        while self.voice.is_speaking():
            time.sleep(0.1)
            
        # 3. Listen
        self._listen_loop()

    def _listen_loop(self):
        # Update UI to indicate listening
        self.root.after(0, lambda: self.lbl_status.config(text="🎤 ESCUCHANDO... (Diga 1, 2, 3...)", foreground="red"))
        
        # Listen without re-calibrating (uses previous threshold)
        text = self.voice.listen(timeout=5, calibrate=False)
        
        self.root.after(0, lambda: self._process_voice_result(text))

    def _listen_moment_thread(self):
        # Retry Logic: Wait for TTS to finish before listening again
        time.sleep(0.5)
        while self.voice.is_speaking():
            time.sleep(0.1)
        self._listen_loop()

    def _process_voice_result(self, text):
        if not self.voice_prompt_active: return

        detected_num = None
        normalized_text = ""
        # Check for error codes first
        if text in ["NO_SPEECH", "UNKNOWN", "ERROR_MIC", "ERROR_SERVICE", "ERROR_IMPORT"]:
            error_code = text
        else:
            normalized_text = self._normalize_voice_text(text)
            detected_num = self._extract_moment_number(normalized_text)
        
        if detected_num:
            # Valid
            idx = detected_num - 1
            if 0 <= idx < len(self.moments_vars):
                # Trigger the checkbox (which triggers callback -> starts video)
                self.moments_vars[idx].set(True)
                
                self.voice.speak(f"Momento {detected_num} seleccionado.", force=True)
                self.moments_voice_prompt_done = True
                self.voice_prompt_active = False
                self.lbl_status.config(
                    text=f"Reconocido: '{normalized_text or text}'. Momento {detected_num} seleccionado.",
                    foreground="green"
                )
                # No need to manually call start_video_playback, callback handles it
        else:
            # Invalid or Error
            self.voice_attempts += 1
            
            # Determine feedback message
            if text == "NO_SPEECH":
                msg = "No te escuché."
            elif text == "UNKNOWN":
                msg = "No entendí."
            elif text == "ERROR_MIC":
                msg = "Error de micrófono."
            elif text == "ERROR_SERVICE":
                msg = "Error del servicio de reconocimiento."
            elif text == "ERROR_IMPORT":
                msg = "Reconocimiento de voz no disponible."
            else:
                msg = "Valor no válido."

            if self.voice_attempts >= self.voice_max_attempts:
                final_msg = f"{msg} Seleccione manualmente."
                self.voice.speak(final_msg, force=True)
                self.moments_voice_prompt_done = True # Stop blocking
                self.voice_prompt_active = False
                detail = f" Texto reconocido: '{normalized_text or text}'." if text and text not in ["NO_SPEECH", "UNKNOWN", "ERROR_MIC", "ERROR_SERVICE", "ERROR_IMPORT"] else ""
                self.lbl_status.config(text=f"Selección manual requerida.{detail}", foreground="orange")
            else:
                retry_msg = f"{msg} Repite el número (1 al 5)."
                self.voice.speak(retry_msg, force=True)
                detail = f" Texto reconocido: '{normalized_text or text}'." if text and text not in ["NO_SPEECH", "UNKNOWN", "ERROR_MIC", "ERROR_SERVICE", "ERROR_IMPORT"] else ""
                self.lbl_status.config(text=f"{msg}{detail}", foreground="orange")
                # Retry
                threading.Thread(target=self._listen_moment_thread, daemon=True).start()

    def _normalize_voice_text(self, text):
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def _extract_moment_number(self, normalized_text):
        if not normalized_text:
            return None

        if re.search(r"\b[1-5]\b", normalized_text):
            return int(re.search(r"\b([1-5])\b", normalized_text).group(1))

        number_patterns = {
            1: [r"\buno\b", r"\buna\b", r"\bun\b", r"\bprimero\b"],
            2: [r"\bdos\b", r"\bsegundo\b"],
            3: [r"\btres\b", r"\btercero\b"],
            4: [r"\bcuatro\b", r"\bcuarto\b"],
            5: [r"\bcinco\b", r"\bquinto\b"],
        }

        for number, patterns in number_patterns.items():
            for pattern in patterns:
                if re.search(pattern, normalized_text):
                    return number

        compact_text = normalized_text.replace(" ", "")
        compact_patterns = {
            1: ["uno", "unoo", "unon", "una"],
            2: ["dos"],
            3: ["tres"],
            4: ["cuatro"],
            5: ["cinco"],
        }

        for number, patterns in compact_patterns.items():
            if any(token in compact_text for token in patterns):
                return number

        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = HandWashApp(root)
    root.mainloop()
