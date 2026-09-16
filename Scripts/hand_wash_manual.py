import tkinter as tk
from tkinter import ttk
import threading
import subprocess
import os
from PIL import Image, ImageTk

class HandWashHelp(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Manual de Ayuda - Técnica de Lavado de Manos OMS")
        self.geometry("500x700")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure("Help.TLabel", font=("Helvetica", 10))
        self.style.configure("HelpTitle.TLabel", font=("Helvetica", 14, "bold"), foreground="#003366")
        self.style.configure("HelpStep.TLabel", font=("Helvetica", 10))
        self.style.configure("HelpHighlight.TLabel", font=("Helvetica", 10, "bold"), foreground="#d9534f")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container with scrollbar
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Content
        self.add_content()
        
    def add_content(self):
        # Header
        header_frame = ttk.Frame(self.scrollable_frame)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(header_frame, text="Técnica de Lavado de Manos OMS", style="HelpTitle.TLabel").pack(anchor="center")
        ttk.Label(header_frame, text="Duración recomendada: 40-60 segundos", style="HelpHighlight.TLabel").pack(anchor="center", pady=(5, 0))
        
        # Image Display
        img_frame = ttk.Frame(self.scrollable_frame)
        img_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Try to locate image
        possible_paths = [
            os.path.join(os.getcwd(), "lavadomanos-24-6.jpg"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lavadomanos-24-6.jpg"),
            "lavadomanos-24-6.jpg"
        ]
        
        image_path = None
        for p in possible_paths:
            if os.path.exists(p):
                image_path = p
                break
        
        if image_path:
            try:
                pil_img = Image.open(image_path)
                # Resize to fit width 450
                base_width = 440
                w_percent = (base_width / float(pil_img.size[0]))
                h_size = int((float(pil_img.size[1]) * float(w_percent)))
                pil_img = pil_img.resize((base_width, h_size), Image.Resampling.LANCZOS)
                
                self.tk_img = ImageTk.PhotoImage(pil_img)
                
                lbl_img = ttk.Label(img_frame, image=self.tk_img)
                lbl_img.pack(anchor="center")
            except Exception as e:
                print(f"Error loading image: {e}")
                self._create_placeholder(img_frame, f"Error cargando imagen:\n{e}")
        else:
            self._create_placeholder(img_frame, "[Ilustración del Proceso OMS]\nSiga los 11 pasos descritos abajo")

        # Steps
        steps_frame = ttk.Frame(self.scrollable_frame)
        steps_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(steps_frame, text="Pasos detallados:", style="HelpTitle.TLabel", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        self.steps = [
            "1. Mójese las manos con agua limpia (tibia o fría).",
            "2. Deposite jabón suficiente en la palma para cubrir toda la superficie de las manos.",
            "3. Frótese las palmas de las manos entre sí. (INICIO EVALUACIÓN)",
            "4. Frótese la palma derecha contra el dorso izquierdo entrelazando los dedos, y viceversa.",
            "5. Frótese las palmas con los dedos entrelazados.",
            "6. Frótese el dorso de los dedos de una mano contra la palma opuesta, agarrando los dedos.",
            "7. Rodee el pulgar izquierdo con la palma derecha en movimiento rotatorio, y viceversa.",
            "8. Frótese las puntas de los dedos derechos contra la palma izquierda en rotación, y viceversa.",
            "9. Enjuáguese las manos con agua limpia.",
            "10. Séquese con toalla desechable de un solo uso.",
            "11. Use la toalla para cerrar el grifo sin tocarlo."
        ]
        
        for i, step in enumerate(self.steps):
            f = ttk.Frame(steps_frame)
            f.pack(fill=tk.X, pady=5)
            
            # Highlight step 3
            style = "HelpStep.TLabel"
            if i == 2: # Step 3 (index 2)
                style = "HelpHighlight.TLabel"
                
            lbl = ttk.Label(f, text=step, style=style, wraplength=400)
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            btn_speak = ttk.Button(f, text="🔊", width=3, command=lambda s=step: self.speak_text(s))
            btn_speak.pack(side=tk.RIGHT, padx=5)

        # Footer Notes
        note_frame = ttk.Frame(self.scrollable_frame)
        note_frame.pack(fill=tk.X, padx=20, pady=20)
        ttk.Label(note_frame, text="Indicaciones adicionales:\nDedique al menos 40-60 segundos al frotado total. Esta técnica aplica en contextos sanitarios y cotidianos.", 
                 style="Help.TLabel", wraplength=440).pack(anchor="w")

        # Global Audio
        btn_all = ttk.Button(self.scrollable_frame, text="Escuchar Todo el Proceso", command=self.speak_all)
        btn_all.pack(pady=10)

    def _create_placeholder(self, parent, text):
        canvas = tk.Canvas(parent, width=400, height=200, bg="#e6f3ff")
        canvas.pack(anchor="center")
        canvas.create_text(200, 100, text=text, justify="center", font=("Helvetica", 12))

    def speak_text(self, text):
        threading.Thread(target=self._run_tts, args=(text,), daemon=True).start()
        
    def speak_all(self):
        intro = "Ejecute estos pasos en orden para cubrir todas las superficies de las manos. "
        steps_text = " ".join(self.steps)
        full_text = intro + steps_text + " Dedique al menos 40 a 60 segundos al frotado total."
        self.speak_text(full_text)

    def _run_tts(self, text):
        # Using PowerShell for TTS on Windows
        try:
            # Escape quotes for PowerShell
            safe_text = text.replace('"', '\\"').replace("'", "''")
            cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe_text}')"
            subprocess.run(["powershell", "-Command", cmd], check=False)
        except Exception as e:
            print(f"TTS Error: {e}")

    def on_close(self):
        self.withdraw() # Hide instead of destroy to keep state if needed, or destroy.
        # If we destroy, we need to recreate it next time. Let's just withdraw.
