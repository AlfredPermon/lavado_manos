#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluador de Lavado de Manos - Protocolo OMS
Interfaz Profesional Minimalista con Tkinter
Autor: Experto en Epidemiología Clínica
"""

import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import json
import os
from datetime import datetime

class HandwashEvaluatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Evaluador de Lavado de Manos - OMS")
        self.root.geometry("900x700")
        self.root.minsize(600, 500)
        
        # Configuración de estilo moderno
        self.setup_styles()
        
        # Variables de estado
        self.selected_steps = set()
        self.has_water = tk.BooleanVar(value=False)
        self.has_soap = tk.BooleanVar(value=False)
        self.has_drying = tk.BooleanVar(value=False)
        
        # Construcción de la UI
        self.create_widgets()
        
    def setup_styles(self):
        """Configura fuentes y colores para un look moderno"""
        self.colors = {
            'bg': '#F5F7FA',
            'card_bg': '#FFFFFF',
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'accent': '#E74C3C',
            'success': '#27AE60',
            'text': '#2C3E50',
            'text_light': '#7F8C8D',
            'border': '#ECF0F1'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Fuentes
        self.font_title = tkfont.Font(family="Segoe UI", size=18, weight="bold")
        self.font_subtitle = tkfont.Font(family="Segoe UI", size=12, weight="normal")
        self.font_body = tkfont.Font(family="Segoe UI", size=11)
        self.font_button = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.font_mono = tkfont.Font(family="Consolas", size=10)
        
        # Estilo para Checkbuttons personalizados
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure("TCheckbutton", 
                            background=self.colors['card_bg'],
                            foreground=self.colors['text'],
                            font=self.font_body)
        self.style.map("TCheckbutton",
                      background=[('active', self.colors['card_bg'])])
                      
        self.style.configure("Title.TLabel", 
                            background=self.colors['bg'], 
                            foreground=self.colors['primary'],
                            font=self.font_title)
                            
        self.style.configure("Subtitle.TLabel", 
                            background=self.colors['bg'], 
                            foreground=self.colors['text_light'],
                            font=self.font_subtitle)
                            
        self.style.configure("Card.TFrame",
                            background=self.colors['card_bg'])

    def create_widgets(self):
        """Construye la interfaz principal"""
        # Contenedor principal con scroll si es necesario (opcional para pantallas pequeñas)
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        lbl_title = tk.Label(header_frame, text="Evaluación de Higiene de Manos", 
                            style="Title.TLabel")
        lbl_title.pack(anchor='w')
        
        lbl_subtitle = tk.Label(header_frame, text="Protocolo OMS: Duración ideal 40-60s", 
                               style="Subtitle.TLabel")
        lbl_subtitle.pack(anchor='w')
        
        # Contenedor de tarjetas (Grid responsivo)
        content_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tarjeta 1: Pasos Críticos
        steps_card = tk.LabelFrame(content_frame, text="Pasos Críticos de Fricción", 
                                  font=self.font_subtitle, fg=self.colors['text_light'],
                                  bg=self.colors['card_bg'], padx=20, pady=20)
        steps_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        
        self.steps_list = [
            (1, "Palmas entre sí"),
            (2, "Palma derecha contra dorso izquierdo (y viceversa)"),
            (3, "Palmas entre sí con dedos entrelazados"),
            (4, "Dorso de los dedos contra palma opuesta"),
            (5, "Fricción rotacional del pulgar"),
            (6, "Fricción rotacional de la punta de los dedos/uñas")
        ]
        
        self.step_vars = {}
        for i, (step_num, desc) in enumerate(self.steps_list):
            var = tk.BooleanVar()
            self.step_vars[step_num] = var
            
            chk = tk.Checkbutton(steps_card, 
                                text=f"Paso {step_num}: {desc}",
                                variable=var,
                                onvalue=True, offvalue=False,
                                bg=self.colors['card_bg'],
                                selectcolor=self.colors['bg'],
                                activebackground=self.colors['card_bg'],
                                activeforeground=self.colors['primary'],
                                font=self.font_body,
                                command=self.update_status)
            chk.pack(anchor='w', pady=5)
            
        # Tarjeta 2: Insumos y Secado
        supplies_card = tk.LabelFrame(content_frame, text="Insumos y Secado", 
                                     font=self.font_subtitle, fg=self.colors['text_light'],
                                     bg=self.colors['card_bg'], padx=20, pady=20)
        supplies_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))
        
        supplies = [
            ("Uso de Agua", self.has_water),
            ("Uso de Jabón", self.has_soap),
            ("Secado con Toalla Desechable", self.has_drying)
        ]
        
        for text, var in supplies:
            chk = tk.Checkbutton(supplies_card,
                                text=text,
                                variable=var,
                                bg=self.colors['card_bg'],
                                selectcolor=self.colors['bg'],
                                activebackground=self.colors['card_bg'],
                                font=self.font_body)
            chk.pack(anchor='w', pady=8)
            
        # Configurar grid weights para responsividad
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Panel de Resultados (Abajo)
        result_frame = tk.Frame(main_frame, bg=self.colors['card_bg'], padx=20, pady=20)
        result_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        
        # Frame interno para resultados
        inner_result = tk.Frame(result_frame, bg=self.colors['card_bg'])
        inner_result.pack(fill=tk.X)
        
        self.lbl_status = tk.Label(inner_result, text="Estado: Pendiente", 
                                  font=self.font_subtitle, bg=self.colors['card_bg'],
                                  fg=self.colors['text_light'])
        self.lbl_status.pack(anchor='w')
        
        self.lbl_efficiency = tk.Label(inner_result, text="Eficiencia: 0%", 
                                      font=self.font_title, bg=self.colors['card_bg'],
                                      fg=self.colors['primary'])
        self.lbl_efficiency.pack(anchor='w', pady=(5, 15))
        
        # Botones de acción
        btn_frame = tk.Frame(inner_result, bg=self.colors['card_bg'])
        btn_frame.pack(fill=tk.X)
        
        btn_eval = tk.Button(btn_frame, text="Generar Evaluación", 
                            command=self.generate_evaluation,
                            bg=self.colors['secondary'], fg='white',
                            font=self.font_button, bd=0, padx=20, pady=10,
                            activebackground='#2980B9', cursor="hand2")
        btn_eval.pack(side=tk.LEFT, padx=(0, 10))
        
        btn_copy = tk.Button(btn_frame, text="Copiar JSON", 
                            command=self.copy_to_clipboard,
                            bg=self.colors['primary'], fg='white',
                            font=self.font_button, bd=0, padx=20, pady=10,
                            activebackground='#34495E', cursor="hand2")
        btn_copy.pack(side=tk.LEFT)
        
        # Área de texto para JSON (oculta inicialmente o pequeña)
        self.txt_json = tk.Text(result_frame, height=6, font=self.font_mono, 
                               bg='#F8F9F9', fg='#2C3E50', bd=0, padx=10, pady=10)
        self.txt_json.pack(fill=tk.X, side=tk.BOTTOM, pady=(15, 0))
        self.txt_json.insert('1.0', "El resultado JSON aparecerá aquí...")
        self.txt_json.config(state='disabled')

    def update_status(self):
        """Actualiza el estado visual mientras se seleccionan pasos"""
        count = sum(1 for v in self.step_vars.values() if v.get())
        self.lbl_status.config(text=f"Pasos seleccionados: {count}/6")

    def generate_evaluation(self):
        """Lógica principal de evaluación epidemiológica"""
        completed = [k for k, v in self.step_vars.items() if v.get()]
        omitted = [k for k in range(1, 7) if k not in completed]
        
        # Cálculo de eficiencia (cada paso vale 16.66%)
        efficiency = round((len(completed) / 6) * 100)
        
        # Determinación de riesgo
        # Regla crítica: Si falta paso 2 o 6 -> Riesgo Alto
        risk_level = "Bajo"
        recommendation = "Técnica adecuada. Continuar monitoreo."
        
        if 2 in omitted or 6 in omitted:
            risk_level = "Alto"
            missing_desc = []
            if 2 in omitted: missing_desc.append("fricción del dorso")
            if 6 in omitted: missing_desc.append("limpieza de uñas/puntas")
            recommendation = f"Riesgo Alto de Contaminación Cruzada. Omisión crítica detectada en: {', '.join(missing_desc)}. Requiere capacitación inmediata."
        elif len(omitted) > 0:
            risk_level = "Medio"
            recommendation = "Se omitieron pasos intermedios. Reforzar técnica completa."
        
        # Verificación de insumos (Adicional)
        if not self.has_water.get() or not self.has_soap.get():
            if risk_level != "Alto":
                risk_level = "Medio"
            recommendation += " [Nota: Falta agua o jabón en el proceso]."

        # Crear objeto JSON
        evaluation_data = {
            "evaluacion_id": f"EVAL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "eficiencia_porcentaje": efficiency,
            "pasos_completados": completed,
            "pasos_omitidos": omitted,
            "nivel_de_riesgo": risk_level,
            "recomendacion_epidemiologia": recommendation
        }
        
        # Actualizar UI
        self.lbl_efficiency.config(text=f"Eficiencia: {efficiency}%")
        
        color_map = {"Alto": self.colors['accent'], "Medio": "#F39C12", "Bajo": self.colors['success']}
        self.lbl_status.config(text=f"Nivel de Riesgo: {risk_level}", fg=color_map[risk_level])
        
        json_str = json.dumps(evaluation_data, indent=2, ensure_ascii=False)
        
        self.txt_json.config(state='normal')
        self.txt_json.delete('1.0', tk.END)
        self.txt_json.insert('1.0', json_str)
        self.txt_json.config(state='disabled')
        
        return evaluation_data

    def copy_to_clipboard(self):
        """Copia el JSON generado al portapapeles"""
        try:
            json_content = self.txt_json.get('1.0', tk.END).strip()
            if not json_content or "aparecerá" in json_content:
                messagebox.showwarning("Sin datos", "Primero genera una evaluación.")
                return
                
            self.root.clipboard_clear()
            self.root.clipboard_append(json_content)
            messagebox.showinfo("Éxito", "JSON copiado al portapapeles correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo copiar: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HandwashEvaluatorApp(root)
    root.mainloop()
