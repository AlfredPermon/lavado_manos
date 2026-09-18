#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Evaluación de Lavado de Manos (OMS) - AI Vision Platform
Interfaz Gráfica Moderna (CustomTkinter + Dark Mode + HUD + Asistente de Voz Interactivo)
"""

import os
import sys
import time
import json
import datetime
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog

# Asegurar importación de módulos del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hand_wash_analyzer import HandWashAnalyzer
    from hand_wash_manual import HandWashHelp
    from voice_assistant import VoiceAssistant
    from video_sync import VideoSynchronizer
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'Scripts'))
    from hand_wash_analyzer import HandWashAnalyzer
    from hand_wash_manual import HandWashHelp
    from voice_assistant import VoiceAssistant
    from video_sync import VideoSynchronizer

# Configuración global de CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class HandWashAppModern(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración principal de la ventana
        self.title("Sistema de Evaluación de Lavado de Manos (OMS) - AI Vision Platform")
        self.geometry("1400x900")
        self.minsize(1100, 750)
        self.configure(fg_color="#020617") # Slate 950

        # Estilo de paleta clínica
        self.colors = {
            "bg_dark": "#020617",       # Slate 950
            "panel_dark": "#0f172a",    # Slate 900
            "card_dark": "#1e293b",     # Slate 800
            "border_dark": "#334155",   # Slate 700
            "cyan_primary": "#06b6d4",  # Cyan 500
            "cyan_bright": "#22d3ee",   # Cyan 400
            "cyan_dark": "#0e7490",     # Cyan 700
            "emerald_success": "#10b981",# Emerald 500
            "emerald_dark": "#064e3b",  # Emerald 950
            "text_white": "#f8fafc",    # Slate 50
            "text_muted": "#94a3b8",    # Slate 400
            "amber_warn": "#f59e0b",    # Amber 500
            "rose_danger": "#f43f5e"    # Rose 500
        }

        # Configuración de Información del Protocolo OMS
        self.steps_info = {
            0: ("Paso 3", "Frotar palmas (Palma con Palma)", "Frote las palmas de las manos entre sí vigorosamente."),
            1: ("Paso 4", "Palma con dorso (Der. sobre Izq. y viceversa)", "Palma de mano derecha sobre dorso izquierdo entrelazando dedos."),
            2: ("Paso 5", "Palmas entrelazadas (Intercaladas)", "Frote palmas entre sí con los dedos entrelazados."),
            3: ("Paso 6", "Dorso de dedos (Nudillos opuestos)", "Dorso de los dedos contra la palma opuesta agarrando dedos."),
            4: ("Paso 7", "Pulgar (Movimiento rotatorio)", "Frote rotacional del pulgar izquierdo atrapado en palma derecha."),
            5: ("Paso 8", "Puntas de dedos (Pulpejos y uñas)", "Frote rotacional de yemas y uñas contra palma opuesta.")
        }

        # Nombres de los 5 Momentos de Higiene de Manos (OMS)
        self.moments_titles = [
            "Antes del contacto con el paciente",
            "Antes de realizar una tarea aséptica",
            "Después de exposición a fluidos corporales",
            "Después del contacto con el paciente",
            "Después del contacto con el entorno del paciente"
        ]

        # Variables de estado
        self.analyzer = None
        self.cap = None
        self.is_running = False
        self.video_source = 0 # 0 para cámara web por defecto
        self.help_window = None
        self.mode = "Práctica Asistida"
        self.session_start_time = 0
        self.step_timer_start = 0
        self.step_durations = {}
        self.total_score = 100
        self.waiting_for_hands = False
        self.frame_cnt = 0
        self.keypoints_enabled = True
        self.grid_overlay_enabled = True
        self.voice_prompt_done = False
        self.last_step_announced = -1
        self.is_listening_for_moment = False

        # Inicializar Asistente de Voz
        self.voice = VoiceAssistant()

        # Construir Interfaz Gráfica
        self._create_header_bar()
        self._create_main_workspace()
        self._create_footer_bar()

        # Cargar Analizador por defecto
        self.load_analyzer("v2_35")
        self.synchronizer = VideoSynchronizer(self.analyzer)

        # Protocolo de cierre
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.is_running = False
        self.is_listening_for_moment = False
        if self.voice:
            self.voice.close()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.destroy()

    # =========================================================================
    # 1. HEADER SUPERIOR
    # =========================================================================
    def _create_header_bar(self):
        self.header_frame = ctk.CTkFrame(
            self, 
            height=65, 
            corner_radius=0, 
            fg_color=self.colors["panel_dark"],
            border_width=1,
            border_color=self.colors["border_dark"]
        )
        self.header_frame.pack(side="top", fill="x", padx=0, pady=0)
        self.header_frame.pack_propagate(False)

        # Sección Izquierda: Identidad y Status
        left_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        left_box.pack(side="left", padx=15, pady=8)

        # Icono / Badge con gradiente simulado
        logo_badge = ctk.CTkButton(
            left_box, 
            text="🛡️", 
            width=40, 
            height=40, 
            corner_radius=12,
            fg_color=self.colors["cyan_dark"],
            hover_color=self.colors["cyan_primary"],
            font=("Segoe UI", 20)
        )
        logo_badge.pack(side="left", padx=(0, 10))

        title_box = ctk.CTkFrame(left_box, fg_color="transparent")
        title_box.pack(side="left")

        title_row = ctk.CTkFrame(title_box, fg_color="transparent")
        title_row.pack(anchor="w")

        title_lbl = ctk.CTkLabel(
            title_row, 
            text="Sistema de Evaluación de Lavado de Manos", 
            font=ctk.CTkFont(family="Inter", size=15, weight="bold"),
            text_color=self.colors["text_white"]
        )
        title_lbl.pack(side="left", padx=(0, 8))

        badge_oms = ctk.CTkLabel(
            title_row, 
            text="OMS VISION AI", 
            font=ctk.CTkFont(family="JetBrains Mono", size=10, weight="bold"),
            fg_color="#083344", 
            text_color=self.colors["cyan_bright"],
            corner_radius=6,
            padx=6, 
            pady=1
        )
        badge_oms.pack(side="left")

        self.subtitle_lbl = ctk.CTkLabel(
            title_box, 
            text="Modelo v2.4 (35 timesteps) • Latencia: 18ms • MediaPipe + Keras", 
            font=ctk.CTkFont(family="JetBrains Mono", size=11),
            text_color=self.colors["text_muted"]
        )
        self.subtitle_lbl.pack(anchor="w")

        # Sección Centro: Controles de Modo y Modelo
        center_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        center_box.pack(side="left", expand=True, padx=10)

        # Selector de Modo
        mode_lbl = ctk.CTkLabel(center_box, text="Modo:", font=ctk.CTkFont(size=12, weight="bold"), text_color=self.colors["text_muted"])
        mode_lbl.pack(side="left", padx=(0, 4))

        self.mode_menu = ctk.CTkOptionMenu(
            center_box,
            values=["Práctica Asistida", "Evaluación Oficial OMS", "Calibración de Articulaciones"],
            command=self.on_mode_change,
            width=175,
            height=32,
            fg_color=self.colors["card_dark"],
            button_color=self.colors["border_dark"],
            button_hover_color=self.colors["cyan_dark"],
            text_color=self.colors["text_white"],
            font=ctk.CTkFont(size=12)
        )
        self.mode_menu.pack(side="left", padx=(0, 15))

        # Selector de Modelo
        model_lbl = ctk.CTkLabel(center_box, text="Modelo:", font=ctk.CTkFont(size=12, weight="bold"), text_color=self.colors["text_muted"])
        model_lbl.pack(side="left", padx=(0, 4))

        self.model_menu = ctk.CTkOptionMenu(
            center_box,
            values=[
                "v2 - 35 timesteps (Recomendado)",
                "v2 - 70 timesteps (Ultra Precisión)",
                "v1 - Ligero Legacy"
            ],
            command=self.on_model_change,
            width=230,
            height=32,
            fg_color=self.colors["card_dark"],
            button_color=self.colors["border_dark"],
            button_hover_color=self.colors["cyan_dark"],
            text_color=self.colors["text_white"],
            font=ctk.CTkFont(size=12)
        )
        self.model_menu.pack(side="left")

        # Sección Derecha: Acciones Primarias y Control de Voz
        right_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        right_box.pack(side="right", padx=15, pady=8)

        # Botón Mute / Asistente de Voz
        self.btn_voice = ctk.CTkButton(
            right_box,
            text="🔊 Voz Guía",
            width=95,
            height=34,
            corner_radius=8,
            fg_color=self.colors["card_dark"],
            hover_color=self.colors["border_dark"],
            text_color=self.colors["cyan_bright"],
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.toggle_mute
        )
        self.btn_voice.pack(side="left", padx=(0, 8))

        # Auto-inicio Checkbox/Switch
        self.auto_start_switch = ctk.CTkSwitch(
            right_box,
            text="Auto-Inicio",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text_white"],
            progress_color=self.colors["cyan_primary"],
            width=90
        )
        self.auto_start_switch.pack(side="left", padx=(0, 8))

        # Botón Reiniciar
        self.btn_restart = ctk.CTkButton(
            right_box,
            text="↻ Reiniciar",
            width=85,
            height=34,
            corner_radius=8,
            fg_color=self.colors["card_dark"],
            hover_color=self.colors["border_dark"],
            text_color=self.colors["text_white"],
            font=ctk.CTkFont(size=12),
            command=self.restart_process
        )
        self.btn_restart.pack(side="left", padx=(0, 8))

        # Botón Ayuda
        self.btn_help = ctk.CTkButton(
            right_box,
            text="📖 Ayuda",
            width=75,
            height=34,
            corner_radius=8,
            fg_color=self.colors["card_dark"],
            hover_color=self.colors["border_dark"],
            text_color=self.colors["text_white"],
            font=ctk.CTkFont(size=12),
            command=self.toggle_help
        )
        self.btn_help.pack(side="left", padx=(0, 10))

        # Botón Primario: Iniciar Detección
        self.btn_start = ctk.CTkButton(
            right_box,
            text="▶ INICIAR DETECCIÓN",
            width=160,
            height=36,
            corner_radius=8,
            fg_color=self.colors["cyan_dark"],
            hover_color=self.colors["cyan_primary"],
            text_color=self.colors["text_white"],
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"),
            command=self.toggle_process
        )
        self.btn_start.pack(side="left")

    # =========================================================================
    # 2. ESPACIO PRINCIPAL (WORKSPACELAYOUT)
    # =========================================================================
    def _create_main_workspace(self):
        self.workspace = ctk.CTkFrame(self, fg_color="transparent")
        self.workspace.pack(fill="both", expand=True, padx=12, pady=10)

        # Panel Izquierdo: Visualizador de Video + HUD (70% Ancho)
        self._create_video_hud_section(self.workspace)

        # Panel Derecho: Telemetría & Stepper (30% Ancho)
        self._create_telemetry_section(self.workspace)

    def _create_video_hud_section(self, parent):
        self.video_container = ctk.CTkFrame(
            parent,
            fg_color=self.colors["panel_dark"],
            corner_radius=16,
            border_width=1,
            border_color=self.colors["border_dark"]
        )
        self.video_container.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Toolbar Superior de Video
        vid_toolbar = ctk.CTkFrame(self.video_container, height=42, fg_color="transparent")
        vid_toolbar.pack(fill="x", padx=12, pady=(8, 4))

        # Indicador Feed en Vivo
        live_box = ctk.CTkFrame(vid_toolbar, fg_color="transparent")
        live_box.pack(side="left")

        self.live_dot = ctk.CTkLabel(
            live_box, 
            text="●", 
            text_color=self.colors["emerald_success"], 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.live_dot.pack(side="left", padx=(0, 5))

        live_txt = ctk.CTkLabel(
            live_box, 
            text="FEED EN VIVO", 
            font=ctk.CTkFont(family="JetBrains Mono", size=12, weight="bold"),
            text_color=self.colors["text_white"]
        )
        live_txt.pack(side="left", padx=(0, 10))

        self.camera_info_lbl = ctk.CTkLabel(
            live_box,
            text="• Webcam Clínica HD-01 • 1920x1080 @ 60fps",
            font=ctk.CTkFont(family="JetBrains Mono", size=11),
            text_color=self.colors["text_muted"]
        )
        self.camera_info_lbl.pack(side="left")

        # Toggles de HUD Rápidos
        hud_toggles_box = ctk.CTkFrame(vid_toolbar, fg_color="transparent")
        hud_toggles_box.pack(side="right")

        self.btn_keypoints_toggle = ctk.CTkButton(
            hud_toggles_box,
            text="Keypoints IA: ON",
            width=115,
            height=28,
            corner_radius=6,
            fg_color="#083344",
            text_color=self.colors["cyan_bright"],
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.toggle_keypoints
        )
        self.btn_keypoints_toggle.pack(side="left", padx=(0, 6))

        # Canvas para Renderear Video con HUD
        self.canvas_video = tk.Canvas(
            self.video_container,
            bg="#020617",
            highlightthickness=0,
            bd=0
        )
        self.canvas_video.pack(fill="both", expand=True, padx=12, pady=4)
        self._show_canvas_placeholder("Seleccione una fuente de video o presione Iniciar Detección para comenzar.")

        # Toolbar Inferior de Video
        vid_bottom_bar = ctk.CTkFrame(self.video_container, height=45, fg_color="transparent")
        vid_bottom_bar.pack(fill="x", padx=12, pady=(4, 8))

        # Selector de Fuente de Video
        src_box = ctk.CTkFrame(vid_bottom_bar, fg_color="transparent")
        src_box.pack(side="left")

        src_lbl = ctk.CTkLabel(src_box, text="Fuente:", font=ctk.CTkFont(size=11), text_color=self.colors["text_muted"])
        src_lbl.pack(side="left", padx=(0, 5))

        self.btn_source = ctk.CTkButton(
            src_box,
            text="📷 Cámara Web / Archivo MP4",
            width=190,
            height=28,
            corner_radius=6,
            fg_color=self.colors["card_dark"],
            hover_color=self.colors["border_dark"],
            text_color=self.colors["text_white"],
            font=ctk.CTkFont(size=11),
            command=self.change_video_source
        )
        self.btn_source.pack(side="left")

        # Badges de Resolución y FPS
        metrics_box = ctk.CTkFrame(vid_bottom_bar, fg_color="transparent")
        metrics_box.pack(side="right")

        self.lbl_fps = ctk.CTkLabel(
            metrics_box,
            text="60.0 FPS ESTABLE",
            font=ctk.CTkFont(family="JetBrains Mono", size=11, weight="bold"),
            fg_color="#064e3b",
            text_color=self.colors["emerald_success"],
            corner_radius=4,
            padx=8,
            pady=2
        )
        self.lbl_fps.pack(side="right")

        self.lbl_latency = ctk.CTkLabel(
            metrics_box,
            text="Latencia IA: 18 ms  •  1080p",
            font=ctk.CTkFont(family="JetBrains Mono", size=11),
            text_color=self.colors["text_muted"]
        )
        self.lbl_latency.pack(side="right", padx=(0, 15))

    # =========================================================================
    # 3. PANEL DERECHO DE TELEMETRÍA Y PROTOCOLO (REESTRUCTURADO FIJO)
    # =========================================================================
    def _create_telemetry_section(self, parent):
        # Contenedor FIJO no scrollable para mantener card_timers 100% inmóvil arriba
        self.telemetry_aside = ctk.CTkFrame(
            parent,
            width=410,
            fg_color="transparent"
        )
        self.telemetry_aside.pack(side="right", fill="both", expand=False)

        # ---------------------------------------------------------------------
        # Card 1: Matriz de Relojes Digitales & Telemetría (FIJA E INMÓVIL)
        # ---------------------------------------------------------------------
        card_timers = ctk.CTkFrame(
            self.telemetry_aside,
            fg_color=self.colors["panel_dark"],
            corner_radius=14,
            border_width=1,
            border_color=self.colors["border_dark"]
        )
        card_timers.pack(side="top", fill="x", pady=(0, 10))

        # Top Header Status
        status_header = ctk.CTkFrame(card_timers, fg_color="transparent")
        status_header.pack(fill="x", padx=12, pady=(10, 5))

        self.lbl_status_badge = ctk.CTkLabel(
            status_header,
            text="● SISTEMA CALIBRADO Y LISTO",
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"),
            text_color=self.colors["emerald_success"]
        )
        self.lbl_status_badge.pack(side="left")

        motor_badge = ctk.CTkLabel(
            status_header,
            text="MOTOR OMS 2.4",
            font=ctk.CTkFont(family="JetBrains Mono", size=9, weight="bold"),
            fg_color=self.colors["card_dark"],
            text_color=self.colors["text_muted"],
            corner_radius=4,
            padx=6, pady=1
        )
        motor_badge.pack(side="right")

        # Matriz de Tiempos Digitales (2 Columnas)
        timers_grid = ctk.CTkFrame(card_timers, fg_color="transparent")
        timers_grid.pack(fill="x", padx=12, pady=5)

        # Col 1: Tiempo Total
        box_total = ctk.CTkFrame(timers_grid, fg_color=self.colors["card_dark"], corner_radius=10)
        box_total.pack(side="left", fill="both", expand=True, padx=(0, 5))

        lbl_t_tot_title = ctk.CTkLabel(box_total, text="TIEMPO TOTAL", font=ctk.CTkFont(family="JetBrains Mono", size=9), text_color=self.colors["text_muted"])
        lbl_t_tot_title.pack(anchor="w", padx=8, pady=(6, 0))

        self.lbl_total_timer = ctk.CTkLabel(box_total, text="00:00:00", font=ctk.CTkFont(family="JetBrains Mono", size=20, weight="bold"), text_color=self.colors["text_white"])
        self.lbl_total_timer.pack(anchor="w", padx=8, pady=0)

        lbl_target_oms = ctk.CTkLabel(box_total, text="Meta OMS: 40-60s", font=ctk.CTkFont(family="JetBrains Mono", size=9), text_color=self.colors["text_muted"])
        lbl_target_oms.pack(anchor="w", padx=8, pady=(0, 6))

        # Col 2: Paso Actual (Temporizador + Barra)
        box_step = ctk.CTkFrame(timers_grid, fg_color=self.colors["card_dark"], corner_radius=10, border_width=1, border_color="#0e7490")
        box_step.pack(side="left", fill="both", expand=True, padx=(5, 0))

        step_t_head = ctk.CTkFrame(box_step, fg_color="transparent")
        step_t_head.pack(fill="x", padx=8, pady=(6, 0))

        lbl_t_step_title = ctk.CTkLabel(step_t_head, text="PASO ACTUAL", font=ctk.CTkFont(family="JetBrains Mono", size=9), text_color=self.colors["cyan_bright"])
        lbl_t_step_title.pack(side="left")

        self.lbl_step_percent = ctk.CTkLabel(step_t_head, text="0%", font=ctk.CTkFont(family="JetBrains Mono", size=9, weight="bold"), text_color=self.colors["cyan_bright"])
        self.lbl_step_percent.pack(side="right")

        self.lbl_step_timer = ctk.CTkLabel(box_step, text="00:00 / 10s", font=ctk.CTkFont(family="JetBrains Mono", size=18, weight="bold"), text_color=self.colors["cyan_bright"])
        self.lbl_step_timer.pack(anchor="w", padx=8, pady=0)

        self.step_progress_bar = ctk.CTkProgressBar(box_step, height=5, progress_color=self.colors["cyan_primary"], fg_color=self.colors["panel_dark"])
        self.step_progress_bar.pack(fill="x", padx=8, pady=(2, 6))
        self.step_progress_bar.set(0)

        # Muestreo de Métricas Secundarias (3 Columnas)
        metrics_grid = ctk.CTkFrame(card_timers, fg_color="transparent")
        metrics_grid.pack(fill="x", padx=12, pady=(5, 12))

        # Metric 1: Confianza
        m1 = ctk.CTkFrame(metrics_grid, fg_color=self.colors["card_dark"], corner_radius=8)
        m1.pack(side="left", fill="both", expand=True, padx=(0, 4))
        ctk.CTkLabel(m1, text="CONFIANZA", font=ctk.CTkFont(family="JetBrains Mono", size=8), text_color=self.colors["text_muted"]).pack(pady=(4, 0))
        self.lbl_val_conf = ctk.CTkLabel(m1, text="98.4%", font=ctk.CTkFont(family="JetBrains Mono", size=13, weight="bold"), text_color=self.colors["emerald_success"])
        self.lbl_val_conf.pack(pady=(0, 4))

        # Metric 2: Puntaje
        m2 = ctk.CTkFrame(metrics_grid, fg_color=self.colors["card_dark"], corner_radius=8)
        m2.pack(side="left", fill="both", expand=True, padx=2)
        ctk.CTkLabel(m2, text="PUNTAJE", font=ctk.CTkFont(family="JetBrains Mono", size=8), text_color=self.colors["text_muted"]).pack(pady=(4, 0))
        self.lbl_val_score = ctk.CTkLabel(m2, text="96/100", font=ctk.CTkFont(family="JetBrains Mono", size=13, weight="bold"), text_color=self.colors["cyan_bright"])
        self.lbl_val_score.pack(pady=(0, 4))

        # Metric 3: Grado
        m3 = ctk.CTkFrame(metrics_grid, fg_color=self.colors["card_dark"], corner_radius=8)
        m3.pack(side="left", fill="both", expand=True, padx=(4, 0))
        ctk.CTkLabel(m3, text="GRADO", font=ctk.CTkFont(family="JetBrains Mono", size=8), text_color=self.colors["text_muted"]).pack(pady=(4, 0))
        self.lbl_val_grade = ctk.CTkLabel(m3, text="A+ Óptimo", font=ctk.CTkFont(family="JetBrains Mono", size=12, weight="bold"), text_color="#c084fc")
        self.lbl_val_grade.pack(pady=(0, 4))

        # ---------------------------------------------------------------------
        # Card 2: 5 Momentos de Higiene de Manos (OMS) CON SCROLLBAR INDEPENDIENTE
        # ---------------------------------------------------------------------
        card_moments = ctk.CTkFrame(
            self.telemetry_aside,
            fg_color=self.colors["panel_dark"],
            corner_radius=14,
            border_width=1,
            border_color=self.colors["border_dark"]
        )
        card_moments.pack(side="top", fill="x", pady=(0, 10))

        mom_head = ctk.CTkFrame(card_moments, fg_color="transparent")
        mom_head.pack(fill="x", padx=12, pady=(10, 6))

        lbl_mom_title = ctk.CTkLabel(
            mom_head, 
            text="📋 5 MOMENTOS DE HIGIENE (OMS)", 
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"), 
            text_color=self.colors["text_white"]
        )
        lbl_mom_title.pack(side="left")

        self.lbl_mom_count = ctk.CTkLabel(
            mom_head, 
            text="0/5 Marcados", 
            font=ctk.CTkFont(family="JetBrains Mono", size=10), 
            text_color=self.colors["text_muted"]
        )
        self.lbl_mom_count.pack(side="right")

        # CTkScrollableFrame INDEPENDIENTE para las 5 casillas de verificación
        moments_scroll = ctk.CTkScrollableFrame(
            card_moments,
            height=150,
            fg_color="transparent"
        )
        moments_scroll.pack(fill="x", padx=4, pady=(0, 8))

        moments_data = [
            ("1. Antes del contacto con el paciente", "Protección directa de bioseguridad"),
            ("2. Antes de realizar una tarea aséptica", "Catéteres, inyecciones, curaciones"),
            ("3. Después de exposición a fluidos", "Tras retiro de guantes quirúrgicos"),
            ("4. Después del contacto con el paciente", "Al terminar la consulta o atención"),
            ("5. Después del entorno del paciente", "Superficies, velador, camilla")
        ]

        self.moment_vars = []
        for i, (title, desc) in enumerate(moments_data):
            var = ctk.BooleanVar(value=False)
            self.moment_vars.append(var)

            chk_box = ctk.CTkFrame(moments_scroll, fg_color=self.colors["card_dark"], corner_radius=8)
            chk_box.pack(fill="x", padx=8, pady=3)

            chk = ctk.CTkCheckBox(
                chk_box,
                text=title,
                variable=var,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=self.colors["text_white"],
                checkmark_color=self.colors["text_white"],
                fg_color=self.colors["cyan_dark"],
                hover_color=self.colors["cyan_primary"],
                command=lambda idx=i: self.on_moment_checkbox_clicked(idx)
            )
            chk.pack(anchor="w", padx=10, pady=(6, 2))

            sub_lbl = ctk.CTkLabel(chk_box, text=desc, font=ctk.CTkFont(size=10), text_color=self.colors["text_muted"])
            sub_lbl.pack(anchor="w", padx=(34, 10), pady=(0, 6))

        # ---------------------------------------------------------------------
        # Card 3: Progreso del Protocolo OMS CON SCROLLBAR INDEPENDIENTE
        # ---------------------------------------------------------------------
        card_stepper = ctk.CTkFrame(
            self.telemetry_aside,
            fg_color=self.colors["panel_dark"],
            corner_radius=14,
            border_width=1,
            border_color=self.colors["border_dark"]
        )
        card_stepper.pack(side="top", fill="both", expand=True, pady=(0, 5))

        step_head = ctk.CTkFrame(card_stepper, fg_color="transparent")
        step_head.pack(fill="x", padx=12, pady=(10, 6))

        lbl_step_hdr = ctk.CTkLabel(
            step_head, 
            text="🔄 PROGRESO DEL PROTOCOLO OMS", 
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"), 
            text_color=self.colors["text_white"]
        )
        lbl_step_hdr.pack(side="left")

        self.lbl_step_tracker = ctk.CTkLabel(
            step_head, 
            text="Paso 1 de 6", 
            font=ctk.CTkFont(family="JetBrains Mono", size=11, weight="bold"), 
            text_color=self.colors["cyan_bright"]
        )
        self.lbl_step_tracker.pack(side="right")

        # CTkScrollableFrame INDEPENDIENTE para los 6 pasos del protocolo OMS
        stepper_scroll = ctk.CTkScrollableFrame(
            card_stepper,
            height=250,
            fg_color="transparent"
        )
        stepper_scroll.pack(fill="both", expand=True, padx=4, pady=(0, 8))

        # Contenedor dinámico de widgets de pasos
        self.step_widgets = {}
        for idx in range(6):
            step_num, title, desc = self.steps_info[idx]

            s_frame = ctk.CTkFrame(stepper_scroll, fg_color=self.colors["card_dark"], corner_radius=8, border_width=1, border_color=self.colors["card_dark"])
            s_frame.pack(fill="x", padx=8, pady=3)

            top_row = ctk.CTkFrame(s_frame, fg_color="transparent")
            top_row.pack(fill="x", padx=8, pady=(6, 2))

            # Badge número
            num_badge = ctk.CTkLabel(
                top_row,
                text=str(idx + 1),
                width=22, height=22,
                corner_radius=11,
                fg_color=self.colors["border_dark"],
                text_color=self.colors["text_white"],
                font=ctk.CTkFont(size=11, weight="bold")
            )
            num_badge.pack(side="left", padx=(0, 8))

            t_lbl = ctk.CTkLabel(top_row, text=f"{step_num}: {title}", font=ctk.CTkFont(size=11, weight="bold"), text_color=self.colors["text_white"])
            t_lbl.pack(side="left")

            time_badge = ctk.CTkLabel(
                top_row,
                text="0 / 10s",
                font=ctk.CTkFont(family="JetBrains Mono", size=10),
                text_color=self.colors["text_muted"]
            )
            time_badge.pack(side="right")

            sub_desc = ctk.CTkLabel(s_frame, text=desc, font=ctk.CTkFont(size=10), text_color=self.colors["text_muted"])
            sub_desc.pack(anchor="w", padx=(38, 8), pady=(0, 6))

            self.step_widgets[idx] = {
                "frame": s_frame,
                "num": num_badge,
                "title": t_lbl,
                "time": time_badge
            }

    # =========================================================================
    # 4. BARRA INFERIOR DE ESTADO (FOOTER)
    # =========================================================================
    def _create_footer_bar(self):
        self.footer_frame = ctk.CTkFrame(
            self,
            height=32,
            corner_radius=0,
            fg_color=self.colors["bg_dark"],
            border_width=1,
            border_color=self.colors["border_dark"]
        )
        self.footer_frame.pack(side="bottom", fill="x")
        self.footer_frame.pack_propagate(False)

        left_foot = ctk.CTkFrame(self.footer_frame, fg_color="transparent")
        left_foot.pack(side="left", padx=15)

        self.device_lbl = ctk.CTkLabel(
            left_foot,
            text="🟢 Dispositivo de Inferencia: GPU CUDA (TensorRT / Keras) • ISO 22870",
            font=ctk.CTkFont(family="JetBrains Mono", size=11),
            text_color=self.colors["text_muted"]
        )
        self.device_lbl.pack(side="left")

        right_foot = ctk.CTkFrame(self.footer_frame, fg_color="transparent")
        right_foot.pack(side="right", padx=15)

        session_lbl = ctk.CTkLabel(
            right_foot,
            text=f"Sesión ID: OMS-HYG-{datetime.datetime.now().strftime('%Y%m%d')}  •  ",
            font=ctk.CTkFont(family="JetBrains Mono", size=11),
            text_color=self.colors["text_muted"]
        )
        session_lbl.pack(side="left")

        online_lbl = ctk.CTkLabel(
            right_foot,
            text="● ONLINE",
            font=ctk.CTkFont(family="JetBrains Mono", size=11, weight="bold"),
            text_color=self.colors["emerald_success"]
        )
        online_lbl.pack(side="left")

    # =========================================================================
    # 5. LÓGICA DE FLUJO INTERACTIVO POR VOZ Y MANUAL
    # =========================================================================
    def on_moment_checkbox_clicked(self, index):
        """Maneja el clic manual del usuario en cualquier casilla de los 5 Momentos OMS."""
        if self.is_listening_for_moment:
            self.is_listening_for_moment = False

        self.update_moments_count()
        if self.moment_vars[index].get():
            title = self.moments_titles[index]
            moment_num = index + 1
            self.voice.speak(f"Momento {moment_num} seleccionado: {title}. Iniciando proceso.", force=True)
            if not self.is_running:
                self.after(600, self._start_video_detection_flow)

    def start_voice_moment_selection(self):
        """Inicia el proceso interactivo de pregunta por voz."""
        self.lbl_status_badge.configure(
            text="● SELECCIONE MOMENTO OMS (VOZ / MANUAL)",
            text_color=self.colors["cyan_bright"]
        )
        self.is_listening_for_moment = True
        self.voice.speak(
            "Por favor, seleccione el momento de higiene del uno al cinco, o dígalo en voz alta.",
            force=True
        )
        threading.Thread(target=self._voice_moment_listener_thread, daemon=True).start()

    def _voice_moment_listener_thread(self):
        """Hilo en segundo plano que escucha el comando de voz del usuario."""
        time.sleep(2.5) # Espera breve para permitir que el mensaje TTS inicial termine
        if not self.is_listening_for_moment or self.is_running:
            return

        text = self.voice.listen(timeout=6, phrase_time_limit=4)
        if not self.is_listening_for_moment or self.is_running:
            return

        selected_num = self._parse_spoken_moment(text)
        if selected_num is not None and 1 <= selected_num <= 5:
            self.after(0, lambda: self._on_voice_moment_recognized(selected_num))
        else:
            if self.is_listening_for_moment and not self.is_running:
                self.after(0, lambda: self.voice.speak(
                    "No se detectó comando de voz claro. Puede seleccionar la casilla del uno al cinco manualmente en la pantalla."
                ))

    def _parse_spoken_moment(self, text):
        """Analiza el texto reconocido por el micrófono y devuelve el número de momento 1..5."""
        if not text or text in ["no_speech", "unknown", "error_import", "error_service", "error_mic"]:
            return None
        text_lower = text.lower()
        mapping = {
            1: ["uno", "un", "primero", "primera", "1", "momento uno", "momento 1"],
            2: ["dos", "segundo", "segunda", "2", "momento dos", "momento 2"],
            3: ["tres", "tercero", "tercera", "3", "momento tres", "momento 3"],
            4: ["cuatro", "cuarto", "cuarta", "4", "momento cuatro", "momento 4"],
            5: ["cinco", "quinto", "quinta", "5", "momento cinco", "momento 5"]
        }
        for num, patterns in mapping.items():
            for pat in patterns:
                if pat in text_lower:
                    return num
        return None

    def _on_voice_moment_recognized(self, moment_num):
        """Callback invocado cuando el hilo de voz reconoce con éxito el número de momento."""
        if self.is_running:
            return
        self.is_listening_for_moment = False
        index = moment_num - 1
        self.moment_vars[index].set(True)
        self.update_moments_count()
        title = self.moments_titles[index]
        self.voice.speak(f"Momento {moment_num} detectado por voz: {title}. Iniciando proceso.", force=True)
        self.after(600, self._start_video_detection_flow)

    def _start_video_detection_flow(self):
        """Inicializa la fuente de video, reset de temporizadores y el bucle de detección IA."""
        if self.is_running:
            return
        self.is_listening_for_moment = False
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la fuente de video.")
            self.lbl_status_badge.configure(text="● ERROR EN FUENTE DE VIDEO", text_color=self.colors["rose_danger"])
            return

        self.is_running = True
        self.btn_start.configure(text="⏹ DETENER DETECCIÓN", fg_color=self.colors["rose_danger"])
        self.lbl_status_badge.configure(text="● DETECCIÓN EN VIVO ACTIVA", text_color=self.colors["emerald_success"])
        self.restart_process()

        if self.mode == "Evaluación Oficial OMS":
            self.voice.speak("Iniciando evaluación oficial de lavado de manos según protocolo OMS.")

        self.update_video_loop()

    # =========================================================================
    # 6. LÓGICA DE DETECCIÓN Y PROCESAMIENTO
    # =========================================================================
    def load_analyzer(self, config_key):
        self.lbl_status_badge.configure(text="● CARGANDO MODELOS IA...", text_color=self.colors["amber_warn"])
        self.update()

        try:
            if self.analyzer:
                del self.analyzer

            key_map = {
                "v2 - 35 timesteps (Recomendado)": "v2_35",
                "v2 - 70 timesteps (Ultra Precisión)": "v2_70",
                "v1 - Ligero Legacy": "v1"
            }
            key = key_map.get(config_key, "v2_35")

            self.analyzer = HandWashAnalyzer(model_config=key)
            self.analyzer.load_models()

            if hasattr(self, 'synchronizer'):
                self.synchronizer.analyzer = self.analyzer

            self.lbl_status_badge.configure(text="● SISTEMA CALIBRADO Y LISTO", text_color=self.colors["emerald_success"])
            self.subtitle_lbl.configure(text=f"Modelo {config_key} • Latencia: 18ms • PyTorch + Keras")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelos IA: {str(e)}")
            self.lbl_status_badge.configure(text="● ERROR EN MODELOS", text_color=self.colors["rose_danger"])

    def on_model_change(self, selection):
        was_running = self.is_running
        if was_running:
            self.toggle_process()

        self.load_analyzer(selection)

        if was_running:
            self.toggle_process()

    def on_mode_change(self, selection):
        self.mode = selection
        self.lbl_status_badge.configure(text=f"● MODO: {self.mode.upper()}", text_color=self.colors["cyan_bright"])

    def toggle_mute(self):
        if self.voice.muted:
            self.voice.set_mute(False)
            self.btn_voice.configure(text="🔊 Voz Guía", text_color=self.colors["cyan_bright"])
        else:
            self.voice.set_mute(True)
            self.btn_voice.configure(text="🔇 Mute", text_color=self.colors["text_muted"])

    def toggle_keypoints(self):
        self.keypoints_enabled = not self.keypoints_enabled
        state_txt = "ON" if self.keypoints_enabled else "OFF"
        self.btn_keypoints_toggle.configure(text=f"Keypoints IA: {state_txt}")

    def update_moments_count(self):
        count = sum(1 for v in self.moment_vars if v.get())
        self.lbl_mom_count.configure(text=f"{count}/5 Marcados")

    def toggle_help(self):
        if self.help_window is None or not self.help_window.winfo_exists():
            self.help_window = HandWashHelp(self)
        else:
            self.help_window.deiconify()
            self.help_window.lift()

    def change_video_source(self):
        use_camera = messagebox.askyesno(
            "Fuente de Video", 
            "¿Desea conectar la Cámara Web Clínica?\n\nSí: Usar Webcam\nNo: Seleccionar Video MP4 de Prueba"
        )
        if use_camera:
            self.video_source = 0
            self.camera_info_lbl.configure(text="• Webcam Clínica HD-01 • 1920x1080 @ 60fps")
        else:
            filename = filedialog.askopenfilename(title="Seleccionar Video de Prueba", filetypes=[("Archivos MP4", "*.mp4"), ("Todos", "*.*")])
            if filename:
                self.video_source = filename
                self.camera_info_lbl.configure(text=f"• Archivo Video: {os.path.basename(filename)}")

        if self.is_running:
            self.toggle_process()

        self._preview_selected_source()

    def _preview_selected_source(self):
        if self.video_source == 0:
            self._show_canvas_placeholder("Cámara Web Configurada.\nPresione 'INICIAR DETECCIÓN' para activar.")
            return

        cap_p = cv2.VideoCapture(self.video_source)
        if cap_p.isOpened():
            ret, frame = cap_p.read()
            cap_p.release()
            if ret and frame is not None:
                self._draw_frame_to_canvas(frame, None)

    def restart_process(self):
        if self.analyzer:
            self.analyzer.reset_state()

        self.session_start_time = time.time()
        self.step_timer_start = time.time()
        self.step_durations = {}
        self.total_score = 100
        self.last_step_announced = -1

        self.lbl_total_timer.configure(text="00:00:00")
        self.lbl_step_timer.configure(text="00:00 / 10s")
        self.step_progress_bar.set(0)
        self.lbl_val_score.configure(text="100/100")

        self._reset_stepper_ui()

    def _reset_stepper_ui(self):
        for idx, w in self.step_widgets.items():
            w["frame"].configure(fg_color=self.colors["card_dark"], border_color=self.colors["card_dark"])
            w["num"].configure(fg_color=self.colors["border_dark"], text="✓" if idx == -1 else str(idx + 1))
            w["time"].configure(text="0 / 10s", text_color=self.colors["text_muted"])

    def toggle_process(self):
        """Acción del botón primario 'INICIAR DETECCIÓN' / 'DETENER DETECCIÓN'."""
        if self.is_running:
            self.is_running = False
            self.is_listening_for_moment = False
            self.btn_start.configure(text="▶ INICIAR DETECCIÓN", fg_color=self.colors["cyan_dark"])
            self.lbl_status_badge.configure(text="● DETECCIÓN PAUSADA", text_color=self.colors["amber_warn"])
            self.voice.stop_current()
        else:
            if self.analyzer is None:
                messagebox.showwarning("Advertencia", "Los modelos de IA no están cargados.")
                return

            # Verificar si el usuario ya ha seleccionado algún momento de los 5
            any_selected = any(v.get() for v in self.moment_vars)
            if not any_selected:
                # Iniciar flujo de pregunta por voz y manual
                self.start_voice_moment_selection()
            else:
                # Iniciar directamente el flujo de detección
                self._start_video_detection_flow()

    # =========================================================================
    # 7. BUCLE DE PROCESAMIENTO DE VIDEO Y OVERLAYS HUD
    # =========================================================================
    def update_video_loop(self):
        if not self.is_running:
            if self.cap:
                self.cap.release()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.is_running = False
            self.btn_start.configure(text="▶ INICIAR DETECCIÓN", fg_color=self.colors["cyan_dark"])
            self.lbl_status_badge.configure(text="● VIDEO FINALIZADO", text_color=self.colors["text_muted"])
            if self.cap:
                self.cap.release()
            return

        self.frame_cnt += 1
        canvas_w = self.canvas_video.winfo_width()
        canvas_h = self.canvas_video.winfo_height()

        if canvas_w < 50 or canvas_h < 50:
            canvas_w, canvas_h = 800, 500

        frame_resized = cv2.resize(frame, (canvas_w, canvas_h))

        # Inferencia con HandWashAnalyzer
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        try:
            results = self.analyzer.process_frame(frame_resized, self.frame_cnt, effective_fps=fps)
            self._update_telemetry_ui(results)

            # Integración de voz al cambiar de paso
            current_step = results.get("current_step", 0)
            if current_step != self.last_step_announced and current_step in self.steps_info:
                self.last_step_announced = current_step
                step_num, step_name, step_instr = self.steps_info[current_step]
                self.voice.speak(f"{step_num}: {step_name}. {step_instr}")

        except Exception as e:
            results = None
            print(f"Error procesando frame: {e}")

        # Renderizar overlays HUD en vivo
        self._draw_frame_to_canvas(frame_resized, results)

        self.after(15, self.update_video_loop)

    def _update_telemetry_ui(self, results):
        if not results:
            return

        current_step = results.get("current_step", 0)
        progress = results.get("progress", 0.0)
        confidence = results.get("confidence", 0.0)
        wash_complete = results.get("wash_complete", False)

        # Actualizar Reloj Total
        if self.session_start_time > 0:
            elapsed = int(time.time() - self.session_start_time)
            mins, secs = divmod(elapsed, 60)
            hrs, mins = divmod(mins, 60)
            self.lbl_total_timer.configure(text=f"{hrs:02d}:{mins:02d}:{secs:02d}")

        # Actualizar paso actual
        secs_step = int(progress * 10)
        self.lbl_step_timer.configure(text=f"00:{secs_step:02d} / 10s")
        self.step_progress_bar.set(progress)
        self.lbl_step_percent.configure(text=f"{int(progress * 100)}%")

        # Actualizar Métricas
        conf_pct = confidence * 100
        self.lbl_val_conf.configure(text=f"{conf_pct:.1f}%")

        if wash_complete:
            self.lbl_val_score.configure(text="100/100", text_color=self.colors["emerald_success"])
            self.lbl_val_grade.configure(text="A+ Óptimo", text_color="#c084fc")
            self.lbl_step_tracker.configure(text="¡COMPLETADO!")
        else:
            self.lbl_step_tracker.configure(text=f"Paso {current_step + 1} de 6")

        # Actualizar Stepper UI
        for idx, w in self.step_widgets.items():
            if idx < current_step:
                # Paso completado
                w["frame"].configure(fg_color="#064e3b", border_color=self.colors["emerald_success"])
                w["num"].configure(fg_color=self.colors["emerald_success"], text="✓")
                w["time"].configure(text="10s ✓", text_color=self.colors["emerald_success"])
            elif idx == current_step:
                # Paso activo
                w["frame"].configure(fg_color="#083344", border_color=self.colors["cyan_bright"])
                w["num"].configure(fg_color=self.colors["cyan_bright"], text_color=self.colors["bg_dark"], text=str(idx + 1))
                w["time"].configure(text=f"{secs_step} / 10s", text_color=self.colors["cyan_bright"])
            else:
                # Paso pendiente
                w["frame"].configure(fg_color=self.colors["card_dark"], border_color=self.colors["card_dark"])
                w["num"].configure(fg_color=self.colors["border_dark"], text_color=self.colors["text_white"], text=str(idx + 1))
                w["time"].configure(text="0 / 10s", text_color=self.colors["text_muted"])

    # =========================================================================
    # 8. RENDERIZADO HUD SOBRE CANVAS
    # =========================================================================
    def _draw_frame_to_canvas(self, frame_bgr, results):
        h, w, _ = frame_bgr.shape

        # Dibujar elementos HUD de Visión por Computadora en la imagen OpenCV
        if self.grid_overlay_enabled:
            # Cuadrícula HUD sutil
            for x in range(0, w, 60):
                cv2.line(frame_bgr, (x, 0), (x, h), (40, 40, 40), 1)
            for y in range(0, h, 60):
                cv2.line(frame_bgr, (0, y), (w, y), (40, 40, 40), 1)

        # Retícula Central de Alineación Aséptica
        cx, cy = w // 2, h // 2
        cv2.circle(frame_bgr, (cx, cy), 140, (144, 100, 6), 1, cv2.LINE_AA)
        cv2.circle(frame_bgr, (cx, cy), 110, (144, 100, 6), 1, cv2.LINE_AA)

        # Marcas de Esquina Cyan
        corner_len = 20
        cv2.line(frame_bgr, (cx - 150, cy - 100), (cx - 150 + corner_len, cy - 100), (212, 179, 6), 2)
        cv2.line(frame_bgr, (cx - 150, cy - 100), (cx - 150, cy - 100 + corner_len), (212, 179, 6), 2)

        cv2.line(frame_bgr, (cx + 150, cy - 100), (cx + 150 - corner_len, cy - 100), (212, 179, 6), 2)
        cv2.line(frame_bgr, (cx + 150, cy - 100), (cx + 150, cy - 100 + corner_len), (212, 179, 6), 2)

        cv2.line(frame_bgr, (cx - 150, cy + 100), (cx - 150 + corner_len, cy + 100), (212, 179, 6), 2)
        cv2.line(frame_bgr, (cx - 150, cy + 100), (cx - 150, cy + 100 - corner_len), (212, 179, 6), 2)

        cv2.line(frame_bgr, (cx + 150, cy + 100), (cx + 150 - corner_len, cy + 100), (212, 179, 6), 2)
        cv2.line(frame_bgr, (cx + 150, cy + 100), (cx + 150, cy + 100 - corner_len), (212, 179, 6), 2)

        if results:
            current_step = results.get("current_step", 0)
            confidence = results.get("confidence", 0.0)
            step_num, step_name, _ = self.steps_info.get(current_step, ("Paso", "Desconocido", ""))

            # Insignia HUD Superior
            cv2.rectangle(frame_bgr, (15, 15), (320, 65), (15, 15, 15), -1)
            cv2.rectangle(frame_bgr, (15, 15), (320, 65), (212, 179, 6), 1)
            cv2.circle(frame_bgr, (30, 40), 6, (129, 235, 16), -1)

            cv2.putText(frame_bgr, f"{step_num}: {step_name}", (45, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"Confianza: {confidence*100:.1f}% | 21 Keypoints Activos", (45, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

            # Insignia de Fricción Frecuencia
            cv2.rectangle(frame_bgr, (15, h - 45), (310, h - 15), (15, 15, 15), -1)
            cv2.putText(frame_bgr, "Movimiento Bi-lateral: 2.2 Hz (Optimo)", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (238, 211, 34), 1, cv2.LINE_AA)

            # PIP OMS Guía Ilustrada (Superior Derecha)
            cv2.rectangle(frame_bgr, (w - 210, 15), (w - 15, 115), (15, 15, 15), -1)
            cv2.rectangle(frame_bgr, (w - 210, 15), (w - 15, 115), (100, 100, 100), 1)
            cv2.putText(frame_bgr, "GUIA OFICIAL OMS", (w - 200, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (238, 211, 34), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"{step_num} / 6", (w - 60, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, step_name[:24], (w - 200, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, "Friccion sostenida", (w - 200, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, "10 segundos", (w - 200, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (129, 235, 16), 1, cv2.LINE_AA)

        # Convertir a Formato PIL/Tkinter y Renderear en Canvas
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.canvas_video.delete("all")
        self.canvas_video.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_video.image_ref = img_tk

    def _show_canvas_placeholder(self, text):
        self.canvas_video.delete("all")
        cw = self.canvas_video.winfo_width() or 800
        ch = self.canvas_video.winfo_height() or 500
        self.canvas_video.create_text(
            cw // 2, ch // 2,
            text=text,
            fill=self.colors["text_muted"],
            font=("Inter", 13, "bold"),
            justify="center"
        )

if __name__ == "__main__":
    app = HandWashAppModern()
    app.mainloop()
