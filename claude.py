"""
╔══════════════════════════════════════════════════════════════════════╗
║         MEDICAL DIAGNOSTIC SYSTEM  —  ENHANCED v2.0                 ║
║  56 Diseases · 130+ Symptoms · Smart Feedback · Self-Training        ║
║  Dataset in: medical_dataset.py (same folder)                        ║
╚══════════════════════════════════════════════════════════════════════╝

REQUIREMENTS (install once):
    pip install customtkinter scikit-learn pandas numpy

RUN:
    python claude.py
"""

# ════════════════════════════════════════════════════════════════════
#  IMPORTS
# ════════════════════════════════════════════════════════════════════
import customtkinter as ctk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time, re, threading, json, os
from difflib import get_close_matches

# ── Load dataset from sibling file ───────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medical_dataset import (
    DISEASE_DATA, PRECAUTION_DATA, DESCRIPTION_DATA, SEVERITY_DATA
)

# ════════════════════════════════════════════════════════════════════
#  PERSISTENT FEEDBACK / RATINGS  (JSON file next to the script)
# ════════════════════════════════════════════════════════════════════
RATINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ratings.json")

def load_ratings():
    if os.path.exists(RATINGS_FILE):
        try:
            with open(RATINGS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_ratings(data):
    try:
        with open(RATINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════════
#  THEME CONSTANTS  —  Blue & Black Palette
# ════════════════════════════════════════════════════════════════════
COLORS = {
    "bg_dark":       "#030c18",
    "bg_panel":      "#050f20",
    "bg_card":       "#081828",
    "bg_card2":      "#0b2040",
    "bg_input":      "#060e1c",
    "accent_blue":   "#1565c0",
    "accent_mid":    "#1e88e5",
    "accent_bright": "#42a5f5",
    "accent_cyan":   "#00e5ff",
    "accent_glow":   "#80d8ff",
    "text_primary":  "#e3f2fd",
    "text_secondary":"#90caf9",
    "text_muted":    "#37598a",
    "success":       "#00e5a0",
    "warning":       "#ffab40",
    "danger":        "#ff5252",
    "border":        "#0d2545",
    "border_bright": "#1565c0",
    "sidebar_bg":    "#02080f",
    "hover_card":    "#0f2a4a",
    "rank1":         "#00e5ff",
    "rank2":         "#42a5f5",
    "rank3":         "#00c896",
    "rank4":         "#90caf9",
    "star_gold":     "#ffd700",
    "star_empty":    "#1a3050",
    "feedback_bg":   "#0a1f38",
    "pill_green":    "#003d28",
    "pill_blue":     "#0a1f38",
}

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ════════════════════════════════════════════════════════════════════
#  ANIMATED WIDGET HELPERS
# ════════════════════════════════════════════════════════════════════

def animate_in(widget, steps=8, delay=18):
    """Slide-fade-in: grows from 0 to full height."""
    try:
        widget.update_idletasks()
        target = widget.winfo_reqheight() or 60
    except Exception:
        target = 60
    step_h = max(1, target // steps)
    def _step(h):
        if not widget.winfo_exists():
            return
        widget.configure(height=min(h, target))
        if h < target:
            widget.after(delay, _step, h + step_h)
    widget.configure(height=1)
    widget.after(delay, _step, step_h)


def pulse_color(widget, color_a, color_b, times=4, delay=140):
    """Briefly pulses a label between two text colors."""
    def _pulse(n, use_a):
        if not widget.winfo_exists() or n <= 0:
            try:
                widget.configure(text_color=color_a)
            except Exception:
                pass
            return
        try:
            widget.configure(text_color=color_a if use_a else color_b)
        except Exception:
            return
        widget.after(delay, _pulse, n - 1, not use_a)
    _pulse(times * 2, True)


# ════════════════════════════════════════════════════════════════════
#  SESSION VOTE TRACKER  (reset each time the app starts)
# ════════════════════════════════════════════════════════════════════
# Maps disease_name -> True if user already voted this session
SESSION_VOTES: dict = {}


# ════════════════════════════════════════════════════════════════════
#  STAR RATING WIDGET
# ════════════════════════════════════════════════════════════════════

class StarRatingBar(ctk.CTkFrame):
    """
    One star per disease per chat session.
    • "👍 I prefer this"  — adds 1 star (disabled after voting this session)
    • "✕ Remove"          — appears after voting; removes that 1 star & re-enables button
    Stars accumulate across sessions (saved in ratings.json).
    """
    MAX_DISPLAY = 5

    def __init__(self, parent, disease_name, ratings_store, on_feedback, **kw):
        super().__init__(parent, fg_color=COLORS["feedback_bg"],
                         corner_radius=10, **kw)
        self.disease     = disease_name
        self.ratings     = ratings_store
        self.on_feedback = on_feedback
        self._build()

    def _build(self):
        self.row = ctk.CTkFrame(self, fg_color="transparent")
        self.row.pack(padx=14, pady=8, fill="x")

        # ── "I prefer this" button ───────────────────────────────
        already_voted = SESSION_VOTES.get(self.disease, False)
        self.pref_btn = ctk.CTkButton(
            self.row,
            text="👍  I prefer this",
            width=140, height=30,
            font=("Segoe UI", 11, "bold"),
            corner_radius=8,
            fg_color=COLORS["accent_blue"] if not already_voted else COLORS["border"],
            hover_color=COLORS["accent_mid"],
            text_color="white",
            state="disabled" if already_voted else "normal",
            command=self._clicked
        )
        self.pref_btn.pack(side="left", padx=(0, 8))

        # ── "Remove" button (hidden until voted this session) ────
        self.remove_btn = ctk.CTkButton(
            self.row,
            text="✕ Remove",
            width=80, height=30,
            font=("Segoe UI", 10),
            corner_radius=8,
            fg_color=COLORS["danger"],
            hover_color="#cc0000",
            text_color="white",
            command=self._remove
        )
        if already_voted:
            self.remove_btn.pack(side="left", padx=(0, 10))

        # ── Star display ─────────────────────────────────────────
        self.star_frame = ctk.CTkFrame(self.row, fg_color="transparent")
        self.star_frame.pack(side="left", padx=4)

        self.count_lbl = ctk.CTkLabel(
            self.row, text="",
            font=("Segoe UI", 10),
            text_color=COLORS["text_muted"]
        )
        self.count_lbl.pack(side="left", padx=8)

        self._refresh_stars()

    # ── Refresh star visuals ──────────────────────────────────────
    def _refresh_stars(self):
        for w in self.star_frame.winfo_children():
            w.destroy()

        count      = self.ratings.get(self.disease, 0)
        SHOW_STARS = 4
        filled     = min(count, SHOW_STARS)
        overflow   = count - SHOW_STARS

        voted_this_session = SESSION_VOTES.get(self.disease, False)

        for i in range(SHOW_STARS):
            is_filled = i < filled
            color     = COLORS["star_gold"] if is_filled else COLORS["star_empty"]

            star = ctk.CTkLabel(
                self.star_frame,
                text="★",
                font=("Segoe UI", 18),
                text_color=color,
                width=20
            )
            star.pack(side="left", padx=1)

            # Hover effect on filled stars only when user voted this session
            if is_filled and voted_this_session:
                star.configure(cursor="hand2")
                star.bind("<Enter>",
                          lambda e, s=star: s.configure(text_color=COLORS["danger"]))
                star.bind("<Leave>",
                          lambda e, s=star: s.configure(text_color=COLORS["star_gold"]))
                star.bind("<Button-1>", lambda e: self._remove())

        # Overflow counter — small text after the 4 stars
        if overflow > 0:
            ctk.CTkLabel(
                self.star_frame,
                text=f"+{overflow}",
                font=("Segoe UI", 9, "bold"),
                text_color=COLORS["star_gold"],
                width=26
            ).pack(side="left", padx=(2, 0))

        # Voter count label
        if count > 0:
            people = f"{count} {'person' if count == 1 else 'people'} rated this"
            self.count_lbl.configure(text=people)
        else:
            self.count_lbl.configure(text="No votes yet")

    # ── Vote ──────────────────────────────────────────────────────
    def _clicked(self):
        # Guard: only 1 vote per session
        if SESSION_VOTES.get(self.disease, False):
            return

        SESSION_VOTES[self.disease] = True
        cur = self.ratings.get(self.disease, 0)
        self.ratings[self.disease] = cur + 1
        save_ratings(self.ratings)

        # Update button states
        self.pref_btn.configure(state="disabled", fg_color=COLORS["border"])
        # Remove button hidden — clicking any gold star removes the vote instead
        self.remove_btn.pack_forget()

        self._refresh_stars()
        pulse_color(self.count_lbl, COLORS["star_gold"], COLORS["text_muted"])

        if self.on_feedback:
            self.on_feedback(self.disease)

    # ── Remove vote ───────────────────────────────────────────────
    def _remove(self):
        if not SESSION_VOTES.get(self.disease, False):
            return

        SESSION_VOTES[self.disease] = False
        cur = self.ratings.get(self.disease, 0)
        self.ratings[self.disease] = max(0, cur - 1)
        save_ratings(self.ratings)

        # Restore button states
        self.pref_btn.configure(state="normal", fg_color=COLORS["accent_blue"])
        self.remove_btn.pack_forget()

        self._refresh_stars()
        pulse_color(self.count_lbl, COLORS["danger"], COLORS["text_muted"])

        if self.on_feedback:
            self.on_feedback(self.disease)


# ════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION CLASS
# ════════════════════════════════════════════════════════════════════
class MedicalDiagnosticSystem(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MedDiag Pro  —  AI Medical Diagnostic System v2")
        self.geometry("1200x840")
        self.minsize(960, 680)
        self.configure(fg_color=COLORS["bg_dark"])
        self.resizable(True, True)

        # Persistent feedback / ratings
        self.ratings = load_ratings()

        # Extra training samples added from user feedback
        self.feedback_training = []  # list of (disease, symptom_list)
        self.last_matched_symptoms = []

        self._load_and_train()
        self._build_ui()
        self.history = []

    # ─────────────────────────────────────────────────────────────────
    #  MODEL TRAINING
    # ─────────────────────────────────────────────────────────────────
    def _load_and_train(self, extra_data=None):
        self.severity_map = SEVERITY_DATA

        combined = list(DISEASE_DATA) + (extra_data or self.feedback_training)

        all_syms = set()
        for _, syms in combined:
            all_syms.update(syms)
        self.unique_symptoms = sorted(all_syms)

        feat = {s: [] for s in self.unique_symptoms}
        feat["Disease"] = []
        for disease, syms in combined:
            sym_set = set(syms)
            for s in self.unique_symptoms:
                feat[s].append(1 if s in sym_set else 0)
            feat["Disease"].append(disease)

        tdf = pd.DataFrame(feat)
        self.features = [c for c in tdf.columns if c != "Disease"]
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(tdf[self.features], tdf["Disease"])

        self.disease_count = len(set(r[0] for r in combined))
        self.symptom_count = len(self.unique_symptoms)

    # ─────────────────────────────────────────────────────────────────
    #  SELF-TRAINING WHEN USER CLICKS FEEDBACK
    # ─────────────────────────────────────────────────────────────────
    def _on_feedback(self, disease):
        """Called when user clicks 'I prefer this' on a disease card."""
        if self.last_matched_symptoms:
            # Add 3 reinforcement records (weighted self-training)
            for _ in range(3):
                self.feedback_training.append(
                    (disease, list(self.last_matched_symptoms))
                )
            # Retrain silently in background
            threading.Thread(
                target=self._retrain_background, daemon=True
            ).start()

    def _retrain_background(self):
        try:
            self._load_and_train()
            self.after(0, self._retrain_flash)
        except Exception:
            pass

    def _retrain_flash(self):
        """Show a quick "Model Updated" flash in the status dot."""
        old_text  = self.status_dot.cget("text")
        old_color = COLORS["success"]
        self.status_dot.configure(text="✦ MODEL UPDATED", text_color=COLORS["star_gold"])
        self.after(2500, lambda: self.status_dot.configure(
            text=old_text, text_color=old_color))

    # ─────────────────────────────────────────────────────────────────
    #  SORTING HELPER  (stars first, then percentile)
    # ─────────────────────────────────────────────────────────────────
    def _sort_results(self, top_idx, probs):
        """
        Sort top_idx list by:
          1. star rating (descending)
          2. model probability (descending)
        Returns list of (idx, conf, stars).
        """
        items = []
        for idx in top_idx:
            idx   = int(idx)
            disease = self.model.classes_[idx]
            conf  = float(probs[idx]) * 100
            stars = self.ratings.get(disease, 0)
            items.append((idx, conf, stars, disease))
        # Sort: stars desc, then conf desc
        items.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return items

    # ─────────────────────────────────────────────────────────────────
    #  UI BUILD
    # ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    # ─────────────────────────────────────────────────────────────────
    #  SIDEBAR
    # ─────────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkFrame(
            self, width=230, corner_radius=0,
            fg_color=COLORS["sidebar_bg"],
            border_width=1, border_color=COLORS["border"]
        )
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(8, weight=1)

        # Logo
        logo = ctk.CTkFrame(sb, fg_color="transparent")
        logo.pack(pady=(32, 8), padx=20, fill="x")
        ctk.CTkLabel(logo, text="⚕", font=("Segoe UI", 44),
                     text_color=COLORS["accent_cyan"]).pack()
        ctk.CTkLabel(logo, text="MedDiag Pro",
                     font=("Segoe UI", 18, "bold"),
                     text_color=COLORS["text_primary"]).pack(pady=(4, 0))
        ctk.CTkLabel(logo, text="AI Diagnostic Engine v2",
                     font=("Segoe UI", 10),
                     text_color=COLORS["text_muted"]).pack()

        ctk.CTkFrame(sb, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=16, pady=18)

        # Stats cards
        stats = [
            ("🧬", "Diseases",  str(self.disease_count)),
            ("🔬", "Symptoms",  str(self.symptom_count)),
            ("🤖", "Model",     "Random Forest"),
            ("🌳", "Trees",     "200"),
        ]
        for icon, label, val in stats:
            card = ctk.CTkFrame(sb, fg_color=COLORS["bg_card"],
                                corner_radius=8, border_width=1,
                                border_color=COLORS["border"])
            card.pack(padx=14, pady=3, fill="x")
            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(padx=12, pady=8, fill="x")
            ctk.CTkLabel(inner, text=f"{icon}  {label}",
                         font=("Segoe UI", 10),
                         text_color=COLORS["text_muted"]).pack(anchor="w")
            ctk.CTkLabel(inner, text=val,
                         font=("Segoe UI", 13, "bold"),
                         text_color=COLORS["accent_bright"]).pack(anchor="w")

        ctk.CTkFrame(sb, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=16, pady=16)

        # Nav buttons
        nav_items = [("🏠", "Dashboard"), ("⭐", "Ratings"),
                     ("📋", "History"),   ("ℹ️", "About")]
        for icon, label in nav_items:
            ctk.CTkButton(
                sb, text=f"  {icon}   {label}", anchor="w",
                height=42, corner_radius=8,
                fg_color="transparent",
                hover_color=COLORS["bg_card2"],
                text_color=COLORS["text_secondary"],
                font=("Segoe UI", 13),
                command=lambda l=label: self._nav(l)
            ).pack(padx=10, pady=2, fill="x")

        # Feedback counter
        total_votes = sum(self.ratings.values())
        self.vote_lbl = ctk.CTkLabel(
            sb,
            text=f"👍  {total_votes} user vote{'s' if total_votes != 1 else ''}",
            font=("Segoe UI", 9), text_color=COLORS["text_muted"]
        )
        self.vote_lbl.pack(side="bottom", pady=(4, 2))
        ctk.CTkLabel(sb, text=f" •  {self.disease_count} diseases  ",
                     font=("Segoe UI", 9),
                     text_color=COLORS["text_muted"]).pack(side="bottom", pady=(0, 6))

    # ─────────────────────────────────────────────────────────────────
    #  MAIN PANEL
    # ─────────────────────────────────────────────────────────────────
    def _build_main(self):
        mf = ctk.CTkFrame(self, fg_color="transparent")
        mf.grid(row=0, column=1, sticky="nsew")
        mf.grid_columnconfigure(0, weight=1)
        mf.grid_rowconfigure(2, weight=1)

        # Topbar
        topbar = ctk.CTkFrame(mf, fg_color=COLORS["bg_panel"],
                              corner_radius=0, height=68,
                              border_width=1, border_color=COLORS["border"])
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.grid_propagate(False)

        left = ctk.CTkFrame(topbar, fg_color="transparent")
        left.pack(side="left", padx=24, pady=12)
        ctk.CTkLabel(left, text="Symptom Analysis Hub",
                     font=("Segoe UI", 24, "bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w")
        ctk.CTkLabel(left,
                     text="Describe symptoms · Get AI predictions · Rate & improve results",
                     font=("Segoe UI", 11),
                     text_color=COLORS["text_muted"]).pack(anchor="w")

        right = ctk.CTkFrame(topbar, fg_color="transparent")
        right.pack(side="right", padx=24)
        self.status_dot = ctk.CTkLabel(right, text="● READY",
                                       font=("Segoe UI", 12, "bold"),
                                       text_color=COLORS["success"])
        self.status_dot.pack()

        # Input section
        inp_outer = ctk.CTkFrame(mf, fg_color=COLORS["bg_panel"],
                                 corner_radius=0, border_width=1,
                                 border_color=COLORS["border"])
        inp_outer.grid(row=1, column=0, sticky="ew")

        inp = ctk.CTkFrame(inp_outer, fg_color="transparent")
        inp.pack(padx=28, pady=18, fill="x")
        inp.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(inp,
                     text="Enter Symptoms  (separate with commas or spaces)",
                     font=("Segoe UI", 12),
                     text_color=COLORS["text_muted"]).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        self.entry = ctk.CTkEntry(
            inp, height=54, font=("Segoe UI", 14),
            placeholder_text="e.g.  fever, headache, vomiting, joint_pain ...",
            corner_radius=10, border_width=2,
            border_color=COLORS["border_bright"],
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text_primary"],
            placeholder_text_color=COLORS["text_muted"]
        )
        self.entry.grid(row=1, column=0, sticky="ew", padx=(0, 14))
        self.entry.bind("<Return>", lambda e: self.start_analysis())

        self.btn = ctk.CTkButton(
            inp, text="▶  DIAGNOSE", height=54, width=160,
            font=("Segoe UI", 14, "bold"), corner_radius=10,
            fg_color=COLORS["accent_blue"],
            hover_color=COLORS["accent_mid"],
            text_color="white",
            command=self.start_analysis
        )
        self.btn.grid(row=1, column=1)

        self.progress = ctk.CTkProgressBar(
            inp, height=5, corner_radius=3,
            fg_color=COLORS["border"],
            progress_color=COLORS["accent_cyan"]
        )
        self.progress.set(0)
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        self.prog_lbl = ctk.CTkLabel(inp, text="", font=("Segoe UI", 10),
                                     text_color=COLORS["text_muted"])
        self.prog_lbl.grid(row=3, column=0, columnspan=2, sticky="w", pady=(3, 0))

        # Results scrollable area
        self.scroll = ctk.CTkScrollableFrame(
            mf, fg_color=COLORS["bg_dark"],
            corner_radius=0,
            scrollbar_button_color=COLORS["border_bright"],
            scrollbar_button_hover_color=COLORS["accent_mid"]
        )
        self.scroll.grid(row=2, column=0, sticky="nsew")
        self.scroll.grid_columnconfigure(0, weight=1)

        self._show_welcome()

    # ─────────────────────────────────────────────────────────────────
    #  WELCOME SCREEN
    # ─────────────────────────────────────────────────────────────────
    def _show_welcome(self):
        self._clear()
        wrap = ctk.CTkFrame(self.scroll, fg_color="transparent")
        wrap.pack(expand=True, fill="both", padx=30, pady=40)

        # Animated icon
        icon_lbl = ctk.CTkLabel(wrap, text="⚕", font=("Segoe UI", 72),
                                text_color=COLORS["accent_blue"])
        icon_lbl.pack()
        # Cute pulse on welcome
        self.after(300, lambda: pulse_color(icon_lbl,
                                            COLORS["accent_cyan"],
                                            COLORS["accent_blue"], times=3))

        ctk.CTkLabel(wrap, text="Ready for Diagnosis",
                     font=("Segoe UI", 24, "bold"),
                     text_color=COLORS["text_primary"]).pack(pady=(10, 4))
        ctk.CTkLabel(
            wrap,
            text="Type one or more symptoms · Press DIAGNOSE · Rate results to improve accuracy",
            font=("Segoe UI", 13),
            text_color=COLORS["text_muted"], justify="center"
        ).pack()

        # Feature badges
        badges_frame = ctk.CTkFrame(wrap, fg_color="transparent")
        badges_frame.pack(pady=16)
        badges = [
            ("⭐", "Smart Ratings"),
            ("🧠", "Self-Training"),
            ("📊", "Top 4 Results"),
            ("💾", "Saves Feedback"),
        ]
        for icon, txt in badges:
            pill = ctk.CTkFrame(badges_frame, fg_color=COLORS["bg_card2"],
                                corner_radius=20)
            pill.pack(side="left", padx=6)
            ctk.CTkLabel(pill, text=f"  {icon} {txt}  ",
                         font=("Segoe UI", 11),
                         text_color=COLORS["accent_bright"]).pack(padx=4, pady=6)

        # Examples
        hint = ctk.CTkFrame(wrap, fg_color=COLORS["bg_card"],
                            corner_radius=14, border_width=1,
                            border_color=COLORS["border"])
        hint.pack(pady=28, ipadx=10)
        ctk.CTkLabel(hint, text="💡  Click any example to fill it in",
                     font=("Segoe UI", 12, "bold"),
                     text_color=COLORS["accent_bright"]).pack(pady=(16, 8), padx=24)

        examples = [
            ("🦟", "Hepatitis A",        "fever, chills, headache, joint_pain, vomiting"),
            ("🫁", "Respiratory",        "cough, breathlessness, high_fever, fatigue, phlegm"),
            ("💛", "Liver / Jaundice",   "yellowish_skin, dark_urine, vomiting, abdominal_pain"),
            ("❤️", "Cardiac",            "chest_pain, breathlessness, sweating, vomiting"),
            ("🦠", "COVID-19",           "cough, loss_of_smell, fatigue, headache, sore_throat"),
            ("🦴", "Joint Issues",       "joint_pain, swelling_joints, stiff_neck, muscle_weakness"),
        ]
        for icon, label, syms in examples:
            row = ctk.CTkFrame(hint, fg_color=COLORS["bg_card2"],
                               corner_radius=8, cursor="hand2")
            row.pack(padx=20, pady=3, fill="x")
            inner = ctk.CTkFrame(row, fg_color="transparent")
            inner.pack(padx=14, pady=8, fill="x")
            ctk.CTkLabel(inner, text=f"{icon}  {label}",
                         font=("Segoe UI", 12, "bold"),
                         text_color=COLORS["accent_bright"]).pack(anchor="w")
            ctk.CTkLabel(inner, text=syms, font=("Consolas", 11),
                         text_color=COLORS["text_muted"]).pack(anchor="w")
            for w in [row, inner]:
                w.bind("<Button-1>", lambda e, t=syms: self._fill(t))
            ctk.CTkLabel(hint, text="", height=4).pack()

    def _fill(self, text):
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self.entry.focus()

    # ─────────────────────────────────────────────────────────────────
    #  ANALYSIS THREAD
    # ─────────────────────────────────────────────────────────────────
    def start_analysis(self):
        raw = self.entry.get().strip()
        if not raw:
            self._clear()
            self._msg_card("⚠  Please enter at least one symptom before pressing DIAGNOSE.",
                           COLORS["warning"])
            return
        self.btn.configure(state="disabled", text="⏳  Analyzing...")
        self.status_dot.configure(text="● PROCESSING", text_color=COLORS["warning"])
        self.progress.set(0)
        self.prog_lbl.configure(text="Starting...")
        self._clear()
        threading.Thread(target=self._run, args=(raw,), daemon=True).start()

    def _reset_ui(self):
        self.btn.configure(state="normal", text="▶  DIAGNOSE")

    def _run(self, raw):
        try:
            steps = [
                (0.15, "🔤  Tokenizing input..."),
                (0.35, "🔍  Fuzzy-matching symptoms..."),
                (0.60, "🌳  Running Random Forest (200 trees)..."),
                (0.80, "⭐  Applying star-rating boost..."),
                (0.95, "📊  Ranking top 4 predictions..."),
                (1.0,  "✓  Analysis complete"),
            ]
            for val, msg in steps:
                time.sleep(0.10)
                self.after(0, self.progress.set, val)
                self.after(0, self.prog_lbl.configure, {"text": msg})

            words = re.split(r"[,;\s]+", raw.lower())
            vector = np.zeros(len(self.features))
            matched, severity_total = [], 0

            for w in words:
                w = w.strip()
                if not w:
                    continue
                m = get_close_matches(w, self.features, n=1, cutoff=0.55)
                if m:
                    vector[self.features.index(m[0])] = 1
                    matched.append(m[0])
                    severity_total += self.severity_map.get(m[0], 0)

            vector_df = pd.DataFrame([vector], columns=self.features)

            self.after(0, self._reset_ui)

            if not matched:
                self.after(0, self._show_no_match, raw)
                self.after(0, self.status_dot.configure,
                           {"text": "● NO MATCH", "text_color": COLORS["danger"]})
            else:
                # Store for self-training
                self.last_matched_symptoms = list(matched)
                probs   = self.model.predict_proba(vector_df)[0]
                # Get top 6 candidates, will filter to 4 shown
                top_idx = np.argsort(probs)[-6:][::-1]
                self.history.append({"input": raw, "matched": matched})
                self.after(0, self._show_results, matched, severity_total,
                           top_idx.tolist(), probs.tolist())
                self.after(0, self.status_dot.configure,
                           {"text": "● DONE", "text_color": COLORS["success"]})

        except Exception as e:
            self.after(0, self._reset_ui)
            self.after(0, self.progress.set, 0)
            self.after(0, self.prog_lbl.configure, {"text": ""})
            self.after(0, self.status_dot.configure,
                       {"text": "● ERROR", "text_color": COLORS["danger"]})
            self.after(0, self._msg_card,
                       f"⚠  An error occurred: {str(e)}\n\nPlease try again.",
                       COLORS["danger"])

    # ─────────────────────────────────────────────────────────────────
    #  RESULTS — TOP 4 WITH RATINGS & SELF-TRAIN
    # ─────────────────────────────────────────────────────────────────
    def _show_results(self, matched, severity_total, top_idx, probs):
        # Summary bar
        sev_lv   = ("LOW",      COLORS["success"])  if severity_total < 15 else \
                   ("MODERATE", COLORS["warning"])   if severity_total < 30 else \
                   ("HIGH",     COLORS["danger"])
        sev_lbl, sev_col = sev_lv

        summ = ctk.CTkFrame(self.scroll, fg_color=COLORS["bg_card"],
                            corner_radius=14, border_width=1,
                            border_color=COLORS["border_bright"])
        summ.pack(fill="x", padx=24, pady=(22, 6))

        row = ctk.CTkFrame(summ, fg_color="transparent")
        row.pack(padx=20, pady=(14, 6), fill="x")
        ctk.CTkLabel(row, text=f"🔬  {len(matched)} symptom(s) identified",
                     font=("Segoe UI", 14, "bold"),
                     text_color=COLORS["accent_bright"]).pack(side="left")
        ctk.CTkLabel(row,
                     text=f"⚠  Severity: {sev_lbl}  ({severity_total} pts)",
                     font=("Segoe UI", 13, "bold"),
                     text_color=sev_col).pack(side="right")

        sym_text = "   ·   ".join(matched)
        ctk.CTkLabel(summ, text=f"  {sym_text}",
                     font=("Consolas", 11), text_color=COLORS["text_secondary"],
                     wraplength=780, justify="left").pack(
            padx=20, pady=(0, 14), anchor="w")

        # Sort candidates: star-rating first, then confidence
        sorted_items = self._sort_results(top_idx, probs)

        rank_colors  = [COLORS["rank1"], COLORS["rank2"],
                        COLORS["rank3"], COLORS["rank4"]]
        rank_borders = [COLORS["accent_cyan"], COLORS["border_bright"],
                        COLORS["accent_mid"],  COLORS["border"]]
        rank_labels  = ["🥇  Best Match", "🥈  2nd Possibility",
                        "🥉  3rd Possibility", "4️⃣  4th Possibility"]

        shown = 0
        for sort_rank, (idx, conf, stars, disease) in enumerate(sorted_items):
            if conf < 0.5 or shown >= 4:
                break
            shown += 1

            rank = min(sort_rank, 3)

            # ── Card wrapper (animated slide-in) ──────────────────
            card = ctk.CTkFrame(self.scroll, fg_color=COLORS["bg_card"],
                                corner_radius=14, border_width=1,
                                border_color=rank_borders[rank])
            card.pack(fill="x", padx=24, pady=6)
            # Animate: slight delay per card for a cascade effect
            self.after(shown * 120, lambda c=card: animate_in(c, steps=6, delay=14))

            # ── Card header ───────────────────────────────────────
            hdr = ctk.CTkFrame(card, fg_color=COLORS["bg_card2"],
                               corner_radius=10)
            hdr.pack(fill="x", padx=10, pady=(10, 0))
            hdr_in = ctk.CTkFrame(hdr, fg_color="transparent")
            hdr_in.pack(padx=16, pady=12, fill="x")

            # Rank label + star count badge side by side
            top_row = ctk.CTkFrame(hdr_in, fg_color="transparent")
            top_row.pack(fill="x")
            ctk.CTkLabel(top_row, text=rank_labels[rank],
                         font=("Segoe UI", 10),
                         text_color=COLORS["text_muted"]).pack(side="left")
            if stars > 0:
                ctk.CTkLabel(top_row,
                             text=f"  ★ {stars} vote{'s' if stars != 1 else ''} (user preferred)",
                             font=("Segoe UI", 10, "bold"),
                             text_color=COLORS["star_gold"]).pack(side="left", padx=10)

            ctk.CTkLabel(hdr_in, text=disease,
                         font=("Segoe UI", 20, "bold"),
                         text_color=rank_colors[rank]).pack(
                anchor="w", pady=(2, 8))

            # Confidence bar
            bar_row = ctk.CTkFrame(hdr_in, fg_color="transparent")
            bar_row.pack(fill="x")
            bar = ctk.CTkProgressBar(bar_row, height=8, corner_radius=4,
                                     fg_color=COLORS["border"],
                                     progress_color=rank_colors[rank])
            bar.set(max(0.0, min(1.0, float(probs[idx]))))
            bar.pack(side="left", fill="x", expand=True, padx=(0, 14))
            ctk.CTkLabel(bar_row, text=f"{conf:.1f}%",
                         font=("Segoe UI", 15, "bold"),
                         text_color=rank_colors[rank]).pack(side="right")

            # ── Feedback / Rating bar ──────────────────────────────
            fb = StarRatingBar(
                card, disease, self.ratings,
                on_feedback=self._on_feedback_wrapper
            )
            fb.pack(fill="x", padx=10, pady=(8, 0))

            # ── Description ───────────────────────────────────────
            desc_text = DESCRIPTION_DATA.get(disease, "")
            if not desc_text:
                for k in DESCRIPTION_DATA:
                    if (k.lower().strip() in disease.lower() or
                            disease.lower().strip() in k.lower()):
                        desc_text = DESCRIPTION_DATA[k]
                        break
            if desc_text:
                desc_f = ctk.CTkFrame(card, fg_color="transparent")
                desc_f.pack(fill="x", padx=24, pady=(12, 0))
                ctk.CTkLabel(desc_f, text="📖  About this condition",
                             font=("Segoe UI", 11, "bold"),
                             text_color=COLORS["text_muted"]).pack(anchor="w")
                ctk.CTkLabel(desc_f, text=desc_text,
                             font=("Segoe UI", 12),
                             text_color=COLORS["text_secondary"],
                             wraplength=760, justify="left").pack(
                    anchor="w", pady=(3, 0))

            # ── Precautions ───────────────────────────────────────
            prec_steps = PRECAUTION_DATA.get(disease, [])
            if not prec_steps:
                for k in PRECAUTION_DATA:
                    if (k.lower().strip() in disease.lower() or
                            disease.lower().strip() in k.lower()):
                        prec_steps = PRECAUTION_DATA[k]
                        break

            prec_f = ctk.CTkFrame(card, fg_color="transparent")
            prec_f.pack(fill="x", padx=24, pady=(12, 18))
            ctk.CTkLabel(prec_f, text="🛡  Recommended Precautions",
                         font=("Segoe UI", 13, "bold"),
                         text_color=COLORS["accent_bright"]).pack(anchor="w",
                                                                  pady=(0, 6))

            if prec_steps:
                grid_f = ctk.CTkFrame(prec_f, fg_color="transparent")
                grid_f.pack(fill="x")
                for i, step in enumerate(prec_steps):
                    step = str(step).strip()
                    if not step or step.lower() == "nan":
                        continue
                    pill = ctk.CTkFrame(grid_f, fg_color=COLORS["bg_card2"],
                                        corner_radius=8)
                    pill.grid(row=i // 2, column=i % 2, padx=4, pady=3,
                              sticky="ew")
                    grid_f.grid_columnconfigure(0, weight=1)
                    grid_f.grid_columnconfigure(1, weight=1)
                    ctk.CTkLabel(pill,
                                 text=f"  {i + 1}.  {step.capitalize()}",
                                 font=("Segoe UI", 12),
                                 text_color=COLORS["text_primary"],
                                 anchor="w").pack(padx=10, pady=8, anchor="w")
            else:
                ctk.CTkLabel(prec_f,
                             text="No specific precautions listed.",
                             font=("Segoe UI", 12),
                             text_color=COLORS["text_muted"]).pack(anchor="w")

        # ── Disclaimer ────────────────────────────────────────────
        disc = ctk.CTkFrame(self.scroll, fg_color=COLORS["bg_card"],
                            corner_radius=10, border_width=1,
                            border_color=COLORS["border"])
        disc.pack(fill="x", padx=24, pady=(8, 28))
        ctk.CTkLabel(
            disc,
            text="⚕  MEDICAL DISCLAIMER — This tool is for informational purposes only "
                 "and does not replace professional medical advice, diagnosis, or treatment. "
                 "Always consult a qualified healthcare provider for any medical concerns.",
            font=("Segoe UI", 10), text_color=COLORS["text_muted"],
            wraplength=780, justify="left"
        ).pack(padx=18, pady=12, anchor="w")

    def _on_feedback_wrapper(self, disease):
        """Thread-safe wrapper so rating refresh happens on main thread."""
        self._on_feedback(disease)
        # Update sidebar vote count
        total_votes = sum(self.ratings.values())
        self.after(0, self.vote_lbl.configure,
                   {"text": f"👍  {total_votes} user vote{'s' if total_votes != 1 else ''}"})

    # ─────────────────────────────────────────────────────────────────
    #  MISC HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _show_no_match(self, raw):
        card = ctk.CTkFrame(self.scroll, fg_color=COLORS["bg_card"],
                            corner_radius=14, border_width=1,
                            border_color=COLORS["danger"])
        card.pack(fill="x", padx=24, pady=40)
        ctk.CTkLabel(card, text="✗  No Matching Symptoms Found",
                     font=("Segoe UI", 18, "bold"),
                     text_color=COLORS["danger"]).pack(pady=(24, 6))
        ctk.CTkLabel(
            card,
            text=f"Input:  \"{raw}\"\n\n"
                 "Suggestions:\n"
                 "•  Use underscore for multi-word symptoms:  joint_pain,  high_fever\n"
                 "•  Try common terms:  fever  cough  vomiting  fatigue  headache\n"
                 "•  Separate symptoms with commas or spaces\n"
                 "•  The fuzzy matcher accepts minor typos",
            font=("Segoe UI", 13), text_color=COLORS["text_secondary"],
            justify="left"
        ).pack(padx=30, pady=(0, 24), anchor="w")

    def _msg_card(self, msg, color):
        card = ctk.CTkFrame(self.scroll, fg_color=COLORS["bg_card"],
                            corner_radius=12)
        card.pack(fill="x", padx=24, pady=30)
        ctk.CTkLabel(card, text=msg, font=("Segoe UI", 13),
                     text_color=color).pack(pady=20, padx=20)

    def _clear(self):
        for w in self.scroll.winfo_children():
            w.destroy()

    # ─────────────────────────────────────────────────────────────────
    #  NAVIGATION
    # ─────────────────────────────────────────────────────────────────
    def _nav(self, label):
        if label == "Dashboard":
            self._clear()
            self._show_welcome()
        elif label == "Ratings":
            self._clear()
            self._show_ratings()
        elif label == "History":
            self._clear()
            self._show_history()
        elif label == "About":
            self._clear()
            self._show_about()

    def _show_ratings(self):
        ctk.CTkLabel(self.scroll, text="⭐  User Ratings Leaderboard",
                     font=("Segoe UI", 22, "bold"),
                     text_color=COLORS["star_gold"]).pack(anchor="w", padx=28, pady=(24, 4))
        ctk.CTkFrame(self.scroll, height=2,
                     fg_color=COLORS["star_gold"]).pack(fill="x", padx=28, pady=(0, 14))

        if not self.ratings:
            ctk.CTkLabel(self.scroll,
                         text="No ratings yet. Run a diagnosis and click '👍 I prefer this' on any result.",
                         font=("Segoe UI", 13),
                         text_color=COLORS["text_muted"]).pack(pady=30)
            return

        # Sort by rating descending
        sorted_r = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        for rank, (disease, count) in enumerate(sorted_r):
            card = ctk.CTkFrame(self.scroll, fg_color=COLORS["bg_card"],
                                corner_radius=10, border_width=1,
                                border_color=COLORS["border"])
            card.pack(fill="x", padx=24, pady=4)
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(padx=16, pady=12, fill="x")

            # Rank medal
            medal = ["🥇", "🥈", "🥉"][rank] if rank < 3 else f"#{rank+1}"
            ctk.CTkLabel(row, text=medal,
                         font=("Segoe UI", 18),
                         text_color=COLORS["star_gold"]).pack(side="left", padx=(0, 12))

            ctk.CTkLabel(row, text=disease,
                         font=("Segoe UI", 14, "bold"),
                         text_color=COLORS["text_primary"]).pack(side="left")

            # Stars
            star_f = ctk.CTkFrame(row, fg_color="transparent")
            star_f.pack(side="right")
            displayed = min(count, 5)
            for i in range(5):
                ctk.CTkLabel(star_f, text="★",
                             font=("Segoe UI", 16),
                             text_color=COLORS["star_gold"] if i < displayed
                             else COLORS["star_empty"],
                             width=18).pack(side="left", padx=1)
            ctk.CTkLabel(star_f,
                         text=f"  {count} vote{'s' if count != 1 else ''}",
                         font=("Segoe UI", 11),
                         text_color=COLORS["text_muted"]).pack(side="left")

    def _show_history(self):
        ctk.CTkLabel(self.scroll, text="📋  Analysis History",
                     font=("Segoe UI", 22, "bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w", padx=28, pady=(24, 4))
        ctk.CTkFrame(self.scroll, height=2,
                     fg_color=COLORS["accent_blue"]).pack(fill="x", padx=28, pady=(0, 14))
        if not self.history:
            ctk.CTkLabel(self.scroll,
                         text="No analyses yet. Run a diagnosis to see history here.",
                         font=("Segoe UI", 13),
                         text_color=COLORS["text_muted"]).pack(pady=30)
            return
        for i, h in enumerate(reversed(self.history)):
            card = ctk.CTkFrame(self.scroll, fg_color=COLORS["bg_card"],
                                corner_radius=10)
            card.pack(fill="x", padx=24, pady=5)
            ctk.CTkLabel(card,
                         text=f"#{len(self.history) - i}  Input: {h['input']}",
                         font=("Segoe UI", 12, "bold"),
                         text_color=COLORS["accent_bright"]).pack(
                anchor="w", padx=16, pady=(10, 2))
            ctk.CTkLabel(card,
                         text="Matched: " + ", ".join(h["matched"]),
                         font=("Consolas", 11),
                         text_color=COLORS["text_muted"]).pack(
                anchor="w", padx=16, pady=(0, 10))

    def _show_about(self):
        ctk.CTkLabel(self.scroll, text="ℹ️  About MedDiag Pro v2",
                     font=("Segoe UI", 22, "bold"),
                     text_color=COLORS["accent_cyan"]).pack(anchor="w", padx=28, pady=(24, 4))
        ctk.CTkFrame(self.scroll, height=2,
                     fg_color=COLORS["accent_blue"]).pack(fill="x", padx=28, pady=(0, 14))

        items = [
            ("🤖  AI Engine",         "Scikit-learn Random Forest — 200 decision trees"),
            ("🧬  Diseases",           f"{self.disease_count} diseases across multiple categories"),
            ("🔬  Symptoms",           f"{self.symptom_count} unique symptoms with severity weighting"),
            ("⭐  User Ratings",       "Click 'I prefer this' to add a star — saved across sessions"),
            ("🧠  Self-Training",      "Each preference vote adds 3 reinforcement training records"),
            ("📊  Top 4 Results",      "Shows 4 diseases ordered by stars first, then AI confidence"),
            ("📁  Modular Dataset",    "All data lives in medical_dataset.py — easy to expand"),
            ("🔍  Input Matching",     "Fuzzy string matching via difflib — handles typos automatically"),
            ("🎨  UI Framework",       "CustomTkinter · Deep blue & black professional palette"),
        ]
        for title, desc in items:
            card = ctk.CTkFrame(self.scroll, fg_color=COLORS["bg_card"],
                                corner_radius=10, border_width=1,
                                border_color=COLORS["border"])
            card.pack(fill="x", padx=24, pady=5)
            ctk.CTkLabel(card, text=title, font=("Segoe UI", 13, "bold"),
                         text_color=COLORS["accent_bright"]).pack(
                anchor="w", padx=16, pady=(12, 2))
            ctk.CTkLabel(card, text=desc, font=("Segoe UI", 12),
                         text_color=COLORS["text_secondary"]).pack(
                anchor="w", padx=16, pady=(0, 12))


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = MedicalDiagnosticSystem()
    app.mainloop()
