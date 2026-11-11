"""
ui/dashboard.py

The CustomTkinter window.

Thread-safety rule used everywhere in this file:
    Background threads NEVER touch widgets. They only put small messages on
    `self.ui_queue`. The Tk main thread reads that queue every 50 ms
    (`_pump_ui`) and updates the widgets.

Automatic question handling:
    * every new piece of transcribed text goes to the QuestionTracker
    * `_poll_question` runs on its own 200 ms timer (it does NOT wait for the
      audio queue to be empty) and fires once the question looks complete and
      the speaker has been quiet for PAUSE_SECONDS
    * a new question cancels the running answer, clears both panels and starts
      the next answer immediately
"""

import os
import queue
import threading

import customtkinter as ctk

from modules.audio_capture import AudioCapture
from modules.llm_engine import LLMEngine
from modules.question_detector import QuestionTracker
from modules.speech_to_text import SpeechToText, TranscriptionWorker

# How long the speaker must be quiet before a finished question is answered.
# Change it here, or set the environment variable PARAKEET_PAUSE (seconds).
PAUSE_SECONDS = float(os.environ.get("PARAKEET_PAUSE", "1.2"))

# ---- colours
BG = "#0e1016"
CARD = "#171a23"
CARD_BORDER = "#262b38"
TEXT = "#e6e9f2"
MUTED = "#8b93a7"
ACCENT = "#4f8cff"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
GREY = "#6b7280"


class ParakeetApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("Parakeet AI Assistant")
        self.geometry("1040x780")
        self.minsize(820, 640)
        self.configure(fg_color=BG)

        self.ui_queue = queue.Queue()
        self.tracker = QuestionTracker(pause_seconds=PAUSE_SECONDS)
        self.engine = LLMEngine()
        self.stt = None
        self.capture = None
        self.worker = None
        self.recording = False
        self.busy = False            # starting or stopping
        self.stt_ready = False
        self.ollama_ok = False
        self.speech_info = ""
        self.ollama_msg = "checking Ollama..."
        self.window_text = ""        # transcript since the last answered question
        self.current_gen = None      # id of the answer that is allowed to update the UI
        self.answer_started = False
        self.closing = False

        self._build_ui()
        self._set_status("Loading speech model...", AMBER)

        threading.Thread(target=self._load_models, daemon=True).start()
        self.after(50, self._pump_ui)
        self.after(200, self._poll_question)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=3)   # transcription
        self.grid_rowconfigure(3, weight=1)   # question
        self.grid_rowconfigure(4, weight=5)   # answer

        title_font = ctk.CTkFont(family="Segoe UI", size=24, weight="bold")
        sub_font = ctk.CTkFont(family="Segoe UI", size=12)
        btn_font = ctk.CTkFont(family="Segoe UI", size=14, weight="bold")
        status_font = ctk.CTkFont(family="Segoe UI", size=13)

        # header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 4))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="\U0001F99C  Parakeet AI Assistant", font=title_font,
                     text_color=TEXT).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, text="Live transcription • automatic question detection • local AI answers",
                     font=sub_font, text_color=MUTED).grid(row=1, column=0, sticky="w")

        # controls
        controls = ctk.CTkFrame(self, fg_color=CARD, border_color=CARD_BORDER,
                                border_width=1, corner_radius=12)
        controls.grid(row=1, column=0, sticky="ew", padx=24, pady=(10, 10))
        controls.grid_columnconfigure(2, weight=1)

        self.start_btn = ctk.CTkButton(controls, text="▶  Start Recording", width=170, height=40,
                                       font=btn_font, fg_color=GREEN, hover_color="#16a34a",
                                       text_color="#06210f", command=self.start_recording,
                                       state="disabled")
        self.start_btn.grid(row=0, column=0, padx=(14, 8), pady=12)
        self.stop_btn = ctk.CTkButton(controls, text="■  Stop Recording", width=170, height=40,
                                      font=btn_font, fg_color=RED, hover_color="#dc2626",
                                      command=self.stop_recording, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=8, pady=12)

        status_box = ctk.CTkFrame(controls, fg_color="transparent")
        status_box.grid(row=0, column=2, sticky="e", padx=16)
        self.dot = ctk.CTkLabel(status_box, text="●", font=ctk.CTkFont(size=20), text_color=GREY)
        self.dot.grid(row=0, column=0, padx=(0, 8))
        self.status_label = ctk.CTkLabel(status_box, text="", font=status_font, text_color=TEXT,
                                         anchor="w", justify="left", wraplength=520)
        self.status_label.grid(row=0, column=1)

        # panels
        self.transcript_box = self._make_panel(2, "LIVE TRANSCRIPTION", MUTED, ("Segoe UI", 15))
        self.question_box = self._make_panel(3, "DETECTED QUESTION", AMBER, ("Segoe UI", 17, "bold"))
        self.answer_box = self._make_panel(4, "AI ANSWER", ACCENT, ("Segoe UI", 16))
        self._set_text(self.transcript_box, "Press Start Recording and begin speaking...")
        self._set_text(self.question_box, "Waiting for a question...")
        self._set_text(self.answer_box, "The answer will appear here automatically.")

    def _make_panel(self, row, title, title_color, font):
        card = ctk.CTkFrame(self, fg_color=CARD, border_color=CARD_BORDER,
                            border_width=1, corner_radius=12)
        card.grid(row=row, column=0, sticky="nsew", padx=24, pady=(0, 10 if row < 4 else 22))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                     text_color=title_color).grid(row=0, column=0, sticky="w", padx=16, pady=(10, 0))
        box = ctk.CTkTextbox(card, wrap="word", font=ctk.CTkFont(*font), fg_color=CARD,
                             text_color=TEXT, border_width=0, activate_scrollbars=True)
        box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2, 10))
        box.configure(state="disabled")
        return box

    # ------------------------------------------------------------ small helpers
    def _set_status(self, text, color):
        self.dot.configure(text_color=color)
        self.status_label.configure(text=text)

    def _set_text(self, box, text):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("end", text)
        box.configure(state="disabled")

    def _append_text(self, box, text, limit=None):
        box.configure(state="normal")
        box.insert("end", text)
        if limit is not None:
            extra = len(box.get("1.0", "end-1c")) - limit
            if extra > 0:
                box.delete("1.0", "1.0+%dc" % extra)
        box.see("end")
        box.configure(state="disabled")

    def _idle_status(self):
        parts = [self.speech_info or "speech model not ready", self.ollama_msg]
        return " • ".join(parts)

    def _refresh_buttons(self):
        can_start = self.stt_ready and not self.recording and not self.busy
        self.start_btn.configure(state="normal" if can_start else "disabled")
        self.stop_btn.configure(state="normal" if self.recording and not self.busy else "disabled")

    # ---------------------------------------------------------- model loading
    def _load_models(self):
        try:
            stt = SpeechToText()
            info = stt.load()
            self.ui_queue.put(("stt_ready", stt, info, stt.gpu_error))
        except Exception as exc:  # noqa: BLE001
            self.ui_queue.put(("stt_error", str(exc)))
        ok, msg = self.engine.check_server()
        self.ui_queue.put(("ollama", ok, msg))
        if ok:
            self.engine.warm_up()

    # ------------------------------------------------------------- recording
    def start_recording(self):
        if self.recording or self.busy or not self.stt_ready:
            return
        self.busy = True
        self._refresh_buttons()
        self._set_status("Starting audio...", AMBER)
        threading.Thread(target=self._start_worker, daemon=True).start()

    def _start_worker(self):
        try:
            capture = AudioCapture()
            warnings = capture.start()
            worker = TranscriptionWorker(
                self.stt,
                capture.audio_queue,
                on_text=lambda t: self.ui_queue.put(("text", t)),
                on_error=lambda m: self.ui_queue.put(("worker_error", m)),
            )
            worker.start()
            self.ui_queue.put(("started", capture, worker, warnings))
        except Exception as exc:  # noqa: BLE001
            self.ui_queue.put(("start_failed", str(exc)))

    def stop_recording(self):
        if not self.recording or self.busy:
            return
        self.busy = True
        self.recording = False
        self._refresh_buttons()
        self._set_status("Stopping...", AMBER)
        capture, worker = self.capture, self.worker
        self.capture = self.worker = None

        def _stop():
            try:
                if worker:
                    worker.stop()
                if capture:
                    capture.stop()
                if worker:
                    worker.join(timeout=3)
            finally:
                self.ui_queue.put(("stopped",))

        threading.Thread(target=_stop, daemon=True).start()

    # ------------------------------------------------- UI message pump (main thread)
    def _pump_ui(self):
        if self.closing:
            return
        try:
            for _ in range(200):
                self._handle(self.ui_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(50, self._pump_ui)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "stt_ready":
            _, stt, info, gpu_error = msg
            self.stt, self.stt_ready = stt, True
            self.speech_info = "Speech: " + info
            if gpu_error:
                self.speech_info += " (GPU not used)"
            self._refresh_buttons()
            if not self.recording and not self.busy:
                self._set_status(self._idle_status(), GREY)
        elif kind == "stt_error":
            self._set_status("Could not load the speech model: %s" % msg[1], RED)
        elif kind == "ollama":
            _, self.ollama_ok, self.ollama_msg = msg
            if not self.recording and not self.busy and self.stt_ready:
                self._set_status(self._idle_status(), GREY if self.ollama_ok else AMBER)
        elif kind == "started":
            _, self.capture, self.worker, warnings = msg
            self.recording = True
            self.busy = False
            self.window_text = ""
            self.tracker.reset()
            self._set_text(self.transcript_box, "")
            self._set_text(self.question_box, "Waiting for a question...")
            self._set_text(self.answer_box, "The answer will appear here automatically.")
            self._refresh_buttons()
            note = ("  •  " + " ".join(warnings)) if warnings else ""
            self._set_status("Recording • Listening..." + note, GREEN)
        elif kind == "start_failed":
            self.busy = False
            self._refresh_buttons()
            self._set_status("Could not start recording: %s" % msg[1], RED)
        elif kind == "stopped":
            self.busy = False
            self.engine.cancel()
            self.current_gen = None
            self._refresh_buttons()
            self._set_status("Stopped • " + self._idle_status(), GREY)
        elif kind == "worker_error":
            if self.recording:
                self._set_status("Recording • %s" % msg[1], AMBER)
        elif kind == "text":
            self._on_text(msg[1])
        elif kind == "token":
            self._on_token(msg[1], msg[2])
        elif kind == "done":
            if msg[1] == self.current_gen and self.recording:
                self._set_status("Recording • Answer complete. Listening...", GREEN)
        elif kind == "error":
            if msg[1] == self.current_gen:
                self._set_text(self.answer_box, "⚠  " + msg[2])
                self._set_status("Answer failed: %s" % msg[2], RED)

    # ------------------------------------------------------------ transcription
    def _on_text(self, text):
        self._append_text(self.transcript_box, text + " ", limit=4000)
        self.window_text = (self.window_text + " " + text).strip()[-500:]
        self.tracker.update(self.window_text)

    # --------------------------------------------- independent question timer
    def _poll_question(self):
        if self.closing:
            return
        try:
            if self.recording:
                question = self.tracker.poll()
                if question:
                    self._start_question(question)
        finally:
            self.after(200, self._poll_question)

    def _start_question(self, question):
        # Reset detection so the next question starts from fresh text.
        self.window_text = ""
        self.tracker.reset()
        self._set_text(self.question_box, question)
        self._set_text(self.answer_box, "Generating answer...")
        self.answer_started = False
        self._set_status("Recording • Question detected. Generating answer...", AMBER)
        q = self.ui_queue
        # start_answer cancels the previous answer, so rapid-fire questions just work.
        self.current_gen = self.engine.start_answer(
            question,
            on_token=lambda gen, tok: q.put(("token", gen, tok)),
            on_done=lambda gen: q.put(("done", gen)),
            on_error=lambda gen, msg: q.put(("error", gen, msg)),
        )

    def _on_token(self, gen, token):
        if gen != self.current_gen:
            return  # stale token from an old answer
        if not self.answer_started:
            self.answer_started = True
            self._set_text(self.answer_box, "")
            token = token.lstrip()
            self._set_status("Recording • Answering...", ACCENT)
        self._append_text(self.answer_box, token)

    # ------------------------------------------------------------------ close
    def _on_close(self):
        self.closing = True
        try:
            self.engine.cancel()
            if self.worker:
                self.worker.stop()
            if self.capture:
                threading.Thread(target=self.capture.stop, daemon=True).start()
        finally:
            self.after(300, self.destroy)


def run():
    app = ParakeetApp()
    app.mainloop()